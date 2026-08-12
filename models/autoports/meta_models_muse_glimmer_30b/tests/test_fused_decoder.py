# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness and performance tests for the Muse-Glimmer-30B **fused** decoder.

``FusedDecoder`` has the same public contract as ``FunctionalDecoder``, so this
module reuses the functional stage's fixtures, page-table / hidden-state helpers
and HF reference harness (imported from ``test_functional_decoder``) and re-runs
the same correctness surface against the fused implementation: both layer kinds,
non-aligned prefill lengths, caller-chunked continuation prefill, batched paged
prefill+decode, the full 131072 context, real checkpoint weights, an FP32 HF
control, determinism, host-fallback trapping and traced decode.

On top of that it adds what only the fusing stage can assert:

* ``test_fused_vs_functional_equivalence`` — fused against *unfused* on device,
  the graph-fusing skill's equivalence requirement;
* ``test_fused_graph_uses_fused_ops`` — the fused ops are actually dispatched and
  the primitive sequences they replaced are gone, so a silent regression to the
  functional graph fails the suite rather than passing it;
  ``test_fused_graph_uses_fused_ops`` also doubles as the layout-churn audit:
  no ``tilize``/``untilize``/``to_layout``/``typecast`` and no host round-trip
  anywhere in a measured prefill or decode.
* ``test_repeated_run_stress`` — a 64-step traced-decode soak over the new L1
  residual stream and sharded norm configs.
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tests import reference as R
from models.autoports.meta_models_muse_glimmer_30b.tests.test_functional_decoder import (  # noqa: F401
    CONTINUATION_SPLITS,
    HF_ADVERTISED_CONTEXT,
    LAYER_KINDS,
    PAGE_BLOCK_SIZE,
    PREFILL_CHUNK_SIZE,
    PREFILL_SEQ_LENS,
    SHORT_MAX_SEQ,
    _FallbackGuard,
    _reference_prefill_cache_only,
    assert_pcc,
    capture_decode_trace,
    decode_position_tensors,
    decoder_cache,
    layer_idx_for,
    make_page_table,
    mesh_device,
    reference_layers,
    signpost,
    to_device_hidden,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder import LAYER_KIND_SLIDING, FunctionalDecoder
from models.autoports.meta_models_muse_glimmer_30b.tt.fused_decoder import (
    MINIMAL_MATMUL_BLOCKS,
    MINIMAL_MATMUL_MIN_ROWS,
    PREFILL_SDPA_CHUNK,
    FusedDecoder,
    _minimal_matmul_config,
    norm_compute_kernel_config,
)
from models.common.utility_functions import comp_pcc

#: Fused-vs-unfused equivalence bar.  The rewrites are not bit-identical (a
#: dedicated kernel accumulates differently from the primitive sequence it
#: replaces), so equivalence is the *acceptance* bar from the graph-fusing
#: skill, not an arbitrarily tighter one.
EQUIVALENCE_PCC = 0.995

#: How much worse than the unfused graph the fused graph is allowed to be
#: against the HF reference, per mode.
#:
#: **Prefill: zero.**  Every measured prefill comparison improves, by +3.2e-4 to
#: +1.1e-3, so zero is what the evidence supports and any drift should be
#: re-examined rather than absorbed.  Note the two regimes: at 4097 and 12345
#: tokens prefill also swaps matmul kernel for a more accurate one at the same
#: math fidelity (rows >= ``MINIMAL_MATMUL_MIN_ROWS``), while at 100 tokens it
#: runs the *baseline's* kernel and improves purely from the RoPE, norm and
#: activation rewrites — i.e. the same class of rewrite decode has, and it still
#: lands on the right side.
#:
#: **Decode: 5e-4, and the margin is thin.**  Decode keeps ``ttnn.linear`` (a
#: step is 32 rows, below the ``minimal_matmul`` crossover), so its rewrites —
#: the RoPE op, the sharded norm and the activation merges — only *re-associate*
#: BF16 rounding rather than changing precision, and the result can land either
#: side of the baseline's.  Five of the six measured decode comparisons improve
#: and one drifts by -4.3e-4, so 5e-4 bounds the observed negative drift by only
#: about 1.15x: another ~0.7e-4 of re-association would fail this assertion.
#: That is deliberate — it is a tight guard, not a comfortable one — and it is
#: still about 6x inside the headroom from the suite's worst decode PCC
#: (0.998077) to the 0.995 acceptance bar — a full 5e-4 drift would land at
#: 0.997577.  See ``doc/fused_decoder/logs/pcc_summary.txt`` for all twelve
#: controls.
ACCURACY_REGRESSION_TOL = {"prefill": 0.0, "decode": 5e-4}


def build_fused(
    mesh_device,
    decoder_cache,
    kind: str,
    *,
    max_seq_len: int = SHORT_MAX_SEQ,
    max_batch_size: int = 1,
    real_weights: bool = False,
    chunk: int = PREFILL_CHUNK_SIZE,
) -> FusedDecoder:
    layer_idx = layer_idx_for(kind)
    key = ("fused", layer_idx, max_seq_len, max_batch_size, real_weights, chunk)

    def factory():
        state_dict = R.real_state_dict(layer_idx) if real_weights else R.synthetic_state_dict(layer_idx)
        return FusedDecoder.from_state_dict(
            state_dict,
            hf_config=R.hf_config(),
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            page_block_size=PAGE_BLOCK_SIZE,
            prefill_chunk_size=chunk,
        )

    return decoder_cache.get(key, factory)


# ------------------------------------------------------------------- prefill / decode


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("seq_len", PREFILL_SEQ_LENS)
def test_prefill_pcc(mesh_device, decoder_cache, reference_layers, kind, seq_len):
    """Same nine lengths the functional stage used, incl. non-aligned ones.

    1 / 100 / 2049 / 4097 / 8193 / 12345 are divisible by neither the tile
    height, the page block nor the internal prefill chunk, so the public API
    keeps accepting any logical length.
    """
    layer_idx, layer = reference_layers[kind]
    decoder = build_fused(mesh_device, decoder_cache, kind)
    hidden = R.synthetic_hidden_states(1, seq_len, seed=101 + seq_len)
    expected, _ = R.reference_prefill(layer, layer_idx, hidden)

    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ)
    tt_out = decoder.prefill_forward(to_device_hidden(mesh_device, hidden), page_table=page_table, user_id=0)
    actual = ttnn.to_torch(tt_out).reshape(1, seq_len, -1)
    assert tuple(tt_out.shape) == (1, 1, seq_len, decoder.config.hidden_size)
    assert_pcc(f"fused prefill[{kind}] seq_len={seq_len}", expected, actual)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("prompt_len", (100, 2048, 3000))
def test_decode_pcc(mesh_device, decoder_cache, reference_layers, kind, prompt_len):
    """Prefill, then decode four tokens past the prompt (paged update path)."""
    layer_idx, layer = reference_layers[kind]
    decoder = build_fused(mesh_device, decoder_cache, kind)
    hidden = R.synthetic_hidden_states(1, prompt_len, seed=202 + prompt_len)
    _, cache = R.reference_prefill(layer, layer_idx, hidden)

    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ)
    ttnn.deallocate(decoder.prefill_forward(to_device_hidden(mesh_device, hidden), page_table=page_table, user_id=0))

    for step in range(4):
        position = prompt_len + step
        token = R.synthetic_hidden_states(1, 1, seed=909 + step)
        expected = R.reference_decode(
            layer, layer_idx, token, past_key_values=cache, positions=torch.tensor([position])
        )
        current_pos, rope_pos_ids = decode_position_tensors(mesh_device, torch.tensor([position]))
        tt_out = decoder.decode_forward(
            to_device_hidden(mesh_device, token),
            current_pos=current_pos,
            page_table=page_table,
            rope_pos_ids=rope_pos_ids,
        )
        assert_pcc(
            f"fused decode[{kind}] prompt={prompt_len} pos={position}",
            expected,
            ttnn.to_torch(tt_out).reshape(1, 1, -1),
        )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_prefill_decode_pcc_vs_fp32_reference(mesh_device, decoder_cache, kind):
    """FP32 HF control: rules out an error common-mode to two BF16 graphs."""
    layer_idx = layer_idx_for(kind)
    layer = R.reference_layer(layer_idx, R.synthetic_state_dict(layer_idx), dtype=torch.float32)
    decoder = build_fused(mesh_device, decoder_cache, kind)

    seq_len = 2049
    hidden_bf16 = R.synthetic_hidden_states(1, seq_len, seed=606060)
    expected, cache = R.reference_prefill(layer, layer_idx, hidden_bf16.float())

    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ, seed=6060)
    tt_out = decoder.prefill_forward(to_device_hidden(mesh_device, hidden_bf16), page_table=page_table, user_id=0)
    assert_pcc(
        f"fused prefill[{kind}] vs FP32 HF reference seq_len={seq_len}",
        expected,
        ttnn.to_torch(tt_out).reshape(1, seq_len, -1),
    )
    ttnn.deallocate(tt_out)

    token_bf16 = R.synthetic_hidden_states(1, 1, seed=606061)
    expected = R.reference_decode(
        layer, layer_idx, token_bf16.float(), past_key_values=cache, positions=torch.tensor([seq_len])
    )
    current_pos, rope_pos_ids = decode_position_tensors(mesh_device, torch.tensor([seq_len]))
    tt_out = decoder.decode_forward(
        to_device_hidden(mesh_device, token_bf16),
        current_pos=current_pos,
        page_table=page_table,
        rope_pos_ids=rope_pos_ids,
    )
    assert_pcc(
        f"fused decode[{kind}] vs FP32 HF reference pos={seq_len}",
        expected,
        ttnn.to_torch(tt_out).reshape(1, 1, -1),
    )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_prefill_seq_len_equals_max_and_chunk(mesh_device, decoder_cache, reference_layers, kind):
    """``max_seq_len == prefill_chunk_size == seq_len``.

    The fused prefill hands its *persistent* pre-tilized RoPE tables straight to
    ``rotary_embedding_hf`` at ``start_pos == 0`` (no slice at all), so a stray
    deallocate of the op's inputs would destroy the layer.  Two prefills on the
    same decoder prove the tables survive.
    """
    layer_idx, layer = reference_layers[kind]
    seq_len = 4096
    decoder = build_fused(mesh_device, decoder_cache, kind, max_seq_len=seq_len, chunk=seq_len)
    page_table = make_page_table(mesh_device, 1, seq_len, seed=771)
    for attempt in range(2):
        hidden = R.synthetic_hidden_states(1, seq_len, seed=880 + attempt)
        expected, _ = R.reference_prefill(layer, layer_idx, hidden)
        tt_out = decoder.prefill_forward(to_device_hidden(mesh_device, hidden), page_table=page_table, user_id=0)
        assert_pcc(
            f"fused prefill[{kind}] seq_len==max_seq_len==chunk=={seq_len} attempt={attempt}",
            expected,
            ttnn.to_torch(tt_out).reshape(1, seq_len, -1),
        )
        ttnn.deallocate(tt_out)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_multi_chunk_prefill_page_table_bound(mesh_device, decoder_cache, reference_layers, kind):
    """Multi-chunk prefill at a ``max_seq_len`` whose page count is awkward.

    The paged ``chunked_scaled_dot_product_attention`` requires
    ``kv_length >= padded_q + chunk_start_idx``, and ``kv_length`` comes from the
    *page table's* width — so the Q padding the SDPA chunk size implies must not
    reach past the end of the user's pages.  ``max_seq_len = 12416`` is 194
    blocks: a full-length prefill's second internal chunk is 4224 tokens at
    ``start_pos = 8192``, which the fused layer's 256-token SDPA chunk would pad
    to 4352 and overrun (8192 + 4352 = 12544 > 12416).  ``_prefill_sdpa_full``
    halves the chunk until both the alignment and the page-table bound hold; this
    pins that, and every other tested ``max_seq_len`` (4096 / 16384 / 131072) has
    a block count that hides it.
    """
    layer_idx, layer = reference_layers[kind]
    max_seq_len = 12416
    decoder = build_fused(mesh_device, decoder_cache, kind, max_seq_len=max_seq_len)
    page_table = make_page_table(mesh_device, 1, max_seq_len, seed=1216)
    hidden = R.synthetic_hidden_states(1, max_seq_len, seed=12416)
    expected, _ = R.reference_prefill(layer, layer_idx, hidden)

    tt_out = decoder.prefill_forward(to_device_hidden(mesh_device, hidden), page_table=page_table, user_id=0)
    assert tuple(tt_out.shape) == (1, 1, max_seq_len, decoder.config.hidden_size)
    assert_pcc(
        f"fused prefill[{kind}] max_seq_len==seq_len==12416 (194 pages, 2 chunks)",
        expected,
        ttnn.to_torch(tt_out).reshape(1, max_seq_len, -1),
    )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("first_len,second_len", CONTINUATION_SPLITS)
def test_continuation_prefill_pcc(mesh_device, decoder_cache, reference_layers, kind, first_len, second_len):
    """Caller-chunked prefill: two ``start_pos``-separated calls == one call.

    Also the only test that drives the fused prefill's *sliced* RoPE-table path
    (``start_pos > 0``), since the single-shot path uses the persistent tables.
    """
    layer_idx, layer = reference_layers[kind]
    decoder = build_fused(mesh_device, decoder_cache, kind)
    total = first_len + second_len
    hidden = R.synthetic_hidden_states(1, total, seed=90210 + first_len)
    expected, _ = R.reference_prefill(layer, layer_idx, hidden)

    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ, seed=4242)
    first_out, tail = decoder.prefill_forward(
        to_device_hidden(mesh_device, hidden[:, :first_len]),
        page_table=page_table,
        user_id=0,
        return_sliding_kv_tail=True,
    )
    assert (tail is not None) == (kind == LAYER_KIND_SLIDING)
    if tail is not None:
        assert tuple(tail[0].shape) == (
            1,
            decoder.config.num_key_value_heads,
            decoder.sliding_kv_tail_len(first_len),
            decoder.config.head_dim,
        )

    second_out = decoder.prefill_forward(
        to_device_hidden(mesh_device, hidden[:, first_len:]),
        page_table=page_table,
        user_id=0,
        start_pos=first_len,
        sliding_kv_tail=tail,
    )
    actual = torch.cat(
        [
            ttnn.to_torch(first_out).reshape(1, first_len, -1),
            ttnn.to_torch(second_out).reshape(1, second_len, -1),
        ],
        dim=1,
    )
    assert_pcc(f"fused continuation prefill[{kind}] {first_len}+{second_len}", expected, actual)

    _, cache = R.reference_prefill(layer, layer_idx, hidden)
    token = R.synthetic_hidden_states(1, 1, seed=90211)
    expected_decode = R.reference_decode(
        layer, layer_idx, token, past_key_values=cache, positions=torch.tensor([total])
    )
    current_pos, rope_pos_ids = decode_position_tensors(mesh_device, torch.tensor([total]))
    tt_out = decoder.decode_forward(
        to_device_hidden(mesh_device, token),
        current_pos=current_pos,
        page_table=page_table,
        rope_pos_ids=rope_pos_ids,
    )
    assert_pcc(
        f"fused decode after continuation prefill[{kind}] pos={total}",
        expected_decode,
        ttnn.to_torch(tt_out).reshape(1, 1, -1),
    )


@pytest.mark.timeout(600)
def test_continuation_prefill_requires_sliding_tail(mesh_device, decoder_cache, expect_error):
    """A sliding continuation without its window must fail loudly, not silently."""
    decoder = build_fused(mesh_device, decoder_cache, LAYER_KIND_SLIDING)
    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ, seed=4243)
    tt_hidden = to_device_hidden(mesh_device, R.synthetic_hidden_states(1, 128, seed=1))
    with expect_error(ValueError, "sliding_kv_tail"):
        decoder.prefill_forward(tt_hidden, page_table=page_table, user_id=0, start_pos=4096)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("batch", (4, 13, 32))
def test_batched_prefill_decode_pcc(mesh_device, decoder_cache, reference_layers, kind, batch):
    """Independent users share one paged cache: per-user prefill then batched decode.

    ``batch=13`` is prime and larger than the 11-wide grid, so no ``batch``-core
    rectangle exists and the decode head-concat takes its shape-agnostic
    fallback.  It also pins the fused decode's *sharded* RoPE gather and
    width-sharded norms at a batch that is neither 1 nor tile-aligned.
    """
    layer_idx, layer = reference_layers[kind]
    max_seq_len = 4096
    decoder = build_fused(mesh_device, decoder_cache, kind, max_seq_len=max_seq_len, max_batch_size=batch)
    page_table = make_page_table(mesh_device, batch, max_seq_len, seed=31 + batch)

    prompt_lens = [2000 + 37 * user for user in range(batch)]
    assert (decoder._decode_concat_grid_width(batch) is None) == (batch == 13)
    caches = []
    for user, prompt_len in enumerate(prompt_lens):
        hidden = R.synthetic_hidden_states(1, prompt_len, seed=4000 + user)
        expected, cache = R.reference_prefill(layer, layer_idx, hidden)
        caches.append(cache)
        tt_out = decoder.prefill_forward(to_device_hidden(mesh_device, hidden), page_table=page_table, user_id=user)
        assert_pcc(
            f"fused prefill[{kind}] batch={batch} user={user} seq_len={prompt_len}",
            expected,
            ttnn.to_torch(tt_out).reshape(1, prompt_len, -1),
        )
        ttnn.deallocate(tt_out)

    positions = torch.tensor(prompt_lens, dtype=torch.int32)
    tokens = R.synthetic_hidden_states(batch, 1, seed=8123)
    expected = torch.cat(
        [
            R.reference_decode(
                layer,
                layer_idx,
                tokens[user : user + 1],
                past_key_values=caches[user],
                positions=positions[user : user + 1],
            )
            for user in range(batch)
        ],
        dim=0,
    )
    current_pos, rope_pos_ids = decode_position_tensors(mesh_device, positions)
    tt_out = decoder.decode_forward(
        to_device_hidden(mesh_device, tokens),
        current_pos=current_pos,
        page_table=page_table,
        rope_pos_ids=rope_pos_ids,
    )
    assert_pcc(
        f"fused decode[{kind}] batch={batch} ragged positions",
        expected,
        ttnn.to_torch(tt_out).reshape(batch, 1, -1),
    )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_multi_chunk_prefill_nonzero_user(mesh_device, decoder_cache, reference_layers, kind):
    """Multi-chunk prefill into a non-zero cache slot, then decode that slot."""
    layer_idx, layer = reference_layers[kind]
    seq_len = 12345
    decoder = build_fused(mesh_device, decoder_cache, kind, max_seq_len=SHORT_MAX_SEQ, max_batch_size=4)
    page_table = make_page_table(mesh_device, 4, SHORT_MAX_SEQ, seed=555)
    hidden = R.synthetic_hidden_states(1, seq_len, seed=13579)
    expected, cache = R.reference_prefill(layer, layer_idx, hidden)

    tt_out = decoder.prefill_forward(to_device_hidden(mesh_device, hidden), page_table=page_table, user_id=2)
    assert_pcc(
        f"fused multi-chunk prefill[{kind}] user_id=2 seq_len={seq_len}",
        expected,
        ttnn.to_torch(tt_out).reshape(1, seq_len, -1),
    )
    ttnn.deallocate(tt_out)

    positions = torch.tensor([0, 0, seq_len, 0], dtype=torch.int32)
    tokens = R.synthetic_hidden_states(4, 1, seed=13580)
    expected = R.reference_decode(layer, layer_idx, tokens[2:3], past_key_values=cache, positions=positions[2:3])
    current_pos, rope_pos_ids = decode_position_tensors(mesh_device, positions)
    tt_out = decoder.decode_forward(
        to_device_hidden(mesh_device, tokens),
        current_pos=current_pos,
        page_table=page_table,
        rope_pos_ids=rope_pos_ids,
    )
    assert_pcc(
        f"fused decode[{kind}] user_id=2 pos={seq_len}",
        expected,
        ttnn.to_torch(tt_out).reshape(4, 1, -1)[2:3],
    )


# ------------------------------------------------------------------- full 128k context


@pytest.mark.slow
@pytest.mark.timeout(5400)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("seq_len", (HF_ADVERTISED_CONTEXT, HF_ADVERTISED_CONTEXT - 999))
def test_full_context_prefill_tail_pcc(mesh_device, decoder_cache, reference_layers, kind, seq_len):
    """HF-vs-TTNN prefill PCC at (and just under) the advertised 131072 context."""
    from transformers.cache_utils import DynamicCache

    layer_idx, layer = reference_layers[kind]
    decoder = build_fused(mesh_device, decoder_cache, kind, max_seq_len=HF_ADVERTISED_CONTEXT)
    tail = 32
    hidden = R.synthetic_hidden_states(1, seq_len, seed=555 + seq_len)

    page_table = make_page_table(mesh_device, 1, HF_ADVERTISED_CONTEXT, seed=99)
    tt_hidden = to_device_hidden(mesh_device, hidden)
    tt_out = decoder.prefill_forward(tt_hidden, page_table=page_table, user_id=0)
    ttnn.deallocate(tt_hidden)

    offsets = [seq_len - tail]
    if seq_len == HF_ADVERTISED_CONTEXT:
        offsets.insert(0, 8 * PREFILL_CHUNK_SIZE)
    for offset in offsets:
        tt_rows = ttnn.slice(tt_out, [0, 0, offset, 0], [1, 1, offset + tail, decoder.config.hidden_size])
        actual = ttnn.to_torch(tt_rows).reshape(1, tail, -1)
        ttnn.deallocate(tt_rows)

        cache = DynamicCache(config=R.text_config())
        _reference_prefill_cache_only(layer, layer_idx, hidden[:, :offset], cache)
        expected, _ = R.reference_prefill(
            layer, layer_idx, hidden[:, offset : offset + tail], past_key_values=cache, start_pos=offset
        )
        where = "last" if offset == seq_len - tail else f"interior @{offset}"
        assert_pcc(f"fused prefill[{kind}] full-context seq_len={seq_len} ({where} {tail} rows)", expected, actual)
    ttnn.deallocate(tt_out)


@pytest.mark.slow
@pytest.mark.timeout(5400)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_full_context_decode_pcc(mesh_device, decoder_cache, reference_layers, kind):
    """Decode at the last valid position of the advertised context (131071)."""
    from transformers.cache_utils import DynamicCache

    layer_idx, layer = reference_layers[kind]
    decoder = build_fused(mesh_device, decoder_cache, kind, max_seq_len=HF_ADVERTISED_CONTEXT)
    prompt_len = HF_ADVERTISED_CONTEXT - 1
    hidden = R.synthetic_hidden_states(1, prompt_len, seed=777)

    page_table = make_page_table(mesh_device, 1, HF_ADVERTISED_CONTEXT, seed=123)
    tt_hidden = to_device_hidden(mesh_device, hidden)
    ttnn.deallocate(decoder.prefill_forward(tt_hidden, page_table=page_table, user_id=0))
    ttnn.deallocate(tt_hidden)

    cache = DynamicCache(config=R.text_config())
    _reference_prefill_cache_only(layer, layer_idx, hidden, cache)

    position = prompt_len
    token = R.synthetic_hidden_states(1, 1, seed=778)
    expected = R.reference_decode(layer, layer_idx, token, past_key_values=cache, positions=torch.tensor([position]))
    current_pos, rope_pos_ids = decode_position_tensors(mesh_device, torch.tensor([position]))
    tt_out = decoder.decode_forward(
        to_device_hidden(mesh_device, token),
        current_pos=current_pos,
        page_table=page_table,
        rope_pos_ids=rope_pos_ids,
    )
    assert_pcc(f"fused decode[{kind}] full-context pos={position}", expected, ttnn.to_torch(tt_out).reshape(1, 1, -1))


# ----------------------------------------------------------------------- real weights


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_real_weights_prefill_decode_pcc(mesh_device, decoder_cache, kind):
    """Same contract, exercised against the released bf16 checkpoint."""
    layer_idx = layer_idx_for(kind)
    try:
        state_dict = R.real_state_dict(layer_idx)
    except FileNotFoundError as error:  # pragma: no cover - weights not cached
        if os.environ.get("MG_ALLOW_MISSING_WEIGHTS", "0") == "1":
            pytest.skip(str(error))
        raise
    layer = R.reference_layer(layer_idx, state_dict)
    decoder = build_fused(mesh_device, decoder_cache, kind, real_weights=True)

    seq_len = 2049
    hidden = R.synthetic_hidden_states(1, seq_len, seed=31337)
    expected, cache = R.reference_prefill(layer, layer_idx, hidden)

    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ, seed=64)
    tt_out = decoder.prefill_forward(to_device_hidden(mesh_device, hidden), page_table=page_table, user_id=0)
    assert_pcc(
        f"fused real-weights prefill[{kind}] seq_len={seq_len}",
        expected,
        ttnn.to_torch(tt_out).reshape(1, seq_len, -1),
    )
    ttnn.deallocate(tt_out)

    token = R.synthetic_hidden_states(1, 1, seed=31338)
    expected = R.reference_decode(layer, layer_idx, token, past_key_values=cache, positions=torch.tensor([seq_len]))
    current_pos, rope_pos_ids = decode_position_tensors(mesh_device, torch.tensor([seq_len]))
    tt_out = decoder.decode_forward(
        to_device_hidden(mesh_device, token),
        current_pos=current_pos,
        page_table=page_table,
        rope_pos_ids=rope_pos_ids,
    )
    assert_pcc(f"fused real-weights decode[{kind}] pos={seq_len}", expected, ttnn.to_torch(tt_out).reshape(1, 1, -1))


# ------------------------------------------------------- fused vs unfused equivalence


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("seq_len", (100, 4097, 12345))
def test_fused_vs_functional_equivalence(mesh_device, decoder_cache, reference_layers, kind, seq_len):
    """The rewrite is equivalence-preserving *and* does not lose accuracy.

        Two assertions, because either alone is weak:

        1. **Equivalence** — fused output vs *unfused* output on device, at the
           stage acceptance bar.  An HF-only comparison could hide a rewrite that is
           subtly worse, since both graphs are compared to HF at 0.995.
        2. **No accuracy regression** — the fused graph must be at least as close to
           the HF reference as the unfused graph is: *strictly* in prefill, and
           within a documented BF16 re-association band in decode (see
           ``ACCURACY_REGRESSION_TOL``).  This is the assertion that actually has
           teeth: the two TTNN graphs differ from each other by roughly the
           *unfused* graph's own BF16 error (~0.998), so "fused == unfused" would be
           the wrong thing to demand.

    12345 forces the multi-chunk prefill (sliced RoPE tables, sliding tail
        carry, paged chunked SDPA) and 4097 a single ``minimal_matmul`` chunk, while
        100 is sub-tile and below the ``_dense`` row threshold, so the two sides of
        the dense-kernel dispatch are both compared.
    """
    layer_idx, layer = reference_layers[kind]
    state_dict = R.synthetic_state_dict(layer_idx)

    def build(cls, tag):
        def factory():
            return cls.from_state_dict(
                state_dict,
                hf_config=R.hf_config(),
                layer_idx=layer_idx,
                mesh_device=mesh_device,
                max_batch_size=1,
                max_seq_len=SHORT_MAX_SEQ,
                page_block_size=PAGE_BLOCK_SIZE,
                prefill_chunk_size=PREFILL_CHUNK_SIZE,
            )

        return decoder_cache.get((tag, layer_idx, SHORT_MAX_SEQ, 1, False, PREFILL_CHUNK_SIZE), factory)

    hidden = R.synthetic_hidden_states(1, seq_len, seed=4242 + seq_len)
    token = R.synthetic_hidden_states(1, 1, seed=4243)
    hf_prefill, hf_cache = R.reference_prefill(layer, layer_idx, hidden)
    hf_decode = R.reference_decode(layer, layer_idx, token, past_key_values=hf_cache, positions=torch.tensor([seq_len]))

    outputs = {}
    for cls, tag in ((FunctionalDecoder, "equiv-functional"), (FusedDecoder, "fused")):
        decoder = build(cls, tag)
        page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ, seed=808)
        tt_out = decoder.prefill_forward(to_device_hidden(mesh_device, hidden), page_table=page_table, user_id=0)
        prefill = ttnn.to_torch(tt_out).reshape(1, seq_len, -1).clone()
        ttnn.deallocate(tt_out)
        current_pos, rope_pos_ids = decode_position_tensors(mesh_device, torch.tensor([seq_len]))
        tt_out = decoder.decode_forward(
            to_device_hidden(mesh_device, token),
            current_pos=current_pos,
            page_table=page_table,
            rope_pos_ids=rope_pos_ids,
        )
        outputs[tag] = (prefill, ttnn.to_torch(tt_out).reshape(1, 1, -1).clone())
        ttnn.deallocate(tt_out)

    unfused, fused = outputs["equiv-functional"], outputs["fused"]
    for index, (mode, reference) in enumerate((("prefill", hf_prefill), ("decode", hf_decode))):
        assert_pcc(
            f"fused vs unfused {mode}[{kind}] seq_len={seq_len}",
            unfused[index],
            fused[index],
            threshold=EQUIVALENCE_PCC,
        )
        unfused_vs_hf = comp_pcc(reference.float(), unfused[index].float(), 0.0)[1]
        fused_vs_hf = comp_pcc(reference.float(), fused[index].float(), 0.0)[1]
        logger.info(
            f"accuracy vs HF {mode}[{kind}] seq_len={seq_len}: "
            f"unfused={unfused_vs_hf:.6f} fused={fused_vs_hf:.6f} delta={fused_vs_hf - unfused_vs_hf:+.6f}"
        )
        assert fused_vs_hf >= unfused_vs_hf - ACCURACY_REGRESSION_TOL[mode], (
            f"fused {mode}[{kind}] seq_len={seq_len} is less accurate than the unfused graph by more "
            f"than the {ACCURACY_REGRESSION_TOL[mode]} {mode} tolerance: {fused_vs_hf} < {unfused_vs_hf}"
        )


# ------------------------------------------------------------------- fused-graph audit


class _OpTrace:
    """Record which TTNN ops a forward pass actually dispatches.

    ``merged_activations`` additionally counts ``ttnn.mul`` calls that carry an
    input activation, i.e. the *positive* form of the SiLU/sigmoid op merge.
    Asserting only that ``ttnn.silu``/``ttnn.sigmoid`` are absent would also
    pass if the activations had been dropped altogether.
    """

    def __init__(self, names: tuple[str, ...]):
        self.names = names
        self.counts: dict[str, int] = {}
        self.merged_activations = 0
        self._saved: list[tuple[object, str, object]] = []

    def _resolve(self, dotted: str):
        owner = ttnn
        parts = dotted.split(".")
        for part in parts[:-1]:
            owner = getattr(owner, part)
        return owner, parts[-1]

    def __enter__(self):
        for dotted in self.names:
            owner, attr = self._resolve(dotted)
            original = getattr(owner, attr)
            self._saved.append((owner, attr, original))

            def traced(*args, _dotted=dotted, _original=original, **kwargs):
                self.counts[_dotted] = self.counts.get(_dotted, 0) + 1
                if _dotted == "mul" and (
                    kwargs.get("input_tensor_a_activations") or kwargs.get("input_tensor_b_activations")
                ):
                    self.merged_activations += 1
                return _original(*args, **kwargs)

            setattr(owner, attr, traced)
        return self

    def __exit__(self, *exc):
        for owner, attr, original in self._saved:
            setattr(owner, attr, original)
        return False


#: Ops the fused graph must dispatch, per mode.  ``minimal_matmul`` is handled
#: separately because it is row-count dependent: it must appear in prefill only
#: at or above ``MINIMAL_MATMUL_MIN_ROWS`` and never in decode (where
#: ``ttnn.linear``'s DRAM-sharded config is 1.5x faster).  The sharded-residual
#: reshards must appear in decode and must NOT appear in prefill.
_REQUIRED_OPS = {
    "prefill": ("experimental.rotary_embedding_hf",),
    "decode": ("experimental.rotary_embedding_hf", "interleaved_to_sharded", "sharded_to_interleaved"),
}
#: Ops that must NOT appear in a given mode (on top of ``_FORBIDDEN_OPS``).
_MODE_FORBIDDEN_OPS = {
    "prefill": ("interleaved_to_sharded", "sharded_to_interleaved"),
    "decode": ("experimental.minimal_matmul",),
}
#: Ops the fusing removed.  ``neg``/``concat`` were the rotate-half spelling,
#: ``silu``/``sigmoid`` are now matmul pack-time activations, ``repeat`` was the
#: decode cos/sin broadcast, ``to_layout``/``tilize``/``untilize``/``typecast``
#: were layout churn, and ``from_torch``/``to_torch`` would be a host fallback.
_FORBIDDEN_OPS = (
    "silu",
    "sigmoid",
    "repeat",
    "to_layout",
    "tilize",
    "untilize",
    "typecast",
    "from_torch",
    "to_torch",
    "as_tensor",
)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("seq_len", (2048, 3000, 12345))
def test_fused_graph_uses_fused_ops(mesh_device, decoder_cache, kind, seq_len):
    """The fused ops are dispatched and the sequences they replaced are gone.

    Without this, a regression to the functional graph would still pass every
    PCC test in this file — the point of the stage is the graph, so the graph is
    asserted directly.  ``12345`` covers the multi-chunk prefill paths, and the
    three lengths straddle the ``_dense`` row-count crossover (2048 and 3000 pad
    to 2048/3008 rows -> ``ttnn.linear``; 12345 chunks to 8192 + 4160 rows ->
    ``minimal_matmul``), so the dispatch rule itself is asserted rather than
    assumed.
    """
    decoder = build_fused(mesh_device, decoder_cache, kind)
    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ, seed=5)
    tt_hidden = to_device_hidden(mesh_device, R.synthetic_hidden_states(1, seq_len, seed=606))
    token = to_device_hidden(mesh_device, R.synthetic_hidden_states(1, 1, seed=607))
    current_pos, rope_pos_ids = decode_position_tensors(mesh_device, torch.tensor([seq_len]))

    watched = tuple(
        dict.fromkeys(
            sum(_REQUIRED_OPS.values(), ()) + sum(_MODE_FORBIDDEN_OPS.values(), ()) + _FORBIDDEN_OPS + ("mul",)
        )
    )
    with _OpTrace(watched) as prefill_trace:
        ttnn.deallocate(decoder.prefill_forward(tt_hidden, page_table=page_table, user_id=0))
    with _OpTrace(watched) as decode_trace:
        ttnn.deallocate(
            decoder.decode_forward(token, current_pos=current_pos, page_table=page_table, rope_pos_ids=rope_pos_ids)
        )

    logger.info(f"fused prefill[{kind}] op trace: {prefill_trace.counts}")
    logger.info(f"fused decode[{kind}] op trace: {decode_trace.counts}")

    # The dense kernel is chosen by chunk row count, so assert the choice, not
    # just its presence: a prefill chunk is min(chunk_size, padded_len) rows.
    padded = -(-seq_len // 32) * 32
    chunk_rows = [min(PREFILL_CHUNK_SIZE, padded - offset) for offset in range(0, padded, PREFILL_CHUNK_SIZE)]
    expected_minimal = sum(1 for rows in chunk_rows if rows >= MINIMAL_MATMUL_MIN_ROWS) * 6
    assert prefill_trace.counts.get("experimental.minimal_matmul", 0) == expected_minimal, (
        f"prefill[{kind}] seq_len={seq_len} (chunks {chunk_rows}) dispatched "
        f"{prefill_trace.counts.get('experimental.minimal_matmul', 0)} minimal_matmul calls, expected "
        f"{expected_minimal} (6 dense projections per chunk at or above {MINIMAL_MATMUL_MIN_ROWS} rows)"
    )

    for mode, trace in (("prefill", prefill_trace), ("decode", decode_trace)):
        for required in _REQUIRED_OPS[mode]:
            if required == "experimental.rotary_embedding_hf" and kind != LAYER_KIND_SLIDING:
                continue  # full layers are NoPE: there is no rotary embedding at all
            assert trace.counts.get(required), f"{mode} did not dispatch {required}: {trace.counts}"
        forbidden = _FORBIDDEN_OPS + _MODE_FORBIDDEN_OPS[mode]
        used = {name for name in forbidden if trace.counts.get(name)}
        assert not used, f"fused {mode} still dispatches unfused/host/wrong-mode ops: {sorted(used)}"
        # One merged activation per prefill chunk for the attention gate plus
        # one for the SwiGLU; decode has exactly one of each.
        expected_merges = 2 * max(1, -(-seq_len // PREFILL_CHUNK_SIZE)) if mode == "prefill" else 2
        assert trace.merged_activations == expected_merges, (
            f"fused {mode} applied {trace.merged_activations} merged input activations, expected "
            f"{expected_merges} (the SiLU and sigmoid merges are how the unary ops were removed)"
        )
    # full (NoPE) layers must not gain a rotary op from the fusing either.
    if kind != LAYER_KIND_SLIDING:
        assert not decode_trace.counts.get("experimental.rotary_embedding_hf")


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("seq_len", (3000, 12345))
def test_no_host_fallback_in_forward(mesh_device, decoder_cache, kind, seq_len):
    """No torch / host round-trip inside a measured prefill or decode."""
    decoder = build_fused(mesh_device, decoder_cache, kind)
    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ, seed=5)
    tt_hidden = to_device_hidden(mesh_device, R.synthetic_hidden_states(1, seq_len, seed=606))
    token = to_device_hidden(mesh_device, R.synthetic_hidden_states(1, 1, seed=607))
    current_pos, rope_pos_ids = decode_position_tensors(mesh_device, torch.tensor([seq_len]))

    with _FallbackGuard() as guard:
        ttnn.deallocate(decoder.prefill_forward(tt_hidden, page_table=page_table, user_id=0))
        ttnn.deallocate(
            decoder.decode_forward(token, current_pos=current_pos, page_table=page_table, rope_pos_ids=rope_pos_ids)
        )
    assert not guard.violations, f"host fallback inside a measured pass: {sorted(set(guard.violations))}"


# ------------------------------------------------------------ determinism and stress


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_determinism_repeated_inputs(mesh_device, decoder_cache, kind):
    """Identical inputs must produce bit-identical prefill and decode outputs."""
    decoder = build_fused(mesh_device, decoder_cache, kind)
    seq_len = 1024
    hidden = R.synthetic_hidden_states(1, seq_len, seed=246)
    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ, seed=17)

    runs = []
    for _ in range(3):
        tt_out = decoder.prefill_forward(to_device_hidden(mesh_device, hidden), page_table=page_table, user_id=0)
        runs.append(ttnn.to_torch(tt_out).clone())
        ttnn.deallocate(tt_out)
    assert torch.equal(runs[0], runs[1]) and torch.equal(runs[1], runs[2]), "prefill is not deterministic"

    token = R.synthetic_hidden_states(1, 1, seed=247)
    decode_runs = []
    for _ in range(3):
        current_pos, rope_pos_ids = decode_position_tensors(mesh_device, torch.tensor([seq_len]))
        tt_out = decoder.decode_forward(
            to_device_hidden(mesh_device, token),
            current_pos=current_pos,
            page_table=page_table,
            rope_pos_ids=rope_pos_ids,
        )
        decode_runs.append(ttnn.to_torch(tt_out).clone())
        ttnn.deallocate(tt_out)
    assert torch.equal(decode_runs[0], decode_runs[1]) and torch.equal(
        decode_runs[1], decode_runs[2]
    ), "decode is not deterministic"


STRESS_STEPS = int(os.environ.get("MG_STRESS_STEPS", "64"))


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_repeated_run_stress(mesh_device, decoder_cache, reference_layers, kind):
    """Long repeated-run soak over the fused path.

    ``STRESS_STEPS`` traced decode replays advancing the position every step,
    interleaved with re-prefills of the same user, then a final HF-referenced
    decode.  This is where a leaked buffer or a trace-lifetime bug in the new
    L1-resident residual stream would show up — none of the single-shot PCC
    tests would.

    (The decode sharded-norm config cache is keyed on tile-padded rows, and
    every batch the advertised contract allows — 1 to 32 — rounds to the same
    32 rows, so the cache legitimately only ever holds one entry.
    ``test_decode_norm_config_shapes`` pins that mapping directly.)
    """
    layer_idx, layer = reference_layers[kind]
    decoder = build_fused(mesh_device, decoder_cache, kind)
    prompt_len = 1024
    hidden = R.synthetic_hidden_states(1, prompt_len, seed=515)
    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ, seed=616)

    for _ in range(4):
        ttnn.deallocate(
            decoder.prefill_forward(to_device_hidden(mesh_device, hidden), page_table=page_table, user_id=0)
        )
    _, cache = R.reference_prefill(layer, layer_idx, hidden)

    tt_token = to_device_hidden(mesh_device, R.synthetic_hidden_states(1, 1, seed=1))
    current_pos, rope_pos_ids = decode_position_tensors(mesh_device, torch.tensor([prompt_len]))
    trace_id, tt_out = capture_decode_trace(decoder, mesh_device, tt_token, current_pos, page_table, rope_pos_ids)

    hidden_size = decoder.config.hidden_size
    last = None
    for step in range(STRESS_STEPS):
        position = prompt_len + step
        token = R.synthetic_hidden_states(1, 1, seed=2000 + step)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(token.reshape(1, 1, 1, hidden_size), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
            tt_token,
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.tensor([position], dtype=torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32
            ),
            current_pos,
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.tensor([[position]], dtype=torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32
            ),
            rope_pos_ids,
        )
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        actual = ttnn.to_torch(tt_out).reshape(1, 1, -1).clone()
        assert torch.isfinite(actual.float()).all(), f"non-finite decode output at step {step}"
        expected = R.reference_decode(
            layer, layer_idx, token, past_key_values=cache, positions=torch.tensor([position])
        )
        last = (step, position, expected, actual)
        if step % 16 == 0 or step == STRESS_STEPS - 1:
            assert_pcc(f"fused stress decode[{kind}] step={step} pos={position}", expected, actual)
    ttnn.release_trace(mesh_device, trace_id)
    assert last is not None and last[0] == STRESS_STEPS - 1


@pytest.mark.timeout(600)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_decode_norm_config_shapes(mesh_device, decoder_cache, kind):
    """The width-sharded decode norm config is legal for every allowed batch.

    ``nlp_create_qkv_heads_decode`` hard-caps decode at 32 users
    (``..._device_operation.cpp:45-51``), and the config is cached on tile-padded
    row count, so every supported batch shares one 32-row entry — the cache
    legitimately never holds more than one at runtime. The 33/64 rows here are a
    shape check on the *derivation* (a second, 64-row entry must also come out
    tile-aligned, evenly divided over the grid, and matched by its program
    config), not a claim that those batches are supported.
    """
    decoder = build_fused(mesh_device, decoder_cache, kind)
    grid_x, grid_y = decoder.decode_norm_grid
    cores = grid_x * grid_y
    dim = decoder.config.hidden_size
    assert dim % (cores * 32) == 0, f"decode norm grid {grid_x}x{grid_y} does not divide {dim} into tiles"
    seen = {}
    for batch in (1, 4, 13, 32, 33, 64):
        rows = ((batch + 32 - 1) // 32) * 32
        program_config, memory_config = decoder._decode_norm_configs(rows)
        seen[rows] = memory_config
        shard = memory_config.shard_spec
        assert shard.grid.num_cores() == cores
        assert tuple(shard.shape) == (rows, dim // cores)
        assert program_config.block_h == rows // 32
        assert program_config.block_w == dim // cores // 32
    assert len(seen) == 2, f"expected two distinct row shapes across batches 1..64, got {sorted(seen)}"
    assert len(decoder._decode_norm_cache) == len(seen)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_prefill_sdpa_chunk_sizes(mesh_device, decoder_cache, kind):
    """Both prefill SDPA call sites really run the retuned chunk.

    The retune is a shipped constant that no PCC test can see: reverting either
    call site to 128 would keep every other assertion in this file green and only
    show up in a Tracy capture. This pins both, plus the two constraints that
    make the paged site shrink it (`chunk_start_idx` alignment and the
    page-table-derived KV bound).
    """
    decoder = build_fused(mesh_device, decoder_cache, kind)
    blocks = SHORT_MAX_SEQ // PAGE_BLOCK_SIZE

    # The retuned value itself, not just "whatever the constant says": reverting
    # it to the functional layer's 128 has to fail here.
    assert PREFILL_SDPA_CHUNK == 256

    # In-memory site: 256 for any slice at least that long, clamped below it.
    assert decoder._prefill_program_config(8192).q_chunk_size == PREFILL_SDPA_CHUNK
    assert decoder._prefill_program_config(8192).k_chunk_size == PREFILL_SDPA_CHUNK
    assert decoder._prefill_program_config(64).q_chunk_size == 64

    # Paged site: 256 at the aligned offsets an 8192-chunked prefill produces
    # (both reachable at max_seq_len == SHORT_MAX_SEQ == 16384)...
    assert decoder.chunked_sdpa_chunk_size(8192, 8192, blocks) == PREFILL_SDPA_CHUNK
    assert decoder.chunked_sdpa_chunk_size(8192, 4096, blocks) == PREFILL_SDPA_CHUNK
    # ...halved when the caller's continuation start_pos is not a multiple of it
    # (1024 is, so it stays at 256; 64 and 192 are not)...
    assert decoder.chunked_sdpa_chunk_size(1024, 512, blocks) == PREFILL_SDPA_CHUNK
    assert decoder.chunked_sdpa_chunk_size(64, 100, blocks) == 64
    assert decoder.chunked_sdpa_chunk_size(192, 100, blocks) == 64
    # ...and halved when the implied Q padding would run past the page table.
    tight_blocks = 12416 // PAGE_BLOCK_SIZE
    assert decoder.chunked_sdpa_chunk_size(8192, 4224, tight_blocks) == 128
    assert 8192 + 4224 <= tight_blocks * PAGE_BLOCK_SIZE


def test_norm_compute_kernel_config_is_the_documented_uplift(mesh_device):
    """The one fidelity change in this layer is the one that was measured.

    ``ttnn.rms_norm``'s default is ``HiFi4 / approx=True / fp32_dest_acc_en=False``
    (``rmsnorm.cpp:16-20``); this layer turns the approximation off and FP32
    accumulation on, which is worth ~+3.5e-4 of the prefill accuracy gain at
    short lengths and is free in decode. It is a deliberate departure from the
    functional layer's "no config at all", so it is asserted here rather than
    left to a code reading — and it must reach *every* norm, since the accuracy
    controls in ``test_fused_vs_functional_equivalence`` are measured with it on.
    """
    ck = norm_compute_kernel_config(mesh_device.arch())
    assert ck.math_fidelity == ttnn.MathFidelity.HiFi4
    assert ck.math_approx_mode is False
    assert ck.fp32_dest_acc_en is True
    assert ck.packer_l1_acc is True


@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_every_norm_takes_the_uplifted_config(mesh_device, decoder_cache, kind):
    """All four hidden-size norms and the two QK norms carry it."""
    decoder = build_fused(mesh_device, decoder_cache, kind)
    expected = norm_compute_kernel_config(mesh_device.arch())
    norms = [
        decoder.input_layernorm,
        decoder.post_attention_layernorm,
        decoder.pre_feedforward_layernorm,
        decoder.post_feedforward_layernorm,
    ]
    for norm in norms:
        assert norm.compute_kernel_config is not None
        assert norm.compute_kernel_config.fp32_dest_acc_en == expected.fp32_dest_acc_en
        assert norm.compute_kernel_config.math_approx_mode == expected.math_approx_mode
    # the per-head QK norms go through the decoder's own handle
    assert decoder.norm_compute_kernel_config.fp32_dest_acc_en == expected.fp32_dest_acc_en
    assert decoder.norm_compute_kernel_config.math_approx_mode == expected.math_approx_mode


@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_minimal_matmul_block_config(mesh_device, decoder_cache, kind):
    """The tuned ``MinimalMatmulConfig`` blocking reaches the two shapes it won on.

    Same problem as the SDPA chunk: the blocking is a shipped constant no PCC
    test can see, and dropping it would only show up in a Tracy capture. This
    pins the selected blocks, the shapes that must *not* get one (``wqkv`` and
    ``mlp_down`` are faster on the op's default), the row gate that keeps
    ``o_proj`` on the default for tail chunks where the config loses ~6 %, and
    the op's own subblock rule, which the config has to keep to stay legal
    (``M_block % subblock_h == 0``, ``N_block % subblock_w == 0``).
    """
    decoder = build_fused(mesh_device, decoder_cache, kind)
    grid = mesh_device.compute_with_storage_grid_size()

    assert MINIMAL_MATMUL_BLOCKS == {
        (4096, 6656): (16, 4, 8, 8192),
        (6656, 19968): (8, 4, 16, MINIMAL_MATMUL_MIN_ROWS),
    }

    def config_for(weight, rows):
        return _minimal_matmul_config(rows, weight, grid)

    # o_proj: tuned at the full 8192-row chunk, default on every shorter chunk.
    wo = decoder.wo
    assert (wo.shape[-2], wo.shape[-1]) == (4096, 6656)
    cfg = config_for(wo, 8192)
    assert (cfg.M_block_size, cfg.K_block_size, cfg.N_block_size) == (16, 4, 8)
    assert (cfg.subblock_h, cfg.subblock_w) == (4, 2)  # N < M -> op's 4x2 rule
    assert config_for(wo, 4096) is None
    assert config_for(wo, 8160) is None  # a tile short of the full chunk

    # MLP gate/up: tuned at every length that reaches minimal_matmul at all.
    gate = decoder.mlp.gate
    assert (gate.shape[-2], gate.shape[-1]) == (6656, 19968)
    for rows in (MINIMAL_MATMUL_MIN_ROWS, 4096, 8192):
        cfg = config_for(gate, rows)
        assert (cfg.M_block_size, cfg.K_block_size, cfg.N_block_size) == (8, 4, 16)
        assert (cfg.subblock_h, cfg.subblock_w) == (2, 4)  # N >= M -> op's 2x4 rule

    # wqkv, attn gate and mlp_down keep the op's own choice at every length.
    for weight in (decoder.wqkv, decoder.w_attn_gate, decoder.mlp.down):
        for rows in (MINIMAL_MATMUL_MIN_ROWS, 8192):
            assert config_for(weight, rows) is None

    for cfg_blocks in MINIMAL_MATMUL_BLOCKS.values():
        m_block, _k_block, n_block, _min_rows = cfg_blocks
        assert m_block % 2 == 0 and n_block % 4 == 0  # legal under either subblock rule


# --------------------------------------------------------------------- traced decode


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_traced_decode_pcc(mesh_device, decoder_cache, reference_layers, kind):
    """Decode PCC measured from a *trace replay*, not an eager forward."""
    layer_idx, layer = reference_layers[kind]
    decoder = build_fused(mesh_device, decoder_cache, kind)
    prompt_len = 2048
    hidden = R.synthetic_hidden_states(1, prompt_len, seed=1357)
    _, cache = R.reference_prefill(layer, layer_idx, hidden)

    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ, seed=246)
    ttnn.deallocate(decoder.prefill_forward(to_device_hidden(mesh_device, hidden), page_table=page_table, user_id=0))

    positions = torch.tensor([prompt_len])
    token = R.synthetic_hidden_states(1, 1, seed=2468)
    tt_token = to_device_hidden(mesh_device, token)
    current_pos, rope_pos_ids = decode_position_tensors(mesh_device, positions)

    trace_id, tt_out = capture_decode_trace(decoder, mesh_device, tt_token, current_pos, page_table, rope_pos_ids)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    actual = ttnn.to_torch(tt_out).reshape(1, 1, -1)
    ttnn.release_trace(mesh_device, trace_id)

    expected = R.reference_decode(layer, layer_idx, token, past_key_values=cache, positions=positions)
    assert_pcc(f"fused traced decode[{kind}] pos={prompt_len}", expected, actual)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_traced_decode_advances_positions(mesh_device, decoder_cache, reference_layers, kind):
    """One captured trace replays across positions when only tensor contents change."""
    layer_idx, layer = reference_layers[kind]
    decoder = build_fused(mesh_device, decoder_cache, kind)
    prompt_len = 1024
    hidden = R.synthetic_hidden_states(1, prompt_len, seed=112233)
    _, cache = R.reference_prefill(layer, layer_idx, hidden)

    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ, seed=99)
    ttnn.deallocate(decoder.prefill_forward(to_device_hidden(mesh_device, hidden), page_table=page_table, user_id=0))

    tt_token = to_device_hidden(mesh_device, R.synthetic_hidden_states(1, 1, seed=1))
    current_pos, rope_pos_ids = decode_position_tensors(mesh_device, torch.tensor([prompt_len]))
    trace_id, tt_out = capture_decode_trace(decoder, mesh_device, tt_token, current_pos, page_table, rope_pos_ids)

    hidden_size = decoder.config.hidden_size
    for step in range(3):
        position = prompt_len + step
        token = R.synthetic_hidden_states(1, 1, seed=1000 + step)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(token.reshape(1, 1, 1, hidden_size), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
            tt_token,
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.tensor([position], dtype=torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32
            ),
            current_pos,
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.tensor([[position]], dtype=torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32
            ),
            rope_pos_ids,
        )
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        expected = R.reference_decode(
            layer, layer_idx, token, past_key_values=cache, positions=torch.tensor([position])
        )
        assert_pcc(
            f"fused traced decode replay[{kind}] pos={position}",
            expected,
            ttnn.to_torch(tt_out).reshape(1, 1, -1),
        )
    ttnn.release_trace(mesh_device, trace_id)


# --------------------------------------------------------------------------- perf


PERF_DECODE_ITERS = int(os.environ.get("MG_PERF_DECODE_ITERS", "8"))

#: Prefill windows.  8192 is one internal chunk — the regime the functional
#: stage profiled, so it is the like-for-like before/after.  16384 is two, which
#: is the only way to get the *paged* ``chunked_scaled_dot_product_attention``
#: into a profiled window at all: a ``full`` layer uses it for every chunk after
#: the first, and it is the dominant op of any long-context prefill.
PERF_PREFILL_SEQS = (8192, 16384)


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("seq_len", PERF_PREFILL_SEQS)
def test_perf_prefill(mesh_device, decoder_cache, kind, seq_len):
    """Warmed prefill, signposted for tt-perf-report."""
    decoder = build_fused(mesh_device, decoder_cache, kind)
    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ, seed=3)
    tt_hidden = to_device_hidden(mesh_device, R.synthetic_hidden_states(1, seq_len, seed=42))

    for _ in range(2):  # compile + warm
        ttnn.deallocate(decoder.prefill_forward(tt_hidden, page_table=page_table, user_id=0))
    ttnn.synchronize_device(mesh_device)

    signpost("PERF_PREFILL")
    out = decoder.prefill_forward(tt_hidden, page_table=page_table, user_id=0)
    ttnn.synchronize_device(mesh_device)
    signpost("PERF_PREFILL_END")
    logger.info(f"fused prefill perf window done: kind={kind} seq_len={seq_len} shape={tuple(out.shape)}")
    ttnn.deallocate(out)


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("context", (2048, HF_ADVERTISED_CONTEXT - 1))
def test_perf_decode_traced(mesh_device, decoder_cache, kind, context):
    """Warmed traced decode, signposted for tt-perf-report.

    Mirrors the functional stage's window exactly (same prompt, same replay
    count, same signposts) so the before/after numbers compare like with like.
    """
    max_seq_len = SHORT_MAX_SEQ if context < SHORT_MAX_SEQ else HF_ADVERTISED_CONTEXT
    decoder = build_fused(mesh_device, decoder_cache, kind, max_seq_len=max_seq_len)
    page_table = make_page_table(mesh_device, 1, max_seq_len, seed=4)
    warm_prompt = min(2048, context)
    ttnn.deallocate(
        decoder.prefill_forward(
            to_device_hidden(mesh_device, R.synthetic_hidden_states(1, warm_prompt, seed=43)),
            page_table=page_table,
            user_id=0,
        )
    )
    tt_token = to_device_hidden(mesh_device, R.synthetic_hidden_states(1, 1, seed=44))
    current_pos, rope_pos_ids = decode_position_tensors(mesh_device, torch.tensor([context]))
    trace_id, tt_out = capture_decode_trace(decoder, mesh_device, tt_token, current_pos, page_table, rope_pos_ids)

    for _ in range(4):  # warm the replay path
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)

    signpost("PERF_DECODE")
    for _ in range(PERF_DECODE_ITERS):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    signpost("PERF_DECODE_END")
    logger.info(
        f"fused decode perf window done: kind={kind} context={context} iters={PERF_DECODE_ITERS} "
        f"shape={tuple(tt_out.shape)}"
    )
    ttnn.release_trace(mesh_device, trace_id)
