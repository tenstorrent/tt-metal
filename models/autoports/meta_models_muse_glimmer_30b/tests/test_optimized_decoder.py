# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness and performance tests for the Muse-Glimmer-30B **optimized** decoder.

``OptimizedDecoder`` has the same public contract as ``FusedDecoder``, so this
module re-runs the whole inherited correctness surface (both layer kinds,
non-aligned prefill lengths, caller-chunked continuation prefill, batched paged
prefill+decode, the full 131072 context, an FP32 HF control, determinism,
host-fallback trapping, traced decode, a soak) against the optimized path, and
adds what only this stage can assert:

* ``test_decode_uses_dram_sharded_matmuls`` -- the decode projections really are
  ``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`` dispatches on
  width-sharded L1 activations, with the swept ``in0_block_w``.  Every perf claim
  in this stage rests on that, and a silent fall back to the fused decoder's
  ``ttnn.linear`` would still pass every PCC test here;
* ``test_weight_dtype_policy_reaches_the_tensors`` -- the shipped precision
  policy is what the *tensors* are, not just what the policy object says
  ($optimize OPT-013);
* ``test_real_weights_*`` -- widened well past the fused stage's single case,
  because the BFP4 MLP policy is selected on real-weight evidence and OPT-012
  requires that evidence to cover the disputed conditions (non-aligned lengths,
  the paged prefill/decode transition, traced replay, batch > 1, multi-step
  decode off a BFP8 cache).

Two PCC bars
------------

The functional acceptance bar is ``PCC_THRESHOLD = 0.995`` and it is what the
**real-checkpoint** tests hold the optimized layer to.  Those tests are wide on
purpose -- six prefill lengths, an eight-step decode, traced replay, batch 8, both
layer kinds -- because the shipped BFP4 MLP policy is *selected* on real-weight
evidence.

The synthetic harness gets its own, looser bar, and the honest reason is that the
gap is measured but **not explained**.  ``reference.synthetic_state_dict`` draws
i.i.d. Gaussian samples with each real tensor's mean and std, and under BFP4 MLP
weights the layer lands at ~0.9925-0.9939 against the HF reference where the real
checkpoint lands at ~0.9970-0.9979.  ``doc/optimized_decoder/logs/bfp_block_range_probe.log``
rules out the three obvious mechanisms -- per-block dynamic range is the same to
within 4 %, the real weights' on-device BFP4 round-trip error is *larger* not
smaller, and single-projection output PCC under BFP4 is the same for both -- so
this bar records an interaction inside the layer that the stage did not isolate.

``$optimize`` OPT-012 is what makes that acceptable: a synthetic-distribution PCC
does not veto a policy that passes real-weight PCC under the disputed conditions,
and the response is a looser *documented* synthetic bar plus wider real-weight
coverage.  It is deliberately not an expected-failure marker on the synthetic
case, and the slower higher-precision fallbacks (gate/up-only BFP4 at +5.3 %
decode, all-BFP8 at +16 %) are reported in the README with their numbers rather
than shipped.
"""

from __future__ import annotations

import math
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
    PCC_THRESHOLD,
    PREFILL_CHUNK_SIZE,
    PREFILL_SEQ_LENS,
    SHORT_MAX_SEQ,
    _FallbackGuard,
    _reference_prefill_cache_only,
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
from models.autoports.meta_models_muse_glimmer_30b.tests.test_fused_decoder import build_fused
from models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder import LAYER_KIND_SLIDING, TILE_SIZE
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import (
    BOUNDARY_CORES,
    DECODE_MATMUL,
    DEFAULT_PRECISION,
    MCAST_MAX_PER_CORE_M,
    PREFILL_MCAST2D,
    PREFILL_MINIMAL_BLOCKS,
    PREFILL_NORM_SHARD_CORES,
    PREFILL_NORM_SHARD_MAX_ROWS,
    OptimizedDecoder,
    core_rectangle,
    minimal_matmul_blocks,
    prefill_mcast2d_program_config,
    prefill_mcast2d_spec,
)
from models.common.utility_functions import comp_pcc

#: Bar for the i.i.d.-Gaussian synthetic-weight harness under the shipped BFP4
#: MLP policy.  See the module docstring for why it differs from
#: ``PCC_THRESHOLD``; the measured synthetic floor across this whole file is
#: recorded in ``doc/optimized_decoder/logs/pcc_summary.txt``.
SYNTHETIC_PCC_THRESHOLD = 0.99

#: How much *worse* than the fused decoder the optimized decoder may be against
#: the same HF reference.  This is a precision stage, so unlike the fusing stage
#: a regression is expected and bounded rather than forbidden.
#:
#: These are *diagnostic bounds*, not the acceptance gate.  The gate is the
#: absolute real-weight bar (``PCC_THRESHOLD`` = 0.995, held by the whole
#: ``test_real_weights_*`` surface); this test exists to keep the head-to-head
#: delta from drifting silently.  Both bounds are the measured worst delta at
#: ``seq_len = 4097`` plus ~35 % margin, so a real drift trips them while
#: re-measuring the same code does not:
#:
#: ===========  ==============  ==============  =========
#: population   worst prefill   worst decode    bound
#: ===========  ==============  ==============  =========
#: real         2.544e-3        2.996e-3        4.0e-3
#: synthetic    7.316e-3        6.435e-3        1.0e-2
#: ===========  ==============  ==============  =========
#:
#: The real and synthetic bounds differ by 2.5x, and that gap *is* the OPT-012
#: argument in one constant: BFP4 MLP weights cost the i.i.d.-Gaussian synthetic
#: harness several times more accuracy than they cost the real checkpoint.  The
#: numbers above are regenerated by ``logs/full_test_run.log`` ("optimized vs
#: fused" lines) and quoted in ``README.md``.
FUSED_REGRESSION_TOL = {"synthetic": 1.0e-2, "real": 4.0e-3}


def assert_pcc(label: str, expected: torch.Tensor, actual: torch.Tensor, threshold: float = SYNTHETIC_PCC_THRESHOLD):
    passed, message = comp_pcc(expected.float(), actual.float(), threshold)
    logger.info(f"{label}: {message}")
    assert passed, f"{label} below PCC {threshold}: {message}"
    return message


def build_optimized(
    mesh_device,
    decoder_cache,
    kind: str,
    *,
    max_seq_len: int = SHORT_MAX_SEQ,
    max_batch_size: int = 1,
    real_weights: bool = False,
    chunk: int = PREFILL_CHUNK_SIZE,
    **build_kwargs,
) -> OptimizedDecoder:
    layer_idx = layer_idx_for(kind)
    # Values, not just names: keying on ``sorted(build_kwargs)`` alone would let two
    # builds that differ only in a kwarg *value* share a cached decoder.
    key = (
        "optimized",
        layer_idx,
        max_seq_len,
        max_batch_size,
        real_weights,
        chunk,
        tuple(sorted((name, repr(value)) for name, value in build_kwargs.items())),
    )

    def factory():
        state_dict = R.real_state_dict(layer_idx) if real_weights else R.synthetic_state_dict(layer_idx)
        return OptimizedDecoder.from_state_dict(
            state_dict,
            hf_config=R.hf_config(),
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            page_block_size=PAGE_BLOCK_SIZE,
            prefill_chunk_size=chunk,
            **build_kwargs,
        )

    return decoder_cache.get(key, factory)


def require_real_weights(layer_idx: int):
    try:
        return R.real_state_dict(layer_idx)
    except FileNotFoundError as error:  # pragma: no cover - weights not cached
        pytest.skip(f"released checkpoint not cached: {error}")


# ------------------------------------------------------------------- prefill / decode


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("seq_len", PREFILL_SEQ_LENS)
def test_prefill_pcc(mesh_device, decoder_cache, reference_layers, kind, seq_len):
    """The nine inherited lengths, including the non-aligned ones.

    ``seq_len = 1`` matters more here than in the earlier stages: it pads to a
    single 32-row tile, which is the only prefill row count that takes the
    DRAM-sharded decode matmul rather than ``minimal_matmul``
    (``matmul_device_operation.cpp:1287`` allows one M tile only), so it is a
    distinct code path and not just a smoke case.
    """
    layer_idx, layer = reference_layers[kind]
    decoder = build_optimized(mesh_device, decoder_cache, kind)
    hidden = R.synthetic_hidden_states(1, seq_len, seed=101 + seq_len)
    expected, _ = R.reference_prefill(layer, layer_idx, hidden)

    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ)
    tt_out = decoder.prefill_forward(to_device_hidden(mesh_device, hidden), page_table=page_table, user_id=0)
    actual = ttnn.to_torch(tt_out).reshape(1, seq_len, -1)
    assert tuple(tt_out.shape) == (1, 1, seq_len, decoder.config.hidden_size)
    assert_pcc(f"optimized prefill[{kind}] seq_len={seq_len}", expected, actual)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("real", (False, True), ids=("synthetic", "real"))
@pytest.mark.parametrize("seq_len", (224, 256, 288, 320))
def test_prefill_pcc_across_the_norm_shard_band(mesh_device, decoder_cache, reference_layers, kind, real, seq_len):
    """PCC either side of ``PREFILL_NORM_SHARD_MAX_ROWS``, on both weight populations.

    256 rows is the *last* row count where the 16-core prefill norm shard fits L1 --
    at 512 it overflows the circular-buffer budget by 33 % -- so it is the worst
    case inside the band, and 288/320 are the first row counts outside it. The
    inherited length list jumps 128 -> 2048, so without this the band edge and the
    fallback boundary would both go unexercised while being the newest branch in the
    prefill path. 224 and 288 are also deliberately not powers of two: the branch
    keys on a row count, so a tile-aligned non-power-of-two length is the realistic
    case a serving prompt produces.
    """
    layer_idx = layer_idx_for(kind)
    if real:
        state_dict = require_real_weights(layer_idx)
        layer = R.reference_layer(layer_idx, state_dict)
    else:
        layer_idx, layer = reference_layers[kind]
    decoder = build_optimized(mesh_device, decoder_cache, kind, real_weights=real)
    hidden = R.synthetic_hidden_states(1, seq_len, seed=8100 + seq_len)
    expected, _ = R.reference_prefill(layer, layer_idx, hidden)

    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ, seed=61)

    # PCC alone cannot tell the two norm paths apart -- 256 and 288 would both pass
    # if the branch silently fell back to interleaved everywhere -- so record which
    # path each row count actually took.
    sharded_norms = []
    original = ttnn.rms_norm

    def traced(x, **kwargs):
        spec = x.memory_config().shard_spec
        if spec is not None and spec.grid.num_cores() == PREFILL_NORM_SHARD_CORES:
            sharded_norms.append(seq_len)
        return original(x, **kwargs)

    ttnn.rms_norm = traced
    try:
        tt_out = decoder.prefill_forward(to_device_hidden(mesh_device, hidden), page_table=page_table, user_id=0)
    finally:
        ttnn.rms_norm = original

    expect_sharded = seq_len <= PREFILL_NORM_SHARD_MAX_ROWS
    assert bool(sharded_norms) == expect_sharded, (
        f"{seq_len} rows is {'inside' if expect_sharded else 'outside'} the "
        f"<= {PREFILL_NORM_SHARD_MAX_ROWS} sharded-norm band but took the "
        f"{'interleaved' if expect_sharded else 'sharded'} path"
    )
    if expect_sharded:
        assert len(sharded_norms) == 4, f"expected the 4 hidden-size norms sharded, saw {len(sharded_norms)}"

    assert tuple(tt_out.shape) == (1, 1, seq_len, decoder.config.hidden_size)
    assert_pcc(
        f"optimized {'real-weight ' if real else ''}prefill[{kind}] norm-band seq_len={seq_len}",
        expected,
        ttnn.to_torch(tt_out).reshape(1, seq_len, -1),
        threshold=PCC_THRESHOLD if real else SYNTHETIC_PCC_THRESHOLD,
    )
    ttnn.deallocate(tt_out)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("prompt_len", (100, 2048, 3000))
def test_decode_pcc(mesh_device, decoder_cache, reference_layers, kind, prompt_len):
    """Prefill, then decode four tokens past the prompt off the BFP8 paged cache."""
    layer_idx, layer = reference_layers[kind]
    decoder = build_optimized(mesh_device, decoder_cache, kind)
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
            f"optimized decode[{kind}] prompt={prompt_len} pos={position}",
            expected,
            ttnn.to_torch(tt_out).reshape(1, 1, -1),
        )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_prefill_decode_pcc_vs_fp32_reference(mesh_device, decoder_cache, kind):
    """FP32 HF control: rules out an error common-mode to two BF16 graphs."""
    layer_idx = layer_idx_for(kind)
    layer = R.reference_layer(layer_idx, R.synthetic_state_dict(layer_idx), dtype=torch.float32)
    decoder = build_optimized(mesh_device, decoder_cache, kind)

    seq_len = 2049
    hidden_bf16 = R.synthetic_hidden_states(1, seq_len, seed=606060)
    expected, cache = R.reference_prefill(layer, layer_idx, hidden_bf16.float())

    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ, seed=6060)
    tt_out = decoder.prefill_forward(to_device_hidden(mesh_device, hidden_bf16), page_table=page_table, user_id=0)
    assert_pcc(
        f"optimized prefill[{kind}] vs FP32 HF reference seq_len={seq_len}",
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
        f"optimized decode[{kind}] vs FP32 HF reference pos={seq_len}",
        expected,
        ttnn.to_torch(tt_out).reshape(1, 1, -1),
    )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_prefill_seq_len_equals_max_and_chunk(mesh_device, decoder_cache, reference_layers, kind):
    """``max_seq_len == prefill_chunk_size == seq_len``, twice on one decoder.

    Inherited regression: prefill hands its *persistent* pre-tilized RoPE tables
    straight to ``rotary_embedding_hf`` at ``start_pos == 0``, so a stray
    deallocate would destroy the layer on the second call.
    """
    layer_idx, layer = reference_layers[kind]
    seq_len = 4096
    decoder = build_optimized(mesh_device, decoder_cache, kind, max_seq_len=seq_len, chunk=seq_len)
    page_table = make_page_table(mesh_device, 1, seq_len, seed=771)
    for attempt in range(2):
        hidden = R.synthetic_hidden_states(1, seq_len, seed=880 + attempt)
        expected, _ = R.reference_prefill(layer, layer_idx, hidden)
        tt_out = decoder.prefill_forward(to_device_hidden(mesh_device, hidden), page_table=page_table, user_id=0)
        assert_pcc(
            f"optimized prefill[{kind}] seq_len==max_seq_len==chunk=={seq_len} attempt={attempt}",
            expected,
            ttnn.to_torch(tt_out).reshape(1, seq_len, -1),
        )
        ttnn.deallocate(tt_out)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_multi_chunk_prefill_page_table_bound(mesh_device, decoder_cache, reference_layers, kind):
    """Multi-chunk prefill at a ``max_seq_len`` whose page count is awkward (12416)."""
    layer_idx, layer = reference_layers[kind]
    max_seq_len = 12416
    decoder = build_optimized(mesh_device, decoder_cache, kind, max_seq_len=max_seq_len)
    page_table = make_page_table(mesh_device, 1, max_seq_len, seed=1216)
    hidden = R.synthetic_hidden_states(1, max_seq_len, seed=12416)
    expected, _ = R.reference_prefill(layer, layer_idx, hidden)

    tt_out = decoder.prefill_forward(to_device_hidden(mesh_device, hidden), page_table=page_table, user_id=0)
    assert tuple(tt_out.shape) == (1, 1, max_seq_len, decoder.config.hidden_size)
    assert_pcc(
        f"optimized prefill[{kind}] max_seq_len==seq_len==12416 (194 pages, 2 chunks)",
        expected,
        ttnn.to_torch(tt_out).reshape(1, max_seq_len, -1),
    )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("first_len,second_len", CONTINUATION_SPLITS)
def test_continuation_prefill_pcc(mesh_device, decoder_cache, reference_layers, kind, first_len, second_len):
    """Caller-chunked prefill: two ``start_pos``-separated calls == one call."""
    layer_idx, layer = reference_layers[kind]
    decoder = build_optimized(mesh_device, decoder_cache, kind)
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
    assert_pcc(f"optimized continuation prefill[{kind}] {first_len}+{second_len}", expected, actual)

    # ...and a decode past the join, so the cache the two calls wrote is read.
    _, cache = R.reference_prefill(layer, layer_idx, hidden)
    token = R.synthetic_hidden_states(1, 1, seed=555)
    ref = R.reference_decode(layer, layer_idx, token, past_key_values=cache, positions=torch.tensor([total]))
    current_pos, rope_pos_ids = decode_position_tensors(mesh_device, torch.tensor([total]))
    tt_out = decoder.decode_forward(
        to_device_hidden(mesh_device, token),
        current_pos=current_pos,
        page_table=page_table,
        rope_pos_ids=rope_pos_ids,
    )
    assert_pcc(
        f"optimized decode after continuation[{kind}] {first_len}+{second_len} pos={total}",
        ref,
        ttnn.to_torch(tt_out).reshape(1, 1, -1),
    )


@pytest.mark.timeout(600)
def test_continuation_prefill_requires_sliding_tail(mesh_device, decoder_cache, expect_error):
    """A sliding continuation without its K/V window must raise, not guess."""
    decoder = build_optimized(mesh_device, decoder_cache, LAYER_KIND_SLIDING)
    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ, seed=13)
    hidden = to_device_hidden(mesh_device, R.synthetic_hidden_states(1, 64, seed=14))
    with expect_error(ValueError, "sliding_kv_tail"):
        decoder.prefill_forward(hidden, page_table=page_table, user_id=0, start_pos=1024)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("batch", (4, 13, 32))
def test_batched_prefill_decode_pcc(mesh_device, decoder_cache, reference_layers, kind, batch):
    """Independent users share one paged cache: per-user prefill then batched decode.

    ``batch=13`` is prime and larger than the 11-wide grid, so it has no
    ``batch``-core rectangle and takes the decode head-concat's shape-agnostic
    fallback.  Batch is also the field OPT-005 warns about: every batch from 1 to
    32 tile-pads to the same 32 activation rows, and the DRAM-sharded matmul's
    ``per_core_M`` is derived from those *padded* rows, so this pins that the
    padding never becomes extra active users.
    """
    layer_idx, layer = reference_layers[kind]
    max_seq_len = 4096
    decoder = build_optimized(mesh_device, decoder_cache, kind, max_seq_len=max_seq_len, max_batch_size=batch)
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
            f"optimized prefill[{kind}] batch={batch} user={user} seq_len={prompt_len}",
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
        f"optimized decode[{kind}] batch={batch} ragged positions",
        expected,
        ttnn.to_torch(tt_out).reshape(batch, 1, -1),
    )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_multi_chunk_prefill_nonzero_user(mesh_device, decoder_cache, reference_layers, kind):
    """Multi-chunk prefill into a non-zero cache slot, then decode that slot."""
    layer_idx, layer = reference_layers[kind]
    seq_len = 12345
    decoder = build_optimized(mesh_device, decoder_cache, kind, max_seq_len=SHORT_MAX_SEQ, max_batch_size=4)
    page_table = make_page_table(mesh_device, 4, SHORT_MAX_SEQ, seed=555)
    hidden = R.synthetic_hidden_states(1, seq_len, seed=13579)
    expected, cache = R.reference_prefill(layer, layer_idx, hidden)

    tt_out = decoder.prefill_forward(to_device_hidden(mesh_device, hidden), page_table=page_table, user_id=2)
    assert_pcc(
        f"optimized multi-chunk prefill[{kind}] user_id=2 seq_len={seq_len}",
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
        f"optimized decode[{kind}] user_id=2 pos={seq_len}",
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
    decoder = build_optimized(mesh_device, decoder_cache, kind, max_seq_len=HF_ADVERTISED_CONTEXT)
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
        assert_pcc(f"optimized prefill[{kind}] full-context seq_len={seq_len} ({where} {tail} rows)", expected, actual)
    ttnn.deallocate(tt_out)


@pytest.mark.slow
@pytest.mark.timeout(5400)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_full_context_decode_pcc(mesh_device, decoder_cache, reference_layers, kind):
    """Decode at the last valid position of the advertised context (131071).

    This is also the case the BFP8 KV cache has to survive at full depth: the
    ``full`` (NoPE) layer's decode SDPA reads all 131071 cached positions, and
    its cost is what the reduced cache dtype halves.
    """
    from transformers.cache_utils import DynamicCache

    layer_idx, layer = reference_layers[kind]
    decoder = build_optimized(mesh_device, decoder_cache, kind, max_seq_len=HF_ADVERTISED_CONTEXT)
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
    assert_pcc(
        f"optimized decode[{kind}] full-context pos={position}", expected, ttnn.to_torch(tt_out).reshape(1, 1, -1)
    )


# ----------------------------------------------------------------------- real weights

#: Real-checkpoint prefill lengths.  Wider than the fused stage's single case
#: because the shipped BFP4 MLP policy is *selected* on real-weight evidence, so
#: OPT-012 requires that evidence to cover the conditions the synthetic result
#: disputes: non-aligned lengths, sub-tile, multi-chunk, and the tile-only
#: DRAM-sharded prefill branch.
REAL_PREFILL_SEQ_LENS = (1, 100, 2049, 4097, 8193, 12345)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("seq_len", REAL_PREFILL_SEQ_LENS)
def test_real_weights_prefill_pcc(mesh_device, decoder_cache, kind, seq_len):
    """Released-checkpoint prefill at the functional bar, ``PCC_THRESHOLD``."""
    layer_idx = layer_idx_for(kind)
    state_dict = require_real_weights(layer_idx)
    layer = R.reference_layer(layer_idx, state_dict)
    decoder = build_optimized(mesh_device, decoder_cache, kind, real_weights=True)

    hidden = R.synthetic_hidden_states(1, seq_len, seed=31337 + seq_len)
    expected, _ = R.reference_prefill(layer, layer_idx, hidden)
    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ, seed=808)
    tt_out = decoder.prefill_forward(to_device_hidden(mesh_device, hidden), page_table=page_table, user_id=0)
    assert_pcc(
        f"optimized real-weight prefill[{kind}] seq_len={seq_len}",
        expected,
        ttnn.to_torch(tt_out).reshape(1, seq_len, -1),
        threshold=PCC_THRESHOLD,
    )


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_real_weights_decode_pcc(mesh_device, decoder_cache, kind):
    """Released-checkpoint decode: eight steps across the paged BFP8 cache.

    Multi-step rather than single-step on purpose: a reduced-precision cache
    fault compounds across steps, and a single decode off a freshly filled cache
    would not see it (OPT-002's "follow-on cache-use evidence").
    """
    layer_idx = layer_idx_for(kind)
    state_dict = require_real_weights(layer_idx)
    layer = R.reference_layer(layer_idx, state_dict)
    decoder = build_optimized(mesh_device, decoder_cache, kind, real_weights=True)

    prompt_len = 3000
    hidden = R.synthetic_hidden_states(1, prompt_len, seed=1717)
    _, cache = R.reference_prefill(layer, layer_idx, hidden)
    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ, seed=909)
    ttnn.deallocate(decoder.prefill_forward(to_device_hidden(mesh_device, hidden), page_table=page_table, user_id=0))

    for step in range(8):
        position = prompt_len + step
        token = R.synthetic_hidden_states(1, 1, seed=1800 + step)
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
            f"optimized real-weight decode[{kind}] step={step} pos={position}",
            expected,
            ttnn.to_torch(tt_out).reshape(1, 1, -1),
            threshold=PCC_THRESHOLD,
        )


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_real_weights_traced_decode_and_batch(mesh_device, decoder_cache, kind):
    """Real weights through the *traced* replay, and at batch 8.

    The remaining two conditions OPT-012 asks the real-weight evidence to cover:
    trace replay (which is the measured performance path) and batch > 1.
    """
    layer_idx = layer_idx_for(kind)
    state_dict = require_real_weights(layer_idx)
    layer = R.reference_layer(layer_idx, state_dict)
    batch = 8
    max_seq_len = 4096
    decoder = build_optimized(
        mesh_device, decoder_cache, kind, max_seq_len=max_seq_len, max_batch_size=batch, real_weights=True
    )
    page_table = make_page_table(mesh_device, batch, max_seq_len, seed=414)

    prompt_lens = [1000 + 61 * user for user in range(batch)]
    caches = []
    for user, prompt_len in enumerate(prompt_lens):
        hidden = R.synthetic_hidden_states(1, prompt_len, seed=5000 + user)
        _, cache = R.reference_prefill(layer, layer_idx, hidden)
        caches.append(cache)
        ttnn.deallocate(
            decoder.prefill_forward(to_device_hidden(mesh_device, hidden), page_table=page_table, user_id=user)
        )

    positions = torch.tensor(prompt_lens, dtype=torch.int32)
    tokens = R.synthetic_hidden_states(batch, 1, seed=6000)
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
    tt_token = to_device_hidden(mesh_device, tokens)
    current_pos, rope_pos_ids = decode_position_tensors(mesh_device, positions)
    trace_id, tt_out = capture_decode_trace(decoder, mesh_device, tt_token, current_pos, page_table, rope_pos_ids)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    assert_pcc(
        f"optimized real-weight traced decode[{kind}] batch={batch}",
        expected,
        ttnn.to_torch(tt_out).reshape(batch, 1, -1),
        threshold=PCC_THRESHOLD,
    )
    ttnn.release_trace(mesh_device, trace_id)


# ------------------------------------------------------- optimized vs fused accuracy


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("real", (False, True), ids=("synthetic", "real"))
def test_optimized_vs_fused_accuracy(mesh_device, decoder_cache, reference_layers, kind, real):
    """Bound how much accuracy the precision policy costs, against the same HF layer.

    The fusing stage forbade any prefill regression, because a topology rewrite
    should not lose accuracy.  This stage *buys* speed with precision, so the
    question is different: how much, and is it bounded?  Both graphs are compared
    to the same HF reference on the same weights and inputs, and the optimized
    one may be worse by at most ``FUSED_REGRESSION_TOL``.

    The synthetic and real tolerances differ by more than 2x, which is the whole
    OPT-012 argument in one assertion: the i.i.d.-Gaussian synthetic weights cost
    BFP4 several times more accuracy than the real checkpoint does.
    """
    layer_idx = layer_idx_for(kind)
    if real:
        state_dict = require_real_weights(layer_idx)
        layer = R.reference_layer(layer_idx, state_dict)
    else:
        layer_idx, layer = reference_layers[kind]

    seq_len = 4097
    hidden = R.synthetic_hidden_states(1, seq_len, seed=24680)
    expected, cache = R.reference_prefill(layer, layer_idx, hidden)
    token = R.synthetic_hidden_states(1, 1, seed=24681)
    expected_decode = R.reference_decode(
        layer, layer_idx, token, past_key_values=cache, positions=torch.tensor([seq_len])
    )
    tol = FUSED_REGRESSION_TOL["real" if real else "synthetic"]

    scores = {}
    for name in ("fused", "optimized"):
        build = build_fused if name == "fused" else build_optimized
        decoder = build(mesh_device, decoder_cache, kind, real_weights=real)
        page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ, seed=1122)
        tt_out = decoder.prefill_forward(to_device_hidden(mesh_device, hidden), page_table=page_table, user_id=0)
        _, prefill_pcc = comp_pcc(expected.float(), ttnn.to_torch(tt_out).reshape(1, seq_len, -1).float(), 0.0)
        ttnn.deallocate(tt_out)
        current_pos, rope_pos_ids = decode_position_tensors(mesh_device, torch.tensor([seq_len]))
        tt_dec = decoder.decode_forward(
            to_device_hidden(mesh_device, token),
            current_pos=current_pos,
            page_table=page_table,
            rope_pos_ids=rope_pos_ids,
        )
        _, decode_pcc = comp_pcc(expected_decode.float(), ttnn.to_torch(tt_dec).reshape(1, 1, -1).float(), 0.0)
        ttnn.deallocate(tt_dec)
        scores[name] = (float(prefill_pcc), float(decode_pcc))
        logger.info(f"{name}[{kind}] {'real' if real else 'synthetic'}: prefill/decode PCC {scores[name]}")

    for mode, index in (("prefill", 0), ("decode", 1)):
        fused_pcc, opt_pcc = scores["fused"][index], scores["optimized"][index]
        delta = fused_pcc - opt_pcc
        logger.info(
            f"optimized vs fused {mode}[{kind}] {'real' if real else 'synthetic'}: "
            f"{fused_pcc:.6f} -> {opt_pcc:.6f} (delta {delta:+.6f}, tol {tol})"
        )
        assert delta <= tol, (
            f"optimized {mode}[{kind}] on {'real' if real else 'synthetic'} weights lost {delta:.6f} PCC against "
            f"the fused decoder ({fused_pcc:.6f} -> {opt_pcc:.6f}), more than the documented {tol}"
        )


# ------------------------------------------------------------------- the optimized graph


class _MatmulSpy:
    """Record the program config and memory configs of every ``ttnn.linear`` call."""

    def __init__(self):
        self.calls: list[dict] = []
        self._saved = None

    def __enter__(self):
        self._saved = ttnn.linear

        def traced(a, b, **kwargs):
            self.calls.append(
                {
                    "program_config": kwargs.get("program_config"),
                    "in_memory_layout": a.memory_config().memory_layout,
                    "in_cores": (
                        a.memory_config().shard_spec.grid.num_cores() if a.memory_config().shard_spec else None
                    ),
                    "weight_layout": b.memory_config().memory_layout,
                    "weight_dtype": b.dtype,
                    "n": int(b.shape[-1]),
                    "k": int(b.shape[-2]),
                }
            )
            return self._saved(a, b, **kwargs)

        ttnn.linear = traced
        return self

    def __exit__(self, *exc):
        ttnn.linear = self._saved
        return False


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_decode_uses_dram_sharded_matmuls(mesh_device, decoder_cache, kind):
    """Every decode projection is a DRAM-sharded matmul with the swept geometry.

    This is the assertion the whole stage's performance claim rests on.  A silent
    fall back to the fused decoder's auto-configured ``ttnn.linear`` would keep
    every PCC test in this file green while giving up 383 GB/s against 490, so
    the program config, the width-sharded activation, the width-sharded DRAM
    weight and the tuned ``in0_block_w`` are all checked directly rather than
    inferred from a latency number.
    """
    decoder = build_optimized(mesh_device, decoder_cache, kind)
    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ, seed=71)
    ttnn.deallocate(
        decoder.prefill_forward(
            to_device_hidden(mesh_device, R.synthetic_hidden_states(1, 256, seed=72)),
            page_table=page_table,
            user_id=0,
        )
    )
    token = to_device_hidden(mesh_device, R.synthetic_hidden_states(1, 1, seed=73))
    current_pos, rope_pos_ids = decode_position_tensors(mesh_device, torch.tensor([256]))

    with _MatmulSpy() as spy:
        ttnn.deallocate(
            decoder.decode_forward(token, current_pos=current_pos, page_table=page_table, rope_pos_ids=rope_pos_ids)
        )

    assert len(spy.calls) == 6, f"expected 6 decode projections, saw {len(spy.calls)}"
    for call in spy.calls:
        assert isinstance(
            call["program_config"], ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig
        ), f"decode projection K={call['k']} N={call['n']} is not a DRAM-sharded matmul: {call['program_config']}"
        assert call["in_memory_layout"] == ttnn.TensorMemoryLayout.WIDTH_SHARDED
        assert call["weight_layout"] == ttnn.TensorMemoryLayout.WIDTH_SHARDED

    # Each role's (cores, in0_block_w) must be the swept winner for its dtype.
    by_shape = {(c["k"], c["n"]): c for c in spy.calls}
    expected_roles = {
        "wqkv": (decoder.config.hidden_size, 4608),
        "attn_gate": (decoder.config.hidden_size, decoder.config.num_attention_heads * decoder.config.head_dim),
        "o_proj": (decoder.config.num_attention_heads * decoder.config.head_dim, decoder.config.hidden_size),
        "mlp_gate": (decoder.config.hidden_size, decoder.config.intermediate_size),
        "mlp_down": (decoder.config.intermediate_size, decoder.config.hidden_size),
    }
    for role, shape in expected_roles.items():
        call = by_shape[shape]
        cores, in0_block_w = decoder.decode_matmul[role]
        assert call["in_cores"] == cores, f"{role}: activation on {call['in_cores']} cores, expected {cores}"
        assert (
            call["program_config"].in0_block_w == in0_block_w
        ), f"{role}: in0_block_w={call['program_config'].in0_block_w}, expected the swept {in0_block_w}"
        assert call["program_config"].per_core_M == 1, "a decode step must be exactly one M tile"
        assert call["program_config"].per_core_N == math.ceil(shape[1] / (TILE_SIZE * cores))
        assert call["weight_dtype"] == DEFAULT_PRECISION.weight_dtype(role), (
            f"{role}: weight dtype {call['weight_dtype']} is not the policy's "
            f"{DEFAULT_PRECISION.weight_dtype(role)} -- the policy did not reach the measured op (OPT-013)"
        )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_weight_dtype_policy_reaches_the_tensors(mesh_device, decoder_cache, kind):
    """The shipped policy is what the weights *are* (OPT-013), and they are shared.

    Also asserts the single-copy property the whole design depends on: prefill and
    decode use the same tensor object, so the layer holds 314.8 MB of weights
    rather than two copies.
    """
    decoder = build_optimized(mesh_device, decoder_cache, kind)
    roles = {
        "wqkv": decoder.wqkv,
        "attn_gate": decoder.w_attn_gate,
        "o_proj": decoder.wo,
        "mlp_gate": decoder.mlp.gate,
        "mlp_up": decoder.mlp.up,
        "mlp_down": decoder.mlp.down,
    }
    total_bytes = 0
    for role, tensor in roles.items():
        assert tensor.dtype == DEFAULT_PRECISION.weight_dtype(role), f"{role} is {tensor.dtype}"
        assert (
            tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
        ), f"{role} is not DRAM width-sharded, so the decode matmul cannot take it"
        assert tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
        bits = {ttnn.bfloat16: 16.0, ttnn.bfloat8_b: 8.5, ttnn.bfloat4_b: 4.5}[tensor.dtype]
        total_bytes += int(tensor.shape[-2]) * int(tensor.shape[-1]) * bits / 8
    # The fused stage's BF16 layer was 967,835,648 B.
    assert total_bytes < 400e6, f"weight footprint {total_bytes / 1e6:.1f} MB is larger than the policy implies"
    logger.info(f"optimized[{kind}] weight footprint {total_bytes / 1e6:.1f} MB (fused stage: 967.8 MB)")

    # One tensor, two phases: the prefill and decode paths must not have been
    # given separate copies.
    assert decoder.mlp.gate is roles["mlp_gate"]
    seen = {id(t) for t in roles.values()}
    assert len(seen) == 6, "two roles share a weight tensor"


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("seq_len", (32, 128, 2048, 12345))
def test_prefill_uses_the_expected_dense_kernel(mesh_device, decoder_cache, kind, seq_len):
    """Prefill dispatches the right one of *three* kernels for its row count.

    All three read the same DRAM width-sharded weight, and which one is fastest is
    a measured crossover, not a preference:

    * ``seq_len == 32`` (one M tile) -> the DRAM-sharded matmul.  This side of the
      rule is an op contract, not tuning (``matmul_device_operation.cpp:1287``
      allows ``M == 1`` only), so both sides are asserted.
    * ``128`` rows -> ``ttnn.linear`` with a **2D-multicast** program config whose
      grid is exactly ``dram_banks`` columns wide, 1.3-2.0x faster than
      ``minimal_matmul`` through ~1024 rows.
    * ``2048`` and ``12345`` -> ``minimal_matmul`` with the swept per-shape
      blocking, and **no** ``ttnn.linear`` at all.
    """
    decoder = build_optimized(mesh_device, decoder_cache, kind)
    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ, seed=81)
    hidden = to_device_hidden(mesh_device, R.synthetic_hidden_states(1, seq_len, seed=82))

    #: ``(K, N) -> role``, so a traced dispatch can be checked against the table.
    role_of_shape = {
        (decoder.config.hidden_size, 4608): "wqkv",
        (decoder.config.hidden_size, decoder.config.num_attention_heads * decoder.config.head_dim): "attn_gate",
        (decoder.config.num_attention_heads * decoder.config.head_dim, decoder.config.hidden_size): "o_proj",
        (decoder.config.intermediate_size, decoder.config.hidden_size): "mlp_down",
    }

    minimal_calls = []
    original = ttnn.experimental.minimal_matmul

    def traced_minimal(a, b, **kwargs):
        minimal_calls.append(
            {"rows": int(a.shape[-2]), "shape": (int(b.shape[-2]), int(b.shape[-1])), "config": kwargs.get("config")}
        )
        return original(a, b, **kwargs)

    ttnn.experimental.minimal_matmul = traced_minimal
    try:
        with _MatmulSpy() as spy:
            ttnn.deallocate(decoder.prefill_forward(hidden, page_table=page_table, user_id=0))
    finally:
        ttnn.experimental.minimal_matmul = original

    if seq_len == TILE_SIZE:
        assert not minimal_calls, "a one-tile prefill must take the DRAM-sharded matmul"
        assert len(spy.calls) == 6
        for call in spy.calls:
            assert isinstance(call["program_config"], ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig)
        return

    if seq_len == 128:
        banks = mesh_device.dram_grid_size().x
        assert not minimal_calls, f"128 rows should take the 2D-multicast matmul, saw {len(minimal_calls)} minimal"
        assert len(spy.calls) == 6, f"expected 6 2D-multicast projections, saw {len(spy.calls)}"
        for call in spy.calls:
            program_config = call["program_config"]
            assert isinstance(
                program_config, ttnn.MatmulMultiCoreReuseMultiCastProgramConfig
            ), f"K={call['k']} N={call['n']} at 128 rows is {type(program_config).__name__}"
            # The silent-inf guard, on the real dispatch rather than on the table.
            program_grid = program_config.compute_with_storage_grid_size
            assert program_grid.x == banks, "a 2D-multicast grid wider than the DRAM bank count silently returns inf"
            assert program_config.per_core_N * banks == call["n"] // TILE_SIZE
            role = role_of_shape.get((call["k"], call["n"]))
            if role is not None:
                grid_y, in0_block_w = prefill_mcast2d_spec(role, 128, DEFAULT_PRECISION.weight_dtype(role))
                assert (
                    program_config.in0_block_w == in0_block_w
                ), f"{role}@128 rows is not the swept in0_block_w {in0_block_w}"
                assert program_grid.y == grid_y
        return

    assert not spy.calls, f"ttnn.linear was dispatched at {seq_len} rows with a width-sharded weight"
    assert len(minimal_calls) >= 6
    for call in minimal_calls:
        # gate and up share a shape, so only the four unambiguous roles are
        # checked against the table; all six are checked for a legal config.
        assert call["config"] is None or isinstance(call["config"], ttnn.MinimalMatmulConfig)
        role = role_of_shape.get(call["shape"])
        if role is None:
            continue
        expected = minimal_matmul_blocks(role, call["rows"], DEFAULT_PRECISION.weight_dtype(role))
        if expected is None:
            assert call["config"] is None, f"{role}@{call['rows']} rows should use the op's own blocking"
        else:
            m_block, k_block, n_block = expected
            assert (
                call["config"].M_block_size,
                call["config"].K_block_size,
                call["config"].N_block_size,
            ) == (m_block, k_block, n_block), f"{role}@{call['rows']} rows is not the swept blocking {expected}"


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("seq_len", (3000, 12345))
def test_no_host_fallback_in_forward(mesh_device, decoder_cache, kind, seq_len):
    """No torch / host round-trip in a measured prefill or decode."""
    decoder = build_optimized(mesh_device, decoder_cache, kind)
    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ, seed=91)
    tt_hidden = to_device_hidden(mesh_device, R.synthetic_hidden_states(1, seq_len, seed=92))
    token = to_device_hidden(mesh_device, R.synthetic_hidden_states(1, 1, seed=93))
    current_pos, rope_pos_ids = decode_position_tensors(mesh_device, torch.tensor([seq_len]))

    with _FallbackGuard() as guard:
        ttnn.deallocate(decoder.prefill_forward(tt_hidden, page_table=page_table, user_id=0))
        ttnn.deallocate(
            decoder.decode_forward(token, current_pos=current_pos, page_table=page_table, rope_pos_ids=rope_pos_ids)
        )
    assert not guard.violations, f"host fallback in the optimized path: {sorted(set(guard.violations))}"


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_determinism_repeated_inputs(mesh_device, decoder_cache, kind):
    """Identical inputs must produce bit-identical prefill and decode outputs."""
    decoder = build_optimized(mesh_device, decoder_cache, kind)
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


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_traced_decode_pcc(mesh_device, decoder_cache, reference_layers, kind):
    """Decode PCC measured from a *trace replay*, which is the measured perf path."""
    layer_idx, layer = reference_layers[kind]
    decoder = build_optimized(mesh_device, decoder_cache, kind)
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
    expected = R.reference_decode(layer, layer_idx, token, past_key_values=cache, positions=positions)
    assert_pcc(
        f"optimized traced decode replay[{kind}] pos={prompt_len}",
        expected,
        ttnn.to_torch(tt_out).reshape(1, 1, -1),
    )
    ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_traced_decode_advances_positions(mesh_device, decoder_cache, reference_layers, kind):
    """One captured trace, three positions: the replay reads the refreshed inputs."""
    layer_idx, layer = reference_layers[kind]
    decoder = build_optimized(mesh_device, decoder_cache, kind)
    prompt_len = 1024
    hidden = R.synthetic_hidden_states(1, prompt_len, seed=333)
    _, cache = R.reference_prefill(layer, layer_idx, hidden)
    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ, seed=334)
    ttnn.deallocate(decoder.prefill_forward(to_device_hidden(mesh_device, hidden), page_table=page_table, user_id=0))

    hidden_size = decoder.config.hidden_size
    tt_token = to_device_hidden(mesh_device, R.synthetic_hidden_states(1, 1, seed=1))
    current_pos, rope_pos_ids = decode_position_tensors(mesh_device, torch.tensor([prompt_len]))
    trace_id, tt_out = capture_decode_trace(decoder, mesh_device, tt_token, current_pos, page_table, rope_pos_ids)

    for step in range(3):
        position = prompt_len + step
        token = R.synthetic_hidden_states(1, 1, seed=4000 + step)
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
            f"optimized traced decode[{kind}] advanced pos={position}",
            expected,
            ttnn.to_torch(tt_out).reshape(1, 1, -1),
        )
    ttnn.release_trace(mesh_device, trace_id)


STRESS_STEPS = int(os.environ.get("MG_STRESS_STEPS", "64"))


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_repeated_run_stress(mesh_device, decoder_cache, reference_layers, kind):
    """``STRESS_STEPS`` traced decode replays advancing the position every step.

    The soak the two new layout features need: the boundary/MLP reshard pair runs
    every step inside the trace, and the BFP8 cache is written and read every
    step, so a leaked L1 shard or a cache-repack fault compounds here where the
    single-shot tests would miss it.
    """
    layer_idx, layer = reference_layers[kind]
    decoder = build_optimized(mesh_device, decoder_cache, kind)
    prompt_len = 1024
    hidden = R.synthetic_hidden_states(1, prompt_len, seed=515)
    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ, seed=616)

    for _ in range(4):
        ttnn.deallocate(
            decoder.prefill_forward(to_device_hidden(mesh_device, hidden), page_table=page_table, user_id=0)
        )
    _, cache = R.reference_prefill(layer, layer_idx, hidden)

    hidden_size = decoder.config.hidden_size
    tt_token = to_device_hidden(mesh_device, R.synthetic_hidden_states(1, 1, seed=1))
    current_pos, rope_pos_ids = decode_position_tensors(mesh_device, torch.tensor([prompt_len]))
    trace_id, tt_out = capture_decode_trace(decoder, mesh_device, tt_token, current_pos, page_table, rope_pos_ids)

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
        last = step
        if step % 16 == 0 or step == STRESS_STEPS - 1:
            assert_pcc(f"optimized stress decode[{kind}] step={step} pos={position}", expected, actual)
    ttnn.release_trace(mesh_device, trace_id)
    assert last == STRESS_STEPS - 1


@pytest.mark.timeout(600)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_decode_layout_contract(mesh_device, decoder_cache, kind):
    """Every batch the contract allows lands on the same tile-padded shard spec.

    OPT-005: the shard height is the *tile-padded* row count, and every batch from
    1 to the op's 32-user ceiling pads to 32, so the norm/matmul configs must not
    vary with logical batch.
    """
    decoder = build_optimized(mesh_device, decoder_cache, kind)
    specs = set()
    for batch in (1, 2, 13, 31, 32):
        rows = ((batch + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
        program_config, memory_config = decoder._decode_norm_configs(rows)
        assert memory_config.shard_spec.grid.num_cores() == BOUNDARY_CORES
        assert memory_config.shard_spec.shape[1] == decoder.config.hidden_size // BOUNDARY_CORES
        assert program_config.block_h == 1
        specs.add((rows, memory_config.shard_spec.shape[1]))
    assert specs == {(TILE_SIZE, decoder.config.hidden_size // BOUNDARY_CORES)}


@pytest.mark.timeout(600)
def test_prefill_block_table_is_legal():
    """Every ``PREFILL_MINIMAL_BLOCKS`` entry is self-consistent, host-side.

    Cheap guard on the table's shape: thresholds descending, blocks positive, and
    the lookup returning the entry the thresholds imply.
    """
    for (role, dtype), entries in PREFILL_MINIMAL_BLOCKS.items():
        thresholds = [min_rows for min_rows, _ in entries]
        assert thresholds == sorted(thresholds, reverse=True), f"{role}/{dtype} thresholds are not descending"
        assert thresholds[-1] == TILE_SIZE, f"{role}/{dtype} has no entry covering the smallest prefill"
        for min_rows, blocks in entries:
            if blocks is None:
                continue
            assert all(b > 0 for b in blocks), f"{role}/{dtype}@{min_rows} has a non-positive block"
            assert minimal_matmul_blocks(role, min_rows, dtype) == blocks
    # Every role the shipped policy uses must be covered at the dtype it uses.
    for role in ("wqkv", "attn_gate", "o_proj", "mlp_gate", "mlp_up", "mlp_down"):
        assert (
            role,
            DEFAULT_PRECISION.weight_dtype(role),
        ) in PREFILL_MINIMAL_BLOCKS, f"{role} has no swept prefill blocking at its shipped dtype"


@pytest.mark.timeout(600)
def test_prefill_mcast_table_is_legal(mesh_device):
    """``PREFILL_MCAST2D`` is self-consistent, and its grid cannot miscompute.

    The second assertion is the important one and it guards a **silent** failure.
    With a width-sharded DRAM ``input_tensor_b``, the 2D-multicast matmul is only
    correct when the program grid's core-*column* count equals the DRAM bank
    count.  At ``grid_x = 9`` or ``11`` on this part the op validates, launches,
    and returns ``inf`` in tens of thousands of elements
    (``doc/optimized_decoder/logs/mcast_gx_bug_repro.log``); nothing in TTNN
    rejects it.  So the layer must never be able to build such a config, and the
    only way to keep that true across future grid changes is to assert it here
    rather than to rely on the table's authors remembering.
    """
    banks = mesh_device.dram_grid_size().x
    grid = mesh_device.compute_with_storage_grid_size()
    shapes = {
        "wqkv": (6656, 4608),
        "attn_gate": (6656, 4096),
        "o_proj": (4096, 6656),
        "mlp_gate": (6656, 19968),
        "mlp_up": (6656, 19968),
        "mlp_down": (19968, 6656),
    }
    for (role, dtype), entries in PREFILL_MCAST2D.items():
        thresholds = [max_rows for max_rows, _ in entries]
        assert thresholds == sorted(thresholds), f"{role}/{dtype} bands are not ascending"
        assert (
            prefill_mcast2d_spec(role, thresholds[-1] + TILE_SIZE, dtype) is None
        ), f"{role}/{dtype} must hand row counts past its last band back to minimal_matmul"
        k, n = shapes[role]
        # A padded N that is not an exact multiple of 32 * banks would give the
        # matmul a per_core_N that over-runs the shard set, which is the same
        # failure mode as an over-wide grid.
        assert (n // TILE_SIZE) % banks == 0, f"{role}: {n // TILE_SIZE} N tiles do not divide {banks} DRAM banks"
        for max_rows, spec in entries:
            grid_y, in0_block_w = spec
            assert grid_y <= grid.y, f"{role}/{dtype}@{max_rows}: grid_y={grid_y} exceeds the device grid"
            assert (
                k // TILE_SIZE
            ) % in0_block_w == 0, (
                f"{role}/{dtype}@{max_rows}: in0_block_w={in0_block_w} does not divide {k // TILE_SIZE} K tiles"
            )
            assert prefill_mcast2d_spec(role, max_rows, dtype) == spec
            program_config = prefill_mcast2d_program_config(max_rows, n, grid_y, in0_block_w, banks)
            program_grid = program_config.compute_with_storage_grid_size
            assert (program_grid.x, program_grid.y) == (banks, grid_y), (
                f"{role}/{dtype}@{max_rows}: the 2D-multicast grid must be {banks} columns wide "
                "or the width-sharded DRAM weight is read out of range and the output silently contains inf"
            )
            assert program_config.per_core_N * banks == n // TILE_SIZE
            # The L1 bound.  ``max_rows`` is the worst case inside the band, so if
            # per_core_M is small here it is small everywhere the band applies.
            # A lower-bound band would put per_core_M=8 on a 2016-row batched
            # prefill and overflow the static circular buffers; the ascending
            # form cannot, and this is where that is pinned.
            assert program_config.per_core_M <= MCAST_MAX_PER_CORE_M, (
                f"{role}/{dtype}@{max_rows}: per_core_M={program_config.per_core_M} exceeds the measured, "
                f"L1-safe bound of {MCAST_MAX_PER_CORE_M}"
            )
    # Every role the shipped policy uses is covered at the dtype it uses, so no
    # role silently keeps the slower minimal_matmul across the whole short band.
    for role in ("wqkv", "attn_gate", "o_proj", "mlp_gate", "mlp_up", "mlp_down"):
        assert (
            role,
            DEFAULT_PRECISION.weight_dtype(role),
        ) in PREFILL_MCAST2D, f"{role} has no swept 2D-multicast band at its shipped dtype"


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_prefill_norm_is_sharded_and_rectangular(mesh_device, decoder_cache, kind):
    """The short-prefill norms are sharded, on an exact rectangle, and only below the band.

    Three things worth pinning, because the middle one is a silent-corruption guard
    rather than a performance preference:

    * below ``PREFILL_NORM_SHARD_MAX_ROWS`` the four hidden-size prefill norms run
      width-sharded in L1 (~134 us each on 4 cores interleaved -> 33 us on 16);
    * their shard is an exact ``gx * gy`` rectangle and the norm program grid covers
      exactly it.  A wider program grid returns ``inf`` at ``block_h > 1`` with no
      error (``doc/optimized_decoder/logs/sharded_norm_grid_probe.log``);
    * above the band they fall back to interleaved, because the shard stops fitting
      L1 at 512 rows.
    """
    decoder = build_optimized(mesh_device, decoder_cache, kind)
    grid = mesh_device.compute_with_storage_grid_size()

    rect = core_rectangle(PREFILL_NORM_SHARD_CORES, grid)
    assert rect is not None, f"{PREFILL_NORM_SHARD_CORES} cores must have an exact rectangle"
    for rows in (TILE_SIZE, 128, PREFILL_NORM_SHARD_MAX_ROWS):
        program_config, memory_config = decoder._prefill_norm_configs(rows)
        program_grid = program_config.compute_with_storage_grid_size
        shard_cores = memory_config.shard_spec.grid.num_cores()
        assert (program_grid.x, program_grid.y) == rect
        assert (
            program_grid.x * program_grid.y == shard_cores == PREFILL_NORM_SHARD_CORES
        ), "the prefill norm program grid must cover exactly its shard, or it silently miscomputes"
        assert memory_config.shard_spec.grid.num_cores() == PREFILL_NORM_SHARD_CORES

    # Dispatch: sharded inside the band, interleaved outside it.
    seen = []
    original = ttnn.rms_norm

    def traced(x, **kwargs):
        spec = x.memory_config().shard_spec
        seen.append(None if spec is None else spec.grid.num_cores())
        return original(x, **kwargs)

    ttnn.rms_norm = traced
    try:
        for seq_len, expect_sharded in ((128, True), (2048, False)):
            seen.clear()
            page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ, seed=57)
            ttnn.deallocate(
                decoder.prefill_forward(
                    to_device_hidden(mesh_device, R.synthetic_hidden_states(1, seq_len, seed=58)),
                    page_table=page_table,
                    user_id=0,
                )
            )
            # The four hidden-size norms; the two per-head QK norms are height-sharded
            # and inherited, so filter to the hidden-size core count.
            hidden_size_norms = [c for c in seen if c in (None, PREFILL_NORM_SHARD_CORES)]
            if expect_sharded:
                assert (
                    hidden_size_norms.count(PREFILL_NORM_SHARD_CORES) == 4
                ), f"expected 4 sharded hidden-size prefill norms at {seq_len} rows, saw {seen}"
            else:
                assert PREFILL_NORM_SHARD_CORES not in seen, (
                    f"the prefill norm shard does not fit L1 above {PREFILL_NORM_SHARD_MAX_ROWS} rows, "
                    f"but {seq_len} rows dispatched one: {seen}"
                )
    finally:
        ttnn.rms_norm = original


@pytest.mark.timeout(1800)
def test_decode_norm_refuses_the_silently_corrupting_shape(mesh_device, decoder_cache, expect_error):
    """``_decode_norm_configs`` raises rather than build a >1 tile-row unsafe norm.

    The decode boundary shard is a row-major *prefix*, not a rectangle, because the
    DRAM-sharded matmul ignores the output shard grid it is handed and writes its own
    storage layout.  So the decode norm program grid is necessarily wider than its
    shard (``11x2 = 22`` cores over a 16-core shard), which is correct at
    ``block_h == 1`` and returns ``inf`` above it.  A decode step is always one tile
    row, so this is safe -- and the guard is what keeps it safe if that ever changes.
    """
    decoder = build_optimized(mesh_device, decoder_cache, LAYER_KIND_SLIDING)
    grid = mesh_device.compute_with_storage_grid_size()
    program_config, memory_config = decoder._decode_norm_configs(TILE_SIZE)
    program_grid = program_config.compute_with_storage_grid_size
    shard_cores = memory_config.shard_spec.grid.num_cores()
    assert shard_cores == BOUNDARY_CORES
    if program_grid.x * program_grid.y != shard_cores:
        # The forced-prefix case this model is actually in: only block_h == 1 is legal.
        assert program_config.block_h == 1
        with expect_error(ValueError, "silently returns inf"):
            decoder._decode_norm_configs(TILE_SIZE * 2)
    else:
        assert core_rectangle(shard_cores, grid) == (program_grid.x, program_grid.y)


@pytest.mark.timeout(600)
def test_decode_geometry_table_is_legal():
    """``DECODE_MATMUL``'s ``in0_block_w`` divides ``K / (32 * cores)`` everywhere.

    An illegal entry fails at op-compile time inside a trace capture, which is a
    much worse place to find out.
    """
    shapes = {
        "wqkv": (6656, 4608),
        "attn_gate": (6656, 4096),
        "o_proj": (4096, 6656),
        "mlp_gate": (6656, 19968),
        "mlp_up": (6656, 19968),
        "mlp_down": (19968, 6656),
    }
    for (role, dtype), (cores, in0_block_w) in DECODE_MATMUL.items():
        k, _ = shapes[role]
        k_tiles = k // TILE_SIZE
        assert k_tiles % cores == 0, f"{role}/{dtype}: {cores} cores do not divide {k_tiles} K tiles"
        per_core = k_tiles // cores
        assert (
            per_core % in0_block_w == 0
        ), f"{role}/{dtype}: in0_block_w={in0_block_w} does not divide the {per_core}-tile activation shard"


# --------------------------------------------------------------------------- perf

PERF_DECODE_ITERS = int(os.environ.get("MG_PERF_DECODE_ITERS", "8"))
#: 128 is here because it is the row count where this stage's 2D-multicast
#: prefill matmul runs; without it no committed perf table would contain a
#: single ``MatmulMultiCoreReuseMultiCast`` row and the largest prefill win in
#: the stage would be evidenced only by a bench script.
PERF_PREFILL_SEQS = (128, 8192, 16384)


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("seq_len", PERF_PREFILL_SEQS)
def test_perf_prefill(mesh_device, decoder_cache, kind, seq_len):
    """Warmed prefill, signposted for tt-perf-report (same window as the fused stage)."""
    decoder = build_optimized(mesh_device, decoder_cache, kind)
    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ, seed=3)
    tt_hidden = to_device_hidden(mesh_device, R.synthetic_hidden_states(1, seq_len, seed=42))

    for _ in range(2):
        ttnn.deallocate(decoder.prefill_forward(tt_hidden, page_table=page_table, user_id=0))
    ttnn.synchronize_device(mesh_device)

    signpost("PERF_PREFILL")
    out = decoder.prefill_forward(tt_hidden, page_table=page_table, user_id=0)
    ttnn.synchronize_device(mesh_device)
    signpost("PERF_PREFILL_END")
    logger.info(f"optimized prefill perf window done: kind={kind} seq_len={seq_len} shape={tuple(out.shape)}")
    ttnn.deallocate(out)


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("context", (2048, HF_ADVERTISED_CONTEXT - 1))
def test_perf_decode_traced(mesh_device, decoder_cache, kind, context):
    """Warmed traced decode, signposted; mirrors the fused stage's window exactly."""
    max_seq_len = SHORT_MAX_SEQ if context < SHORT_MAX_SEQ else HF_ADVERTISED_CONTEXT
    decoder = build_optimized(mesh_device, decoder_cache, kind, max_seq_len=max_seq_len)
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

    for _ in range(4):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)

    signpost("PERF_DECODE")
    for _ in range(PERF_DECODE_ITERS):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    signpost("PERF_DECODE_END")
    logger.info(
        f"optimized decode perf window done: kind={kind} context={context} iters={PERF_DECODE_ITERS} "
        f"shape={tuple(tt_out.shape)}"
    )
    ttnn.release_trace(mesh_device, trace_id)
