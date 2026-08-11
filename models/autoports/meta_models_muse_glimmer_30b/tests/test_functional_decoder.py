# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness and performance tests for the Muse-Glimmer-30B functional decoder.

The model has two decoder-layer kinds (``sliding`` = sliding-window + RoPE,
``full`` = full-attention + NoPE); every test that can be is parameterised over
both.  Layer 0 is the first ``sliding`` layer and layer 3 the first ``full``
layer of the released 52-layer checkpoint.
"""

from __future__ import annotations

import gc
import os
from collections import OrderedDict

import pytest
import torch
from loguru import logger

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tests import reference as R
from models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder import (
    LAYER_KIND_FULL,
    LAYER_KIND_SLIDING,
    FunctionalDecoder,
    reference_layer_indices,
)
from models.common.utility_functions import comp_pcc

try:
    from tracy import signpost
except ImportError:  # pragma: no cover - only present under the Tracy profiler

    def signpost(header: str) -> None:
        del header


PCC_THRESHOLD = 0.995
PAGE_BLOCK_SIZE = 64
PREFILL_CHUNK_SIZE = 8192
HF_ADVERTISED_CONTEXT = 131072
SHORT_MAX_SEQ = 16384

LAYER_KINDS = (LAYER_KIND_SLIDING, LAYER_KIND_FULL)


def layer_idx_for(kind: str) -> int:
    return reference_layer_indices(R.hf_config())[kind]


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="session")
def mesh_device():
    """One 1x1 mesh for the whole session (the functional stage is single-chip)."""
    if ttnn.get_num_devices() < 1:  # pragma: no cover - no hardware
        pytest.skip("no Tenstorrent device available")
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), trace_region_size=0)
    previous_default = ttnn.GetDefaultDevice()
    ttnn.SetDefaultDevice(mesh)
    try:
        yield mesh
    finally:
        ttnn.SetDefaultDevice(previous_default)
        ttnn.close_mesh_device(mesh)


class _DecoderCache:
    """Small LRU of built decoders — a rebuild uploads ~1 GB of weights."""

    def __init__(self, capacity: int = 2) -> None:
        self._entries: OrderedDict = OrderedDict()
        self._capacity = capacity

    def get(self, key, factory):
        if key in self._entries:
            self._entries.move_to_end(key)
            return self._entries[key]
        while len(self._entries) >= self._capacity:
            self._entries.popitem(last=False)
            gc.collect()
        decoder = factory()
        self._entries[key] = decoder
        return decoder

    def clear(self) -> None:
        self._entries.clear()
        gc.collect()


@pytest.fixture(scope="session")
def decoder_cache():
    cache = _DecoderCache()
    yield cache
    cache.clear()


@pytest.fixture(scope="session")
def reference_layers():
    layers = {}
    for kind in LAYER_KINDS:
        idx = layer_idx_for(kind)
        layers[kind] = (idx, R.reference_layer(idx, R.synthetic_state_dict(idx)))
    return layers


def build_decoder(
    mesh_device,
    decoder_cache,
    kind: str,
    *,
    max_seq_len: int = SHORT_MAX_SEQ,
    max_batch_size: int = 1,
    real_weights: bool = False,
    chunk: int = PREFILL_CHUNK_SIZE,
) -> FunctionalDecoder:
    layer_idx = layer_idx_for(kind)
    key = (layer_idx, max_seq_len, max_batch_size, real_weights, chunk)

    def factory():
        state_dict = R.real_state_dict(layer_idx) if real_weights else R.synthetic_state_dict(layer_idx)
        return FunctionalDecoder.from_state_dict(
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


# ----------------------------------------------------------------------------- utils


def make_page_table(mesh_device, batch: int, max_seq_len: int, *, seed: int = 7) -> ttnn.Tensor:
    """Randomly permuted, non-identity block assignment across all users."""
    blocks_per_seq = (max_seq_len + PAGE_BLOCK_SIZE - 1) // PAGE_BLOCK_SIZE
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(batch * blocks_per_seq, generator=generator)
    return ttnn.from_torch(
        permutation.reshape(batch, blocks_per_seq).to(torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def to_device_hidden(mesh_device, hidden: torch.Tensor) -> ttnn.Tensor:
    """``[batch, seq, hidden]`` torch -> ``[1, 1, batch*seq, hidden]`` TTNN tile tensor."""
    flat = hidden.reshape(1, 1, hidden.shape[0] * hidden.shape[1], hidden.shape[2])
    return ttnn.from_torch(
        flat,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def decode_position_tensors(mesh_device, positions: torch.Tensor):
    current_pos = ttnn.from_torch(
        positions.to(torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    rope_pos_ids = ttnn.from_torch(
        positions.reshape(1, -1).to(torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return current_pos, rope_pos_ids


def assert_pcc(label: str, expected: torch.Tensor, actual: torch.Tensor, threshold: float = PCC_THRESHOLD):
    passed, message = comp_pcc(expected.float(), actual.float(), threshold)
    logger.info(f"{label}: {message}")
    assert passed, f"{label} below PCC {threshold}: {message}"
    return message


# ------------------------------------------------------------------- prefill / decode

# Coverage rationale for the sequence lengths below:
#   1      minimal smoke (sub-tile)
#   100    non-aligned, sub-tile-multiple, inside one page block
#   128    exactly 4 tiles / 2 page blocks
#   2048   exactly the sliding window
#   2049   one token past the sliding window
#   4097   one token past a page-block-aligned length, mid-chunk
#   8192   exactly the prefill chunk size
#   8193   one token past the prefill chunk size (forces a 1-token second chunk)
#   12345  long, divisible by neither the tile, the page block nor the chunk
PREFILL_SEQ_LENS = (1, 100, 128, 2048, 2049, 4097, 8192, 8193, 12345)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("seq_len", PREFILL_SEQ_LENS)
def test_prefill_pcc(mesh_device, decoder_cache, reference_layers, kind, seq_len):
    layer_idx, layer = reference_layers[kind]
    decoder = build_decoder(mesh_device, decoder_cache, kind)
    hidden = R.synthetic_hidden_states(1, seq_len, seed=101 + seq_len)
    expected, _ = R.reference_prefill(layer, layer_idx, hidden)

    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ)
    tt_hidden = to_device_hidden(mesh_device, hidden)
    tt_out = decoder.prefill_forward(tt_hidden, page_table=page_table, user_id=0)
    actual = ttnn.to_torch(tt_out).reshape(1, seq_len, -1)
    assert tuple(tt_out.shape) == (1, 1, seq_len, decoder.config.hidden_size)
    assert_pcc(f"prefill[{kind}] seq_len={seq_len}", expected, actual)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_prefill_decode_pcc_vs_fp32_reference(mesh_device, decoder_cache, kind):
    """Control against an FP32 HF layer, not just the BF16 one.

    Every other PCC number compares BF16 TTNN against a BF16 HF layer; an error
    that is common-mode between two bfloat16 implementations would hide there.
    The same weights and the same (BF16-rounded) inputs are used, upcast to
    FP32, so only the compute precision differs.
    """
    layer_idx = layer_idx_for(kind)
    state_dict = R.synthetic_state_dict(layer_idx)
    layer = R.reference_layer(layer_idx, state_dict, dtype=torch.float32)
    decoder = build_decoder(mesh_device, decoder_cache, kind)

    seq_len = 2049
    hidden_bf16 = R.synthetic_hidden_states(1, seq_len, seed=606060)
    expected, cache = R.reference_prefill(layer, layer_idx, hidden_bf16.float())

    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ, seed=6060)
    tt_out = decoder.prefill_forward(to_device_hidden(mesh_device, hidden_bf16), page_table=page_table, user_id=0)
    assert_pcc(
        f"prefill[{kind}] vs FP32 HF reference seq_len={seq_len}",
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
        f"decode[{kind}] vs FP32 HF reference pos={seq_len}",
        expected,
        ttnn.to_torch(tt_out).reshape(1, 1, -1),
    )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_prefill_seq_len_equals_max_and_chunk(mesh_device, decoder_cache, reference_layers, kind):
    """``max_seq_len == prefill_chunk_size == seq_len``.

    Regression for the full-range ``ttnn.slice`` aliasing hazard: at this shape
    the RoPE-table slice covers the whole persistent cos/sin table, so a blind
    deallocate would free the layer's own buffers.  A second prefill on the same
    decoder proves the tables survived.
    """
    layer_idx, layer = reference_layers[kind]
    seq_len = 4096
    decoder = build_decoder(mesh_device, decoder_cache, kind, max_seq_len=seq_len, max_batch_size=1, chunk=seq_len)
    page_table = make_page_table(mesh_device, 1, seq_len, seed=771)
    for attempt in range(2):
        hidden = R.synthetic_hidden_states(1, seq_len, seed=880 + attempt)
        expected, _ = R.reference_prefill(layer, layer_idx, hidden)
        tt_out = decoder.prefill_forward(to_device_hidden(mesh_device, hidden), page_table=page_table, user_id=0)
        actual = ttnn.to_torch(tt_out).reshape(1, seq_len, -1)
        assert_pcc(f"prefill[{kind}] seq_len==max_seq_len==chunk=={seq_len} attempt={attempt}", expected, actual)
        ttnn.deallocate(tt_out)


# ``first_len`` must be a page-block multiple (the continuation contract);
# ``second_len`` is deliberately not.  64 and 1024 are below the 2048 sliding
# window, so the handed-over tail is *shorter* than the window — the regime the
# internal tail-carry bug (work log bug #1) lived in.
CONTINUATION_SPLITS = ((4096, 3000), (1024, 1024), (64, 100))


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("first_len,second_len", CONTINUATION_SPLITS)
def test_continuation_prefill_pcc(mesh_device, decoder_cache, reference_layers, kind, first_len, second_len):
    """Caller-chunked prefill: two ``start_pos``-separated calls == one call.

    ``full`` layers read the prefix straight out of the paged cache; ``sliding``
    layers are handed the previous call's K/V window explicitly.  The reference
    is a single-shot HF prefill over the concatenated prompt.
    """
    layer_idx, layer = reference_layers[kind]
    decoder = build_decoder(mesh_device, decoder_cache, kind)
    total = first_len + second_len
    hidden = R.synthetic_hidden_states(1, total, seed=90210 + first_len)
    expected, _ = R.reference_prefill(layer, layer_idx, hidden)

    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ, seed=4242)
    first = decoder.prefill_forward(
        to_device_hidden(mesh_device, hidden[:, :first_len]),
        page_table=page_table,
        user_id=0,
        return_sliding_kv_tail=True,
    )
    first_out, tail = first
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
    assert_pcc(f"continuation prefill[{kind}] {first_len}+{second_len}", expected, actual)

    # Decode past the continuation must still read the right paged prefix.
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
        f"decode after continuation prefill[{kind}] pos={total}",
        expected_decode,
        ttnn.to_torch(tt_out).reshape(1, 1, -1),
    )


@pytest.mark.timeout(600)
def test_continuation_prefill_requires_sliding_tail(mesh_device, decoder_cache, expect_error):
    """A sliding continuation without its window must fail loudly, not silently."""
    decoder = build_decoder(mesh_device, decoder_cache, LAYER_KIND_SLIDING)
    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ, seed=4243)
    tt_hidden = to_device_hidden(mesh_device, R.synthetic_hidden_states(1, 128, seed=1))
    with expect_error(ValueError, "sliding_kv_tail"):
        decoder.prefill_forward(tt_hidden, page_table=page_table, user_id=0, start_pos=4096)


def test_resolve_layer_kind_rejects_unsupported_pairings(expect_error):
    """Every checkpoint layer maps to a supported kind; other pairings raise."""
    from copy import deepcopy

    from models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder import resolve_layer_kind

    config = R.hf_config()
    kinds = {resolve_layer_kind(config, i) for i in range(config.text_config.num_hidden_layers)}
    assert kinds == set(LAYER_KINDS)

    broken = deepcopy(config)
    broken.text_config.layer_types = list(broken.text_config.layer_types)
    broken.text_config.layer_rope_theta = list(broken.text_config.layer_rope_theta)
    broken.text_config.layer_types[0] = "full_attention"  # full + RoPE is not a released combination
    with expect_error(ValueError, "unsupported"):
        resolve_layer_kind(broken, 0)
    broken.text_config.layer_rope_theta[1] = 0  # sliding + NoPE is not either
    with expect_error(ValueError, "unsupported"):
        resolve_layer_kind(broken, 1)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("prompt_len", (100, 2048, 3000))
def test_decode_pcc(mesh_device, decoder_cache, reference_layers, kind, prompt_len):
    """Prefill, then decode several tokens past the prompt (paged update path)."""
    layer_idx, layer = reference_layers[kind]
    decoder = build_decoder(mesh_device, decoder_cache, kind)
    hidden = R.synthetic_hidden_states(1, prompt_len, seed=202 + prompt_len)
    _, cache = R.reference_prefill(layer, layer_idx, hidden)

    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ)
    tt_hidden = to_device_hidden(mesh_device, hidden)
    ttnn.deallocate(decoder.prefill_forward(tt_hidden, page_table=page_table, user_id=0))

    for step in range(4):
        position = prompt_len + step
        token = R.synthetic_hidden_states(1, 1, seed=909 + step)
        expected = R.reference_decode(
            layer, layer_idx, token, past_key_values=cache, positions=torch.tensor([position])
        )
        current_pos, rope_pos_ids = decode_position_tensors(mesh_device, torch.tensor([position]))
        tt_token = to_device_hidden(mesh_device, token)
        tt_out = decoder.decode_forward(
            tt_token, current_pos=current_pos, page_table=page_table, rope_pos_ids=rope_pos_ids
        )
        actual = ttnn.to_torch(tt_out).reshape(1, 1, -1)
        assert_pcc(f"decode[{kind}] prompt={prompt_len} pos={position}", expected, actual)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("batch", (4, 13, 32))
def test_batched_prefill_decode_pcc(mesh_device, decoder_cache, reference_layers, kind, batch):
    """Independent users share one paged cache: per-user prefill then batched decode.

    ``batch=13`` is deliberate: 13 is prime and larger than the 11-wide device
    grid, so no ``batch``-core *rectangle* exists and the decode head-concat has
    to take its shape-agnostic fallback path.
    """
    layer_idx, layer = reference_layers[kind]
    max_seq_len = 4096
    decoder = build_decoder(mesh_device, decoder_cache, kind, max_seq_len=max_seq_len, max_batch_size=batch)
    page_table = make_page_table(mesh_device, batch, max_seq_len, seed=31 + batch)

    # Straddle the 2048 sliding window so the batched decode exercises the
    # per-user window-start arithmetic on both sides of the boundary.
    prompt_lens = [2000 + 37 * user for user in range(batch)]
    assert (decoder._decode_concat_grid_width(batch) is None) == (batch == 13)
    caches = []
    for user, prompt_len in enumerate(prompt_lens):
        hidden = R.synthetic_hidden_states(1, prompt_len, seed=4000 + user)
        expected, cache = R.reference_prefill(layer, layer_idx, hidden)
        caches.append(cache)
        tt_hidden = to_device_hidden(mesh_device, hidden)
        tt_out = decoder.prefill_forward(tt_hidden, page_table=page_table, user_id=user)
        actual = ttnn.to_torch(tt_out).reshape(1, prompt_len, -1)
        assert_pcc(f"prefill[{kind}] batch={batch} user={user} seq_len={prompt_len}", expected, actual)
        ttnn.deallocate(tt_out)

    # Decode with per-user (different) current positions in one batched call.
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
    tt_tokens = to_device_hidden(mesh_device, tokens)
    tt_out = decoder.decode_forward(
        tt_tokens, current_pos=current_pos, page_table=page_table, rope_pos_ids=rope_pos_ids
    )
    actual = ttnn.to_torch(tt_out).reshape(batch, 1, -1)
    assert_pcc(f"decode[{kind}] batch={batch} ragged positions", expected, actual)


@pytest.mark.timeout(900)
@pytest.mark.parametrize("cur_pos", (2047, 2048, 2049, 5000))
def test_decode_sdpa_sliding_window_semantics(mesh_device, cur_pos):
    """Direct control for the *decode* kernel's sliding-window boundary.

    ``sdpa_sliding_window_chunk_repro.py`` pins the prefill op's window against
    an explicit ``kv_idx > q_idx - W`` torch mask, but decode uses a different
    kernel (``SdpaDecodeDeviceOperation``).  An off-by-one there would move
    end-to-end PCC by ~1e-4, i.e. invisible under the BF16 floor, so this probes
    the op directly with a known paged cache — including the two positions
    either side of the window boundary.
    """
    text_config = R.text_config()
    window = text_config.sliding_window
    n_q, n_kv, head_dim = text_config.num_attention_heads, text_config.num_key_value_heads, text_config.head_dim
    block = PAGE_BLOCK_SIZE
    kv_len = ((cur_pos + 1 + block - 1) // block) * block
    blocks = kv_len // block

    generator = torch.Generator().manual_seed(4711)
    keys = torch.randn(1, n_kv, kv_len, head_dim, generator=generator) / 3
    values = torch.randn(1, n_kv, kv_len, head_dim, generator=generator) / 3
    query = torch.randn(1, 1, n_q, head_dim, generator=generator) / 3
    permutation = torch.randperm(blocks, generator=generator)

    def paged(source):
        paged_cache = torch.zeros(blocks, n_kv, block, head_dim)
        for logical in range(blocks):
            paged_cache[permutation[logical]] = source[0, :, logical * block : (logical + 1) * block, :]
        return paged_cache

    def to_dev(tensor, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
        return ttnn.from_torch(
            tensor, device=mesh_device, layout=layout, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    page_table = to_dev(permutation.reshape(1, blocks).to(torch.int32), ttnn.ROW_MAJOR_LAYOUT, ttnn.int32)
    out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        to_dev(query.to(torch.bfloat16)),
        to_dev(paged(keys).to(torch.bfloat16)),
        to_dev(paged(values).to(torch.bfloat16)),
        cur_pos_tensor=to_dev(torch.tensor([cur_pos], dtype=torch.int32), ttnn.ROW_MAJOR_LAYOUT, ttnn.int32),
        page_table_tensor=page_table,
        scale=0.342063,
        sliding_window_size=window,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    actual = ttnn.to_torch(out).reshape(n_q, head_dim).float()

    # HF: attends kv in (cur_pos - window, cur_pos], i.e. window tokens incl. self.
    index = torch.arange(kv_len)
    mask = (index <= cur_pos) & (index > cur_pos - window)
    q_grouped = query.reshape(n_kv, n_q // n_kv, head_dim).float()
    scores = torch.einsum("hgd,hkd->hgk", q_grouped, keys[0].float()) * 0.342063
    scores = scores.masked_fill(~mask, float("-inf"))
    expected = torch.einsum("hgk,hkd->hgd", torch.softmax(scores, dim=-1), values[0].float())
    assert_pcc(
        f"decode SDPA sliding-window control cur_pos={cur_pos} (window={window})",
        expected.reshape(n_q, head_dim),
        actual,
        threshold=0.999,
    )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_multi_chunk_prefill_nonzero_user(mesh_device, decoder_cache, reference_layers, kind):
    """Multi-chunk prefill into a non-zero cache slot.

    Combines the two things the batched test does not: an internal chunk
    boundary (``seq_len > prefill_chunk_size``) *and* ``user_id > 0``, so the
    page-table row slicing and the chunked paged read both run off row 2.
    """
    layer_idx, layer = reference_layers[kind]
    seq_len = 12345
    decoder = build_decoder(mesh_device, decoder_cache, kind, max_seq_len=SHORT_MAX_SEQ, max_batch_size=4)
    page_table = make_page_table(mesh_device, 4, SHORT_MAX_SEQ, seed=555)
    hidden = R.synthetic_hidden_states(1, seq_len, seed=13579)
    expected, cache = R.reference_prefill(layer, layer_idx, hidden)

    tt_out = decoder.prefill_forward(to_device_hidden(mesh_device, hidden), page_table=page_table, user_id=2)
    assert_pcc(
        f"multi-chunk prefill[{kind}] user_id=2 seq_len={seq_len}",
        expected,
        ttnn.to_torch(tt_out).reshape(1, seq_len, -1),
    )
    ttnn.deallocate(tt_out)

    # Decode for that slot only; the other three rows stay at position 0.
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
    actual = ttnn.to_torch(tt_out).reshape(4, 1, -1)[2:3]
    assert_pcc(f"decode[{kind}] user_id=2 pos={seq_len}", expected, actual)


# ------------------------------------------------------------------- full 128k context


def _reference_prefill_cache_only(layer, layer_idx, hidden: torch.Tensor, cache, *, chunk: int = 8192):
    """Fill the HF cache with the K/V of a long prompt without running attention.

    Mirrors ``MuseGlimmerTextAttention``'s K/V half exactly (norm -> k/v proj ->
    scale-less QK-norm -> RoPE -> ``cache.update``).  Running the real HF
    attention over 131072 queries on CPU is intractable; the K/V path is not,
    and it is what the decode step and the tail-of-prompt queries actually read.
    """
    from transformers.models.muse_glimmer.modeling_muse_glimmer import apply_rotary_pos_emb

    attn = layer.self_attn
    head_dim = attn.head_dim
    total = hidden.shape[1]
    with torch.no_grad():
        for start in range(0, total, chunk):
            piece = hidden[:, start : start + chunk]
            normed = layer.input_layernorm(piece)
            shape = (*normed.shape[:-1], -1, head_dim)
            key = attn.qk_norm(attn.k_proj(normed).view(shape).transpose(1, 2))
            value = attn.v_proj(normed).view(shape).transpose(1, 2)
            if R.uses_rope(layer_idx):
                position_ids = torch.arange(start, start + piece.shape[1]).unsqueeze(0)
                cos, sin = R.rope_embeddings(position_ids, piece.dtype)
                key, _ = apply_rotary_pos_emb(key, key, cos, sin)
            cache.update(key, value, layer_idx)
    return cache


@pytest.mark.slow
@pytest.mark.timeout(5400)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("seq_len", (HF_ADVERTISED_CONTEXT, HF_ADVERTISED_CONTEXT - 999))
def test_full_context_prefill_tail_pcc(mesh_device, decoder_cache, reference_layers, kind, seq_len):
    """HF-vs-TTNN prefill PCC at (and just under) the advertised 131072 context.

    The reference is a reduced layer-level harness: the HF cache is filled with
    the first ``seq_len - 32`` positions' K/V, then the *real* HF decoder layer
    runs over the final 32 query positions against that full prefix.  Those 32
    rows are compared against the same rows of a full ``seq_len`` TTNN prefill.
    """
    layer_idx, layer = reference_layers[kind]
    decoder = build_decoder(mesh_device, decoder_cache, kind, max_seq_len=HF_ADVERTISED_CONTEXT, max_batch_size=1)
    tail = 32
    hidden = R.synthetic_hidden_states(1, seq_len, seed=555 + seq_len)

    page_table = make_page_table(mesh_device, 1, HF_ADVERTISED_CONTEXT, seed=99)
    tt_hidden = to_device_hidden(mesh_device, hidden)
    tt_out = decoder.prefill_forward(tt_hidden, page_table=page_table, user_id=0)
    ttnn.deallocate(tt_hidden)

    from transformers.cache_utils import DynamicCache

    # The last 32 rows, plus an *interior* block of 32 rows at an internal
    # prefill-chunk boundary.  For a sliding layer the tail rows only see the
    # last ~2080 tokens, so the interior block is what actually validates the
    # mid-prompt chunk hand-offs (row 65536 is the first row of internal chunk
    # 9 and depends entirely on the carried window).
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
        assert_pcc(f"prefill[{kind}] full-context seq_len={seq_len} ({where} {tail} rows)", expected, actual)
    ttnn.deallocate(tt_out)


@pytest.mark.slow
@pytest.mark.timeout(5400)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_full_context_decode_pcc(mesh_device, decoder_cache, reference_layers, kind):
    """Decode at the last valid position of the advertised context (131071)."""
    layer_idx, layer = reference_layers[kind]
    decoder = build_decoder(mesh_device, decoder_cache, kind, max_seq_len=HF_ADVERTISED_CONTEXT, max_batch_size=1)
    prompt_len = HF_ADVERTISED_CONTEXT - 1
    hidden = R.synthetic_hidden_states(1, prompt_len, seed=777)

    page_table = make_page_table(mesh_device, 1, HF_ADVERTISED_CONTEXT, seed=123)
    tt_hidden = to_device_hidden(mesh_device, hidden)
    ttnn.deallocate(decoder.prefill_forward(tt_hidden, page_table=page_table, user_id=0))
    ttnn.deallocate(tt_hidden)

    from transformers.cache_utils import DynamicCache

    cache = DynamicCache(config=R.text_config())
    _reference_prefill_cache_only(layer, layer_idx, hidden, cache)

    position = prompt_len  # 131071, the last valid absolute position
    token = R.synthetic_hidden_states(1, 1, seed=778)
    expected = R.reference_decode(layer, layer_idx, token, past_key_values=cache, positions=torch.tensor([position]))
    current_pos, rope_pos_ids = decode_position_tensors(mesh_device, torch.tensor([position]))
    tt_out = decoder.decode_forward(
        to_device_hidden(mesh_device, token),
        current_pos=current_pos,
        page_table=page_table,
        rope_pos_ids=rope_pos_ids,
    )
    actual = ttnn.to_torch(tt_out).reshape(1, 1, -1)
    assert_pcc(f"decode[{kind}] full-context pos={position}", expected, actual)


# ----------------------------------------------------------------------- real weights


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_real_weights_prefill_decode_pcc(mesh_device, decoder_cache, kind):
    """Same contract, exercised against the released bf16 checkpoint."""
    layer_idx = layer_idx_for(kind)
    try:
        state_dict = R.real_state_dict(layer_idx)
    except FileNotFoundError as error:  # pragma: no cover - weights not cached
        # The stage contract requires real-weight evidence, so a missing
        # checkpoint is a failure unless the operator explicitly opts out.
        if os.environ.get("MG_ALLOW_MISSING_WEIGHTS", "0") == "1":
            pytest.skip(str(error))
        raise
    layer = R.reference_layer(layer_idx, state_dict)
    decoder = build_decoder(mesh_device, decoder_cache, kind, real_weights=True)

    seq_len = 2049  # non-aligned and past the sliding window
    hidden = R.synthetic_hidden_states(1, seq_len, seed=31337)
    expected, cache = R.reference_prefill(layer, layer_idx, hidden)

    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ, seed=64)
    tt_out = decoder.prefill_forward(to_device_hidden(mesh_device, hidden), page_table=page_table, user_id=0)
    actual = ttnn.to_torch(tt_out).reshape(1, seq_len, -1)
    assert_pcc(f"real-weights prefill[{kind}] seq_len={seq_len}", expected, actual)
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
    actual = ttnn.to_torch(tt_out).reshape(1, 1, -1)
    assert_pcc(f"real-weights decode[{kind}] pos={seq_len}", expected, actual)


def test_real_state_dict_key_and_shape_contract():
    """``from_state_dict`` consumes exactly the checkpoint's keys and shapes."""
    stats = R.weight_stats()["layers"]
    for kind in LAYER_KINDS:
        layer_idx = layer_idx_for(kind)
        try:
            real = R.real_state_dict(layer_idx)
        except FileNotFoundError as error:  # pragma: no cover - weights not cached
            if os.environ.get("MG_ALLOW_MISSING_WEIGHTS", "0") == "1":
                pytest.skip(str(error))
            raise
        expected_keys = {f"{R.layer_prefix(layer_idx)}.{suffix}" for suffix in R.LAYER_WEIGHT_SUFFIXES}
        assert set(real) == expected_keys
        for suffix in R.LAYER_WEIGHT_SUFFIXES:
            tensor = real[f"{R.layer_prefix(layer_idx)}.{suffix}"]
            assert list(tensor.shape) == stats[str(layer_idx)][suffix]["shape"]
            assert tensor.dtype is torch.bfloat16


# ------------------------------------------------------------------------ determinism


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_determinism_repeated_inputs(mesh_device, decoder_cache, kind):
    """Identical inputs must produce bit-identical prefill and decode outputs."""
    decoder = build_decoder(mesh_device, decoder_cache, kind)
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


# -------------------------------------------------------------------- fallback audit


#: Host-side entry points that would indicate a fallback inside a measured
#: prefill/decode pass.  ``ttnn.from_torch``/``to_torch`` are the device
#: boundary; the ``torch`` entries cover tensor creation and the compute /
#: layout helpers a hidden CPU path would realistically reach for.
_TTNN_FALLBACK_NAMES = ("from_torch", "to_torch", "as_tensor")
_TORCH_FALLBACK_NAMES = (
    "matmul",
    "tensor",
    "as_tensor",
    "from_numpy",
    "zeros",
    "ones",
    "empty",
    "full",
    "arange",
    "cat",
    "stack",
    "einsum",
    "softmax",
)


class _FallbackGuard:
    """Fail if a measured forward pass touches torch or a host round-trip."""

    def __init__(self):
        self.violations: list[str] = []
        self._saved: dict = {}

    def _trap(self, name, original):
        def trap(*args, **kwargs):
            self.violations.append(name)
            return original(*args, **kwargs)

        return trap

    def __enter__(self):
        import ttnn as ttnn_module

        for name in _TTNN_FALLBACK_NAMES:
            original = getattr(ttnn_module, name, None)
            if original is None:
                continue
            self._saved[("ttnn", name)] = original
            setattr(ttnn_module, name, self._trap(f"ttnn.{name}", original))
        for name in _TORCH_FALLBACK_NAMES:
            original = getattr(torch, name, None)
            if original is None:
                continue
            self._saved[("torch", name)] = original
            setattr(torch, name, self._trap(f"torch.{name}", original))
        return self

    def __exit__(self, *exc):
        import ttnn as ttnn_module

        module = {"ttnn": ttnn_module, "torch": torch}
        for (where, name), original in self._saved.items():
            setattr(module[where], name, original)
        return False


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("seq_len", (3000, 12345))
def test_no_host_fallback_in_forward(mesh_device, decoder_cache, kind, seq_len):
    """3000 is single-chunk; 12345 exercises the multi-chunk prefill paths
    (chunked paged SDPA, sliding tail concat/clone, page-table and RoPE-table
    slicing) inside the guard as well."""
    decoder = build_decoder(mesh_device, decoder_cache, kind)
    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ, seed=5)
    tt_hidden = to_device_hidden(mesh_device, R.synthetic_hidden_states(1, seq_len, seed=606))
    token = to_device_hidden(mesh_device, R.synthetic_hidden_states(1, 1, seed=607))
    current_pos, rope_pos_ids = decode_position_tensors(mesh_device, torch.tensor([seq_len]))

    with _FallbackGuard() as guard:
        out = decoder.prefill_forward(tt_hidden, page_table=page_table, user_id=0)
        ttnn.deallocate(out)
        out = decoder.decode_forward(token, current_pos=current_pos, page_table=page_table, rope_pos_ids=rope_pos_ids)
        ttnn.deallocate(out)
    assert not guard.violations, f"host fallback inside a measured pass: {sorted(set(guard.violations))}"


# --------------------------------------------------------------------- traced decode


def capture_decode_trace(decoder, mesh_device, tt_token, current_pos, page_table, rope_pos_ids):
    """Compile, capture and return ``(trace_id, output_tensor)`` for decode."""
    warm = decoder.decode_forward(tt_token, current_pos=current_pos, page_table=page_table, rope_pos_ids=rope_pos_ids)
    ttnn.deallocate(warm)
    ttnn.synchronize_device(mesh_device)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    tt_out = decoder.decode_forward(tt_token, current_pos=current_pos, page_table=page_table, rope_pos_ids=rope_pos_ids)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    return trace_id, tt_out


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_traced_decode_pcc(mesh_device, decoder_cache, reference_layers, kind):
    """Decode PCC measured from a *trace replay*, not an eager forward."""
    layer_idx, layer = reference_layers[kind]
    decoder = build_decoder(mesh_device, decoder_cache, kind)
    prompt_len = 2048
    hidden = R.synthetic_hidden_states(1, prompt_len, seed=1357)
    _, cache = R.reference_prefill(layer, layer_idx, hidden)

    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ, seed=246)
    ttnn.deallocate(decoder.prefill_forward(to_device_hidden(mesh_device, hidden), page_table=page_table, user_id=0))

    positions = torch.tensor([prompt_len])
    token = R.synthetic_hidden_states(1, 1, seed=2468)
    tt_token = to_device_hidden(mesh_device, token)
    current_pos, rope_pos_ids = decode_position_tensors(mesh_device, positions)

    # The warm-up call inside capture_decode_trace already consumed position
    # ``prompt_len``; re-running it during capture and replay writes the same
    # K/V to the same slot, so the reference stays a single decode step.
    trace_id, tt_out = capture_decode_trace(decoder, mesh_device, tt_token, current_pos, page_table, rope_pos_ids)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    actual = ttnn.to_torch(tt_out).reshape(1, 1, -1)
    ttnn.release_trace(mesh_device, trace_id)

    expected = R.reference_decode(layer, layer_idx, token, past_key_values=cache, positions=positions)
    assert_pcc(f"traced decode[{kind}] pos={prompt_len}", expected, actual)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_traced_decode_advances_positions(mesh_device, decoder_cache, reference_layers, kind):
    """One captured trace replays across positions when only tensor contents change."""
    layer_idx, layer = reference_layers[kind]
    decoder = build_decoder(mesh_device, decoder_cache, kind)
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
        # Refresh the stable input buffers *outside* the traced region.
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
        actual = ttnn.to_torch(tt_out).reshape(1, 1, -1)
        expected = R.reference_decode(
            layer, layer_idx, token, past_key_values=cache, positions=torch.tensor([position])
        )
        assert_pcc(f"traced decode replay[{kind}] pos={position}", expected, actual)
    ttnn.release_trace(mesh_device, trace_id)


# --------------------------------------------------------------------------- perf


PERF_PREFILL_SEQ = int(os.environ.get("MG_PERF_PREFILL_SEQ", "8192"))
PERF_DECODE_ITERS = int(os.environ.get("MG_PERF_DECODE_ITERS", "32"))


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_perf_prefill(mesh_device, decoder_cache, kind):
    """Warmed prefill, signposted for tt-perf-report."""
    decoder = build_decoder(mesh_device, decoder_cache, kind)
    page_table = make_page_table(mesh_device, 1, SHORT_MAX_SEQ, seed=3)
    tt_hidden = to_device_hidden(mesh_device, R.synthetic_hidden_states(1, PERF_PREFILL_SEQ, seed=42))

    for _ in range(2):  # compile + warm
        ttnn.deallocate(decoder.prefill_forward(tt_hidden, page_table=page_table, user_id=0))
    ttnn.synchronize_device(mesh_device)

    signpost("PERF_PREFILL")
    out = decoder.prefill_forward(tt_hidden, page_table=page_table, user_id=0)
    ttnn.synchronize_device(mesh_device)
    signpost("PERF_PREFILL_END")
    logger.info(f"prefill perf window done: kind={kind} seq_len={PERF_PREFILL_SEQ} shape={tuple(out.shape)}")
    ttnn.deallocate(out)


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize("context", (2048, HF_ADVERTISED_CONTEXT - 1))
def test_perf_decode_traced(mesh_device, decoder_cache, kind, context):
    """Warmed traced decode, signposted for tt-perf-report.

    ``context`` is the absolute decode position, i.e. the number of KV tokens
    the decode SDPA reads.  ``full`` (NoPE) layers read all of them, so the long
    case is materially more expensive; ``sliding`` layers read at most the 2048
    window either way.  For the long case the KV cache is not filled by a
    131071-token prefill (that would cost minutes of profiled setup and can
    overflow the profiler's marker buffer) — only ``current_pos`` is advanced.
    Decode cost does not depend on cache *contents*, only on how many tokens the
    op reads, which ``current_pos`` alone determines.
    """
    max_seq_len = SHORT_MAX_SEQ if context < SHORT_MAX_SEQ else HF_ADVERTISED_CONTEXT
    decoder = build_decoder(mesh_device, decoder_cache, kind, max_seq_len=max_seq_len)
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
        f"decode perf window done: kind={kind} context={context} iters={PERF_DECODE_ITERS} "
        f"shape={tuple(tt_out.shape)}"
    )
    ttnn.release_trace(mesh_device, trace_id)
