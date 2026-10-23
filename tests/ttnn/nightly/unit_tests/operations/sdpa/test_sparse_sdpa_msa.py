# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""sparse_sdpa_msa (MSA block-sparse prefill) nightly coverage, Blackhole.

Causal behavior is encoded by the block ids in `indices`; the op has no token-level causal mask.
`make_msa_inputs(causal=...)` only changes the selected blocks.

This file carries the broader shape, dtype, cache, production-sampled, and determinism coverage. The smaller
post-commit smoke lives in tests/ttnn/unit_tests/operations/sdpa/test_sparse_sdpa_msa.py.
"""

import pytest
import torch

import ttnn

from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.sdpa.sparse_sdpa_msa_test_utils import (
    BLK_KV,
    SENTINEL,
    sparse_attention_ref_msa,
    sparse_attention_ref_msa_sampled_tokens,
    dense_grouped_kv_attention,
    make_msa_inputs,
    run_op_msa_composed,
    run_op_msa_native,
    pcc,
)


# CPU reference checks.


@pytest.mark.parametrize("H,n_kv", [(16, 1), (32, 1)])
@pytest.mark.parametrize("S", [8, 33])
@pytest.mark.parametrize("nblk", [4, 8])
def test_golden_all_blocks_selected_equals_dense(H, n_kv, S, nblk):
    """Sparse attention with every block selected should match dense attention."""
    d = 128
    T = nblk * BLK_KV
    q, k, v, indices = make_msa_inputs(H, n_kv, S, T, topk=nblk, d=d, causal=False, seed=1)
    scale = d**-0.5
    ref = sparse_attention_ref_msa(q, k, v, indices, scale)
    dense = dense_grouped_kv_attention(q, k, v, scale)
    assert pcc(ref, dense) > 0.99999, f"sparse(all)=dense mismatch, pcc={pcc(ref, dense)}"


def test_golden_sentinel_tail_is_truncation():
    """A sentinel (-1) tail must contribute nothing: golden with k valid blocks + sentinels == golden with
    exactly those k blocks (no sentinel slots)."""
    d, H, n_kv, S, nblk = 128, 16, 1, 6, 8
    T = nblk * BLK_KV
    scale = d**-0.5
    gen = torch.Generator().manual_seed(3)
    q = torch.randn(1, H, S, d, generator=gen)
    k = torch.randn(1, n_kv, T, d, generator=gen)
    v = torch.randn(1, n_kv, T, d, generator=gen)
    chosen = torch.tensor([1, 3, 5], dtype=torch.int32)
    full = torch.full((1, n_kv, S, 6), SENTINEL, dtype=torch.int32)
    tight = torch.empty((1, n_kv, S, 3), dtype=torch.int32)
    for s in range(S):
        full[0, 0, s, :3] = chosen
        tight[0, 0, s] = chosen
    out_pad = sparse_attention_ref_msa(q, k, v, full, scale)
    out_tight = sparse_attention_ref_msa(q, k, v, tight, scale)
    assert pcc(out_pad, out_tight) > 0.99999


def test_golden_all_masked_row_is_zero_not_nan():
    """Reference-only behavior: all sentinels yield 0, but the device op requires at least one valid block."""
    d, H, n_kv, S, nblk = 128, 16, 1, 4, 8
    T = nblk * BLK_KV
    gen = torch.Generator().manual_seed(4)
    q = torch.randn(1, H, S, d, generator=gen)
    k = torch.randn(1, n_kv, T, d, generator=gen)
    v = torch.randn(1, n_kv, T, d, generator=gen)
    indices = torch.full((1, n_kv, S, 4), SENTINEL, dtype=torch.int32)  # nothing selected anywhere
    out = sparse_attention_ref_msa(q, k, v, indices, d**-0.5)
    assert torch.isfinite(out).all(), "all-masked row produced NaN/Inf"
    assert torch.count_nonzero(out) == 0, "all-masked row should be exactly zero"


# Op registration smoke.


def test_op_is_registered():
    assert hasattr(ttnn.transformer, "sparse_sdpa_msa"), "ttnn.transformer.sparse_sdpa_msa not registered"


pytestmark = pytest.mark.use_module_device


# Native device tests: separate K/V, block ids in `indices`. topk=16 keeps index rows DRAM-aligned.


@run_for_blackhole()
@pytest.mark.parametrize("H", [16, 32])
@pytest.mark.parametrize("S", [8, 33])
@pytest.mark.parametrize("topk,nblk", [(16, 16), (16, 32)])
def test_msa_native_pcc_random_selection(device, H, S, topk, nblk):
    d = 128
    T = nblk * BLK_KV
    q, k, v, indices = make_msa_inputs(H, 1, S, T, topk, d, causal=False, seed=1)
    gold = sparse_attention_ref_msa(q, k, v, indices, d**-0.5)
    out = run_op_msa_native(q, k, v, indices, device)
    p = pcc(out, gold)
    assert p > 0.99, f"native random-selection PCC {p}"


@run_for_blackhole()
@pytest.mark.parametrize("H", [16, 32])
@pytest.mark.parametrize("topk,nblk", [(16, 16), (16, 32)])
def test_msa_native_pcc_causal_selection(device, H, topk, nblk):
    """Causal block selection is encoded in `indices`; the op should match the reference."""
    d, S = 128, 200
    T = nblk * BLK_KV
    q, k, v, indices = make_msa_inputs(H, 1, S, T, topk, d, causal=True, seed=2)
    gold = sparse_attention_ref_msa(q, k, v, indices, d**-0.5)
    out = run_op_msa_native(q, k, v, indices, device)
    p = pcc(out, gold)
    assert p > 0.99, f"native causal-selection PCC {p}"


@run_for_blackhole()
def test_msa_native_multi_token_per_core(device):
    """Covers per-core loops with multiple query tokens per active core."""
    d, H, S, topk, nblk = 128, 16, 300, 16, 32
    T = nblk * BLK_KV
    q, k, v, indices = make_msa_inputs(H, 1, S, T, topk, d, causal=False, seed=7)
    gold = sparse_attention_ref_msa(q, k, v, indices, d**-0.5)
    out = run_op_msa_native(q, k, v, indices, device)
    p = pcc(out, gold)
    assert p > 0.99, f"multi-token/core PCC {p}"


@run_for_blackhole()
def test_msa_native_prod_shape_sampled_accuracy(device):
    """Accuracy check for the production-shape run used by the perf smoke."""
    d, H, S, T, topk = 128, 16, 640, 56320, 16
    q, k, v, indices = make_msa_inputs(H, 1, S, T, topk, d, causal=False, seed=7)
    sample_tokens = [0, 1, 127, 128, 319, 639]
    gold = sparse_attention_ref_msa_sampled_tokens(q, k, v, indices, d**-0.5, sample_tokens)
    out = run_op_msa_native(q, k, v, indices, device, kv_dtype=ttnn.bfloat8_b)
    out_sample = out[:, :, sample_tokens, :]
    p = pcc(out_sample, gold)
    assert p > 0.99, f"prod-shape sampled PCC {p}"


@run_for_blackhole()
@pytest.mark.parametrize("H,n_kv", [(32, 2), (64, 4), (64, 2)])
def test_msa_native_gqa_pcc_random_selection(device, H, n_kv):
    """Native GQA: each KV group serves H/n_kv query heads and owns its own block-id rows."""
    d, S, topk, nblk = 128, 33, 16, 16
    T = nblk * BLK_KV
    q, k, v, indices = make_msa_inputs(H, n_kv, S, T, topk, d, causal=False, seed=17 + n_kv)
    gold = sparse_attention_ref_msa(q, k, v, indices, d**-0.5)
    out = run_op_msa_native(q, k, v, indices, device, kv_dtype=ttnn.bfloat8_b)
    p = pcc(out, gold)
    assert p > 0.99, f"native GQA random-selection PCC {p}"


@run_for_blackhole()
def test_msa_native_gqa_group_isolation(device):
    """Distinct constant V per KV group catches accidentally reusing group 0 K/V or indices for every group."""
    d, H, n_kv, S, topk, nblk = 128, 64, 4, 16, 16, 16
    T = nblk * BLK_KV
    q, k, _, indices = make_msa_inputs(H, n_kv, S, T, topk, d, causal=False, seed=23)
    v = torch.empty(1, n_kv, T, d)
    for g in range(n_kv):
        v[:, g].fill_(float(g + 1))
    gold = sparse_attention_ref_msa(q, k, v, indices, d**-0.5)
    out = run_op_msa_native(q, k, v, indices, device)
    assert torch.max(torch.abs(out.float() - gold.float())).item() < 0.1


@run_for_blackhole()
def test_msa_native_prod_shape_gqa_sampled_accuracy(device):
    """Single-chip MiniMax3-style GQA shape: four KV groups, each with 16 query heads."""
    d, H, n_kv, S, T, topk = 128, 64, 4, 640, 56320, 16
    q, k, v, indices = make_msa_inputs(H, n_kv, S, T, topk, d, causal=False, seed=7)
    sample_tokens = [0, 1, 127, 128, 319, 639]
    gold = sparse_attention_ref_msa_sampled_tokens(q, k, v, indices, d**-0.5, sample_tokens)
    out = run_op_msa_native(q, k, v, indices, device, kv_dtype=ttnn.bfloat8_b)
    out_sample = out[:, :, sample_tokens, :]
    p = pcc(out_sample, gold)
    assert p > 0.99, f"prod-shape GQA sampled PCC {p}"


@run_for_blackhole()
def test_msa_native_pcc_sentinel_tail(device):
    """Rows with < topk valid blocks (sentinel tail) still match the golden."""
    d, H, S, topk, nblk = 128, 16, 12, 16, 32
    T = nblk * BLK_KV
    q, k, v, indices = make_msa_inputs(H, 1, S, T, topk, d, causal=False, seed=5)
    indices[0, 0, 0, 3:] = SENTINEL  # 3 valid blocks
    indices[0, 0, 1, 2:] = SENTINEL  # 2 valid blocks
    gold = sparse_attention_ref_msa(q, k, v, indices, d**-0.5)
    out = run_op_msa_native(q, k, v, indices, device)
    p = pcc(out, gold)
    assert p > 0.99, f"native sentinel-tail PCC {p}"


@run_for_blackhole()
@pytest.mark.parametrize("H", [16, 32])
@pytest.mark.parametrize("topk,nblk", [(16, 16), (16, 32)])
def test_msa_native_pcc_bfp8_cache(device, H, topk, nblk):
    """K/V bfp8 cache should stay within PCC tolerance against the bf16 reference."""
    d, S = 128, 64
    T = nblk * BLK_KV
    for causal in (False, True):
        q, k, v, indices = make_msa_inputs(H, 1, S, T, topk, d, causal=causal, seed=7)
        gold = sparse_attention_ref_msa(q, k, v, indices, d**-0.5)
        out = run_op_msa_native(q, k, v, indices, device, kv_dtype=ttnn.bfloat8_b)
        p = pcc(out, gold)
        assert p > 0.99, f"native bfp8 cache PCC {p} (causal={causal})"


# Q dtype coverage. Python tests bf16 and fp8 q; fp16 is not exposed as a host dtype here.


@run_for_blackhole()
@pytest.mark.parametrize("q_dtype", [ttnn.bfloat16, ttnn.fp8_e4m3])
def test_msa_native_q_dtype(device, q_dtype):
    """Covers bf16 and fp8 q; fp8 uses a lower PCC threshold for quantization loss."""
    d, H, S, topk, nblk = 128, 32, 33, 16, 16
    T = nblk * BLK_KV
    q, k, v, indices = make_msa_inputs(H, 1, S, T, topk, d, causal=False, seed=11)
    gold = sparse_attention_ref_msa(q, k, v, indices, d**-0.5)
    out = run_op_msa_native(q, k, v, indices, device, kv_dtype=ttnn.bfloat8_b, q_dtype=q_dtype)
    p = pcc(out, gold)
    thresh = 0.99 if q_dtype == ttnn.bfloat16 else 0.985
    assert p > thresh, f"q_dtype={q_dtype} PCC {p} (threshold {thresh})"


@run_for_blackhole()
def test_msa_native_output_dtype_matches_q_and_distinct_programs(device):
    """Output dtype follows q, and q dtype is part of the program hash."""
    d, H, S, topk, nblk = 128, 32, 16, 16, 16
    T = nblk * BLK_KV
    q, k, v, indices = make_msa_inputs(H, 1, S, T, topk, d, causal=False, seed=12)
    scale = d**-0.5

    def call(q_dtype):
        mc = ttnn.DRAM_MEMORY_CONFIG
        tt_q = ttnn.from_torch(
            q.to(torch.float32), dtype=q_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc
        )
        tt_k = ttnn.from_torch(
            k.to(torch.bfloat16), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
        )
        tt_v = ttnn.from_torch(
            v.to(torch.bfloat16), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
        )
        tt_i = ttnn.from_torch(
            indices.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc
        )
        return ttnn.transformer.sparse_sdpa_msa(tt_q, tt_k, tt_v, tt_i, scale=scale).dtype

    device.clear_program_cache()
    assert call(ttnn.bfloat16) == ttnn.bfloat16, "bf16 q must yield bf16 output"
    n1 = device.num_program_cache_entries()
    assert call(ttnn.fp8_e4m3) == ttnn.fp8_e4m3, "fp8 q must yield fp8 output"
    n2 = device.num_program_cache_entries()
    assert n2 > n1, "bf16-q and fp8-q must build distinct programs"


@run_for_blackhole()
def test_msa_native_v_dim_distinct_programs_and_correct(device):
    """Different v_dim values must build distinct programs and match the reference."""
    d, H, S, topk, nblk = 128, 32, 16, 16, 16
    T = nblk * BLK_KV
    scale = d**-0.5
    gen = torch.Generator().manual_seed(13)
    q = torch.randn(1, H, S, d, generator=gen)
    k = torch.randn(1, 1, T, d, generator=gen)
    indices = torch.full((1, 1, S, topk), SENTINEL, dtype=torch.int32)
    for s in range(S):
        indices[0, 0, s] = torch.randperm(nblk, generator=gen)[:topk].sort().values.to(torch.int32)
    mc = ttnn.DRAM_MEMORY_CONFIG

    def call(v_dim):
        v = torch.randn(1, 1, T, v_dim, generator=gen)
        gold = sparse_attention_ref_msa(q, k, v, indices, scale)  # [1, H, S, v_dim]
        tt_q = ttnn.from_torch(
            q.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc
        )
        tt_k = ttnn.from_torch(
            k.to(torch.bfloat16), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
        )
        tt_v = ttnn.from_torch(
            v.to(torch.bfloat16), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
        )
        tt_i = ttnn.from_torch(
            indices.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc
        )
        out = ttnn.to_torch(ttnn.transformer.sparse_sdpa_msa(tt_q, tt_k, tt_v, tt_i, scale=scale))
        return out, gold

    device.clear_program_cache()
    o1, g1 = call(128)
    n1 = device.num_program_cache_entries()
    assert pcc(o1, g1) > 0.99, f"v_dim=128 PCC {pcc(o1, g1)}"
    o2, g2 = call(256)  # same q/k/indices/dtypes, only v_dim differs
    n2 = device.num_program_cache_entries()
    assert pcc(o2, g2) > 0.99, f"v_dim=256 PCC {pcc(o2, g2)}"
    assert n2 > n1, "different v_dim must build distinct programs"


# Composition baseline: map MSA onto DSA sparse_sdpa for n_kv=1.


@run_for_blackhole()
@pytest.mark.parametrize("H", [16, 32])
@pytest.mark.parametrize("topk,nblk", [(4, 8), (8, 8)])
def test_msa_composed_pcc(device, H, topk, nblk):
    d, S = 128, 33
    T = nblk * BLK_KV
    q, k, v, indices = make_msa_inputs(H, 1, S, T, topk, d, causal=False, seed=1)
    gold = sparse_attention_ref_msa(q, k, v, indices, d**-0.5)
    out = run_op_msa_composed(q, k, v, indices, device)
    p = pcc(out, gold)
    assert p > 0.99, f"composed PCC {p}"


# Tensor-feature coverage: runtime T, persistent/indexed/sharded K/V, cache-hit validation, and hash keys.

_D = 128  # head dim (== v_dim here)


def _rm(t, device, dtype):
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _tile(t, device, dtype=ttnn.bfloat16, mem=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t.to(torch.bfloat16), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)


def _nd_sharded_dram_config(device, rows_per_shard, width):
    """One DRAM shard per bank, round-robin; each [1,1,rows_per_shard,width] slot is one shard."""
    num_banks = device.dram_grid_size().x
    cores = [ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(num_banks)]
    spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, rows_per_shard, width],
        grid=ttnn.CoreRangeSet(cores),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    return ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=spec)


def _msa_indexed_inputs(H, S, T, topk, B, *, seed=0):
    """q [1,H,S,d]; k_full/v_full [B,1,T,d] (B distinct slots); block-id indices [1,1,S,topk] (-1 tail)."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(1, H, S, _D, generator=gen)
    k_full = torch.randn(B, 1, T, _D, generator=gen)
    v_full = torch.randn(B, 1, T, _D, generator=gen)
    nblk = T // BLK_KV
    indices = torch.full((1, 1, S, topk), SENTINEL, dtype=torch.int32)
    for s in range(S):
        chosen = torch.randperm(nblk, generator=gen)[:topk].sort().values
        indices[0, 0, s, : chosen.numel()] = chosen.to(torch.int32)
    return q, k_full, v_full, indices


def _msa_op(device, q, k_tt, v_tt, indices, **kw):
    out = ttnn.transformer.sparse_sdpa_msa(
        _rm(q.to(torch.bfloat16), device, ttnn.bfloat16),
        k_tt,
        v_tt,
        _rm(indices.to(torch.int32), device, ttnn.uint32),
        scale=_D**-0.5,
        block_size=BLK_KV,
        **kw,
    )
    return ttnn.to_torch(out)


# Interleaved K/V length is a runtime shape and should not recompile.
@run_for_blackhole()
def test_msa_kv_len_no_recompile(device):
    H, S, topk = 32, 8, 16
    device.clear_program_cache()
    for T in (
        256,
        512,
        1024,
    ):  # only K/V length differs; sentinel tail keeps topk fixed
        q, k_full, v_full, indices = _msa_indexed_inputs(H, S, T, topk, 1, seed=T)
        out = _msa_op(device, q, _tile(k_full, device), _tile(v_full, device), indices)
        gold = sparse_attention_ref_msa(q, k_full, v_full, indices, _D**-0.5)
        assert pcc(out, gold) > 0.99, f"T={T}"
    n = device.num_program_cache_entries()
    assert n == 1, f"changing K/V length recompiled: {n} entries (expected 1)"


@run_for_blackhole()
def test_msa_gqa_kv_len_no_recompile(device):
    """GQA K/V group strides depend on runtime T and must be patched on cache hits."""
    H, n_kv, S, topk = 64, 4, 8, 16
    device.clear_program_cache()
    for T in (2048, 4096):
        q, k_full, v_full, indices = make_msa_inputs(H, n_kv, S, T, topk, _D, causal=False, seed=T)
        out = _msa_op(device, q, _tile(k_full, device), _tile(v_full, device), indices)
        gold = sparse_attention_ref_msa(q, k_full, v_full, indices, _D**-0.5)
        assert pcc(out, gold) > 0.99, f"GQA T={T}"
    n = device.num_program_cache_entries()
    assert n == 1, f"changing GQA K/V length recompiled: {n} entries (expected 1)"


# Oversized persistent K/V cache: queries attend only a populated prefix.
@run_for_blackhole()
def test_msa_oversized_persistent_kv(device):
    H, S, T_MAX, topk = 32, 8, 1024, 16
    device.clear_program_cache()
    gen = torch.Generator().manual_seed(0)
    k_full = torch.randn(1, 1, T_MAX, _D, generator=gen)
    v_full = torch.randn(1, 1, T_MAX, _D, generator=gen)
    k_tt, v_tt = _tile(k_full, device), _tile(v_full, device)  # persistent oversized cache
    for valid_blocks in (2, 8):  # queries attend only the [0, valid_blocks) populated prefix
        q = torch.randn(1, H, S, _D, generator=gen)
        indices = torch.full((1, 1, S, topk), SENTINEL, dtype=torch.int32)
        for s in range(S):
            chosen = torch.randperm(valid_blocks, generator=gen)[:topk].sort().values
            indices[0, 0, s, : chosen.numel()] = chosen.to(torch.int32)  # rest stays sentinel
        gold = sparse_attention_ref_msa(q, k_full, v_full, indices, _D**-0.5)
        out = _msa_op(device, q, k_tt, v_tt, indices)
        assert pcc(out, gold) > 0.99, f"valid_blocks={valid_blocks}"
    n = device.num_program_cache_entries()
    assert n == 1, f"oversized reuse recompiled: {n} entries (expected 1)"


# Indexed KV cache: cache_batch_idx selects one slot from a shared [B,1,T,d] cache.
@run_for_blackhole()
def test_msa_indexed_kv_cache(device):
    H, S, T, topk, B = 32, 8, 256, 16, 3
    device.clear_program_cache()
    q, k_full, v_full, indices = _msa_indexed_inputs(H, S, T, topk, B)
    k_tt, v_tt = _tile(k_full, device), _tile(v_full, device)  # [B,1,T,d] shared cache
    for cb in range(B):
        out = _msa_op(device, q, k_tt, v_tt, indices, cache_batch_idx=cb)
        gold = sparse_attention_ref_msa(q, k_full[cb : cb + 1], v_full[cb : cb + 1], indices, _D**-0.5)
        assert pcc(out, gold) > 0.99, f"cache_batch_idx={cb}"
    n = device.num_program_cache_entries()
    assert n == 1, f"indexing a different slot recompiled: {n} entries (expected 1)"


@run_for_blackhole()
def test_msa_indexed_kv_cache_first_call_nonzero_slot(device):
    """The cache-miss build must bake the requested cache_batch_idx, not default to slot 0."""
    H, S, T, topk, B = 32, 8, 256, 16, 3
    device.clear_program_cache()
    q, k_full, v_full, indices = _msa_indexed_inputs(H, S, T, topk, B, seed=31)
    k_tt, v_tt = _tile(k_full, device), _tile(v_full, device)
    out = _msa_op(device, q, k_tt, v_tt, indices, cache_batch_idx=1)
    gold = sparse_attention_ref_msa(q, k_full[1:2], v_full[1:2], indices, _D**-0.5)
    assert pcc(out, gold) > 0.99, "first indexed call with cache_batch_idx=1 read the wrong slot"
    assert device.num_program_cache_entries() == 1


# Indexed KV cache, ND-sharded across DRAM banks.
@run_for_blackhole()
def test_msa_indexed_nd_sharded_kv(device):
    H, S, T, topk, B = 32, 8, 256, 16, 2
    device.clear_program_cache()
    q, k_full, v_full, indices = _msa_indexed_inputs(H, S, T, topk, B)
    nd_cfg = _nd_sharded_dram_config(device, rows_per_shard=T, width=_D)  # each [1,1,T,d] slot is one shard
    k_tt = _tile(k_full, device, mem=nd_cfg)
    v_tt = _tile(v_full, device, mem=nd_cfg)
    for cb in range(B):
        out = _msa_op(device, q, k_tt, v_tt, indices, cache_batch_idx=cb)
        gold = sparse_attention_ref_msa(q, k_full[cb : cb + 1], v_full[cb : cb + 1], indices, _D**-0.5)
        assert pcc(out, gold) > 0.99, f"nd-sharded cache_batch_idx={cb}"


# Changing T changes sharded K/V layout, so it must recompile.
@run_for_blackhole()
def test_msa_nd_sharded_kv_len_change(device):
    H, S, topk = 32, 8, 16
    device.clear_program_cache()
    nd_cfg = _nd_sharded_dram_config(device, rows_per_shard=128, width=_D)  # fixed shard shape, independent of T
    for T in (256, 512):
        q, k_full, v_full, indices = _msa_indexed_inputs(H, S, T, topk, 1, seed=T)
        out = _msa_op(device, q, _tile(k_full, device, mem=nd_cfg), _tile(v_full, device, mem=nd_cfg), indices)
        gold = sparse_attention_ref_msa(q, k_full, v_full, indices, _D**-0.5)
        assert pcc(out, gold) > 0.99, f"nd-sharded T={T}"
    assert device.num_program_cache_entries() == 2, "sharded K/V must recompile per shape (2 T -> 2 entries)"


# cache_batch_idx is hash-excluded, so out-of-range slots must be rejected on cache hits.
@run_for_blackhole()
def test_msa_indexed_oob_rejected_on_hit(device, expect_error):
    H, S, T, topk, B = 32, 8, 256, 16, 2
    device.clear_program_cache()
    q, k_full, v_full, indices = _msa_indexed_inputs(H, S, T, topk, B)
    k_tt, v_tt = _tile(k_full, device), _tile(v_full, device)
    _msa_op(device, q, k_tt, v_tt, indices, cache_batch_idx=0)  # miss: builds
    assert device.num_program_cache_entries() == 1
    with expect_error(RuntimeError, "cache_batch_idx"):  # cache hit with out-of-range slot
        _msa_op(device, q, k_tt, v_tt, indices, cache_batch_idx=B)


@run_for_blackhole()
def test_msa_bad_runtime_t_rejected_on_hit(device, expect_error):
    """Interleaved K/V T is hash-excluded, so a cache-hit T that violates block_size must still be rejected."""
    H, S, topk = 32, 8, 16
    device.clear_program_cache()
    q, k_full, v_full, indices = _msa_indexed_inputs(H, S, 256, topk, 1, seed=37)
    _msa_op(device, q, _tile(k_full, device), _tile(v_full, device), indices)
    assert device.num_program_cache_entries() == 1

    bad_T = 320  # tile-aligned, but not divisible by BLK_KV=128
    q_bad, k_bad, v_bad, indices_bad = _msa_indexed_inputs(H, S, bad_T, topk, 1, seed=bad_T)
    with expect_error(RuntimeError, "block_size must divide T"):
        _msa_op(device, q_bad, _tile(k_bad, device), _tile(v_bad, device), indices_bad)


# Program hash covers K/V dtype.
@run_for_blackhole()
def test_msa_hash_distinct_kv_dtype(device):
    H, S, T, topk = 32, 8, 256, 16
    device.clear_program_cache()
    q, k_full, v_full, indices = _msa_indexed_inputs(H, S, T, topk, 1)
    gold = sparse_attention_ref_msa(q, k_full, v_full, indices, _D**-0.5)
    for dt in (ttnn.bfloat16, ttnn.bfloat8_b):
        out = _msa_op(device, q, _tile(k_full, device, dt), _tile(v_full, device, dt), indices)
        assert pcc(out, gold) > 0.99, f"dtype={dt}"
    assert device.num_program_cache_entries() == 2, "bf16 vs bfp8_b K/V must be distinct programs (2 entries)"


# Determinism: identical inputs should produce bit-exact outputs across cached runs.
@run_for_blackhole()
@pytest.mark.parametrize(
    "q_dtype,kv_dtype",
    [(ttnn.bfloat16, ttnn.bfloat16), (ttnn.fp8_e4m3, ttnn.bfloat8_b)],
    ids=["bf16", "fp8"],
)
def test_msa_native_determinism(device, q_dtype, kv_dtype):
    d, H, S, nblk, topk, iters = 128, 32, 128, 16, 16, 10  # multi-chunk; some cores process two tokens
    T = nblk * BLK_KV
    q, k, v, indices = make_msa_inputs(H, 1, S, T, topk, d, causal=False, seed=11)
    # Upload once; rerun the cached program on the same input bits.
    mc = ttnn.DRAM_MEMORY_CONFIG
    tt_q = ttnn.from_torch(
        q.to(torch.float32), dtype=q_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc
    )
    tt_k = ttnn.from_torch(
        k.to(torch.bfloat16), dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
    )
    tt_v = ttnn.from_torch(
        v.to(torch.bfloat16), dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
    )
    tt_idx = ttnn.from_torch(
        indices.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc
    )

    def comparable(o):  # eltwise comparison uses TILE bf16
        if o.dtype == ttnn.fp8_e4m3:
            o = ttnn.typecast(o, ttnn.bfloat16)
        return ttnn.to_layout(o, ttnn.TILE_LAYOUT)

    ref, marker = None, None
    for _ in range(iters):
        cur = comparable(ttnn.transformer.sparse_sdpa_msa(tt_q, tt_k, tt_v, tt_idx, scale=d**-0.5, block_size=BLK_KV))
        if ref is None:
            ref = cur
        else:
            m = ttnn.max(ttnn.ne(ref, cur, dtype=ttnn.bfloat16))  # 0 iff bit-exact
            marker = m if marker is None else ttnn.maximum(marker, m)
    mismatch = float(ttnn.to_torch(ttnn.from_device(marker)).max())  # one scalar to host, once
    assert mismatch == 0.0, "sparse_sdpa_msa output is not deterministic across repeated runs"


# Causal chunk_start_idx is hash-excluded (only causal on/off is hashed), so it must be re-applied on cache
# hits. cs=BLK_KV shifts every query one block forward, changing which selected blocks are masked; each
# dispatch must match its own reference (a frozen offset fails the second) while reusing one cached program.
@run_for_blackhole()
def test_msa_native_causal_chunk_start_no_recompile(device):
    d, H, n_kv, S, nblk = _D, 64, 4, 320, 16
    T = nblk * BLK_KV
    device.clear_program_cache()
    q, k, v, indices = make_msa_inputs(H, n_kv, S, T, topk=nblk, d=d, causal=True, seed=31)
    for cs in (0, BLK_KV):
        gold = sparse_attention_ref_msa(q, k, v, indices, d**-0.5, causal=True, chunk_start_idx=cs)
        out = run_op_msa_native(q, k, v, indices, device, chunk_start_idx=cs)
        assert pcc(out, gold) > 0.99, f"chunk_start_idx={cs} pcc={pcc(out, gold)}"
    n = device.num_program_cache_entries()
    assert n == 1, f"changing chunk_start_idx recompiled: {n} entries (expected 1)"


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((8, 4), {"fabric_config": ttnn.FabricConfig.FABRIC_1D})],
    indirect=True,
)
@run_for_blackhole()
def test_msa_native_causal_per_rank_sp(mesh_device):
    """Multi-device (8x4): q/indices are sharded on the sequence axis across the SP (row) axis and replicated
    across the TP (col) axis, so SP rank r owns global query positions [r*s_local, (r+1)*s_local). The op must
    derive its causal chunk_start from each device's mesh coordinate (device_index = rank along cluster_axis 0);
    otherwise rank>0 masks against a local position, its diagonal block isn't in the selected set, and it
    attends future tokens. K/V are the full context, replicated. Gathering col 0 of each SP row must match one
    global-causal reference. Needs 32 devices (skips otherwise)."""
    rows, cols = tuple(mesh_device.shape)  # SP rows (cluster_axis 0) x TP cols
    d, H, n_kv, nblk = 128, 16, 1, 16
    s_local = 128
    S = rows * s_local
    T = nblk * BLK_KV
    scale = d**-0.5
    q, k, v, indices = make_msa_inputs(H, n_kv, S, T, topk=nblk, d=d, causal=True, seed=31)
    gold = sparse_attention_ref_msa(q, k, v, indices, scale, causal=True)

    def shard_seq_rm(t, dt):  # row-major q/indices: seq sharded on the SP (row) axis, replicated on TP (col)
        return ttnn.from_torch(
            t,
            dtype=dt,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(2, None), mesh_shape=(rows, cols)),
        )

    def repl_tile(t, dt):  # pre-tiled K/V full context, replicated to every device
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    tt_out = ttnn.transformer.sparse_sdpa_msa(
        shard_seq_rm(q.to(torch.float32), ttnn.bfloat16),
        repl_tile(k, ttnn.bfloat16),
        repl_tile(v, ttnn.bfloat16),
        shard_seq_rm(indices.to(torch.int32), ttnn.uint32),
        scale=scale,
        block_size=BLK_KV,
        chunk_start_idx=0,  # base 0; each device adds rank*s_local from its coordinate along cluster_axis 0
        cluster_axis=0,
    )
    # seq is sharded across SP rows and replicated across TP cols -> take col 0 of each row, concat on seq.
    dts = ttnn.get_device_tensors(tt_out)
    out = torch.cat([ttnn.to_torch(dts[r * cols]).float()[:, :H] for r in range(rows)], dim=2)
    p = pcc(out, gold)
    assert p > 0.99, f"per-rank SP causal PCC {p}"
