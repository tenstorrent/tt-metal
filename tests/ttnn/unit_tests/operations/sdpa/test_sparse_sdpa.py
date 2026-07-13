# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""sparse_sdpa (sparse MLA prefill) — BASIC post-commit smoke (Blackhole only).

This file is a fast subset: op-runs/shape, output-dtype (incl. fp8), and a minimal single- and multi-chunk
PCC with all-valid + boundary masking. The full coverage (sweeps over shapes / heads / fp8 q+kv /
query-subblocking / sparsity, plus the perf-only test) lives in the nightly suite:
tests/ttnn/nightly/unit_tests/operations/sdpa/test_sparse_sdpa.py
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.sdpa.sparse_sdpa_test_utils import (
    make_inputs,
    golden,
    to_dev,
    run_op,
    pcc,
)

K_DIM = 576  # head dim (q/kv width)
V_DIM = 512  # V width / output width (op arg)
SCALE_BLOCK_WIDTH = 128

# Open the device ONCE per module (not per test) — all tests share one device config, so this avoids the
# ~1.7s open/close on every test. The program cache persists across tests, which is fine: the recompile and
# indexed tests clear_program_cache() at their start, and the hash keeps distinct configs separate.
pytestmark = pytest.mark.use_module_device


def make_scaled_kv_cache(device, batch, T, seed, round_scale):
    """Build a packed cache and its reconstructed BF16 reference."""
    gen = torch.Generator().manual_seed(seed)
    latent = torch.randn(batch, 1, T, V_DIM, generator=gen, dtype=torch.float32)
    block_scales = torch.logspace(-3, 0, steps=V_DIM // SCALE_BLOCK_WIDTH)
    latent *= block_scales.repeat_interleave(SCALE_BLOCK_WIDTH)
    rope = torch.randn(batch, 1, T, K_DIM - V_DIM, generator=gen, dtype=torch.float32).to(torch.bfloat16)

    tt_latent = to_dev(latent.to(torch.bfloat16), device, ttnn.bfloat16)
    tt_fp8, tt_scales = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(
        tt_latent, round_scale_to_power_of_two=round_scale
    )
    tt_packed = ttnn.experimental.deepseek_prefill.pack_scaled_fp8_kv_cache(
        tt_fp8, tt_scales, to_dev(rope, device, ttnn.bfloat16)
    )
    tt_reconstructed = ttnn.experimental.deepseek_prefill.per_token_cast_back(
        tt_fp8, tt_scales, output_dtype=ttnn.bfloat16
    )
    reconstructed = torch.cat([ttnn.to_torch(tt_reconstructed).float(), rope.float()], dim=-1)
    return tt_packed, reconstructed


# ---- op runs, output shape/layout correct ----
@run_for_blackhole()
def test_sparse_sdpa_runs(device):
    H, S, T, TOPK, kc = 32, 64, 256, 64, 32
    q, kv, indices = make_inputs(H, S, T, TOPK, K_DIM, lambda s: TOPK)
    out, _ = run_op(q, kv, indices, device, kc, V_DIM)
    assert tuple(out.shape) == (1, H, S, V_DIM), f"got {tuple(out.shape)}"


@run_for_blackhole()
@pytest.mark.parametrize(
    "S,T,TOPK,kc,all_valid,round_scale",
    [
        (32, 128, 64, 32, False, True),
        (32, 128, 64, 64, False, True),
        (32, 128, 64, 32, True, True),
        (4, 512, 256, 256, True, True),
        (4, 128, 64, 32, True, False),
        (4, 256, 128, 128, False, False),
    ],
    ids=[
        "multi_chunk",
        "single_chunk",
        "gather_ring_wrap",
        "two_slab_pipeline_wrap",
        "arbitrary_scale",
        "scaled_e4m3_latent_bf16_rope",
    ],
)
def test_sparse_sdpa_scaled_fp8_kv(device, S, T, TOPK, kc, all_valid, round_scale):
    """Selected FP8 latent rows use per-token/block scales while RoPE remains BF16."""
    H = 32
    q, _, indices = make_inputs(
        H, S, T, TOPK, K_DIM, (lambda s: TOPK) if all_valid else (lambda s: 1 + (s * 7) % TOPK), seed=41
    )
    tt_packed, reconstructed = make_scaled_kv_cache(device, 1, T, seed=42, round_scale=round_scale)
    tt_q = to_dev(q.to(torch.bfloat16), device, ttnn.bfloat16)
    tt_idx = to_dev(indices.to(torch.int32), device, ttnn.uint32)

    block_cyclic_args = {"block_cyclic_sp_axis": 0, "block_cyclic_chunk_local": S} if kc == TOPK and TOPK == 64 else {}
    tt_out = ttnn.transformer.sparse_sdpa(
        tt_q,
        tt_packed,
        tt_idx,
        V_DIM,
        scale=K_DIM**-0.5,
        k_chunk_size=kc,
        **block_cyclic_args,
    )

    expected = golden(q, reconstructed, indices, K_DIM**-0.5, V_DIM)
    score = pcc(ttnn.to_torch(tt_out), expected)
    assert score >= 0.999, f"scaled FP8 sparse SDPA PCC {score:.5f}"


@run_for_blackhole()
def test_sparse_sdpa_bf16_multi_query_tile(device):
    """Two query tiles must pack into distinct score rows when qsb is two."""
    H, S, T, TOPK, kc = 64, 64, 128, 64, 32
    q, kv, indices = make_inputs(H, S, T, TOPK, K_DIM, lambda s: TOPK, seed=60)
    out, scale = run_op(q, kv, indices, device, k_chunk_size=kc, v_dim=V_DIM)
    expected = golden(q, kv, indices, scale, V_DIM)
    score = pcc(out, expected)
    tile_scores = [pcc(out[:, start : start + 32], expected[:, start : start + 32]) for start in (0, 32)]
    assert score >= 0.999, f"BF16 H=64 sparse SDPA PCC {score:.5f}; head-tile PCCs {tile_scores}"


@run_for_blackhole()
def test_sparse_sdpa_scaled_fp8_kv_multi_query_tile(device):
    """GLM geometry: two 32-head query tiles share each gathered packed-KV chunk."""
    H, S, T, TOPK, kc = 64, 64, 128, 64, 32
    q, _, indices = make_inputs(H, S, T, TOPK, K_DIM, lambda s: TOPK, seed=61)
    tt_packed, reconstructed = make_scaled_kv_cache(device, 1, T, seed=62, round_scale=True)
    out = ttnn.transformer.sparse_sdpa(
        to_dev(q.to(torch.bfloat16), device, ttnn.bfloat16),
        tt_packed,
        to_dev(indices.to(torch.int32), device, ttnn.uint32),
        V_DIM,
        scale=K_DIM**-0.5,
        k_chunk_size=kc,
    )

    expected = golden(q, reconstructed, indices, K_DIM**-0.5, V_DIM)
    actual = ttnn.to_torch(out)
    score = pcc(actual, expected)
    tile_scores = [pcc(actual[:, start : start + 32], expected[:, start : start + 32]) for start in (0, 32)]
    assert score >= 0.999, f"scaled FP8 H=64 sparse SDPA PCC {score:.5f}; head-tile PCCs {tile_scores}"


@run_for_blackhole()
def test_sparse_sdpa_scaled_fp8_rejects_malformed_packed_width(device, expect_error):
    H, S, T, TOPK = 32, 32, 128, 32
    q, kv, indices = make_inputs(H, S, T, TOPK, K_DIM, lambda s: TOPK)
    malformed = to_dev(torch.randn(1, 1, T, 600), device, ttnn.fp8_e4m3)
    with expect_error(RuntimeError, "kv must be"):
        ttnn.transformer.sparse_sdpa(
            to_dev(q.to(torch.bfloat16), device, ttnn.bfloat16),
            malformed,
            to_dev(indices.to(torch.int32), device, ttnn.uint32),
            V_DIM,
        )


@run_for_blackhole()
def test_sparse_sdpa_rejects_v_dim_larger_than_k_dim(device, expect_error):
    H, S, T, TOPK = 32, 32, 128, 32
    q, kv, indices = make_inputs(H, S, T, TOPK, K_DIM, lambda s: TOPK)
    with expect_error(RuntimeError, "v_dim must be in"):
        ttnn.transformer.sparse_sdpa(
            to_dev(q.to(torch.bfloat16), device, ttnn.bfloat16),
            to_dev(kv.to(torch.bfloat16), device, ttnn.bfloat16),
            to_dev(indices.to(torch.int32), device, ttnn.uint32),
            K_DIM + 32,
        )


# ---- output dtype matches q (bf16 q -> bf16 out, fp8 q -> fp8 out) ----
@run_for_blackhole()
@pytest.mark.parametrize("q_dtype", [ttnn.bfloat16, ttnn.fp8_e4m3], ids=["q_bf16", "q_fp8"])
def test_sparse_sdpa_output_dtype(device, q_dtype):
    H, S, T, TOPK, kc = 32, 64, 256, 64, 32
    q, kv, indices = make_inputs(H, S, T, TOPK, K_DIM, lambda s: TOPK)
    q_host = q.to(torch.bfloat16) if q_dtype == ttnn.bfloat16 else q.to(torch.float32)
    tt_q = to_dev(q_host, device, q_dtype)
    tt_kv = to_dev(kv.to(torch.bfloat16), device, ttnn.bfloat16)
    tt_idx = to_dev(indices.to(torch.int32), device, ttnn.uint32)
    out = ttnn.transformer.sparse_sdpa(tt_q, tt_kv, tt_idx, V_DIM, scale=K_DIM**-0.5, k_chunk_size=kc)
    assert out.dtype == q_dtype, f"output dtype {out.dtype} != q dtype {q_dtype}"


# ---- minimal PCC vs the sparse_mla golden: single chunk + multi chunk, all-valid + a boundary mask ----
@run_for_blackhole()
@pytest.mark.parametrize(
    "nv_fn,nv_id",
    [(lambda s: 10**9, "all_valid"), (lambda s: 1 + (s * 3) % 20, "boundary")],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_sparse_sdpa_pcc_single_chunk(device, nv_fn, nv_id):
    H, S, T, TOPK = 32, 64, 256, 32  # k_chunk == TOPK -> single chunk
    q, kv, indices = make_inputs(H, S, T, TOPK, K_DIM, nv_fn)
    out, scale = run_op(q, kv, indices, device, k_chunk_size=TOPK, v_dim=V_DIM)
    p = pcc(out, golden(q, kv, indices, scale, V_DIM))
    assert p >= 0.99, f"PCC {p:.5f} ({nv_id})"


@run_for_blackhole()
@pytest.mark.parametrize(
    "nv_fn,nv_id",
    [(lambda s: 10**9, "all_valid"), (lambda s: 1 + (s * 3) % 50, "mid_tile_boundary")],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_sparse_sdpa_pcc_multi_chunk(device, nv_fn, nv_id):
    H, S, T, TOPK, kc = 32, 64, 256, 64, 32  # 2 chunks -> flash accumulation
    q, kv, indices = make_inputs(H, S, T, TOPK, K_DIM, nv_fn)
    out, scale = run_op(q, kv, indices, device, kc, V_DIM)
    p = pcc(out, golden(q, kv, indices, scale, V_DIM))
    assert p >= 0.99, f"PCC {p:.5f} ({nv_id})"


# ---- the kv (K cache) length T rides on the accessor's runtime args, so changing T must NOT recompile ----
@run_for_blackhole()
def test_sparse_sdpa_kv_len_no_recompile(device):
    H, S, TOPK, kc = 32, 64, 64, 32
    device.clear_program_cache()
    for T in (256, 512, 1024):  # only the kv (K cache) length differs across these calls
        q, kv, indices = make_inputs(H, S, T, TOPK, K_DIM, lambda s: TOPK)
        out, scale = run_op(q, kv, indices, device, kc, V_DIM)
        p = pcc(out, golden(q, kv, indices, scale, V_DIM))
        assert p >= 0.99, f"PCC {p:.5f} (T={T})"
    n = device.num_program_cache_entries()
    assert n == 1, f"changing kv length recompiled: {n} program-cache entries (expected 1)"


# ---- oversized persistent kv buffer (same idea as ring_mla's reuse_max): one max-size K cache, allocated
# ---- once and reused across calls. The cache is far larger than the keys any query attends to (T >> TOPK)
# ---- and reads are index-driven, so the unpopulated suffix is simply never addressed. ----
@run_for_blackhole()
def test_sparse_sdpa_oversized_persistent_kv(device):
    H, S, T_MAX, TOPK, kc = 32, 64, 1024, 64, 32
    device.clear_program_cache()
    gen = torch.Generator().manual_seed(0)
    kv = torch.randn(1, 1, T_MAX, K_DIM, generator=gen, dtype=torch.float32)
    tt_kv = to_dev(kv.to(torch.bfloat16), device, ttnn.bfloat16)  # ONE persistent oversized K cache
    scale = K_DIM**-0.5
    for valid_len in (256, 1024):  # each call's queries attend only to the [0, valid_len) populated prefix
        q = torch.randn(1, H, S, K_DIM, generator=gen, dtype=torch.float32)
        indices = torch.empty((1, 1, S, TOPK), dtype=torch.int64)
        for s in range(S):
            indices[0, 0, s, :] = torch.randperm(valid_len, generator=gen)[:TOPK]
        tt_q = to_dev(q.to(torch.bfloat16), device, ttnn.bfloat16)
        tt_idx = to_dev(indices.to(torch.int32), device, ttnn.uint32)
        tt_out = ttnn.transformer.sparse_sdpa(tt_q, tt_kv, tt_idx, V_DIM, scale=scale, k_chunk_size=kc)
        p = pcc(ttnn.to_torch(tt_out), golden(q, kv, indices, scale, V_DIM))
        assert p >= 0.99, f"PCC {p:.5f} (valid_len={valid_len})"
    n = device.num_program_cache_entries()  # reusing the same oversized buffer must not realloc/recompile
    assert n == 1, f"oversized persistent kv reuse recompiled: {n} program-cache entries (expected 1)"


# ---- block-cyclic remap: indices are NATURAL token positions but the kv cache is stored block-cyclic. `sp` is
# ---- DERIVED from the mesh (block_cyclic_sp_axis = the mesh axis the cache was striped over). On ONE device the
# ---- mesh is 1x1 so sp=1 and the remap reduces to identity (shard=0 and BC_SLAB_STRIDE_GAP=0 => page=n) — enough
# ---- to smoke the whole BC path here (API, the chunk_local cross-check, the BC_ENABLE kernel branch). All remap
# ---- constants incl. BC_SHARD_STRIDE_GAP (= T/sp - chunk_local) are compile-time, with cache length T folded into the program hash, so
# ---- each DISTINCT cache size T is its own program (the cache is expected to be a consistent-size prealloc).
# ---- The sp>1 PERMUTATION arithmetic needs a real SP mesh; that multi-device coverage lives in
# ---- tests/nightly/blackhole/sdpa/test_sparse_sdpa_multidevice.py (run on QuietBox-2 in BH post-commit — a
# ---- mesh_device fixture can't share this single-device use_module_device file). ----
@run_for_blackhole()
def test_sparse_sdpa_block_cyclic_sp1_identity(device):
    """sp=1 (read from the 1x1 device-mesh): block-cyclic == natural, so the op must reproduce the natural golden
    while exercising the BC_ENABLE path, across cache sizes T. Since T is hashed for this path (BC_SHARD_STRIDE_GAP
    is a compile-time define), each distinct T is a DISTINCT program — asserted below."""
    H, S, TOPK, kc = 32, 64, 64, 32
    Ts = (256, 512, 1024)
    device.clear_program_cache()
    for T in Ts:
        q, kv, indices = make_inputs(H, S, T, TOPK, K_DIM, lambda s: TOPK, seed=T)
        out = ttnn.transformer.sparse_sdpa(
            to_dev(q.to(torch.bfloat16), device, ttnn.bfloat16),
            to_dev(kv.to(torch.bfloat16), device, ttnn.bfloat16),  # sp=1 => block-cyclic order == natural
            to_dev(indices.to(torch.int32), device, ttnn.uint32),
            V_DIM,
            scale=K_DIM**-0.5,
            k_chunk_size=kc,
            block_cyclic_sp_axis=0,
            block_cyclic_chunk_local=S,  # == q_isl (sp=1, tp=1)
        )
        p = pcc(ttnn.to_torch(out), golden(q, kv, indices, K_DIM**-0.5, V_DIM))
        assert p >= 0.99, f"PCC {p:.5f} (sp=1 identity block-cyclic, T={T})"
    # T is hashed for the block-cyclic path (compile-time BC_SHARD_STRIDE_GAP), so each distinct cache size is its own program.
    n = device.num_program_cache_entries()
    assert n == len(Ts), f"block-cyclic should hash cache size T: got {n} program-cache entries (expected {len(Ts)})"


@run_for_blackhole()
def test_sparse_sdpa_block_cyclic_chunk_local_rejected(device, expect_error):
    """The chunk_local cross-check rejects a value that is neither q_isl nor tp*q_isl — the footgun guard. On a
    single device sp=1/tp=1, so the only legal chunk_local is q_isl (= S); any other value must raise."""
    H, S, T, TOPK, kc = 32, 64, 256, 64, 32
    q, kv_nat, indices = make_inputs(H, S, T, TOPK, K_DIM, lambda s: TOPK, seed=1)
    with expect_error(RuntimeError, "block_cyclic_chunk_local"):
        ttnn.transformer.sparse_sdpa(
            to_dev(q.to(torch.bfloat16), device, ttnn.bfloat16),
            to_dev(kv_nat.to(torch.bfloat16), device, ttnn.bfloat16),
            to_dev(indices.to(torch.int32), device, ttnn.uint32),
            V_DIM,
            scale=K_DIM**-0.5,
            k_chunk_size=kc,
            block_cyclic_sp_axis=0,
            block_cyclic_chunk_local=S - 16,  # != S (=q_isl); rejected before dispatch
        )


def _indexed_inputs(H, S, T, TOPK, B, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(1, H, S, K_DIM, generator=gen, dtype=torch.float32)
    kv_full = torch.randn(B, 1, T, K_DIM, generator=gen, dtype=torch.float32)  # B DISTINCT cache slots
    indices = torch.empty((1, 1, S, TOPK), dtype=torch.int64)
    for s in range(S):
        indices[0, 0, s, :] = torch.randperm(T, generator=gen)[:TOPK]
    return q, kv_full, indices


# ---- indexed KV cache: kv is a shared [B,1,T,K_DIM] cache; cache_batch_idx selects the slot. Distinct
# ---- random data per slot means a correct PCC for every slot proves the right slot is read, and a single
# ---- program-cache entry across slots proves indexing is a runtime arg (no recompile). ----
@run_for_blackhole()
def test_sparse_sdpa_indexed_kv_cache(device):
    H, S, T, TOPK, kc, B = 32, 64, 256, 64, 32, 3
    device.clear_program_cache()
    scale = K_DIM**-0.5
    q, kv_full, indices = _indexed_inputs(H, S, T, TOPK, B)
    tt_q = to_dev(q.to(torch.bfloat16), device, ttnn.bfloat16)
    tt_kv = to_dev(kv_full.to(torch.bfloat16), device, ttnn.bfloat16)  # [B,1,T,K_DIM] shared cache
    tt_idx = to_dev(indices.to(torch.int32), device, ttnn.uint32)
    for cb in range(B):
        tt_out = ttnn.transformer.sparse_sdpa(
            tt_q, tt_kv, tt_idx, V_DIM, scale=scale, k_chunk_size=kc, cache_batch_idx=cb
        )
        p = pcc(ttnn.to_torch(tt_out), golden(q, kv_full[cb : cb + 1], indices, scale, V_DIM))  # golden uses slot cb
        assert p >= 0.99, f"PCC {p:.5f} (cache_batch_idx={cb})"
    n = device.num_program_cache_entries()  # changing the indexed slot must NOT recompile
    assert n == 1, f"indexing into a different slot recompiled: {n} program-cache entries (expected 1)"


# ---- indexed KV cache that is ND-sharded across DRAM banks (each batch slot is one shard). ----
def _nd_sharded_dram_config(device, rows_per_shard, width=K_DIM):
    num_banks = device.dram_grid_size().x
    cores = [ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(num_banks)]
    spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, rows_per_shard, width],
        grid=ttnn.CoreRangeSet(cores),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    return ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=spec)


@run_for_blackhole()
def test_sparse_sdpa_indexed_nd_sharded_kv(device):
    H, S, T, TOPK, kc, B = 32, 64, 256, 64, 32, 2
    device.clear_program_cache()
    scale = K_DIM**-0.5
    q, kv_full, indices = _indexed_inputs(H, S, T, TOPK, B)
    nd_cfg = _nd_sharded_dram_config(device, rows_per_shard=T)  # each [1,1,T,K_DIM] slot is one shard
    tt_q = to_dev(q.to(torch.bfloat16), device, ttnn.bfloat16)
    tt_kv = ttnn.from_torch(
        kv_full.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=nd_cfg,
    )
    tt_idx = to_dev(indices.to(torch.int32), device, ttnn.uint32)
    for cb in range(B):
        tt_out = ttnn.transformer.sparse_sdpa(
            tt_q, tt_kv, tt_idx, V_DIM, scale=scale, k_chunk_size=kc, cache_batch_idx=cb
        )
        p = pcc(ttnn.to_torch(tt_out), golden(q, kv_full[cb : cb + 1], indices, scale, V_DIM))
        assert p >= 0.99, f"PCC {p:.5f} (nd-sharded, cache_batch_idx={cb})"


@run_for_blackhole()
def test_sparse_sdpa_scaled_fp8_indexed_nd_sharded_kv(device):
    """Production cache geometry: one packed ND-sharded cache and a dynamic user/layer slot."""
    H, S, T, TOPK, kc, B = 32, 32, 128, 64, 32, 2
    scale = K_DIM**-0.5
    q, _, indices = make_inputs(H, S, T, TOPK, K_DIM, lambda s: TOPK, seed=51)
    tt_packed, reconstructed = make_scaled_kv_cache(device, B, T, seed=52, round_scale=True)
    tt_packed = ttnn.to_memory_config(tt_packed, _nd_sharded_dram_config(device, T, tt_packed.shape[-1]))
    tt_q = to_dev(q.to(torch.bfloat16), device, ttnn.bfloat16)
    tt_idx = to_dev(indices.to(torch.int32), device, ttnn.uint32)

    for cache_batch_idx in range(B):
        out = ttnn.transformer.sparse_sdpa(
            tt_q,
            tt_packed,
            tt_idx,
            V_DIM,
            scale=scale,
            k_chunk_size=kc,
            cache_batch_idx=cache_batch_idx,
        )
        expected = golden(q, reconstructed[cache_batch_idx : cache_batch_idx + 1], indices, scale, V_DIM)
        score = pcc(ttnn.to_torch(out), expected)
        assert score >= 0.99, f"scaled ND-sharded PCC {score:.5f} (slot={cache_batch_idx})"


# ---- A SHARDED kv's per-page bank mapping derives from its tensor shape (the accessor's shard strides), which
# ---- are baked at create time and NOT re-emitted on a cache-hit fast path. So unlike interleaved kv, changing
# ---- a sharded kv's T must recompile — else the 2nd call would reuse stale strides and read wrong banks. Here
# ---- both T values share ONE shard spec (same memory_config), so the hash must distinguish them by shape;
# ---- correct PCC for both proves it does. ----
@run_for_blackhole()
def test_sparse_sdpa_nd_sharded_kv_len_change(device):
    H, S, TOPK, kc = 32, 64, 64, 32
    device.clear_program_cache()
    scale = K_DIM**-0.5
    nd_cfg = _nd_sharded_dram_config(device, rows_per_shard=128)  # FIXED shard shape, independent of T
    for T in (256, 512):  # multiples of 128; same shard spec, different T (different shard count)
        q, kv, indices = make_inputs(H, S, T, TOPK, K_DIM, lambda s: TOPK)
        tt_q = to_dev(q.to(torch.bfloat16), device, ttnn.bfloat16)
        tt_kv = ttnn.from_torch(
            kv.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=nd_cfg,
        )
        tt_idx = to_dev(indices.to(torch.int32), device, ttnn.uint32)
        out = ttnn.transformer.sparse_sdpa(tt_q, tt_kv, tt_idx, V_DIM, scale=scale, k_chunk_size=kc)
        p = pcc(ttnn.to_torch(out), golden(q, kv, indices, scale, V_DIM))
        assert p >= 0.99, f"PCC {p:.5f} (nd-sharded, T={T})"
    assert device.num_program_cache_entries() == 2, "sharded kv must recompile per shape (2 distinct T -> 2 entries)"


# ---- cache_batch_idx is excluded from the program hash, so an out-of-range slot must be rejected even on a
# ---- program-cache HIT (validate_on_program_cache_hit re-checks the bound; without it the gather would run
# ---- out of bounds silently). ----
@run_for_blackhole()
def test_sparse_sdpa_indexed_oob_rejected_on_hit(device, expect_error):
    H, S, T, TOPK, kc, B = 32, 64, 256, 64, 32, 2
    device.clear_program_cache()
    q, kv_full, indices = _indexed_inputs(H, S, T, TOPK, B)
    tt_q = to_dev(q.to(torch.bfloat16), device, ttnn.bfloat16)
    tt_kv = to_dev(kv_full.to(torch.bfloat16), device, ttnn.bfloat16)
    tt_idx = to_dev(indices.to(torch.int32), device, ttnn.uint32)
    ttnn.transformer.sparse_sdpa(tt_q, tt_kv, tt_idx, V_DIM, k_chunk_size=kc, cache_batch_idx=0)  # miss: builds
    assert device.num_program_cache_entries() == 1
    # cache HIT, slot B is out of range [0,B) -> rejected by the hit validator
    with expect_error(RuntimeError, "cache_batch_idx"):
        ttnn.transformer.sparse_sdpa(tt_q, tt_kv, tt_idx, V_DIM, k_chunk_size=kc, cache_batch_idx=B)


# ---- q/indices layout/memory are NOT in the program hash, so an incompatible (e.g. TILE-layout) q with the
# ---- same shape/dtype must be rejected on a program-cache HIT too (validate_on_program_cache_hit re-checks
# ---- all non-hashed input invariants; without it the cached row-major program would run on a tiled tensor). ----
@run_for_blackhole()
def test_sparse_sdpa_bad_layout_rejected_on_hit(device, expect_error):
    H, S, T, TOPK, kc = 32, 64, 256, 64, 32
    device.clear_program_cache()
    q, kv, indices = make_inputs(H, S, T, TOPK, K_DIM, lambda s: TOPK)
    tt_q = to_dev(q.to(torch.bfloat16), device, ttnn.bfloat16)
    tt_kv = to_dev(kv.to(torch.bfloat16), device, ttnn.bfloat16)
    tt_idx = to_dev(indices.to(torch.int32), device, ttnn.uint32)
    ttnn.transformer.sparse_sdpa(tt_q, tt_kv, tt_idx, V_DIM, k_chunk_size=kc)  # miss: builds the row-major program
    assert device.num_program_cache_entries() == 1
    tt_q_tile = ttnn.to_layout(tt_q, ttnn.TILE_LAYOUT)  # same shape+dtype (same hash) but wrong layout -> HIT
    with expect_error(RuntimeError, "ROW_MAJOR"):
        ttnn.transformer.sparse_sdpa(tt_q_tile, tt_kv, tt_idx, V_DIM, k_chunk_size=kc)
