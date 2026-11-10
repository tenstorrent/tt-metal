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
from tt_py_test_utils_common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.sdpa.sparse_sdpa_test_utils import (
    ATTENTION_SINK_SHAPES,
    make_attention_sink_inputs,
    make_inputs,
    golden,
    to_dev,
    run_op,
    pcc,
)

K_DIM = 576  # head dim (q/kv width)
V_DIM = 512  # V width / output width (op arg)
SCALE_BLOCK_WIDTH = 128
BF16_KV = ttnn.transformer.SparseKVFormat.BF16
FP8_KV = ttnn.transformer.SparseKVFormat.FP8_E4M3
SCALED_FP8_KV = ttnn.transformer.SparseKVFormat.SCALED_FP8
SCALE = K_DIM**-0.5

# Open the device ONCE per module (not per test) — all tests share one device config, so this avoids the
# ~1.7s open/close on every test. The program cache persists across tests, which is fine: the recompile and
# indexed tests clear_program_cache() at their start, and the hash keeps distinct configs separate.
pytestmark = pytest.mark.use_module_device


def bf16_dev(t, device):
    return to_dev(t.to(torch.bfloat16), device, ttnn.bfloat16)


def idx_dev(indices, device):
    return to_dev(indices.to(torch.int32), device, ttnn.uint32)


def scaled_sdpa(tt_q, tt_packed, tt_idx, kc, scale=SCALE, **kwargs):
    return ttnn.transformer.sparse_sdpa(
        tt_q, tt_packed, tt_idx, V_DIM, kv_format=SCALED_FP8_KV, scale=scale, k_chunk_size=kc, **kwargs
    )


def nv_pattern(nv_id, TOPK):
    """Valid keys per query row."""
    return {
        "all": lambda s: TOPK,
        "mixed": lambda s: 1 + (s * 37) % TOPK,
        "causal": lambda s: min(TOPK, 1 + s * 3),
    }[nv_id]


def assert_pcc_per_token(actual, expected, what=""):
    """Whole-output PCC >= 0.999 and every query row's PCC >= 0.995."""
    score = pcc(actual, expected)
    worst = min(pcc(actual[:, :, s], expected[:, :, s]) for s in range(actual.shape[2]))
    assert score >= 0.999 and worst >= 0.995, f"{what}PCC {score:.5f}, worst token {worst:.5f}"


def make_scaled_kv_cache(device, batch, T, seed, round_scale, block_amax=None):
    """Build a packed scaled-FP8 cache and its reconstructed BF16 reference. The latent's 128-column scale blocks
    span 1e-3..1, or have the per-block max |x| given in `block_amax`."""
    gen = torch.Generator().manual_seed(seed)
    latent = torch.randn(batch, 1, T, V_DIM, generator=gen, dtype=torch.float32)
    if block_amax is None:
        latent *= torch.logspace(-3, 0, steps=V_DIM // SCALE_BLOCK_WIDTH).repeat_interleave(SCALE_BLOCK_WIDTH)
    else:
        blocks = latent.reshape(batch, 1, T, -1, SCALE_BLOCK_WIDTH)
        blocks = blocks / blocks.abs().amax(-1, keepdim=True) * torch.tensor(block_amax).reshape(-1, 1)
        latent = blocks.reshape(batch, 1, T, V_DIM)
    rope = torch.randn(batch, 1, T, K_DIM - V_DIM, generator=gen, dtype=torch.float32).to(torch.bfloat16)

    tt_latent = bf16_dev(latent, device)
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


def split_k_ran(device, tt_q, tt_kv, tt_idx, kv_format, kc, v_dim=V_DIM):
    """True if the program has the split-K merge CBs: more CBs than the whole-token split at S = cores.
    On a cache miss, NO_DISPATCH builds the program without compiling, running, or caching it.
    On a cache hit, it reports the cached program."""
    grid = device.compute_with_storage_grid_size()
    H, k_dim, topk = tt_q.shape[1], tt_q.shape[3], tt_idx.shape[3]
    whole_q = to_dev(torch.zeros(1, H, grid.x * grid.y, k_dim, dtype=torch.bfloat16), device, ttnn.bfloat16)
    whole_idx = to_dev(torch.zeros(1, 1, grid.x * grid.y, topk, dtype=torch.int32), device, ttnn.uint32)

    def num_cbs(q, idx):
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)
        try:
            ttnn.transformer.sparse_sdpa(
                q, tt_kv, idx, v_dim, kv_format=kv_format, scale=k_dim**-0.5, k_chunk_size=kc
            )
        finally:
            graph = ttnn.graph.end_graph_capture()
        return sum(node["node_type"] == "circular_buffer_allocate" for node in graph)

    return num_cbs(tt_q, tt_idx) > num_cbs(whole_q, whole_idx)


def test_sparse_kv_format_api_is_explicit(expect_error):
    assert ttnn.transformer.SparseKVFormat.BF16.name == "BF16"
    assert ttnn.transformer.SparseKVFormat.FP8_E4M3.name == "FP8_E4M3"
    assert ttnn.transformer.SparseKVFormat.SCALED_FP8.name == "SCALED_FP8"
    with expect_error(TypeError, "kv_format"):
        ttnn.transformer.sparse_sdpa(None, None, None, V_DIM)


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
    tt_q = bf16_dev(q, device)
    tt_idx = idx_dev(indices, device)

    block_cyclic_args = {"block_cyclic_sp_axis": 0, "block_cyclic_chunk_local": S} if kc == TOPK and TOPK == 64 else {}
    tt_out = scaled_sdpa(tt_q, tt_packed, tt_idx, kc, **block_cyclic_args)
    expected = golden(q, reconstructed, indices, SCALE, V_DIM)
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
    out = scaled_sdpa(bf16_dev(q, device), tt_packed, idx_dev(indices, device), kc)
    expected = golden(q, reconstructed, indices, SCALE, V_DIM)
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
            bf16_dev(q, device),
            malformed,
            idx_dev(indices, device),
            V_DIM,
            kv_format=SCALED_FP8_KV,
        )


@run_for_blackhole()
def test_sparse_sdpa_scaled_fp8_rejects_unaligned_rope_offset(device, expect_error):
    """Scaled rows require the latent+scale prefix to be addressable in the kernel's 16-byte units."""
    H, S, T, TOPK = 32, 32, 128, 32
    k_dim, v_dim = 192, 128
    scale_bytes = (v_dim // SCALE_BLOCK_WIDTH) * 4
    packed_width = v_dim + scale_bytes + (k_dim - v_dim) * 2
    q, _, indices = make_inputs(H, S, T, TOPK, k_dim, lambda s: TOPK)
    packed = to_dev(torch.randn(1, 1, T, packed_width), device, ttnn.fp8_e4m3)

    with expect_error(RuntimeError, "scale/RoPE boundary"):
        ttnn.transformer.sparse_sdpa(
            bf16_dev(q, device),
            packed,
            idx_dev(indices, device),
            v_dim,
            kv_format=SCALED_FP8_KV,
            k_chunk_size=TOPK,
        )


@run_for_blackhole()
def test_sparse_sdpa_rejects_v_dim_larger_than_k_dim(device, expect_error):
    H, S, T, TOPK = 32, 32, 128, 32
    q, kv, indices = make_inputs(H, S, T, TOPK, K_DIM, lambda s: TOPK)
    with expect_error(RuntimeError, "v_dim must be in"):
        ttnn.transformer.sparse_sdpa(
            bf16_dev(q, device),
            bf16_dev(kv, device),
            idx_dev(indices, device),
            K_DIM + 32,
            kv_format=BF16_KV,
        )


@run_for_blackhole()
@pytest.mark.parametrize(
    "kv_dtype,kv_format,message",
    [
        (ttnn.bfloat16, FP8_KV, "requires fp8_e4m3 storage"),
        (ttnn.fp8_e4m3, BF16_KV, "requires bfloat16 storage"),
    ],
)
def test_sparse_sdpa_rejects_kv_format_storage_mismatch(device, expect_error, kv_dtype, kv_format, message):
    H, S, T, TOPK = 32, 32, 128, 32
    q, kv, indices = make_inputs(H, S, T, TOPK, K_DIM, lambda s: TOPK)
    kv_host = kv.to(torch.bfloat16) if kv_dtype == ttnn.bfloat16 else kv
    with expect_error(RuntimeError, message):
        ttnn.transformer.sparse_sdpa(
            bf16_dev(q, device),
            to_dev(kv_host, device, kv_dtype),
            idx_dev(indices, device),
            V_DIM,
            kv_format=kv_format,
            k_chunk_size=TOPK,
        )


# ---- output dtype matches q (bf16 q -> bf16 out, fp8 q -> fp8 out) ----
@run_for_blackhole()
@pytest.mark.parametrize("q_dtype", [ttnn.bfloat16, ttnn.fp8_e4m3], ids=["q_bf16", "q_fp8"])
def test_sparse_sdpa_output_dtype(device, q_dtype):
    H, S, T, TOPK, kc = 32, 64, 256, 64, 32
    q, kv, indices = make_inputs(H, S, T, TOPK, K_DIM, lambda s: TOPK)
    q_host = q.to(torch.bfloat16) if q_dtype == ttnn.bfloat16 else q.to(torch.float32)
    tt_q = to_dev(q_host, device, q_dtype)
    tt_kv = bf16_dev(kv, device)
    tt_idx = idx_dev(indices, device)
    out = ttnn.transformer.sparse_sdpa(tt_q, tt_kv, tt_idx, V_DIM, kv_format=BF16_KV, scale=SCALE, k_chunk_size=kc)
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
    tt_kv = bf16_dev(kv, device)  # ONE persistent oversized K cache
    for valid_len in (256, 1024):  # each call's queries attend only to the [0, valid_len) populated prefix
        q = torch.randn(1, H, S, K_DIM, generator=gen, dtype=torch.float32)
        indices = torch.empty((1, 1, S, TOPK), dtype=torch.int64)
        for s in range(S):
            indices[0, 0, s, :] = torch.randperm(valid_len, generator=gen)[:TOPK]
        tt_q = bf16_dev(q, device)
        tt_idx = idx_dev(indices, device)
        tt_out = ttnn.transformer.sparse_sdpa(
            tt_q, tt_kv, tt_idx, V_DIM, kv_format=BF16_KV, scale=SCALE, k_chunk_size=kc
        )
        p = pcc(ttnn.to_torch(tt_out), golden(q, kv, indices, SCALE, V_DIM))
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
            bf16_dev(q, device),
            bf16_dev(kv, device),  # sp=1 => block-cyclic order == natural
            idx_dev(indices, device),
            V_DIM,
            kv_format=BF16_KV,
            scale=SCALE,
            k_chunk_size=kc,
            block_cyclic_sp_axis=0,
            block_cyclic_chunk_local=S,  # == q_isl (sp=1, tp=1)
        )
        p = pcc(ttnn.to_torch(out), golden(q, kv, indices, SCALE, V_DIM))
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
            bf16_dev(q, device),
            bf16_dev(kv_nat, device),
            idx_dev(indices, device),
            V_DIM,
            kv_format=BF16_KV,
            scale=SCALE,
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
    q, kv_full, indices = _indexed_inputs(H, S, T, TOPK, B)
    tt_q = bf16_dev(q, device)
    tt_kv = bf16_dev(kv_full, device)  # [B,1,T,K_DIM] shared cache
    tt_idx = idx_dev(indices, device)
    for cb in range(B):
        tt_out = ttnn.transformer.sparse_sdpa(
            tt_q, tt_kv, tt_idx, V_DIM, kv_format=BF16_KV, scale=SCALE, k_chunk_size=kc, cache_batch_idx=cb
        )
        p = pcc(ttnn.to_torch(tt_out), golden(q, kv_full[cb : cb + 1], indices, SCALE, V_DIM))  # golden uses slot cb
        assert p >= 0.99, f"PCC {p:.5f} (cache_batch_idx={cb})"
    n = device.num_program_cache_entries()  # changing the indexed slot must NOT recompile
    assert n == 1, f"indexing into a different slot recompiled: {n} program-cache entries (expected 1)"


# ---- override_runtime_arguments on a cache HIT must re-apply BOTH the excluded slot offset
# ---- (kv_batch_page_offset = cache_batch_idx*T) AND every tensor buffer address. Here each iteration changes
# ---- cache_batch_idx AND freshly allocates q/kv/indices (prior tensors kept alive so the new buffers get
# ---- different addresses) — correct PCC for every step, on ONE cache entry, proves the hit path re-applies
# ---- both (not stale baked values). ----
@run_for_blackhole()
def test_sparse_sdpa_indexed_addr_change_on_hit(device):
    H, S, T, TOPK, kc, B = 32, 64, 256, 64, 32, 3
    device.clear_program_cache()
    keep_alive = []  # hold device tensors so each reallocation lands at a fresh address
    for cb in range(B):
        q, kv_full, indices = _indexed_inputs(H, S, T, TOPK, B, seed=cb)  # fresh data + fresh buffers each step
        tt_q = bf16_dev(q, device)
        tt_kv = bf16_dev(kv_full, device)
        tt_idx = idx_dev(indices, device)
        keep_alive += [tt_q, tt_kv, tt_idx]
        tt_out = ttnn.transformer.sparse_sdpa(
            tt_q, tt_kv, tt_idx, V_DIM, kv_format=BF16_KV, scale=SCALE, k_chunk_size=kc, cache_batch_idx=cb
        )
        p = pcc(ttnn.to_torch(tt_out), golden(q, kv_full[cb : cb + 1], indices, SCALE, V_DIM))
        assert p >= 0.99, f"PCC {p:.5f} (cache_batch_idx={cb}, reallocated buffers)"
    n = device.num_program_cache_entries()  # slot + address changes are runtime args, not a recompile
    assert n == 1, f"addr/slot change on hit recompiled: {n} program-cache entries (expected 1)"


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
    q, kv_full, indices = _indexed_inputs(H, S, T, TOPK, B)
    nd_cfg = _nd_sharded_dram_config(device, rows_per_shard=T)  # each [1,1,T,K_DIM] slot is one shard
    tt_q = bf16_dev(q, device)
    tt_kv = ttnn.from_torch(
        kv_full.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=nd_cfg,
    )
    tt_idx = idx_dev(indices, device)
    for cb in range(B):
        tt_out = ttnn.transformer.sparse_sdpa(
            tt_q, tt_kv, tt_idx, V_DIM, kv_format=BF16_KV, scale=SCALE, k_chunk_size=kc, cache_batch_idx=cb
        )
        p = pcc(ttnn.to_torch(tt_out), golden(q, kv_full[cb : cb + 1], indices, SCALE, V_DIM))
        assert p >= 0.99, f"PCC {p:.5f} (nd-sharded, cache_batch_idx={cb})"


@run_for_blackhole()
def test_sparse_sdpa_scaled_fp8_indexed_nd_sharded_kv(device):
    """Production cache geometry: one packed ND-sharded cache and a dynamic user/layer slot."""
    H, S, T, TOPK, kc, B = 32, 32, 128, 64, 32, 2
    q, _, indices = make_inputs(H, S, T, TOPK, K_DIM, lambda s: TOPK, seed=51)
    tt_packed, reconstructed = make_scaled_kv_cache(device, B, T, seed=52, round_scale=True)
    tt_packed = ttnn.to_memory_config(tt_packed, _nd_sharded_dram_config(device, T, tt_packed.shape[-1]))
    tt_q = bf16_dev(q, device)
    tt_idx = idx_dev(indices, device)

    for cache_batch_idx in range(B):
        out = scaled_sdpa(tt_q, tt_packed, tt_idx, kc, cache_batch_idx=cache_batch_idx)
        expected = golden(q, reconstructed[cache_batch_idx : cache_batch_idx + 1], indices, SCALE, V_DIM)
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
    nd_cfg = _nd_sharded_dram_config(device, rows_per_shard=128)  # FIXED shard shape, independent of T
    for T in (256, 512):  # multiples of 128; same shard spec, different T (different shard count)
        q, kv, indices = make_inputs(H, S, T, TOPK, K_DIM, lambda s: TOPK)
        tt_q = bf16_dev(q, device)
        tt_kv = ttnn.from_torch(
            kv.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=nd_cfg,
        )
        tt_idx = idx_dev(indices, device)
        out = ttnn.transformer.sparse_sdpa(tt_q, tt_kv, tt_idx, V_DIM, kv_format=BF16_KV, scale=SCALE, k_chunk_size=kc)
        p = pcc(ttnn.to_torch(out), golden(q, kv, indices, SCALE, V_DIM))
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
    tt_q = bf16_dev(q, device)
    tt_kv = bf16_dev(kv_full, device)
    tt_idx = idx_dev(indices, device)
    ttnn.transformer.sparse_sdpa(
        tt_q, tt_kv, tt_idx, V_DIM, kv_format=BF16_KV, k_chunk_size=kc, cache_batch_idx=0
    )  # miss: builds
    assert device.num_program_cache_entries() == 1
    # cache HIT, slot B is out of range [0,B) -> rejected by the hit validator
    with expect_error(RuntimeError, "cache_batch_idx"):
        ttnn.transformer.sparse_sdpa(tt_q, tt_kv, tt_idx, V_DIM, kv_format=BF16_KV, k_chunk_size=kc, cache_batch_idx=B)


# ---- q/indices layout/memory are NOT in the program hash, so an incompatible (e.g. TILE-layout) q with the
# ---- same shape/dtype must be rejected on a program-cache HIT too (validate_on_program_cache_hit re-checks
# ---- all non-hashed input invariants; without it the cached row-major program would run on a tiled tensor). ----
@run_for_blackhole()
def test_sparse_sdpa_bad_layout_rejected_on_hit(device, expect_error):
    H, S, T, TOPK, kc = 32, 64, 256, 64, 32
    device.clear_program_cache()
    q, kv, indices = make_inputs(H, S, T, TOPK, K_DIM, lambda s: TOPK)
    tt_q = bf16_dev(q, device)
    tt_kv = bf16_dev(kv, device)
    tt_idx = idx_dev(indices, device)
    ttnn.transformer.sparse_sdpa(
        tt_q, tt_kv, tt_idx, V_DIM, kv_format=BF16_KV, k_chunk_size=kc
    )  # miss: builds the row-major program
    assert device.num_program_cache_entries() == 1
    tt_q_tile = ttnn.to_layout(tt_q, ttnn.TILE_LAYOUT)  # same shape+dtype (same hash) but wrong layout -> HIT
    with expect_error(RuntimeError, "ROW_MAJOR"):
        ttnn.transformer.sparse_sdpa(tt_q_tile, tt_kv, tt_idx, V_DIM, kv_format=BF16_KV, k_chunk_size=kc)


@run_for_blackhole()
@pytest.mark.parametrize("H,S,TOPK,dim,kc", ATTENTION_SINK_SHAPES)
def test_sparse_sdpa_attention_sink(device, H, S, TOPK, dim, kc):
    q, kv, indices, sinks, scale = make_attention_sink_inputs(H, S, TOPK, dim)
    tt_q = to_dev(q, device, ttnn.bfloat16)
    tt_kv = to_dev(kv, device, ttnn.bfloat16)
    tt_indices = idx_dev(indices, device)
    tt_sink = to_dev(sinks, device, ttnn.bfloat16)
    out = ttnn.transformer.sparse_sdpa(
        tt_q, tt_kv, tt_indices, dim, kv_format=BF16_KV, scale=scale, k_chunk_size=kc, attention_sink=tt_sink
    )
    actual = ttnn.to_torch(out).float()
    # Compute the PyTorch reference in batches of query positions to limit host memory use.
    # This batch size is independent of hardware tiles and the device k_chunk_size.
    # Concatenating along the query axis restores the full expected output for every head.
    reference_query_batch_size = 32
    expected = torch.cat(
        [
            golden(
                q[:, :, start : start + reference_query_batch_size].float(),
                kv.float(),
                indices[:, :, start : start + reference_query_batch_size],
                scale,
                dim,
                attention_sink=sinks.float().reshape(1, H, 1, 1),
            )
            for start in range(0, S, reference_query_batch_size)
        ],
        dim=2,
    )
    score = pcc(actual, expected)
    assert score >= 0.99, f"Attention sink PCC {score:.5f}"
    # Correlation alone would miss the sink's suppression of the output magnitude.
    torch.testing.assert_close(actual, expected, atol=0.035, rtol=0.08)


@run_for_blackhole()
def test_sparse_sdpa_attention_sink_cache(device):
    device.clear_program_cache()
    grid = device.compute_with_storage_grid_size()
    H, S, dim, scale = 64, grid.x * grid.y + 1, 64, 0.25
    # Zero scores and unit values give the exact output nv / (nv + exp(scale * sink)).
    tt_q = to_dev(torch.zeros(1, H, S, dim), device, ttnn.bfloat16)
    tt_kv = to_dev(torch.ones(1, 1, 32, dim), device, ttnn.bfloat16)
    indices = torch.arange(32).reshape(1, 1, 1, 32).expand(1, 1, S, 32).contiguous()
    tt_indices = idx_dev(indices, device)
    sink_tensors = []  # keep allocations alive to guarantee distinct addresses on cache hits
    for value in (None, 0.0, 16.0, -32.0, None):
        sink = None
        if value is not None:
            sink = ttnn.from_torch(
                torch.full((1, 1, 1, H), value), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
            )
            sink_tensors.append(sink)
        out = ttnn.transformer.sparse_sdpa(
            tt_q, tt_kv, tt_indices, dim, kv_format=BF16_KV, scale=scale, k_chunk_size=32, attention_sink=sink
        )
        expected = 1.0 if value is None else 32 / (32 + torch.exp(torch.tensor(scale * value)).item())
        torch.testing.assert_close(
            ttnn.to_torch(out).float(), torch.full((1, H, S, dim), expected), atol=0.012, rtol=0.01
        )
        assert device.num_program_cache_entries() == (1 if not sink_tensors else 2)


@run_for_blackhole()
@pytest.mark.parametrize("invalid", ["heads", "layout", "memory", "dtype", "shape", "bfp8", "bfp4"])
def test_sparse_sdpa_attention_sink_validation(device, expect_error, invalid):
    q, kv, indices = make_inputs(32, 1, 64, 32, 64, lambda s: 32)
    heads = 64 if invalid == "heads" else 32
    sink_shape = (1, heads, 1, 1) if invalid == "shape" else (1, 1, 1, heads)
    dtype = {"dtype": ttnn.float32, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}.get(invalid, ttnn.bfloat16)
    sink = ttnn.from_torch(
        torch.zeros(sink_shape),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT if invalid in ("layout", "bfp8", "bfp4") else ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG if invalid == "memory" else ttnn.DRAM_MEMORY_CONFIG,
        device=device,
    )
    with expect_error(RuntimeError, "Attention sink"):
        ttnn.transformer.sparse_sdpa(
            to_dev(q, device, ttnn.bfloat16),
            to_dev(kv, device, ttnn.bfloat16),
            idx_dev(indices, device),
            64,
            kv_format=BF16_KV,
            k_chunk_size=32,
            attention_sink=sink,
        )


@run_for_blackhole()
def test_sparse_sdpa_attention_sink_dense_parity(device):
    # Selecting every key must match classic SDPA, including its sink scaling convention.
    H, S, T, dim, scale = 64, 32, 32, 64, 0.25
    torch.manual_seed(0)
    q = torch.randn(1, H, S, dim, dtype=torch.bfloat16)
    kv = torch.randn(1, 1, T, dim, dtype=torch.bfloat16)
    indices = torch.arange(T).reshape(1, 1, 1, T).expand(1, 1, S, T).contiguous()
    sink_values = torch.linspace(-16, 32, H)
    sparse_sink = ttnn.from_torch(
        sink_values.reshape(1, 1, 1, H),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    dense_sink = ttnn.from_torch(
        sink_values.reshape(1, H, 1, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    sparse = ttnn.transformer.sparse_sdpa(
        to_dev(q, device, ttnn.bfloat16),
        to_dev(kv, device, ttnn.bfloat16),
        idx_dev(indices, device),
        dim,
        kv_format=BF16_KV,
        scale=scale,
        k_chunk_size=T,
        attention_sink=sparse_sink,
    )
    tiled_q = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tiled_kv = ttnn.from_torch(kv, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    dense = ttnn.transformer.scaled_dot_product_attention(
        tiled_q,
        tiled_kv,
        tiled_kv,
        is_causal=False,
        scale=scale,
        attention_sink=dense_sink,
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(), q_chunk_size=S, k_chunk_size=T
        ),
    )
    actual, expected = ttnn.to_torch(sparse).float(), ttnn.to_torch(dense).float()
    score = pcc(actual, expected)
    assert score >= 0.99, f"Sparse/dense attention sink PCC {score:.5f}"
    torch.testing.assert_close(actual, expected, atol=0.035, rtol=0.08)


@run_for_blackhole()
@pytest.mark.parametrize("scale", [None, 0.25])
def test_sparse_sdpa_deepseek_attention_sink(device, scale):
    # DeepSeek-V4 appends the learned sink AFTER scaling QK. Verify caller conversion against that
    # convention directly, rather than a reference that scales the sink along with QK.
    H, S, T, dim = 64, 4, 32, 64
    resolved_scale = dim**-0.5 if scale is None else scale
    # Constant key/value rows and binary-fraction queries make QK exact, isolating sink scaling
    # from random-matmul rounding. Nonzero QK scores still exercise the supplied/default scale.
    q = (torch.arange(H, dtype=torch.float32) - H // 2).reshape(1, H, 1, 1) / 256
    q = q.expand(1, H, S, dim).to(torch.bfloat16).contiguous()
    kv = torch.ones(1, 1, T, dim, dtype=torch.bfloat16)
    indices = torch.arange(T).reshape(1, 1, 1, T).expand(1, 1, S, T).contiguous()
    model_sink = torch.linspace(-4, 8, H).reshape(1, H, 1, 1).to(torch.bfloat16).float()
    sink = ttnn.from_torch(
        (model_sink / resolved_scale).reshape(1, 1, 1, H),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    out = ttnn.transformer.sparse_sdpa(
        to_dev(q, device, ttnn.bfloat16),
        to_dev(kv, device, ttnn.bfloat16),
        idx_dev(indices, device),
        dim,
        kv_format=BF16_KV,
        scale=scale,
        k_chunk_size=T,
        attention_sink=sink,
    )
    logits = q.float() @ kv.float().transpose(-1, -2) * resolved_scale
    probs = torch.cat([logits, model_sink.expand(1, H, S, 1)], dim=-1).softmax(dim=-1)[..., :-1]
    expected = probs @ kv.float()
    torch.testing.assert_close(ttnn.to_torch(out).float(), expected, atol=0.012, rtol=0.01)


def check_scaled_split_k(device, H, S, T, TOPK, kc, nv_id, split, seed):
    """Scaled-FP8 KV: split-K is (or is not) planned as expected, and the output matches the golden per token."""
    q, _, indices = make_inputs(H, S, T, TOPK, K_DIM, nv_pattern(nv_id, TOPK), seed=seed)
    tt_packed, reconstructed = make_scaled_kv_cache(device, 1, T, seed=seed + 1, round_scale=True)
    tt_q = bf16_dev(q, device)
    tt_idx = idx_dev(indices, device)
    assert split_k_ran(device, tt_q, tt_packed, tt_idx, SCALED_FP8_KV, kc) == split, f"expected split-K {split}"
    out = scaled_sdpa(tt_q, tt_packed, tt_idx, kc)
    assert_pcc_per_token(ttnn.to_torch(out), golden(q, reconstructed, indices, SCALE, V_DIM))


# ---- split-K work distribution (S > cores, S % cores != 0): tokens straddling two cores are merged ----
@run_for_blackhole()
@pytest.mark.parametrize(
    "H,S,T,TOPK,kc,nv_id,split",
    [
        (64, 160, 4096, 2048, 128, "all", True),  # GLM per-chip shape (16 chunks/token)
        (64, 160, 1024, 512, 128, "mixed", True),  # sentinel tails: empty HEAD segments, masks in either segment
        (32, 250, 1024, 512, 128, "causal", True),  # 2+ tokens/core, prefill-like valid counts
        (64, None, 1024, 256, 64, "mixed", True),  # 1 token + a sliver per core: S = cores + 10
        (128, 160, 1024, 512, 128, "mixed", True),  # Sqt=4: the merge fills DST
        (96, 160, 1024, 512, 128, "mixed", True),  # Sqt=3
        (128, 160, 4096, 1344, 448, "mixed", False),  # split-K CBs over the L1 budget: whole-token split
    ],
)
def test_sparse_sdpa_scaled_fp8_split_k(device, H, S, T, TOPK, kc, nv_id, split):
    """Split-K must run (or fall back) as expected and match the torch golden (PCC), incl. sentinel tails."""
    grid = device.compute_with_storage_grid_size()
    cores = grid.x * grid.y
    S = cores + 10 if S is None else S
    if S <= cores:
        pytest.skip(f"split-K needs S > cores ({cores})")
    check_scaled_split_k(device, H, S, T, TOPK, kc, nv_id, split, seed=71)


# ---- split-K for S < cores (small chunks, e.g. GLM chunk 1024/2048 -> 32/64 rows per chip): <= 2 cores per token ----
@run_for_blackhole()
@pytest.mark.parametrize(
    "H,S,T,TOPK,kc,nv_id,split",
    [
        (64, 32, 4096, 2048, 128, "all", True),  # GLM chunk 1024 per chip: 64 cores, half a token each
        (64, 64, 4096, 2048, 128, "all", True),  # GLM chunk 2048: 96 cores, 2/3 token each (HEAD+TAIL cores)
        (64, 32, 1024, 512, 128, "mixed", True),  # sentinel tails: empty HEAD segments
        (64, 64, 1024, 512, 128, "causal", True),  # prefill-like valid counts
        (32, 40, 1024, 256, 64, "mixed", True),  # 4 chunks/token, odd S
        (128, 32, 4096, 1344, 448, "mixed", False),  # split-K CBs over the L1 budget: whole-token split
    ],
)
def test_sparse_sdpa_scaled_fp8_split_k_small_s(device, H, S, T, TOPK, kc, nv_id, split):
    """Small-S split-K must run (or fall back) as expected and match the torch golden (PCC), incl. sentinel tails."""
    grid = device.compute_with_storage_grid_size()
    if S >= grid.x * grid.y:
        pytest.skip("needs S < cores")
    check_scaled_split_k(device, H, S, T, TOPK, kc, nv_id, split, seed=73)


# ---- BF16 and plain FP8 KV: split-K only when every token gets two cores (2S <= cores), e.g. Qwen 3.8 QSA ----
@run_for_blackhole()
@pytest.mark.parametrize("kv_dtype", [ttnn.bfloat16, ttnn.fp8_e4m3], ids=["bf16", "fp8"])
@pytest.mark.parametrize(
    "H,S,T,TOPK,k_dim,v_dim,kc,nv_id,split",
    [
        (64, 32, 4096, 2048, K_DIM, V_DIM, 128, "all", True),  # GLM chunk 1024 per chip: half a token per core
        (64, 32, 1024, 512, K_DIM, V_DIM, 128, "mixed", True),  # sentinel tails
        (64, 40, 1024, 512, K_DIM, V_DIM, 128, "causal", True),  # prefill-like valid counts
        (128, 16, 1024, 512, K_DIM, V_DIM, 64, "mixed", True),  # Sqt=4
        (32, 1, 4096, 2048, 512, 256, 32, "all", True),  # Qwen 3.8 QSA decode: 1 row, 64 chunks on 2 cores
        (32, 5, 4096, 2080, 512, 256, 32, "mixed", True),  # QSA rows with 65 chunks: 32 + 33 per token
        (64, 64, 1024, 512, K_DIM, V_DIM, 128, "all", False),  # 2S > cores: whole-token split
        (64, 160, 1024, 512, K_DIM, V_DIM, 128, "mixed", False),  # S > cores: whole-token split
    ],
)
def test_sparse_sdpa_split_k_kv_formats(device, kv_dtype, H, S, T, TOPK, k_dim, v_dim, kc, nv_id, split):
    """BF16 and plain FP8 KV must use split-K only when 2S <= cores and match the torch golden (PCC)."""
    grid = device.compute_with_storage_grid_size()
    cores = grid.x * grid.y
    if split != (2 * S <= cores):
        pytest.skip(f"case needs 2S {'<=' if split else '>'} cores ({cores})")
    q, kv, indices = make_inputs(H, S, T, TOPK, k_dim, nv_pattern(nv_id, TOPK), seed=77)
    fp8 = kv_dtype == ttnn.fp8_e4m3
    kv_format = FP8_KV if fp8 else BF16_KV
    tt_q = bf16_dev(q, device)
    tt_kv = to_dev(kv if fp8 else kv.to(torch.bfloat16), device, kv_dtype)
    tt_idx = idx_dev(indices, device)
    assert split_k_ran(device, tt_q, tt_kv, tt_idx, kv_format, kc, v_dim) == split, f"expected split-K {split}"
    out = ttnn.transformer.sparse_sdpa(
        tt_q, tt_kv, tt_idx, v_dim, kv_format=kv_format, scale=k_dim**-0.5, k_chunk_size=kc
    )
    kv_held = kv.to(torch.float8_e4m3fn if fp8 else torch.bfloat16).float()
    assert_pcc_per_token(ttnn.to_torch(out), golden(q, kv_held, indices, k_dim**-0.5, v_dim))


# ---- the split-K plan reads the free L1, so the program hash includes it: less free L1 must give a new program ----
@run_for_blackhole()
def test_sparse_sdpa_split_k_plan_follows_free_l1(device):
    """An L1 tensor that leaves room for the whole-token CBs but not for the split-K CBs must give a new
    whole-token program. A cache hit would reuse the split-K program, whose CB region overlaps the tensor."""
    H, S, T, TOPK, kc = 64, 160, 4096, 2048, 128
    grid = device.compute_with_storage_grid_size()
    cores = grid.x * grid.y
    if S <= cores or S % cores == 0:
        pytest.skip(f"split-K needs S > cores ({cores}) and S % cores != 0")
    q, _, indices = make_inputs(H, S, T, TOPK, K_DIM, lambda s: TOPK, seed=75)
    tt_packed, reconstructed = make_scaled_kv_cache(device, 1, T, seed=76, round_scale=True)
    tt_q = bf16_dev(q, device)
    tt_idx = idx_dev(indices, device)
    whole_q = to_dev(torch.zeros(1, H, cores, K_DIM, dtype=torch.bfloat16), device, ttnn.bfloat16)
    whole_idx = to_dev(torch.zeros(1, 1, cores, TOPK, dtype=torch.int32), device, ttnn.uint32)
    expected = golden(q, reconstructed, indices, SCALE, V_DIM)

    def run(q_in, idx_in):
        """Run the op. Return the output and the end of the CB region (0 on a program-cache hit)."""
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        try:
            out = scaled_sdpa(q_in, tt_packed, idx_in, kc)
        finally:
            graph = ttnn.graph.end_graph_capture()
        cbs = [node["params"] for node in graph if node["node_type"] == "circular_buffer_allocate"]
        return out, max((int(p["address"]) + int(p["size"]) for p in cbs), default=0)

    def l1_tensor(tiles_per_core):
        shape = [1, 1, cores * tiles_per_core * 32, 32]
        mem = ttnn.create_sharded_memory_config(shape, ttnn.CoreGrid(y=grid.y, x=grid.x), ttnn.ShardStrategy.HEIGHT)
        data = torch.randn(shape, dtype=torch.bfloat16)
        return data, ttnn.from_torch(
            data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem
        )

    device.clear_program_cache()
    out_split, split_end = run(tt_q, tt_idx)
    _, whole_end = run(whole_q, whole_idx)
    assert split_k_ran(device, tt_q, tt_packed, tt_idx, SCALED_FP8_KV, kc)
    assert 0 < whole_end < split_end, f"CB region ends: whole-token {whole_end}, split-K {split_end}"

    # L1 tensors are allocated top-down, so a one-tile probe finds the top of the free L1.
    _, probe = l1_tensor(1)
    top = probe.buffer_address() + 2048
    ttnn.deallocate(probe)
    data, blocker = l1_tensor(-(-(top - (whole_end + split_end) // 2) // 2048))
    low = blocker.buffer_address()
    assert whole_end <= low < split_end, f"L1 tensor at {low}, not in [{whole_end}, {split_end})"

    assert not split_k_ran(device, tt_q, tt_packed, tt_idx, SCALED_FP8_KV, kc)
    entries = device.num_program_cache_entries()
    out_whole, whole_run_end = run(tt_q, tt_idx)
    assert device.num_program_cache_entries() == entries + 1, "less free L1 must give a new program"
    assert 0 < whole_run_end <= low, f"CB region ends at {whole_run_end}, L1 tensor starts at {low}"
    assert torch.equal(ttnn.to_torch(blocker), data), "the op wrote over the L1 tensor"
    ttnn.deallocate(blocker)
    for name, out in (("split-K", out_split), ("whole-token", out_whole)):
        score = pcc(ttnn.to_torch(out), expected)
        assert score >= 0.999, f"{name} PCC {score:.5f}"


# ---- production shapes: 160 query rows per chip (more rows than cores, not a multiple of the core count), all
#      heads after the head-to-sequence reshard (GLM-5.3: 64, Kimi K3.1: 96), TOPK=2048, k_chunk=128, scaled-FP8 KV ----
@run_for_blackhole()
@pytest.mark.parametrize("H", [64, 96], ids=["glm53", "k31"])
@pytest.mark.parametrize("nv_id", ["all", "causal"])
def test_sparse_sdpa_production_pcc(device, H, nv_id):
    S, T, TOPK, kc = 160, 8192, 2048, 128
    nv_fn = {"all": lambda s: TOPK, "causal": lambda s: min(TOPK, 1 + s * 13)}[nv_id]
    q, _, indices = make_inputs(H, S, T, TOPK, K_DIM, nv_fn, seed=81)
    tt_packed, reconstructed = make_scaled_kv_cache(device, 1, T, seed=82, round_scale=True)
    out = scaled_sdpa(bf16_dev(q, device), tt_packed, idx_dev(indices, device), kc)
    assert_pcc_per_token(ttnn.to_torch(out), golden(q, reconstructed, indices, SCALE, V_DIM), f"H={H} ")


# ---- fused-dequant edge cases on a small scaled-FP8 shape ----
SMALL_SCALED_SHAPE = (32, 32, 256, 128, 64)  # H, S, T, TOPK, kc

# Per-128-column block amax: below the cast's 1e-4 clamp, just above it (an unrounded scale of ~2^-22.1),
# mid-range, and large (a power-of-two scale of 2^7, the top FP16 binade after the 2^9 offset).
SCALE_RANGE_AMAX = (5e-5, 1.05e-4, 3.0, 4.0e4)


@run_for_blackhole()
@pytest.mark.parametrize("round_scale", [True, False], ids=["pow2_scale", "arbitrary_scale"])
def test_sparse_sdpa_scaled_fp8_scale_range(device, round_scale):
    """Checks magnitudes per scale block, not just correlation (PCC passes for a uniformly rescaled output)."""
    H, S, T, TOPK, kc = SMALL_SCALED_SHAPE
    q, _, indices = make_inputs(H, S, T, TOPK, K_DIM, lambda s: 1 + (s * 7) % TOPK, seed=91)
    # Zero Q's latent so the scores come from the BF16 RoPE alone: the 4e4 block would otherwise make the logits
    # so large that rounding decides the softmax. The output still reads every dequantized latent block as V.
    q[..., :V_DIM] = 0
    tt_packed, reconstructed = make_scaled_kv_cache(
        device, 1, T, seed=92, round_scale=round_scale, block_amax=SCALE_RANGE_AMAX
    )
    out = scaled_sdpa(bf16_dev(q, device), tt_packed, idx_dev(indices, device), kc)
    actual = ttnn.to_torch(out).float()
    expected = golden(q, reconstructed, indices, SCALE, V_DIM)
    for b, amax in enumerate(SCALE_RANGE_AMAX):
        cols = slice(b * SCALE_BLOCK_WIDTH, (b + 1) * SCALE_BLOCK_WIDTH)
        ref = expected[..., cols].abs().max()
        err = (actual[..., cols] - expected[..., cols]).abs().max() / ref
        assert err < 0.03, f"block {b} (amax {amax}): max error {err:.4f} of the block's max |out| {ref:.3e}"


@run_for_blackhole()
def test_sparse_sdpa_scaled_fp8_attention_sink(device):
    """Scaled FP8 KV with an attention sink keeps the two-pass dequant."""
    H, S, T, TOPK, kc = SMALL_SCALED_SHAPE
    q, _, indices = make_inputs(H, S, T, TOPK, K_DIM, lambda s: 1 + (s * 7) % TOPK, seed=93)
    tt_packed, reconstructed = make_scaled_kv_cache(device, 1, T, seed=94, round_scale=True)
    sinks = (torch.linspace(-4, 8, H) / SCALE).reshape(1, 1, 1, H).to(torch.bfloat16)
    tt_sinks = to_dev(sinks, device, ttnn.bfloat16)
    out = scaled_sdpa(bf16_dev(q, device), tt_packed, idx_dev(indices, device), kc, attention_sink=tt_sinks)
    actual = ttnn.to_torch(out).float()
    expected = golden(q, reconstructed, indices, SCALE, V_DIM, attention_sink=sinks.float().reshape(1, H, 1, 1))
    score = pcc(actual, expected)
    assert score >= 0.999, f"scaled FP8 + sink PCC {score:.5f}"
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)  # latent values up to ~4


@run_for_blackhole()
def test_sparse_sdpa_scaled_fp8_kv_fp8_q(device):
    """FP8 Q with scaled FP8 KV: the fused-dequant Q prescale runs on the BFP8 cb_q_in."""
    H, S, T, TOPK, kc = SMALL_SCALED_SHAPE
    q, _, indices = make_inputs(H, S, T, TOPK, K_DIM, lambda s: TOPK, seed=95)
    tt_packed, reconstructed = make_scaled_kv_cache(device, 1, T, seed=96, round_scale=True)
    tt_q = to_dev(q.to(torch.float32), device, ttnn.fp8_e4m3)
    out = scaled_sdpa(tt_q, tt_packed, idx_dev(indices, device), kc)
    actual = ttnn.to_torch(ttnn.typecast(out, ttnn.bfloat16)).float()  # no FP8 to_torch
    expected = golden(q.to(torch.float8_e4m3fn).float(), reconstructed, indices, SCALE, V_DIM)  # the Q the op sees
    score = pcc(actual, expected)
    assert score >= 0.99, f"FP8 Q + scaled FP8 KV PCC {score:.5f}"
    # FP8 output: compare magnitudes coarsely (PCC alone would pass a rescaled output).
    ratio = actual.norm() / expected.norm()
    assert 0.95 < ratio < 1.05, f"output norm ratio {ratio:.4f}"
