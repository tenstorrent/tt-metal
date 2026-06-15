# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""sparse_sdpa (sparse MLA prefill) unit tests — Blackhole only. See PLAN_sparse_sdpa.md.

Correctness uses SMALL parametric shapes (the golden gathers sel[S,k,D]; full 640/2048/56320 is ~2.8 GiB).
"""

import math

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v32.reference_cpu.sparse_sdpa_prefill import sparse_mla, MASKED_INDEX

K_DIM = 576
V_DIM = 512


def _make_inputs(H, S, T, TOPK, n_valid_fn, seed=0):
    """Build (q, kv, indices) torch tensors matching the producer contract (tail-shaped sentinels)."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(1, H, S, K_DIM, generator=gen, dtype=torch.float32)
    kv = torch.randn(1, 1, T, K_DIM, generator=gen, dtype=torch.float32)
    indices = torch.full((1, 1, S, TOPK), MASKED_INDEX, dtype=torch.int64)
    for s in range(S):
        nv = max(1, min(TOPK, n_valid_fn(s)))
        perm = torch.randperm(T, generator=gen)[:nv]
        indices[0, 0, s, :nv] = perm
    return q, kv, indices


def _golden(q, kv, indices, scale):
    # sparse_mla expects kvpe [T,576] and indices reshaped (it accepts [..,S,k]).
    out = sparse_mla(q, kv[0, 0], indices.to(torch.int64), scale)  # [1,H,S,512]
    return out


def _to_dev(t, device, dtype):
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _run_op(q, kv, indices, device, k_chunk_size):
    tt_q = _to_dev(q.to(torch.bfloat16), device, ttnn.bfloat16)
    tt_kv = _to_dev(kv.to(torch.bfloat16), device, ttnn.bfloat16)
    tt_idx = _to_dev(indices.to(torch.int32), device, ttnn.uint32)
    scale = K_DIM**-0.5
    tt_out = ttnn.transformer.sparse_sdpa(tt_q, tt_kv, tt_idx, V_DIM, scale=scale, k_chunk_size=k_chunk_size)
    return ttnn.to_torch(tt_out), scale


# ---- Phase 1: op runs, output shape/layout correct (compute is a no-op → zeros) ----
@run_for_blackhole()
def test_sparse_sdpa_phase1_runs(device):
    H, S, T, TOPK, kc = 32, 64, 256, 64, 32
    q, kv, indices = _make_inputs(H, S, T, TOPK, lambda s: TOPK)
    out, _ = _run_op(q, kv, indices, device, kc)
    assert tuple(out.shape) == (1, H, S, V_DIM), f"got {tuple(out.shape)}"


# ---- First passing target: SINGLE-CHUNK PCC (k_chunk == TOPK) vs sparse_mla golden ----
@run_for_blackhole()
@pytest.mark.parametrize("S,T,TOPK", [(64, 256, 32), (64, 512, 64), (32, 1024, 128)])
@pytest.mark.parametrize(
    "nv_fn,nv_id",
    [(lambda s: 10**9, "all_valid"), (lambda s: 1 + (s % 7), "few_valid"), (lambda s: 1 + (s * 3) % 20, "boundary")],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_sparse_sdpa_pcc_single_chunk(device, S, T, TOPK, nv_fn, nv_id):
    H = 32
    q, kv, indices = _make_inputs(H, S, T, TOPK, nv_fn)
    out, scale = _run_op(q, kv, indices, device, k_chunk_size=TOPK)  # single chunk
    golden = _golden(q, kv, indices, scale).to(torch.float32)
    pcc = torch.corrcoef(torch.stack([out.flatten().float(), golden.flatten()]))[0, 1].item()
    assert pcc >= 0.99, f"PCC {pcc:.5f} (S={S},T={T},TOPK={TOPK},{nv_id})"


# ---- Phase 5 target: multi-chunk PCC vs sparse_mla golden (needs flash accumulation) ----
@run_for_blackhole()
@pytest.mark.parametrize("S,T,TOPK,kc", [(64, 256, 64, 32), (64, 512, 128, 64), (32, 1024, 256, 128)])
@pytest.mark.parametrize(
    "nv_fn,nv_id",
    [
        (lambda s: 10**9, "all_valid"),
        (lambda s: 1 + (s % 7), "few_valid"),
        (lambda s: 1 + (s * 3) % 50, "mid_tile_boundary"),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_sparse_sdpa_pcc(device, S, T, TOPK, kc, nv_fn, nv_id):
    H = 32
    q, kv, indices = _make_inputs(H, S, T, TOPK, nv_fn)
    out, scale = _run_op(q, kv, indices, device, kc)
    golden = _golden(q, kv, indices, scale).to(torch.float32)
    pcc = torch.corrcoef(torch.stack([out.flatten().float(), golden.flatten()]))[0, 1].item()
    assert pcc >= 0.99, f"PCC {pcc:.5f} (S={S},T={T},TOPK={TOPK},kc={kc},{nv_id})"


# ---- Multi-token-per-core: S (256) > num_cores (110) => each core processes 2-3 tokens.
#      Guards the per-token CB lifecycle (Q_IN/max/sum/out must be clean between tokens). ----
@run_for_blackhole()
@pytest.mark.parametrize("TOPK,kc", [(64, 64), (64, 32), (128, 32)], ids=["1chunk", "2chunk", "4chunk"])
def test_sparse_sdpa_multitoken(device, TOPK, kc):
    H, S, T = 32, 256, 512
    q, kv, indices = _make_inputs(H, S, T, TOPK, lambda s: 1 + (s * 5) % TOPK)
    out, scale = _run_op(q, kv, indices, device, kc)
    golden = _golden(q, kv, indices, scale).to(torch.float32)
    pcc = torch.corrcoef(torch.stack([out.flatten().float(), golden.flatten()]))[0, 1].item()
    assert pcc >= 0.99, f"PCC {pcc:.5f} (S={S},T={T},TOPK={TOPK},kc={kc})"


# ---- Arbitrary head count: H = Sqt*32 query tile-rows processed as one subblock. The upper bound is
#      per-core L1 (flash state + Q scale with H), so it depends on k_chunk (see the aggressive-block test).
#      Exercises all_valid + a mid-tile boundary mask, multi-token/core (S>cores guards the cb_q_in pop). ----
@run_for_blackhole()
@pytest.mark.parametrize("H", [32, 64, 96, 128], ids=["h32", "h64", "h96", "h128"])
@pytest.mark.parametrize(
    "nv_fn,nv_id",
    [(lambda s: 10**9, "all_valid"), (lambda s: 1 + (s * 3) % 50, "mid_tile_boundary")],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_sparse_sdpa_heads(device, H, nv_fn, nv_id):
    S, T, TOPK, kc = 256, 512, 128, 64  # 2 chunks (Skt=2); S>cores => multi-token/core (guards cb_q_in pop)
    q, kv, indices = _make_inputs(H, S, T, TOPK, nv_fn)
    out, scale = _run_op(q, kv, indices, device, kc)
    golden = _golden(q, kv, indices, scale).to(torch.float32)
    pcc = torch.corrcoef(torch.stack([out.flatten().float(), golden.flatten()]))[0, 1].item()
    assert pcc >= 0.99, f"PCC {pcc:.5f} (H={H},{nv_id})"


# ---- Aggressive blocking: k_chunk=32 (smallest) minimizes the per-chunk K/score L1, freeing room for more
#      query tile-rows. H=192 fits at k_chunk=32 but OOMs at k_chunk=256 (measured max H: 96@256, 192@32). ----
@run_for_blackhole()
@pytest.mark.parametrize(
    "nv_fn,nv_id",
    [(lambda s: 10**9, "all_valid"), (lambda s: 1 + (s * 3) % 50, "mid_tile_boundary")],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_sparse_sdpa_heads_aggressive_block(device, nv_fn, nv_id):
    H, S, T, TOPK, kc = 192, 64, 512, 128, 32  # k_chunk=32 (Skt=1) => 4 chunks; large H fits via tiny per-chunk L1
    q, kv, indices = _make_inputs(H, S, T, TOPK, nv_fn)
    out, scale = _run_op(q, kv, indices, device, kc)
    golden = _golden(q, kv, indices, scale).to(torch.float32)
    pcc = torch.corrcoef(torch.stack([out.flatten().float(), golden.flatten()]))[0, 1].item()
    assert pcc >= 0.99, f"PCC {pcc:.5f} (H={H},kc={kc},{nv_id})"


# ---- Perf-only (no golden; full-size golden gather is ~GBs). Profile with:
#      python -m tracy -p -r -v -m pytest <thisfile>::test_sparse_sdpa_perf
#      then read "DEVICE KERNEL DURATION [ns]" for SparseSDPAOperation. ----
# nv (valid keys/token) patterns. Chunk-skip benefit is sparsity-dependent, so sweep it.
_NV = {
    "dense": lambda s, T, K: K,  # all TOPK valid (no skip) -> worst case, proves no regression
    "half": lambda s, T, K: K // 2,
    "causal": lambda s, T, K: min(s + 1, K),  # realistic prefill: position p has p+1 candidates
    "sparse": lambda s, T, K: 256,
    "mixed": lambda s, T, K: 1 + (s * 7) % K,  # earlier arbitrary distribution (for before/after compare)
}


@run_for_blackhole()
@pytest.mark.parametrize(
    "S,T,TOPK,kc,nv",
    [
        (640, 56320, 2048, 256, "dense"),  # production shape (Q[32,640,576], KV[56320,576], idx[640,2048])
        (640, 56320, 2048, 256, "half"),
        (640, 56320, 2048, 256, "causal"),
        (640, 56320, 2048, 256, "sparse"),
        (640, 56320, 2048, 256, "mixed"),
        (110, 56320, 2048, 256, "dense"),  # 1 token/core, 8 chunks, real T -> representative DRAM locality
        (8, 56320, 2048, 256, "dense"),  # 8 cores -> low DRAM contention; isolates BW headroom vs floor
    ],
    ids=["prod-dense", "prod-half", "prod-causal", "prod-sparse", "prod-mixed", "zone1tok", "lowcore"],
)
def test_sparse_sdpa_perf(device, S, T, TOPK, kc, nv):
    H = 32
    nv_fn = _NV[nv]
    q, kv, indices = _make_inputs(H, S, T, TOPK, lambda s: nv_fn(s, T, TOPK))
    out, _ = _run_op(q, kv, indices, device, kc)  # no golden; correctness covered by PCC tests
    assert tuple(out.shape) == (1, H, S, V_DIM)
