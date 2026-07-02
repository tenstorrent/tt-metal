# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""sparse_sdpa (sparse MLA prefill) — NIGHTLY full coverage (Blackhole only).

Comprehensive sweeps over shapes, sparsity, head count, query sub-blocking, and fp8 q/kv, plus a perf-only
case on the production shape. The fast post-commit smoke is in
tests/ttnn/unit_tests/operations/sdpa/test_sparse_sdpa.py (shared helpers in sparse_sdpa_test_utils.py).
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

# Open the device ONCE per module (not per test) — all tests share one device config, so this avoids the
# ~1.7s open/close on every parametrized case. The program cache persists across tests, which is fine: the
# recompile/indexed tests clear_program_cache() at their start, and the hash keeps distinct configs separate.
pytestmark = pytest.mark.use_module_device


# ---- SINGLE-CHUNK PCC (k_chunk == TOPK) vs sparse_mla golden, across shapes + sparsity ----
@run_for_blackhole()
@pytest.mark.parametrize("S,T,TOPK", [(64, 256, 32), (64, 512, 64), (32, 1024, 128)])
@pytest.mark.parametrize(
    "nv_fn,nv_id",
    [(lambda s: 10**9, "all_valid"), (lambda s: 1 + (s % 7), "few_valid"), (lambda s: 1 + (s * 3) % 20, "boundary")],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_sparse_sdpa_pcc_single_chunk(device, S, T, TOPK, nv_fn, nv_id):
    H = 32
    q, kv, indices = make_inputs(H, S, T, TOPK, K_DIM, nv_fn)
    out, scale = run_op(q, kv, indices, device, k_chunk_size=TOPK, v_dim=V_DIM)  # single chunk
    p = pcc(out, golden(q, kv, indices, scale, V_DIM))
    assert p >= 0.99, f"PCC {p:.5f} (S={S},T={T},TOPK={TOPK},{nv_id})"


# ---- multi-chunk PCC vs sparse_mla golden (needs flash accumulation), across shapes + sparsity ----
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
    q, kv, indices = make_inputs(H, S, T, TOPK, K_DIM, nv_fn)
    out, scale = run_op(q, kv, indices, device, kc, V_DIM)
    p = pcc(out, golden(q, kv, indices, scale, V_DIM))
    assert p >= 0.99, f"PCC {p:.5f} (S={S},T={T},TOPK={TOPK},kc={kc},{nv_id})"


# ---- Multi-token-per-core: S (256) > num_cores (110) => each core processes 2-3 tokens.
#      Guards the per-token CB lifecycle (Q_IN/max/sum/out must be clean between tokens). ----
@run_for_blackhole()
@pytest.mark.parametrize("TOPK,kc", [(64, 64), (64, 32), (128, 32)], ids=["1chunk", "2chunk", "4chunk"])
def test_sparse_sdpa_multitoken(device, TOPK, kc):
    H, S, T = 32, 256, 512
    q, kv, indices = make_inputs(H, S, T, TOPK, K_DIM, lambda s: 1 + (s * 5) % TOPK)
    out, scale = run_op(q, kv, indices, device, kc, V_DIM)
    p = pcc(out, golden(q, kv, indices, scale, V_DIM))
    assert p >= 0.99, f"PCC {p:.5f} (S={S},T={T},TOPK={TOPK},kc={kc})"


# ---- Arbitrary head count: H = Sqt*32 query tile-rows processed in DST-sized groups. The upper bound is
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
    q, kv, indices = make_inputs(H, S, T, TOPK, K_DIM, nv_fn)
    out, scale = run_op(q, kv, indices, device, kc, V_DIM)
    p = pcc(out, golden(q, kv, indices, scale, V_DIM))
    assert p >= 0.99, f"PCC {p:.5f} (H={H},{nv_id})"


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
    q, kv, indices = make_inputs(H, S, T, TOPK, K_DIM, nv_fn)
    out, scale = run_op(q, kv, indices, device, kc, V_DIM)
    p = pcc(out, golden(q, kv, indices, scale, V_DIM))
    assert p >= 0.99, f"PCC {p:.5f} (H={H},kc={kc},{nv_id})"


# ---- fp8 K/V cache: kv stored as FP8_E4M3 (halves the dominant K-gather bytes). Compute tilizes fp8 ->
#      bfp8_b with a 32-bit dest (auto-enabled for fp8). Golden uses the fp8-quantized kv. ----
@run_for_blackhole()
@pytest.mark.parametrize("S,T,TOPK,kc", [(64, 512, 128, 64), (32, 1024, 256, 128)])
@pytest.mark.parametrize(
    "nv_fn,nv_id",
    [(lambda s: 10**9, "all_valid"), (lambda s: 1 + (s * 3) % 50, "mid_tile_boundary")],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_sparse_sdpa_fp8_kv(device, S, T, TOPK, kc, nv_fn, nv_id):
    H = 32
    q, kv, indices = make_inputs(H, S, T, TOPK, K_DIM, nv_fn)
    out, scale = run_op(q, kv, indices, device, kc, V_DIM, kv_dtype=ttnn.fp8_e4m3)
    kv_fp8 = kv.to(torch.float8_e4m3fn).to(torch.float32)  # what the device holds after fp8 quantization
    p = pcc(out, golden(q, kv_fp8, indices, scale, V_DIM))
    assert p >= 0.99, f"PCC {p:.5f} (fp8 kv, S={S},T={T},TOPK={TOPK},kc={kc},{nv_id})"


# ---- fp8 Q: q stored as FP8_E4M3 and tilized to bfp8_b cb_q_in. Q is the QK matmul's srcB; the kernel
#      restores srcB's bf16 format after the QK. Sweeps fp8 q alone and fp8 q + fp8 kv (both srcs bfp8). ----
@run_for_blackhole()
@pytest.mark.parametrize("S,T,TOPK,kc", [(64, 512, 128, 64), (32, 1024, 256, 128)])
@pytest.mark.parametrize("kv_dtype,kv_id", [(ttnn.bfloat16, "kv_bf16"), (ttnn.fp8_e4m3, "kv_fp8")])
def test_sparse_sdpa_fp8_q(device, S, T, TOPK, kc, kv_dtype, kv_id):
    H = 32
    q, kv, indices = make_inputs(H, S, T, TOPK, K_DIM, lambda s: 1 + (s * 3) % 50)
    out, scale = run_op(q, kv, indices, device, kc, V_DIM, kv_dtype=kv_dtype, q_dtype=ttnn.fp8_e4m3)
    q_fp8 = q.to(torch.float8_e4m3fn).to(torch.float32)  # what the device holds after fp8 quantization
    kv_g = kv.to(torch.float8_e4m3fn).to(torch.float32) if kv_dtype == ttnn.fp8_e4m3 else kv
    p = pcc(out, golden(q_fp8, kv_g, indices, scale, V_DIM))
    assert p >= 0.99, f"PCC {p:.5f} (fp8 q, {kv_id}, S={S},T={T},TOPK={TOPK},kc={kc})"


# ---- Query sub-blocking: when Sqt = H/32 exceeds dst_size, the compute kernel processes the query
#      tile-rows in DST-sized groups (qsb = largest divisor of Sqt <= dst_size). fp32_dest_acc_en halves
#      dst_size (8 -> 4), forcing multi-group at modest H: H=192 (Sqt=6, qsb=3, 2 groups) and H=160
#      (Sqt=5, qsb=1, 5 groups — the qsb=1 stress path). Without fp32-acc these would be one group. ----
@run_for_blackhole()
@pytest.mark.parametrize("H,groups", [(192, 2), (160, 5)], ids=["h192_2grp", "h160_5grp"])
def test_sparse_sdpa_query_subblock(device, H, groups):
    S, T, TOPK, kc = 64, 512, 128, 32  # k_chunk=32 keeps per-chunk L1 small so the large H fits
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    q, kv, indices = make_inputs(H, S, T, TOPK, K_DIM, lambda s: 1 + (s * 3) % 50)
    out, scale = run_op(q, kv, indices, device, kc, V_DIM, compute_kernel_config=cfg)
    p = pcc(out, golden(q, kv, indices, scale, V_DIM))
    assert p >= 0.99, f"PCC {p:.5f} (H={H}, {groups} query groups, fp32_dest_acc)"


# ---- Determinism: identical inputs must yield BIT-EXACT-identical output across repeated runs. Efficient by
#      design — cheap inputs, NO golden, and the comparison runs ON DEVICE: ttnn.ne -> ttnn.max reduces each
#      iteration's diff to a 1-element marker (accumulated with ttnn.maximum), so only that scalar is pulled to
#      host, not the full output every iteration. Dense multi-chunk exercises the flash accumulation +
#      reciprocal (softmax normalize) path, the determinism-sensitive part; bf16 + fp8. ----
@run_for_blackhole()
@pytest.mark.parametrize(
    "q_dtype,kv_dtype",
    [(ttnn.bfloat16, ttnn.bfloat16), (ttnn.fp8_e4m3, ttnn.fp8_e4m3)],
    ids=["bf16", "fp8"],
)
def test_sparse_sdpa_determinism(device, q_dtype, kv_dtype):
    H, S, T, TOPK, kc, iters = 32, 128, 512, 256, 64, 10  # dense, 4 chunks/token; S>cores => some 2 tokens/core
    q, kv, indices = make_inputs(H, S, T, TOPK, K_DIM, lambda s: TOPK)
    q_host = q.to(torch.bfloat16) if q_dtype == ttnn.bfloat16 else q.to(torch.float32)
    kv_host = kv.to(torch.bfloat16) if kv_dtype == ttnn.bfloat16 else kv.to(torch.float32)
    tt_q = to_dev(q_host, device, q_dtype)  # upload ONCE; rerun the cached program on the same input bits
    tt_kv = to_dev(kv_host, device, kv_dtype)
    tt_idx = to_dev(indices.to(torch.int32), device, ttnn.uint32)

    def comparable(o):  # ttnn.ne/max need TILE bf16 (fp8 has no eltwise); both casts are deterministic 1:1 maps
        if o.dtype == ttnn.fp8_e4m3:
            o = ttnn.typecast(o, ttnn.bfloat16)
        return ttnn.to_layout(o, ttnn.TILE_LAYOUT)

    ref, marker = None, None
    for _ in range(iters):
        cur = comparable(ttnn.transformer.sparse_sdpa(tt_q, tt_kv, tt_idx, V_DIM, scale=K_DIM**-0.5, k_chunk_size=kc))
        if ref is None:
            ref = cur
        else:
            m = ttnn.max(ttnn.ne(ref, cur, dtype=ttnn.bfloat16))  # 0 iff bit-exact
            marker = m if marker is None else ttnn.maximum(marker, m)
    mismatch = float(ttnn.to_torch(ttnn.from_device(marker)).max())  # one scalar pulled to host, once
    assert mismatch == 0.0, "sparse_sdpa output is not deterministic across repeated runs"


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
    q, kv, indices = make_inputs(H, S, T, TOPK, K_DIM, lambda s: nv_fn(s, T, TOPK))
    out, _ = run_op(q, kv, indices, device, kc, V_DIM)  # no golden; correctness covered by the PCC tests
    assert tuple(out.shape) == (1, H, S, V_DIM)
