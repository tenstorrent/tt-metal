# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Probe + latency of the decode-time indexer primitives on random data (no checkpoint): fused ``indexer_score_dsa`` vs a matmul composite,
``topk_large_indices`` (k=512) over N entries, for U users per chip. Prints ``IDXOPS`` lines (us per call inside a trace)."""

import time

import pytest
import torch

import ttnn

DP = {"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000}
MESH = pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
PARAMS = pytest.mark.parametrize("device_params", [pytest.param(DP, id="ring")], indirect=True)
REPS = 5


def dev(md, t, dtype, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        device=md,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )


def bench(md, fn, tag, reps=REPS):
    out = fn()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    for _ in range(reps):
        fn()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ts = []
    for _ in range(3):
        t = time.perf_counter()
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        ts.append((time.perf_counter() - t) / reps * 1e6)
    ttnn.release_trace(md, tid)
    print(f"IDXOPS {tag:70s} {min(ts):9.1f} us/call", flush=True)
    return out


@MESH
@PARAMS
@pytest.mark.parametrize(
    "users", [1]
)  # indexer_score_dsa takes ONE user per call (k batch must be 1; several users = one call per user with cache_batch_idx)
@pytest.mark.parametrize("n_entries", [2048, 8192, 32768, 131072])
@torch.no_grad()
def test_indexer_ops(mesh_device, users, n_entries):
    md, U, N = mesh_device, users, n_entries
    T = N + 32  # allocated key length (tile aligned, row 0 of the query tile must be able to see every valid key)
    H, D = 32, 128
    gen = torch.Generator().manual_seed(1)
    k = dev(md, torch.randn(U, 1, T, D, generator=gen).to(torch.bfloat16), ttnn.bfloat16)
    k8 = dev(md, torch.randn(U, 1, T, D, generator=gen).to(torch.bfloat16), ttnn.bfloat8_b)
    q = dev(
        md, torch.randn(U, H, 32, D, generator=gen).to(torch.bfloat16), ttnn.bfloat16
    )  # Sq must be a tile multiple: row 0 real, 31 padding rows
    q8 = ttnn.typecast(q, ttnn.bfloat8_b)
    w = dev(md, torch.randn(U, 1, 32, H, generator=gen).to(torch.bfloat16), ttnn.bfloat16)
    score = None
    for tag, kk, qq in (("bf16 k, bf16 q", k, q), ("bfp8 k, bfp8 q", k8, q8)):
        for hg in (0, 8):
            for kc in (128, 256, 512):
                try:
                    cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=min(kc, T), head_group_size=hg)
                    score = bench(
                        md,
                        lambda: ttnn.experimental.indexer_score_dsa(
                            qq, kk, w, chunk_start_idx=T - 32, kv_len=T, program_config=cfg
                        ),
                        f"indexer_score_dsa U={U} N={N} {tag} head_group={hg} k_chunk={kc}",
                    )
                except Exception as e:  # noqa: BLE001
                    print(
                        f"IDXOPS indexer_score_dsa U={U} N={N} {tag} hg={hg} kc={kc} FAILED: {str(e)[:200]}", flush=True
                    )
    if score is not None:
        print(f"IDXOPS score shape {score.shape} layout {score.layout} dtype {score.dtype}", flush=True)
        for kvl in (N, N // 2):
            try:
                idx = bench(
                    md,
                    lambda: ttnn.experimental.topk_large_indices(score, k=512, valid_length=kvl),
                    f"topk_large_indices k=512 U={U} T={T} valid={kvl}",
                )
            except Exception as e:  # noqa: BLE001
                print(f"IDXOPS topk_large_indices FAILED: {str(e)[:300]}", flush=True)
        vl = dev(md, torch.full((1, 1, 1, 1), N, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        try:
            bench(
                md,
                lambda: ttnn.experimental.topk_large_indices(score, k=512, valid_length_tensor=vl),
                f"topk_large_indices k=512 U={U} T={T} valid_length_tensor",
            )
        except Exception as e:  # noqa: BLE001
            print(f"IDXOPS topk(valid_length_tensor) FAILED: {str(e)[:300]}", flush=True)
    # matmul composite: relu(q_heads[32,128] @ K^T) then head-weighted sum (heads = the 32 rows of one tile: no row padding waste)
    qm = dev(md, torch.randn(U, 1, H, D, generator=gen).to(torch.bfloat16), ttnn.bfloat16)
    wm = dev(md, torch.randn(U, 1, 1, H, generator=gen).to(torch.bfloat16), ttnn.bfloat16)
    try:

        def comp():
            s = ttnn.matmul(qm, k, transpose_b=True, activation="relu", compute_kernel_config=None)  # [U,1,32,T]
            return ttnn.matmul(wm, s)  # [U,1,1(32),T]

        bench(md, comp, f"matmul composite U={U} N={N} (q K^T relu, w @ s)")
    except Exception as e:  # noqa: BLE001
        print(f"IDXOPS matmul composite FAILED: {str(e)[:300]}", flush=True)
