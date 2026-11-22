# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.FBM (INDEXER_BMAX_COMPUTE): the indexer's 32-key block maxes made on the score op's COMPUTE kernel
(ring_indexer_score_dsa block_max_compute + ttnn.bringup.block_max_bf16(strip_major=True)) vs the writer core's
(block_keys 8 fp32 + the fold, PB.BM), op level on the held perf model's plain indexer objects (no layer runs):

    (1) the fork checks on this mesh: check_block_max_compute (compute maxes == writer maxes == torch, scores
        unchanged, nothing past kv_len) and check_block_select (the strip-major transpose), byte-exact
    (2) per history h (FBM_POS, default 139264 and 1024000), a 1048576-token slot's capacity, random q / gates / keys in
        the model's dtypes, the single score form (ring_indexer_score_dsa as TtIndexer.scores calls it; MB.8: the
        batching branch's packed "ranges" form is left out, model_bringup has no lane_ranges / keys_gathered), each with no epilogue (off), the writer's maxes (w8) and the compute maxes (cmax): ms a call of the score, of
        the fold / transpose, and the bf16 [Sq, NB] maxes of w8 + fold vs cmax + transpose byte-compared on every chip
        (metric fbm_bmax_mismatch_<form>_p<h>, must be 0).
Time: FBM_REPS calls back to back between syncs, median of FBM_ITERS (DS41-385: compare within one run).

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "<the walls env> \\
        scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/perf/test_fbm_probe.py"
"""

from __future__ import annotations

import os
import statistics
import time

import pytest
import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import batch
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec

S = spec()
BATCH_SHARED = True
BATCH_LAYERS = "executed"
BATCH_ITEM_VARS = ("FBM_POS", "FBM_ITERS", "FBM_REPS", "FBM_PARTS", "FBM_EPS")


@mesh_parametrize(timeout_s=3600)
def test_fbm_probe(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.common import SP_AXIS, TP_AXIS
    from models.demos.deepseek_v41_flash_d_p.tt.cores import max_grid
    from models.demos.deepseek_v41_flash_d_p.tt.indexer import BSEL_BLOCK, SCORE_BLOCK, bsel_blocks

    parts = os.environ.get("FBM_PARTS", "check,time").split(",")
    layers = batch.run_layers(S, list(S.layers()), last_chunk=False)
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    mesh = model.mesh
    sp, tp = int(mesh.shape[SP_AXIS]), int(mesh.shape[TP_AXIS])
    n = sp * tp
    ix = next(b.indexer for b in model.blocks.values() if type(getattr(b, "indexer", None)).__name__ == "TtIndexer")
    mg = max_grid(mesh)
    if "check" in parts:
        from ttnn.bringup.block_select_ttnn.tests.unit.test_block_select import check_block_select
        from ttnn.bringup.indexer_score.tests.unit.test_block_max_compute import check_block_max_compute

        r = check_block_max_compute(
            mesh, ratio=ix.ratio, sq=64, ckc=ix.score_ckc, max_grid=None if mg is None else (mg.x, mg.y)
        )
        print("check_block_max_compute:", r)
        metrics.record("fbm_check_block_max_compute_cases", r["cases"])
        print("check_block_select:", check_block_select(mesh))
    if "time" not in parts:
        return
    R, nh, hd, ratio = ix.sq, ix.nh, ix.hd, ix.ratio
    T = 1048576 // ratio
    iters = int(os.environ.get("FBM_ITERS", "3"))
    reps = int(os.environ.get("FBM_REPS", "4"))
    pts = [int(x) for x in os.environ.get("FBM_POS", "139264,1024000").split(",")]
    q_dtype = ttnn.bfloat4_b if ix.q_bfp4 else ttnn.bfloat16

    def up(t, dtype, dims):
        return ttnn.from_torch(
            t.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(sp, tp), dims=dims),
        )

    def zeros_rep(shape, dtype, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            torch.zeros(shape),
            dtype=dtype,
            layout=layout,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    k_local = up(torch.randn(sp, 1, T // sp, hd), ix.k_dtype, (0, None))
    kbuf = zeros_rep([1, 1, T, hd], ix.k_dtype)
    nb = bsel_blocks(T)
    bm8 = zeros_rep([1, 1, R, nb * (BSEL_BLOCK // SCORE_BLOCK)], ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)
    strip = zeros_rep([1, 1, R // 32 * nb // 8, 256], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    bm_a = zeros_rep([1, 1, R, nb], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    bm_b = zeros_rep([1, 1, R, nb], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    qd = up(torch.randn(n, nh, R, hd).view(sp, tp * nh, R, hd), q_dtype, (0, 1))
    wd = up(torch.randn(n, 1, R, nh).view(sp, tp, R, nh), ttnn.bfloat16, (0, 1))
    common = dict(
        cluster_axis=SP_AXIS,
        topology=ix.topology,
        num_links=ix.num_links,
        key_compression_ratio=ratio,
        compute_kernel_config=ix.score_ckc,
        seq_subshard_axis=TP_AXIS if tp > 1 else None,
        block_cyclic_sp_axis=SP_AXIS,
        block_cyclic_chunk_local=ix.key_rows,
        natural_query_blocks=True,
        natural_query_rows=0 if ix.key_rows == ix.w_rows else ix.w_rows,
    )
    eps = {
        "off": {},
        "w8": dict(block_keys=SCORE_BLOCK, block_max_out=bm8),
        "cmax": dict(block_keys=BSEL_BLOCK, block_max_out=strip, block_max_compute=True),
    }

    def timed(fn, keep=False) -> float:
        out = fn()
        if not keep and out is not None:
            ttnn.deallocate(out)
        ts = []
        for _ in range(iters):
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(reps):
                out = fn()
                if not keep and out is not None:
                    ttnn.deallocate(out)
            ttnn.synchronize_device(mesh)
            ts.append((time.perf_counter() - t0) * 1e3 / reps)
        return statistics.median(ts)

    def chips(x, rows, cols):
        return [ttnn.to_torch(c).reshape(rows, cols) for c in ttnn.get_device_tensors(x)]

    for h in pts:
        start = min(h, T * ratio - 8192)
        kv = min(T, -(-((start + model.chunk) // ratio) // 32) * 32)
        nbv = kv // BSEL_BLOCK
        forms = ["single"]
        want_eps = os.environ.get("FBM_EPS", "off,w8,cmax").split(",")
        for form in forms:
            for ep_name, ep in eps.items():
                if ep_name not in want_eps:
                    continue

                def score(ep=ep):
                    return ttnn.bringup.ring_indexer_score_dsa(
                        qd,
                        kbuf,
                        wd,
                        k_local,
                        ix.ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=SP_AXIS),
                        chunk_start_idx=start,
                        program_config=ix.score_cfg,
                        kv_len=kv,
                        **ep,
                        **common,
                    )

                ms = timed(score)
                metrics.record(f"fbm_score_{form}_{ep_name}_ms_p{h}", round(ms, 3))
                print(f"FBM score {form} {ep_name} p{h}: {ms:.3f} ms a call (kv_len {kv})")
            if "w8" not in want_eps or "cmax" not in want_eps:
                continue
            # the maxes: one fresh call of each, then the fold / transpose timed and their outputs compared
            ttnn.deallocate(score(eps["w8"]))
            ms_f = timed(
                lambda: ttnn.bringup.block_max_bf16(bm8, factor=BSEL_BLOCK // SCORE_BLOCK, n=nbv, output_tensor=bm_a),
                keep=True,
            )
            ttnn.deallocate(score(eps["cmax"]))
            ms_t = timed(
                lambda: ttnn.bringup.block_max_bf16(strip, strip_major=True, n=nbv, output_tensor=bm_b), keep=True
            )
            a, b = chips(bm_a, R, nb), chips(bm_b, R, nb)
            bad = sum(int((x[:, :nbv].view(torch.int16) != y[:, :nbv].view(torch.int16)).sum()) for x, y in zip(a, b))
            metrics.record(f"fbm_fold_ms_p{h}_{form}", round(ms_f, 3))
            metrics.record(f"fbm_transpose_ms_p{h}_{form}", round(ms_t, 3))
            metrics.record(f"fbm_bmax_mismatch_{form}_p{h}", bad)
            print(
                f"FBM {form} p{h}: fold {ms_f:.3f} ms, transpose {ms_t:.3f} ms; maxes w8 vs cmax: {bad} of "
                f"{n * R * nbv} differ"
            )
            assert bad == 0, (form, h, bad)
    for t in (k_local, kbuf, bm8, strip, bm_a, bm_b, qd, wd):
        ttnn.deallocate(t)
