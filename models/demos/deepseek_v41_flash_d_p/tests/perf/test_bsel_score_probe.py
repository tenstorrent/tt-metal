# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.BM (INDEXER_BLOCK_SELECT): the cost of the score op's block-max epilogue (block_max_out, block_keys 8, no pin)
in the two ring forms the model runs, op level on the held perf model's plain indexer objects (no layer runs):

    single   ring_indexer_score_dsa as TtIndexer.scores calls it (the model's q chunk, natural blocks, one start)
    ranges   the glued one-call ring (lane_ranges, q chunk 32): four chip-aligned 2048-row segments at history h,
             each its own key buffer (one capacity)

at histories h (BSEL_SCORE_POS, default 139264 and 1024000), the capacity of a 1048576-token slot; random q / gates /
keys in the model's dtypes. Time: ms a call, BSEL_SCORE_REPS calls back to back between syncs, median of
BSEL_SCORE_ITERS; metrics bsel_score_<form>_<off|max>_ms_p<h>.

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "<the walls env> \\
        scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/perf/test_bsel_score_probe.py"
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
BATCH_ITEM_VARS = ("BSEL_SCORE_POS", "BSEL_SCORE_ITERS", "BSEL_SCORE_REPS")
RANGE_STRIDE = 41


@mesh_parametrize(timeout_s=3600)
def test_bsel_score_probe(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.common import SP_AXIS, TP_AXIS
    from models.demos.deepseek_v41_flash_d_p.tt.indexer import K_CHUNK, SCORE_BLOCK

    layers = batch.run_layers(S, list(S.layers()), last_chunk=False)
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    mesh = model.mesh
    sp, tp = int(mesh.shape[SP_AXIS]), int(mesh.shape[TP_AXIS])
    n = sp * tp
    ix = next(b.indexer for b in model.blocks.values() if type(getattr(b, "indexer", None)).__name__ == "TtIndexer")
    R, nh, hd, ratio = ix.sq, ix.nh, ix.hd, ix.ratio
    T = 1048576 // ratio
    iters = int(os.environ.get("BSEL_SCORE_ITERS", "3"))
    reps = int(os.environ.get("BSEL_SCORE_REPS", "4"))
    pts = [int(x) for x in os.environ.get("BSEL_SCORE_POS", "139264,1024000").split(",")]
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

    k_locals = [up(torch.randn(sp, 1, T // sp, hd), ix.k_dtype, (0, None)) for _ in range(4)]
    kbufs = [zeros_rep([1, 1, T, hd], ix.k_dtype) for _ in range(4)]
    nb8 = -(-(T // SCORE_BLOCK) // 128) * 128
    bm = zeros_rep([1, 1, R, nb8], ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)
    qd = up(torch.randn(n, nh, R, hd).view(sp, tp * nh, R, hd), q_dtype, (0, 1))
    wd = up(torch.randn(n, 1, R, nh).view(sp, tp, R, nh), ttnn.bfloat16, (0, 1))
    cfg32 = ttnn.bringup.IndexerScoreProgramConfig(
        q_chunk_size=32,
        k_chunk_size=K_CHUNK,
        head_group_size=0,
        max_grid_x=ix.score_cfg.max_grid_x,
        max_grid_y=ix.score_cfg.max_grid_y,
    )
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

    def timed(fn) -> float:
        fn()
        ts = []
        for _ in range(iters):
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(reps):
                ttnn.deallocate(fn())
            ttnn.synchronize_device(mesh)
            ts.append((time.perf_counter() - t0) * 1e3 / reps)
        return statistics.median(ts)

    res = {}
    for h in pts:
        start = min(h, T * ratio - 8192)
        kv = min(T, -(-((start + model.chunk) // ratio) // 32) * 32)
        # four chip-aligned 2048-row segments at history h (each 8 chips), each its own key buffer
        chips_per = n // 4
        words = []
        for q in range(n):
            j = q // chips_per
            words += [1, 0, R, j, start + (q % chips_per) * R, kv] + [0] * (RANGE_STRIDE - 6)
        for ep_name, ep in (("off", {}), ("max", dict(block_keys=SCORE_BLOCK, block_max_out=bm))):

            def single(ep=ep):
                return ttnn.bringup.ring_indexer_score_dsa(
                    qd,
                    kbufs[0],
                    wd,
                    k_locals[0],
                    ix.ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=SP_AXIS),
                    chunk_start_idx=start,
                    program_config=ix.score_cfg,
                    kv_len=kv,
                    **ep,
                    **common,
                )

            def ranges(ep=ep):
                return ttnn.bringup.ring_indexer_score_dsa(
                    qd,
                    kbufs[0],
                    wd,
                    k_locals[0],
                    ix.ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=SP_AXIS),
                    chunk_start_idx=0,
                    program_config=cfg32,
                    kv_len=kv,
                    k_extra=kbufs[1:4],
                    k_local_extra=k_locals[1:4],
                    lane_ranges=words,
                    **ep,
                    **common,
                )

            for form, fn in (("single", single), ("ranges", ranges)):
                ms = timed(fn)
                res[(form, ep_name, h)] = ms
                metrics.record(f"bsel_score_{form}_{ep_name}_ms_p{h}", round(ms, 3))
                print(f"BSEL_SCORE {form} {ep_name} p{h}: {ms:.3f} ms a call (kv_len {kv})")
    for t in k_locals + kbufs + [bm, qd, wd]:
        ttnn.deallocate(t)
