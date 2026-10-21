# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""LM-head candidates for the decode terminal path (traced, profiler off) and their logit error.

Input: the real normalized final hidden state of a 48-layer prefill is not needed for a precision comparison of
the head alone, so the probe takes real-scale inputs: real token embeddings (rows of the tied table, unscaled) mixed
like a normalized hidden state, against the float32 host product with the BF16 checkpoint weight.

  python models/autoports/google_gemma_4_12b_it/tests/lm_head_probe.py --out doc/full_model/logs/lm_head_probe.json
"""

import argparse
import json
import statistics
import time

import torch

import ttnn
from models.autoports.google_gemma_4_12b_it.tests.hf_reference import pcc
from models.autoports.google_gemma_4_12b_it.tt.model import TILE, Checkpoint
from models.autoports.google_gemma_4_12b_it.tt.multichip_decoder import close_qb2_mesh, open_qb2_mesh
from models.autoports.google_gemma_4_12b_it.tt.optimized_decoder import FIDELITY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(16)
    emb = Checkpoint().tensor("embed_tokens.weight")  # [V, H] bf16
    V, H = emb.shape
    g = torch.Generator().manual_seed(0)
    # hidden: a normalized mixture of real embedding rows (the LM head sees RMS-normalized states)
    mix = emb[torch.randint(0, V, (32, 8), generator=g)].float().sum(1)
    h = mix * torch.rsqrt(mix.pow(2).mean(-1, keepdim=True) + 1e-6) * 3.0
    want = h.to(torch.bfloat16).float() @ emb.float().T  # [32, V]
    mesh = open_qb2_mesh(trace_region_size=0)
    res = {}
    try:
        grid = mesh.compute_with_storage_grid_size()
        banks = mesh.dram_grid_size().x
        n = V // 4

        def width_sharded(width, cores):
            crs = ttnn.num_cores_to_corerangeset(cores, grid, row_wise=True)
            return ttnn.create_sharded_memory_config(
                (TILE, width // cores),
                core_grid=crs,
                strategy=ttnn.ShardStrategy.WIDTH,
                use_height_and_width_as_shard_shape=True,
            )

        res_mem = width_sharded(H, 15)
        x = ttnn.from_torch(
            h.reshape(1, 1, 32, H),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=res_mem,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        dram_sharded = ttnn.from_torch(
            emb.T.contiguous(),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
                    (H, n // banks),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
        )
        interleaved = ttnn.from_torch(
            emb.T.contiguous(),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
        )
        x_dram = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        def ck(name):
            fid, fp32, l1 = FIDELITY[name]
            return ttnn.init_device_compute_kernel_config(
                mesh.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=l1
            )

        variants = {}
        x_l1 = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        workers = banks * 2
        cfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=1,
            per_core_M=1,
            per_core_N=n // TILE // workers,
            fused_activation=None,
            num_workers_per_dram_bank=2,
        )
        out_mem = width_sharded(n, workers)
        variants["dram_sharded_r2_hifi2"] = lambda: ttnn.linear(
            x,
            dram_sharded,
            program_config=cfg,
            compute_kernel_config=ck("hifi2"),
            memory_config=out_mem,
            dtype=ttnn.bfloat16,
        )
        cores = grid.x * grid.y
        nt = n // TILE
        for in0w in (4, 8, 16, 40):
            for src, xin in (("dram", x_dram), ("l1", x_l1)):
                for gx, gy in ((grid.x, grid.y),):
                    per_core_n = -(-nt // (gx * gy))
                    cfg1d = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                        in0_block_w=in0w,
                        out_subblock_h=1,
                        out_subblock_w=1,
                        out_block_h=1,
                        out_block_w=per_core_n,
                        per_core_M=1,
                        per_core_N=per_core_n,
                        fuse_batch=True,
                        fused_activation=None,
                        mcast_in0=True,
                    )
                    variants[f"mcast1d_in0w{in0w}_{src}_hifi2"] = lambda cfg1d=cfg1d, xin=xin: ttnn.linear(
                        xin,
                        interleaved,
                        program_config=cfg1d,
                        compute_kernel_config=ck("hifi2"),
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        dtype=ttnn.bfloat16,
                    )
        for name, fn in variants.items():
            try:
                out = fn()
                got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=3)).float().reshape(32, -1)
                ttnn.deallocate(out)
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                o = fn()
                ttnn.deallocate(o)
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                times = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    for _ in range(a.reps):
                        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    times.append((time.perf_counter() - t0) / a.reps * 1e6)
                ttnn.release_trace(mesh, tid)
                agree = float((got.argmax(-1) == want.argmax(-1)).float().mean())
                top5 = sum(
                    len(set(got[r].topk(5).indices.tolist()) & set(want[r].topk(5).indices.tolist())) for r in range(32)
                ) / (32 * 5)
                res[name] = {
                    "us": round(statistics.median(times), 1),
                    "pcc": round(pcc(want, got), 6),
                    "max_abs_err": round(float((want - got).abs().max()), 4),
                    "argmax_agree": agree,
                    "top5_overlap": round(top5, 4),
                }
            except Exception as e:  # noqa: BLE001
                res[name] = {"error": str(e).splitlines()[0][:300]}
            print(name, res[name], flush=True)
    finally:
        close_qb2_mesh(mesh)
    print("LMHEAD", json.dumps(res), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
