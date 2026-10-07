# SPDX-License-Identifier: Apache-2.0
"""Decode matmul sweep on the mesh (per device shapes): qkv [B,4096]x[4096,1920] and o_proj [B,1536]x[1536,4096], bfp8 interleaved weights. default ttnn.linear vs the 1D mcast_in0 program config
on N cores (qwen36 tp_common recipe); traced time per matmul (50 repeats per trace) and agreement with the default result."""
import json
import math
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import close_mesh, open_mesh_full  # noqa: E402

TILE = 32


def prog_1d(m, k, n, cores, fp32_acc=True, grid_w=8):
    cols = min(grid_w, cores)
    rows = math.ceil(cores / cols)
    m_t, k_t, n_t = math.ceil(m / TILE), math.ceil(k / TILE), math.ceil(n / TILE)
    per_core_k = max(i for i in range(1, k_t + 1) if k_t % i == 0 and i <= 8)  # largest divisor <= 8 tiles
    per_core_n = math.ceil(n_t / (cols * rows))
    cap = 4 if fp32_acc else 8
    sub_w = max(i for i in range(1, cap + 1) if per_core_n % i == 0)
    sub_h = max(i for i in range(1, cap + 1) if m_t % i == 0 and i * sub_w <= cap)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(cols, rows),
        in0_block_w=per_core_k,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        per_core_M=m_t,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def main():
    mesh = open_mesh_full()
    try:
        ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        g = torch.Generator().manual_seed(0)
        rep = ttnn.ReplicateTensorToMesh(mesh)
        for B in (8, 32):
            for name, K, N in (("qkv", 4096, 1920), ("o_proj", 1536, 4096)):
                x = torch.randn(1, 1, B, K, generator=g).bfloat16()
                w = (torch.randn(1, 1, K, N, generator=g) / 64).bfloat16()
                xt = ttnn.from_torch(x, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep)
                wt = ttnn.from_torch(w, device=mesh, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep)
                ref = ttnn.to_torch(
                    ttnn.get_device_tensors(ttnn.linear(xt, wt, dtype=ttnn.bfloat16, compute_kernel_config=ckc))[0]
                ).float()
                for cores in [0, 8, 12, 15, 16, 20, 24, 30, 32, 40, 60, 64]:
                    try:
                        if cores == 0:
                            f = lambda: ttnn.linear(
                                xt,
                                wt,
                                dtype=ttnn.bfloat16,
                                compute_kernel_config=ckc,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            )
                        else:
                            pc = prog_1d(B, K, N, cores)
                            f = lambda pc=pc: ttnn.linear(
                                xt,
                                wt,
                                dtype=ttnn.bfloat16,
                                compute_kernel_config=ckc,
                                program_config=pc,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            )
                        o = f()
                        d = ttnn.to_torch(ttnn.get_device_tensors(o)[0]).float()
                        ttnn.synchronize_device(mesh)
                        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                        for _ in range(50):
                            ttnn.deallocate(f())
                        ttnn.end_trace_capture(mesh, tid, cq_id=0)
                        ts = []
                        for _ in range(6):
                            t0 = time.perf_counter()
                            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                            ts.append(time.perf_counter() - t0)
                        ttnn.release_trace(mesh, tid)
                        us = round(sorted(ts)[2] / 50 * 1e6, 1)
                        err = float((d - ref).abs().max())
                        print(
                            "MM_RESULT",
                            json.dumps(dict(B=B, op=name, cores=cores, us=us, max_abs_diff_vs_default=err)),
                            flush=True,
                        )
                    except Exception as e:  # noqa
                        print("MM_RESULT", json.dumps(dict(B=B, op=name, cores=cores, error=str(e)[:120])), flush=True)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
