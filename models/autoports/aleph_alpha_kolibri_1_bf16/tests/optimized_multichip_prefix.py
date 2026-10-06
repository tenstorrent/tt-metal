# SPDX-License-Identifier: Apache-2.0
"""Exhaustive supported aligned-count values; never feed inexact offsets to MoE."""

import json
import statistics
import time

import torch

import ttnn

from .multichip_sweep import OUT


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=10000000)
    rows = []
    try:
        counts = (torch.arange(32 * 384).reshape(1, 1, 32, 384) % 257 * 32).float()
        expected = counts.cumsum(-1)

        def tt(x):
            return ttnn.from_torch(
                x,
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        a = tt(counts)
        w = tt(torch.triu(torch.ones(384, 384)))
        for fidelity in ["HiFi4", "HiFi2", "LoFi"]:
            for cores in [12, 6, 3]:
                pn = 12 // cores
                program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(cores if cores <= 6 else 4, 1 if cores <= 6 else 3),
                    in0_block_w=12,
                    out_subblock_h=1,
                    out_subblock_w=min(pn, 4),
                    per_core_M=1,
                    per_core_N=pn,
                    fuse_batch=False,
                    mcast_in0=True,
                )
                compute = ttnn.WormholeComputeKernelConfig(
                    math_fidelity=getattr(ttnn.MathFidelity, fidelity),
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=True,
                )
                for location in ["DRAM", "L1"]:
                    x = ttnn.to_memory_config(a, ttnn.L1_MEMORY_CONFIG) if location == "L1" else a

                    def call():
                        return ttnn.matmul(
                            x,
                            w,
                            dtype=ttnn.float32,
                            program_config=program,
                            compute_kernel_config=compute,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )

                    output = call()
                    errors = [float((ttnn.to_torch(t) - expected).abs().max()) for t in ttnn.get_device_tensors(output)]
                    del output
                    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                    output = call()
                    ttnn.end_trace_capture(mesh, tid, cq_id=0)
                    for _ in range(3):
                        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                    rounds = []
                    for _ in range(3):
                        start = time.monotonic()
                        for _ in range(100):
                            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        rounds.append((time.monotonic() - start) * 10)
                    ms = statistics.median(rounds)
                    ttnn.release_trace(mesh, tid)
                    row = dict(
                        fidelity=fidelity,
                        cores=cores,
                        input_memory=location,
                        max_abs_error=errors,
                        exact=not any(errors),
                        decode_ms=ms,
                        rounds_ms=rounds,
                        kblock=12,
                        subblock=min(pn, 4),
                    )
                    rows.append(row)
                    print(json.dumps(row), flush=True)
        (OUT / "prefix_integer_probe.json").write_text(
            json.dumps(dict(mesh=[1, 4], count_values=list(range(0, 8193, 32)), rows=rows), indent=2) + "\n"
        )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
