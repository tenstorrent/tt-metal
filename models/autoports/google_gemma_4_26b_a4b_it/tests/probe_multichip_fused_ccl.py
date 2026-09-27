# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""WO -> distributed norm/residual or distributed norm -> QKV CCL probe.

Synthetic shape-faithful component evidence, not real-weight or whole-layer evidence.
Run one candidate per process; reset devices after a failed multichip process.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

import ttnn


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--family", choices=["mm_rs", "ag_mm"], required=True)
    parser.add_argument("--layer", choices=["sliding_attention", "full_attention"], default="sliding_attention")
    parser.add_argument("--topology", choices=["linear", "ring"], default="linear")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--grid-x", type=int, default=8)
    parser.add_argument("--block-k", type=int, default=2)
    parser.add_argument("--out-block-w", type=int)
    parser.add_argument("--subblock-w", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(73)
    torch.set_num_threads(4)
    dtype = ttnn.float32 if args.dtype == "fp32" else ttnn.bfloat16
    h, local_h, rows = 2816, 704, 32
    k = 1024 if args.layer == "sliding_attention" else 2048
    n = 2048 if args.layer == "sliding_attention" else 3072
    eps = 1e-6
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING if args.topology == "ring" else ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=16 * 1024**2)
    try:
        grid = mesh.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        manager = mesh.create_sub_device_manager([ttnn.SubDevice([cores])], 0)
        mesh.load_sub_device_manager(manager)
        mesh.set_sub_device_stall_group([ttnn.SubDeviceId(0)])
        compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        def upload(t, shard=None, dt=dtype):
            mapper = ttnn.ReplicateTensorToMesh(mesh) if shard is None else ttnn.ShardTensorToMesh(mesh, dim=shard)
            return ttnn.from_torch(
                t,
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=dt,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        def sems(count):
            return [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(count)]

        # Distinct ownership per operation; each full invocation resets its semaphores internally.
        rs_sems, ag_sems, stats_sems = sems(3), sems(2), sems(2)
        rs_barrier, ag_barrier, stats_barrier = sems(3)
        common = dict(
            dim=3,
            num_links=1,
            topology=ttnn.Topology.Ring if args.topology == "ring" else ttnn.Topology.Linear,
            subdevice_id=ttnn.SubDeviceId(0),
        )

        def gather(x, handles, barrier):
            return ttnn.experimental.all_gather_async(
                x,
                multi_device_global_semaphore=handles,
                barrier_semaphore=barrier,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                **common,
            )

        def norm(x):
            stats = ttnn.rms_norm_pre_all_gather(x, dtype=ttnn.float32, compute_kernel_config=compute)
            stats = gather(stats, stats_sems, stats_barrier)
            return ttnn.rms_norm_post_all_gather(x, stats, epsilon=eps, compute_kernel_config=compute)

        out_width = h if args.family == "mm_rs" else n
        per_n = (out_width // 32 + args.grid_x - 1) // args.grid_x
        kwargs = {} if args.out_block_w is None else {"out_block_w": args.out_block_w}
        config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(args.grid_x, 6),
            in0_block_w=args.block_k,
            out_subblock_h=1,
            out_subblock_w=args.subblock_w,
            per_core_M=1,
            per_core_N=per_n,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
            **kwargs,
        )
        mm_kwargs = dict(program_config=config, compute_kernel_config=compute, dtype=dtype)
        residual = upload(torch.randn(1, 1, rows, h).bfloat16(), 3)
        if args.family == "mm_rs":
            inp = upload(torch.randn(1, 1, rows, k * 4).bfloat16(), 3)
            weights = upload(torch.randn(1, 1, k * 4, h).bfloat16() / (k * 4) ** 0.5, 2, ttnn.bfloat8_b)
            intermediate = upload(torch.zeros(1, 1, rows, h))
            rs_buffer = upload(torch.zeros(1, 1, rows, local_h))

            def run(fused):
                if fused:
                    _, reduced = ttnn.experimental.matmul_reduce_scatter_async(
                        inp,
                        weights,
                        persistent_intermediate_buffer=intermediate,
                        persistent_output_buffer=rs_buffer,
                        multi_device_global_semaphore=rs_sems,
                        barrier_semaphore=rs_barrier,
                        reduce_scatter_core_grid_offset=(0, 6),
                        memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
                        memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
                        **common,
                        **mm_kwargs,
                    )
                else:
                    product = ttnn.matmul(inp, weights, memory_config=ttnn.DRAM_MEMORY_CONFIG, **mm_kwargs)
                    reduced = ttnn.experimental.reduce_scatter_minimal_async(
                        product,
                        multi_device_global_semaphore=rs_sems,
                        barrier_semaphore=rs_barrier,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        **common,
                    )
                # Consume H/4 directly: no hidden all-gather before norm or residual.
                return ttnn.add(residual, norm(reduced))

        else:
            inp = upload(torch.randn(1, 1, rows, h).bfloat16(), 3)
            weights = upload(torch.randn(1, 1, h, n * 4).bfloat16() / h**0.5, 3, ttnn.bfloat8_b)
            gathered_buffer = upload(torch.zeros(1, 1, rows, h))

            def run(fused):
                normalized = ttnn.typecast(norm(inp), dtype)
                if fused:
                    _, projected = ttnn.experimental.all_gather_matmul_async(
                        normalized,
                        weights,
                        persistent_output_buffer=gathered_buffer,
                        multi_device_global_semaphore=ag_sems,
                        barrier_semaphore=ag_barrier,
                        all_gather_core_grid_offset=(0, 6),
                        memory_config_ag=ttnn.DRAM_MEMORY_CONFIG,
                        memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
                        **common,
                        **mm_kwargs,
                    )
                    return projected
                gathered = gather(normalized, ag_sems, ag_barrier)
                return ttnn.matmul(gathered, weights, memory_config=ttnn.DRAM_MEMORY_CONFIG, **mm_kwargs)

        def host(x):
            return ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=3)).float()

        print("UNFUSED_REFERENCE_BEGIN", flush=True)
        reference = host(run(False))
        print("UNFUSED_REFERENCE_PASS", flush=True)
        print("FUSED_CANDIDATE_BEGIN", flush=True)
        candidate = host(run(True))
        print("FUSED_CANDIDATE_RETURNED", flush=True)
        pcc = torch.corrcoef(torch.stack([reference.flatten(), candidate.flatten()]))[0, 1].item()
        assert pcc >= 0.995, pcc
        results = {}
        for fused in (False, True):
            for _ in range(2):
                output = run(fused)
            ttnn.synchronize_device(mesh)
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            output = run(fused)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            times = []
            for _ in range(args.iterations):
                start = time.perf_counter_ns()
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                times.append((time.perf_counter_ns() - start) / 1000)
            first = host(output)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            assert torch.equal(first, host(output)), "Trace replay nondeterminism"
            ttnn.release_trace(mesh, trace)
            results["fused" if fused else "unfused"] = {"trace_host_us": statistics.median(times)}
        packet = dict(
            vars(args),
            passed=True,
            pcc=pcc,
            measurements=results,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scope="synthetic component, whole producer/consumer window; host timing, not device timing",
        )
        Path(args.output).write_text(json.dumps(packet, indent=2) + "\n")
        print(json.dumps(packet), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
