# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolate the linear all-gather endpoint under watcher; run one case per process."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.multichip_decoder import _MeshCCLManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, choices=[1, 2], default=1)
    parser.add_argument("--dtype", choices=["bfloat16", "bfloat8_b"], default="bfloat16")
    parser.add_argument("--memory", choices=["L1", "DRAM"], default="L1")
    parser.add_argument("--persistent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--trace-replays", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    dtype = getattr(ttnn, args.dtype)
    memory = getattr(ttnn, args.memory + "_MEMORY_CONFIG")
    shape = [1, args.channels, args.rows, 704]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=16777216)
    trace_id = None
    try:
        mesh.enable_program_cache()
        manager = _MeshCCLManager(mesh, 1, ttnn.Topology.Linear)
        host_parts = [
            (torch.arange(torch.tensor(shape).prod().item()).reshape(shape) % 4 + rank * 4).to(torch.bfloat16)
            for rank in range(4)
        ]
        value = ttnn.from_torch(
            torch.cat(host_parts, dim=0),
            device=mesh,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            memory_config=memory,
        )
        actual_parts = [ttnn.to_torch(part).float() for part in ttnn.get_device_tensors(value)]
        reference = torch.cat(actual_parts, dim=-1)
        output_shape = shape[:-1] + [2816]
        output = (
            ttnn.empty(output_shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=memory)
            if args.persistent
            else None
        )

        def gather():
            return ttnn.experimental.all_gather_async(
                value,
                dim=3,
                cluster_axis=1,
                mesh_device=mesh,
                topology=ttnn.Topology.Linear,
                multi_device_global_semaphore=manager.get_ag_ping_pong_semaphore(),
                barrier_semaphore=manager.get_barrier_semaphore(),
                persistent_output_tensor=output,
                num_links=1,
                num_workers_per_link=args.workers,
                memory_config=memory,
            )

        def check(result):
            mismatches = [
                int((ttnn.to_torch(part).float() != reference).sum()) for part in ttnn.get_device_tensors(result)
            ]
            print("AG_MISMATCHES", mismatches, flush=True)
            assert mismatches == [0, 0, 0, 0], mismatches
            return mismatches

        print("AG_START", vars(args), flush=True)
        eager = gather()
        ttnn.synchronize_device(mesh)
        eager_mismatches = check(eager)
        # Warm both ping-pong signatures before capture.
        second = gather()
        ttnn.synchronize_device(mesh)
        check(second)
        replay_mismatches = []
        if args.trace_replays:
            trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
            traced = gather()
            ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
            ttnn.synchronize_device(mesh)
            check(traced)
            for _ in range(args.trace_replays):
                ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                replay_mismatches.append(check(traced))
        kernel = Path(
            "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_writer.cpp"
        )
        report = dict(
            case={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            mesh=[1, 4],
            local_input_shape=shape,
            output_shape=output_shape,
            kernel_sha256=hashlib.sha256(kernel.read_bytes()).hexdigest(),
            watcher_environment={key: os.environ.get(key) for key in ["TT_METAL_WATCHER", "TT_METAL_WATCHER_NOINLINE"]},
            eager_mismatches=eager_mismatches,
            replay_mismatches=replay_mismatches,
            passed=True,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("AG_PASS", args.output, flush=True)
    finally:
        if trace_id is not None:
            ttnn.release_trace(mesh, trace_id)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
