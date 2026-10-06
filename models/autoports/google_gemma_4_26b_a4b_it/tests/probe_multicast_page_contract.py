# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Watcher regression for scatter and single-page sampler multicast gathers."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

import ttnn
from models.common.modules.tt_ccl import get_tt_ccl
from models.common.sampling.tt_sampling import TTSampling

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--dtypes", nargs="+", choices=("bfloat16", "uint32"), default=["bfloat16", "uint32"])
parser.add_argument("--ring", action="store_true")
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
rows = []
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING if args.ring else ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=10000000)
trace = None
try:
    ccl = get_tt_ccl(mesh)
    sampler_gather_state = SimpleNamespace(_line_all_gather=getattr(ccl, "line_all_gather", None))
    assert sampler_gather_state._line_all_gather is None
    for name in args.dtypes:
        print(f"GATHER_START dtype={name} local_shape=[1,1,32,32] ring={args.ring}", flush=True)
        cpu = torch.arange(128, dtype=torch.int32).reshape(1, 1, 1, 128).expand(1, 1, 32, 128).clone()
        if name == "bfloat16":
            cpu = cpu.to(torch.bfloat16)
        tensor = ttnn.from_torch(
            cpu,
            dtype=getattr(ttnn, name),
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
        )

        def gather():
            return TTSampling._perform_all_gather(
                sampler_gather_state,
                tensor,
                dim=3,
                cluster_axis=None,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                num_links=1,
            )

        def check(out, expected, phase):
            checks = [
                torch.equal(ttnn.to_torch(t).to(torch.int32), expected.to(torch.int32))
                for t in ttnn.get_device_tensors(out)
            ]
            row = dict(
                dtype=name,
                page_bytes=2048 if name == "bfloat16" else 4096,
                local_shape=[1, 1, 32, 32],
                ring=args.ring,
                phase=phase,
                devices_passed=checks,
                passed=all(checks),
            )
            rows.append(row)
            args.output.write_text(json.dumps(rows, indent=2) + "\n")
            print(row, flush=True)
            assert row["passed"], row

        warmed = gather()
        ttnn.synchronize_device(mesh)
        check(warmed, cpu, "eager")
        ttnn.deallocate(warmed)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        traced = gather()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        for step in range(2):
            expected = cpu + 32 * (step + 1)
            host = ttnn.from_torch(
                expected,
                dtype=getattr(ttnn, name),
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
            )
            ttnn.copy_host_to_device_tensor(host, tensor)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            check(traced, expected, f"traced_{step}")
        ttnn.release_trace(mesh, trace)
        trace = None
        ttnn.deallocate(traced)
        ttnn.deallocate(tensor)
finally:
    if trace is not None:
        ttnn.release_trace(mesh, trace)
    ttnn.close_mesh_device(mesh)
    print("DEVICES_CLOSED", flush=True)
