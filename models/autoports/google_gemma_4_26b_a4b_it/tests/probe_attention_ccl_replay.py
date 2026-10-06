# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolate attention-shaped cast -> native RS -> AG exact replay."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gpt_oss.tt.ccl import CCLManager


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--dtype", choices=["bfloat16", "bfloat8_b"], required=True)
    p.add_argument("--precast", action="store_true")
    p.add_argument("--seeds", type=int, default=128)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=4 * 1024**2)
    result = dict(
        dtype=a.dtype,
        precast=a.precast,
        seeds=a.seeds,
        repeats=a.repeats,
        failures=[],
        passed=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    try:
        mapper = ttnn.ShardTensorToMesh(mesh, dim=-1)
        ccl = CCLManager(mesh, 1, ttnn.Topology.Linear)
        config = MeshConfig(mesh.shape, decode=ModeConfig(tp=4))

        def sample(seed):
            generator = torch.Generator().manual_seed(seed)
            return torch.randn(1, 1, 1, 2816 * 4, generator=generator)

        input_dtype = getattr(ttnn, a.dtype) if a.precast else ttnn.float32
        x = ttnn.from_torch(
            sample(0),
            dtype=input_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        def forward():
            return config.allreduce(x if a.precast else ttnn.typecast(x, getattr(ttnn, a.dtype)), ccl, axis=1)

        def read(t):
            return [ttnn.to_torch(v).float() for v in ttnn.get_device_tensors(t)]

        for _ in range(2):
            y = forward()
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        y = forward()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        for seed in range(a.seeds):
            host = ttnn.from_torch(sample(seed), dtype=input_dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
            ttnn.copy_host_to_device_tensor(host, x)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            reference = read(y)
            for repeat in range(a.repeats):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                actual = read(y)
                for rank, (want, got) in enumerate(zip(reference, actual)):
                    if not torch.equal(want, got):
                        result["failures"].append(
                            dict(
                                seed=seed,
                                repeat=repeat,
                                rank=rank,
                                changed_elements=int(torch.count_nonzero(want != got)),
                                max_abs_diff=float((want - got).abs().max()),
                            )
                        )
                if result["failures"]:
                    break
            if result["failures"]:
                break
        result["completed_seeds"] = seed + 1
        result["passed"] = not result["failures"]
        ttnn.release_trace(mesh, trace)
        a.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        assert result["passed"], result["failures"]
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
