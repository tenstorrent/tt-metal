# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One captured selection derivation follows changing actual_start."""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        ((2, 4), fabric_1d_device_params(trace_region_size=1024 * 1024)),
        ((1, 8), fabric_1d_device_params(trace_region_size=1024 * 1024)),
    ],
    indirect=True,
)
@pytest.mark.parametrize("sp_axis", [0, 1])
def test_chronological_selections(mesh_device, device_params, sp_axis):
    shape = tuple(mesh_device.shape)
    p = shape[sp_axis]
    rows = 640
    actual_start = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int64),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def run():
        return ttnn.experimental.kda.chronological_selections(actual_start, sp_axis, rows, 4, 128, 128)

    for _ in range(2):
        ttnn.deallocate(run())
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    result = run()
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    try:
        for actual_start_value in [*range(0, 2 * p * rows + 32, 32), 2**32 - 32]:
            source = ttnn.from_torch(
                torch.tensor([actual_start_value], dtype=torch.int64),
                device=mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            ttnn.copy(source, actual_start)
            ttnn.deallocate(source)
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            for i, shard in enumerate(ttnn.get_device_tensors(result)):
                rank = (i // shape[1]) if sp_axis == 0 else i % shape[1]
                first_rank = (actual_start_value // rows) % p
                split = actual_start_value % rows != 0 and p > 1
                got = ttnn.to_torch(shard).to(torch.int64)
                assert tuple(got.shape) == (7 + 2 * p, 8)
                end = rows - actual_start_value % rows if split and rank == first_rank else rows
                assert got[0, :3].tolist() == list(range(end - 3, end))
                for step in range(p):
                    assert got[7 + step * 2, 0].item() == (first_rank + step) % p
        print("CHRONOLOGY_DYNAMIC_REPLAY_PASS")
    finally:
        ttnn.release_trace(mesh_device, trace)
