# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One captured topology derivation follows changing device start metadata."""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params
from models.demos.deepseek_v3_d_p.tt.kda.device_chronology import rank_tensor


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        ((2, 4), fabric_1d_device_params(trace_region_size=1024 * 1024)),
        ((1, 8), fabric_1d_device_params(trace_region_size=1024 * 1024)),
    ],
    indirect=True,
)
@pytest.mark.parametrize("sp_axis", [0, 1])
def test_device_chronology(mesh_device, device_params, sp_axis):
    shape = tuple(mesh_device.shape)
    p = shape[sp_axis]
    rows = 640
    start = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ranks = rank_tensor(mesh_device, sp_axis)

    def run():
        return ttnn.experimental.kda.chronological_topology(start, ranks, p, rows, 4, 128, 128)

    for _ in range(2):
        ttnn.deallocate(run())
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    result = run()
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    try:
        for s in range(0, 2 * p * rows + 32, 32):
            source = ttnn.from_torch(
                torch.tensor([s], dtype=torch.int32),
                device=mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            ttnn.copy(source, start)
            ttnn.deallocate(source)
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            for i, shard in enumerate(ttnn.get_device_tensors(result)):
                rank = (i // shape[1]) if sp_axis == 0 else i % shape[1]
                boundary = (s // rows) % p
                split = s % rows != 0 and p > 1
                expected = [
                    boundary,
                    rows - s % rows,
                    int(split and rank == boundary),
                    rank,
                    boundary if split else (boundary + p - 1) % p,
                    int(split),
                    rows,
                    0,
                ]
                got = ttnn.to_torch(shard).to(torch.int64)
                assert got[0].tolist() == expected
                end = expected[1] if expected[2] else rows
                assert got[1, :3].tolist() == list(range(end - 3, end))
                for step in range(p):
                    assert got[8 + step * 2, 0].item() == (boundary + step) % p
        print("CHRONOLOGY_DYNAMIC_REPLAY_PASS")
    finally:
        ttnn.release_trace(mesh_device, trace)
