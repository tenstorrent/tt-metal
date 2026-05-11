# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Trace capture / replay on the full system mesh (all user-visible devices).

Tensors are replicated across every device in the ``mesh_device`` fixture
(system mesh when not parametrized; see ``conftest.py::mesh_device``).

Inside the trace, a minimal forward pass (two linear blocks with ReLU) runs
many times—same pattern as a tiny inference loop, not a single op in a loop.
"""

import pytest
import torch

import ttnn


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 1996800, "dispatch_core_type": ttnn.DispatchCoreType.WORKER}],
    indirect=True,
)
def test_with_ops_all_devices_mesh(mesh_device):
    torch.manual_seed(0)
    dim = 512
    torch_x = torch.randn((dim, dim), dtype=torch.bfloat16)
    torch_w1 = torch.randn((dim, dim), dtype=torch.bfloat16)
    torch_w2 = torch.randn((dim, dim), dtype=torch.bfloat16)

    def replicate_bf16(t):
        return ttnn.from_torch(
            t,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.distributed.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    x = replicate_bf16(torch_x)
    w1 = replicate_bf16(torch_w1)
    w2 = replicate_bf16(torch_w2)

    grid = ttnn.CoreGrid(y=8, x=8)

    def inference_step():
        h = ttnn.matmul(x, w1, core_grid=grid)
        h = ttnn.relu(h)
        return ttnn.matmul(h, w2, core_grid=grid)

    inference_step()
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for _ in range(100):
        inference_step()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)

    for _ in range(5):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    ttnn.release_trace(mesh_device, tid)
