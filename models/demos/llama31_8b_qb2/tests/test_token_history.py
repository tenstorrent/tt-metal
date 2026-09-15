# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The device history cursor must record whole rows and wrap without a host reset."""

import torch

import ttnn
from models.demos.llama31_8b_qb2.tt.token_history import history_program


def test_history_wraps_on_device(qb2_mesh):
    def device(value):
        return ttnn.from_torch(
            value,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=qb2_mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(qb2_mesh),
        )

    tokens = device(torch.zeros(1, 1, 1, 32, dtype=torch.int32))
    index = device(torch.zeros(1, 1, 1, 32, dtype=torch.int32))
    history = device(torch.zeros(1, 1, 3, 32, dtype=torch.int32))
    program = history_program(tokens, index, history)
    expected = torch.zeros(3, 32, dtype=torch.int64)
    for step in range(5):
        row = torch.arange(32, dtype=torch.int32) + 100 * step
        host = ttnn.from_torch(
            row.reshape(1, 1, 1, 32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(qb2_mesh),
        )
        ttnn.copy_host_to_device_tensor(host, tokens)
        ttnn.generic_op([tokens, index, history], program)
        expected[step % 3] = row
    for replica in ttnn.get_device_tensors(history):
        assert torch.equal(ttnn.to_torch(replica).reshape(3, 32).long(), expected)
    for replica in ttnn.get_device_tensors(index):
        assert ttnn.to_torch(replica).flatten()[0].item() == 2
