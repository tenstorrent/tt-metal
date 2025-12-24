# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from models.common.auto_compose import to_torch_auto_compose


@torch.no_grad()
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16,),
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
def test_topk(dtype, mesh_device):
    torch_input = torch.load(f"tt_logits_torch_8_actual_3264_expected_21862.pt")

    sub_core_grid_topk = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
        ]
    )

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(3, None),
            mesh_shape=(8, 4),
        ),
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_topk_values, tt_topk_indices = ttnn.topk(
        tt_input, k=32, dim=-1, sub_core_grids=None, indices_tensor=None, sorted=True, largest=True
    )

    tt_topk_values_torch = to_torch_auto_compose(tt_topk_values)
    tt_topk_indices_torch = to_torch_auto_compose(tt_topk_indices)

    tt_per_device_values_torch = []
    tt_per_device_indices_torch = []
    for i in range(8):
        tt_per_device_values_torch.append(tt_topk_values_torch[:, :, :, i * 32 : (i + 1) * 32])
        tt_per_device_indices_torch.append(tt_topk_indices_torch[:, :, :, i * 32 : (i + 1) * 32])

    per_device_size_last_dim = torch_input.shape[-1] // 8
    reference_values_per_device = []
    reference_indices_per_device = []
    for i in range(8):
        reference_values, reference_indices = torch.topk(
            torch_input[:, :, :, i * per_device_size_last_dim : (i + 1) * per_device_size_last_dim], k=32, dim=-1
        )
        reference_values_per_device.append(reference_values)
        reference_indices_per_device.append(reference_indices)

    for i in range(8):
        breakpoint()
        assert torch.allclose(
            tt_per_device_values_torch[i].to(torch.float32), reference_values_per_device[i].to(torch.float32)
        )


@torch.no_grad()
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16,),
)
@pytest.mark.parametrize(
    "per_device_size_last_dim",
    (19456,),
)
@pytest.mark.parametrize(
    "per_device_index",
    (0,),
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (1, 1),
    ],
    indirect=True,
)
def test_topk_single_device(dtype, mesh_device, per_device_size_last_dim, per_device_index):
    torch_input = torch.load(f"tt_logits_torch_8_actual_3264_expected_21862.pt")
    torch_input = torch_input[
        :, :, :, per_device_index * per_device_size_last_dim : (per_device_index + 1) * per_device_size_last_dim
    ]

    sub_core_grid_topk = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
        ]
    )

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_topk_values, tt_topk_indices = ttnn.topk(
        tt_input, k=32, dim=-1, sub_core_grids=None, indices_tensor=None, sorted=True, largest=True
    )

    tt_topk_values_torch = to_torch_auto_compose(tt_topk_values)
    tt_topk_indices_torch = to_torch_auto_compose(tt_topk_indices)

    reference_values, reference_indices = torch.topk(torch_input, k=32, dim=-1)

    assert torch.allclose(tt_topk_values_torch.to(torch.float32), reference_values.to(torch.float32))
