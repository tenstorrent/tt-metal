# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from ...utils.check import assert_quality
from ...layers.conv2d import Conv2d
from ...parallel.manager import CCLManager
from ...parallel.config import vae_all_gather


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 20000000}],
    indirect=True,
)
@pytest.mark.parametrize(
    (
        "batch",
        "height",
        "width",
        "in_channels",
        "out_channels",
        "kernel",
        "padding",
        "stride",
        "shard_input",
        "mesh_axis",
    ),
    [
        (1, 128, 128, 16, 512, 3, 1, 1, False, 1),
        (1, 128, 128, 512, 512, 3, 1, 1, True, 1),
        (1, 256, 256, 512, 512, 3, 1, 1, True, 1),
        (1, 512, 512, 512, 512, 3, 1, 1, True, 1),
        (1, 512, 512, 512, 256, 3, 1, 1, True, 1),
        (1, 512, 512, 256, 256, 3, 1, 1, True, 1),
        (1, 1024, 1024, 256, 256, 3, 1, 1, True, 1),
        (1, 1024, 1024, 256, 128, 3, 1, 1, True, 1),
        (1, 1024, 1024, 128, 128, 3, 1, 1, True, 1),
        (1, 1024, 1024, 128, 3, 3, 1, 1, True, None),
    ],
)
def test_conv2d(
    *,
    mesh_device: ttnn.Device,
    batch: int,
    height: int,
    width: int,
    in_channels: int,
    out_channels: int,
    kernel: int,
    padding: int,
    stride: int,
    shard_input: bool,
    mesh_axis: int,
) -> None:
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)

    # models
    torch_model = torch.nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, padding=padding, stride=stride
    )
    torch.nn.init.normal_(torch_model.weight)
    torch.nn.init.normal_(torch_model.bias)
    torch_model.eval()

    tt_model = Conv2d.from_torch(
        torch_ref=torch_model, mesh_device=mesh_device, mesh_axis=mesh_axis, ccl_manager=ccl_manager
    )

    torch_input = torch.randn(batch, in_channels, height, width)

    tt_input_tensor = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1) if shard_input else None,
        layout=ttnn.TILE_LAYOUT,
    )

    with torch.no_grad():
        torch_output = torch_model(torch_input)

    tt_out = tt_model(tt_input_tensor)

    if mesh_axis is not None:
        tt_out = vae_all_gather(ccl_manager, tt_out)

    tt_final_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).permute(0, 3, 1, 2)
    assert_quality(torch_output, tt_final_out_torch, pcc=0.999_500)
