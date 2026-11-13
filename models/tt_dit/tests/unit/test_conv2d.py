# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

from ...layers.conv2d import Conv2d, configure_conv2d
from ...parallel.config import vae_all_gather
from ...parallel.manager import CCLManager
from ...utils.check import assert_quality


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
        (1, 1024, 1024, 128, 3, 3, 1, 1, False, None),
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
        torch_ref=torch_model, mesh_device=mesh_device, out_mesh_axis=mesh_axis, ccl_manager=ccl_manager
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
        "mesh_axis",
    ),
    [
        (1, 128, 128, 16, 512, 3, 1, 1, 1),
        (1, 128, 128, 512, 512, 3, 1, 1, 1),
        (1, 256, 256, 512, 512, 3, 1, 1, 1),
        (1, 512, 512, 512, 512, 3, 1, 1, 1),
        (1, 512, 512, 512, 256, 3, 1, 1, 1),
        (1, 512, 512, 256, 256, 3, 1, 1, 1),
        (1, 1024, 1024, 256, 256, 3, 1, 1, 1),
        (1, 1024, 1024, 256, 128, 3, 1, 1, 1),
        (1, 1024, 1024, 128, 128, 3, 1, 1, 1),
    ],
)
def test_conv2d_input_parallel(
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
        torch_ref=torch_model, mesh_device=mesh_device, in_mesh_axis=mesh_axis, ccl_manager=ccl_manager
    )

    torch_input = torch.randn(batch, in_channels, height, width)

    tt_input_tensor = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        layout=ttnn.TILE_LAYOUT,
    )

    with torch.no_grad():
        torch_output = torch_model(torch_input)

    tt_out = tt_model(tt_input_tensor)

    tt_out = vae_all_gather(ccl_manager, tt_out)

    tt_final_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).permute(0, 3, 1, 2)
    assert_quality(torch_output, tt_final_out_torch, pcc=0.999_500)


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 20000000}],
    indirect=True,
)
@pytest.mark.parametrize(
    ("in_channels", "out_channels", "size_a", "size_b", "extra_slices"),
    [
        # 16 slices at 1024x1024 (shared table), 40 at 720x1280 (added for this instance). Weights
        # prepared for 1024x1024 give NaN at 720x1280 (job 171948).
        (256, 128, (1024, 1024), (720, 1280), 40),
        # Shared table at both sizes: 4 then 8 slices.
        (512, 512, (128, 128), (256, 256), None),
    ],
    ids=["instance_table", "shared_table"],
)
def test_conv2d_input_size_change(
    *,
    mesh_device: ttnn.Device,
    in_channels: int,
    out_channels: int,
    size_a: tuple[int, int],
    size_b: tuple[int, int],
    extra_slices: int | None,
) -> None:
    """One Conv2d instance run at size A, then B, then A again, with a different slice count at
    each size. Weights prepared for one size can give wrong results at another (on the Ideogram 4
    VAE, 32 px output stripes), so every output must still match torch, and returning to A must
    reuse A's prepared weights."""
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)

    torch_model = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
    torch.nn.init.normal_(torch_model.weight)
    torch.nn.init.normal_(torch_model.bias)
    torch_model.eval()

    tt_model = Conv2d.from_torch(
        torch_ref=torch_model, mesh_device=mesh_device, out_mesh_axis=1, ccl_manager=ccl_manager
    )
    if extra_slices is not None:
        mesh_shape = tuple(mesh_device.shape)
        configure_conv2d(tt_model, slice_params={mesh_shape: {(*size_b, in_channels, out_channels): extra_slices}})

    for height, width in (size_a, size_b, size_a):
        torch_input = torch.randn(1, in_channels, height, width)
        tt_input = ttnn.from_torch(
            torch_input.permute(0, 2, 3, 1),
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
            layout=ttnn.TILE_LAYOUT,
        )

        with torch.no_grad():
            torch_output = torch_model(torch_input)

        tt_out = vae_all_gather(ccl_manager, tt_model(tt_input))
        tt_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).permute(0, 3, 1, 2)
        assert_quality(torch_output, tt_out_torch, pcc=0.999_500)

    assert len(tt_model._prepared) == 2
