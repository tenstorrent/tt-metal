# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
import ttnn
from loguru import logger
import time
from ..parallel_config import VAEParallelConfig

slice_params = {
    (2, 2): {
        (128, 128, 16, 512): 8,
        (128, 128, 512, 512): 4,
        (256, 256, 512, 512): 8,
        (512, 512, 512, 512): 16,
        (512, 512, 512, 256): 16,
        (512, 512, 256, 256): 4,
        (1024, 1024, 256, 256): 16,
        (1024, 1024, 256, 128): 16,
        (1024, 1024, 128, 128): 16,
        (1024, 1024, 128, 3): 8,
    },
}


def get_slice_config(mesh_device, height, width, in_channels, out_channels):
    num_slices = 32
    try:
        num_slices = slice_params[tuple(mesh_device.shape)][(height, width, in_channels, out_channels)]
    except Exception as e:
        logger.debug(f"Error encountered. Using defaut num_slices: {num_slices}")

    return ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceWidth, num_slices=num_slices)


@dataclass
class TtConv2dParameters:
    weight: ttnn.Tensor
    bias: ttnn.Tensor | None
    in_channels: int
    out_channels: int
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int] | tuple[int, int, int, int]
    stride: tuple[int, int]
    dilation: tuple[int, int]
    compute_config: ttnn.DeviceComputeKernelConfig
    conv_config: ttnn.Conv2dConfig
    parallel_config: VAEParallelConfig
    multi_device: bool

    @classmethod
    def from_torch(
        cls,
        torch_conv: torch.nn.Module,
        *,
        dtype: ttnn.DataType | None = None,
        parallel_config: VAEParallelConfig,
    ) -> TtConv2dParameters:
        weight = torch_conv.state_dict()["weight"]
        bias = torch_conv.state_dict()["bias"]

        compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        multi_device = weight.shape[0] > 3  # heuristic for the last conv2d layer.

        return cls(
            weight=ttnn.from_torch(
                weight,
                dtype=dtype,
                mesh_mapper=(ttnn.ShardTensorToMesh(parallel_config.device, dim=0) if multi_device else None),
            ),
            # bias=ttnn.from_torch(bias.reshape((1, 1, 1, -1)), dtype=dtype,mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0)),
            bias=ttnn.from_torch(
                bias.reshape((1, 1, 1, -1)),
                dtype=dtype,
                mesh_mapper=(ttnn.ShardTensorToMesh(parallel_config.device, dim=-1) if multi_device else None),
            ),
            out_channels=torch_conv.out_channels,
            in_channels=torch_conv.in_channels,
            kernel_size=torch_conv.kernel_size,
            padding=torch_conv.padding,
            stride=torch_conv.stride,
            dilation=torch_conv.dilation,
            compute_config=compute_config,
            conv_config=None,  # conv_config,
            parallel_config=parallel_config,
            multi_device=multi_device,
        )


def vae_conv2d(x, parameters):
    b, h, w, c = x.shape
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    logger.info(f" CONV: In shape: {x.shape}, channels: {parameters.out_channels}")

    distr_st = time.time()

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(parameters.parallel_config.device)):
        x = ttnn.to_device(x, parameters.parallel_config.device)

    conv_st = time.time()

    # TODO: compute optimal slice config per height or width.
    slice_config = get_slice_config(
        parameters.parallel_config.device, h, w, parameters.in_channels, parameters.out_channels
    )
    o_c, _, _, _ = parameters.weight.shape
    # slice_config = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceWidth, num_slices=min(256, w // 2))
    conv_config = parameters.conv_config

    output_tensor, [_out_height, _out_width] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=parameters.weight,
        bias_tensor=parameters.bias,
        in_channels=c,
        out_channels=o_c,  # parameters.out_channels//4,
        device=parameters.parallel_config.device,
        kernel_size=parameters.kernel_size,
        stride=parameters.stride,
        padding=parameters.padding,
        batch_size=b,
        input_height=h,
        input_width=w,
        conv_config=conv_config,
        compute_config=parameters.compute_config,
        # memory_config=ttnn.DRAM_MEMORY_CONFIG,
        slice_config=slice_config,
        return_output_dim=True,
    )

    reshap_st = time.time()
    # print(f" output layout: {output_tensor.layout}")
    # output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.reshape(output_tensor, (x.shape[0], _out_height, _out_width, output_tensor.shape[3]))
    # print(f"output_tensor.shape: {output_tensor.shape}")
    # call all gather. For now, we assume the original device and see.
    # output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    # gather from all mesh. See if we need to call synchronize
    # sync_st = time.time()

    # composer = ttnn.concat_mesh_to_tensor_composer(mesh_device=parameters.device,dim=3)
    # output_tensor = ttnn.aggregate_tensor(output_tensor,composer)
    if not parameters.multi_device:
        return output_tensor

    gather_st = time.time()
    for mesh_dim in [1, 0]:  # Assuming a 2d mesh
        output_tensor = ttnn.experimental.all_gather_async(
            input_tensor=output_tensor,
            dim=3,
            multi_device_global_semaphore=parameters.parallel_config.ccl_global_semaphore,
            topology=ttnn.Topology.Linear,
            mesh_device=parameters.parallel_config.device,
            cluster_axis=mesh_dim,
        )

    # sync_st = time.time()
    ttnn.synchronize_device(parameters.parallel_config.device)

    print(f"dist_time: {conv_st - distr_st}")
    print(f"conv_time: {reshap_st- conv_st}")
    print(f"reshap_time: {gather_st - reshap_st}")
    # print(f"sync_time: {time.time()-sync_st}")
    print(f"Gather tim: {time.time() - gather_st}")
    return output_tensor
