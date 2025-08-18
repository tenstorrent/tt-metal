# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
import ttnn
from ..parallel_config import VAEParallelConfig
from enum import Enum


class ConvStrategy(Enum):
    TP = 1
    DP = 2
    SP = 3


slice_params = {
    (1, 4): {
        (512, 512, 512, 64): (16, ttnn.Conv2dSliceWidth),
        (128, 128, 16, 512): (8, ttnn.Conv2dSliceWidth),
        (128, 128, 512, 512): (4, ttnn.Conv2dSliceWidth),
        (256, 256, 512, 512): (8, ttnn.Conv2dSliceWidth),
        (512, 512, 512, 512): (16, ttnn.Conv2dSliceWidth),
        (512, 512, 512, 256): (16, ttnn.Conv2dSliceWidth),
        (512, 512, 256, 256): (4, ttnn.Conv2dSliceWidth),
        (1024, 1024, 256, 256): (16, ttnn.Conv2dSliceWidth),
        (1024, 1024, 256, 128): (16, ttnn.Conv2dSliceWidth),
        (1024, 1024, 128, 128): (16, ttnn.Conv2dSliceWidth),
        (1024, 1024, 128, 3): (8, ttnn.Conv2dSliceWidth),
    },
    (2, 2): {
        (512, 512, 512, 64): (16, ttnn.Conv2dSliceWidth),
        (128, 128, 16, 512): (8, ttnn.Conv2dSliceWidth),
        (128, 128, 512, 512): (4, ttnn.Conv2dSliceWidth),
        (256, 256, 512, 512): (8, ttnn.Conv2dSliceWidth),
        (512, 512, 512, 512): (16, ttnn.Conv2dSliceWidth),
        (512, 512, 512, 256): (16, ttnn.Conv2dSliceWidth),
        (512, 512, 256, 256): (4, ttnn.Conv2dSliceWidth),
        (1024, 1024, 256, 256): (16, ttnn.Conv2dSliceWidth),
        (1024, 1024, 256, 128): (16, ttnn.Conv2dSliceWidth),
        (1024, 1024, 128, 128): (16, ttnn.Conv2dSliceWidth),
        (1024, 1024, 128, 3): (8, ttnn.Conv2dSliceWidth),
    },
    (4, 4): {
        (512, 512, 512, 64): (16, ttnn.Conv2dSliceWidth),
        (128, 128, 16, 512): (8, ttnn.Conv2dSliceWidth),
        (128, 128, 512, 512): (4, ttnn.Conv2dSliceWidth),
        (256, 256, 512, 512): (8, ttnn.Conv2dSliceWidth),
        (512, 512, 512, 512): (16, ttnn.Conv2dSliceWidth),
        (512, 512, 512, 256): (16, ttnn.Conv2dSliceWidth),
        (512, 512, 256, 256): (4, ttnn.Conv2dSliceWidth),
        (1024, 1024, 256, 256): (16, ttnn.Conv2dSliceWidth),
        (1024, 1024, 256, 128): (16, ttnn.Conv2dSliceWidth),
        (1024, 1024, 128, 128): (16, ttnn.Conv2dSliceWidth),
        (1024, 1024, 128, 3): (8, ttnn.Conv2dSliceWidth),
    },
    (2, 4): {
        (128, 128, 16, 512): (8, ttnn.Conv2dSliceWidth),
        (128, 128, 512, 512): (4, ttnn.Conv2dSliceWidth),
        (256, 256, 512, 512): (8, ttnn.Conv2dSliceWidth),
        (512, 512, 512, 512): (16, ttnn.Conv2dSliceWidth),
        (512, 512, 512, 256): (16, ttnn.Conv2dSliceWidth),
        (512, 512, 256, 256): (4, ttnn.Conv2dSliceWidth),
        (1024, 1024, 256, 256): (16, ttnn.Conv2dSliceWidth),
        (1024, 1024, 256, 128): (16, ttnn.Conv2dSliceWidth),
        (1024, 1024, 128, 128): (16, ttnn.Conv2dSliceWidth),
        (1024, 1024, 128, 3): (8, ttnn.Conv2dSliceWidth),
    },
    (1, 1): {
        (128, 128, 16, 512): (8, ttnn.Conv2dSliceWidth),
        (128, 128, 512, 512): (4, ttnn.Conv2dSliceWidth),
        (256, 256, 512, 512): (8, ttnn.Conv2dSliceWidth),
        (512, 512, 512, 512): (16, ttnn.Conv2dSliceWidth),
        (512, 512, 512, 256): (16, ttnn.Conv2dSliceWidth),
        (512, 512, 256, 256): (4, ttnn.Conv2dSliceWidth),
        (1024, 1024, 256, 256): (16, ttnn.Conv2dSliceWidth),
        (1024, 1024, 256, 128): (16, ttnn.Conv2dSliceWidth),
        (1024, 1024, 128, 128): (16, ttnn.Conv2dSliceWidth),
        (1024, 1024, 128, 3): (8, ttnn.Conv2dSliceWidth),
    },
}


def get_slice_config(mesh_device, height, width, in_channels, out_channels):
    sl_config = slice_params[tuple(mesh_device.shape)][(height, width, in_channels, out_channels)]
    return ttnn.Conv2dSliceConfig(num_slices=sl_config[0], slice_type=sl_config[1])


# Assumption. The input is replicated across mesh unless specified. Output is either replicated or sharded across mesh depending on mesh_sharded_output
# TODO: Address situations where mesh_sharded_output is True, but conv_parallel_strategy is not TP
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
    conv_parallel_strategy: ConvStrategy
    device_slice_mask: ttnn.Tensor | None
    mesh_sharded_input: bool  # Indicates the input is sharded across the mesh devices
    mesh_sharded_output: bool  # Indicates if the output should be left sharded across the mesh devices

    @classmethod
    def from_torch(
        cls,
        torch_conv: torch.nn.Module,
        *,
        dtype: ttnn.DataType | None = None,
        parallel_config: VAEParallelConfig,
        mesh_sharded_output: bool = True,
        mesh_sharded_input: bool = False,
    ) -> TtConv2dParameters:
        weight = torch_conv.state_dict()["weight"]
        bias = (
            torch_conv.state_dict()["bias"]
            if "bias" in torch_conv.state_dict()
            else torch.zeros(torch_conv.out_channels)
        )

        compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        # setup strategy
        o_c, i_c, _, _ = weight.shape
        mesh_shape = list(parallel_config.device.shape)
        device_count = mesh_shape[1]

        # configure weight distribution. Default is DP (Patch based)
        conv_parallel_strategy = ConvStrategy.DP
        w_mesh_mapper = None  # ttnn.ReplicateTensorToMesh(parallel_config.device)
        b_mesh_mapper = None  # ttnn.ReplicateTensorToMesh(parallel_config.device)
        device_slice_mask = None
        if (o_c // device_count) % 32 == 0 < (o_c // device_count):
            conv_parallel_strategy = ConvStrategy.TP
            w_mesh_mapper = ttnn.ShardTensor2dMesh(
                parallel_config.device, tuple(parallel_config.device.shape), dims=[None, 0]
            )
            b_mesh_mapper = ttnn.ShardTensor2dMesh(
                parallel_config.device, tuple(parallel_config.device.shape), dims=[None, -1]
            )
        elif False:  # ((i_c//device_count) % 32 == 0 < (i_c//device_count)) or i_c == o_c:
            conv_parallel_strategy = ConvStrategy.TP_DP
            w_mesh_mapper = ttnn.ShardTensor2dMesh(
                parallel_config.device, tuple(parallel_config.device.shape), dims=[None, 1]
            )
            b_mesh_mapper = ttnn.ShardTensor2dMesh(
                parallel_config.device, tuple(parallel_config.device.shape), dims=[None, -1]
            )
            bias = torch.cat([bias, torch.zeros((device_count - 1) * o_c)])  # Force bias only on first device
            device_slice_mask = ttnn.from_torch(
                torch.eye(i_c),
                dtype=dtype,
                mesh_mapper=w_mesh_mapper,
                layout=ttnn.TILE_LAYOUT,
                device=parallel_config.device,
            )

        return cls(
            weight=ttnn.from_torch(weight, dtype=dtype, mesh_mapper=w_mesh_mapper),
            bias=ttnn.from_torch(bias.reshape((1, 1, 1, -1)), dtype=dtype, mesh_mapper=b_mesh_mapper)
            if bias is not None
            else None,
            out_channels=torch_conv.out_channels,
            in_channels=torch_conv.in_channels,
            kernel_size=torch_conv.kernel_size,
            padding=torch_conv.padding,
            stride=torch_conv.stride,
            dilation=torch_conv.dilation,
            compute_config=compute_config,
            conv_config=None,  # conv_config,
            parallel_config=parallel_config,
            conv_parallel_strategy=conv_parallel_strategy,
            device_slice_mask=device_slice_mask,
            mesh_sharded_output=mesh_sharded_output,
            mesh_sharded_input=mesh_sharded_input,
        )


def run_conv2d(x, parameters):
    o_c = parameters.weight.shape[0]
    b, h, w, c = x.shape
    conv_config = parameters.conv_config
    slice_config = get_slice_config(
        parameters.parallel_config.device, h, w, parameters.in_channels, parameters.out_channels
    )

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

    # ttnn.synchronize_device(parameters.parallel_config.device)
    output_tensor = ttnn.reshape(output_tensor, (x.shape[0], _out_height, _out_width, output_tensor.shape[3]))
    return output_tensor


# TODO: Try out padded/unpadded variant (unpadded_all_gather_async  from utils.py)
def vae_conv2d(x, parameters):
    if parameters.mesh_sharded_input:
        x = parameters.parallel_config.vae_all_gather(x)

    if parameters.conv_parallel_strategy == ConvStrategy.TP:
        output_tensor = run_conv2d(x, parameters)

        if not parameters.mesh_sharded_output:  # If output is sharded, we need to gather the output
            output_tensor = parameters.parallel_config.vae_all_gather(output_tensor)

    elif parameters.conv_parallel_strategy == ConvStrategy.SP:
        # Get device slice
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT) @ parameters.device_slice_mask

        # TODO: Use the intermediate buffer
        output_tensor = run_conv2d(x, parameters)
        output_tensor = ttnn.reshape(
            output_tensor, (1, 1, -1, 32)
        )  # Helps resolve the issue with padding when last 2 dims are not multiple of 32
        output_tensor = ttnn.experimental.all_reduce_async(  # TODO: Move to parallel config
            input_tensor=output_tensor,
            cluster_axis=1,
            mesh_device=parameters.parallel_config.device,
            from_remote_multi_device_global_semaphore=parameters.parallel_config.reduce_from_semaphore,
            to_remote_multi_device_global_semaphore=parameters.parallel_config.reduce_to_semaphore,
            gather_multi_device_global_semaphore=parameters.parallel_config.gather_semaphore,
            math_op=ttnn.ReduceType.Sum,
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
            subdevice_id=None,
        )

        output_tensor = ttnn.reshape(output_tensor, (x.shape[0], x.shape[1], x.shape[2], parameters.weight.shape[0]))

    elif parameters.conv_parallel_strategy == ConvStrategy.DP:
        output_tensor = run_conv2d(x, parameters)

    else:
        output_tensor = run_conv2d(x, parameters)
    return output_tensor
