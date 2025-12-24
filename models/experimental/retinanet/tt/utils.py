# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import ttnn
import tt_lib.fallback_ops as fallback_ops
from dataclasses import replace
from typing import Optional

from models.tt_cnn.tt.builder import (
    TtConv2d,
    Conv2dConfiguration,
    MaxPool2dConfiguration,
    AutoShardedStrategyConfiguration,
    L1FullSliceStrategyConfiguration,
)
from ttnn.dot_access import make_dot_access_dict
from ttnn.torch_tracer import trace, visualize

conv_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
    "WEIGHTS_DTYPE": ttnn.bfloat16,
    "ACTIVATIONS_DTYPE": ttnn.bfloat16,
}


class MaxPoolConfiguration(MaxPool2dConfiguration):
    @classmethod
    def from_model_args(cls, maxpool2d_args, **kwargs):
        return cls(
            input_height=maxpool2d_args.input_height,
            input_width=maxpool2d_args.input_width,
            channels=maxpool2d_args.input_channels,
            batch_size=maxpool2d_args.batch_size,
            kernel_size=(maxpool2d_args.kernel_size, maxpool2d_args.kernel_size),
            stride=(maxpool2d_args.stride, maxpool2d_args.stride),
            padding=(maxpool2d_args.padding, maxpool2d_args.padding),
            dilation=(maxpool2d_args.dilation, maxpool2d_args.dilation),
            **kwargs,
        )


def override_conv_config(config, override_dict):
    """Apply override dictionary to Conv2dConfiguration using dataclasses.replace"""
    if not isinstance(config, Conv2dConfiguration):
        return config
    return replace(config, **override_dict)


class Conv2dNormActivation:
    """Conv2d followed by GroupNorm and ReLU activation - reusable building block."""

    def __init__(
        self,
        parameters: dict,
        device: ttnn.Device,
        in_channels: int = 256,
        out_channels: int = 256,
        kernel_size: tuple = (3, 3),
        stride: tuple = (1, 1),
        padding: tuple = (1, 1),
        num_groups: int = 32,
        grid_size: Optional[ttnn.CoreGrid] = None,
        input_mask: Optional[ttnn.Tensor] = None,
        model_config: dict = None,
        compute_config: Optional[ttnn.DeviceComputeKernelConfig] = None,
        conv_config: dict = None,
        batch_size: int = 1,
        input_height: int = 64,
        input_width: int = 64,
    ):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_groups = num_groups
        self.model_config = model_config
        self.compute_config = compute_config
        self.input_height = input_height
        self.input_width = input_width

        self.conv_weight = parameters["weight"]
        self.conv_bias = parameters["bias"]
        self.norm_weight = parameters["norm_weight"]
        self.norm_bias = parameters["norm_bias"]
        self.norm_weight = ttnn.to_device(self.norm_weight, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.norm_bias = ttnn.to_device(self.norm_bias, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        self.fallback_on_groupnorm = os.environ.get("FALLBACK_ON_GROUPNORM", "1") == "1"
        self.grid_size = grid_size if grid_size is not None else ttnn.CoreGrid(y=8, x=8)
        self.input_mask = input_mask

        base_conv_config = _create_conv_config_from_params(
            input_height=input_height,
            input_width=input_width,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            batch_size=batch_size,
            parameters=parameters,
            stride=self.stride,
            padding=self.padding,
        )

        if conv_config:
            self.conv_config = override_conv_config(base_conv_config, conv_config)
        else:
            self.conv_config = base_conv_config

        self.conv = TtConv2d(self.conv_config, self.device)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x, [H_out, W_out] = self.conv(x, return_output_dim=True)
        N, H_out, W_out, C = x.shape

        if self.fallback_on_groupnorm:
            if ttnn.is_sharded(x):
                x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)

            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x = ttnn.reshape(x, (N, self.input_height, self.input_width, C))
            x = ttnn.permute(x, (0, 3, 1, 2))

            x = fallback_ops.group_norm(
                x,
                num_groups=self.num_groups,
                weight=self.norm_weight,
                bias=self.norm_bias,
            )
            x = x.to(self.device)
            x = ttnn.permute(x, (0, 2, 3, 1))
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        else:
            if ttnn.is_sharded(x):
                x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
            spatial_size = H_out * W_out
            required_size = ((spatial_size + self.grid_size.y * 32 - 1) // (self.grid_size.y * 32)) * (
                self.grid_size.y * 32
            )

            if spatial_size != required_size:
                pad_amount = required_size - spatial_size
                x_flat = ttnn.reshape(x, (N, 1, spatial_size, C))
                x_padded = ttnn.pad(x_flat, padding=((0, 0), (0, 0), (0, pad_amount), (0, 0)), value=0.0)
            else:
                x_padded = ttnn.reshape(x, (N, 1, spatial_size, C))

            x_padded = ttnn.to_device(x_padded, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            x_normalized = ttnn.group_norm(
                x_padded,
                epsilon=1e-5,
                num_groups=self.num_groups,
                input_mask=self.input_mask,
                weight=self.norm_weight,
                bias=self.norm_bias,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                core_grid=self.grid_size,
                inplace=False,
                compute_kernel_config=self.compute_config,
            )

            if spatial_size != required_size:
                x_normalized = x_normalized[:, :, :spatial_size, :]

            x = ttnn.reshape(x_normalized, (N, self.input_height, self.input_width, C))

        x = ttnn.relu(x)
        return x


# Helper function to create Conv2dConfiguration from parameters
def _create_conv_config_from_params(
    input_height: int,
    input_width: int,
    in_channels: int,
    out_channels: int,
    batch_size: int,
    parameters: dict,
    kernel_size=(1, 1),
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
    activation=None,
    deallocate_activation=False,
    activation_dtype=None,
    weights_dtype=None,
    output_dtype=None,
    math_fidelity=None,
    fp32_dest_acc_en=False,
    packer_l1_acc=False,
    sharding_strategy=AutoShardedStrategyConfiguration(),
) -> Conv2dConfiguration:
    """
    Conv2dConfiguration from parameters dict for SqueezeExcitation.
    """

    return Conv2dConfiguration(
        input_height=input_height,
        input_width=input_width,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        dilation=dilation,
        weight=parameters["weight"],
        bias=parameters["bias"],
        activation=activation,
        activation_dtype=activation_dtype or conv_config["ACTIVATIONS_DTYPE"],
        weights_dtype=weights_dtype or conv_config["WEIGHTS_DTYPE"],
        output_dtype=output_dtype or conv_config["ACTIVATIONS_DTYPE"],
        math_fidelity=math_fidelity or conv_config["MATH_FIDELITY"],
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=packer_l1_acc,
        sharding_strategy=sharding_strategy,
        slice_strategy=L1FullSliceStrategyConfiguration(),
        enable_act_double_buffer=True,
        enable_weights_double_buffer=True,
        deallocate_activation=deallocate_activation,
        reallocate_halo_output=True,
    )


class TTUpsample:
    def __init__(
        self,
        scale_factor: int = 1,
        mode: str = "nearest",
        memory_config=ttnn.L1_MEMORY_CONFIG,
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
    ) -> None:
        self.scale_factor = scale_factor
        self.mode = mode
        self.memory_config = memory_config

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=math_fidelity,
            math_approx_mode=math_approx_mode,
            fp32_dest_acc_en=fp32_dest_acc_en,
        )

    def __call__(
        self,
        device,
        input_tensor,
        input_shape=None,
        reshape_output=False,
        pad_ch_to_32=False,
        sent_to_dram=False,
        dtype=ttnn.bfloat8_b,
    ):
        if sent_to_dram:
            input_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.DRAM_MEMORY_CONFIG)
        else:
            input_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.L1_MEMORY_CONFIG)

        input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor = ttnn.reshape(input_tensor, input_shape)

        # Optionally pad channels to a multiple of 32 to match TT tile/channel alignment.
        if pad_ch_to_32:
            input_tensor = ttnn.pad(input_tensor, [(0, 0), (0, 0), (0, 0), (0, 32 - input_tensor.shape[-1] % 32)], 0)

        output_tensor = ttnn.upsample(
            input_tensor,
            scale_factor=self.scale_factor,
            mode=self.mode,
            memory_config=self.memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )

        # Remove channel padding if added.
        if pad_ch_to_32:
            output_tensor = ttnn.slice(
                output_tensor,
                [0, 0, 0, 0],
                [output_tensor.shape[0], output_tensor.shape[1], output_tensor.shape[2], input_shape[-1]],
            )

        if reshape_output:
            B, H, W, C = output_tensor.shape
            output_tensor = ttnn.reshape(output_tensor, [1, 1, B * H * W, C])

        return output_tensor


class ModuleArgs(dict):
    ...


class Conv2dArgs(ModuleArgs):
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        return super().__repr__()


class ConvTranspose2dArgs(ModuleArgs):
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        return super().__repr__()


class MaxPool2dArgs(ModuleArgs):
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        return super().__repr__()


class GroupNormArgs(ModuleArgs):
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        return super().__repr__()


def infer_ttnn_module_args(*, model, run_model, device):
    if run_model is None:
        return None

    # Run model under TTNN tracing
    with trace():
        output = run_model(model)

    visualize(output, file_name=ttnn.CONFIG.tmp_dir / "model_graph.svg")

    # Helper: insert value into nested dict using module path
    def insert_nested(d, path, value):
        for key in path[:-1]:
            key = int(key) if isinstance(key, str) and key.isdigit() else key
            d = d.setdefault(key, {})
        last = path[-1]
        last = int(last) if isinstance(last, str) and last.isdigit() else last
        d[last] = value

    # Recursive graph walk
    def _infer_ttnn_module_args(graph):
        ttnn_module_args = {}

        for node in graph:
            attributes = graph.nodes[node]
            operation = attributes.get("operation")

            if not isinstance(operation, ttnn.tracer.TorchModule):
                continue

            module_path = operation.module.__ttnn_tracer_name__.split(".")

            in_edges = list(graph.in_edges(node, data=True))
            if not in_edges:
                continue

            input_node, _, edge_data = in_edges[0]
            input_shape = graph.nodes[input_node]["shapes"][edge_data["source_output_index"]]

            module = operation.module

            # Conv2d
            if isinstance(module, torch.nn.Conv2d):
                insert_nested(
                    ttnn_module_args,
                    module_path,
                    Conv2dArgs(
                        in_channels=module.in_channels,
                        out_channels=module.out_channels,
                        kernel_size=module.kernel_size,
                        stride=module.stride,
                        padding=module.padding,
                        dilation=module.dilation,
                        groups=module.groups,
                        padding_mode=module.padding_mode,
                        batch_size=input_shape[0],
                        input_height=input_shape[-2],
                        input_width=input_shape[-1],
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        dtype=ttnn.bfloat16,
                        weights_dtype=ttnn.bfloat16,
                        use_1d_systolic_array=True,
                        enable_auto_formatting=False,
                        conv_blocking_and_parallelization_config_override={},
                        device=device,
                    ),
                )

            # ConvTranspose2d
            elif isinstance(module, torch.nn.ConvTranspose2d):
                insert_nested(
                    ttnn_module_args,
                    module_path,
                    ConvTranspose2dArgs(
                        in_channels=module.in_channels,
                        out_channels=module.out_channels,
                        kernel_size=module.kernel_size,
                        stride=module.stride,
                        padding=module.padding,
                        output_padding=module.output_padding,
                        dilation=module.dilation,
                        groups=module.groups,
                        padding_mode=module.padding_mode,
                        batch_size=input_shape[0],
                        input_height=input_shape[-2],
                        input_width=input_shape[-1],
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        dtype=ttnn.bfloat16,
                        weights_dtype=ttnn.bfloat16,
                        use_1d_systolic_array=True,
                        enable_auto_formatting=False,
                        conv_blocking_and_parallelization_config_override={},
                        device=device,
                    ),
                )

            # MaxPool2d
            elif isinstance(module, torch.nn.MaxPool2d):
                insert_nested(
                    ttnn_module_args,
                    module_path,
                    MaxPool2dArgs(
                        kernel_size=module.kernel_size,
                        stride=module.stride,
                        padding=module.padding,
                        dilation=module.dilation,
                        batch_size=input_shape[0],
                        input_channels=input_shape[1],
                        input_height=input_shape[-2],
                        input_width=input_shape[-1],
                        dtype=ttnn.bfloat16,
                    ),
                )

            # GroupNorm
            elif isinstance(module, torch.nn.GroupNorm):
                insert_nested(
                    ttnn_module_args,
                    module_path,
                    GroupNormArgs(
                        num_groups=module.num_groups,
                        num_channels=module.num_channels,
                        eps=module.eps,
                        affine=module.affine,
                        batch_size=input_shape[0],
                        input_height=input_shape[-2],
                        input_width=input_shape[-1],
                        dtype=ttnn.bfloat16,
                    ),
                )

            else:
                nested = _infer_ttnn_module_args(operation.graph)
                if nested:
                    insert_nested(
                        ttnn_module_args,
                        module_path,
                        nested,
                    )

        return make_dot_access_dict(ttnn_module_args, ignore_types=(ModuleArgs,))

    full_args = _infer_ttnn_module_args(ttnn.tracer.get_graph(output))

    # Root module is stored under empty name ""
    return full_args.get("", full_args)
