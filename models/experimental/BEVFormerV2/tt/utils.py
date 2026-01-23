# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    MaxPool2dConfiguration,
    AutoShardedStrategyConfiguration,
    HeightShardedStrategyConfiguration,
    BlockShardedStrategyConfiguration,
)

_model_config_default = {
    "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
    "WEIGHTS_DTYPE": ttnn.bfloat16,
    "ACTIVATIONS_DTYPE": ttnn.bfloat16,
}


def create_conv2d_config(
    input_height: int,
    input_width: int,
    in_channels: int,
    out_channels: int,
    batch_size: int,
    kernel_size,
    weight,
    bias=None,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups: int = 1,
    model_config: dict = None,
    conv_config: dict = None,
    activation=None,
    shard_layout=None,
    deallocate_activation=True,
    reallocate_halo_output=False,
    reshard_if_not_optimal=False,
    act_block_h_override=0,
    enable_act_double_buffer=False,
    enable_weights_double_buffer=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
    config_tensors_in_dram=True,
) -> Conv2dConfiguration:
    cfg = model_config if model_config is not None else _model_config_default

    kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    stride = (stride, stride) if isinstance(stride, int) else stride
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation

    if isinstance(padding, int):
        padding = (padding, padding)
    elif isinstance(padding, tuple) and len(padding) == 4:
        padding = (padding[0], padding[2])

    if isinstance(weight, ttnn.Tensor):
        weight = ttnn.to_torch(weight)
    if bias is not None and isinstance(bias, ttnn.Tensor):
        bias = ttnn.to_torch(bias)
        if len(bias.shape) > 1:
            bias = bias.flatten()

    ttnn_weight, ttnn_bias = Conv2dConfiguration.convert_torch_weight_and_bias_to_ttnn(weight, bias)

    if conv_config:
        activation = activation or conv_config.get("activation")
        shard_layout = shard_layout or conv_config.get("shard_layout")
        deallocate_activation = conv_config.get("deallocate_activation", deallocate_activation)
        reallocate_halo_output = conv_config.get("reallocate_halo_output", reallocate_halo_output)
        enable_act_double_buffer = conv_config.get("enable_act_double_buffer", enable_act_double_buffer)
        enable_weights_double_buffer = conv_config.get("enable_weights_double_buffer", enable_weights_double_buffer)
        packer_l1_acc = conv_config.get("packer_l1_acc", packer_l1_acc)

    strategy_configs = {
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED: HeightShardedStrategyConfiguration,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED: BlockShardedStrategyConfiguration,
    }
    strategy_class = strategy_configs.get(shard_layout, AutoShardedStrategyConfiguration)

    strategy_args = (
        {"reshard_if_not_optimal": reshard_if_not_optimal, "act_block_h_override": act_block_h_override}
        if shard_layout in strategy_configs
        else {}
    )
    sharding_strategy = strategy_class(**strategy_args)

    return Conv2dConfiguration(
        input_height=input_height,
        input_width=input_width,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        weight=ttnn_weight,
        bias=ttnn_bias,
        activation=activation,
        activation_dtype=cfg["ACTIVATIONS_DTYPE"],
        weights_dtype=cfg["WEIGHTS_DTYPE"],
        output_dtype=cfg["ACTIVATIONS_DTYPE"],
        math_fidelity=cfg["MATH_FIDELITY"],
        sharding_strategy=sharding_strategy,
        deallocate_activation=deallocate_activation,
        reallocate_halo_output=reallocate_halo_output,
        enable_act_double_buffer=enable_act_double_buffer,
        enable_weights_double_buffer=enable_weights_double_buffer,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=packer_l1_acc,
        config_tensors_in_dram=config_tensors_in_dram,
    )


def create_maxpool_config(
    input_height: int,
    input_width: int,
    channels: int,
    batch_size: int,
    kernel_size=(2, 2),
    stride=(2, 2),
    padding=(0, 0),
    dilation=(1, 1),
    ceil_mode: bool = False,
    dtype: ttnn.DataType = None,
) -> MaxPool2dConfiguration:
    kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
    dtype = dtype or _model_config_default["ACTIVATIONS_DTYPE"]

    return MaxPool2dConfiguration(
        input_height=input_height,
        input_width=input_width,
        channels=channels,
        batch_size=batch_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        dtype=dtype,
    )


def post_process_conv_output(
    output_tensor,
    batch_size: int,
    out_height: int,
    out_width: int,
    out_channels: int = None,
    to_dram: bool = True,
    reshape_4d: bool = True,
):
    if output_tensor.is_sharded():
        memory_config = ttnn.DRAM_MEMORY_CONFIG if to_dram else ttnn.L1_MEMORY_CONFIG
        output_tensor = ttnn.sharded_to_interleaved(output_tensor, memory_config)

    if reshape_4d:
        channels = out_channels or output_tensor.shape[-1]
        shape = output_tensor.shape
        target_shape = (batch_size, out_height, out_width, channels)

        is_reshape_required = (
            len(shape) == 3
            or (len(shape) == 4 and shape[1] == 1 and shape[2] == out_height * out_width)
            or (len(shape) == 4 and shape[0] == 1 and shape[1] == 1 and shape[2] == batch_size * out_height * out_width)
        )

        if is_reshape_required:
            output_tensor = ttnn.reshape(output_tensor, target_shape)

    return output_tensor


def ensure_memory_config(tensor, target_memory_config=None, reference_tensor=None):
    target = target_memory_config
    if target is None and reference_tensor is not None:
        target = reference_tensor.memory_config()

    if target is not None and tensor.memory_config() != target:
        tensor = ttnn.to_memory_config(tensor, target)

    return tensor
