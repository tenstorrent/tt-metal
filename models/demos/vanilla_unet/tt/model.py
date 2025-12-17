# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.vanilla_unet.tt.config import TtUNetLayerConfigs, UpconvConfiguration
from models.tt_cnn.tt.builder import TtConv2d, TtMaxPool2d


def concatenate_skip_connection(
    upsampled: ttnn.Tensor, skip: ttnn.Tensor, use_row_major_layout_for_inputs=True
) -> ttnn.Tensor:
    assert (
        upsampled.shape == skip.shape
    ), f"Expected input tensors to have identical shapes for concatenation (was {upsampled.shape} and {skip.shape})"

    # If residual is in DRAM we can reshard it to the same spec as the activation since their shapes always match
    if not skip.is_sharded():
        input_core_grid = upsampled.memory_config().shard_spec.grid
        input_shard_shape = upsampled.memory_config().shard_spec.shape
        input_shard_spec = ttnn.ShardSpec(input_core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        input_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec
        )
        skip = ttnn.to_memory_config(skip, input_memory_config)

    output_core_grid = input_core_grid
    output_shard_shape = (
        input_shard_shape[0],
        input_shard_shape[1] * 2,
    )  # Assumes activation and residual channels match
    output_shard_spec = ttnn.ShardSpec(output_core_grid, output_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec
    )
    if use_row_major_layout_for_inputs:
        upsampled_rm = ttnn.to_layout(upsampled, ttnn.ROW_MAJOR_LAYOUT)
        skip_rm = ttnn.to_layout(skip, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(upsampled)
        ttnn.deallocate(skip)

        concatenated = ttnn.concat([upsampled_rm, skip_rm], dim=3, memory_config=output_memory_config)
        ttnn.deallocate(upsampled_rm)
        ttnn.deallocate(skip_rm)

        concat_tiled = ttnn.to_layout(concatenated, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(concatenated)
        return concat_tiled
    else:
        concatenated = ttnn.concat([upsampled, skip], dim=3, memory_config=output_memory_config)
        ttnn.deallocate(upsampled)
        ttnn.deallocate(skip)
        return concatenated


def transpose_conv2d(
    input_tensor: ttnn.Tensor,
    upconv_config: UpconvConfiguration,
    act_block_h_override=32,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
) -> ttnn.Tensor:
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        deallocate_activation=True,
        enable_act_double_buffer=False,
        output_layout=ttnn.TILE_LAYOUT,
        act_block_h_override=act_block_h_override,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        input_tensor.device().arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=packer_l1_acc,
    )
    output, [upconv_config.weight, upconv_config.bias] = ttnn.conv_transpose2d(
        input_tensor=input_tensor,
        weight_tensor=upconv_config.weight,
        bias_tensor=upconv_config.bias,
        in_channels=upconv_config.in_channels,
        out_channels=upconv_config.out_channels,
        kernel_size=upconv_config.kernel_size,
        stride=upconv_config.stride,
        padding=upconv_config.padding,
        batch_size=upconv_config.batch_size,
        input_height=upconv_config.input_height,
        input_width=upconv_config.input_width,
        conv_config=conv_config,
        compute_config=compute_config,
        device=input_tensor.device(),
        return_output_dim=False,
        return_weights_and_bias=True,
    )
    return output


class TtUNetEncoder:
    def __init__(self, conv1, conv2, pool, device):
        self.conv1 = TtConv2d(conv1, device)
        self.conv2 = TtConv2d(conv2, device)
        self.pool = TtMaxPool2d(pool, device)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        skip = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x = self.pool(x)
        return x, skip


class TtUNet:
    def __init__(self, configs: TtUNetLayerConfigs, device: ttnn.Device):
        self.device = device
        self.configs = configs

        self.downblock1 = TtUNetEncoder(configs.encoder1_conv1, configs.encoder1_conv2, configs.encoder1_pool, device)
        self.downblock2 = TtUNetEncoder(configs.encoder2_conv1, configs.encoder2_conv2, configs.encoder2_pool, device)
        self.downblock3 = TtUNetEncoder(configs.encoder3_conv1, configs.encoder3_conv2, configs.encoder3_pool, device)
        self.downblock4 = TtUNetEncoder(configs.encoder4_conv1, configs.encoder4_conv2, configs.encoder4_pool, device)

        self.bottleneck_conv1 = TtConv2d(configs.bottleneck_conv1, device)
        self.bottleneck_conv2 = TtConv2d(configs.bottleneck_conv2, device)

        self.upconv4_config = configs.upconv4
        self.decoder4_conv1 = TtConv2d(configs.decoder4_conv1, device)
        self.decoder4_conv2 = TtConv2d(configs.decoder4_conv2, device)

        self.upconv3_config = configs.upconv3
        self.decoder3_conv1 = TtConv2d(configs.decoder3_conv1, device)
        self.decoder3_conv2 = TtConv2d(configs.decoder3_conv2, device)

        self.upconv2_config = configs.upconv2
        self.decoder2_conv1 = TtConv2d(configs.decoder2_conv1, device)
        self.decoder2_conv2 = TtConv2d(configs.decoder2_conv2, device)

        self.upconv1_config = configs.upconv1
        self.decoder1_conv1 = TtConv2d(configs.decoder1_conv1, device)
        self.decoder1_conv2 = TtConv2d(configs.decoder1_conv2, device)

        self.final_conv = TtConv2d(configs.final_conv, device)

    def preprocess_input_tensor(self, x: ttnn.Tensor, deallocate_input_activation: bool):
        output = ttnn.experimental.convert_to_hwc(x)
        if deallocate_input_activation:
            ttnn.deallocate(x)  # Some use-cases have a persistent input tensor that we don't want to delete
        return output

    def __call__(self, input_tensor: ttnn.Tensor, deallocate_input_activation: bool = True) -> ttnn.Tensor:
        input_tensor = self.preprocess_input_tensor(input_tensor, deallocate_input_activation)

        enc1, skip1 = self.downblock1(input_tensor)
        enc2, skip2 = self.downblock2(enc1)
        enc3, skip3 = self.downblock3(enc2)
        enc4, skip4 = self.downblock4(enc3)

        bottleneck = self.bottleneck_conv1(enc4)
        bottleneck = self.bottleneck_conv2(bottleneck)

        dec4 = transpose_conv2d(bottleneck, self.upconv4_config, act_block_h_override=3 * 32)
        dec4 = concatenate_skip_connection(dec4, skip4)
        dec4 = self.decoder4_conv1(dec4)
        dec4 = self.decoder4_conv2(dec4)

        dec3 = transpose_conv2d(dec4, self.upconv3_config, act_block_h_override=5 * 32)
        dec3 = concatenate_skip_connection(dec3, skip3)
        dec3 = self.decoder3_conv1(dec3)
        dec3 = self.decoder3_conv2(dec3)

        dec2 = transpose_conv2d(dec3, self.upconv2_config, act_block_h_override=2 * 32)
        dec2 = concatenate_skip_connection(dec2, skip2)
        dec2 = self.decoder2_conv1(dec2)
        dec2 = self.decoder2_conv2(dec2)

        dec1 = transpose_conv2d(
            dec2, self.upconv1_config, fp32_dest_acc_en=False, packer_l1_acc=False, act_block_h_override=5 * 32
        )
        dec1 = concatenate_skip_connection(dec1, skip1, use_row_major_layout_for_inputs=False)
        dec1 = self.decoder1_conv1(dec1)
        dec1 = self.decoder1_conv2(dec1)

        output = self.final_conv(dec1)

        return ttnn.experimental.convert_to_chw(output, dtype=ttnn.bfloat16)


def create_unet_from_configs(configs: TtUNetLayerConfigs, device: ttnn.Device) -> TtUNet:
    """
    Construct a TtUnet instance using TtUnetLayerConfigs configuration object
    """
    return TtUNet(configs, device)
