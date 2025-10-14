# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.vanilla_unet.tt.unet_config import TtUNetLayerConfigs, UpconvConfiguration
from models.tt_cnn.tt.builder import TtConv2d, TtMaxPool2d


class TtUNet:
    """
    TT-CNN UNet implementation built from UNetConfig configurations
    """

    def __init__(self, configs: TtUNetLayerConfigs, device: ttnn.Device):
        self.device = device
        self.configs = configs

        self.encoder1_conv1 = TtConv2d(configs.encoder1_conv1, device)
        self.encoder1_conv2 = TtConv2d(configs.encoder1_conv2, device)
        self.encoder1_pool = TtMaxPool2d(configs.encoder1_pool, device)

        self.encoder2_conv1 = TtConv2d(configs.encoder2_conv1, device)
        self.encoder2_conv2 = TtConv2d(configs.encoder2_conv2, device)
        self.encoder2_pool = TtMaxPool2d(configs.encoder2_pool, device)

        self.encoder3_conv1 = TtConv2d(configs.encoder3_conv1, device)
        self.encoder3_conv2 = TtConv2d(configs.encoder3_conv2, device)
        self.encoder3_pool = TtMaxPool2d(configs.encoder3_pool, device)

        self.encoder4_conv1 = TtConv2d(configs.encoder4_conv1, device)
        self.encoder4_conv2 = TtConv2d(configs.encoder4_conv2, device)
        self.encoder4_pool = TtMaxPool2d(configs.encoder4_pool, device)

        self.bottleneck_conv1 = TtConv2d(configs.bottleneck_conv1, device)
        self.bottleneck_conv2 = TtConv2d(configs.bottleneck_conv2, device)

        self.decoder4_conv1 = TtConv2d(configs.decoder4_conv1, device)
        self.decoder4_conv2 = TtConv2d(configs.decoder4_conv2, device)

        self.decoder3_conv1 = TtConv2d(configs.decoder3_conv1, device)
        self.decoder3_conv2 = TtConv2d(configs.decoder3_conv2, device)

        self.decoder2_conv1 = TtConv2d(configs.decoder2_conv1, device)
        self.decoder2_conv2 = TtConv2d(configs.decoder2_conv2, device)

        self.decoder1_conv1 = TtConv2d(configs.decoder1_conv1, device)
        self.decoder1_conv2 = TtConv2d(configs.decoder1_conv2, device)

        self.final_conv = TtConv2d(configs.final_conv, device)

        # Store upconv configurations
        self.upconv4_config = configs.upconv4
        self.upconv3_config = configs.upconv3
        self.upconv2_config = configs.upconv2
        self.upconv1_config = configs.upconv1

    def __call__(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        # Encoder 1
        enc1 = self.encoder1_conv1(input_tensor)
        enc1 = self.encoder1_conv2(enc1)
        skip1 = ttnn.to_memory_config(enc1, ttnn.DRAM_MEMORY_CONFIG)
        enc1 = self.encoder1_pool(enc1)

        # Encoder 2
        enc2 = self.encoder2_conv1(enc1)
        enc2 = self.encoder2_conv2(enc2)
        skip2 = ttnn.to_memory_config(enc2, ttnn.DRAM_MEMORY_CONFIG)
        enc2 = self.encoder2_pool(enc2)

        # Encoder 3
        enc3 = self.encoder3_conv1(enc2)
        enc3 = self.encoder3_conv2(enc3)
        skip3 = ttnn.to_memory_config(enc3, ttnn.DRAM_MEMORY_CONFIG)
        enc3 = self.encoder3_pool(enc3)

        # Encoder 4
        enc4 = self.encoder4_conv1(enc3)
        enc4 = self.encoder4_conv2(enc4)
        skip4 = ttnn.to_memory_config(enc4, ttnn.DRAM_MEMORY_CONFIG)
        enc4 = self.encoder4_pool(enc4)

        # Bottleneck
        bottleneck = self.bottleneck_conv1(enc4)
        bottleneck = self.bottleneck_conv2(bottleneck)

        # Decoder 4
        print("decoder 4")
        dec4 = self._transpose_conv(bottleneck, self.upconv4_config)
        dec4 = self._concatenate_skip_connection(dec4, skip4)
        dec4 = self.decoder4_conv1(dec4)
        dec4 = self.decoder4_conv2(dec4)

        # Decoder 3
        print("decoder 3")
        dec3 = self._transpose_conv(dec4, self.upconv3_config)
        dec3 = self._concatenate_skip_connection(dec3, skip3)
        dec3 = self.decoder3_conv1(dec3)
        dec3 = self.decoder3_conv2(dec3)

        # Decoder 2
        print("decoder 2")
        dec2 = self._transpose_conv(dec3, self.upconv2_config)
        dec2 = self._concatenate_skip_connection(dec2, skip2)
        dec2 = self.decoder2_conv1(dec2)
        dec2 = self.decoder2_conv2(dec2)

        # Decoder 1
        print("decoder 1")
        dec1 = self._transpose_conv(dec2, self.upconv1_config)
        dec1 = self._concatenate_skip_connection(dec1, skip1, rm=False)
        dec1 = self.decoder1_conv1(dec1)
        dec1 = self.decoder1_conv2(dec1)

        print("final guy")
        output = self.final_conv(dec1)

        return output

    def _transpose_conv(self, input_tensor: ttnn.Tensor, upconv_config: UpconvConfiguration) -> ttnn.Tensor:
        """
        Perform transpose convolution using UpconvConfiguration

        Args:
            input_tensor: Input tensor
            upconv_config: UpconvConfiguration with all parameters

        Returns:
            Upsampled tensor
        """
        print(f"running conv2d transpose: {upconv_config}")
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat8_b,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            enable_act_double_buffer=False,
            output_layout=ttnn.TILE_LAYOUT,
            act_block_h_override=32,
        )
        compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        return ttnn.conv_transpose2d(
            input_tensor=input_tensor,
            weight_tensor=upconv_config.weight,
            bias_tensor=upconv_config.bias,
            in_channels=upconv_config.in_channels,
            out_channels=upconv_config.out_channels,
            device=self.device,
            kernel_size=upconv_config.kernel_size,
            stride=upconv_config.stride,
            padding=upconv_config.padding,
            batch_size=upconv_config.batch_size,
            input_height=upconv_config.input_height,
            input_width=upconv_config.input_width,
            conv_config=conv_config,
            compute_config=compute_config,
        )

    def _concatenate_skip_connection(self, upsampled: ttnn.Tensor, skip: ttnn.Tensor, rm=True) -> ttnn.Tensor:
        assert upsampled.shape[-1] == skip.shape[-1]

        input_core_grid = upsampled.memory_config().shard_spec.grid
        input_shard_shape = upsampled.memory_config().shard_spec.shape
        input_shard_spec = ttnn.ShardSpec(input_core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        input_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec
        )

        output_core_grid = input_core_grid
        output_shard_shape = (input_shard_shape[0], input_shard_shape[1] * 2)
        output_shard_spec = ttnn.ShardSpec(output_core_grid, output_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        output_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec
        )

        skip = ttnn.to_memory_config(skip, input_memory_config)

        if rm:
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
            print(upsampled, skip)
            concatenated = ttnn.concat([upsampled, skip], dim=3, memory_config=output_memory_config)
            ttnn.deallocate(upsampled)
            ttnn.deallocate(skip)
            return concatenated


def create_unet_from_configs(configs: TtUNetLayerConfigs, device: ttnn.Device) -> TtUNet:
    return TtUNet(configs, device)
