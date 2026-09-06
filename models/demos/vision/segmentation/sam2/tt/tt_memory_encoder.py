# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI U.S. Corp., for the TTNN port. Reference code © Meta Platforms, Inc. (Apache-2.0).
# SPDX-License-Identifier: Apache-2.0
import ttnn
from models.demos.vision.segmentation.sam2.tt.tt_hiera import _sharded_linear
from models.tt_cnn.tt.builder import (
    BlockShardedStrategyConfiguration,
    Conv2dConfiguration,
    HeightShardedStrategyConfiguration,
    TtConv2d,
)


def _height_sharded_conv(parameters, device, input_size, in_channels, out_channels, kernel_size, stride=1, padding=0):
    return TtConv2d(
        Conv2dConfiguration(
            input_height=input_size,
            input_width=input_size,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=1,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            weight=parameters.weight,
            bias=parameters.bias,
            sharding_strategy=HeightShardedStrategyConfiguration(),
            enable_weights_double_buffer=False,
            deallocate_activation=True,
        ),
        device,
    )


class TtMemoryEncoder:
    def __init__(self, parameters, device):
        self.device = device
        self.p = parameters
        self.batch_size = 1
        self.out_dim = 64
        self.in_dim = 256
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        self._layer_norm_compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.fuser_layers = parameters.fuser.layers
        self.mask_input_memory_config = ttnn.create_sharded_memory_config(
            (1024 * 1024 // 64, 8),
            ttnn.CoreGrid(x=8, y=8),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        input_size = 1024
        in_channels = 1
        self.mask_convs = []
        for block in parameters.mask_downsampler.blocks:
            out_channels = in_channels * 4
            self.mask_convs.append(
                _height_sharded_conv(block.conv, device, input_size, in_channels, out_channels, 3, stride=2, padding=1)
            )
            input_size //= 2
            in_channels = out_channels
        self.mask_final = _height_sharded_conv(
            parameters.mask_downsampler.final, device, input_size, in_channels, self.in_dim, 1
        )
        self.pix_feat_proj = _height_sharded_conv(parameters.pix_feat_proj, device, 64, self.in_dim, self.in_dim, 1)
        self.fuser_dwconvs = [
            TtConv2d(
                Conv2dConfiguration(
                    input_height=64,
                    input_width=64,
                    in_channels=self.in_dim,
                    out_channels=self.in_dim,
                    batch_size=1,
                    kernel_size=(7, 7),
                    padding=(3, 3),
                    groups=self.in_dim,
                    weight=block.dwconv.weight,
                    bias=None,
                    sharding_strategy=BlockShardedStrategyConfiguration(reshard_if_not_optimal=True),
                    enable_act_double_buffer=True,
                    enable_weights_double_buffer=False,
                    deallocate_activation=False,
                ),
                device,
            )
            for block in self.fuser_layers
        ]
        self.out_proj = _height_sharded_conv(parameters.out_proj, device, 64, self.in_dim, self.out_dim, 1)

    def __call__(self, pix_feat, masks):
        x = masks
        for conv, block in zip(self.mask_convs, self.p.mask_downsampler.blocks):
            x = conv(x)
            memory_config = x.memory_config()
            normalized = ttnn.moreh_layer_norm(
                x,
                1,
                1e-6,
                block.norm.weight,
                block.norm.bias,
                memory_config=memory_config,
                compute_kernel_config=self._layer_norm_compute_config,
            )[0]
            ttnn.deallocate(x)
            x = ttnn.gelu(
                normalized,
                fast_and_approximate_mode=False,
                memory_config=memory_config,
            )
            ttnn.deallocate(normalized)
        mask_ds = self.mask_final(x)
        pix = self.pix_feat_proj(pix_feat)
        x = ttnn.add_(pix, mask_ds)
        ttnn.deallocate(mask_ds)
        for block, dwconv in zip(self.fuser_layers, self.fuser_dwconvs):
            # Depthwise conv must not free x because it is also the residual.
            residual = x
            residual_memory_config = residual.memory_config()
            y = dwconv(x)
            restored = ttnn.to_memory_config(y, residual_memory_config)
            if restored is not y:
                ttnn.deallocate(y)
            y = restored
            memory_config = y.memory_config()
            y = ttnn.add_(y, block.dwconv.post_bias)
            normalized = ttnn.moreh_layer_norm(
                y,
                1,
                1e-6,
                block.norm.weight,
                block.norm.bias,
                memory_config=memory_config,
                compute_kernel_config=self._layer_norm_compute_config,
            )[0]
            ttnn.deallocate(y)
            y4 = ttnn.reshape(normalized, (self.batch_size, 64, 64, self.in_dim))
            hidden = _sharded_linear(
                y4,
                block.pwconv1.weight,
                block.pwconv1.bias,
                self.device,
                self.compute_config,
                out_sharded=True,
                fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 0.0),
            )
            output = _sharded_linear(
                hidden,
                block.pwconv2.weight,
                block.pwconv2.bias,
                self.device,
                self.compute_config,
                out_sharded=True,
            )
            ttnn.deallocate(hidden)
            y = ttnn.reshape(output, (self.batch_size, 1, 64 * 64, self.in_dim))
            ttnn.deallocate(normalized)
            y = ttnn.add_(y, residual)
            ttnn.deallocate(residual)
            x = y
        x = self.out_proj(x)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(x, (self.batch_size, 64, 64, self.out_dim))
