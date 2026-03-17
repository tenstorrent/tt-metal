# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN ChannelMapper neck for DINO-5scale.

Architecture (from mmdet config):
  4 input levels (from Swin-L backbone) -> 5 output levels, all 256-d:
    P2: Conv2d(192  -> 256, 1x1) + GroupNorm(32)
    P3: Conv2d(384  -> 256, 1x1) + GroupNorm(32)
    P4: Conv2d(768  -> 256, 1x1) + GroupNorm(32)
    P5: Conv2d(1536 -> 256, 1x1) + GroupNorm(32)
    P6: Conv2d(1536 -> 256, 3x3, stride=2) + GroupNorm(32)  (from last backbone feature)

All ops on device. GroupNorm uses DRAM multi-core with Welford reciprocals
(same approach as SDXL VAE) to handle large spatial dimensions.
P6 conv uses BLOCK_SHARDED to avoid NOC burst size limit with 1536 in_channels.
"""

import math

import ttnn
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores


def _nearest_32_per_core(x, core):
    """Round up x to the nearest multiple of (32 * core)."""
    return math.ceil(x / core / 32) * 32 * core


# Spatial shapes for each level at 800x1333 input (backbone output shapes)
# These are used to precompute reciprocals for Welford GN.
DINO_NECK_LEVEL_SHAPES = {
    # level: (N, C, H, W)
    0: (1, 256, 200, 334),  # P2
    1: (1, 256, 100, 167),  # P3
    2: (1, 256, 50, 84),  # P4
    3: (1, 256, 25, 42),  # P5
    4: (1, 256, 13, 21),  # P6
}


class TtDINONeck:
    """TTNN ChannelMapper neck: 4 backbone levels -> 5 FPN levels (256-d)."""

    def __init__(
        self,
        device,
        parameters,
        in_channels=(192, 384, 768, 1536),
        out_channels=256,
        num_groups=32,
        level_shapes=None,
    ):
        self.device = device
        self.parameters = parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_groups = num_groups

        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
            math_approx_mode=False,
        )

        # Conv weights stay on device (loaded via _to_device_recursive) for reuse across runs

        # Precompute Welford reciprocals for each GN level (following SDXL VAE pattern)
        if level_shapes is None:
            level_shapes = DINO_NECK_LEVEL_SHAPES

        self.gn_core_grid = ttnn.CoreGrid(y=8, x=8)
        # Welford only for large spatial where it provides real benefit;
        # basic DRAM GN is fine for smaller spatial where padding ratio is low.
        self.welford_threshold = 50000
        self.num_out_blocks = {}
        self.reciprocals_sharded = {}
        self.gn_params_per_level = {}

        for lvl, (N, C, H, W) in level_shapes.items():
            spatial = H * W
            use_welford = spatial >= self.welford_threshold

            if spatial > 50000:
                nob = 32
            else:
                nob = 1
            self.num_out_blocks[lvl] = nob

            gn_w = parameters["convs" if lvl < 4 else "extra_convs"][lvl if lvl < 4 else 0]["gn"]["_torch_w"]
            gn_b = parameters["convs" if lvl < 4 else "extra_convs"][lvl if lvl < 4 else 0]["gn"]["_torch_b"]
            [gamma, beta], input_mask = ttnn.dram_group_norm_params_from_torch(
                [gn_w, gn_b],
                out_channels,
                num_groups,
                device,
                core_grid=self.gn_core_grid,
                return_mask=True,
            )
            self.gn_params_per_level[lvl] = {
                "weight": gamma,
                "bias": beta,
                "input_mask": input_mask,
            }

            if use_welford:
                torch_recip = ttnn.create_group_norm_reciprocals(N, C, H, W, num_groups, self.gn_core_grid)
                inner_dim = torch_recip.shape[1]
                page_bytes = inner_dim * 4
                n_rows = torch_recip.shape[0]
                if page_bytes % 64 != 0:
                    aligned_inner = ((page_bytes + 63) // 64 * 64) // 4
                    recip_tt = ttnn.from_torch(
                        torch_recip,
                        dtype=ttnn.DataType.FLOAT32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    # Trace-safe: use pad instead of zeros + concat
                    recip_tt = ttnn.pad(
                        recip_tt,
                        padding=[(0, 0), (0, aligned_inner - inner_dim)],
                        value=0.0,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                else:
                    recip_tt = ttnn.from_torch(
                        torch_recip,
                        dtype=ttnn.DataType.FLOAT32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                shard_cfg = ttnn.create_sharded_memory_config(
                    shape=list(recip_tt.shape),
                    core_grid=self.gn_core_grid,
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                )
                self.reciprocals_sharded[lvl] = ttnn.to_memory_config(recip_tt, shard_cfg)
            else:
                self.reciprocals_sharded[lvl] = None

    def _group_norm_dram(self, x_nhwc, level_idx, out_h, out_w):
        """
        Apply GroupNorm via DRAM multi-core (optionally Welford).
        Input: x_nhwc — conv2d output, shape [N, 1, H*W_padded, C] in TILE on device.
        Output: NCHW tensor on device.

        Key: untilize first, then re-tilize with zero padding to ensure GN
        statistics are not corrupted by tile padding garbage values.
        """
        N = x_nhwc.shape[0]
        C = self.out_channels
        spatial = out_h * out_w

        # Untilize to ROW_MAJOR so we can cleanly reshape/slice
        x = ttnn.to_memory_config(x_nhwc, ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

        # Slice to valid spatial positions only: [N, 1, spatial, C]
        x = x[:, :, :spatial, :]

        # Re-tilize with zero-padding aligned to GN core grid distribution
        grid_x = self.gn_core_grid.x
        grid_y = self.gn_core_grid.y
        padded_h = _nearest_32_per_core(spatial, grid_x)
        padded_w = _nearest_32_per_core(C, grid_y)
        out_shape = [N, 1, padded_h, padded_w]
        x = ttnn.tilize_with_val_padding(x, output_tensor_shape=out_shape, pad_value=0, use_multicore=True)

        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        recip_sharded = self.reciprocals_sharded[level_idx]
        gn_level = self.gn_params_per_level[level_idx]
        use_welford = recip_sharded is not None

        gn_kwargs = dict(
            num_groups=self.num_groups,
            input_mask=gn_level["input_mask"],
            weight=gn_level["weight"],
            bias=gn_level["bias"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_layout=ttnn.TILE_LAYOUT,
            core_grid=self.gn_core_grid,
            inplace=False,
            num_out_blocks=self.num_out_blocks[level_idx],
            epsilon=1e-5,
        )
        if use_welford:
            gn_kwargs["use_welford"] = True
            gn_kwargs["reciprocals"] = recip_sharded

        x = ttnn.group_norm(x, **gn_kwargs)

        # Reshape [N, 1, padded_spatial, C] -> slice -> [N, H, W, C] -> [N, C, H, W]
        x = x[:, :, :spatial, :C]
        x = ttnn.reshape(x, (N, out_h, out_w, C))
        x = ttnn.permute(x, (0, 3, 1, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return x

    def _conv1x1_gn(self, x, conv_params, in_ch, level_idx):
        """1x1 Conv2d (HEIGHT_SHARDED) + GroupNorm (DRAM Welford). NCHW in/out."""
        # N, C, H, W = x.shape
        N, H, W, C = x.shape

        shard_grid = get_shard_grid_from_num_cores(64, self.device)
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
        )
        conv_config.core_grid = shard_grid
        conv_config.override_sharding_config = True

        [output, [out_h, out_w], [conv_params["weight"], conv_params["bias"]]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=conv_params["weight"],
            bias_tensor=conv_params["bias"],
            in_channels=in_ch,
            out_channels=self.out_channels,
            device=self.device,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            batch_size=N,
            input_height=H,
            input_width=W,
            conv_config=conv_config,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=ttnn.bfloat16,
        )
        output = ttnn.sharded_to_interleaved(output, ttnn.DRAM_MEMORY_CONFIG)

        return self._group_norm_dram(output, level_idx, out_h, out_w)

    def _conv3x3_s2_gn(self, x, conv_params, in_ch, level_idx):
        """3x3 stride-2 Conv2d (BLOCK_SHARDED) + GroupNorm (DRAM). NCHW in/out."""
        # N, C, H, W = x.shape
        N, H, W, C = x.shape

        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            reshard_if_not_optimal=True,
        )

        [output, [out_h, out_w], [conv_params["weight"], conv_params["bias"]]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=conv_params["weight"],
            bias_tensor=conv_params["bias"],
            in_channels=in_ch,
            out_channels=self.out_channels,
            device=self.device,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            batch_size=N,
            input_height=H,
            input_width=W,
            conv_config=conv_config,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=ttnn.bfloat16,
        )
        output = ttnn.sharded_to_interleaved(output, ttnn.DRAM_MEMORY_CONFIG)

        return self._group_norm_dram(output, level_idx, out_h, out_w)

    def __call__(self, features):
        """
        features: list of 4 NCHW tensors from backbone.
        Returns: list of 5 NCHW tensors with out_channels=256.
        """
        assert len(features) == 4, f"Expected 4 backbone features, got {len(features)}"

        feat3_copy = ttnn.clone(features[3], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        outputs = []
        for i in range(4):
            out = self._conv1x1_gn(
                features[i],
                conv_params=self.parameters["convs"][i]["conv"],
                in_ch=self.in_channels[i],
                level_idx=i,
            )
            outputs.append(out)

        p6 = self._conv3x3_s2_gn(
            feat3_copy,
            conv_params=self.parameters["extra_convs"][0]["conv"],
            in_ch=self.in_channels[3],
            level_idx=4,
        )
        outputs.append(p6)

        return outputs
