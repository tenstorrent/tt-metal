# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Feature Pyramid Network for ATSS.

ATSS FPN config:
  - 3 lateral convs (1x1): in=[384, 768, 1536] → out=256
  - 3 FPN convs (3x3):     in=256 → out=256
  - 2 extra convs (3x3, stride=2): in=256 → out=256 (add_extra_convs='on_output')
  - Top-down pathway with 2x nearest-neighbor upsampling

Input:  3 NCHW feature maps from backbone stages 1, 2, 3.
Output: 5 NCHW feature maps (P3, P4, P5, P6, P7).
"""

import ttnn
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores


class TtFPN:
    """TTNN FPN producing 5-level feature pyramid from 3 backbone outputs."""

    def __init__(
        self,
        device,
        parameters,
        in_channels=(384, 768, 1536),
        out_channels=256,
        num_outs=5,
    ):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.num_ins = len(in_channels)

        self.lateral_weights = []
        self.lateral_biases = []
        for i in range(self.num_ins):
            w = parameters["lateral_convs"][i]["weight"]
            b = parameters["lateral_convs"][i]["bias"]
            self.lateral_weights.append(ttnn.from_device(w))
            self.lateral_biases.append(ttnn.from_device(b))

        self.fpn_weights = []
        self.fpn_biases = []
        for i in range(self.num_ins):
            w = parameters["fpn_convs"][i]["weight"]
            b = parameters["fpn_convs"][i]["bias"]
            self.fpn_weights.append(ttnn.from_device(w))
            self.fpn_biases.append(ttnn.from_device(b))

        self.extra_weights = []
        self.extra_biases = []
        num_extra = num_outs - self.num_ins
        for i in range(num_extra):
            w = parameters["fpn_convs"][self.num_ins + i]["weight"]
            b = parameters["fpn_convs"][self.num_ins + i]["bias"]
            self.extra_weights.append(ttnn.from_device(w))
            self.extra_biases.append(ttnn.from_device(b))

    def _conv2d(self, x, weight, bias, in_ch, out_ch, kernel_size, stride, padding, batch, h, w):
        """Run a single conv2d on NHWC input, return NHWC output with (out_h, out_w)."""
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
        )
        grid = self.device.compute_with_storage_grid_size()
        max_cores = grid.x * grid.y
        num_cores = min(max_cores, batch * h * w)
        if num_cores > 0:
            shard_grid = get_shard_grid_from_num_cores(num_cores, self.device)
            conv_config.core_grid = shard_grid
            conv_config.override_sharding_config = True

        compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
            math_approx_mode=True,
        )

        [output, [out_h, out_w], [weight, bias]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=weight,
            bias_tensor=bias,
            in_channels=in_ch,
            out_channels=out_ch,
            device=self.device,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            batch_size=batch,
            input_height=h,
            input_width=w,
            conv_config=conv_config,
            compute_config=compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=ttnn.bfloat16,
        )
        output = ttnn.sharded_to_interleaved(output, ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.reshape(output, (batch, out_h, out_w, out_ch))
        return output, out_h, out_w, weight, bias

    def __call__(self, inputs):
        """
        inputs: list of 3 NCHW tensors from backbone.
        Returns: list of 5 NCHW tensors (P3..P7).
        """
        assert len(inputs) == self.num_ins

        # Convert NCHW → NHWC and run lateral 1x1 convs
        laterals = []
        for i in range(self.num_ins):
            feat = inputs[i]
            N, C, H, W = feat.shape

            nhwc = ttnn.permute(feat, (0, 2, 3, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            nhwc = ttnn.to_layout(nhwc, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            lat, out_h, out_w, self.lateral_weights[i], self.lateral_biases[i] = self._conv2d(
                nhwc,
                self.lateral_weights[i],
                self.lateral_biases[i],
                in_ch=C,
                out_ch=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                batch=N,
                h=H,
                w=W,
            )
            laterals.append(lat)

        # Top-down pathway: upsample higher-level and add to lower-level
        for i in range(self.num_ins - 1, 0, -1):
            lat_higher = laterals[i]
            lat_higher_rm = ttnn.to_layout(lat_higher, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            upsampled = ttnn.upsample(lat_higher_rm, scale_factor=2, mode="nearest")

            target_h = laterals[i - 1].shape[1]
            target_w = laterals[i - 1].shape[2]
            up_h = upsampled.shape[1]
            up_w = upsampled.shape[2]
            if up_h != target_h or up_w != target_w:
                upsampled = upsampled[:, :target_h, :target_w, :]

            upsampled = ttnn.to_layout(upsampled, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            laterals[i - 1] = ttnn.to_layout(laterals[i - 1], ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            laterals[i - 1] = ttnn.add(laterals[i - 1], upsampled, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(upsampled)

        # FPN 3x3 convs on each lateral
        outs = []
        for i in range(self.num_ins):
            lat = laterals[i]
            lat = ttnn.to_layout(lat, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            N = lat.shape[0]
            H = lat.shape[1]
            W = lat.shape[2]

            fpn_out, _, _, self.fpn_weights[i], self.fpn_biases[i] = self._conv2d(
                lat,
                self.fpn_weights[i],
                self.fpn_biases[i],
                in_ch=self.out_channels,
                out_ch=self.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                batch=N,
                h=H,
                w=W,
            )
            outs.append(fpn_out)

        # Extra levels: stride-2 convs on output (add_extra_convs='on_output')
        extra_src = outs[-1]
        for i in range(len(self.extra_weights)):
            extra_src = ttnn.to_layout(extra_src, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            N = extra_src.shape[0]
            H = extra_src.shape[1]
            W = extra_src.shape[2]

            extra_out, _, _, self.extra_weights[i], self.extra_biases[i] = self._conv2d(
                extra_src,
                self.extra_weights[i],
                self.extra_biases[i],
                in_ch=self.out_channels,
                out_ch=self.out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                batch=N,
                h=H,
                w=W,
            )
            outs.append(extra_out)
            extra_src = extra_out

        # Convert all outputs NHWC → NCHW
        nchw_outs = []
        for out in outs:
            out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            nchw = ttnn.permute(out, (0, 3, 1, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            nchw_outs.append(nchw)

        return nchw_outs
