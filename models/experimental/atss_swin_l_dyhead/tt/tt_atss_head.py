# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN ATSS detection head.

With DyHead providing the stacked convolutions (stacked_convs=0), the ATSS
head is simply three 1×1 convolution branches per FPN level:
  - Classification: 256 → 80
  - Regression:     256 → 4  (scaled by a learnable per-level Scale)
  - Centerness:     256 → 1

Input:  5 NCHW feature maps from DyHead.
Output: (cls_scores, bbox_preds, centernesses) — each a list of 5 NCHW tensors.
"""

import ttnn
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores


class TtATSSHead:
    """TTNN ATSS detection head (stacked_convs=0, 1x1 prediction convs)."""

    def __init__(
        self,
        device,
        parameters,
        num_classes=80,
        in_channels=256,
        num_anchors=1,
        num_levels=5,
    ):
        self.device = device
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_anchors = num_anchors
        self.num_levels = num_levels

        self.cls_weight = ttnn.from_device(parameters["atss_cls"]["weight"])
        self.cls_bias = ttnn.from_device(parameters["atss_cls"]["bias"])

        self.reg_weight = ttnn.from_device(parameters["atss_reg"]["weight"])
        self.reg_bias = ttnn.from_device(parameters["atss_reg"]["bias"])

        self.centerness_weight = ttnn.from_device(parameters["atss_centerness"]["weight"])
        self.centerness_bias = ttnn.from_device(parameters["atss_centerness"]["bias"])

        self.scales = []
        for i in range(num_levels):
            self.scales.append(parameters["scales"][i])

    def _conv1x1(self, x, weight, bias, in_ch, out_ch, batch, h, w):
        """1x1 conv on NHWC input."""
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=False,
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
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
            math_approx_mode=False,
        )

        [output, [out_h, out_w], [weight, bias]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=weight,
            bias_tensor=bias,
            in_channels=in_ch,
            out_channels=out_ch,
            device=self.device,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            batch_size=batch,
            input_height=h,
            input_width=w,
            conv_config=conv_config,
            compute_config=compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=ttnn.bfloat16,
        )
        # output = ttnn.sharded_to_interleaved(output, ttnn.DRAM_MEMORY_CONFIG)
        # output = ttnn.reshape(output, (batch, out_h, out_w, out_ch))
        return output, weight, bias

    def __call__(self, feats):
        """
        feats: list of 5 NCHW tensors.
        Returns: (cls_scores, bbox_preds, centernesses) — each list of 5 NCHW tensors.
        """
        cls_scores = []
        bbox_preds = []
        centernesses = []

        for level, feat in enumerate(feats):
            # N, C, H, W = feat.shape
            N, H, W, C = feat.shape
            # nhwc = ttnn.permute(feat, (0, 2, 3, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            # nhwc = ttnn.to_layout(nhwc, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            cls_out, self.cls_weight, self.cls_bias = self._conv1x1(
                # nhwc,
                feat,
                self.cls_weight,
                self.cls_bias,
                in_ch=self.in_channels,
                out_ch=self.num_anchors * self.num_classes,
                batch=N,
                h=H,
                w=W,
            )

            # cls_nchw = ttnn.permute(cls_out, (0, 3, 1, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            # cls_scores.append(cls_nchw)
            cls_scores.append(cls_out)

            reg_out, self.reg_weight, self.reg_bias = self._conv1x1(
                feat,
                self.reg_weight,
                self.reg_bias,
                in_ch=self.in_channels,
                out_ch=self.num_anchors * 4,
                batch=N,
                h=H,
                w=W,
            )
            # reg_out = ttnn.to_layout(reg_out, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            # reg_out = ttnn.multiply(reg_out, self.scales[level], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            reg_out = ttnn.multiply(reg_out, self.scales[level], memory_config=ttnn.L1_MEMORY_CONFIG)
            # reg_nchw = ttnn.permute(reg_out, (0, 3, 1, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            # bbox_preds.append(reg_nchw)
            bbox_preds.append(reg_out)

            cent_out, self.centerness_weight, self.centerness_bias = self._conv1x1(
                # nhwc,
                feat,
                self.centerness_weight,
                self.centerness_bias,
                in_ch=self.in_channels,
                out_ch=self.num_anchors * 1,
                batch=N,
                h=H,
                w=W,
            )
            # cent_nchw = ttnn.permute(cent_out, (0, 3, 1, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            # centernesses.append(cent_nchw)
            centernesses.append(cent_out)

            # ttnn.deallocate(nhwc)
            ttnn.deallocate(feat)

        return cls_scores, bbox_preds, centernesses
