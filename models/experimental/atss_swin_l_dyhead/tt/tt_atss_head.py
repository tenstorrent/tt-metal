# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores


class TtATSSHead:
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

        raw_cls_w = ttnn.from_device(parameters["atss_cls"]["weight"])
        raw_cls_b = ttnn.from_device(parameters["atss_cls"]["bias"])
        raw_reg_w = ttnn.from_device(parameters["atss_reg"]["weight"])
        raw_reg_b = ttnn.from_device(parameters["atss_reg"]["bias"])
        raw_cent_w = ttnn.from_device(parameters["atss_centerness"]["weight"])
        raw_cent_b = ttnn.from_device(parameters["atss_centerness"]["bias"])

        self.cls_weights = [raw_cls_w] * num_levels
        self.cls_biases = [raw_cls_b] * num_levels
        self.reg_weights = [raw_reg_w] * num_levels
        self.reg_biases = [raw_reg_b] * num_levels
        self.cent_weights = [raw_cent_w] * num_levels
        self.cent_biases = [raw_cent_b] * num_levels

        self.scales = [parameters["scales"][i] for i in range(num_levels)]

        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
            math_approx_mode=False,
        )

    def _conv1x1(self, x, weight, bias, in_ch, out_ch, batch, h, w):
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=False,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
        )
        grid = self.device.compute_with_storage_grid_size()
        num_cores = min(grid.x * grid.y, batch * h * w)
        if num_cores > 0:
            conv_config.core_grid = get_shard_grid_from_num_cores(num_cores, self.device)
            conv_config.override_sharding_config = True

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
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=ttnn.bfloat16,
        )
        output = ttnn.sharded_to_interleaved(output, ttnn.L1_MEMORY_CONFIG)
        output = ttnn.reshape(output, (batch, out_h, out_w, out_ch))
        return output, weight, bias

    def __call__(self, feats):
        cls_scores = []
        bbox_preds = []
        centernesses = []

        for level, feat in enumerate(feats):
            N, H, W, C = feat.shape

            cls_out, self.cls_weights[level], self.cls_biases[level] = self._conv1x1(
                feat,
                self.cls_weights[level],
                self.cls_biases[level],
                in_ch=self.in_channels,
                out_ch=self.num_anchors * self.num_classes,
                batch=N,
                h=H,
                w=W,
            )
            cls_scores.append(cls_out)

            reg_out, self.reg_weights[level], self.reg_biases[level] = self._conv1x1(
                feat,
                self.reg_weights[level],
                self.reg_biases[level],
                in_ch=self.in_channels,
                out_ch=self.num_anchors * 4,
                batch=N,
                h=H,
                w=W,
            )
            reg_out = ttnn.multiply(reg_out, self.scales[level], memory_config=ttnn.L1_MEMORY_CONFIG)
            bbox_preds.append(reg_out)

            cent_out, self.cent_weights[level], self.cent_biases[level] = self._conv1x1(
                feat,
                self.cent_weights[level],
                self.cent_biases[level],
                in_ch=self.in_channels,
                out_ch=self.num_anchors * 1,
                batch=N,
                h=H,
                w=W,
            )
            centernesses.append(cent_out)

            ttnn.deallocate(feat)

        return cls_scores, bbox_preds, centernesses
