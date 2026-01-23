# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from dataclasses import dataclass
from models.tt_cnn.tt.builder import TtConv2d
from .utils import create_conv2d_config, post_process_conv_output


@dataclass
class FPNOptimizations:
    """TTNN implementation of FPNOptimizations"""

    lateral_conv: dict
    fpn_conv: dict
    extra_conv: dict


fpn_optimizations = FPNOptimizations(
    lateral_conv={
        "activation": None,
        "deallocate_activation": False,
        "reallocate_halo_output": False,
        "packer_l1_acc": False,
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
    },
    fpn_conv={
        "activation": None,
        "deallocate_activation": False,
        "reallocate_halo_output": False,
        "packer_l1_acc": False,
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
    },
    extra_conv={
        "activation": None,
        "deallocate_activation": False,
        "reallocate_halo_output": False,
        "packer_l1_acc": False,
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
    },
)


class TtFPN:
    """TTNN implementation of FPN"""

    def __init__(
        self,
        conv_args,
        parameters,
        device,
        in_channels=[512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs="on_output",
        num_outs=5,
        relu_before_extra_convs=True,
        model_config=None,
        optimizations=None,
    ):
        self.device = device
        self.conv_args = conv_args
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level
        self.add_extra_convs = add_extra_convs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.model_config = model_config or {
            "WEIGHTS_DTYPE": ttnn.bfloat16,
            "ACTIVATIONS_DTYPE": ttnn.bfloat16,
            "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
        }
        self.optimizations = optimizations or fpn_optimizations

        self.lateral_weights = []
        self.lateral_biases = []
        for i in range(self.start_level, len(in_channels)):
            lateral_conv = parameters.lateral_convs[i]
            self.lateral_weights.append(lateral_conv["weight"])
            self.lateral_biases.append(
                lateral_conv["bias"] if "bias" in lateral_conv and lateral_conv["bias"] is not None else None
            )

        self.fpn_weights = []
        self.fpn_biases = []
        for i in range(self.start_level, len(in_channels)):
            fpn_conv = parameters.fpn_convs[i]
            self.fpn_weights.append(fpn_conv["weight"])
            self.fpn_biases.append(fpn_conv["bias"] if "bias" in fpn_conv and fpn_conv["bias"] is not None else None)

        extra_levels = num_outs - self.start_level - len(in_channels)
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                idx = len(in_channels) + i
                extra_conv = parameters.fpn_convs[idx]
                self.fpn_weights.append(extra_conv["weight"])
                self.fpn_biases.append(
                    extra_conv["bias"] if "bias" in extra_conv and extra_conv["bias"] is not None else None
                )

    def _get_lateral_conv(self, level_idx, batch_size, height, width):
        opt = self.optimizations.lateral_conv
        out_channels, in_channels = self.lateral_weights[level_idx].shape[0], self.lateral_weights[level_idx].shape[1]
        config = create_conv2d_config(
            input_height=height,
            input_width=width,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            kernel_size=(1, 1),
            weight=self.lateral_weights[level_idx],
            bias=self.lateral_biases[level_idx],
            stride=(1, 1),
            padding=(0, 0),
            model_config=self.model_config,
            activation=opt.get("activation"),
            deallocate_activation=opt.get("deallocate_activation", False),
            reallocate_halo_output=opt.get("reallocate_halo_output", False),
            packer_l1_acc=opt.get("packer_l1_acc", False),
            enable_act_double_buffer=opt.get("enable_act_double_buffer", False),
            enable_weights_double_buffer=opt.get("enable_weights_double_buffer", False),
        )
        return TtConv2d(config, self.device)

    def _get_fpn_conv(self, level_idx, batch_size, height, width):
        opt = self.optimizations.fpn_conv
        out_channels, in_channels = self.fpn_weights[level_idx].shape[0], self.fpn_weights[level_idx].shape[1]
        config = create_conv2d_config(
            input_height=height,
            input_width=width,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            kernel_size=(3, 3),
            weight=self.fpn_weights[level_idx],
            bias=self.fpn_biases[level_idx],
            stride=(1, 1),
            padding=(1, 1),
            model_config=self.model_config,
            activation=opt.get("activation"),
            deallocate_activation=opt.get("deallocate_activation", False),
            reallocate_halo_output=opt.get("reallocate_halo_output", False),
            packer_l1_acc=opt.get("packer_l1_acc", False),
            enable_act_double_buffer=opt.get("enable_act_double_buffer", False),
            enable_weights_double_buffer=opt.get("enable_weights_double_buffer", False),
        )
        return TtConv2d(config, self.device)

    def _get_extra_conv(self, level_idx, batch_size, height, width):
        opt = self.optimizations.extra_conv
        out_channels, in_channels = self.fpn_weights[level_idx].shape[0], self.fpn_weights[level_idx].shape[1]
        config = create_conv2d_config(
            input_height=height,
            input_width=width,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            kernel_size=(3, 3),
            weight=self.fpn_weights[level_idx],
            bias=self.fpn_biases[level_idx],
            stride=(2, 2),
            padding=(1, 1),
            model_config=self.model_config,
            activation=opt.get("activation"),
            deallocate_activation=opt.get("deallocate_activation", False),
            reallocate_halo_output=opt.get("reallocate_halo_output", False),
            packer_l1_acc=opt.get("packer_l1_acc", False),
            enable_act_double_buffer=opt.get("enable_act_double_buffer", False),
            enable_weights_double_buffer=opt.get("enable_weights_double_buffer", False),
        )
        return TtConv2d(config, self.device)

    def __call__(self, inputs, batch_size=1):
        assert len(inputs) == len(self.in_channels)

        laterals = []
        input_heights = []
        input_widths = []

        for i, feat in enumerate(inputs):
            if feat.shape[1] == 1 and feat.shape[2] > 1:
                if feat.is_sharded():
                    feat = ttnn.sharded_to_interleaved(feat, ttnn.DRAM_MEMORY_CONFIG)
                _, _, hw, c = feat.shape
                h = int(hw**0.5)
                w = hw // h
                while h * w != hw and h > 0:
                    h -= 1
                    w = hw // h if h > 0 else hw
                feat = ttnn.reshape(feat, (batch_size, h, w, c))
                if feat.is_sharded():
                    feat = ttnn.sharded_to_interleaved(feat, ttnn.DRAM_MEMORY_CONFIG)
                height, width = h, w
            else:
                height, width = feat.shape[1], feat.shape[2]

            input_heights.append(height)
            input_widths.append(width)

            feat_flat = ttnn.reshape(feat, (1, 1, batch_size * height * width, feat.shape[3]))
            lateral_conv = self._get_lateral_conv(i, batch_size, height, width)
            lateral, _ = lateral_conv(feat_flat, return_output_dim=True)
            lateral = post_process_conv_output(
                lateral, batch_size, height, width, self.out_channels, to_dram=True, reshape_4d=True
            )
            laterals.append(lateral)

        for i in range(len(laterals) - 1, 0, -1):
            upper = laterals[i]
            lower = laterals[i - 1]

            if upper.shape[1] == 1 and upper.shape[2] > 1:
                if upper.is_sharded():
                    upper = ttnn.sharded_to_interleaved(upper, ttnn.DRAM_MEMORY_CONFIG)
                _, _, hw, c = upper.shape
                h = int(hw**0.5)
                w = hw // h
                while h * w != hw and h > 0:
                    h -= 1
                    w = hw // h if h > 0 else hw
                upper = ttnn.reshape(upper, (batch_size, h, w, c))
                if upper.is_sharded():
                    upper = ttnn.sharded_to_interleaved(upper, ttnn.DRAM_MEMORY_CONFIG)

            target_h, target_w = input_heights[i - 1], input_widths[i - 1]

            if upper.shape[1] != target_h or upper.shape[2] != target_w:
                # Use ttnn upsample - repeat each element to match target size
                scale_h = target_h // upper.shape[1]
                scale_w = target_w // upper.shape[2]
                upper = ttnn.repeat_interleave(upper, scale_h, dim=1)
                upper = ttnn.repeat_interleave(upper, scale_w, dim=2)

            if lower.memory_config() != upper.memory_config():
                lower = ttnn.to_memory_config(lower, upper.memory_config())
            if lower.layout != upper.layout:
                lower = ttnn.to_layout(lower, upper.layout)

            laterals[i - 1] = ttnn.add_(lower, upper)

        outs = []
        used_backbone_levels = len(laterals)

        for i in range(used_backbone_levels):
            lateral = laterals[i]
            height, width = input_heights[i], input_widths[i]

            if lateral.shape[1] == 1 and lateral.shape[2] > 1:
                if lateral.is_sharded():
                    lateral = ttnn.sharded_to_interleaved(lateral, ttnn.DRAM_MEMORY_CONFIG)
                _, _, hw, c = lateral.shape
                h = int(hw**0.5)
                w = hw // h
                while h * w != hw and h > 0:
                    h -= 1
                    w = hw // h if h > 0 else hw
                lateral = ttnn.reshape(lateral, (batch_size, h, w, c))
                if lateral.is_sharded():
                    lateral = ttnn.sharded_to_interleaved(lateral, ttnn.DRAM_MEMORY_CONFIG)
                height, width = h, w

            lateral_flat = ttnn.reshape(lateral, (1, 1, batch_size * height * width, lateral.shape[3]))
            fpn_conv = self._get_fpn_conv(i, batch_size, height, width)
            out, _ = fpn_conv(lateral_flat, return_output_dim=True)
            out = post_process_conv_output(
                out, batch_size, height, width, self.out_channels, to_dram=True, reshape_4d=True
            )
            outs.append(out)

        for lateral in laterals:
            if isinstance(lateral, ttnn.Tensor):
                ttnn.deallocate(lateral)

        if self.num_outs > len(outs):
            if self.add_extra_convs == "on_output":
                extra_source = outs[-1]

                if extra_source.shape[1] == 1 and extra_source.shape[2] > 1:
                    if extra_source.is_sharded():
                        extra_source = ttnn.sharded_to_interleaved(extra_source, ttnn.DRAM_MEMORY_CONFIG)
                    _, _, hw, c = extra_source.shape
                    h = int(hw**0.5)
                    w = hw // h
                    while h * w != hw and h > 0:
                        h -= 1
                        w = hw // h if h > 0 else hw
                    extra_source = ttnn.reshape(extra_source, (batch_size, h, w, c))
                    if extra_source.is_sharded():
                        extra_source = ttnn.sharded_to_interleaved(extra_source, ttnn.DRAM_MEMORY_CONFIG)
                    height, width = h, w
                else:
                    height, width = extra_source.shape[1], extra_source.shape[2]

                if used_backbone_levels < len(self.fpn_weights):
                    extra_source_flat = ttnn.reshape(
                        extra_source, (1, 1, batch_size * height * width, extra_source.shape[3])
                    )
                    extra_conv = self._get_extra_conv(used_backbone_levels, batch_size, height, width)
                    extra_out, (out_h, out_w) = extra_conv(extra_source_flat, return_output_dim=True)
                    extra_out = post_process_conv_output(
                        extra_out, batch_size, out_h, out_w, self.out_channels, to_dram=True, reshape_4d=True
                    )
                    outs.append(extra_out)
                    height, width = out_h, out_w

                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        next_input = ttnn.relu(outs[-1])
                    else:
                        next_input = outs[-1]

                    if next_input.shape[1] == 1 and next_input.shape[2] > 1:
                        if next_input.is_sharded():
                            next_input = ttnn.sharded_to_interleaved(next_input, ttnn.DRAM_MEMORY_CONFIG)
                        _, _, hw, c = next_input.shape
                        h = int(hw**0.5)
                        w = hw // h
                        while h * w != hw and h > 0:
                            h -= 1
                            w = hw // h if h > 0 else hw
                        next_input = ttnn.reshape(next_input, (batch_size, h, w, c))
                        if next_input.is_sharded():
                            next_input = ttnn.sharded_to_interleaved(next_input, ttnn.DRAM_MEMORY_CONFIG)
                        height, width = h, w
                    else:
                        height, width = next_input.shape[1], next_input.shape[2]

                    if i < len(self.fpn_weights):
                        next_input_flat = ttnn.reshape(
                            next_input, (1, 1, batch_size * height * width, next_input.shape[3])
                        )
                        extra_conv = self._get_extra_conv(i, batch_size, height, width)
                        extra_out, (out_h, out_w) = extra_conv(next_input_flat, return_output_dim=True)
                        extra_out = post_process_conv_output(
                            extra_out, batch_size, out_h, out_w, self.out_channels, to_dram=True, reshape_4d=True
                        )
                        outs.append(extra_out)
                    else:
                        # Use ttnn max_pool2d for downsampling
                        if next_input.shape[1] == 1 and next_input.shape[2] > 1:
                            _, _, hw, c = next_input.shape
                            h = int(hw**0.5)
                            w = hw // h
                            next_input = ttnn.reshape(next_input, (batch_size, h, w, c))

                        next_input_flat = ttnn.reshape(
                            next_input, (1, 1, batch_size * height * width, next_input.shape[3])
                        )
                        next_input = ttnn.max_pool2d(
                            input_tensor=next_input_flat,
                            batch_size=batch_size,
                            input_h=height,
                            input_w=width,
                            channels=next_input.shape[3],
                            kernel_size=[1, 1],
                            stride=[2, 2],
                            padding=[0, 0],
                            dilation=[1, 1],
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )
                        out_h = height // 2
                        out_w = width // 2
                        next_input = post_process_conv_output(
                            next_input, batch_size, out_h, out_w, self.out_channels, to_dram=True, reshape_4d=True
                        )
                        outs.append(next_input)

        return tuple(outs)
