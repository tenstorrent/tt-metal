# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.petr.tt.common import Conv
from models.tt_cnn.tt.builder import TtUpsample, UpsampleConfiguration


class ttnn_ConvModule:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        parameters=None,
    ):
        self.conv = Conv([stride, stride, padding, padding], parameters["conv"])

    def __call__(
        self,
        device,
        x,
    ):
        x = self.conv(device, x)
        return x


class ttnn_CPFPN:
    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        add_extra_convs=False,
        relu_before_extra_convs=False,
        no_norm_on_lateral=False,
        upsample_cfg=dict(mode="nearest"),
        parameters=None,
    ):
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            assert add_extra_convs in ("on_input", "on_lateral", "on_output")
        elif add_extra_convs:
            self.add_extra_convs = "on_input"

        self.lateral_convs = []
        self.fpn_convs = []

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ttnn_ConvModule(in_channels[i], out_channels, 1, parameters=parameters["lateral_convs"][i])
            self.lateral_convs.append(l_conv)
            if i == 0:
                fpn_conv = ttnn_ConvModule(
                    out_channels, out_channels, 3, padding=1, parameters=parameters["fpn_convs"][i]
                )
                self.fpn_convs.append(fpn_conv)

        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == "on_input":
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ttnn_ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                )
                self.fpn_convs.append(extra_fpn_conv)

    def __call__(self, device, inputs):
        assert len(inputs) == len(self.in_channels)

        laterals = [
            lateral_conv(device, inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            batch_size, input_height, input_width, channels = laterals[i].shape
            input_tensor = laterals[i]
            input_tensor = ttnn.to_layout(input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)

            if "scale_factor" in self.upsample_cfg:
                upsample_config = UpsampleConfiguration(
                    input_height=input_height,
                    input_width=input_width,
                    channels=channels,
                    batch_size=batch_size,
                )
                upsample = TtUpsample(upsample_config, device)

            else:
                upsample_config = UpsampleConfiguration(
                    input_height=input_height,
                    input_width=input_width,
                    channels=channels,
                    batch_size=batch_size,
                    scale_factor=(2, 2),
                )

                upsample = TtUpsample(upsample_config, device)

            upsampled = upsample(input_tensor)
            upsampled = ttnn.to_layout(upsampled, layout=ttnn.TILE_LAYOUT)
            laterals[i - 1] += upsampled

        outs = [self.fpn_convs[i](device, laterals[i]) if i == 0 else laterals[i] for i in range(used_backbone_levels)]

        return tuple(outs)
