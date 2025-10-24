# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_petr.tt.common import Conv


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
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ("on_input", "on_lateral", "on_output")
        elif add_extra_convs:  # True
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
            if "scale_factor" in self.upsample_cfg:
                # This is not invoked, So, This part is not tested.
                laterals[i - 1] += ttnn.upsample(laterals[i], **self.upsample_cfg)
            else:
                laterals[i - 1] += ttnn.to_layout(
                    ttnn.upsample(
                        ttnn.to_layout(laterals[i], layout=ttnn.ROW_MAJOR_LAYOUT),
                        scale_factor=(2, 2),
                        **self.upsample_cfg,
                    ),
                    layout=ttnn.TILE_LAYOUT,
                )

        outs = [self.fpn_convs[i](device, laterals[i]) if i == 0 else laterals[i] for i in range(used_backbone_levels)]

        ### This case is not invoked in our flow ###
        # # part 2: add extra levels
        # if self.num_outs > len(outs):
        #     # use max pool to get more levels on top of outputs
        #     # (e.g., Faster R-CNN, Mask R-CNN)
        #     if not self.add_extra_convs:
        #         for i in range(self.num_outs - used_backbone_levels):
        #             outs.append(ttnn.max_pool2d(outs[-1], 1, stride=2))
        #     # add conv layers on top of original feature maps (RetinaNet)
        #     else:
        #         if self.add_extra_convs == "on_input":
        #             extra_source = inputs[self.backbone_end_level - 1]
        #         elif self.add_extra_convs == "on_lateral":
        #             extra_source = laterals[-1]
        #         elif self.add_extra_convs == "on_output":
        #             extra_source = outs[-1]
        #         else:
        #             raise NotImplementedError
        #         outs.append(self.fpn_convs[used_backbone_levels](extra_source))
        #         for i in range(used_backbone_levels + 1, self.num_outs):
        #             if self.relu_before_extra_convs:
        #                 outs.append(self.fpn_convs[i](ttnn.relu(outs[-1])))
        #             else:
        #                 outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
