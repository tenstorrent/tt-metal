# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Union
import torch.nn.functional as F
from models.experimental.functional_pointpillars.tt.common import TtConv
import ttnn


class TtFPN:
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs: Union[bool, str] = False,
        relu_before_extra_convs: bool = False,
        no_norm_on_lateral: bool = False,
        upsample_cfg=dict(mode="nearest"),
        parameters=None,
        device=None,
    ) -> None:
        assert isinstance(in_channels, list)
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
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
            l_conv = TtConv(
                parameters=parameters["lateral_convs"][i]["ConvModule"],
                device=device,
                input_params=[1, 1, 0, out_channels, in_channels[i]],
                activation="relu",
                reshape_tensor=True,
                deallocate_activation=True,
            )

            fpn_conv = TtConv(
                parameters=parameters["fpn_convs"][i]["ConvModule"],
                device=device,
                input_params=[3, 1, 1, out_channels, out_channels],
                activation="relu",
                deallocate_activation=True,
                change_shard=True,
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == "on_input":
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = TtConv(
                    parameters=parameters["fpn_convs"][i]["ConvModule"],
                    device=device,
                    input_params=[3, 2, 1, out_channels, in_channels],
                    activation="relu",
                    deallocate_activation=True,
                    change_shard=True,
                )
                self.fpn_convs.append(extra_fpn_conv)

    def __call__(self, inputs) -> tuple:
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if "scale_factor" in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], **self.upsample_cfg)
            else:
                prev_shape = [laterals[i - 1].shape[1], laterals[i - 1].shape[2]]

                laterals[i - 1] = laterals[i - 1] + ttnn.to_layout(
                    ttnn.upsample(
                        ttnn.to_layout(
                            laterals[i], layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
                        ),
                        scale_factor=prev_shape[0] // laterals[i].shape[1],
                        **self.upsample_cfg,
                    ),
                    layout=ttnn.TILE_LAYOUT,
                )
                laterals[i - 1] = ttnn.to_layout(
                    laterals[i - 1], layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )

        # build outputs
        # part 1: from original levels
        outs = [self.fpn_convs[i](laterals[i])[0] for i in range(used_backbone_levels)]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            assert False, "This is not invoked So, not implemented"
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == "on_input":
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == "on_lateral":
                    extra_source = laterals[-1]
                elif self.add_extra_convs == "on_output":
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source)[0])
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
