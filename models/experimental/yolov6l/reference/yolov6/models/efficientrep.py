# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from torch import nn
from yolov6l import RepVGGBlock, BepC3, SPPF, ConvBNSiLU


class CSPBepBackbone(nn.Module):
    """
    CSPBepBackbone module.
    """

    def __init__(
        self,
        in_channels=3,
        channels_list=None,
        num_repeats=None,
        block=RepVGGBlock,
        csp_e=float(1) / 2,
        fuse_P2=False,
        cspsppf=False,
        stage_block_type="BepC3",
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        if stage_block_type == "BepC3":
            stage_block = BepC3
        elif stage_block_type == "MBLABlock":
            stage_block = MBLABlock
        else:
            raise NotImplementedError

        self.fuse_P2 = fuse_P2

        self.stem = block(in_channels=in_channels, out_channels=channels_list[0], kernel_size=3, stride=2)

        self.ERBlock_2 = nn.Sequential(
            block(in_channels=channels_list[0], out_channels=channels_list[1], kernel_size=3, stride=2),
            stage_block(
                in_channels=channels_list[1],
                out_channels=channels_list[1],
                n=num_repeats[1],
                e=csp_e,
                block=block,
            ),
        )

        self.ERBlock_3 = nn.Sequential(
            block(in_channels=channels_list[1], out_channels=channels_list[2], kernel_size=3, stride=2),
            stage_block(
                in_channels=channels_list[2],
                out_channels=channels_list[2],
                n=num_repeats[2],
                e=csp_e,
                block=block,
            ),
        )

        self.ERBlock_4 = nn.Sequential(
            block(in_channels=channels_list[2], out_channels=channels_list[3], kernel_size=3, stride=2),
            stage_block(
                in_channels=channels_list[3],
                out_channels=channels_list[3],
                n=num_repeats[3],
                e=csp_e,
                block=block,
            ),
        )

        channel_merge_layer = SPPF if block == ConvBNSiLU else SimSPPF
        if cspsppf:
            channel_merge_layer = CSPSPPF if block == ConvBNSiLU else SimCSPSPPF

        self.ERBlock_5 = nn.Sequential(
            block(
                in_channels=channels_list[3],
                out_channels=channels_list[4],
                kernel_size=3,
                stride=2,
            ),
            stage_block(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                n=num_repeats[4],
                e=csp_e,
                block=block,
            ),
            channel_merge_layer(in_channels=channels_list[4], out_channels=channels_list[4], kernel_size=5),
        )

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        if self.fuse_P2:
            outputs.append(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)

        return tuple(outputs)
