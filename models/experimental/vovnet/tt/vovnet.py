# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import timm
import ttnn

from models.experimental.vovnet.tt.classifier_head import TtClassifierHead
from models.experimental.vovnet.tt.conv_norm_act import TtConvNormAct
from models.experimental.vovnet.tt.osa_stage import TtOsaStage
from models.experimental.vovnet.tt.separable_conv_norm_act import TtSeparableConvNormAct

model_cfgs = dict(
    stem_chs=[64, 64, 64],
    stage_conv_chs=[128, 160, 192, 224],
    stage_out_chs=[256, 512, 768, 1024],
    layer_per_block=3,
    block_per_stage=[1, 1, 1, 1],
    residual=True,
    depthwise=True,
    attn="ese",
)


class TtVoVNet(nn.Module):
    def __init__(
        self,
        cfg=model_cfgs,
        in_chans=3,
        num_classes=1000,
        global_pool="avg",
        output_stride=32,
        device=None,
        state_dict=None,
        base_address=None,
    ):
        """
        Args:
            cfg (dict): Model architecture configuration
            in_chans (int): Number of input channels (default: 3)
            num_classes (int): Number of classifier classes (default: 1000)
            global_pool (str): Global pooling type (default: 'avg')
            output_stride (int): Output stride of network, one of (8, 16, 32) (default: 32)
            norm_layer (Union[str, nn.Module]): normalization layer
            act_layer (Union[str, nn.Module]): activation layer
            kwargs (dict): Extra kwargs overlayed onto cfg
        """
        super(TtVoVNet, self).__init__()
        self.num_classes = num_classes
        assert output_stride == 32  # FIXME support dilation
        cfg = dict(cfg)
        stem_stride = cfg.get("stem_stride", 4)
        stem_chs = cfg["stem_chs"]
        stage_conv_chs = cfg["stage_conv_chs"]
        stage_out_chs = cfg["stage_out_chs"]
        block_per_stage = cfg["block_per_stage"]
        layer_per_block = cfg["layer_per_block"]
        self.device = device
        self.state_dict = state_dict
        self.base_address = base_address

        # Stem module
        last_stem_stride = stem_stride // 2
        conv_type = TtSeparableConvNormAct if cfg["depthwise"] else TtConvNormAct
        self.stem = nn.Sequential(
            *[
                TtConvNormAct(
                    in_channels=in_chans,
                    out_channels=stem_chs[0],
                    kernel_size=3,
                    stride=2,
                    base_address=f"stem.0",
                    device=self.device,
                    state_dict=self.state_dict,
                ),
                conv_type(
                    in_channels=stem_chs[0],
                    out_channels=stem_chs[1],
                    kernel_size=3,
                    stride=1,
                    groups=64,
                    base_address=f"stem.1",
                    device=device,
                    state_dict=state_dict,
                ),
                conv_type(
                    in_channels=stem_chs[1],
                    out_channels=stem_chs[2],
                    kernel_size=3,
                    stride=last_stem_stride,
                    groups=64,
                    base_address=f"stem.2",
                    device=device,
                    state_dict=state_dict,
                ),
            ]
        )

        current_stride = stem_stride

        in_ch_list = stem_chs[-1:] + stage_out_chs[:-1]

        stages = []
        for i in range(4):  # num_stages
            downsample = stem_stride == 2 or i > 0  # first stage has no stride/downsample if stem_stride is 4
            stages += [
                TtOsaStage(
                    in_chs=in_ch_list[i],
                    mid_chs=stage_conv_chs[i],
                    out_chs=stage_out_chs[i],
                    block_per_stage=block_per_stage[i],
                    layer_per_block=layer_per_block,
                    downsample=downsample,
                    residual=False,
                    depthwise=True,
                    base_address=f"stages.{i}",
                    state_dict=self.state_dict,
                    device=self.device,
                    groups=stage_conv_chs[i],
                )
            ]
            self.num_features = stage_out_chs[i]
            current_stride *= 2 if downsample else 1

        self.stages = nn.Sequential(*stages)

        self.head = TtClassifierHead(
            in_features=1024,
            num_classes=1000,
            pool_type=global_pool,
            base_address=f"head",
            device=self.device,
            state_dict=self.state_dict,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.head(x)
        return x


def _vovnet(device, state_dict) -> TtVoVNet:
    tt_model = TtVoVNet(
        state_dict=state_dict,
        device=device,
    )
    return tt_model


def vovnet_for_image_classification(device) -> TtVoVNet:
    model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True)
    state_dict = model.state_dict()
    tt_model = _vovnet(device=device, state_dict=state_dict)
    return tt_model
