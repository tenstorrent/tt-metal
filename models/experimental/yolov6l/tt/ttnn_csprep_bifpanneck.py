# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.yolov6l.tt.ttnn_bifusion import TtBiFusion
from models.experimental.yolov6l.tt.ttnn_bepc3 import TtBepC3
from models.experimental.yolov6l.tt.common import Yolov6l_Conv2D


class TtCSPRepBiFPANNeck:
    def __init__(self, device, parameters, model_params):
        self.parameters = parameters
        self.model_params = model_params
        self.reduce_layer0 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.reduce_layer0.block.conv,
            conv_pth=parameters.reduce_layer0.block.conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhwc=True,
            reshape=True,
        )
        self.Bifusion0 = TtBiFusion(device, parameters.Bifusion0, model_params.Bifusion0)
        self.Rep_p4 = TtBepC3(device, parameters.Rep_p4, model_params.Rep_p4, n=12)

        self.reduce_layer1 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.reduce_layer1.block.conv,
            conv_pth=parameters.reduce_layer1.block.conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhwc=True,
            reshape=True,
        )
        self.Bifusion1 = TtBiFusion(device, parameters.Bifusion1, model_params.Bifusion1)
        self.Rep_p3 = TtBepC3(device, parameters.Rep_p3, model_params.Rep_p3, n=12)

        self.downsample2 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.downsample2.block.conv,
            conv_pth=parameters.downsample2.block.conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhwc=True,
            reshape=True,
        )
        self.Rep_n3 = TtBepC3(device, parameters.Rep_n3, model_params.Rep_n3, n=12)

        self.downsample1 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.downsample1.block.conv,
            conv_pth=parameters.downsample1.block.conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhwc=True,
            reshape=True,
        )
        self.Rep_n4 = TtBepC3(device, parameters.Rep_n4, model_params.Rep_n4, n=12)

    def __call__(self, input_list):
        (input_tensor_3, input_tensor_2, input_tensor_1, input_tensor_0) = input_list

        fpn_out0 = self.reduce_layer0(input_tensor_0)
        f_concat_layer0 = self.Bifusion0([fpn_out0, input_tensor_1, input_tensor_2])
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        f_concat_layer1 = self.Bifusion1([fpn_out1, input_tensor_2, input_tensor_3])
        pan_out2 = self.Rep_p3(f_concat_layer1)

        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = ttnn.concat([down_feat1, fpn_out1], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        pan_out1 = self.Rep_n3(p_concat_layer1)
        pan_out_1 = ttnn.clone(pan_out1)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = ttnn.concat([down_feat0, fpn_out0], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        pan_out0 = self.Rep_n4(p_concat_layer2)

        outputs = [pan_out2, pan_out_1, pan_out0]

        return outputs
