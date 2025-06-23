# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.yolov6l.tt.ttnn_bepc3 import TtBepC3
from models.experimental.yolov6l.tt.ttnn_sppf import TtSppf
from models.experimental.yolov6l.tt.common import Yolov6l_Conv2D


class TtCSPBepBackbone:
    def __init__(self, device, parameters, model_params):
        self.parameters = parameters
        self.model_params = model_params
        self.stem = Yolov6l_Conv2D(
            device=device,
            conv=model_params.stem.block.conv,
            conv_pth=parameters.stem.block.conv,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            # auto_shard=True,
            activation="silu",
            activation_dtype=ttnn.bfloat16,
            is_nhwc=True,
            reshape=True,
        )
        self.erblock2_0 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.ERBlock_2[0].block.conv,
            conv_pth=parameters.ERBlock_2[0].block.conv,
            shard_layout=None,
            auto_shard=True,
            activation="silu",
            is_nhwc=True,
            reshape=True,
        )
        self.erblock2_1 = TtBepC3(device, parameters.ERBlock_2[1], model_params.ERBlock_2[1], n=6)

        self.erblock3_0 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.ERBlock_3[0].block.conv,
            conv_pth=parameters.ERBlock_3[0].block.conv,
            shard_layout=None,
            auto_shard=True,
            activation="silu",
            is_nhwc=True,
            reshape=True,
        )
        self.erblock3_1 = TtBepC3(device, parameters.ERBlock_3[1], model_params.ERBlock_3[1], n=12)

        self.erblock4_0 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.ERBlock_4[0].block.conv,
            conv_pth=parameters.ERBlock_4[0].block.conv,
            shard_layout=None,
            auto_shard=True,
            activation="silu",
            is_nhwc=True,
            reshape=True,
        )
        self.erblock4_1 = TtBepC3(device, parameters.ERBlock_4[1], model_params.ERBlock_4[1], n=18)

        self.erblock5_0 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.ERBlock_5[0].block.conv,
            conv_pth=parameters.ERBlock_5[0].block.conv,
            shard_layout=None,
            auto_shard=True,
            activation="silu",
            is_nhwc=True,
            reshape=True,
        )
        self.erblock5_1 = TtBepC3(device, parameters.ERBlock_5[1], model_params.ERBlock_5[1], n=6)
        self.erblock5_2 = TtSppf(device, parameters.ERBlock_5[2].sppf, model_params.ERBlock_5[2].sppf)

    def __call__(self, x):
        outputs = []
        stem = self.stem(x)  # 0.9998324299933435
        erblock2_0 = self.erblock2_0(stem)  # 0.9995408313699374
        erblock2_1 = self.erblock2_1(erblock2_0)  # 0.9973235311581851
        outputs.append(erblock2_1)

        erblock3_0 = self.erblock3_0(erblock2_1)  # 0.9980346895595409
        erblock3_1 = self.erblock3_1(erblock3_0)  # 0.9659582791446282
        outputs.append(erblock3_1)

        erblock4_0 = self.erblock4_0(erblock3_1)  # 0.9683885830291584
        erblock4_1 = self.erblock4_1(erblock4_0)  # 0.9733628369619753
        erblock4 = ttnn.clone(erblock4_1)
        outputs.append(erblock4)

        erblock5_0 = self.erblock5_0(erblock4_1)  # 0.9761851206152861
        erblock5_1 = self.erblock5_1(erblock5_0)  # 0.9824179477964243
        erblock5_2 = self.erblock5_2(erblock5_1)  # 0.9840913142957943
        outputs.append(erblock5_2)
        return tuple(outputs)
