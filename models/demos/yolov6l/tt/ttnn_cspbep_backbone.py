# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov6l.tt.common import Yolov6l_Conv2D
from models.demos.yolov6l.tt.ttnn_bepc3 import TtBepC3
from models.demos.yolov6l.tt.ttnn_sppf import TtSppf


class TtCSPBepBackbone:
    def __init__(self, device, parameters, model_params):
        self.parameters = parameters
        self.model_params = model_params
        self.stem = Yolov6l_Conv2D(
            device=device,
            conv=model_params.stem.block.conv,
            conv_pth=parameters.stem.block.conv,
            activation="silu",
            activation_dtype=ttnn.bfloat16,
            reshape=True,
            deallocate_activation=True,
        )
        self.erblock2_0 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.ERBlock_2[0].block.conv,
            conv_pth=parameters.ERBlock_2[0].block.conv,
            activation="silu",
            reshape=True,
        )
        self.erblock2_1 = TtBepC3(device, parameters.ERBlock_2[1], model_params.ERBlock_2[1], n=6)

        self.erblock3_0 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.ERBlock_3[0].block.conv,
            conv_pth=parameters.ERBlock_3[0].block.conv,
            activation="silu",
            reshape=True,
        )
        self.erblock3_1 = TtBepC3(device, parameters.ERBlock_3[1], model_params.ERBlock_3[1], n=12)

        self.erblock4_0 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.ERBlock_4[0].block.conv,
            conv_pth=parameters.ERBlock_4[0].block.conv,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            activation="silu",
            reshape=True,
        )
        self.erblock4_1 = TtBepC3(device, parameters.ERBlock_4[1], model_params.ERBlock_4[1], n=18)

        self.erblock5_0 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.ERBlock_5[0].block.conv,
            conv_pth=parameters.ERBlock_5[0].block.conv,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            activation="silu",
            reshape=True,
        )
        self.erblock5_1 = TtBepC3(
            device,
            parameters.ERBlock_5[1],
            model_params.ERBlock_5[1],
            n=6,
            shard_layout_cv2=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            shard_layout_rep_block_first_two=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )
        self.erblock5_2 = TtSppf(device, parameters.ERBlock_5[2].sppf, model_params.ERBlock_5[2].sppf)

    def __call__(self, x):
        outputs = []
        stem = self.stem(x)
        erblock2_0 = self.erblock2_0(stem)
        erblock2_1 = self.erblock2_1(erblock2_0)
        outputs.append(erblock2_1)

        erblock3_0 = self.erblock3_0(erblock2_1)
        erblock3_1 = self.erblock3_1(erblock3_0)
        outputs.append(erblock3_1)

        erblock4_0 = self.erblock4_0(erblock3_1)
        erblock4_1 = self.erblock4_1(erblock4_0)
        erblock4 = ttnn.clone(erblock4_1)
        outputs.append(erblock4)

        erblock5_0 = self.erblock5_0(erblock4_1)
        erblock5_1 = self.erblock5_1(erblock5_0)
        erblock5_2 = self.erblock5_2(erblock5_1)
        outputs.append(erblock5_2)
        return tuple(outputs)
