# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import sys
import pytest
import torch
import ttnn
from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d, preprocess_model_parameters
from models.experimental.yolov7.reference import yolov7_model, yolov7_utils
from models.experimental.yolov7.reference.model import Yolov7_model
from models.experimental.yolov7.reference.yolov7_utils import download_yolov7_weights
from models.experimental.yolov7.tt.ttnn_yolov7 import ttnn_yolov7
from tests.ttnn.utils_for_testing import assert_with_pcc

sys.modules["models.common"] = sys.modules["models.experimental.yolov7.reference.yolov7_utils"]
sys.modules["models.yolo"] = sys.modules["models.experimental.yolov7.reference.yolov7_model"]


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, Yolov7_model):
            parameters["0"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[0].conv, model.model[0].bn)
            parameters["0"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["0"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["1"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[1].conv, model.model[1].bn)
            parameters["1"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["1"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["2"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[2].conv, model.model[2].bn)
            parameters["2"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["2"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["3"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[3].conv, model.model[3].bn)
            parameters["3"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["3"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["4"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[4].conv, model.model[4].bn)
            parameters["4"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["4"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["5"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[5].conv, model.model[5].bn)
            parameters["5"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["5"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["6"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[6].conv, model.model[6].bn)
            parameters["6"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["6"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["7"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[7].conv, model.model[7].bn)
            parameters["7"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["7"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["8"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[8].conv, model.model[8].bn)
            parameters["8"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["8"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["9"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[9].conv, model.model[9].bn)
            parameters["9"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["9"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["11"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[11].conv, model.model[11].bn)
            parameters["11"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["11"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["13"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[13].conv, model.model[13].bn)
            parameters["13"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["13"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["14"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[14].conv, model.model[14].bn)
            parameters["14"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["14"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["15"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[15].conv, model.model[15].bn)
            parameters["15"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["15"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["17"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[17].conv, model.model[17].bn)
            parameters["17"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["17"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["18"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[18].conv, model.model[18].bn)
            parameters["18"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["18"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["19"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[19].conv, model.model[19].bn)
            parameters["19"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["19"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["20"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[20].conv, model.model[20].bn)
            parameters["20"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["20"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["21"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[21].conv, model.model[21].bn)
            parameters["21"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["21"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["22"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[22].conv, model.model[22].bn)
            parameters["22"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["22"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["24"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[24].conv, model.model[24].bn)
            parameters["24"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["24"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["26"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[26].conv, model.model[26].bn)
            parameters["26"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["26"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["27"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[27].conv, model.model[27].bn)
            parameters["27"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["27"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["28"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[28].conv, model.model[28].bn)
            parameters["28"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["28"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["30"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[30].conv, model.model[30].bn)
            parameters["30"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["30"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["31"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[31].conv, model.model[31].bn)
            parameters["31"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["31"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["32"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[32].conv, model.model[32].bn)
            parameters["32"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["32"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["33"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[33].conv, model.model[33].bn)
            parameters["33"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["33"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["34"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[34].conv, model.model[34].bn)
            parameters["34"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["34"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["35"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[35].conv, model.model[35].bn)
            parameters["35"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["35"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["37"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[37].conv, model.model[37].bn)
            parameters["37"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["37"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["39"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[39].conv, model.model[39].bn)
            parameters["39"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["39"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["40"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[40].conv, model.model[40].bn)
            parameters["40"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["40"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["41"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[41].conv, model.model[41].bn)
            parameters["41"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["41"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["43"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[43].conv, model.model[43].bn)
            parameters["43"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["43"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["44"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[44].conv, model.model[44].bn)
            parameters["44"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["44"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["45"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[45].conv, model.model[45].bn)
            parameters["45"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["45"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["46"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[46].conv, model.model[46].bn)
            parameters["46"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["46"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["47"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[47].conv, model.model[47].bn)
            parameters["47"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["47"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["48"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[48].conv, model.model[48].bn)
            parameters["48"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["48"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["50"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[50].conv, model.model[50].bn)
            parameters["50"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["50"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["51"] = {}
            parameters["51"]["cv1"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[51].cv1.conv, model.model[51].cv1.bn)
            parameters["51"]["cv1"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["51"]["cv1"]["bias"] = ttnn.from_torch(
                torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
            )

            parameters["51"]["cv2"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[51].cv2.conv, model.model[51].cv2.bn)
            parameters["51"]["cv2"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["51"]["cv2"]["bias"] = ttnn.from_torch(
                torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
            )

            parameters["51"]["cv3"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[51].cv3.conv, model.model[51].cv3.bn)
            parameters["51"]["cv3"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["51"]["cv3"]["bias"] = ttnn.from_torch(
                torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
            )

            parameters["51"]["cv4"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[51].cv4.conv, model.model[51].cv4.bn)
            parameters["51"]["cv4"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["51"]["cv4"]["bias"] = ttnn.from_torch(
                torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
            )

            parameters["51"]["cv5"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[51].cv5.conv, model.model[51].cv5.bn)
            parameters["51"]["cv5"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["51"]["cv5"]["bias"] = ttnn.from_torch(
                torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
            )

            parameters["51"]["cv6"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[51].cv6.conv, model.model[51].cv6.bn)
            parameters["51"]["cv6"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["51"]["cv6"]["bias"] = ttnn.from_torch(
                torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
            )

            parameters["51"]["cv7"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[51].cv7.conv, model.model[51].cv7.bn)
            parameters["51"]["cv7"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["51"]["cv7"]["bias"] = ttnn.from_torch(
                torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
            )

            parameters["52"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[52].conv, model.model[52].bn)
            parameters["52"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["52"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["54"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[54].conv, model.model[54].bn)
            parameters["54"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["54"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["56"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[56].conv, model.model[56].bn)
            parameters["56"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["56"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["57"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[57].conv, model.model[57].bn)
            parameters["57"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["57"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["58"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[58].conv, model.model[58].bn)
            parameters["58"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["58"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["59"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[59].conv, model.model[59].bn)
            parameters["59"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["59"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["60"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[60].conv, model.model[60].bn)
            parameters["60"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["60"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["61"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[61].conv, model.model[61].bn)
            parameters["61"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["61"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["63"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[63].conv, model.model[63].bn)
            parameters["63"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["63"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["64"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[64].conv, model.model[64].bn)
            parameters["64"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["64"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["66"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[66].conv, model.model[66].bn)
            parameters["66"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["66"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["68"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[68].conv, model.model[68].bn)
            parameters["68"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["68"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["69"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[69].conv, model.model[69].bn)
            parameters["69"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["69"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["70"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[70].conv, model.model[70].bn)
            parameters["70"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["70"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["71"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[71].conv, model.model[71].bn)
            parameters["71"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["71"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["72"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[72].conv, model.model[72].bn)
            parameters["72"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["72"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["73"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[73].conv, model.model[73].bn)
            parameters["73"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["73"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["75"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[75].conv, model.model[75].bn)
            parameters["75"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["75"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["77"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[77].conv, model.model[77].bn)
            parameters["77"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["77"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["78"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[78].conv, model.model[78].bn)
            parameters["78"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["78"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["79"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[79].conv, model.model[79].bn)
            parameters["79"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["79"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["81"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[81].conv, model.model[81].bn)
            parameters["81"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["81"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["82"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[82].conv, model.model[82].bn)
            parameters["82"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["82"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["83"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[83].conv, model.model[83].bn)
            parameters["83"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["83"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["84"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[84].conv, model.model[84].bn)
            parameters["84"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["84"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["85"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[85].conv, model.model[85].bn)
            parameters["85"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["85"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["86"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[86].conv, model.model[86].bn)
            parameters["86"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["86"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["88"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[88].conv, model.model[88].bn)
            parameters["88"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["88"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["90"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[90].conv, model.model[90].bn)
            parameters["90"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["90"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["91"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[91].conv, model.model[91].bn)
            parameters["91"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["91"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["92"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[92].conv, model.model[92].bn)
            parameters["92"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["92"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["94"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[94].conv, model.model[94].bn)
            parameters["94"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["94"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["95"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[95].conv, model.model[95].bn)
            parameters["95"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["95"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["96"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[96].conv, model.model[96].bn)
            parameters["96"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["96"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["97"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[97].conv, model.model[97].bn)
            parameters["97"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["97"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["98"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[98].conv, model.model[98].bn)
            parameters["98"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["98"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["99"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[99].conv, model.model[99].bn)
            parameters["99"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["99"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["101"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.model[101].conv, model.model[101].bn)
            parameters["101"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["101"]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)

            parameters["102"] = {}
            parameters["102"]["0"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(
                model.model[102].rbr_dense[0], model.model[102].rbr_dense[1]
            )
            parameters["102"]["0"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["102"]["0"]["bias"] = ttnn.from_torch(
                torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
            )

            parameters["102"]["1"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(
                model.model[102].rbr_1x1[0], model.model[102].rbr_1x1[1]
            )
            parameters["102"]["1"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["102"]["1"]["bias"] = ttnn.from_torch(
                torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
            )

            parameters["103"] = {}
            parameters["103"]["0"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(
                model.model[103].rbr_dense[0], model.model[103].rbr_dense[1]
            )
            parameters["103"]["0"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["103"]["0"]["bias"] = ttnn.from_torch(
                torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
            )

            parameters["103"]["1"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(
                model.model[103].rbr_1x1[0], model.model[103].rbr_1x1[1]
            )
            parameters["103"]["1"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["103"]["1"]["bias"] = ttnn.from_torch(
                torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
            )

            parameters["104"] = {}
            parameters["104"]["0"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(
                model.model[104].rbr_dense[0], model.model[104].rbr_dense[1]
            )
            parameters["104"]["0"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["104"]["0"]["bias"] = ttnn.from_torch(
                torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
            )

            parameters["104"]["1"] = {}
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(
                model.model[104].rbr_1x1[0], model.model[104].rbr_1x1[1]
            )
            parameters["104"]["1"]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            parameters["104"]["1"]["bias"] = ttnn.from_torch(
                torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
            )

            parameters["105"] = {}
            parameters["105"]["0"] = {}
            parameters["105"]["0"]["weight"] = ttnn.from_torch(model.model[105].m[0].weight, dtype=ttnn.bfloat16)
            parameters["105"]["0"]["bias"] = ttnn.from_torch(
                torch.reshape(model.model[105].m[0].bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
            )

            parameters["105"]["1"] = {}
            parameters["105"]["1"]["weight"] = ttnn.from_torch(model.model[105].m[1].weight, dtype=ttnn.bfloat16)
            parameters["105"]["1"]["bias"] = ttnn.from_torch(
                torch.reshape(model.model[105].m[1].bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
            )

            parameters["105"]["2"] = {}
            parameters["105"]["2"]["weight"] = ttnn.from_torch(model.model[105].m[2].weight, dtype=ttnn.bfloat16)
            parameters["105"]["2"]["bias"] = ttnn.from_torch(
                torch.reshape(model.model[105].m[2].bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
            )

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_yolov7(device, reset_seeds):
    def load_weights(model, weights_path):
        ckpt = torch.load(weights_path, map_location="cpu")
        state_dict = ckpt["model"].float().state_dict()
        model.load_state_dict(state_dict, strict=False)

    torch_model = Yolov7_model()

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    ds_state_dict = {k: v for k, v in torch_model.state_dict().items()}
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    torch_input_tensor = torch.randn(1, 3, 640, 640)
    weights_path = "tests/ttnn/integration_tests/yolov7/yolov7.pt"
    weights_path = download_yolov7_weights(weights_path)
    load_weights(torch_model, weights_path)
    torch_output_tensor = torch_model(torch_input_tensor)

    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(None), device=None
    )

    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    grid = [torch.randn(1)] * 3
    nx_ny = [80, 40, 20]
    grid_tensors = []
    for i in range(3):
        yv, xv = torch.meshgrid([torch.arange(nx_ny[i]), torch.arange(nx_ny[i])])
        grid_tensors.append(torch.stack((xv, yv), 2).view((1, 1, nx_ny[i], nx_ny[i], 2)).float())

    ttnn_model = ttnn_yolov7(device, parameters, grid_tensors)
    output = ttnn_model(ttnn_input)[0]

    output = ttnn.to_torch(output)
    assert_with_pcc(torch_output_tensor[0], output, pcc=0.999)
