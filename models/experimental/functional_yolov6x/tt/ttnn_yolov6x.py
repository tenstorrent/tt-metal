# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_yolov6x.tt.common import Yolov6x_Conv2D, Yolov6x_Conv_T_2D, deallocate_tensors
from models.experimental.functional_yolov6x.tt.ttnn_sppf import Ttnn_Sppf
from models.experimental.functional_yolov6x.tt.ttnn_detect import Ttnn_Detect


class Ttnn_Yolov6x:
    def __init__(self, device, parameter, model_params):
        self.parameter = parameter
        self.model_params = model_params
        self.conv0 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[0].conv,
            conv_pth=parameter.model[0].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhwc=True,
        )
        self.conv1 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[1].conv,
            conv_pth=parameter.model[1].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.conv2_0 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[2][0].conv,
            conv_pth=parameter.model[2][0].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv2_1 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[2][1].conv,
            conv_pth=parameter.model[2][1].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv2_2 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[2][2].conv,
            conv_pth=parameter.model[2][2].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv2_3 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[2][3].conv,
            conv_pth=parameter.model[2][3].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv2_4 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[2][4].conv,
            conv_pth=parameter.model[2][4].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv2_5 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[2][5].conv,
            conv_pth=parameter.model[2][5].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.conv3 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[3].conv,
            conv_pth=parameter.model[3].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.conv4_0 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[4][0].conv,
            conv_pth=parameter.model[4][0].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv4_1 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[4][1].conv,
            conv_pth=parameter.model[4][1].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv4_2 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[4][2].conv,
            conv_pth=parameter.model[4][2].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv4_3 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[4][3].conv,
            conv_pth=parameter.model[4][3].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv4_4 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[4][4].conv,
            conv_pth=parameter.model[4][4].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv4_5 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[4][5].conv,
            conv_pth=parameter.model[4][5].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv4_6 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[4][6].conv,
            conv_pth=parameter.model[4][6].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv4_7 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[4][7].conv,
            conv_pth=parameter.model[4][7].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv4_8 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[4][8].conv,
            conv_pth=parameter.model[4][8].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv4_9 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[4][9].conv,
            conv_pth=parameter.model[4][9].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv4_10 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[4][10].conv,
            conv_pth=parameter.model[4][10].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv4_11 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[4][11].conv,
            conv_pth=parameter.model[4][11].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.conv5 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[5].conv,
            conv_pth=parameter.model[5].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.conv6_0 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][0].conv,
            conv_pth=parameter.model[6][0].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_1 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][1].conv,
            conv_pth=parameter.model[6][1].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_2 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][2].conv,
            conv_pth=parameter.model[6][2].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_3 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][3].conv,
            conv_pth=parameter.model[6][3].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_4 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][4].conv,
            conv_pth=parameter.model[6][4].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_5 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][5].conv,
            conv_pth=parameter.model[6][5].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_6 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][6].conv,
            conv_pth=parameter.model[6][6].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_7 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][7].conv,
            conv_pth=parameter.model[6][7].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_8 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][8].conv,
            conv_pth=parameter.model[6][8].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_9 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][9].conv,
            conv_pth=parameter.model[6][9].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_10 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][10].conv,
            conv_pth=parameter.model[6][10].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_11 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][11].conv,
            conv_pth=parameter.model[6][11].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_12 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][12].conv,
            conv_pth=parameter.model[6][12].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_13 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][13].conv,
            conv_pth=parameter.model[6][13].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_14 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][14].conv,
            conv_pth=parameter.model[6][14].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_15 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][15].conv,
            conv_pth=parameter.model[6][15].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_16 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][16].conv,
            conv_pth=parameter.model[6][16].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_17 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][17].conv,
            conv_pth=parameter.model[6][17].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.conv7 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[7].conv,
            conv_pth=parameter.model[7].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.conv8_0 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[8][0].conv,
            conv_pth=parameter.model[8][0].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv8_1 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[8][1].conv,
            conv_pth=parameter.model[8][1].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv8_2 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[8][2].conv,
            conv_pth=parameter.model[8][2].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv8_3 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[8][3].conv,
            conv_pth=parameter.model[8][3].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv8_4 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[8][4].conv,
            conv_pth=parameter.model[8][4].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv8_5 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[8][5].conv,
            conv_pth=parameter.model[8][5].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.sppf = Ttnn_Sppf(device, parameter.model[9], model_params.model[9])

        self.conv10 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[10].conv,
            conv_pth=parameter.model[10].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.conv_t_11 = Yolov6x_Conv_T_2D(
            model_params.model[11],
            parameter.model[11],
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            device=device,
        )

        self.conv13 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[13].conv,
            conv_pth=parameter.model[13].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.conv14_0 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[14][0].conv,
            conv_pth=parameter.model[14][0].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv14_1 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[14][1].conv,
            conv_pth=parameter.model[14][1].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv14_2 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[14][2].conv,
            conv_pth=parameter.model[14][2].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv14_3 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[14][3].conv,
            conv_pth=parameter.model[14][3].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv14_4 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[14][4].conv,
            conv_pth=parameter.model[14][4].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv14_5 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[14][5].conv,
            conv_pth=parameter.model[14][5].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv14_6 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[14][6].conv,
            conv_pth=parameter.model[14][6].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv14_7 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[14][7].conv,
            conv_pth=parameter.model[14][7].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv14_8 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[14][8].conv,
            conv_pth=parameter.model[14][8].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.conv15 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[15].conv,
            conv_pth=parameter.model[15].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.conv_t_16 = Yolov6x_Conv_T_2D(
            model_params.model[16],
            parameter.model[16],
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            device=device,
        )

        self.conv18 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[18].conv,
            conv_pth=parameter.model[18].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.conv19_0 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[19][0].conv,
            conv_pth=parameter.model[19][0].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv19_1 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[19][1].conv,
            conv_pth=parameter.model[19][1].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv19_2 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[19][2].conv,
            conv_pth=parameter.model[19][2].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv19_3 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[19][3].conv,
            conv_pth=parameter.model[19][3].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv19_4 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[19][4].conv,
            conv_pth=parameter.model[19][4].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv19_5 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[19][5].conv,
            conv_pth=parameter.model[19][5].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv19_6 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[19][6].conv,
            conv_pth=parameter.model[19][6].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv19_7 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[19][7].conv,
            conv_pth=parameter.model[19][7].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv19_8 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[19][8].conv,
            conv_pth=parameter.model[19][8].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.conv20 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[20].conv,
            conv_pth=parameter.model[20].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv22 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[22].conv,
            conv_pth=parameter.model[22].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.conv23_0 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[23][0].conv,
            conv_pth=parameter.model[23][0].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv23_1 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[23][1].conv,
            conv_pth=parameter.model[23][1].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv23_2 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[23][2].conv,
            conv_pth=parameter.model[23][2].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv23_3 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[23][3].conv,
            conv_pth=parameter.model[23][3].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv23_4 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[23][4].conv,
            conv_pth=parameter.model[23][4].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv23_5 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[23][5].conv,
            conv_pth=parameter.model[23][5].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv23_6 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[23][6].conv,
            conv_pth=parameter.model[23][6].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv23_7 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[23][7].conv,
            conv_pth=parameter.model[23][7].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv23_8 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[23][8].conv,
            conv_pth=parameter.model[23][8].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.conv24 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[24].conv,
            conv_pth=parameter.model[24].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv26 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[26].conv,
            conv_pth=parameter.model[26].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.conv27_0 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[27][0].conv,
            conv_pth=parameter.model[27][0].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv27_1 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[27][1].conv,
            conv_pth=parameter.model[27][1].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv27_2 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[27][2].conv,
            conv_pth=parameter.model[27][2].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv27_3 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[27][3].conv,
            conv_pth=parameter.model[27][3].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv27_4 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[27][4].conv,
            conv_pth=parameter.model[27][4].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv27_5 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[27][5].conv,
            conv_pth=parameter.model[27][5].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv27_6 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[27][6].conv,
            conv_pth=parameter.model[27][6].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv27_7 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[27][7].conv,
            conv_pth=parameter.model[27][7].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv27_8 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[27][8].conv,
            conv_pth=parameter.model[27][8].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.detect = Ttnn_Detect(device, parameter.model[28], model_params.model[28])

    def __call__(self, device, x):
        x0 = self.conv0(x)  # 0.9975022195673564
        x1 = self.conv1(x0)  # 0.999

        x2_0 = self.conv2_0(x1)  # 0.999
        x2_1 = self.conv2_1(x2_0)  # 0.9217840283448264
        x2_2 = self.conv2_2(x2_1)
        x2_3 = self.conv2_3(x2_2)
        x2_4 = self.conv2_4(x2_3)
        x2_5 = self.conv2_5(x2_4)
        deallocate_tensors(x0, x1, x2_0, x2_1, x2_2, x2_3, x2_4)

        x3 = self.conv3(x2_5)

        x4_0 = self.conv4_0(x3)
        x4_1 = self.conv4_1(x4_0)
        x4_2 = self.conv4_2(x4_1)
        x4_3 = self.conv4_3(x4_2)
        x4_4 = self.conv4_4(x4_3)
        x4_5 = self.conv4_5(x4_4)
        x4_6 = self.conv4_6(x4_5)
        x4_7 = self.conv4_7(x4_6)
        x4_8 = self.conv4_8(x4_7)
        x4_9 = self.conv4_9(x4_8)
        x4_10 = self.conv4_10(x4_9)
        x4_11 = self.conv4_11(x4_10)
        x5 = self.conv5(x4_11)
        deallocate_tensors(x2_5, x3, x4_0, x4_1, x4_2, x4_3, x4_4, x4_5, x4_6, x4_7, x4_8, x4_9, x4_10)

        x6_0 = self.conv6_0(x5)
        x6_1 = self.conv6_1(x6_0)
        x6_2 = self.conv6_2(x6_1)
        x6_3 = self.conv6_3(x6_2)
        x6_4 = self.conv6_4(x6_3)
        x6_5 = self.conv6_5(x6_4)
        x6_6 = self.conv6_6(x6_5)
        x6_7 = self.conv6_7(x6_6)
        x6_8 = self.conv6_8(x6_7)
        x6_9 = self.conv6_9(x6_8)
        x6_10 = self.conv6_10(x6_9)
        x6_11 = self.conv6_11(x6_10)
        x6_12 = self.conv6_12(x6_11)
        x6_13 = self.conv6_13(x6_12)
        x6_14 = self.conv6_14(x6_13)
        x6_15 = self.conv6_15(x6_14)
        x6_16 = self.conv6_16(x6_15)
        x6_17 = self.conv6_17(x6_16)  # 0.9796952769406674
        deallocate_tensors(
            x5,
            x6_0,
            x6_1,
            x6_2,
            x6_3,
            x6_4,
            x6_5,
            x6_6,
            x6_7,
            x6_8,
            x6_9,
            x6_10,
            x6_11,
            x6_12,
            x6_13,
            x6_14,
            x6_15,
            x6_16,
        )

        x7 = self.conv7(x6_17)

        x8_0 = self.conv8_0(x7)
        x8_1 = self.conv8_1(x8_0)
        x8_2 = self.conv8_2(x8_1)
        x8_3 = self.conv8_3(x8_2)
        x8_4 = self.conv8_4(x8_3)
        x8_5 = self.conv8_5(x8_4)
        deallocate_tensors(x7, x8_0, x8_1, x8_2, x8_3, x8_4)

        x9 = self.sppf(device, x8_5)

        x10 = self.conv10(x9)  # 0.9923337476011245

        x11 = self.conv_t_11(x10)  # 0.999
        deallocate_tensors(x8_5, x9)

        x11 = ttnn.sharded_to_interleaved(x11, ttnn.L1_MEMORY_CONFIG)
        x12 = ttnn.concat([x11, x6_17], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)

        x13 = self.conv13(x12)
        deallocate_tensors(x11, x12)

        x14_0 = self.conv14_0(x13)
        x14_1 = self.conv14_1(x14_0)
        x14_2 = self.conv14_2(x14_1)
        x14_3 = self.conv14_3(x14_2)
        x14_4 = self.conv14_4(x14_3)
        x14_5 = self.conv14_5(x14_4)
        x14_6 = self.conv14_6(x14_5)
        x14_7 = self.conv14_7(x14_6)
        x14_8 = self.conv14_8(x14_7)  # 0.9970819747146192
        deallocate_tensors(x13, x14_0, x14_1, x14_2, x14_3, x14_4, x14_5, x14_6, x14_7)

        x15 = self.conv15(x14_8)
        x16 = self.conv_t_16(x15)

        x16 = ttnn.sharded_to_interleaved(x16, ttnn.L1_MEMORY_CONFIG)
        x17 = ttnn.concat([x16, x4_11], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        x18 = self.conv18(x17)  # 0.999
        deallocate_tensors(x14_8, x16, x17, x4_11)

        x19_0 = self.conv19_0(x18)
        x19_1 = self.conv19_1(x19_0)
        x19_2 = self.conv19_2(x19_1)
        x19_3 = self.conv19_3(x19_2)
        x19_4 = self.conv19_4(x19_3)
        x19_5 = self.conv19_5(x19_4)
        x19_6 = self.conv19_6(x19_5)
        x19_7 = self.conv19_7(x19_6)
        x19_8 = self.conv19_8(x19_7)
        deallocate_tensors(x18, x19_0, x19_1, x19_2, x19_3, x19_4, x19_5, x19_6, x19_7)

        x20 = self.conv20(x19_8)

        x21 = ttnn.concat([x20, x15], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        x22 = self.conv22(x21)  # 0.996656566141146
        deallocate_tensors(x20, x15, x21)

        x23_0 = self.conv23_0(x22)
        x23_1 = self.conv23_1(x23_0)
        x23_2 = self.conv23_2(x23_1)
        x23_3 = self.conv23_3(x23_2)
        x23_4 = self.conv23_4(x23_3)
        x23_5 = self.conv23_5(x23_4)
        x23_6 = self.conv23_6(x23_5)
        x23_7 = self.conv23_7(x23_6)
        x23_8 = self.conv23_8(x23_7)
        deallocate_tensors(x22, x23_0, x23_1, x23_2, x23_3, x23_4, x23_5, x23_6, x23_7)

        x24 = self.conv24(x23_8)

        x10 = ttnn.from_device(x10)
        x10 = ttnn.to_dtype(x10, dtype=ttnn.bfloat8_b)
        x10 = ttnn.to_device(x10, device)

        x25 = ttnn.concat([x24, x10], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        x26 = self.conv26(x25)
        deallocate_tensors(x24, x25, x10)

        x27_0 = self.conv27_0(x26)
        x27_1 = self.conv27_1(x27_0)
        x27_2 = self.conv27_2(x27_1)
        x27_3 = self.conv27_3(x27_2)
        x27_4 = self.conv27_4(x27_3)
        x27_5 = self.conv27_5(x27_4)
        x27_6 = self.conv27_6(x27_5)
        x27_7 = self.conv27_7(x27_6)
        x27_8 = self.conv27_8(x27_7)  # 0.9937663709395512
        deallocate_tensors(x26, x27_0, x27_1, x27_2, x27_3, x27_4, x27_5, x27_6, x27_7)

        x28 = self.detect(device, x19_8, x23_8, x27_8)

        deallocate_tensors(x19_8, x23_8, x27_8)

        return x28
