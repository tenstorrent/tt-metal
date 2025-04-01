# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_yolov6x.tt.common import Yolov6x_Conv2D, deallocate_tensors


class Ttnn_Detect:
    def __init__(self, device, parameter, model_params):
        self.cv2_0_0 = Yolov6x_Conv2D(
            model_params.cv2[0][0].conv,
            parameter.cv2[0][0].conv,
            auto_shard=True,
            shard_layout=None,
            activation="relu",
            device=device,
            is_nhw_c=True,
        )
        self.cv2_0_1 = Yolov6x_Conv2D(
            model_params.cv2[0][1].conv,
            parameter.cv2[0][1].conv,
            auto_shard=True,
            shard_layout=None,
            activation="relu",
            device=device,
            is_nhw_c=True,
        )
        self.cv2_0_2 = Yolov6x_Conv2D(
            model_params.cv2[0][2],
            parameter.cv2[0][2],
            auto_shard=True,
            shard_layout=None,
            device=device,
            is_nhw_c=True,
        )

        self.cv2_1_0 = Yolov6x_Conv2D(
            model_params.cv2[1][0].conv,
            parameter.cv2[1][0].conv,
            auto_shard=True,
            shard_layout=None,
            activation="relu",
            device=device,
            is_nhw_c=True,
        )
        self.cv2_1_1 = Yolov6x_Conv2D(
            model_params.cv2[1][1].conv,
            parameter.cv2[1][1].conv,
            auto_shard=True,
            shard_layout=None,
            activation="relu",
            device=device,
            is_nhw_c=True,
        )
        self.cv2_1_2 = Yolov6x_Conv2D(
            model_params.cv2[1][2],
            parameter.cv2[1][2],
            auto_shard=True,
            shard_layout=None,
            device=device,
            is_nhw_c=True,
        )

        self.cv2_2_0 = Yolov6x_Conv2D(
            model_params.cv2[2][0].conv,
            parameter.cv2[2][0].conv,
            auto_shard=True,
            shard_layout=None,
            activation="relu",
            device=device,
            is_nhw_c=True,
        )
        self.cv2_2_1 = Yolov6x_Conv2D(
            model_params.cv2[2][1].conv,
            parameter.cv2[2][1].conv,
            auto_shard=True,
            shard_layout=None,
            activation="relu",
            device=device,
            is_nhw_c=True,
        )
        self.cv2_2_2 = Yolov6x_Conv2D(
            model_params.cv2[2][2],
            parameter.cv2[2][2],
            auto_shard=True,
            shard_layout=None,
            device=device,
            is_nhw_c=True,
        )

        self.cv3_0_0 = Yolov6x_Conv2D(
            model_params.cv3[0][0].conv,
            parameter.cv3[0][0].conv,
            auto_shard=True,
            shard_layout=None,
            activation="relu",
            device=device,
            is_nhw_c=True,
        )
        self.cv3_0_1 = Yolov6x_Conv2D(
            model_params.cv3[0][1].conv,
            parameter.cv3[0][1].conv,
            auto_shard=True,
            shard_layout=None,
            activation="relu",
            device=device,
            is_nhw_c=True,
        )
        self.cv3_0_2 = Yolov6x_Conv2D(
            model_params.cv3[0][2],
            parameter.cv3[0][2],
            auto_shard=True,
            shard_layout=None,
            device=device,
            is_nhw_c=True,
        )

        self.cv3_1_0 = Yolov6x_Conv2D(
            model_params.cv3[1][0].conv,
            parameter.cv3[1][0].conv,
            auto_shard=True,
            shard_layout=None,
            activation="relu",
            device=device,
            is_nhw_c=True,
        )
        self.cv3_1_1 = Yolov6x_Conv2D(
            model_params.cv3[1][1].conv,
            parameter.cv3[1][1].conv,
            auto_shard=True,
            shard_layout=None,
            activation="relu",
            device=device,
            is_nhw_c=True,
        )
        self.cv3_1_2 = Yolov6x_Conv2D(
            model_params.cv3[1][2],
            parameter.cv3[1][2],
            auto_shard=True,
            shard_layout=None,
            device=device,
            is_nhw_c=True,
        )

        self.cv3_2_0 = Yolov6x_Conv2D(
            model_params.cv3[2][0].conv,
            parameter.cv3[2][0].conv,
            auto_shard=True,
            shard_layout=None,
            activation="relu",
            device=device,
            is_nhw_c=True,
        )
        self.cv3_2_1 = Yolov6x_Conv2D(
            model_params.cv3[2][1].conv,
            parameter.cv3[2][1].conv,
            auto_shard=True,
            shard_layout=None,
            activation="relu",
            device=device,
            is_nhw_c=True,
        )
        self.cv3_2_2 = Yolov6x_Conv2D(
            model_params.cv3[2][2],
            parameter.cv3[2][2],
            auto_shard=True,
            shard_layout=None,
            device=device,
            is_nhw_c=True,
        )

        self.dfl = Yolov6x_Conv2D(
            model_params.dfl.conv, parameter.dfl.conv, auto_shard=True, shard_layout=None, device=device, is_nhw_c=True
        )
        self.anchors = parameter.anchors
        self.strides = parameter.strides

    def __call__(self, device, y1, y2, y3):
        x1 = self.cv2_0_0(y1)
        x1 = self.cv2_0_1(x1)
        x1 = self.cv2_0_2(x1)

        x2 = self.cv2_1_0(y2)
        x2 = self.cv2_1_1(x2)
        x2 = self.cv2_1_2(x2)

        x3 = self.cv2_2_0(y3)
        x3 = self.cv2_2_1(x3)
        x3 = self.cv2_2_2(x3)

        x4 = self.cv3_0_0(y1)
        x4 = self.cv3_0_1(x4)
        x4 = self.cv3_0_2(x4)

        x5 = self.cv3_1_0(y2)
        x5 = self.cv3_1_1(x5)
        x5 = self.cv3_1_2(x5)

        x6 = self.cv3_2_0(y3)
        x6 = self.cv3_2_1(x6)
        x6 = self.cv3_2_2(x6)

        x1 = ttnn.sharded_to_interleaved(x1, memory_config=ttnn.L1_MEMORY_CONFIG)
        x2 = ttnn.sharded_to_interleaved(x2, memory_config=ttnn.L1_MEMORY_CONFIG)
        x3 = ttnn.sharded_to_interleaved(x3, memory_config=ttnn.L1_MEMORY_CONFIG)
        x4 = ttnn.sharded_to_interleaved(x4, memory_config=ttnn.L1_MEMORY_CONFIG)
        x5 = ttnn.sharded_to_interleaved(x5, memory_config=ttnn.L1_MEMORY_CONFIG)
        x6 = ttnn.sharded_to_interleaved(x6, memory_config=ttnn.L1_MEMORY_CONFIG)

        y1 = ttnn.concat((x1, x4), -1, memory_config=ttnn.L1_MEMORY_CONFIG)
        y2 = ttnn.concat((x2, x5), -1, memory_config=ttnn.L1_MEMORY_CONFIG)
        y3 = ttnn.concat((x3, x6), -1, memory_config=ttnn.L1_MEMORY_CONFIG)

        y = ttnn.concat((y1, y2, y3), dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
        y = ttnn.squeeze(y, dim=0)

        ya, yb = y[:, :, :64], y[:, :, 64:144]

        deallocate_tensors(y1, y2, y3, x1, x2, x3, x4, x5, x6, y)

        ya = ttnn.reshape(ya, (ya.shape[0], y.shape[1], 4, 16))
        ya = ttnn.permute(ya, (0, 2, 1, 3))  # 0.999
        ya = ttnn.softmax(ya, dim=-1, numeric_stable=True)  # 0.9949745397952091
        c = self.dfl(ya)  # 0.9654762051555557 (0.9654096135662267 after auto shard)

        ttnn.deallocate(ya)
        c = ttnn.sharded_to_interleaved(c, memory_config=ttnn.L1_MEMORY_CONFIG)
        c = ttnn.to_layout(c, layout=ttnn.ROW_MAJOR_LAYOUT)
        c = ttnn.permute(c, (0, 3, 1, 2))
        c = ttnn.reshape(c, (c.shape[0], 1, 4, int(c.shape[3] / 4)))
        c = ttnn.reshape(c, (c.shape[0], c.shape[1] * c.shape[2], c.shape[3]))

        c1, c2 = c[:, :2, :], c[:, 2:4, :]
        anchor, strides = self.anchors, self.strides
        anchor = ttnn.to_memory_config(anchor, memory_config=ttnn.L1_MEMORY_CONFIG)
        strides = ttnn.to_memory_config(strides, memory_config=ttnn.L1_MEMORY_CONFIG)
        c1 = ttnn.to_layout(c1, layout=ttnn.TILE_LAYOUT)
        c2 = ttnn.to_layout(c2, layout=ttnn.TILE_LAYOUT)

        c1 = anchor - c1  # 0.999
        c2 = anchor + c2  # 0.999
        z2 = c1 + c2
        z2 = ttnn.div(z2, 2)  # 0.999
        z1 = c2 - c1  # 0.7676696715044726
        z = ttnn.concat((z2, z1), dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)  # 0.999
        z = ttnn.multiply(z, strides)

        yb = ttnn.permute(yb, (0, 2, 1))
        yb = ttnn.sigmoid(yb)  # 0.9987866613751747

        deallocate_tensors(c, z1, z2, c1, c2, anchor, strides)

        z = ttnn.reallocate(z)
        yb = ttnn.reallocate(yb)

        z = ttnn.to_layout(z, layout=ttnn.ROW_MAJOR_LAYOUT)
        yb = ttnn.to_layout(yb, layout=ttnn.ROW_MAJOR_LAYOUT)
        out = ttnn.concat((z, yb), dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)  # 0.999

        deallocate_tensors(yb, z)
        return out
