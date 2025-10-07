# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov6l.tt.common import Yolov6l_Conv2D

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


class TtDetect:
    def __init__(self, device, parameters, model_params):
        self.parameters = parameters
        self.model_params = model_params
        self.stem_0 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.stems[0].block.conv,
            conv_pth=parameters.stems[0].block.conv,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
        )
        self.stem_1 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.stems[1].block.conv,
            conv_pth=parameters.stems[1].block.conv,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
        )
        self.stem_2 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.stems[2].block.conv,
            conv_pth=parameters.stems[2].block.conv,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )

        self.cls_convs_0 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.cls_convs[0].block.conv,
            conv_pth=parameters.cls_convs[0].block.conv,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
        )
        self.cls_convs_1 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.cls_convs[1].block.conv,
            conv_pth=parameters.cls_convs[1].block.conv,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
        )
        self.cls_convs_2 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.cls_convs[2].block.conv,
            conv_pth=parameters.cls_convs[2].block.conv,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
        )

        self.reg_convs_0 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.reg_convs[0].block.conv,
            conv_pth=parameters.reg_convs[0].block.conv,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
            deallocate_activation=True,
        )
        self.reg_convs_1 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.reg_convs[1].block.conv,
            conv_pth=parameters.reg_convs[1].block.conv,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
            deallocate_activation=True,
        )
        self.reg_convs_2 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.reg_convs[2].block.conv,
            conv_pth=parameters.reg_convs[2].block.conv,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
            deallocate_activation=True,
        )

        self.cls_preds_0 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.cls_preds[0],
            conv_pth=parameters.cls_preds[0],
        )
        self.cls_preds_1 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.cls_preds[1],
            conv_pth=parameters.cls_preds[1],
        )
        self.cls_preds_2 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.cls_preds[2],
            conv_pth=parameters.cls_preds[2],
        )

        self.reg_preds_0 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.reg_preds[0],
            conv_pth=parameters.reg_preds[0],
            reshape=True,
            deallocate_activation=True,
        )
        self.reg_preds_1 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.reg_preds[1],
            conv_pth=parameters.reg_preds[1],
            reshape=True,
            deallocate_activation=True,
        )
        self.reg_preds_2 = Yolov6l_Conv2D(
            device=device,
            conv=model_params.reg_preds[2],
            conv_pth=parameters.reg_preds[2],
            reshape=True,
            deallocate_activation=True,
        )
        self.proj_conv = Yolov6l_Conv2D(
            device=device,
            conv=model_params.proj_conv,
            conv_pth=parameters.proj_conv,
            return_height_width=True,
        )

        self.anchors = parameters.anchors
        self.strides = parameters.strides
        self.ones_tensor = parameters.ones_tensor

    def __call__(self, input_list):
        if use_signpost:
            signpost(header="TtDetect Start")
        stems_0 = self.stem_0(input_list[0])
        stems_1 = self.stem_1(input_list[1])
        stems_2 = self.stem_2(input_list[2])

        cls_feat_0 = self.cls_convs_0(stems_0)
        cls_feat_1 = self.cls_convs_1(stems_1)
        cls_feat_2 = self.cls_convs_2(stems_2)

        cls_output_0 = self.cls_preds_0(cls_feat_0)
        cls_output_1 = self.cls_preds_1(cls_feat_1)
        cls_output_2 = self.cls_preds_2(cls_feat_2)

        reg_feat_0 = self.reg_convs_0(stems_0)
        reg_feat_1 = self.reg_convs_1(stems_1)
        reg_feat_2 = self.reg_convs_2(stems_2)

        reg_output_0 = self.reg_preds_0(reg_feat_0)
        reg_output_1 = self.reg_preds_1(reg_feat_1)
        reg_output_2 = self.reg_preds_2(reg_feat_2)

        reg_output_0 = ttnn.permute(reg_output_0, (0, 3, 1, 2))
        reg_output_1 = ttnn.permute(reg_output_1, (0, 3, 1, 2))
        reg_output_2 = ttnn.permute(reg_output_2, (0, 3, 1, 2))

        reg_output_0 = ttnn.reshape(
            reg_output_0, (reg_output_0.shape[0], 4, 17, reg_output_0.shape[2] * reg_output_0.shape[3])
        )
        reg_output_1 = ttnn.reshape(
            reg_output_1, (reg_output_0.shape[0], 4, 17, reg_output_1.shape[2] * reg_output_1.shape[3])
        )
        reg_output_2 = ttnn.reshape(
            reg_output_2, (reg_output_0.shape[0], 4, 17, reg_output_2.shape[2] * reg_output_2.shape[3])
        )
        reg_output_0 = ttnn.to_layout(reg_output_0, ttnn.TILE_LAYOUT)
        reg_output_1 = ttnn.to_layout(reg_output_1, ttnn.TILE_LAYOUT)
        reg_output_2 = ttnn.to_layout(reg_output_2, ttnn.TILE_LAYOUT)

        reg_output_0 = ttnn.permute(reg_output_0, (0, 1, 3, 2))
        reg_output_1 = ttnn.permute(reg_output_1, (0, 1, 3, 2))
        reg_output_2 = ttnn.permute(reg_output_2, (0, 1, 3, 2))

        reg_output_0 = ttnn.softmax(reg_output_0, -1)
        reg_output_1 = ttnn.softmax(reg_output_1, -1)
        reg_output_2 = ttnn.softmax(reg_output_2, -1)

        reg_output_0, h, w = self.proj_conv(reg_output_0)
        reg_output_0 = ttnn.sharded_to_interleaved(reg_output_0, memory_config=ttnn.L1_MEMORY_CONFIG)
        reg_output_0 = ttnn.reshape(
            reg_output_0, (reg_output_0.shape[0], h, w, reg_output_0.shape[3]), memory_config=ttnn.L1_MEMORY_CONFIG
        )
        reg_output_1, h, w = self.proj_conv(reg_output_1)
        reg_output_1 = ttnn.sharded_to_interleaved(reg_output_1, memory_config=ttnn.L1_MEMORY_CONFIG)
        reg_output_1 = ttnn.reshape(
            reg_output_1, (reg_output_1.shape[0], h, w, reg_output_1.shape[3]), memory_config=ttnn.L1_MEMORY_CONFIG
        )
        reg_output_2, h, w = self.proj_conv(reg_output_2)
        reg_output_2 = ttnn.sharded_to_interleaved(reg_output_2, memory_config=ttnn.L1_MEMORY_CONFIG)
        reg_output_2 = ttnn.reshape(
            reg_output_2, (reg_output_2.shape[0], h, w, reg_output_2.shape[3]), memory_config=ttnn.L1_MEMORY_CONFIG
        )

        cls_output_0 = ttnn.to_layout(cls_output_0, ttnn.TILE_LAYOUT)
        cls_output_1 = ttnn.to_layout(cls_output_1, ttnn.TILE_LAYOUT)
        cls_output_2 = ttnn.to_layout(cls_output_2, ttnn.TILE_LAYOUT)
        cls_output_0 = ttnn.sigmoid_accurate(cls_output_0)
        cls_output_1 = ttnn.sigmoid_accurate(cls_output_1)
        cls_output_2 = ttnn.sigmoid_accurate(cls_output_2)

        cls_output_0 = ttnn.squeeze(cls_output_0, dim=0)
        cls_output_1 = ttnn.squeeze(cls_output_1, dim=0)
        cls_output_2 = ttnn.squeeze(cls_output_2, dim=0)

        cls_output_0 = ttnn.sharded_to_interleaved(cls_output_0, memory_config=ttnn.L1_MEMORY_CONFIG)
        cls_output_1 = ttnn.sharded_to_interleaved(cls_output_1, memory_config=ttnn.L1_MEMORY_CONFIG)
        cls_output_2 = ttnn.sharded_to_interleaved(cls_output_2, memory_config=ttnn.L1_MEMORY_CONFIG)
        cls_output_0 = ttnn.permute(cls_output_0, (0, 2, 1))
        cls_output_1 = ttnn.permute(cls_output_1, (0, 2, 1))
        cls_output_2 = ttnn.permute(cls_output_2, (0, 2, 1))

        reg_output_0 = ttnn.permute(reg_output_0, (0, 3, 1, 2))
        reg_output_0 = ttnn.reshape(reg_output_0, (1, 4, reg_output_0.shape[3]))

        reg_output_1 = ttnn.permute(reg_output_1, (0, 3, 1, 2))
        reg_output_1 = ttnn.reshape(reg_output_1, (1, 4, reg_output_1.shape[3]))

        reg_output_2 = ttnn.permute(reg_output_2, (0, 3, 1, 2))
        reg_output_2 = ttnn.reshape(reg_output_2, (1, 4, reg_output_2.shape[3]))

        cls_score_list = ttnn.concat(
            [cls_output_0, cls_output_1, cls_output_2], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG
        )

        ttnn.deallocate(cls_output_0)
        ttnn.deallocate(cls_output_1)
        ttnn.deallocate(cls_output_2)

        reg_dist_list = ttnn.concat(
            [reg_output_0, reg_output_1, reg_output_2], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ttnn.deallocate(reg_output_0)
        ttnn.deallocate(reg_output_1)
        ttnn.deallocate(reg_output_2)

        c1, c2 = reg_dist_list[:, :2, :], reg_dist_list[:, 2:4, :]

        x1y1 = ttnn.sub(self.anchors, c1, memory_config=ttnn.L1_MEMORY_CONFIG)
        x2y2 = ttnn.add(self.anchors, c2, memory_config=ttnn.L1_MEMORY_CONFIG)

        c_xy = x1y1 + x2y2
        c_xy = ttnn.div(c_xy, 2)
        wh = x2y2 - x1y1
        bbox = ttnn.concat([c_xy, wh], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(c_xy)
        ttnn.deallocate(wh)

        bbox = ttnn.multiply(bbox, self.strides)

        output = ttnn.concat([bbox, self.ones_tensor, cls_score_list], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        output = ttnn.permute(output, (0, 2, 1))

        if use_signpost:
            signpost(header="TtDetect End")

        return output
