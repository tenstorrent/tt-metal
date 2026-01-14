# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for YUNet model with TTNN backend."""

import pytest
import torch
from torch import nn

from models.experimental.tt_symbiote.core.run_config import DispatchManager
from models.experimental.tt_symbiote.modules.activation import TTNNReLU
from models.experimental.tt_symbiote.modules.conv import TTNNConv2dNHWC, TTNNMaxPool2dNHWC, TTNNUpsampleNHWC
from models.experimental.tt_symbiote.modules.linear import TTNNLinear
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict

try:
    from models.experimental.tt_symbiote.YUNet.nets import nn as YUNet_nn
except:
    print(
        "YUNet import failed. Make sure you have the YUnet https://github.com/jahongir7174/YUNet/tree/master in models.experimental.tt_symbiote.YUNet"
    )
    exit(1)


class RewrittenHead(torch.nn.Module):
    def __init__(self, original_layer):
        super().__init__()
        self.nc = original_layer.nc  # number of classes
        self.nk = original_layer.nk  # number of keypoints
        self.nl = original_layer.nl  # number of detection layers
        self.strides = original_layer.strides

        self.m = original_layer.m
        self.cls = original_layer.cls
        self.box = original_layer.box
        self.obj = original_layer.obj
        self.kpt = original_layer.kpt

    @classmethod
    def from_torch(cls, head: YUNet_nn.Head):
        """Create RewrittenHead from PyTorch YUNet Head layer."""
        new_head = RewrittenHead(head)
        return new_head

    def forward(self, x):
        x = [m(i) for i, m in zip(x, self.m)]

        cls = [m(i) for i, m in zip(x, self.cls)]
        box = [m(i) for i, m in zip(x, self.box)]
        obj = [m(i) for i, m in zip(x, self.obj)]
        kpt = [m(i) for i, m in zip(x, self.kpt)]

        # if self.training:
        return cls, box, obj, kpt

        n = cls[0].shape[0]
        sizes = [i.shape[1:3] for i in cls]
        anchors = self.__make_anchors(sizes, self.strides, cls[0].device, cls[0].dtype)

        cls = [i.reshape(n, -1, self.nc) for i in cls]
        box = [i.reshape(n, -1, 4) for i in box]
        obj = [i.reshape(n, -1) for i in obj]
        kpt = [i.reshape(n, -1, self.nk * 2) for i in kpt]
        cls = torch.cat(cls, dim=1).sigmoid()
        box = torch.cat(box, dim=1)
        obj = torch.cat(obj, dim=1).sigmoid()
        kpt = torch.cat(kpt, dim=1)

        box = self.__box_decode(torch.cat(anchors), box)
        kpt = self.__kpt_decode(torch.cat(anchors), kpt)
        return cls, box, obj, kpt

    @staticmethod
    def __box_decode(anchors, box):
        xys = (box[..., :2] * anchors[..., 2:]) + anchors[..., :2]
        whs = box[..., 2:].exp() * anchors[..., 2:]

        tl_x = xys[..., 0] - whs[..., 0] / 2
        tl_y = xys[..., 1] - whs[..., 1] / 2
        br_x = xys[..., 0] + whs[..., 0] / 2
        br_y = xys[..., 1] + whs[..., 1] / 2

        return torch.stack(tensors=[tl_x, tl_y, br_x, br_y], dim=-1)

    @staticmethod
    def __kpt_decode(anchors, kpt):
        num_kpt = int(kpt.shape[-1] / 2)
        decoded_kpt = [(kpt[..., [2 * i, 2 * i + 1]] * anchors[..., 2:]) + anchors[..., :2] for i in range(num_kpt)]

        return torch.cat(decoded_kpt, dim=-1)

    @staticmethod
    def __make_anchors(sizes, strides, device, dtype, offset=0.0):
        anchors = []
        assert len(sizes) == len(strides)
        for stride, size in zip(strides, sizes):
            # keep size as Tensor instead of int, so that we can convert to ONNX correctly
            shift_x = ((torch.arange(0, size[1]) + offset) * stride).to(dtype)
            shift_y = ((torch.arange(0, size[0]) + offset) * stride).to(dtype)

            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            shift_x, shift_y = shift_x.reshape(-1), shift_y.reshape(-1)
            stride_w = shift_x.new_full((shift_x.shape[0],), stride).to(dtype)
            stride_h = shift_x.new_full((shift_y.shape[0],), stride).to(dtype)
            anchors.append(torch.stack(tensors=[shift_x, shift_y, stride_w, stride_h], dim=-1).to(device))
        return anchors


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_resnet(device):
    """Test Resnet model with TTNN acceleration."""

    model = YUNet_nn.version_n().fuse().to(torch.bfloat16)

    nn_to_nn = {
        YUNet_nn.Head: RewrittenHead,
    }
    model_config = {
        "backbone.p1.0.conv": {"input_shapes": [[1, 224, 224, 3]], "reshape_output": False},
        "backbone.p1.1.conv1": {"input_shapes": [[1, 112, 112, 16]], "reshape_output": False},
        "backbone.p1.1.conv2.conv": {"input_shapes": [[1, 112, 112, 16]], "reshape_output": False},
        "backbone.p2.0": {"input_shapes": [[1, 112, 112, 16]], "reshape_output": False},
        "backbone.p2.1.conv1": {"input_shapes": [[1, 56, 56, 16]], "reshape_output": False},
        "backbone.p2.1.conv2.conv": {"input_shapes": [[1, 56, 56, 64]], "reshape_output": False},
        "backbone.p2.2.conv1": {"input_shapes": [[1, 56, 56, 64]], "reshape_output": False},
        "backbone.p2.2.conv2.conv": {"input_shapes": [[1, 56, 56, 64]], "reshape_output": True},
        "backbone.p2.3.conv1": {"input_shapes": [[1, 56, 56, 64]], "reshape_output": False},
        "backbone.p2.3.conv2.conv": {"input_shapes": [[1, 56, 56, 64]], "reshape_output": True},
        "backbone.p3.0": {"input_shapes": [[1, 56, 56, 64]], "reshape_output": False},
        "backbone.p3.1.conv1": {"input_shapes": [[1, 28, 28, 64]], "reshape_output": False},
        "backbone.p3.1.conv2.conv": {"input_shapes": [[1, 28, 28, 64]], "reshape_output": False},
        "backbone.p3.2.conv1": {"input_shapes": [[1, 28, 28, 64]], "reshape_output": False},
        "backbone.p3.2.conv2.conv": {"input_shapes": [[1, 28, 28, 64]], "reshape_output": True},
        "backbone.p4.0": {"input_shapes": [[1, 28, 28, 64]], "reshape_output": False},
        "backbone.p4.1.conv1": {"input_shapes": [[1, 14, 14, 64]], "reshape_output": False},
        "backbone.p4.1.conv2.conv": {"input_shapes": [[1, 14, 14, 64]], "reshape_output": False},
        "backbone.p4.2.conv1": {"input_shapes": [[1, 14, 14, 64]], "reshape_output": False},
        "backbone.p4.2.conv2.conv": {"input_shapes": [[1, 14, 14, 64]], "reshape_output": True},
        "backbone.p5.0": {"input_shapes": [[1, 14, 14, 64]], "reshape_output": False},
        "backbone.p5.1.conv1": {"input_shapes": [[1, 7, 7, 64]], "reshape_output": False},
        "backbone.p5.1.conv2.conv": {"input_shapes": [[1, 7, 7, 64]], "reshape_output": False},
        "backbone.p5.2.conv1": {"input_shapes": [[1, 7, 7, 64]], "reshape_output": False},
        "backbone.p5.2.conv2.conv": {"input_shapes": [[1, 7, 7, 64]], "reshape_output": True},
        "neck.conv1.conv1": {"input_shapes": [[1, 7, 7, 64]], "reshape_output": True},
        "neck.conv1.conv2.conv": {"input_shapes": [[1, 7, 7, 64]], "reshape_output": True},
        "neck.conv2.conv1": {"input_shapes": [[1, 14, 14, 64]], "reshape_output": True},
        "neck.conv2.conv2.conv": {"input_shapes": [[1, 14, 14, 64]], "reshape_output": True},
        "neck.conv3.conv1": {"input_shapes": [[1, 28, 28, 64]], "reshape_output": True},
        "neck.conv3.conv2.conv": {"input_shapes": [[1, 28, 28, 64]], "reshape_output": True},
        "head.m.0.conv1": {"input_shapes": [[1, 28, 28, 64]], "reshape_output": True},
        "head.m.0.conv2.conv": {"input_shapes": [[1, 28, 28, 64]], "reshape_output": True},
        "head.m.1.conv1": {"input_shapes": [[1, 14, 14, 64]], "reshape_output": True},
        "head.m.1.conv2.conv": {"input_shapes": [[1, 14, 14, 64]], "reshape_output": True},
        "head.m.2.conv1": {"input_shapes": [[1, 7, 7, 64]], "reshape_output": True},
        "head.m.2.conv2.conv": {"input_shapes": [[1, 7, 7, 64]], "reshape_output": True},
        "head.cls.0": {"input_shapes": [[1, 28, 28, 64]], "reshape_output": True},
        "head.cls.1": {"input_shapes": [[1, 14, 14, 64]], "reshape_output": True},
        "head.cls.2": {"input_shapes": [[1, 7, 7, 64]], "reshape_output": True},
        "head.box.0": {"input_shapes": [[1, 28, 28, 64]], "reshape_output": True},
        "head.box.1": {"input_shapes": [[1, 14, 14, 64]], "reshape_output": True},
        "head.box.2": {"input_shapes": [[1, 7, 7, 64]], "reshape_output": True},
        "head.obj.0": {"input_shapes": [[1, 28, 28, 64]], "reshape_output": True},
        "head.obj.1": {"input_shapes": [[1, 14, 14, 64]], "reshape_output": True},
        "head.obj.2": {"input_shapes": [[1, 7, 7, 64]], "reshape_output": True},
        "head.kpt.0": {"input_shapes": [[1, 28, 28, 64]], "reshape_output": True},
        "head.kpt.1": {"input_shapes": [[1, 14, 14, 64]], "reshape_output": True},
        "head.kpt.2": {"input_shapes": [[1, 7, 7, 64]], "reshape_output": True},
    }
    modules = register_module_replacement_dict(model, nn_to_nn, model_config=model_config)

    nn_to_ttnn = {
        nn.Linear: TTNNLinear,
        nn.Conv2d: TTNNConv2dNHWC,
        nn.ReLU: TTNNReLU,
        nn.MaxPool2d: TTNNMaxPool2dNHWC,
        nn.Upsample: TTNNUpsampleNHWC,
    }
    modules = register_module_replacement_dict(model, nn_to_ttnn, model_config=model_config)
    set_device(model, device)
    for k, v in modules.items():
        v.preprocess_weights()
        v.move_weights_to_device()
    model.eval()  # Disables dropout, batch norm updates
    torch.set_grad_enabled(False)  # Disables autograd overhead
    result = model(torch.randn(1, 224, 224, 3, dtype=torch.bfloat16))
    DispatchManager.clear_timings()
    for i in range(60):
        _ = model(torch.randn(1, 224, 224, 3, dtype=torch.bfloat16))
    DispatchManager.save_stats_to_file("yunet_timing_stats.csv")
    print(result)
