# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pkg_resources as pkg


def dist2bbox(distance, anchor_points, box_format="xyxy"):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = torch.split(distance, 2, -1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if box_format == "xyxy":
        bbox = torch.cat([x1y1, x2y2], -1)
    elif box_format == "xywh":
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        bbox = torch.cat([c_xy, wh], -1)
    return bbox


def check_version(current="0.0.0", minimum="0.0.0", name="version ", pinned=False, hard=False, verbose=False):
    # Check whether the package's version is match the required version.
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    if hard:
        info = f"⚠️ {name}{minimum} is required by YOLOv6, but {name}{current} is currently installed"
        assert result, info  # assert minimum version requirement
    return result


torch_1_10_plus = check_version(torch.__version__, minimum="1.10.0")


def generate_anchors(
    feats, fpn_strides, grid_cell_size=5.0, grid_cell_offset=0.5, device="cpu", is_eval=False, mode="af"
):
    """Generate anchors from features."""
    anchors = []
    anchor_points = []
    stride_tensor = []
    num_anchors_list = []
    assert feats is not None
    if is_eval:
        for i, stride in enumerate(fpn_strides):
            _, _, h, w = feats[i].shape
            shift_x = torch.arange(end=w, device=device) + grid_cell_offset
            shift_y = torch.arange(end=h, device=device) + grid_cell_offset
            shift_y, shift_x = (
                torch.meshgrid(shift_y, shift_x, indexing="ij") if torch_1_10_plus else torch.meshgrid(shift_y, shift_x)
            )
            anchor_point = torch.stack([shift_x, shift_y], axis=-1).to(torch.float)
            if mode == "af":  # anchor-free
                anchor_points.append(anchor_point.reshape([-1, 2]))
                stride_tensor.append(torch.full((h * w, 1), stride, dtype=torch.float, device=device))
            elif mode == "ab":  # anchor-based
                anchor_points.append(anchor_point.reshape([-1, 2]).repeat(3, 1))
                stride_tensor.append(torch.full((h * w, 1), stride, dtype=torch.float, device=device).repeat(3, 1))
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        return anchor_points, stride_tensor
    else:
        for i, stride in enumerate(fpn_strides):
            _, _, h, w = feats[i].shape
            cell_half_size = grid_cell_size * stride * 0.5
            shift_x = (torch.arange(end=w, device=device) + grid_cell_offset) * stride
            shift_y = (torch.arange(end=h, device=device) + grid_cell_offset) * stride
            shift_y, shift_x = (
                torch.meshgrid(shift_y, shift_x, indexing="ij") if torch_1_10_plus else torch.meshgrid(shift_y, shift_x)
            )
            anchor = (
                torch.stack(
                    [
                        shift_x - cell_half_size,
                        shift_y - cell_half_size,
                        shift_x + cell_half_size,
                        shift_y + cell_half_size,
                    ],
                    axis=-1,
                )
                .clone()
                .to(feats[0].dtype)
            )
            anchor_point = torch.stack([shift_x, shift_y], axis=-1).clone().to(feats[0].dtype)

            if mode == "af":  # anchor-free
                anchors.append(anchor.reshape([-1, 4]))
                anchor_points.append(anchor_point.reshape([-1, 2]))
            elif mode == "ab":  # anchor-based
                anchors.append(anchor.reshape([-1, 4]).repeat(3, 1))
                anchor_points.append(anchor_point.reshape([-1, 2]).repeat(3, 1))
            num_anchors_list.append(len(anchors[-1]))
            stride_tensor.append(torch.full([num_anchors_list[-1], 1], stride, dtype=feats[0].dtype))
        anchors = torch.cat(anchors)
        anchor_points = torch.cat(anchor_points).to(device)
        stride_tensor = torch.cat(stride_tensor).to(device)
        return anchors, anchor_points, num_anchors_list, stride_tensor


class Detect(nn.Module):
    export = False
    """Efficient Decoupled Head
    With hardware-aware degisn, the decoupled head is optimized with
    hybridchannels methods.
    """

    def __init__(
        self, num_classes=80, num_layers=3, inplace=True, head_layers=None, use_dfl=True, reg_max=16
    ):  # detection layer
        super().__init__()
        assert head_layers is not None
        self.nc = num_classes  # number of classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.nl = num_layers  # number of detection layers
        self.grid = [torch.zeros(1)] * num_layers
        self.prior_prob = 1e-2
        self.inplace = inplace
        stride = [8, 16, 32] if num_layers == 3 else [8, 16, 32, 64]  # strides computed during build
        self.stride = torch.tensor(stride)
        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0

        # Init decouple head
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        # Efficient decoupled head layers
        for i in range(num_layers):
            idx = i * 5
            self.stems.append(head_layers[idx])
            self.cls_convs.append(head_layers[idx + 1])
            self.reg_convs.append(head_layers[idx + 2])
            self.cls_preds.append(head_layers[idx + 3])
            self.reg_preds.append(head_layers[idx + 4])

    def initialize_biases(self):
        for conv in self.cls_preds:
            b = conv.bias.view(
                -1,
            )
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.0)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        for conv in self.reg_preds:
            b = conv.bias.view(
                -1,
            )
            b.data.fill_(1.0)
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.0)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.proj_conv.weight = nn.Parameter(
            self.proj.view([1, self.reg_max + 1, 1, 1]).clone().detach(), requires_grad=False
        )

    def forward(self, x):
        if self.training:
            cls_score_list = []
            reg_distri_list = []

            for i in range(self.nl):
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)

                cls_output = torch.sigmoid(cls_output)
                cls_score_list.append(cls_output.flatten(2).permute((0, 2, 1)))
                reg_distri_list.append(reg_output.flatten(2).permute((0, 2, 1)))

            cls_score_list = torch.cat(cls_score_list, axis=1)
            reg_distri_list = torch.cat(reg_distri_list, axis=1)

            return x, cls_score_list, reg_distri_list
        else:
            cls_score_list = []
            reg_dist_list = []

            for i in range(self.nl):
                b, _, h, w = x[i].shape
                l = h * w
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)

                if self.use_dfl:
                    reg_output = reg_output.reshape([-1, 4, self.reg_max + 1, l]).permute(0, 2, 1, 3)
                    reg_output = self.proj_conv(F.softmax(reg_output, dim=1))

                cls_output = torch.sigmoid(cls_output)

                if self.export:
                    cls_score_list.append(cls_output)
                    reg_dist_list.append(reg_output)
                else:
                    cls_score_list.append(cls_output.reshape([b, self.nc, l]))
                    reg_dist_list.append(reg_output.reshape([b, 4, l]))

            if self.export:
                return tuple(torch.cat([cls, reg], 1) for cls, reg in zip(cls_score_list, reg_dist_list))

            cls_score_list = torch.cat(cls_score_list, axis=-1).permute(0, 2, 1)
            reg_dist_list = torch.cat(reg_dist_list, axis=-1).permute(0, 2, 1)

            anchor_points, stride_tensor = generate_anchors(
                x, self.stride, self.grid_cell_size, self.grid_cell_offset, device=x[0].device, is_eval=True, mode="af"
            )

            pred_bboxes = dist2bbox(reg_dist_list, anchor_points, box_format="xywh")
            pred_bboxes *= stride_tensor
            return torch.cat(
                [
                    pred_bboxes,
                    torch.ones((b, pred_bboxes.shape[1], 1), device=pred_bboxes.device, dtype=pred_bboxes.dtype),
                    cls_score_list,
                ],
                axis=-1,
            )
