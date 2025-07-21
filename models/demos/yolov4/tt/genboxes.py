# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import torch

import ttnn
from models.utility_functions import _nearest_32


def create_conv_bias_tensor(torch_tensor, N, K, pad=0):
    bias_shape = [1, 1, N, K]
    bias_padded_shape = [1, 1, _nearest_32(N), _nearest_32(K)]
    tt_tensor = ttnn.Tensor(torch.flatten(torch_tensor).tolist(), bias_shape, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT).pad(
        bias_shape, (0, 0, 0, 0), 0.0
    )
    tt_tensor = tt_tensor.pad_to_tile(pad).to(ttnn.TILE_LAYOUT)
    return tt_tensor


class TtGenBoxes:
    def __init__(self, device, resolution) -> None:
        self.thresh = 0.6
        self.num_classes = 80
        self.num_anchors = 3
        self.resolution = resolution

        if resolution == (320, 320):
            h1, h2, h3 = 40, 20, 10
        elif resolution == (640, 640):
            h1, h2, h3 = 80, 40, 20
        else:
            raise ValueError(f"Unsupported resolution: {resolution}")

        self.grid_x = []
        self.grid_y = []
        for H in (h1, h2, h3):
            grid_x_i = torch.reshape(
                torch.flatten(
                    torch.from_numpy(
                        np.expand_dims(
                            np.expand_dims(np.expand_dims(np.linspace(0, H - 1, H), axis=0).repeat(H, 0), axis=0),
                            axis=0,
                        )
                    )
                ),
                (1, 1, 1, H * H),
            ).repeat(1, 3, 1, 1)

            grid_y_i = torch.reshape(
                torch.flatten(
                    torch.from_numpy(
                        np.expand_dims(
                            np.expand_dims(np.expand_dims(np.linspace(0, H - 1, H), axis=1).repeat(H, 1), axis=0),
                            axis=0,
                        )
                    )
                ),
                (1, 1, 1, H * H),
            ).repeat(1, 3, 1, 1)

            self.grid_x.append(
                ttnn.from_torch(grid_x_i, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            )  # , 1, H*H))
            self.grid_y.append(
                ttnn.from_torch(grid_y_i, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            )  # , 1, H*H))

    def __call__(self, device, input_tensor):
        B, __, HW, dim = input_tensor.shape
        H = W = int(math.sqrt(HW))
        AHW = self.num_anchors * HW
        A = self.num_anchors

        if self.resolution[0] == 320:
            if HW == 1600:
                group = 0
            elif HW == 400:
                group = 1
            elif HW == 100:
                group = 2
        else:
            if HW == 6400:
                group = 0
            elif HW == 1600:
                group = 1
            elif HW == 400:
                group = 2

        # Pre-derived from the torch function
        if group == 0:
            anchor_w_a = 1.5
            anchor_w_b = 2.375
            anchor_w_c = 5.0
            anchor_h_a = 2.0
            anchor_h_b = 4.5
            anchor_h_c = 3.5
        elif group == 1:
            anchor_w_a = 2.25
            anchor_w_b = 4.75
            anchor_w_c = 4.5
            anchor_h_a = 4.6875
            anchor_h_b = 3.4375
            anchor_h_c = 9.125
        elif group == 2:
            anchor_w_a = 4.4375
            anchor_w_b = 6.0
            anchor_w_c = 14.34375
            anchor_h_a = 3.4375
            anchor_h_b = 7.59375
            anchor_h_c = 12.53125

        input_tensor_i = ttnn.to_memory_config(input_tensor, ttnn.L1_MEMORY_CONFIG)
        input_tensor_i = ttnn.to_layout(input_tensor_i, ttnn.ROW_MAJOR_LAYOUT)
        input_tensor_i = ttnn.permute(input_tensor_i, (0, 1, 3, 2))

        # first anchor
        bx_a = ttnn.slice(input_tensor_i, [0, 0, 0, 0], [1, 1, 1, HW])
        by_a = ttnn.slice(input_tensor_i, [0, 0, 1, 0], [1, 1, 2, HW])
        bw_a = ttnn.slice(input_tensor_i, [0, 0, 2, 0], [1, 1, 3, HW])
        bh_a = ttnn.slice(input_tensor_i, [0, 0, 3, 0], [1, 1, 4, HW])
        det_confs_a = ttnn.slice(input_tensor_i, [0, 0, 4, 0], [1, 1, 5, HW])
        cls_confs_a = ttnn.slice(input_tensor_i, [0, 0, 5, 0], [1, 1, 85, HW])
        # second anchor
        bx_b = ttnn.slice(input_tensor_i, [0, 0, 85, 0], [1, 1, 86, HW])
        by_b = ttnn.slice(input_tensor_i, [0, 0, 86, 0], [1, 1, 87, HW])
        bw_b = ttnn.slice(input_tensor_i, [0, 0, 87, 0], [1, 1, 88, HW])
        bh_b = ttnn.slice(input_tensor_i, [0, 0, 88, 0], [1, 1, 89, HW])
        det_confs_b = ttnn.slice(input_tensor_i, [0, 0, 89, 0], [1, 1, 90, HW])
        cls_confs_b = ttnn.slice(input_tensor_i, [0, 0, 90, 0], [1, 1, 170, HW])
        # third anchor
        bx_c = ttnn.slice(input_tensor_i, [0, 0, 170, 0], [1, 1, 171, HW])
        by_c = ttnn.slice(input_tensor_i, [0, 0, 171, 0], [1, 1, 172, HW])
        bw_c = ttnn.slice(input_tensor_i, [0, 0, 172, 0], [1, 1, 173, HW])
        bh_c = ttnn.slice(input_tensor_i, [0, 0, 173, 0], [1, 1, 174, HW])
        det_confs_c = ttnn.slice(input_tensor_i, [0, 0, 174, 0], [1, 1, 175, HW])
        cls_confs_c = ttnn.slice(input_tensor_i, [0, 0, 175, 0], [1, 1, 255, HW])

        #############
        # Confs
        #############

        det_confs = ttnn.concat([det_confs_a, det_confs_b, det_confs_c], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        cls_confs = ttnn.concat([cls_confs_a, cls_confs_b, cls_confs_c], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)

        det_confs = ttnn.to_layout(det_confs, ttnn.TILE_LAYOUT)
        cls_confs = ttnn.to_layout(cls_confs, ttnn.TILE_LAYOUT)

        det_confs = ttnn.sigmoid(det_confs, memory_config=ttnn.L1_MEMORY_CONFIG)
        cls_confs = ttnn.sigmoid(cls_confs, memory_config=ttnn.L1_MEMORY_CONFIG)

        confs = ttnn.multiply(det_confs, cls_confs, memory_config=ttnn.L1_MEMORY_CONFIG)
        confs = ttnn.permute(confs, (0, 1, 3, 2))
        confs = ttnn.reshape(confs, (B, AHW, self.num_classes))

        #################
        ## Boxes
        #################

        bx = ttnn.concat([bx_a, bx_b, bx_c], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        by = ttnn.concat([by_a, by_b, by_c], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        bw = ttnn.concat([bw_a, bw_b, bw_c], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        bh = ttnn.concat([bh_a, bh_b, bh_c], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)

        # expensive TilizeWithValPadding
        bx = ttnn.to_layout(bx, ttnn.TILE_LAYOUT)
        by = ttnn.to_layout(by, ttnn.TILE_LAYOUT)
        bw = ttnn.to_layout(bw, ttnn.TILE_LAYOUT)
        bh = ttnn.to_layout(bh, ttnn.TILE_LAYOUT)

        bx = ttnn.sigmoid(bx, memory_config=ttnn.L1_MEMORY_CONFIG)
        by = ttnn.sigmoid(by, memory_config=ttnn.L1_MEMORY_CONFIG)
        bw = ttnn.exp(bw, memory_config=ttnn.L1_MEMORY_CONFIG)
        bh = ttnn.exp(bh, memory_config=ttnn.L1_MEMORY_CONFIG)

        ####
        ## Grid tensor derivation
        ####

        grid_x = self.grid_x[group]  # .to(device, mem_config=ttnn.L1_MEMORY_CONFIG)
        grid_y = self.grid_y[group]  # .to(device, mem_config=ttnn.L1_MEMORY_CONFIG)

        bx = ttnn.add(bx, grid_x, memory_config=ttnn.L1_MEMORY_CONFIG)
        by = ttnn.add(by, grid_y, memory_config=ttnn.L1_MEMORY_CONFIG)

        bx = ttnn.multiply(bx, 1 / W, memory_config=ttnn.L1_MEMORY_CONFIG)
        by = ttnn.multiply(by, 1 / H, memory_config=ttnn.L1_MEMORY_CONFIG)

        ######

        bw_a = ttnn.slice(bw, [0, 0, 0, 0], [1, 1, 1, HW])
        bw_b = ttnn.slice(bw, [0, 1, 0, 0], [1, 2, 1, HW])
        bw_c = ttnn.slice(bw, [0, 2, 0, 0], [1, 3, 1, HW])

        bh_a = ttnn.slice(bh, [0, 0, 0, 0], [1, 1, 1, HW])
        bh_b = ttnn.slice(bh, [0, 1, 0, 0], [1, 2, 1, HW])
        bh_c = ttnn.slice(bh, [0, 2, 0, 0], [1, 3, 1, HW])

        bw_a = bw_a * (anchor_w_a / W)
        bw_b = bw_b * (anchor_w_b / W)
        bw_c = bw_c * (anchor_w_c / W)

        bh_a = bh_a * (anchor_h_a / H)
        bh_b = bh_b * (anchor_h_b / H)
        bh_c = bh_c * (anchor_h_c / H)

        bw = ttnn.concat([bw_a, bw_b, bw_c], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        bh = ttnn.concat([bh_a, bh_b, bh_c], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)

        bw_half = bw * (0.5)
        bh_half = bh * (0.5)

        ####

        bx1 = bx - bw_half
        by1 = by - bh_half
        bx2 = bx1 + bw
        by2 = by1 + bh

        bx1 = ttnn.to_layout(bx1, ttnn.ROW_MAJOR_LAYOUT)
        bx2 = ttnn.to_layout(bx2, ttnn.ROW_MAJOR_LAYOUT)
        by1 = ttnn.to_layout(by1, ttnn.ROW_MAJOR_LAYOUT)
        by2 = ttnn.to_layout(by2, ttnn.ROW_MAJOR_LAYOUT)

        # Shape: [batch, num_anchors * h * w, 4] -> [batch, num_anchors * h * w, 1, 4]
        boxes = ttnn.concat((bx1, by1, bx2, by2), dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
        boxes = ttnn.permute(boxes, (0, 2, 1, 3))

        return boxes, confs
