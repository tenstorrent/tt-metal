# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import math
import numpy as np
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
    def __init__(self, device) -> None:
        self.thresh = 0.6
        self.num_classes = 80
        self.num_anchors = 3

        self.grid_x = []
        self.grid_y = []
        for H in (40, 20, 10):
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
            )

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
            )
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

        if HW == 1600:
            group = 0
        elif HW == 400:
            group = 1
        elif HW == 100:
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

        det_confs_a = ttnn.to_layout(det_confs_a, ttnn.TILE_LAYOUT)
        det_confs_b = ttnn.to_layout(det_confs_b, ttnn.TILE_LAYOUT)
        det_confs_c = ttnn.to_layout(det_confs_c, ttnn.TILE_LAYOUT)
        cls_confs_a = ttnn.to_layout(cls_confs_a, ttnn.TILE_LAYOUT)
        cls_confs_b = ttnn.to_layout(cls_confs_b, ttnn.TILE_LAYOUT)
        cls_confs_c = ttnn.to_layout(cls_confs_c, ttnn.TILE_LAYOUT)

        det_confs_a = ttnn.sigmoid(det_confs_a)
        det_confs_b = ttnn.sigmoid(det_confs_b)
        det_confs_c = ttnn.sigmoid(det_confs_c)
        cls_confs_a = ttnn.sigmoid(cls_confs_a)
        cls_confs_b = ttnn.sigmoid(cls_confs_b)
        cls_confs_c = ttnn.sigmoid(cls_confs_c)

        confs_a = ttnn.multiply(det_confs_a, cls_confs_a)
        confs_b = ttnn.multiply(det_confs_b, cls_confs_b)
        confs_c = ttnn.multiply(det_confs_c, cls_confs_c)

        confs = ttnn.concat([confs_a, confs_b, confs_c], dim=1)
        confs = ttnn.permute(confs, (0, 1, 3, 2))
        confs = ttnn.reshape(confs, (B, AHW, self.num_classes))

        #################
        ## Boxes
        #################

        # expensive TilizeWithValPadding
        bx_a = ttnn.to_layout(bx_a, ttnn.TILE_LAYOUT)
        by_a = ttnn.to_layout(by_a, ttnn.TILE_LAYOUT)
        bw_a = ttnn.to_layout(bw_a, ttnn.TILE_LAYOUT)
        bh_a = ttnn.to_layout(bh_a, ttnn.TILE_LAYOUT)
        bx_a = ttnn.sigmoid(bx_a)
        by_a = ttnn.sigmoid(by_a)
        bw_a = ttnn.exp(bw_a)
        bh_a = ttnn.exp(bh_a)

        bx_b = ttnn.to_layout(bx_b, ttnn.TILE_LAYOUT)
        by_b = ttnn.to_layout(by_b, ttnn.TILE_LAYOUT)
        bw_b = ttnn.to_layout(bw_b, ttnn.TILE_LAYOUT)
        bh_b = ttnn.to_layout(bh_b, ttnn.TILE_LAYOUT)
        bx_b = ttnn.sigmoid(bx_b)
        by_b = ttnn.sigmoid(by_b)
        bw_b = ttnn.exp(bw_b)
        bh_b = ttnn.exp(bh_b)

        bx_c = ttnn.to_layout(bx_c, ttnn.TILE_LAYOUT)
        by_c = ttnn.to_layout(by_c, ttnn.TILE_LAYOUT)
        bw_c = ttnn.to_layout(bw_c, ttnn.TILE_LAYOUT)
        bh_c = ttnn.to_layout(bh_c, ttnn.TILE_LAYOUT)
        bx_c = ttnn.sigmoid(bx_c)
        by_c = ttnn.sigmoid(by_c)
        bw_c = ttnn.exp(bw_c)
        bh_c = ttnn.exp(bh_c)

        ####
        ## Grid tensor derivation
        ####

        grid_x = self.grid_x[group]  # .to(device, mem_config=ttnn.L1_MEMORY_CONFIG)
        grid_y = self.grid_y[group]  # .to(device, mem_config=ttnn.L1_MEMORY_CONFIG)

        bx_a = ttnn.add(bx_a, grid_x)
        by_a = ttnn.add(by_a, grid_y)
        bx_b = ttnn.add(bx_b, grid_x)
        by_b = ttnn.add(by_b, grid_y)
        bx_c = ttnn.add(bx_c, grid_x)
        by_c = ttnn.add(by_c, grid_y)

        bx_a = ttnn.multiply(bx_a, 1 / W)
        by_a = ttnn.multiply(by_a, 1 / H)
        bx_b = ttnn.multiply(bx_b, 1 / W)
        by_b = ttnn.multiply(by_b, 1 / H)
        bx_c = ttnn.multiply(bx_c, 1 / W)
        by_c = ttnn.multiply(by_c, 1 / H)

        bw_a = bw_a * (anchor_w_a / W)
        bw_b = bw_b * (anchor_w_b / W)
        bw_c = bw_c * (anchor_w_c / W)

        bh_a = bh_a * (anchor_h_a / H)
        bh_b = bh_b * (anchor_h_b / H)
        bh_c = bh_c * (anchor_h_c / H)

        bw_a_half = bw_a * (0.5)
        bw_b_half = bw_b * (0.5)
        bw_c_half = bw_c * (0.5)

        bh_a_half = bh_a * (0.5)
        bh_b_half = bh_b * (0.5)
        bh_c_half = bh_c * (0.5)

        bx1_a = bx_a - bw_a_half
        by1_a = by_a - bh_a_half
        bx2_a = bx1_a + bw_a
        by2_a = by1_a + bh_a

        bx1_b = bx_b - bw_b_half
        by1_b = by_b - bh_b_half
        bx2_b = bx1_b + bw_b
        by2_b = by1_b + bh_b

        bx1_c = bx_c - bw_c_half
        by1_c = by_c - bh_c_half
        bx2_c = bx1_c + bw_c
        by2_c = by1_c + bh_c

        bx1_a = ttnn.to_layout(bx1_a, ttnn.ROW_MAJOR_LAYOUT)
        bx2_a = ttnn.to_layout(bx2_a, ttnn.ROW_MAJOR_LAYOUT)
        by1_a = ttnn.to_layout(by1_a, ttnn.ROW_MAJOR_LAYOUT)
        by2_a = ttnn.to_layout(by2_a, ttnn.ROW_MAJOR_LAYOUT)

        bx1_b = ttnn.to_layout(bx1_b, ttnn.ROW_MAJOR_LAYOUT)
        bx2_b = ttnn.to_layout(bx2_b, ttnn.ROW_MAJOR_LAYOUT)
        by1_b = ttnn.to_layout(by1_b, ttnn.ROW_MAJOR_LAYOUT)
        by2_b = ttnn.to_layout(by2_b, ttnn.ROW_MAJOR_LAYOUT)

        bx1_c = ttnn.to_layout(bx1_c, ttnn.ROW_MAJOR_LAYOUT)
        bx2_c = ttnn.to_layout(bx2_c, ttnn.ROW_MAJOR_LAYOUT)
        by1_c = ttnn.to_layout(by1_c, ttnn.ROW_MAJOR_LAYOUT)
        by2_c = ttnn.to_layout(by2_c, ttnn.ROW_MAJOR_LAYOUT)

        bx1 = ttnn.concat([bx1_a, bx1_b, bx1_c], dim=2)
        by1 = ttnn.concat([by1_a, by1_b, by1_c], dim=2)
        bx2 = ttnn.concat([bx2_a, bx2_b, bx2_c], dim=2)
        by2 = ttnn.concat([by2_a, by2_b, by2_c], dim=2)

        # Shape: [batch, num_anchors * h * w, 4] -> [batch, num_anchors * h * w, 1, 4]
        boxes = ttnn.concat((bx1, by1, bx2, by2), dim=1)

        return boxes, confs
