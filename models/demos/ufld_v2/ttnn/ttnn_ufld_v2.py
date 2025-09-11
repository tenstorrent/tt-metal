# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.ufld_v2.ttnn.common import TtnnUFLDV2Conv2D
from models.demos.ufld_v2.ttnn.ttnn_resnet_34 import TtnnResnet34


class TtnnUFLDv2:
    def __init__(self, conv_args, conv_pth, device):
        self.input_height = 320
        self.input_width = 800
        self.num_grid_row = 100
        self.num_cls_row = 56
        self.num_grid_col = 100
        self.num_cls_col = 41
        self.num_lane_on_row = 4
        self.num_lane_on_col = 4
        self.use_aux = False
        self.dim1 = self.num_grid_row * self.num_cls_row * self.num_lane_on_row
        self.dim2 = self.num_grid_col * self.num_cls_col * self.num_lane_on_col
        self.dim3 = 2 * self.num_cls_row * self.num_lane_on_row
        self.dim4 = 2 * self.num_cls_col * self.num_lane_on_col
        self.total_dim = self.dim1 + self.dim2 + self.dim3 + self.dim4
        self.input_height = self.input_height
        self.input_width = self.input_width
        self.input_dim = self.input_height // 32 * self.input_width // 32 * 8
        self.conv_pth = conv_pth
        self.res_model = TtnnResnet34(conv_args, conv_pth.res_model, device=device)
        self.pool = TtnnUFLDV2Conv2D(conv_args.pool, conv_pth.pool, activation=None, device=device)

    def __call__(self, input, batch_size=1, grid_size=(8, 8), tile_size=32):
        fea = self.res_model(input, batch_size=batch_size)
        fea, out_h, out_w = self.pool(fea)
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(grid_size[0] - 1, grid_size[1] - 1),
                )
            }
        )
        mem_config = ttnn.create_sharded_memory_config_(
            ttnn.Shape([fea.shape[-2] * fea.shape[-1], tile_size]),
            shard_grid,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
            tile_layout=True,
        )
        fea = ttnn.to_memory_config(fea, mem_config)
        fea = ttnn.permute(fea, (0, 1, 3, 2))
        fea = ttnn.reshape(fea, (fea.shape[0], fea.shape[1], 1, fea.shape[2] * fea.shape[3]))
        compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        out = ttnn.linear(
            fea,
            self.conv_pth.cls.linear_1.weight,
            bias=self.conv_pth.cls.linear_1.bias,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=compute_config,
        )
        out = ttnn.relu(out)
        out = ttnn.linear(
            out,
            self.conv_pth.cls.linear_2.weight,
            bias=self.conv_pth.cls.linear_2.bias,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=compute_config,
        )
        return out
