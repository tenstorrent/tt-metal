# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch


# Giving low pcc
class TtPointPillarsScatter:
    def __init__(self, in_channels=64, output_shape=[496, 432], device=None):
        self.output_shape = output_shape
        self.ny = output_shape[0]
        self.nx = output_shape[1]
        self.in_channels = in_channels
        self.fp16_enabled = False
        self.device = device

    def __call__(self, voxel_features, coors, batch_size):
        batch_canvas = []
        for batch_itt in range(batch_size):
            canvas = ttnn.zeros([self.in_channels, self.nx * self.ny], dtype=voxel_features.dtype, device=self.device)

            # Only include non-empty pillars
            coors = ttnn.typecast(coors, dtype=ttnn.bfloat16)
            batch_mask = coors[:, 0] == batch_itt

            batch_mask = ttnn.unsqueeze(batch_mask, 0)
            batch_mask = ttnn.unsqueeze(batch_mask, 0)
            batch_mask = ttnn.unsqueeze(batch_mask, 0)
            batch_mask = ttnn.to_layout(batch_mask, ttnn.ROW_MAJOR_LAYOUT)
            nonzero_out = ttnn.nonzero(batch_mask)

            row_indices = nonzero_out[1]
            row_indices = ttnn.reshape(row_indices, (-1,))
            row_indices = ttnn.to_layout(row_indices, ttnn.TILE_LAYOUT)
            row_indices = ttnn.typecast(row_indices, dtype=ttnn.uint32)
            coors = ttnn.typecast(coors, dtype=ttnn.bfloat16)
            this_coors = ttnn.embedding(row_indices, coors)

            indices = this_coors[:, 2] * self.nx + this_coors[:, 3]

            voxel_features = ttnn.typecast(voxel_features, dtype=ttnn.bfloat16)
            voxels = ttnn.embedding(row_indices, voxel_features)
            voxels_t = ttnn.permute(voxels, (1, 0))  # PCC: 0.9999981845508428

            print("indices: ", indices)
            print("canvas: ", canvas)
            print("voxels_t: ", voxels_t)

            # indices = ttnn.typecast(indices, ttnn.uint32)

            # canvas = ttnn.to_layout(canvas, ttnn.ROW_MAJOR_LAYOUT)
            # indices = ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT)
            # voxels_t = ttnn.to_layout(voxels_t, ttnn.ROW_MAJOR_LAYOUT)

            # expanded_indices = ttnn.unsqueeze(indices, 0)          # [1, 6522]
            # print("expanded_indices: ", expanded_indices)
            # expanded_indices = ttnn.repeat(expanded_indices, [64, 1])
            # # print("expanded_indices: ", expanded_indices)

            # canvas = ttnn.scatter(
            #     canvas,  # input_tensor (base canvas)
            #     1,  # dim = 1 (your axis=1)
            #     expanded_indices,  # integer indices
            #     voxels_t,  # updates
            # )
            indices = ttnn.to_torch(indices).to(torch.long)
            canvas = ttnn.to_torch(canvas)
            voxels_t = ttnn.to_torch(voxels_t)
            print("##### indices: ", indices, type(indices))
            print("##### canvas: ", canvas, type(canvas))
            print("##### voxels_t: ", voxels_t, type(voxels_t))

            canvas[:, indices] = voxels_t
            print("canvas: ", canvas.shape)

            canvas = ttnn.from_torch(canvas, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

            # return canvas
            batch_canvas.append(canvas)

        batch_canvas = batch_canvas[0]
        print("batch_canvas: ", batch_canvas.shape)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = ttnn.reshape(batch_canvas, (batch_size, self.in_channels, self.ny, self.nx))
        print("batch_canvas: ", batch_canvas.shape)

        return batch_canvas
