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
        batch_size = 1
        for batch_itt in range(batch_size):
            canvas = ttnn.zeros([self.in_channels, self.nx * self.ny], dtype=ttnn.bfloat16, device=self.device)

            # Only include non-empty pillars
            coors = ttnn.typecast(coors, dtype=ttnn.bfloat16)
            batch_mask = coors[:, 0] == batch_itt

            batch_mask = ttnn.unsqueeze(batch_mask, 0)
            batch_mask = ttnn.unsqueeze(batch_mask, 0)
            batch_mask = ttnn.unsqueeze(batch_mask, 0)
            batch_mask = ttnn.to_layout(batch_mask, ttnn.ROW_MAJOR_LAYOUT)
            nonzero_out = ttnn.nonzero(batch_mask)

            no_of_non_zero_indices = nonzero_out[0][..., 0].item()
            row_indices = nonzero_out[1][:, :, :, :no_of_non_zero_indices]

            this_coors = ttnn.embedding(row_indices, coors)
            this_coors = ttnn.squeeze(this_coors, dim=0)

            this_coors = ttnn.to_torch(this_coors).to(dtype=torch.int32)
            indices = this_coors[:, 2] * self.nx + this_coors[:, 3]
            indices = ttnn.from_torch(indices, dtype=ttnn.uint32, device=self.device)

            voxels = ttnn.embedding(row_indices, voxel_features)
            voxels = ttnn.squeeze(voxels, dim=0)
            voxels_t = ttnn.permute(voxels, (1, 0))  # PCC: 0.9999981845508428

            canvas = ttnn.to_layout(canvas, ttnn.ROW_MAJOR_LAYOUT)
            indices = ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT)
            voxels_t = ttnn.to_layout(voxels_t, ttnn.ROW_MAJOR_LAYOUT)
            expanded_indices = ttnn.unsqueeze(indices, 0)
            expanded_indices = ttnn.repeat(expanded_indices, [64, 1])

            canvas = ttnn.scatter(
                canvas,
                1,
                expanded_indices,
                voxels_t,
            )

            batch_canvas.append(canvas)

        batch_canvas = batch_canvas[0]

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = ttnn.reshape(batch_canvas, (batch_size, self.in_channels, self.ny, self.nx))

        return batch_canvas
