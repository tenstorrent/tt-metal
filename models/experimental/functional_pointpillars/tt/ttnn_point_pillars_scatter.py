# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch

# Giving low pcc
# class TtPointPillarsScatter:
#     def __init__(self, in_channels=64, output_shape=[496, 432], device=None):
#         self.output_shape = output_shape
#         self.ny = output_shape[0]
#         self.nx = output_shape[1]
#         self.in_channels = in_channels
#         self.fp16_enabled = False
#         self.device = device

#     def __call__(self, voxel_features, coors, batch_size):
#         batch_canvas = []
#         for batch_itt in range(batch_size):
#             canvas = ttnn.zeros([self.in_channels, self.nx * self.ny], dtype=voxel_features.dtype, device=self.device)

#             # Only include non-empty pillars
#             batch_mask = coors[:, 0] == batch_itt
#             this_coors = ttnn.to_torch(coors)[ttnn.to_torch(batch_mask).to(torch.int), :]
#             this_coors = ttnn.from_torch(this_coors, device=self.device, layout=ttnn.TILE_LAYOUT)
#             indices = this_coors[:, 2] * self.nx + this_coors[:, 3]
#             # indices = indices.type(torch.long)
#             voxels = ttnn.to_torch(voxel_features)[ttnn.to_torch(batch_mask).to(torch.int), :]
#             voxels = ttnn.from_torch(voxels, device=self.device, layout=ttnn.TILE_LAYOUT)
#             if len(voxels.shape) == 2:
#                 voxels = ttnn.permute(voxels, (1, 0))
#             else:
#                 assert False, "voxels size is different"

#             canvas = ttnn.to_torch(canvas)
#             voxels = ttnn.to_torch(voxels)
#             indices = ttnn.to_torch(indices).to(torch.long)
#             canvas[:, indices] = voxels
#             canvas = ttnn.from_torch(canvas, device=self.device, layout=ttnn.TILE_LAYOUT)

#             batch_canvas.append(canvas)

#         # Stack to 3-dim tensor (batch-size, in_channels, nrows*ncols)
#         if len(batch_canvas) > 1:
#             batch_canvas = torch.stack(batch_canvas, 0)  # need to convert to ttnn
#         else:
#             batch_canvas = batch_canvas[0]

#         # Undo the column stacking to final 4-dim tensor
#         batch_canvas = ttnn.reshape(batch_canvas, (batch_size, self.in_channels, self.ny, self.nx))

#         return batch_canvas


class TtPointPillarsScatter:
    def __init__(self, in_channels=64, output_shape=[496, 432], device=None):
        self.output_shape = output_shape
        self.ny = output_shape[0]
        self.nx = output_shape[1]
        self.in_channels = in_channels
        self.fp16_enabled = False
        self.device = device

    def __call__(self, voxel_features, coors, batch_size):
        voxel_features = ttnn.to_torch(voxel_features)
        coors = ttnn.to_torch(coors)
        # batch_canvas will be the final output.
        batch_canvas = []
        # print("batch_size",batch_size)
        for batch_itt in range(1):
            # Create the canvas for this sample
            canvas = torch.zeros(
                self.in_channels, self.nx * self.ny, dtype=voxel_features.dtype, device=voxel_features.device
            )

            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors[:, 2] * self.nx + this_coors[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, in_channels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(1, self.in_channels, self.ny, self.nx)

        batch_canvas = ttnn.from_torch(batch_canvas, device=self.device, layout=ttnn.TILE_LAYOUT)
        return batch_canvas
