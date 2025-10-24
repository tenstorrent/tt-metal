# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import numpy as np
from models.common.lightweightmodule import LightweightModule
from models.experimental.detr3d.ttnn.utils import shift_scale_points_ttnn


class TtnnPositionEmbeddingCoordsSine(LightweightModule):
    def __init__(
        self,
        normalize=False,
        pos_type="fourier",
        parameters=None,
        device=None,
    ):
        super().__init__()
        self.normalize = normalize
        self.device = device
        self.pos_type = pos_type
        if self.pos_type == "fourier":
            assert parameters is not None
            self.gauss_B = parameters.gauss_B
        else:
            raise ValueError(f"Unknown {self.pos_type}, only fourier is currently supported")

    def get_fourier_embeddings(self, xyz, input_range=None):
        # Follows - https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

        bsize, npoints = xyz.shape[0], xyz.shape[1]
        d_in = self.gauss_B.shape[0]
        d_out = self.gauss_B.shape[1]
        assert d_in == xyz.shape[-1]

        # clone coords so that shift/scale operations do not affect original tensor
        xyz_clone = ttnn.clone(xyz)
        src_range = [ttnn.clone(t) for t in input_range]

        if self.normalize:
            xyz_clone = shift_scale_points_ttnn(xyz_clone, src_range=src_range, device=self.device)

        xyz_clone = xyz_clone * (2 * np.pi)
        xyz_clone = ttnn.reshape(xyz_clone, (-1, d_in))
        xyz_proj = ttnn.matmul(xyz_clone, self.gauss_B)
        xyz_proj = ttnn.reshape(xyz_proj, (bsize, npoints, d_out))

        final_embeds = [ttnn.sin(xyz_proj), ttnn.cos(xyz_proj)]

        # return batch x d_pos x npoints embedding
        final_embeds = ttnn.concat(final_embeds, dim=2)

        ttnn.deallocate(xyz_clone)
        ttnn.deallocate(xyz_proj)
        ttnn.deallocate(src_range[0])
        ttnn.deallocate(src_range[1])

        return final_embeds

    def forward(self, xyz, input_range=None):
        if isinstance(xyz, torch.Tensor):
            xyz_ttnn = ttnn.from_torch(xyz, dtype=ttnn.bfloat16, device=self.device, layout=ttnn.TILE_LAYOUT)
        else:
            xyz_ttnn = xyz
        if isinstance(input_range[0], torch.Tensor):
            input_range_ttnn = [
                ttnn.from_torch(t, dtype=ttnn.bfloat16, device=self.device, layout=ttnn.TILE_LAYOUT)
                for t in input_range
            ]
        else:
            input_range_ttnn = input_range
        assert len(xyz_ttnn.shape) == 3
        return self.get_fourier_embeddings(xyz_ttnn, input_range_ttnn)
