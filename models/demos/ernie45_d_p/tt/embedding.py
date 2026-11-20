# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch

import ttnn
from models.demos.ernie45_d_p.tt.common import cache_name, replicate


class TtEmbedding:
    """Replicated [V, H] bf16 table (0.49 GB/chip); output [1, 1, S, H] tiled, replicated."""

    def __init__(self, mesh, weight: torch.Tensor):
        self.mesh = mesh
        self.weight = replicate(
            mesh, weight.to(torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT, cache=cache_name("embed")
        )

    def __call__(self, tokens: torch.Tensor):
        ids = ttnn.from_torch(
            tokens.reshape(1, 1, 1, -1).to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        x = ttnn.embedding(ids, self.weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        ttnn.deallocate(ids)
        return ttnn.reshape(x, [1, 1, tokens.numel(), x.shape[-1]])
