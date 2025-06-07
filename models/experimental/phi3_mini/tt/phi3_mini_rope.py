# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn import ReplicateTensorToMesh
from models.tt_transformers.tt.rope import RotarySetup
from models.experimental.phi3_mini.tt.phi3_mini_common import precompute_freqs
from models.tt_transformers.tt.common import gather_cos_sin


def compute_gather_cos_sin(dhead, end, theta, scale_factor, ext_scale_tensor, position_ids):
    cos, sin = precompute_freqs(dhead, end, theta, scale_factor, ext_scale_tensor)
    return gather_cos_sin(position_ids, cos, sin)


class Phi3MiniRotarySetup(RotarySetup):
    def __init__(
        self,
        device,
        batch_size: int,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float,
        scale_factor: float,
        ext_scale_tensors: dict,
        orig_context_len: int,
        datatype=ttnn.bfloat16,
    ):
        super().__init__(
            device=device,
            batch_size=batch_size,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            scale_factor=scale_factor,
            orig_context_len=orig_context_len,
        )
        # Generate the cos/sin matrices needed for ttnn.embedding op
        if max_seq_len > orig_context_len:
            cos_matrix, sin_matrix = compute_gather_cos_sin(
                dhead=head_dim,
                end=max_seq_len,
                theta=rope_theta,
                scale_factor=scale_factor,
                ext_scale_tensor=torch.tensor(ext_scale_tensors["long_factor"]),
                position_ids=torch.arange(max_seq_len),
            )
        else:
            cos_matrix, sin_matrix = compute_gather_cos_sin(
                dhead=head_dim,
                end=max_seq_len,
                theta=rope_theta,
                scale_factor=scale_factor,
                ext_scale_tensor=torch.tensor(ext_scale_tensors["short_factor"]),
                position_ids=torch.arange(max_seq_len),
            )

        self.cos_matrix = ttnn.from_torch(
            cos_matrix,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            mesh_mapper=ReplicateTensorToMesh(device) if self.is_mesh_device else None,
        )
        self.sin_matrix = ttnn.from_torch(
            sin_matrix,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            mesh_mapper=ReplicateTensorToMesh(device) if self.is_mesh_device else None,
        )
