# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.common import precompute_vision_freqs
from ttnn import ReplicateTensorToMesh


def compute_gather_cos_sin(dhead, max_patches_per_side, theta, scale_factor, orig_context_len, position_ids):
    cos, sin = precompute_vision_freqs(dhead, max_patches_per_side, theta, scale_factor, orig_context_len)
    return cos, sin


class VisionRotarySetup(LightweightModule):
    def __init__(
        self,
        device,
        batch_size: int,
        head_dim: int,
        image_size: int,
        patch_size: int,
        max_seq_len: int,
        rope_theta: float,
        scale_factor: float,  # use None to disable rope scaling
        orig_context_len: int,  # only used if scaling enabled
        datatype=ttnn.bfloat16,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.head_dim = head_dim
        self.device = device
        self.is_mesh_device = isinstance(device, ttnn._ttnn.multi_device.MeshDevice)
        self.num_devices = device.get_num_devices() if self.is_mesh_device else 1
        if self.num_devices == 32:
            self.batch_size_per_device_group = max(self.batch_size // list(device.shape)[1], 1)
        else:
            self.batch_size_per_device_group = self.batch_size
        self.core_grid = device.compute_with_storage_grid_size()

        max_patches_per_side = image_size // patch_size

        # Generate the cos/sin matrices needed for ttnn.embedding op
        cos_matrix, sin_matrix = compute_gather_cos_sin(
            dhead=head_dim,
            max_patches_per_side=max_patches_per_side,
            theta=rope_theta,
            scale_factor=scale_factor,
            orig_context_len=orig_context_len,
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

    def get_rot_mats(self, position_idxs, return_rot_idxs=False):
        device = self.device

        # return self.cos_matrix, self.sin_matrix
        # If position_idxs is a torch tensor, get the TTNN version of it
        if isinstance(position_idxs, torch.Tensor):
            rot_idxs = position_idxs.unsqueeze(0)
        else:
            rot_idxs = position_idxs
            assert len(rot_idxs.shape) == 2 and rot_idxs.shape[0] == 1, "rot_idxs must be a [1, batch] tensor"

        rot_idxs = ttnn.from_torch(
            rot_idxs,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ReplicateTensorToMesh(device) if self.is_mesh_device else None,
        )
        # Send the idxs to device
        if rot_idxs.device != device:
            rot_idxs = ttnn.to_device(rot_idxs, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        embedding_layout = ttnn.TILE_LAYOUT
        cos = ttnn.embedding(rot_idxs, self.cos_matrix, layout=embedding_layout)  # [1, batch, head_dim]
        sin = ttnn.embedding(rot_idxs, self.sin_matrix, layout=embedding_layout)  # [1, batch, head_dim]

        if return_rot_idxs:
            return [cos, sin], rot_idxs
        ttnn.deallocate(rot_idxs)
        return [cos, sin]
