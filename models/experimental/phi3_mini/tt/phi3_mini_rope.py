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
        self.orig_context_len = orig_context_len

        # Generate the cos/sin matrices needed for ttnn.embedding op
        short_scaled_cos_matrix, short_scaled_sin_matrix = compute_gather_cos_sin(
            dhead=head_dim,
            end=max_seq_len,
            theta=rope_theta,
            scale_factor=scale_factor,
            ext_scale_tensor=torch.tensor(ext_scale_tensors["short_factor"]),
            position_ids=torch.arange(max_seq_len),
        )
        long_scaled_cos_matrix, long_scaled_sin_matrix = compute_gather_cos_sin(
            dhead=head_dim,
            end=max_seq_len,
            theta=rope_theta,
            scale_factor=scale_factor,
            ext_scale_tensor=torch.tensor(ext_scale_tensors["long_factor"]),
            position_ids=torch.arange(max_seq_len),
        )

        self.cos_matrix, self.sin_matrix = {}, {}
        self.cos_matrix["short_scaled"] = ttnn.from_torch(
            short_scaled_cos_matrix,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            mesh_mapper=ReplicateTensorToMesh(device) if self.is_mesh_device else None,
        )
        self.sin_matrix["short_scaled"] = ttnn.from_torch(
            short_scaled_sin_matrix,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            mesh_mapper=ReplicateTensorToMesh(device) if self.is_mesh_device else None,
        )
        self.cos_matrix["long_scaled"] = ttnn.from_torch(
            long_scaled_cos_matrix,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            mesh_mapper=ReplicateTensorToMesh(device) if self.is_mesh_device else None,
        )
        self.sin_matrix["long_scaled"] = ttnn.from_torch(
            long_scaled_sin_matrix,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            mesh_mapper=ReplicateTensorToMesh(device) if self.is_mesh_device else None,
        )

    def get_rot_mats(self, position_idxs, return_rot_idxs=False):
        device = self.device

        # If position_idxs is a torch tensor, get the TTNN version of it
        if isinstance(position_idxs, torch.Tensor):
            rot_idxs = self.get_rot_idxs(position_idxs)
        else:
            rot_idxs = position_idxs
            assert len(rot_idxs.shape) == 2 and rot_idxs.shape[0] == 1, "rot_idxs must be a [1, batch] tensor"

        # Send the idxs to device
        if rot_idxs.device != device:
            rot_idxs = ttnn.to_device(rot_idxs, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        embedding_layout = ttnn.TILE_LAYOUT

        # Position ids need to be in float32 for ttnn.gt/lt to give ttnn.tensor(1) when true
        float_rot_idxs = ttnn.to_layout(rot_idxs, ttnn.TILE_LAYOUT)
        float_rot_idxs = ttnn.typecast(float_rot_idxs, ttnn.float32)

        # Highest position id between the batches
        max_rot_id = ttnn.max(float_rot_idxs)  # [1, batch]
        ttnn.deallocate(float_rot_idxs)

        # Condition checking for sequence_length > original_context_length
        is_larger = ttnn.gt(max_rot_id, (self.orig_context_len - 1))
        is_smaller = ttnn.lt(max_rot_id, (self.orig_context_len - 1))
        ttnn.deallocate(max_rot_id)

        # Selecting correct embedding tensors based on postion ids
        cos = self.cos_matrix["long_scaled"] * is_larger + self.cos_matrix["short_scaled"] * is_smaller
        sin = self.sin_matrix["long_scaled"] * is_larger + self.sin_matrix["short_scaled"] * is_smaller
        ttnn.deallocate(is_larger)
        ttnn.deallocate(is_smaller)

        cos = ttnn.embedding(rot_idxs, cos, layout=embedding_layout)  # [1, batch, head_dim]
        sin = ttnn.embedding(rot_idxs, sin, layout=embedding_layout)  # [1, batch, head_dim]

        cos = ttnn.unsqueeze_to_4D(cos)  # [1, 1, batch, head_dim]
        sin = ttnn.unsqueeze_to_4D(sin)  # [1, 1, batch, head_dim]

        cos = ttnn.transpose(cos, 1, 2)  # [1, batch, 1[32], head_dim]
        sin = ttnn.transpose(sin, 1, 2)  # [1, batch, 1[32], head_dim]

        if self.batch_size_per_device_group % ttnn.TILE_SIZE != 0:
            cos = cos[:, : self.batch_size_per_device_group, :, :]
            sin = sin[:, : self.batch_size_per_device_group, :, :]

        mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=self.batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        cos = ttnn.interleaved_to_sharded(cos, mem_config)  # [1, 1 (= batch / shard_num_cores), 1[32], self.head_dim]
        sin = ttnn.interleaved_to_sharded(sin, mem_config)  # [1, 1 (= batch / shard_num_cores), 1[32], self.head_dim]

        if return_rot_idxs:
            return [cos, sin], rot_idxs
        return [cos, sin]
