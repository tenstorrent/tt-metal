# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from typing_extensions import override

import ttnn
from models.tt_transformers.tt.rope import RotarySetup as TTTransformerRotarySetup


class RotarySetup(TTTransformerRotarySetup):
    def __init__(
        self,
        device,
        batch_size: int,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float,
        scale_factor: float,  # use None to disable rope scaling
        orig_context_len: int,  # only used if scaling enabled
        datatype=ttnn.bfloat16,
    ):
        # Call parent constructor - this will initialize all the necessary attributes
        super().__init__(
            device=device,
            batch_size=batch_size,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            scale_factor=scale_factor,
            orig_context_len=orig_context_len,
            datatype=datatype,
        )

    @override
    def set_cos_sin(self, cos_matrix, sin_matrix):
        # [INFO] we avoid re-allocating the cos_matrix and sin_matrix tensors to allow for correct processing of captured trace
        if hasattr(self, "cos_matrix"):
            assert (
                cos_matrix.shape == self.cos_matrix.shape
            ), "cos_matrix must be the same size as the existing cos_matrix"
            assert (
                sin_matrix.shape == self.sin_matrix.shape
            ), "sin_matrix must be the same size as the existing sin_matrix"

            for mat, mat_tt in zip((cos_matrix, sin_matrix), (self.cos_matrix, self.sin_matrix)):
                mat = ttnn.from_torch(
                    mat,
                    device=None,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=self.datatype,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
                )
                mat = ttnn.unsqueeze_to_4D(mat)
                ttnn.copy_host_to_device_tensor(mat, mat_tt)
        else:
            # [INFO] tt-transformers RotarySetup uses a single cos_matrix and sin_matrix for all batches
            assert (
                cos_matrix.shape[0] == 1 and sin_matrix.shape[0] == 1
            ), "Init values of cos_matrix and sin_matrix must have batch size 1"
            for mat, attr_name in zip((cos_matrix, sin_matrix), ("cos_matrix", "sin_matrix")):
                setattr(
                    self,
                    attr_name,
                    ttnn.from_torch(
                        mat.expand(
                            self.batch_size, -1, -1, -1
                        ),  # [INFO] Qwen2.5 VL produces cos and sin matrices with shape [batch_size, 1, seq_len, head_dim]
                        device=self.device,
                        layout=ttnn.TILE_LAYOUT,
                        dtype=self.datatype,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
                    ),
                )

    @override
    def get_rot_idxs(self, position_idxs, on_host=False):
        assert isinstance(position_idxs, torch.Tensor), "Position ids must be a torch tensor"
        assert len(position_idxs.shape) == 1, "position idxs must be a [batch] tensor"
        assert torch.min(position_idxs) >= 0, "position idxs must be non-negative"

        if on_host:  # If tensor is on host, don't pass a mesh mapper if single-device
            rot_idxs = ttnn.as_tensor(
                position_idxs,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if self.is_mesh_device else None,
            )
        else:  # On device
            rot_idxs = ttnn.as_tensor(
                position_idxs,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if self.is_mesh_device else None,
            )

        return rot_idxs  # [batch] tensor

    @override
    def get_rot_mats(self, position_idxs, return_rot_idxs=False):
        device = self.device

        # If position_idxs is a torch tensor, get the TTNN version of it
        assert not isinstance(position_idxs, torch.Tensor), "position_idxs must be a ttnn tensor"
        assert position_idxs.device != device, "rot_idxs must be on device"

        # [INFO] Qwen2.5 VL produces cos and sin matrices with shape [batch_size, 1, seq_len, head_dim]
        # todo)) { Optimize the slicing work-around below
        assert len(position_idxs.shape) == 1, "position_idxs must be a [batch] tensor"
        batch_size = position_idxs.shape[0]
        cos, sin = None, None
        for i in range(batch_size):
            # [INFO] This is a work-around to avoid the slicing issue in position_idxs[i:i+1]
            # todo)) this workaround can be removed after pulling changes from `main` --> the bug is fixed there
            pos_i = ttnn.squeeze(ttnn.reshape(position_idxs, (batch_size, 1))[i : i + 1], dim=-1)
            cos_i = ttnn.embedding(pos_i, self.cos_matrix[i : i + 1, ...])  # [1, head_dim]
            sin_i = ttnn.embedding(pos_i, self.sin_matrix[i : i + 1, ...])  # [1, head_dim]

            cos = cos_i if cos is None else ttnn.concat([cos, cos_i], dim=0)  # towards [batch_size, head_dim]
            sin = sin_i if sin is None else ttnn.concat([sin, sin_i], dim=0)  # towards [batch_size, head_dim]

        cos = ttnn.to_layout(cos, ttnn.TILE_LAYOUT)
        sin = ttnn.to_layout(sin, ttnn.TILE_LAYOUT)
        # } todo))

        cos = ttnn.unsqueeze_to_4D(cos)  # [1, 1, batch_size, head_dim]
        sin = ttnn.unsqueeze_to_4D(sin)  # [1, 1, batch_size, head_dim]

        cos = ttnn.transpose(cos, 1, 2)  # [1, batch_size, 1[32], head_dim]
        sin = ttnn.transpose(sin, 1, 2)  # [1, batch_size, 1[32], head_dim]

        if self.batch_size_per_device_group % ttnn.TILE_SIZE != 0:
            cos = cos[:, : self.batch_size_per_device_group, :, :]
            sin = sin[:, : self.batch_size_per_device_group, :, :]

        grid = ttnn.num_cores_to_corerangeset(self.batch_size, self.core_grid, row_wise=True)
        mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        cos = ttnn.interleaved_to_sharded(cos, mem_config)  # [1, 1 (= batch / shard_num_cores), 1[32], self.head_dim]
        sin = ttnn.interleaved_to_sharded(sin, mem_config)  # [1, 1 (= batch / shard_num_cores), 1[32], self.head_dim]

        if return_rot_idxs:
            return [cos, sin], position_idxs
        return [cos, sin]
