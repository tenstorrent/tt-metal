# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn import ReplicateTensorToMesh, ShardTensor2dMesh
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.rope import RotarySetup as TTTransformerRotarySetup


class RotarySetup(LightweightModule):
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
        super().__init__()

        # favor composition over inheritance: __ is convention for private variables
        self.__tt_transformer_rotary_setup = TTTransformerRotarySetup(
            device=device,
            batch_size=batch_size,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            scale_factor=scale_factor,
            orig_context_len=orig_context_len,
            datatype=datatype,
        )

        self.batch_size = batch_size
        self.head_dim = head_dim
        self.device = device
        self.datatype = datatype
        self.is_mesh_device = self.__tt_transformer_rotary_setup.is_mesh_device
        self.num_devices = self.__tt_transformer_rotary_setup.num_devices
        self.batch_size_per_device_group = self.__tt_transformer_rotary_setup.batch_size_per_device_group
        self.core_grid = self.__tt_transformer_rotary_setup.core_grid

    @property
    def transformation_mat(self):
        return self.__tt_transformer_rotary_setup.transformation_mat

    @property
    def transformation_mat_prefill(self):
        return self.__tt_transformer_rotary_setup.transformation_mat_prefill

    @property
    def cos_matrix(self):
        return self.__tt_transformer_rotary_setup.cos_matrix

    @property
    def sin_matrix(self):
        return self.__tt_transformer_rotary_setup.sin_matrix

    def set_cos_sin(self, cos_matrix, sin_matrix):
        self.__tt_transformer_rotary_setup.set_cos_sin(cos_matrix, sin_matrix)

    def get_both_trans_mats(self):
        return self.__tt_transformer_rotary_setup.get_both_trans_mats()

    def get_rot_idxs(self, position_idxs, on_host=False):
        assert isinstance(position_idxs, torch.Tensor), "Position ids must be a torch tensor"
        assert len(position_idxs.shape) == 1, "position idxs must be a [batch] tensor"
        assert torch.min(position_idxs) >= 0, "position idxs must be non-negative"

        if on_host:  # If tensor is on host, don't pass a mesh mapper if single-device
            rot_idxs = ttnn.as_tensor(
                position_idxs,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ReplicateTensorToMesh(self.device) if self.is_mesh_device else None,
            )
        else:  # On device
            rot_idxs = ttnn.as_tensor(
                position_idxs,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ReplicateTensorToMesh(self.device) if self.is_mesh_device else None,
            )

        return rot_idxs  # [batch] tensor

    def get_rot_mats(self, position_idxs, return_rot_idxs=False):
        device = self.device

        # If position_idxs is a torch tensor, get the TTNN version of it
        assert not isinstance(position_idxs, torch.Tensor), "position_idxs must be a ttnn tensor"
        assert position_idxs.device != device, "rot_idxs must be on device"

        # [INFO] Qwen2.5 VL produces cos and sin matrices with shape [batch_size, 1, seq_len, head_dim]
        # todo)) { Optimize the slicing work-around below
        batch_size = position_idxs.shape[0]
        cos, sin = None, None
        for i in range(batch_size):
            cos_i = ttnn.embedding(position_idxs[i : i + 1, ...], self.cos_matrix[i : i + 1, ...])  # [1, head_dim]
            sin_i = ttnn.embedding(position_idxs[i : i + 1, ...], self.sin_matrix[i : i + 1, ...])  # [1, head_dim]

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
