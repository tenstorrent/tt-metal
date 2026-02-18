# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import nearest_32
from models.tt_transformers.tt.common import RopeScaling, gather_cos_sin, get_rot_transformation_mat, precompute_freqs
from ttnn import ReplicateTensorToMesh, ShardTensor2dMesh


def compute_gather_cos_sin(dhead, end, theta, scale_factor, orig_context_len, position_ids):
    cos, sin = precompute_freqs(dhead, end, theta, scale_factor, orig_context_len)
    return gather_cos_sin(position_ids, cos, sin)


class RotarySetup(LightweightModule):
    def __init__(
        self,
        device,
        batch_size: int,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float,
        rope_scaling: Optional[RopeScaling],
        use_qk_fused: bool = False,  # For Qwen2.5 VL, we do not use qk fused ops (rotary embedding + paged cache update)
        datatype=ttnn.bfloat16,
    ):
        super().__init__()

        self.use_qk_fused = use_qk_fused
        self.batch_size = batch_size
        self.rope_deltas = torch.zeros(batch_size, dtype=torch.int32)
        self.head_dim = head_dim
        self.device = device
        self.is_mesh_device = isinstance(device, ttnn._ttnn.multi_device.MeshDevice)
        self.num_devices = device.get_num_devices() if self.is_mesh_device else 1

        # When fused QK ops are enabled, the transformation matrix and batch grid
        # must cover 2x the batch (one set for Q, one for K)
        doubled_batch_size = batch_size * 2 if use_qk_fused else batch_size
        if self.num_devices == 32:
            self.batch_size_per_device_group = max(doubled_batch_size // list(device.shape)[1], 1)
        else:
            self.batch_size_per_device_group = doubled_batch_size
        self.core_grid = device.compute_with_storage_grid_size()
        self.datatype = datatype

        # Generate the cos/sin matrices needed for ttnn.embedding op
        self.cos_matrix_pt, self.sin_matrix_pt = compute_gather_cos_sin(
            dhead=head_dim,
            end=max_seq_len * 2,
            theta=rope_theta,
            scale_factor=rope_scaling.factor if rope_scaling is not None else None,
            orig_context_len=rope_scaling.original_max_position_embeddings if rope_scaling is not None else None,
            position_ids=torch.arange(max_seq_len),
        )
        self.setup_cos_sin()

        self.batch_grid = (
            ttnn.CoreGrid(y=4, x=8)
            if ttnn.get_arch_name() == "blackhole"
            else ttnn.num_cores_to_corerangeset(self.batch_size_per_device_group, self.core_grid, row_wise=True)
        )
        # Generate the transformation matrix (doubled when fused QK is enabled)
        trans_mat = get_rot_transformation_mat(dhead=ttnn.TILE_SIZE).repeat(
            1,
            1,
            self.batch_size_per_device_group,
            1,
        )  # Repeat across all cores on device
        trans_mat_mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
            core_grid=self.batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.transformation_mat = ttnn.from_torch(
            trans_mat,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            memory_config=trans_mat_mem_config,
            mesh_mapper=(
                ShardTensor2dMesh(
                    device,
                    dims=(None, 2) if (self.num_devices == 32 and batch_size > 1) else (None, None),
                    mesh_shape=list(device.shape),
                )
                if self.is_mesh_device
                else None
            ),
        )

        # TODO: Colman, should this be TILE_SIZE or head_dim? Why should it be different for prefill and decode?
        prefill_trans_mat_torch = get_rot_transformation_mat(dhead=head_dim)
        self.transformation_mat_prefill = ttnn.from_torch(
            prefill_trans_mat_torch,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ReplicateTensorToMesh(device) if self.is_mesh_device else None,
        )

    def update_cos_sin(self, cos_matrix_pt=None, sin_matrix_pt=None):
        if cos_matrix_pt is not None:
            self.cos_matrix_pt.copy_(cos_matrix_pt)
        if sin_matrix_pt is not None:
            self.sin_matrix_pt.copy_(sin_matrix_pt)

        # [INFO] we avoid re-allocating the cos_matrix and sin_matrix tensors to allow for correct processing of captured trace
        assert hasattr(self, "cos_matrix")
        assert hasattr(self, "sin_matrix")
        assert (
            self.cos_matrix_pt.shape == self.cos_matrix.shape
        ), "cos_matrix must be the same size as the existing cos_matrix"
        assert (
            self.sin_matrix_pt.shape == self.sin_matrix.shape
        ), "sin_matrix must be the same size as the existing sin_matrix"
        for mat, mat_tt in zip((self.cos_matrix_pt, self.sin_matrix_pt), (self.cos_matrix, self.sin_matrix)):
            ttnn.copy_host_to_device_tensor(
                ttnn.unsqueeze_to_4D(
                    ttnn.from_torch(
                        mat,
                        device=None,
                        layout=ttnn.TILE_LAYOUT,
                        dtype=self.datatype,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
                    )
                ),
                mat_tt,
            )

    def setup_cos_sin(self):
        for mat, attr_name in zip((self.cos_matrix_pt, self.sin_matrix_pt), ("cos_matrix", "sin_matrix")):
            setattr(
                self,
                attr_name,
                ttnn.from_torch(
                    mat,
                    device=self.device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=self.datatype,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
                ),
            )

    def get_rot_idxs(self, position_idxs, on_host=False):
        assert isinstance(position_idxs, torch.Tensor), "Position ids must be a torch tensor"
        assert len(position_idxs.shape) == 1, "position idxs must be a [batch] tensor"
        assert torch.min(position_idxs) >= 0, "position idxs must be non-negative"

        position_idxs = position_idxs + self.rope_deltas
        if self.use_qk_fused:
            position_idxs = position_idxs.repeat(2)

        batch = position_idxs.shape[0]
        position_idxs_2d = position_idxs.reshape(1, batch)
        pad_size = nearest_32(batch) - batch
        position_idxs_2d = torch.nn.functional.pad(position_idxs_2d, (0, pad_size), "constant", 0)

        if on_host:
            rot_idxs = ttnn.as_tensor(
                position_idxs_2d,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if self.is_mesh_device else None,
            )
        else:
            rot_idxs = ttnn.as_tensor(
                position_idxs_2d,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if self.is_mesh_device else None,
            )

        return rot_idxs  # [1, padded_batch] tensor

    def get_rot_mats(self, position_idxs, return_rot_idxs=False):
        device = self.device

        assert not isinstance(position_idxs, torch.Tensor), "position_idxs must be a ttnn tensor"
        assert position_idxs.device != device, "rot_idxs must be on device"
        assert len(position_idxs.shape) == 2 and position_idxs.shape[0] == 1, "rot_idxs must be a [1, batch] tensor"

        if position_idxs.device != device:
            position_idxs = ttnn.to_device(position_idxs, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        cos = ttnn.embedding(position_idxs, self.cos_matrix, layout=ttnn.TILE_LAYOUT)  # [1, batch, head_dim]
        sin = ttnn.embedding(position_idxs, self.sin_matrix, layout=ttnn.TILE_LAYOUT)  # [1, batch, head_dim]

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
            return [cos, sin], position_idxs
        return [cos, sin]

    def get_both_trans_mats(self):
        assert self.transformation_mat is not None, "Transformation matrix not initialized"
        assert self.transformation_mat_prefill is not None, "Prefill Transformation matrix not initialized"
        return {"decode": self.transformation_mat, "prefill": self.transformation_mat_prefill}
