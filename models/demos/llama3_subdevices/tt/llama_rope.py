# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.llama3_subdevices.tt.llama_common import precompute_freqs, get_rot_transformation_mat, gather_cos_sin
from models.utility_functions import nearest_32


def compute_gather_cos_sin(dhead, end, theta, position_ids, use_scaled_rope, scale_factor):
    cos, sin = precompute_freqs(dhead, end, theta, use_scaled_rope, scale_factor)
    return gather_cos_sin(position_ids, cos, sin)


class TtLlamaRotarySetup(LightweightModule):
    def __init__(
        self,
        device,
        batch_size: int,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float = 10000,
        use_scaled_rope: bool = False,
        scale_factor: float = 8,
        datatype=ttnn.bfloat16,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.head_dim = head_dim
        self.n_kv_heads = 8
        self.device = device
        self.is_mesh_device = isinstance(device, ttnn._ttnn.multi_device.MeshDevice)
        self.num_devices = device.get_num_devices() if self.is_mesh_device else 1
        if self.num_devices == 32:
            self.batch_size_per_device_group = (
                max(self.batch_size // list(device.shape)[1], 1) * 2
            )  # TODO: fix for batch=1
        else:
            self.batch_size_per_device_group = self.batch_size
        self.core_grid = device.compute_with_storage_grid_size()
        num_cores = self.core_grid.x * self.core_grid.y
        self.sub_core_grids = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
            ]
        )
        self.start_core = ttnn.CoreCoord(1, 0)

        # Generate the cos/sin matrices needed for ttnn.embedding op
        cos_matrix, sin_matrix = compute_gather_cos_sin(
            dhead=head_dim,
            end=max_seq_len * 2,
            theta=rope_theta,
            position_ids=torch.arange(max_seq_len),
            use_scaled_rope=use_scaled_rope,
            scale_factor=scale_factor,
        )

        self.cos_matrix = ttnn.from_torch(
            cos_matrix,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=datatype,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if self.is_mesh_device else None,
        )
        self.sin_matrix = ttnn.from_torch(
            sin_matrix,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=datatype,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if self.is_mesh_device else None,
        )

        self.core_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(
            self.start_core, self.batch_size_per_device_group, self.sub_core_grids, row_wise=True
        )
        # Generate the transformation matrix
        trans_mat = get_rot_transformation_mat(dhead=ttnn.TILE_SIZE).repeat(
            1,
            1,
            self.batch_size_per_device_group,
            1
            # 1, 1, num_cores, 1
        )  # Repeat across all cores on device
        trans_mat_mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
            core_grid=self.core_grid,
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
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if self.is_mesh_device else None,
        )

        # TODO: Colman, should this be TILE_SIZE or head_dim? Why should it be different for prefill and decode?
        prefill_trans_mat_torch = get_rot_transformation_mat(dhead=head_dim)
        self.transformation_mat_prefill = ttnn.from_torch(
            prefill_trans_mat_torch,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if self.is_mesh_device else None,
        )

    def get_both_trans_mats(self):
        assert self.transformation_mat is not None, "Transformation matrix not initialized"
        assert self.transformation_mat_prefill is not None, "Prefill Transformation matrix not initialized"
        return {"decode": self.transformation_mat, "prefill": self.transformation_mat_prefill}

    def get_rot_idxs(self, position_idxs, on_host=False):
        assert isinstance(position_idxs, torch.Tensor), "Position ids must be a torch tensor"
        assert len(position_idxs.shape) == 1, "position idxs must be a [batch] tensor"
        assert position_idxs.shape[0] == 32, "position idxs must be a [32] tensor"
        # repeating twice at every 8th position for fused kv rope
        position_idxs = position_idxs.view(-1, 8)  # [4, 8]
        position_idxs = position_idxs.repeat(1, 2)  # [4, 16]
        position_idxs = position_idxs.view(-1)  # [64]
        batch = position_idxs.shape[0]
        position_idxs = position_idxs.reshape(1, batch)  # [1, 1, 1, batch]
        assert position_idxs.shape == (1, batch), "position idxs must be a [1, batch] tensor"
        assert torch.min(position_idxs) >= 0, "position idxs must be non-negative"

        # Add padding if needed
        pad_size = nearest_32(batch) - batch
        position_idxs = torch.nn.functional.pad(position_idxs, (0, pad_size), "constant", 0)
        position_idxs = position_idxs.transpose(1, 0)
        position_idxs = torch.nn.functional.pad(position_idxs, (0, 31), "constant", 0)

        if on_host:  # If tensor is on host, don't pass a mesh mapper if single-device
            rot_idxs = ttnn.as_tensor(
                position_idxs,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.device,
                    dims=(None, 0) if (self.num_devices == 32 and batch > 1) else (None, None),
                    mesh_shape=list(self.device.shape),
                )
                if self.is_mesh_device
                else None,
            )
        else:  # On device
            rot_idxs = ttnn.as_tensor(
                position_idxs,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.device,
                    dims=(None, 0) if (self.num_devices == 32 and batch > 1) else (None, None),
                    mesh_shape=list(self.device.shape),
                )
                if self.is_mesh_device
                else None,
            )

        return rot_idxs

    def get_rm_rot_idxs(self, position_idxs, on_host=False):
        assert isinstance(position_idxs, torch.Tensor), "Position ids must be a torch tensor"
        assert len(position_idxs.shape) == 1, "position idxs must be a [batch] tensor"
        assert position_idxs.shape[0] == 32, "position idxs must be a [32] tensor"
        position_idxs = position_idxs.view(-1, 8)  # [4, 8]
        position_idxs = position_idxs.repeat(1, 2)  # [4, 16]
        position_idxs = position_idxs.view(-1, 1)  # [64, 1]
        position_idxs = position_idxs.repeat(1, self.n_kv_heads)  # [64, 8]

        batch = position_idxs.shape[0]

        rot_idxs = ttnn.as_tensor(
            position_idxs,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=None if on_host else self.device,
            memory_config=None if on_host else ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.device,
                dims=(None, 0) if (self.num_devices == 32 and batch > 1) else (None, None),
                mesh_shape=list(self.device.shape),
            )
            if self.is_mesh_device
            else None,
        )

        return rot_idxs

    def get_rot_mats(self, position_idxs, return_rot_idxs=False):
        device = self.device

        # If position_idxs is a torch tensor, get the TTNN version of it
        if isinstance(position_idxs, torch.Tensor):
            rot_idxs = self.get_rot_idxs(position_idxs)
        else:
            rot_idxs = position_idxs
            # assert len(rot_idxs.shape) == 2 and rot_idxs.shape[0] == 1, "rot_idxs must be a [1, batch] tensor"

        # Send the idxs to device
        if rot_idxs.device != device:
            rot_idxs = ttnn.to_device(rot_idxs, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=self.core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        embedding_layout = ttnn.TILE_LAYOUT
        cos = ttnn.embedding(
            rot_idxs, self.cos_matrix, layout=embedding_layout, memory_config=mem_config
        )  # [1, batch, head_dim]
        sin = ttnn.embedding(
            rot_idxs, self.sin_matrix, layout=embedding_layout, memory_config=mem_config
        )  # [1, batch, head_dim]

        cos = ttnn.reshape(
            cos,
            ttnn.Shape(
                [self.batch_size_per_device_group, 1, cos.shape[-1]]
                # (self.batch_size_per_device_group, 32, cos.shape[-1]),
            ),
            ttnn.Shape(
                # [self.batch_size_per_device_group, 1, sin.shape[-1]]
                (self.batch_size_per_device_group, 32, sin.shape[-1]),
            ),
        )
        sin = ttnn.reshape(
            sin,
            ttnn.Shape(
                [self.batch_size_per_device_group, 1, sin.shape[-1]]
                # (self.batch_size_per_device_group, 32, sin.shape[-1]),
            ),
            ttnn.Shape(
                # [self.batch_size_per_device_group, 1, sin.shape[-1]]
                (self.batch_size_per_device_group, 32, sin.shape[-1]),
            ),
        )

        cos = ttnn.unsqueeze_to_4D(cos)  # [1, 1, batch, head_dim]
        sin = ttnn.unsqueeze_to_4D(sin)  # [1, 1, batch, head_dim]

        if return_rot_idxs:
            return [cos, sin], rot_idxs
        return [cos, sin]

    def get_rm_rot_mats(self, position_idxs, return_rot_idxs=False):
        device = self.device

        # If position_idxs is a torch tensor, get the TTNN version of it
        if isinstance(position_idxs, torch.Tensor):
            rot_idxs = self.get_rm_rot_idxs(position_idxs)
        else:
            rot_idxs = position_idxs

        # Send the idxs to device
        if rot_idxs.device != device:
            rot_idxs = ttnn.to_device(rot_idxs, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        mem_config = ttnn.create_sharded_memory_config(
            shape=(self.n_kv_heads, self.head_dim),
            core_grid=self.core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        embedding_layout = ttnn.ROW_MAJOR_LAYOUT
        cos = ttnn.embedding(
            rot_idxs, self.cos_matrix, layout=embedding_layout, memory_config=mem_config
        )  # [batch, n_kv_heads, head_dim]
        sin = ttnn.embedding(
            rot_idxs, self.sin_matrix, layout=embedding_layout, memory_config=mem_config
        )  # [batch, n_kv_heads, head_dim]

        cos = ttnn.unsqueeze_to_4D(cos)  # [1, batch, n_kv_heads, head_dim]
        sin = ttnn.unsqueeze_to_4D(sin)  # [1, batch, n_kv_heads, head_dim]

        if return_rot_idxs:
            return [cos, sin], rot_idxs
        return [cos, sin]
