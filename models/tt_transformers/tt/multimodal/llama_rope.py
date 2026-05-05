# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Any, List, Optional

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.common import RopeScaling, get_rot_transformation_mat
from models.tt_transformers.tt.multimodal.tensor_utils import from_torch_host_to_device
from models.tt_transformers.tt.prefetcher import Prefetcher
from models.tt_transformers.tt.rope import RotarySetup as BaseRotarySetup
from models.tt_transformers.tt.rope import compute_gather_cos_sin
from ttnn import replicate_tensor_to_mesh_mapper


def get_rot_mats(
    head_dim: int,
    device: Any,
    seq_len: int,
    theta: float,
    rope_scaling: Optional[RopeScaling],
    datatype: Any = ttnn.bfloat16,
    rot_mats_layout: ttnn.Layout = ttnn.TILE_LAYOUT,
) -> List[ttnn.Tensor]:
    cos_matrix, sin_matrix = compute_gather_cos_sin(
        dhead=head_dim,
        end=2 * seq_len,
        theta=theta,
        rope_scaling=rope_scaling,
    )

    cos_matrix = from_torch_host_to_device(
        cos_matrix,
        device=device,
        layout=rot_mats_layout,
        dtype=datatype,
        mesh_mapper=replicate_tensor_to_mesh_mapper(device),
    )
    sin_matrix = from_torch_host_to_device(
        sin_matrix,
        device=device,
        layout=rot_mats_layout,
        dtype=datatype,
        mesh_mapper=replicate_tensor_to_mesh_mapper(device),
    )
    return [cos_matrix, sin_matrix]


class RotarySetup(BaseRotarySetup):
    def __init__(
        self,
        device: Any,
        batch_size: int,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float,
        rope_scaling: Optional[RopeScaling] = None,
        use_qk_fused: bool = False,
        datatype: ttnn.DataType = ttnn.bfloat16,
        shard_batch_to_mesh_dim: Optional[int] = 1,
        prefetcher: Optional[Prefetcher] = None,
    ) -> None:
        LightweightModule.__init__(self)

        self.use_qk_fused = use_qk_fused
        self.original_batch_size = batch_size
        self.prefetcher = prefetcher

        # NOTE: If qk fused ops (rotary embedding + paged cache update) are used
        # we need to double the batch size in order to replicate the transformation matrix on double the batch size number of cores
        self.doubled_batch_size = self.original_batch_size * 2 if use_qk_fused else self.original_batch_size
        self.head_dim = head_dim
        self.device = device
        self.is_mesh_device = isinstance(device, ttnn._ttnn.multi_device.MeshDevice)
        self.num_devices = device.get_num_devices() if self.is_mesh_device else 1
        if self.num_devices == 32:
            self.batch_size_per_device_group = max(
                self.doubled_batch_size // list(device.shape)[shard_batch_to_mesh_dim], 1
            )
        else:
            self.batch_size_per_device_group = self.doubled_batch_size

        self.core_grid = (
            device.compute_with_storage_grid_size() if ttnn.get_arch_name() == "blackhole" else ttnn.CoreCoord(8, 8)
        )
        self.start_core = ttnn.CoreCoord(1, 0)

        self.cos_matrix, self.sin_matrix = get_rot_mats(
            head_dim=head_dim,
            device=device,
            seq_len=max_seq_len,
            theta=rope_theta,
            rope_scaling=rope_scaling,
            datatype=datatype,
            rot_mats_layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        self.cos_matrix_prefill, self.sin_matrix_prefill = get_rot_mats(
            head_dim=head_dim,
            device=device,
            seq_len=max_seq_len,
            theta=rope_theta,
            rope_scaling=rope_scaling,
            datatype=datatype,
            rot_mats_layout=ttnn.TILE_LAYOUT,
        )

        def get_batch_grid(batch_size, core_grid, start_core, batch_size_per_device_group, prefetcher):
            if ttnn.get_arch_name() == "blackhole":
                if prefetcher is not None:
                    return ttnn.num_cores_to_corerangeset_in_subcoregrids(
                        start_core,
                        batch_size_per_device_group,
                        prefetcher.all_worker_cores_range_set,
                        row_wise=True,
                    )
                if batch_size % 32 == 0:
                    return ttnn.CoreGrid(y=8, x=8)
                return ttnn.num_cores_to_corerangeset(batch_size, core_grid, row_wise=True)
            return ttnn.num_cores_to_corerangeset(batch_size, core_grid, row_wise=True)

        self.batch_grid = get_batch_grid(
            self.batch_size_per_device_group,
            self.core_grid,
            self.start_core,
            self.batch_size_per_device_group,
            self.prefetcher,
        )

        trans_mat = get_rot_transformation_mat(dhead=ttnn.TILE_SIZE).repeat(
            1,
            1,
            self.batch_size_per_device_group,
            1,
        )
        trans_mat_mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
            core_grid=self.batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.transformation_mat = from_torch_host_to_device(
            trans_mat,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            memory_config=trans_mat_mem_config,
            mesh_mapper=replicate_tensor_to_mesh_mapper(device),
        )

        prefill_trans_mat_torch = get_rot_transformation_mat(dhead=head_dim)
        self.transformation_mat_prefill = from_torch_host_to_device(
            prefill_trans_mat_torch,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate_tensor_to_mesh_mapper(device),
        )
