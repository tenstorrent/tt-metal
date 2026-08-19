# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Real-hardware qualification for the common Attention2D on WH Galaxy."""

from __future__ import annotations

import gc
import math
import traceback
from dataclasses import dataclass
from typing import Any

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.common.models.galaxy import GalaxyCollectivePlan, GalaxyResourceKey, GalaxyResourcesConfig, GalaxyTensorSpec
from models.common.modules.attention.attention_2d import (
    Attention2D,
    Attention2DConfig,
    Attention2DLowLevelCallables,
    Attention2DSequenceConfig,
    DecodeMetadata,
    KVCacheBinding,
    PrefillAttentionMode,
    PrefillCollectiveMode,
    PrefillMetadata,
    PrefillRecipeIdentity,
    PrefillRowMode,
)
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.rmsnorm.rmsnorm_2d import RMSNorm2DConfig, RMSNorm2DGeometry
from models.common.tests.modules._wh_galaxy_hardware import (
    deallocate_module_weights,
    deallocate_tensor,
    exact_tensor_resource,
    galaxy_mode_plan,
    require_galaxy_ccl_hardware_resources,
)
from models.common.utility_functions import comp_pcc

_MESH_SHAPE = (8, 4)
_BATCH_SIZE = 32
_HEAD_DIM = 128
_MAX_SEQ_LEN = 2048
_PCC = 0.99


@dataclass(frozen=True)
class _ModelSpec:
    name: str
    dim: int
    n_heads: int
    n_kv_heads: int
    qk_norm: bool
    norm_eps: float

    @property
    def qkv_size(self) -> int:
        return (self.n_heads + 2 * self.n_kv_heads) * _HEAD_DIM

    @property
    def weight_count(self) -> int:
        return 4 if self.qk_norm else 2


_MODEL_SPECS = (
    _ModelSpec("llama-70b", dim=8192, n_heads=64, n_kv_heads=8, qk_norm=False, norm_eps=1e-5),
    _ModelSpec("qwen3-32b", dim=5120, n_heads=40, n_kv_heads=8, qk_norm=True, norm_eps=1e-6),
)


def _mesh_mapper(*placements: Any) -> ttnn.MeshMapperConfig:
    return ttnn.MeshMapperConfig(
        placements=list(placements),
        mesh_shape_override=ttnn.MeshShape(*_MESH_SHAPE),
    )


def _lazy_weight(
    source: torch.Tensor,
    mesh_device: ttnn.MeshDevice,
    mapper: ttnn.MeshMapperConfig,
    *,
    dtype: ttnn.DataType = ttnn.bfloat8_b,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> LazyWeight:
    return LazyWeight(
        source=source,
        device=mesh_device,
        mesh_mapper_config=mapper,
        memory_config=memory_config,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
    )


def _compute_kernel() -> ttnn.WormholeComputeKernelConfig:
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


def _matmul_program(sequence_length: int, local_k: int, local_n: int) -> Any:
    grid_x = 7
    grid_y = min(4, max(1, math.ceil(sequence_length / 32)))
    m_tiles = max(1, math.ceil(sequence_length / 32))
    n_tiles = local_n // 32
    k_tiles = local_k // 32
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=math.gcd(k_tiles, 8),
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=math.ceil(m_tiles / grid_y),
        per_core_N=math.ceil(n_tiles / grid_x),
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )


def _decode_ring_config(spec: _ModelSpec) -> dict[str, Any]:
    ring_coords = (
        (6, 6),
        (6, 7),
        (6, 9),
        (6, 0),
        (6, 1),
        (6, 2),
        (6, 4),
        (6, 5),
        (5, 5),
        (5, 6),
        (5, 7),
        (5, 9),
        (5, 0),
        (5, 1),
        (5, 2),
        (5, 4),
        (1, 4),
        (1, 5),
        (1, 9),
        (1, 0),
        (2, 0),
        (2, 4),
        (2, 5),
        (2, 9),
    )
    receiver_coords = (
        (1, 9),
        (2, 9),
        (1, 0),
        (2, 0),
        (1, 4),
        (2, 4),
        (1, 5),
        (2, 5),
        (5, 0),
        (6, 0),
        (5, 9),
        (6, 9),
        (5, 1),
        (6, 1),
        (5, 7),
        (6, 7),
        (5, 6),
        (6, 6),
        (5, 2),
        (6, 2),
        (5, 4),
        (6, 4),
        (5, 5),
        (6, 5),
    )

    def points(coords: tuple[tuple[int, int], ...]) -> ttnn.CoreRangeSet:
        return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*coord), ttnn.CoreCoord(*coord)) for coord in coords])

    def padded(value: int) -> int:
        return math.ceil(value / (32 * 24)) * 32 * 24

    def memory_config(width: int, cores: ttnn.CoreRangeSet) -> ttnn.MemoryConfig:
        return ttnn.create_sharded_memory_config(
            shape=(32, width // 24),
            core_grid=cores,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def program_config(k: int, n: int) -> Any:
        out_block_w = n // 24 // 32
        out_subblock_w = min(8, out_block_w)
        while out_block_w % out_subblock_w:
            out_subblock_w -= 1
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 3),
            in0_block_w=k // 24 // 32,
            out_subblock_h=1,
            out_subblock_w=out_subblock_w,
            per_core_M=1,
            per_core_N=out_block_w,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
            gather_in0=True,
            hop_cores=points(((3, 6),)),
            num_global_cb_receivers=2,
            untilize_out=False,
        )

    qkv_k = padded(spec.dim // _MESH_SHAPE[1])
    qkv_n = padded(spec.qkv_size // _MESH_SHAPE[0])
    wo_k = padded(spec.dim // _MESH_SHAPE[0])
    wo_n = padded(spec.dim // _MESH_SHAPE[1])
    ring_cores, output_cores = points(ring_coords), points(receiver_coords)
    return {
        "qkv_k": qkv_k,
        "qkv_n": qkv_n,
        "wo_k": wo_k,
        "wo_n": wo_n,
        "input_memcfg": memory_config(qkv_k, ring_cores),
        "qkv_output_memcfg": memory_config(qkv_n, output_cores),
        "wo_input_memcfg": memory_config(wo_k, ring_cores),
        "wo_output_memcfg": memory_config(wo_n, output_cores),
        "qkv_program": program_config(spec.dim // _MESH_SHAPE[1], qkv_n),
        "wo_program": program_config(spec.dim // _MESH_SHAPE[0], wo_n),
        "ring_cores": ring_cores,
    }


def _sdpa_program(
    sequence_length: int, *, decode: bool, sub_core_grids: ttnn.CoreRangeSet | None = None
) -> ttnn.SDPAProgramConfig:
    if decode:
        q_chunk_size = k_chunk_size = 0
    elif sequence_length < 2048:
        q_chunk_size = k_chunk_size = 64
    else:
        q_chunk_size, k_chunk_size = 256, 512
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(8, 4) if decode else (7, 10),
        sub_core_grids=sub_core_grids,
        exp_approx_mode=False,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
    )


class _AttentionCollectives:
    """Attention low-level adapters backed only by production Galaxy resources."""

    def __init__(
        self,
        resources: Any,
        mesh_device: ttnn.MeshDevice,
        decode_input_memcfgs: dict[int, ttnn.MemoryConfig],
        decode_output_memcfgs: dict[int, ttnn.MemoryConfig],
        decode_reduce_scatter_memcfgs: dict[int, ttnn.MemoryConfig],
        decode_qkv_input_memcfg: ttnn.MemoryConfig,
        decode_heads_memcfg: ttnn.MemoryConfig,
        decode_gather_users_memcfg: ttnn.MemoryConfig,
        persistent_decode: bool = True,
    ):
        self.resources = resources
        self.mesh_device = mesh_device
        self.decode_input_memcfgs = decode_input_memcfgs
        self.decode_output_memcfgs = decode_output_memcfgs
        self.decode_reduce_scatter_memcfgs = decode_reduce_scatter_memcfgs
        self.decode_qkv_input_memcfg = decode_qkv_input_memcfg
        self.decode_heads_memcfg = decode_heads_memcfg
        self.decode_gather_users_memcfg = decode_gather_users_memcfg
        self.persistent_decode = persistent_decode
        self._reduce_scatter_staging: tuple[Any, ...] = ()
        self._fused_batch_offsets: Any = None
        self._fused_reduced_outputs: list[Any] = []

    def cleanup(self) -> None:
        for tensor in self._fused_reduced_outputs:
            deallocate_tensor(tensor)
        self._fused_reduced_outputs.clear()
        deallocate_tensor(self._fused_batch_offsets)
        self._fused_batch_offsets = None
        for tensor in self._reduce_scatter_staging:
            deallocate_tensor(tensor)
        self._reduce_scatter_staging = ()

    def _all_reduce(self, tensor: ttnn.Tensor, *, mode: str, cluster_axis: int) -> ttnn.Tensor:
        context = self.resources.context(mode)
        operation = (
            "reduce_scatter" if mode == "decode" and self.persistent_decode and cluster_axis == 1 else "all_reduce"
        )
        resource = exact_tensor_resource(context, operation, cluster_axis, tensor)
        if not resource.persistent_output_buffers:
            raise ValueError(f"{mode} Attention2D all_reduce axis {cluster_axis} requires a persistent output")
        output_memcfg = self.decode_output_memcfgs[cluster_axis] if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG
        if mode == "decode" and self.persistent_decode:
            interleaved_input = ttnn.to_memory_config(tensor, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat8_b)
            collective_input = ttnn.to_memory_config(
                interleaved_input, self.decode_input_memcfgs[cluster_axis], ttnn.bfloat8_b
            )
            deallocate_tensor(interleaved_input)
            key = resource.key
            if cluster_axis == 1:
                if resource.topology == ttnn.Topology.Ring and not self._reduce_scatter_staging:
                    self._reduce_scatter_staging = tuple(
                        ttnn.experimental.reduce_scatter_minimal_async_create_intermediate_buffer(
                            collective_input,
                            dim=3,
                            topology=resource.topology,
                            cluster_axis=cluster_axis,
                        )
                    )
                persistent_buffers = [
                    resource.intermediate_output_buffers[0],
                    resource.persistent_output_buffers[0],
                ]
                if self._reduce_scatter_staging:
                    persistent_buffers = [
                        self._reduce_scatter_staging[0],
                        resource.persistent_output_buffers[0],
                        self._reduce_scatter_staging[1],
                    ]
                reduced = ttnn.experimental.reduce_scatter_minimal_async(
                    collective_input,
                    persistent_output_buffers=persistent_buffers,
                    dim=3,
                    multi_device_global_semaphore=context.next_semaphore_handles(
                        key.operation, key.cluster_axis, key.geometry, key.sequence_key
                    ),
                    barrier_semaphore=context.next_barrier_semaphore_handle(
                        key.operation, key.cluster_axis, key.geometry, key.sequence_key
                    ),
                    num_links=resource.num_links,
                    cluster_axis=cluster_axis,
                    memory_config=self.decode_reduce_scatter_memcfgs[cluster_axis],
                    topology=resource.topology,
                    subdevice_id=context.worker_sub_device_id,
                    intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    chunks_per_sync=10,
                    num_workers_per_link=2,
                    num_buffers_per_channel=2,
                )
                gather = exact_tensor_resource(context, "all_gather", cluster_axis, reduced)
                gather_key = gather.key
                output = ttnn.experimental.all_gather_async(
                    reduced,
                    3,
                    multi_device_global_semaphore=context.next_semaphore_window(
                        gather_key.operation,
                        gather_key.cluster_axis,
                        gather_key.geometry,
                        gather_key.sequence_key,
                        count=2,
                    ),
                    num_links=gather.num_links,
                    cluster_axis=cluster_axis,
                    mesh_device=self.mesh_device,
                    topology=gather.topology,
                    memory_config=output_memcfg,
                    persistent_output_tensor=gather.persistent_output_buffers[0],
                    barrier_semaphore=None,
                    subdevice_id=context.worker_sub_device_id,
                    use_optimal_ccl_for_llama=gather.topology == ttnn.Topology.Ring,
                )
            else:
                output = ttnn.experimental.all_reduce_async(
                    collective_input,
                    resource.persistent_output_buffers[0],
                    cluster_axis=cluster_axis,
                    mesh_device=self.mesh_device,
                    multi_device_global_semaphore=context.next_semaphore_handles(
                        key.operation, key.cluster_axis, key.geometry, key.sequence_key
                    ),
                    num_links=resource.num_links,
                    memory_config=output_memcfg,
                    dtype=ttnn.bfloat8_b,
                    topology=resource.topology,
                    subdevice_id=context.worker_sub_device_id,
                    use_optimal_ccl_for_llama=True,
                )
            deallocate_tensor(collective_input)
            restored_interleaved = ttnn.to_memory_config(output, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)
            restored = ttnn.to_memory_config(restored_interleaved, output_memcfg, ttnn.bfloat16)
            deallocate_tensor(restored_interleaved)
            return restored

        reduced_memcfg = ttnn.DRAM_MEMORY_CONFIG
        collective_input = tensor
        reduced = ttnn.reduce_scatter(
            collective_input,
            3,
            cluster_axis=cluster_axis,
            num_links=resource.num_links,
            topology=resource.topology,
            memory_config=reduced_memcfg,
            subdevice_id=context.worker_sub_device_id,
        )
        if collective_input is not tensor:
            deallocate_tensor(collective_input)
        output = ttnn.all_gather(
            reduced,
            3,
            cluster_axis=cluster_axis,
            num_links=resource.num_links,
            topology=resource.topology,
            memory_config=output_memcfg,
            subdevice_id=context.worker_sub_device_id,
        )
        self.resources.synchronize(mode)
        deallocate_tensor(reduced)
        return output

    def is_borrowed_output(self, tensor: Any) -> bool:
        return False

    @staticmethod
    def rotary(q: ttnn.Tensor, k: ttnn.Tensor, rot_mats: Any, **_: Any) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        # RoPE has its own Milestone-A hardware qualification. Identity rotation keeps
        # this test focused on Attention2D projection, SDPA, cache, and CCL ownership.
        assert rot_mats is None
        return q, k

    def reduce_qkv(self, tensor: ttnn.Tensor, *, mode: str, **_: Any) -> ttnn.Tensor:
        return self._all_reduce(tensor, mode=mode, cluster_axis=1)

    def reduce_create_qkv_heads(
        self, tensor: ttnn.Tensor, *, mode: str, config: Attention2DConfig, **_: Any
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        if mode != "decode":
            raise ValueError("fused QKV head creation is decode-only")
        context = self.resources.context(mode)
        resource = exact_tensor_resource(context, "all_reduce_create_qkv_heads", 1, tensor)
        key = resource.key
        collective_input = ttnn.to_memory_config(tensor, self.decode_qkv_input_memcfg, dtype=ttnn.bfloat8_b)
        if self._fused_batch_offsets is None:
            self._fused_batch_offsets = ttnn.as_tensor(
                torch.tensor(config.batch_offsets, dtype=torch.int32).reshape(4, 1),
                dtype=ttnn.int32,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device=self.mesh_device, dims=(None, 0), mesh_shape=_MESH_SHAPE
                ),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        reduced, q, k, v = ttnn.experimental.all_reduce_create_qkv_heads(
            collective_input,
            resource.persistent_output_buffers[0],
            cluster_axis=1,
            mesh_device=self.mesh_device,
            multi_device_global_semaphore=context.next_semaphore_handles(
                key.operation, key.cluster_axis, key.geometry, key.sequence_key
            ),
            num_heads=config.n_heads // _MESH_SHAPE[0],
            memory_config=self.decode_output_memcfgs[1],
            topology=resource.topology,
            num_links=resource.num_links,
            subdevice_id=context.worker_sub_device_id,
            num_kv_heads=config.n_kv_heads // _MESH_SHAPE[0],
            final_memory_config=self.decode_heads_memcfg,
            batch_offset=self._fused_batch_offsets,
            slice_size=config.users_per_column,
            dtype=ttnn.bfloat16,
        )
        self.resources.synchronize(mode)
        deallocate_tensor(collective_input)
        self._fused_reduced_outputs.append(reduced)
        return q, k, v

    @staticmethod
    def gather_heads(tensor: ttnn.Tensor, **_: Any) -> ttnn.Tensor:
        # Users are gathered before concat; the result is already the row-local K shard.
        return tensor

    def gather_users(self, tensor: ttnn.Tensor, *, mode: str, **_: Any) -> ttnn.Tensor:
        if mode != "decode":
            raise ValueError("user gather is decode-only")
        context = self.resources.context(mode)
        resource = exact_tensor_resource(context, "all_gather", 1, tensor)
        output = ttnn.all_gather(
            tensor,
            1,
            cluster_axis=1,
            num_links=resource.num_links,
            topology=resource.topology,
            memory_config=self.decode_gather_users_memcfg,
            subdevice_id=context.worker_sub_device_id,
        )
        self.resources.synchronize(mode)
        return output

    def reduce_output(self, tensor: ttnn.Tensor, *, mode: str, **_: Any) -> ttnn.Tensor:
        return self._all_reduce(tensor, mode=mode, cluster_axis=0)

    def callables(self) -> Attention2DLowLevelCallables:
        return Attention2DLowLevelCallables(
            rotary=self.rotary,
            reduce_qkv=self.reduce_qkv,
            gather_heads=self.gather_heads,
            reduce_output=self.reduce_output,
            is_borrowed_output=self.is_borrowed_output,
            reduce_create_qkv_heads=self.reduce_create_qkv_heads,
            gather_users=self.gather_users,
        )


def _runtime_tensor_factory(offsets: tuple[int, ...], lower: tuple[int, ...], upper: tuple[int, ...], mesh: Any):
    mapper = ttnn.ReplicateTensorToMesh(mesh)

    def make(values: tuple[int, ...]) -> ttnn.Tensor:
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32),
            device=mesh,
            mesh_mapper=mapper,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return make(offsets), make(lower), make(upper)


def _fused_qkv_weight(wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor, spec: _ModelSpec) -> torch.Tensor:
    q_chunks = torch.chunk(wq, _MESH_SHAPE[0], dim=-1)
    k_chunks = torch.chunk(wk, _MESH_SHAPE[0], dim=-1)
    v_chunks = torch.chunk(wv, _MESH_SHAPE[0], dim=-1)
    return torch.cat(
        [torch.cat((q_chunks[row], k_chunks[row], v_chunks[row]), dim=-1) for row in range(_MESH_SHAPE[0])],
        dim=-1,
    ).contiguous()


def _sequence_config(
    spec: _ModelSpec, sequence_length: int, sub_core_grids: ttnn.CoreRangeSet
) -> Attention2DSequenceConfig:
    identity = PrefillRecipeIdentity(
        sequence_length,
        PrefillRowMode.SINGLE_ROW,
        PrefillCollectiveMode.REGULAR,
        PrefillAttentionMode.REGULAR,
    )
    local_qkv = spec.qkv_size // _MESH_SHAPE[0]
    local_input = spec.dim // _MESH_SHAPE[1]
    local_hidden = spec.dim // _MESH_SHAPE[0]
    local_output = spec.dim // _MESH_SHAPE[1]
    kernel = _compute_kernel()
    return Attention2DSequenceConfig(
        identity=identity,
        qkv_program_config=_matmul_program(sequence_length, local_input, local_qkv),
        sdpa_program_config=_sdpa_program(sequence_length, decode=False, sub_core_grids=sub_core_grids),
        wo_program_config=_matmul_program(sequence_length, local_hidden, local_output),
        qkv_output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        heads_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        kv_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        sdpa_output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        concat_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        wo_output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        qkv_kernel_config=kernel,
        sdpa_kernel_config=kernel,
        wo_kernel_config=kernel,
        activation_dtype=ttnn.bfloat16,
    )


def _resource_plan(
    cluster_axis: int,
    input_shape: tuple[int, ...],
    *,
    persistent_output_spec: GalaxyTensorSpec | None = None,
) -> GalaxyCollectivePlan:
    return GalaxyCollectivePlan(
        key=GalaxyResourceKey("all_reduce", cluster_axis, input_shape, math.prod(input_shape[:-1])),
        topology=ttnn.Topology.Linear if cluster_axis == 1 else ttnn.Topology.Ring,
        num_links=1 if cluster_axis == 1 else 4,
        persistent_output_specs=(
            persistent_output_spec
            or GalaxyTensorSpec(input_shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
        ),
    )


def _resources_config(
    mesh_device: ttnn.MeshDevice,
    spec: _ModelSpec,
    decode_all_reduce: dict[str, Any],
) -> GalaxyResourcesConfig:
    def plans(sequence_length: int, *, decode: bool = False) -> tuple[GalaxyCollectivePlan, ...]:
        leading = (1, 1, sequence_length)
        qkv_width = spec.qkv_size // _MESH_SHAPE[0]
        output_width = spec.dim // _MESH_SHAPE[1]
        if decode:
            local_heads = spec.n_heads // _MESH_SHAPE[0]
            qkv_heads = GalaxyCollectivePlan(
                key=GalaxyResourceKey("all_reduce_create_qkv_heads", 1, (*leading, qkv_width), sequence_length),
                topology=ttnn.Topology.Ring,
                num_links=3,
                persistent_output_specs=(decode_all_reduce["qkv_scratch_spec"],),
            )
            return (
                qkv_heads,
                GalaxyCollectivePlan(
                    key=GalaxyResourceKey(
                        "all_gather",
                        1,
                        (1, 8, local_heads, _HEAD_DIM),
                        8 * local_heads,
                    ),
                    topology=ttnn.Topology.Ring,
                    num_links=1,
                    persistent_output_specs=(
                        GalaxyTensorSpec(
                            (1, 32, local_heads, _HEAD_DIM),
                            ttnn.bfloat16,
                            ttnn.TILE_LAYOUT,
                            decode_all_reduce["gather_users_memcfg"],
                        ),
                    ),
                ),
                _resource_plan(
                    0,
                    (*leading, output_width),
                    persistent_output_spec=decode_all_reduce["output_buffer_spec"],
                ),
            )
        return (
            _resource_plan(
                1,
                (*leading, qkv_width),
            ),
            _resource_plan(
                0,
                (*leading, output_width),
            ),
        )

    return GalaxyResourcesConfig(
        architecture=ttnn.device.Arch.WORMHOLE_B0,
        prefill=galaxy_mode_plan(
            "prefill", tuple(plan for length in (128, 2048) for plan in plans(length)), mesh_device
        ),
        decode=galaxy_mode_plan(
            "decode",
            plans(_BATCH_SIZE, decode=True),
            mesh_device,
            semaphore_cores=decode_all_reduce["worker_cores"],
        ),
    )


def _decode_all_reduce_config(
    spec: _ModelSpec, mesh_device: ttnn.MeshDevice, ring_cores: ttnn.CoreRangeSet
) -> dict[str, Any]:
    worker_cores = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )
    qkv_core_count = (spec.qkv_size // _MESH_SHAPE[0]) // _HEAD_DIM
    qkv_cores = ttnn.num_cores_to_corerangeset_in_subcoregrids(
        ttnn.CoreCoord(1, 0), qkv_core_count, worker_cores, row_wise=False
    )
    head_cores = ttnn.num_cores_to_corerangeset_in_subcoregrids(
        ttnn.CoreCoord(1, 0), _BATCH_SIZE, worker_cores, row_wise=False
    )
    sdpa_cores = ttnn.num_cores_to_corerangeset_in_subcoregrids(
        ttnn.CoreCoord(1, 0), _BATCH_SIZE, worker_cores, row_wise=True
    )
    gather_user_cores = ttnn.num_cores_to_corerangeset_in_subcoregrids(
        ttnn.CoreCoord(1, 0), _BATCH_SIZE, worker_cores, row_wise=True
    )
    kv_cores = ttnn.num_cores_to_corerangeset_in_subcoregrids(
        ttnn.CoreCoord(1, 0), _BATCH_SIZE // _MESH_SHAPE[1], worker_cores, row_wise=False
    )
    sdpa_output_cores = ttnn.num_cores_to_corerangeset_in_subcoregrids(
        ttnn.CoreCoord(1, 0), _BATCH_SIZE // _MESH_SHAPE[1], worker_cores, row_wise=True
    )
    output_core_count = (spec.dim // _MESH_SHAPE[1]) // _HEAD_DIM
    output_cores = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(1, 0),
                ttnn.CoreCoord(2, output_core_count // 2 - 1),
            )
        }
    )
    output_reduce_scatter_cores = ttnn.num_cores_to_corerangeset_in_subcoregrids(
        ttnn.CoreCoord(1, 0), 8, worker_cores, row_wise=False
    )

    def output_memcfg(cores: ttnn.CoreRangeSet) -> ttnn.MemoryConfig:
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(cores, (32, _HEAD_DIM), ttnn.ShardOrientation.ROW_MAJOR),
        )

    def buffer_spec(cluster_axis: int) -> GalaxyTensorSpec:
        shard_width = max(1024, _HEAD_DIM * _MESH_SHAPE[cluster_axis])
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(worker_cores, (32, shard_width), ttnn.ShardOrientation.ROW_MAJOR),
        )
        return GalaxyTensorSpec(
            (*_MESH_SHAPE, 32, shard_width * worker_cores.num_cores()),
            ttnn.bfloat8_b,
            ttnn.TILE_LAYOUT,
            memory_config,
            ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=_MESH_SHAPE),
        )

    return {
        "qkv_input_memcfg": ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(ring_cores, (32, 64), ttnn.ShardOrientation.ROW_MAJOR),
        ),
        "qkv_output_memcfg": output_memcfg(qkv_cores),
        "output_memcfg": output_memcfg(output_cores),
        "qkv_reduce_scatter_memcfg": ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(qkv_cores, (32, 32), ttnn.ShardOrientation.ROW_MAJOR),
        ),
        "output_reduce_scatter_memcfg": ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(output_reduce_scatter_cores, (32, 32), ttnn.ShardOrientation.ROW_MAJOR),
        ),
        "heads_output_memcfg": ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(head_cores, (32, _HEAD_DIM), ttnn.ShardOrientation.ROW_MAJOR),
        ),
        "kv_output_memcfg": ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(kv_cores, (32, _HEAD_DIM), ttnn.ShardOrientation.ROW_MAJOR),
        ),
        "sdpa_output_memcfg": ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(sdpa_output_cores, (32, _HEAD_DIM), ttnn.ShardOrientation.ROW_MAJOR),
        ),
        "gather_users_memcfg": ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(gather_user_cores, (32, _HEAD_DIM), ttnn.ShardOrientation.ROW_MAJOR),
        ),
        "head_cores": head_cores,
        "sdpa_cores": sdpa_cores,
        "kv_cores": kv_cores,
        "worker_cores": worker_cores,
        "qkv_scratch_spec": GalaxyTensorSpec(
            (*_MESH_SHAPE, 32, (spec.qkv_size // _MESH_SHAPE[0]) * _MESH_SHAPE[1]),
            ttnn.bfloat8_b,
            ttnn.TILE_LAYOUT,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(qkv_cores, (32, 512), ttnn.ShardOrientation.ROW_MAJOR),
            ),
            ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=_MESH_SHAPE),
        ),
        "qkv_buffer_spec": buffer_spec(1),
        "output_buffer_spec": buffer_spec(0),
    }


def _qk_norm_config(
    weight: torch.Tensor | None, spec: _ModelSpec, mesh_device: ttnn.MeshDevice
) -> RMSNorm2DConfig | None:
    if weight is None:
        return None
    return RMSNorm2DConfig(
        weight=LazyWeight(source=weight, device=mesh_device, dtype=ttnn.bfloat16),
        mesh_device=mesh_device,
        geometry=RMSNorm2DGeometry.HEAD_LOCAL,
        eps=spec.norm_eps,
        decode_input_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        prefill_input_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        decode_output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        prefill_output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
    )


def _make_module(
    spec: _ModelSpec,
    mesh_device: ttnn.MeshDevice,
    resources: Any,
    lazy_wqkv: LazyWeight,
    lazy_wo: LazyWeight,
    prefill_wqkv: LazyWeight,
    prefill_wo: LazyWeight,
    decode_ring: dict[str, Any],
    decode_all_reduce: dict[str, Any],
    q_norm: torch.Tensor | None,
    k_norm: torch.Tensor | None,
) -> Attention2D:
    wqkv_mapper = lazy_wqkv.mesh_mapper_config
    wo_mapper = lazy_wo.mesh_mapper_config
    collectives = _AttentionCollectives(
        resources,
        mesh_device,
        {1: decode_ring["qkv_output_memcfg"], 0: decode_ring["wo_output_memcfg"]},
        {1: decode_all_reduce["qkv_output_memcfg"], 0: decode_all_reduce["output_memcfg"]},
        {
            1: decode_all_reduce["qkv_reduce_scatter_memcfg"],
            0: decode_all_reduce["output_reduce_scatter_memcfg"],
        },
        decode_all_reduce["qkv_input_memcfg"],
        decode_all_reduce["heads_output_memcfg"],
        decode_all_reduce["gather_users_memcfg"],
        persistent_decode=False,
    )
    local_qkv = spec.qkv_size // _MESH_SHAPE[0]
    local_input = spec.dim // _MESH_SHAPE[1]
    local_hidden = spec.dim // _MESH_SHAPE[0]
    local_output = spec.dim // _MESH_SHAPE[1]
    kernel = _compute_kernel()
    module = Attention2D.from_config(
        Attention2DConfig(
            wqkv=lazy_wqkv,
            wo=lazy_wo,
            prefill_wqkv=prefill_wqkv,
            prefill_wo=prefill_wo,
            n_heads=spec.n_heads,
            n_kv_heads=spec.n_kv_heads,
            head_dim=_HEAD_DIM,
            max_batch_size=_BATCH_SIZE,
            max_seq_len=_MAX_SEQ_LEN,
            low_level=collectives.callables(),
            runtime_tensor_factory=_runtime_tensor_factory,
            runtime_tensor_releaser=deallocate_tensor,
            q_norm_config=_qk_norm_config(q_norm, spec, mesh_device),
            k_norm_config=_qk_norm_config(k_norm, spec, mesh_device),
            mesh_device=mesh_device,
            architecture=ttnn.device.Arch.WORMHOLE_B0,
            wqkv_mesh_mapper_config=wqkv_mapper,
            wo_mesh_mapper_config=wo_mapper,
            weight_memory_config=lazy_wqkv.memory_config,
            wo_weight_memory_config=lazy_wo.memory_config,
            weight_layout=ttnn.TILE_LAYOUT,
            wqkv_dtype=ttnn.bfloat8_b,
            wo_dtype=ttnn.bfloat8_b,
            decode_input_placement=ttnn.DRAM_MEMORY_CONFIG,
            decode_output_placement=decode_all_reduce["output_memcfg"],
            prefill_input_placement=ttnn.DRAM_MEMORY_CONFIG,
            prefill_output_placement=ttnn.DRAM_MEMORY_CONFIG,
            decode_qkv_output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            decode_heads_memory_config=decode_all_reduce["heads_output_memcfg"],
            decode_kv_memory_config=decode_all_reduce["kv_output_memcfg"],
            decode_sdpa_output_memory_config=decode_all_reduce["sdpa_output_memcfg"],
            decode_concat_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            decode_concat_sub_core_grids=decode_all_reduce["gather_users_memcfg"].shard_spec.grid,
            decode_wo_output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            decode_program_config=_matmul_program(_BATCH_SIZE, local_input, local_qkv),
            decode_sdpa_program_config=_sdpa_program(
                _MAX_SEQ_LEN, decode=True, sub_core_grids=decode_all_reduce["sdpa_cores"]
            ),
            decode_wo_program_config=_matmul_program(_BATCH_SIZE, local_hidden, local_output),
            decode_qkv_kernel_config=kernel,
            decode_sdpa_kernel_config=kernel,
            decode_wo_kernel_config=kernel,
            decode_activation_dtype=ttnn.bfloat16,
            decode_prefetch_context=None,
            prefill_prefetch_context=None,
            prefill_sequence_configs={
                recipe.identity: recipe
                for recipe in (
                    _sequence_config(spec, 128, decode_all_reduce["worker_cores"]),
                    _sequence_config(spec, 2048, decode_all_reduce["worker_cores"]),
                )
            },
        )
    )
    module._hardware_collectives = collectives
    return module


def _make_cache(module: Attention2D, mesh_device: ttnn.MeshDevice) -> KVCacheBinding:
    shape = (_BATCH_SIZE, module.config.n_kv_heads // _MESH_SHAPE[0], _MAX_SEQ_LEN, _HEAD_DIM)
    mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=_MESH_SHAPE)
    tensors = [
        ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16),
            device=mesh_device,
            mesh_mapper=mapper,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for _ in range(2)
    ]
    return KVCacheBinding(tensors[0], tensors[1], owner=object(), mesh_device=mesh_device)


def _to_device_input(
    x: torch.Tensor, mesh_device: ttnn.MeshDevice, memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
) -> ttnn.Tensor:
    tensor = ttnn.from_torch(
        x,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=_MESH_SHAPE),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if memory_config != ttnn.DRAM_MEMORY_CONFIG:
        tensor = ttnn.to_memory_config(tensor, memory_config)
    return tensor


def _compose_output(output: ttnn.Tensor, mesh_device: ttnn.MeshDevice, spec: _ModelSpec) -> torch.Tensor:
    composed = ttnn.to_torch(
        output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=_MESH_SHAPE),
    )
    return composed[:, :1, :, : spec.dim]


def _compose_cache(cache: ttnn.Tensor, mesh_device: ttnn.MeshDevice, spec: _ModelSpec) -> torch.Tensor:
    composed = ttnn.to_torch(
        cache,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 0), mesh_shape=_MESH_SHAPE),
    )
    return composed[:_BATCH_SIZE, : spec.n_kv_heads, :, :_HEAD_DIM]


def _rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    return x if weight is None else F.rms_norm(x.float(), (_HEAD_DIM,), weight.float(), eps).to(x.dtype)


def _project_qkv(
    x: torch.Tensor,
    spec: _ModelSpec,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    q_norm: torch.Tensor | None,
    k_norm: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = x.reshape(-1, spec.dim).float()
    q = torch.matmul(flat, wq.float()).reshape(*x.shape[:-1], spec.n_heads, _HEAD_DIM)
    k = torch.matmul(flat, wk.float()).reshape(*x.shape[:-1], spec.n_kv_heads, _HEAD_DIM)
    v = torch.matmul(flat, wv.float()).reshape(*x.shape[:-1], spec.n_kv_heads, _HEAD_DIM)
    q = _rms_norm(q, q_norm, spec.norm_eps)
    k = _rms_norm(k, k_norm, spec.norm_eps)
    return q, k, v


def _decode_reference(
    x: torch.Tensor,
    positions: torch.Tensor,
    spec: _ModelSpec,
    weights: tuple[torch.Tensor, ...],
    cache: tuple[torch.Tensor, torch.Tensor],
    norms: tuple[torch.Tensor | None, torch.Tensor | None],
) -> torch.Tensor:
    wq, wk, wv, wo = weights
    q, k, v = _project_qkv(x[0, 0], spec, wq, wk, wv, *norms)
    rows = torch.arange(_BATCH_SIZE)
    cache[0][rows, :, positions, :] = k
    cache[1][rows, :, positions, :] = v
    keys = cache[0][:, :, : int(positions.max()) + 1]
    values = cache[1][:, :, : int(positions.max()) + 1]
    q = q.transpose(0, 1).unsqueeze(2).transpose(0, 1)
    keys = keys.repeat_interleave(spec.n_heads // spec.n_kv_heads, dim=1)
    values = values.repeat_interleave(spec.n_heads // spec.n_kv_heads, dim=1)
    attention = F.scaled_dot_product_attention(q.float(), keys.float(), values.float(), scale=_HEAD_DIM**-0.5)
    merged = attention.transpose(1, 2).reshape(_BATCH_SIZE, 1, spec.dim)
    return torch.matmul(merged, wo.float()).unsqueeze(0)


def _prefill_reference(
    x: torch.Tensor,
    spec: _ModelSpec,
    weights: tuple[torch.Tensor, ...],
    norms: tuple[torch.Tensor | None, torch.Tensor | None],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    wq, wk, wv, wo = weights
    q, k, v = _project_qkv(x[0, 0], spec, wq, wk, wv, *norms)
    q = q.transpose(0, 1).unsqueeze(0)
    keys = k.transpose(0, 1).unsqueeze(0).repeat_interleave(spec.n_heads // spec.n_kv_heads, dim=1)
    values = v.transpose(0, 1).unsqueeze(0).repeat_interleave(spec.n_heads // spec.n_kv_heads, dim=1)
    attention = F.scaled_dot_product_attention(
        q.float(), keys.float(), values.float(), is_causal=True, scale=_HEAD_DIM**-0.5
    )
    merged = attention.transpose(1, 2).reshape(1, x.shape[-2], spec.dim)
    return torch.matmul(merged, wo.float()).unsqueeze(0), k, v


def _assert_pcc(expected: torch.Tensor, actual: torch.Tensor, case: str) -> None:
    passing, message = comp_pcc(expected.float(), actual.float(), _PCC)
    if not passing and case.startswith("decode") and expected.numel() == actual.numel():
        expected_rows = expected.float().reshape(-1, expected.shape[-1])
        actual_rows = actual.float().reshape(-1, actual.shape[-1])
        expected_centered = expected_rows - expected_rows.mean(dim=1, keepdim=True)
        actual_centered = actual_rows - actual_rows.mean(dim=1, keepdim=True)
        correlations = (expected_centered @ actual_centered.T) / (
            expected_centered.norm(dim=1, keepdim=True) * actual_centered.norm(dim=1).unsqueeze(0)
        ).clamp_min(1e-12)
        best_values, best_users = correlations.max(dim=1)
        norm_ratios = actual_rows.norm(dim=1) / expected_rows.norm(dim=1).clamp_min(1e-12)
        message = (
            f"{message}; best_users={best_users.tolist()}, "
            f"best_pcc={[round(value, 4) for value in best_values.tolist()]}, "
            f"norm_ratios={[round(value, 3) for value in norm_ratios.tolist()]}"
        )
    assert passing, f"{case} failed PCC>={_PCC}: {message}"


def _assert_cache(
    binding: KVCacheBinding,
    mesh_device: ttnn.MeshDevice,
    spec: _ModelSpec,
    expected_k: torch.Tensor,
    expected_v: torch.Tensor,
    index: Any,
    case: str,
) -> None:
    actual_k = _compose_cache(binding.keys, mesh_device, spec)[index]
    actual_v = _compose_cache(binding.values, mesh_device, spec)[index]
    _assert_pcc(expected_k, actual_k, f"{case} K cache")
    _assert_pcc(expected_v, actual_v, f"{case} V cache")


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("spec", _MODEL_SPECS, ids=lambda spec: spec.name)
@torch.no_grad()
def test_attention_2d_wh_galaxy_decode_and_prefill_repeat(mesh_device: ttnn.MeshDevice, spec: _ModelSpec):
    torch.manual_seed(17)
    weight_scale = 1.0 / math.sqrt(spec.dim)
    wq = torch.randn(spec.dim, spec.n_heads * _HEAD_DIM, dtype=torch.bfloat16) * weight_scale
    wk = torch.randn(spec.dim, spec.n_kv_heads * _HEAD_DIM, dtype=torch.bfloat16) * weight_scale
    wv = torch.randn(spec.dim, spec.n_kv_heads * _HEAD_DIM, dtype=torch.bfloat16) * weight_scale
    wo = torch.randn(spec.dim, spec.dim, dtype=torch.bfloat16) * weight_scale
    wqkv = _fused_qkv_weight(wq, wk, wv, spec)
    q_norm = 1.0 + 0.05 * torch.randn(_HEAD_DIM, dtype=torch.bfloat16) if spec.qk_norm else None
    k_norm = 1.0 + 0.05 * torch.randn(_HEAD_DIM, dtype=torch.bfloat16) if spec.qk_norm else None
    weights = (wq, wk, wv, wo)
    norms = (q_norm, k_norm)
    decode_ring = _decode_ring_config(spec)
    decode_all_reduce = _decode_all_reduce_config(spec, mesh_device, decode_ring["ring_cores"])
    dram_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(mesh_device.dram_grid_size().x - 1, 0))}
    )

    def dram_sharded(local_k: int, local_n: int) -> ttnn.MemoryConfig:
        padded_n = math.ceil(local_n / (32 * 12)) * 32 * 12
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_grid, (local_k, padded_n // 12), ttnn.ShardOrientation.ROW_MAJOR),
        )

    wqkv_mapper = _mesh_mapper(ttnn.PlacementShard(1), ttnn.PlacementShard(0))
    wo_mapper = _mesh_mapper(ttnn.PlacementShard(0), ttnn.PlacementShard(1))
    lazy_wqkv = _lazy_weight(
        wqkv,
        mesh_device,
        wqkv_mapper,
        memory_config=dram_sharded(spec.dim // _MESH_SHAPE[1], spec.qkv_size // _MESH_SHAPE[0]),
    )
    lazy_wo = _lazy_weight(
        wo,
        mesh_device,
        wo_mapper,
        memory_config=dram_sharded(spec.dim // _MESH_SHAPE[0], spec.dim // _MESH_SHAPE[1]),
    )
    prefill_wqkv = _lazy_weight(wqkv, mesh_device, wqkv_mapper)
    prefill_wo = _lazy_weight(wo, mesh_device, wo_mapper)
    resources = require_galaxy_ccl_hardware_resources(
        mesh_device,
        config=_resources_config(mesh_device, spec, decode_all_reduce),
    )
    module = binding = None
    try:
        module = _make_module(
            spec,
            mesh_device,
            resources,
            lazy_wqkv,
            lazy_wo,
            prefill_wqkv,
            prefill_wo,
            decode_ring,
            decode_all_reduce,
            q_norm,
            k_norm,
        )
        binding = _make_cache(module, mesh_device)
        module.bind_kv_cache(binding)
        reference_cache = (
            torch.zeros(_BATCH_SIZE, spec.n_kv_heads, _MAX_SEQ_LEN, _HEAD_DIM),
            torch.zeros(_BATCH_SIZE, spec.n_kv_heads, _MAX_SEQ_LEN, _HEAD_DIM),
        )

        for invocation, position in enumerate((127, 128)):
            resources.activate("decode")
            x = torch.randn(1, 1, _BATCH_SIZE, spec.dim, dtype=torch.bfloat16) * 0.05
            positions = torch.full((_BATCH_SIZE,), position, dtype=torch.long)
            expected = _decode_reference(x, positions, spec, weights, reference_cache, norms)
            tt_x = _to_device_input(x, mesh_device)
            tt_positions = ttnn.from_torch(
                positions[: module.config.users_per_column].to(torch.int32),
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            output = None
            try:
                output = module.decode_forward(tt_x, None, DecodeMetadata(tt_positions))
                _assert_cache(
                    binding,
                    mesh_device,
                    spec,
                    reference_cache[0][:, :, position],
                    reference_cache[1][:, :, position],
                    (slice(None), slice(None), position),
                    f"decode invocation {invocation}",
                )
                _assert_pcc(expected, _compose_output(output, mesh_device, spec), f"decode invocation {invocation}")
            except BaseException:
                traceback.print_exc()
                raise
            finally:
                if output is not None and not module.output_is_borrowed(output):
                    deallocate_tensor(output)
                deallocate_tensor(tt_positions)
                deallocate_tensor(tt_x)

        resources.activate("prefill")
        for invocation in range(2):
            for sequence_length in (128, 2048):
                x = torch.randn(1, 1, sequence_length, spec.dim, dtype=torch.bfloat16) * 0.05
                expected, expected_k, expected_v = _prefill_reference(x, spec, weights, norms)
                tt_x = _to_device_input(x, mesh_device)
                output = None
                try:
                    output = module.prefill_forward(tt_x, None, PrefillMetadata(sequence_length, user_ids=(0,)))
                    _assert_pcc(
                        expected,
                        _compose_output(output, mesh_device, spec),
                        f"prefill {sequence_length} invocation {invocation}",
                    )
                    _assert_cache(
                        binding,
                        mesh_device,
                        spec,
                        expected_k.transpose(0, 1),
                        expected_v.transpose(0, 1),
                        (0, slice(None), slice(0, sequence_length)),
                        f"prefill {sequence_length} invocation {invocation}",
                    )
                except BaseException:
                    traceback.print_exc()
                    raise
                finally:
                    if output is not None and not module.output_is_borrowed(output):
                        deallocate_tensor(output)
                    deallocate_tensor(tt_x)
    finally:
        try:
            resources.cleanup()
            if module is not None:
                module._hardware_collectives.cleanup()
                module.close()
                deallocate_module_weights(module, "wqkv", "wo", "prefill_wqkv", "prefill_wo")
                for norm in (module._q_norm, module._k_norm):
                    deallocate_tensor(getattr(norm, "weight", None))
            if binding is not None:
                deallocate_tensor(binding.keys)
                deallocate_tensor(binding.values)
        finally:
            del wq, wk, wv, wo, wqkv, weights
            gc.collect()
