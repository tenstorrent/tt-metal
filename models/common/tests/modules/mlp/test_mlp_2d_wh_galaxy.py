# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Real-hardware correctness tests for the common WH Galaxy MLP2D."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.common.models.galaxy import GalaxyCollectivePlan, GalaxyResourceKey, GalaxyResourcesConfig, GalaxyTensorSpec
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.mlp.mlp_2d import MLP2D, MLP2DConfig, _load_input_device_tensor
from models.common.tests.modules._wh_galaxy_hardware import (
    compose_2d_sharded_tensor,
    deallocate_module_weights,
    deallocate_tensor,
    exact_tensor_resource,
    galaxy_mode_plan,
    galaxy_prefetch_decode_mode_plan,
    require_galaxy_hardware_resources,
)
from models.common.utility_functions import comp_pcc


def _torch_mlp(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
    gate = F.silu(torch.matmul(x, w1))
    up = torch.matmul(x, w3)
    return torch.matmul(gate * up, w2)


def _lazy(source: torch.Tensor, mesh_device: ttnn.MeshDevice, dtype=ttnn.bfloat8_b) -> LazyWeight:
    return LazyWeight(source=source, device=mesh_device, dtype=dtype)


def _assert_pcc(expected: torch.Tensor, actual: torch.Tensor, *, case: str) -> None:
    passing, message = comp_pcc(expected.float(), actual.float(), 0.99)
    assert passing, f"{case} failed PCC>=0.99: {message}"


def _tensor_spec(
    shape: tuple[int, ...],
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=None,
    dtype=ttnn.bfloat8_b,
) -> GalaxyTensorSpec:
    return GalaxyTensorSpec(shape, dtype, ttnn.TILE_LAYOUT, memory_config, mesh_mapper)


def _collective(
    operation: str,
    input_shape: tuple[int, ...],
    *,
    cluster_axis: int | None = None,
    cluster_size: int | None = None,
    sequence_key: object | None = None,
    output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    persistent_output_shape: tuple[int, ...] | None = None,
    output_mesh_mapper=None,
    intermediate_output_shape: tuple[int, ...] | None = None,
    intermediate_output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    intermediate_output_mesh_mapper=None,
    topology=None,
    num_links: int | None = None,
    dtype=ttnn.bfloat8_b,
) -> GalaxyCollectivePlan:
    cluster_axis = (0 if operation == "all_reduce" else 1) if cluster_axis is None else cluster_axis
    cluster_size = {0: 8, 1: 4}[cluster_axis] if cluster_size is None else cluster_size
    width_scale = {
        "reduce_scatter": 1 / cluster_size,
        "all_gather": cluster_size,
        "all_reduce": 1,
    }[operation]
    output_shape = (*input_shape[:-1], int(input_shape[-1] * width_scale))
    intermediate = (
        (
            _tensor_spec(
                intermediate_output_shape or input_shape,
                intermediate_output_memory_config,
                intermediate_output_mesh_mapper,
                dtype,
            ),
        )
        if operation == "reduce_scatter"
        else ()
    )
    return GalaxyCollectivePlan(
        key=GalaxyResourceKey(
            operation,
            cluster_axis,
            input_shape,
            math.prod(input_shape[:-1]) if sequence_key is None else sequence_key,
        ),
        topology=topology or (ttnn.Topology.Ring if operation == "all_reduce" else ttnn.Topology.Linear),
        num_links=num_links or {"reduce_scatter": 1, "all_gather": 4, "all_reduce": 4}[operation],
        semaphores_per_slot={"reduce_scatter": 3, "all_gather": 1, "all_reduce": 1}[operation],
        persistent_output_specs=(
            _tensor_spec(persistent_output_shape or output_shape, output_memory_config, output_mesh_mapper, dtype),
        ),
        intermediate_output_specs=intermediate,
    )


def _resources_config(
    mesh_device: ttnn.MeshDevice,
    dim: int,
    hidden_dim: int,
    *,
    decode_w2_input_memcfg,
    decode_reduce_scatter_memcfg,
    decode_all_reduce_buffer_memcfg,
) -> GalaxyResourcesConfig:
    packet_cores = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(3, 2)),
            ttnn.CoreRange(ttnn.CoreCoord(1, 3), ttnn.CoreCoord(2, 3)),
        ]
    )
    decode_intermediate_memcfg = ttnn.create_sharded_memory_config(
        shape=(32, 512),
        core_grid=packet_cores,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    def plans(sequence: int, *, decode: bool = False) -> tuple[GalaxyCollectivePlan, ...]:
        collective_dtype = ttnn.bfloat8_b
        leading = (1, sequence // 1024, 1024) if sequence >= 1024 else (1, 1, sequence)
        local_hidden = hidden_dim // 8
        padded_local_hidden = math.ceil(local_hidden / (32 * 24)) * 32 * 24
        input_shard_width = padded_local_hidden // 24
        decode_reduced_width = (padded_local_hidden if local_hidden % input_shard_width else local_hidden) // 4
        axis1_reduce_scatters = tuple(
            _collective(
                "reduce_scatter",
                (*leading, hidden_dim // 8),
                sequence_key=(sequence, stage) if stage is not None else None,
                output_memory_config=decode_reduce_scatter_memcfg if decode else ttnn.DRAM_MEMORY_CONFIG,
                output_mesh_mapper=None if decode else ttnn.ReplicateTensorToMesh(mesh_device),
                intermediate_output_shape=(8, 4, sequence, 512 * 8) if decode else None,
                intermediate_output_memory_config=(decode_intermediate_memcfg if decode else ttnn.DRAM_MEMORY_CONFIG),
                intermediate_output_mesh_mapper=(
                    ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(8, 4))
                    if decode
                    else ttnn.ReplicateTensorToMesh(mesh_device)
                ),
                topology=ttnn.Topology.Ring,
                num_links=4,
                dtype=collective_dtype,
            )
            for stage in ((None,) if decode else ("w1", "w3"))
        )
        axis1_collectives = axis1_reduce_scatters + (
            _collective(
                "all_gather",
                (*leading, decode_reduced_width if decode else hidden_dim // 32),
                sequence_key=None if decode else (sequence, "gated"),
                output_memory_config=decode_w2_input_memcfg if decode else ttnn.DRAM_MEMORY_CONFIG,
                persistent_output_shape=((*leading, local_hidden) if decode else None),
                output_mesh_mapper=None if decode else ttnn.ReplicateTensorToMesh(mesh_device),
                topology=ttnn.Topology.Ring,
                dtype=collective_dtype,
            ),
        )
        if not decode:
            return axis1_collectives + (
                _collective(
                    "reduce_scatter",
                    (1, 1, sequence, dim // 4),
                    cluster_axis=0,
                    sequence_key=(sequence, "final"),
                    output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    output_mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                    intermediate_output_mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                    topology=ttnn.Topology.Ring,
                    num_links=4,
                    dtype=collective_dtype,
                ),
                _collective(
                    "all_gather",
                    (1, 1, sequence, dim // 32),
                    cluster_axis=0,
                    sequence_key=(sequence, "final"),
                    output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    output_mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                    topology=ttnn.Topology.Ring,
                    dtype=collective_dtype,
                ),
            )
        return axis1_collectives + (
            _collective(
                "all_reduce",
                (1, 1, sequence, dim // 4),
                output_memory_config=(decode_all_reduce_buffer_memcfg if decode else ttnn.DRAM_MEMORY_CONFIG),
                persistent_output_shape=(8, 4, sequence, 50 * 1024) if decode else None,
                output_mesh_mapper=(
                    ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(8, 4)) if decode else None
                ),
                dtype=collective_dtype,
            ),
        )

    prefill = tuple(plan for sequence in (128, 2048) for plan in plans(sequence))
    return GalaxyResourcesConfig(
        architecture=ttnn.device.Arch.WORMHOLE_B0,
        prefill=galaxy_mode_plan("prefill", prefill, mesh_device),
        decode=galaxy_prefetch_decode_mode_plan(plans(32, decode=True)),
    )


def _weight_lazies(w1, w2, w3, mesh_device, dtype):
    mesh_shape = ttnn.MeshShape(8, 4)
    dram_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(mesh_device.dram_grid_size().x - 1, mesh_device.dram_grid_size().y - 1),
            )
        }
    )

    def dram_sharded(local_k, local_n):
        padded_n = math.ceil(local_n / (32 * 24)) * 32 * 24
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_grid, (local_k, padded_n // 12), ttnn.ShardOrientation.ROW_MAJOR),
        )

    w1_w3_memcfg = dram_sharded(w1.shape[-2] // 4, w1.shape[-1] // 8)
    w2_memcfg = dram_sharded(w2.shape[-2] // 8, w2.shape[-1] // 4)
    w1_w3_mapper = ttnn.MeshMapperConfig(
        placements=[ttnn.PlacementShard(-1), ttnn.PlacementShard(-2)], mesh_shape_override=mesh_shape
    )
    w2_mapper = ttnn.MeshMapperConfig(
        placements=[ttnn.PlacementShard(-2), ttnn.PlacementShard(-1)], mesh_shape_override=mesh_shape
    )

    def make(source, mapper, memory_config):
        return LazyWeight(
            source=source,
            device=mesh_device,
            mesh_mapper_config=mapper,
            memory_config=memory_config,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
        )

    return (
        make(w1, w1_w3_mapper, w1_w3_memcfg),
        make(w2, w2_mapper, w2_memcfg),
        make(w3, w1_w3_mapper, w1_w3_memcfg),
    )


def _prefill_weight_lazies(w1, w2, w3, mesh_device, dtype):
    mesh_shape = ttnn.MeshShape(8, 4)
    w1_w3_mapper = ttnn.MeshMapperConfig(
        placements=[ttnn.PlacementShard(-1), ttnn.PlacementShard(-2)], mesh_shape_override=mesh_shape
    )
    w2_mapper = ttnn.MeshMapperConfig(
        placements=[ttnn.PlacementShard(-2), ttnn.PlacementShard(-1)], mesh_shape_override=mesh_shape
    )

    def make(source, mapper):
        return LazyWeight(
            source=source,
            device=mesh_device,
            mesh_mapper_config=mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
        )

    return make(w1, w1_w3_mapper), make(w2, w2_mapper), make(w3, w1_w3_mapper)


def _decode_ring_config(dim, hidden_dim):
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

    def points(coords):
        return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*coord), ttnn.CoreCoord(*coord)) for coord in coords])

    ring_cores = points(ring_coords)
    receiver_cores = points(receiver_coords)
    padded_dim = math.ceil((dim // 4) / (32 * 24)) * 32 * 24
    padded_hidden = math.ceil((hidden_dim // 8) / (32 * 24)) * 32 * 24

    def memory_config(width, cores):
        return ttnn.create_sharded_memory_config(
            shape=(32, width // 24),
            core_grid=cores,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def program_config(k, n):
        out_block_w = n // 24 // 32
        out_subblock_w = 8
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

    return {
        "decode_input_memcfg": memory_config(padded_dim, ring_cores),
        "decode_w2_input_memcfg": memory_config(padded_hidden, ring_cores),
        "decode_w1_w3_output_memcfg": memory_config(padded_hidden, receiver_cores),
        "decode_w2_output_memcfg": memory_config(padded_dim, receiver_cores),
        "decode_w1_w3_prg_config": program_config(dim // 4, padded_hidden),
        "decode_w2_prg_config": program_config(hidden_dim // 8, padded_dim),
    }


def _decode_all_reduce_configs(dim):
    residual_core_count = (dim // 4) // 128
    assert residual_core_count in (10, 16)
    residual_cores = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(1, 0),
                ttnn.CoreCoord(2, residual_core_count // 2 - 1),
            )
        }
    )
    worker_cores = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )
    output_memcfg = ttnn.create_sharded_memory_config(
        shape=(32, 128),
        core_grid=residual_cores,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    buffer_memcfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(worker_cores, [32, 1024], ttnn.ShardOrientation.ROW_MAJOR),
    )
    return output_memcfg, buffer_memcfg


def _decode_reduce_scatter_memcfg():
    worker_cores = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )
    output_cores = ttnn.num_cores_to_corerangeset_in_subcoregrids(ttnn.CoreCoord(1, 0), 30, worker_cores, row_wise=True)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(output_cores, [32, 32], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _invoke(
    module: MLP2D,
    resources,
    mesh_device: ttnn.MeshDevice,
    x: torch.Tensor,
    *,
    mode: str,
    expected: torch.Tensor,
    case: str,
) -> None:
    input_dtype = module.config.decode_activation_dtype if mode == "decode" else module.config.prefill_activation_dtype
    device_input = _load_input_device_tensor(_lazy(x, mesh_device, input_dtype), module.config, mode)
    resources.activate(mode)
    output = module(device_input, mode=mode)
    try:
        resources.synchronize(mode)
        actual = compose_2d_sharded_tensor(output, mesh_device)
        _assert_pcc(expected, actual, case=case)
    finally:
        deallocate_tensor(output)
        deallocate_tensor(device_input)


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "dim,hidden_dim",
    [(8192, 28672), (5120, 25600)],
    ids=["llama-8192x28672", "qwen-5120x25600"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@torch.no_grad()
def test_mlp_2d_wh_galaxy_decode_batch_32_repeat(mesh_device, dim, hidden_dim):
    torch.manual_seed(0)
    scale = 1.0 / math.sqrt(dim)
    w1 = torch.randn(dim, hidden_dim, dtype=torch.bfloat16) * scale
    w2 = torch.randn(hidden_dim, dim, dtype=torch.bfloat16) / math.sqrt(hidden_dim)
    w3 = torch.randn(dim, hidden_dim, dtype=torch.bfloat16) * scale
    x = torch.randn(1, 1, 32, dim, dtype=torch.bfloat16)
    expected = _torch_mlp(x, w1, w2, w3)
    weight_dtype = ttnn.bfloat16 if dim == 5120 else ttnn.bfloat8_b
    activation_dtype = ttnn.bfloat8_b
    lazy_w1, lazy_w2, lazy_w3 = _weight_lazies(w1, w2, w3, mesh_device, weight_dtype)
    decode_ring = _decode_ring_config(dim, hidden_dim)
    reduce_scatter_memcfg = _decode_reduce_scatter_memcfg()
    all_reduce_output_memcfg, all_reduce_buffer_memcfg = _decode_all_reduce_configs(dim)
    resources = require_galaxy_hardware_resources(
        mesh_device,
        config=_resources_config(
            mesh_device,
            dim,
            hidden_dim,
            decode_w2_input_memcfg=decode_ring["decode_w2_input_memcfg"],
            decode_reduce_scatter_memcfg=reduce_scatter_memcfg,
            decode_all_reduce_buffer_memcfg=all_reduce_buffer_memcfg,
        ),
        prefetch_weights=(
            ("mlp.w1", lazy_w1.get_device_weight()),
            ("mlp.w3", lazy_w3.get_device_weight()),
            ("mlp.w2", lazy_w2.get_device_weight()),
        ),
    )
    module = None
    try:
        module = MLP2D.from_config(
            MLP2DConfig(
                w1=lazy_w1,
                w2=lazy_w2,
                w3=lazy_w3,
                mesh_device=mesh_device,
                tt_ccl=resources.ccl,
                collective_resource_selector=exact_tensor_resource,
                w1_w3_memcfg=lazy_w1.memory_config,
                w2_memcfg=lazy_w2.memory_config,
                **decode_ring,
                ff1_out_reduce_scatter_memcfg=reduce_scatter_memcfg,
                ff2_out_reduce_scatter_memcfg=all_reduce_output_memcfg,
                sharded_attn_input_memcfg=ttnn.DRAM_MEMORY_CONFIG,
                decode_prefetch_context=resources.prefetch_context("decode"),
                prefill_prefetch_context=resources.prefetch_context("prefill"),
                activation_dtype=activation_dtype,
                ccl_dtype=activation_dtype,
                mul_dtype=activation_dtype,
            )
        )
        for invocation in range(2):
            _invoke(
                module,
                resources,
                mesh_device,
                x,
                mode="decode",
                expected=expected,
                case=f"decode invocation {invocation}",
            )
    finally:
        try:
            resources.cleanup()
        finally:
            deallocate_module_weights(module, "w1", "w2", "w3")


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "dim,hidden_dim",
    [(8192, 28672), (5120, 25600)],
    ids=["llama-8192x28672", "qwen-5120x25600"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@torch.no_grad()
def test_mlp_2d_wh_galaxy_prefill_128_then_2048_repeat(mesh_device, dim, hidden_dim):
    torch.manual_seed(1)
    scale = 1.0 / math.sqrt(dim)
    w1 = torch.randn(dim, hidden_dim, dtype=torch.bfloat16) * scale
    w2 = torch.randn(hidden_dim, dim, dtype=torch.bfloat16) / math.sqrt(hidden_dim)
    w3 = torch.randn(dim, hidden_dim, dtype=torch.bfloat16) * scale
    weight_dtype = ttnn.bfloat16 if dim == 5120 else ttnn.bfloat8_b
    activation_dtype = ttnn.bfloat8_b
    lazy_w1, lazy_w2, lazy_w3 = _weight_lazies(w1, w2, w3, mesh_device, weight_dtype)
    prefill_w1, prefill_w2, prefill_w3 = _prefill_weight_lazies(w1, w2, w3, mesh_device, weight_dtype)
    decode_ring = _decode_ring_config(dim, hidden_dim)
    reduce_scatter_memcfg = _decode_reduce_scatter_memcfg()
    all_reduce_output_memcfg, all_reduce_buffer_memcfg = _decode_all_reduce_configs(dim)
    resources = require_galaxy_hardware_resources(
        mesh_device,
        config=_resources_config(
            mesh_device,
            dim,
            hidden_dim,
            decode_w2_input_memcfg=decode_ring["decode_w2_input_memcfg"],
            decode_reduce_scatter_memcfg=reduce_scatter_memcfg,
            decode_all_reduce_buffer_memcfg=all_reduce_buffer_memcfg,
        ),
        prefetch_weights=(
            ("mlp.w1", lazy_w1.get_device_weight()),
            ("mlp.w3", lazy_w3.get_device_weight()),
            ("mlp.w2", lazy_w2.get_device_weight()),
        ),
    )
    module = None
    try:
        module = MLP2D.from_config(
            MLP2DConfig(
                w1=lazy_w1,
                w2=lazy_w2,
                w3=lazy_w3,
                prefill_w1=prefill_w1,
                prefill_w2=prefill_w2,
                prefill_w3=prefill_w3,
                mesh_device=mesh_device,
                tt_ccl=resources.ccl,
                collective_resource_selector=exact_tensor_resource,
                w1_w3_memcfg=lazy_w1.memory_config,
                w2_memcfg=lazy_w2.memory_config,
                ff1_out_reduce_scatter_memcfg=ttnn.DRAM_MEMORY_CONFIG,
                ff2_out_reduce_scatter_memcfg=ttnn.DRAM_MEMORY_CONFIG,
                sharded_attn_input_memcfg=ttnn.DRAM_MEMORY_CONFIG,
                decode_prefetch_context=resources.prefetch_context("decode"),
                prefill_prefetch_context=resources.prefetch_context("prefill"),
                activation_dtype=activation_dtype,
                ccl_dtype=activation_dtype,
                mul_dtype=activation_dtype,
            )
        )
        for invocation in range(2):
            for seq_len in (128, 2048):
                x = torch.randn(1, 1, seq_len, dim, dtype=torch.bfloat16)
                _invoke(
                    module,
                    resources,
                    mesh_device,
                    x,
                    mode="prefill",
                    expected=_torch_mlp(x, w1, w2, w3),
                    case=f"prefill {seq_len} invocation {invocation}",
                )
    finally:
        try:
            resources.cleanup()
        finally:
            deallocate_module_weights(module, "w1", "w2", "w3", "prefill_w1", "prefill_w2", "prefill_w3")
