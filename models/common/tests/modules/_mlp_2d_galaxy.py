# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared WH Galaxy MLP2D geometry, weights, and reference plumbing.

`test_mlp_2d_wh_galaxy.py` is the qualified MLP2D device suite, and its resource
plan, ring core map and DRAM-sharded weight layout are the only combination
known to work on a 6U `(8, 4)` Galaxy. The Prefetcher2D hardware suite drives
the same payload so its numbers are comparable with the MLP evidence, so the
geometry lives here and both suites import it rather than re-deriving it - a
wrong core grid is expensive to debug and impossible to spot by inspection.

This follows the precedent set by `_hf_reference.py`: cross-suite test plumbing
lives in a shared module under `models/common/tests/modules/`.
"""

from __future__ import annotations

import math

import torch
from transformers import AutoModelForCausalLM, LlamaConfig

# transformers 5.x moved no_init_weights to transformers.initialization; fall back
# to the old location for transformers < 5.x.
try:
    from transformers.initialization import no_init_weights
except ImportError:
    from transformers.modeling_utils import no_init_weights

import ttnn
from models.common.models.galaxy import GalaxyCollectivePlan, GalaxyResourceKey, GalaxyResourcesConfig, GalaxyTensorSpec
from models.common.modules.lazy_weight import LazyWeight
from models.common.tests.modules._wh_galaxy_hardware import galaxy_mode_plan, galaxy_prefetch_decode_mode_plan
from models.common.utility_functions import comp_pcc

MLP_PCC_THRESHOLD = 0.99


def reference_mlp(dim: int, hidden_dim: int):
    """HuggingFace MLP the 2D suites qualify MLP2D against - the same reference the
    1D suite uses, instead of a hand-written silu(x@w1) * (x@w3) @ w2.

    The config is built locally rather than downloaded: only the MLP block matters,
    so the vocabulary is shrunk and the rest of the layer is never run. Weights are
    drawn at the scale the Galaxy numeric budget expects (bfloat8_b activations and,
    for the wider model, bfloat8_b weights).
    """
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=dim,
        intermediate_size=hidden_dim,
        num_hidden_layers=1,
        num_attention_heads=dim // 128,
        num_key_value_heads=dim // 128,
        max_position_embeddings=2048,
    )
    with no_init_weights():
        hf_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    reference = hf_model.model.layers[0].mlp
    with torch.no_grad():
        reference.gate_proj.weight.copy_(torch.randn(hidden_dim, dim, dtype=torch.bfloat16) / math.sqrt(dim))
        reference.up_proj.weight.copy_(torch.randn(hidden_dim, dim, dtype=torch.bfloat16) / math.sqrt(dim))
        reference.down_proj.weight.copy_(torch.randn(dim, hidden_dim, dtype=torch.bfloat16) / math.sqrt(hidden_dim))
    return reference


def lazy_activation(source: torch.Tensor, mesh_device: ttnn.MeshDevice, dtype=ttnn.bfloat8_b) -> LazyWeight:
    return LazyWeight(source=source, device=mesh_device, dtype=dtype)


def mlp_pcc(expected: torch.Tensor, actual: torch.Tensor) -> tuple[bool, float]:
    """Correlate an MLP2D output against the HF reference at the suite threshold."""

    return comp_pcc(expected.float(), actual.float(), MLP_PCC_THRESHOLD)


def assert_mlp_pcc(expected: torch.Tensor, actual: torch.Tensor, *, case: str) -> float:
    passing, value = mlp_pcc(expected, actual)
    assert passing, f"{case} failed PCC>={MLP_PCC_THRESHOLD}: {value}"
    return value


def tensor_spec(
    shape: tuple[int, ...],
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=None,
    dtype=ttnn.bfloat8_b,
) -> GalaxyTensorSpec:
    return GalaxyTensorSpec(shape, dtype, ttnn.TILE_LAYOUT, memory_config, mesh_mapper)


def collective_plan(
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
            tensor_spec(
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
            tensor_spec(persistent_output_shape or output_shape, output_memory_config, output_mesh_mapper, dtype),
        ),
        intermediate_output_specs=intermediate,
    )


def resources_config(
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
            collective_plan(
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
            collective_plan(
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
                collective_plan(
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
                collective_plan(
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
            collective_plan(
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


def weight_lazies(w1, w2, w3, mesh_device, dtype):
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


def prefill_weight_lazies(w1, w2, w3, mesh_device, dtype):
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


def decode_ring_config(dim, hidden_dim):
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


def decode_all_reduce_configs(dim):
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


def decode_reduce_scatter_memcfg():
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
