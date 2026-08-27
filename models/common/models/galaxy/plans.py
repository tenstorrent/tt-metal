# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Production Galaxy collective-resource plans for one dense 2D transformer.

The plans are the union of the collectives that the hardware-qualified
Milestone A recipes issue: the MLP2D axis-1 reduce-scatter/all-gather pair and
final axis-0 reduction, the Attention2D fused create-QKV-heads collective, its
decode user gather and output reduction, and the RMSNorm2D distributed
statistics gather. Keys are derived from the exact tensor geometry each TTNN
operation observes so that :func:`select_galaxy_resource` can never pick a
resource allocated for a different shape.

Every formula here mirrors the qualified module test recipes. Where a formula
is only proven for one model geometry the comment says so; those are the
Milestone B hardware confirmation points.
"""

from __future__ import annotations

import math
from typing import Any

import ttnn
from models.common.models.galaxy.ccl import GalaxyMode
from models.common.models.galaxy.recipes import (
    GALAXY_COLUMNS,
    GALAXY_MESH_SHAPE,
    GALAXY_PHYSICAL_BATCH,
    GALAXY_ROWS,
    TILE,
    GalaxyDecodePlacements,
    GalaxyDenseGeometry,
    core_ranges,
    distributed_norm_stats_memory_config,
    galaxy_prefill_mode_plan_cores,
    pad_ring_width,
    prefetch_sender_cores,
    validate_galaxy_mesh,
    worker_cores,
)
from models.common.models.galaxy.resources import (
    GalaxyCollectivePlan,
    GalaxyModePlan,
    GalaxyResourceKey,
    GalaxyResourcesConfig,
    GalaxyTensorSpec,
)

_ALL_REDUCE_BUFFER_WIDTH = 50 * 1024
_DECODE_PACKET_SHARD_WIDTH = 512
_DECODE_PACKET_SHARDS = 8


def _spec(
    shape: tuple[int, ...],
    memory_config: Any,
    *,
    dtype: Any = ttnn.bfloat8_b,
    mesh_mapper: Any = None,
) -> GalaxyTensorSpec:
    return GalaxyTensorSpec(shape, dtype, ttnn.TILE_LAYOUT, memory_config, mesh_mapper)


def _sequence_key(shape: tuple[int, ...], stage: Any = None) -> Any:
    """Return the key `select_galaxy_resource` derives from a tensor."""

    leading = math.prod(shape[:-1])
    return leading if stage is None else (leading, stage)


def _decode_packet_memory_config() -> ttnn.MemoryConfig:
    """Return the qualified eight-core packet buffer for decode reduce-scatter."""

    packet_cores = core_ranges((1, 1, 3, 2), (1, 3, 2, 3))
    return ttnn.create_sharded_memory_config(
        shape=(TILE, _DECODE_PACKET_SHARD_WIDTH),
        core_grid=packet_cores,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def build_galaxy_decode_collectives(
    mesh_device: Any,
    geometry: GalaxyDenseGeometry,
    placements: GalaxyDecodePlacements,
    *,
    residual_dtype: Any = ttnn.bfloat16,
    # The LM head all-reduce buffer follows the *logits*, which both Galaxy
    # precision recipes store at bfloat8_b -- the production dtype, and the only
    # one whose buffer fits beside the ring matmul's circular buffers.
    lm_head_dtype: Any = ttnn.bfloat8_b,
) -> tuple[GalaxyCollectivePlan, ...]:
    """Return the decode collectives for attention, MLP, and distributed norm.

    ``residual_dtype`` sizes the shared axis-0 all-reduce buffer. It must match
    the dtype the *consumers* hand ``all_reduce_async``, because that op sizes
    its circular buffer from the data and checks it against the buffer's L1 bank:
    a bfloat16 reduction against a bfloat8_b buffer fails with

        TT_FATAL ... Cannot set circular buffer size to 65536. This is larger
                     than the associated dynamically allocated L1 buffer bank
                     size of 34816 B

    Both Galaxy models set ``MLP2D``'s ``decode_ccl_dtype`` to their
    ``decode_residual_dtype``, which is bfloat16 in both precision recipes -
    deliberately, so that an 80-layer running residual sum is never re-quantized
    - so bfloat16 is the default here rather than ``_spec``'s bfloat8_b. It is a
    parameter and not a literal so a model with a different residual dtype can
    say so instead of silently mismatching.
    """

    row_shard = ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=GALAXY_MESH_SHAPE)
    batch = GALAXY_PHYSICAL_BATCH
    qkv_shape = (1, 1, batch, geometry.local_qkv_size)
    users_shape = (1, geometry.users_per_column, geometry.local_heads, geometry.head_dim)
    hidden_shape = (1, 1, batch, geometry.local_hidden_dim)
    scattered_shape = (1, 1, batch, geometry.decode_reduce_scatter_width)
    output_shape = (1, 1, batch, geometry.local_dim)
    stats_shape = (1, 1, batch, TILE)
    # The key carries the width TTNN *reports* for the logits, which is the
    # logical `local_padded_vocab_size`, not the ring-padded physical width. The
    # ring matmul's output shard spec over-covers -- 24 cores x 672 = 16128 for a
    # 16032-wide tensor -- and `select_galaxy_resource` keys on `tensor.shape`.
    #
    # Measured, not assumed. `GalaxyDenseGeometry.decode_reduce_scatter_width`
    # records the opposite rule for the MLP's reduce-scatter output ("the
    # collective scatters the padded width"), and applying it here gave
    #     KeyError: no all_reduce resources for axis=1,
    #               geometry=(1, 1, 32, 16032), sequence=32
    # so a *matmul* output keeps its logical width where a *reduce-scatter*
    # output does not. Do not generalise one to the other.
    geometry_padded_local_vocab = pad_ring_width(geometry.local_padded_vocab_size)
    logits_shape = (1, 1, batch, geometry.local_padded_vocab_size)

    fused_qkv = GalaxyCollectivePlan(
        key=GalaxyResourceKey("all_reduce_create_qkv_heads", 1, qkv_shape, _sequence_key(qkv_shape)),
        topology=ttnn.Topology.Ring,
        num_links=3,
        persistent_output_specs=(
            _spec(
                (*GALAXY_MESH_SHAPE, TILE, geometry.local_qkv_size * GALAXY_COLUMNS),
                placements.attention_qkv_scratch_memcfg,
                mesh_mapper=row_shard,
            ),
        ),
    )
    gather_users = GalaxyCollectivePlan(
        key=GalaxyResourceKey("all_gather", 1, users_shape, _sequence_key(users_shape)),
        topology=ttnn.Topology.Ring,
        num_links=1,
        persistent_output_specs=(
            _spec(
                (1, batch, geometry.local_heads, geometry.head_dim),
                placements.attention_gather_users_memcfg,
                dtype=ttnn.bfloat16,
            ),
        ),
    )
    mlp_reduce_scatter = GalaxyCollectivePlan(
        key=GalaxyResourceKey("reduce_scatter", 1, hidden_shape, _sequence_key(hidden_shape)),
        topology=ttnn.Topology.Ring,
        num_links=4,
        semaphores_per_slot=3,
        persistent_output_specs=(_spec(scattered_shape, placements.mlp_reduce_scatter_memcfg),),
        intermediate_output_specs=(
            _spec(
                (*GALAXY_MESH_SHAPE, TILE, _DECODE_PACKET_SHARD_WIDTH * _DECODE_PACKET_SHARDS),
                _decode_packet_memory_config(),
                mesh_mapper=row_shard,
            ),
        ),
    )
    mlp_all_gather = GalaxyCollectivePlan(
        key=GalaxyResourceKey("all_gather", 1, scattered_shape, _sequence_key(scattered_shape)),
        topology=ttnn.Topology.Ring,
        num_links=4,
        persistent_output_specs=(_spec(hidden_shape, placements.mlp_w2_input_memcfg),),
    )
    # Attention and MLP finish decode with the same axis-0 hidden reduction, so
    # they share one keyed resource and one persistent buffer.
    output_all_reduce = GalaxyCollectivePlan(
        key=GalaxyResourceKey("all_reduce", 0, output_shape, _sequence_key(output_shape)),
        topology=ttnn.Topology.Ring,
        num_links=4,
        persistent_output_specs=(
            _spec(
                (*GALAXY_MESH_SHAPE, TILE, _ALL_REDUCE_BUFFER_WIDTH),
                placements.all_reduce_buffer_memcfg,
                dtype=residual_dtype,
                mesh_mapper=row_shard,
            ),
        ),
    )
    # The decode LM head reduces the hidden dimension over the four mesh columns.
    # It needs its own keyed resource and its own persistent buffer, and it cannot
    # borrow the axis-0 one: that buffer is sized for the `local_dim`-wide residual
    # stream, and `all_reduce_async` validates
    #     buffer_shard_volume >= output_shard_volume * ring_size
    # against the *logits*, which are `padded_local_vocab` wide.
    #
    # Without a persistent buffer, `ttnn.all_reduce` falls back to
    # `composite_common::composite_all_gather`, whose `ttnn::concat` is handed no
    # `sub_core_grids` and builds over the full compute grid:
    #     TT_FATAL ... Kernel group cores do not match sub device cores
    #                  for programmable core type TENSIX
    # from `ttnn::prim::concat`. The persistent-buffer overload takes the fused
    # path instead, which honours `subdevice_id`. This mirrors the production
    # `tt_ccl.line_all_reduce(..., lm_head=True, buffer_key="LM_HEAD")`, which
    # keeps a dedicated LM-head buffer for exactly this reason.
    lm_head_all_reduce = GalaxyCollectivePlan(
        key=GalaxyResourceKey("all_reduce", 1, logits_shape, _sequence_key(logits_shape)),
        topology=ttnn.Topology.Ring,
        num_links=4,
        persistent_output_specs=(
            _spec(
                (*GALAXY_MESH_SHAPE, TILE, geometry_padded_local_vocab * GALAXY_COLUMNS),
                # DRAM, deliberately. This buffer is four times the width of the
                # logits, which at bfloat16 is about 129 kB per core -- far too
                # much to keep resident in L1 for the whole decode step, where it
                # competes with 80 layers' worth of activations:
                #     TT_THROW ... Statically allocated circular buffers in
                #     program 250 clash with L1 buffers on core range [1-0 - 3-9]
                # The collective brings it into L1 for the duration of the
                # reduction and frees the L1 copy immediately after, which is what
                # the production code does: `tt_lm_head_buffer` is created with
                # `ttnn.DRAM_MEMORY_CONFIG`, `llama_model.py` materialises
                # `tt_lm_head_buffer_l1` just before the LM head, and
                # `line_all_reduce` ends with `persistent_buffer.deallocate(True)`.
                ttnn.DRAM_MEMORY_CONFIG,
                dtype=lm_head_dtype,
                mesh_mapper=row_shard,
            ),
        ),
    )
    norm_stats = GalaxyCollectivePlan(
        key=GalaxyResourceKey("all_gather", 1, stats_shape, _sequence_key(stats_shape)),
        topology=ttnn.Topology.Ring,
        num_links=1,
        semaphores_per_slot=1,
        persistent_output_specs=(
            _spec(
                (1, 1, batch, TILE * GALAXY_COLUMNS),
                distributed_norm_stats_memory_config(placements.residual_memcfg),
                dtype=ttnn.bfloat16,
            ),
        ),
    )
    return (
        fused_qkv,
        gather_users,
        mlp_reduce_scatter,
        mlp_all_gather,
        output_all_reduce,
        lm_head_all_reduce,
        norm_stats,
    )


def build_galaxy_prefill_collectives(
    mesh_device: Any,
    geometry: GalaxyDenseGeometry,
    sequence_length: int,
) -> tuple[GalaxyCollectivePlan, ...]:
    """Return every prefill collective for one padded sequence length."""

    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    leading = geometry.prefill_leading_shape(sequence_length)
    token_leading = (1, 1, sequence_length)
    qkv_shape = (*token_leading, geometry.local_qkv_size)
    output_shape = (*token_leading, geometry.local_dim)
    scattered_output_shape = (*token_leading, geometry.local_dim // GALAXY_ROWS)
    hidden_shape = (*leading, geometry.local_hidden_dim)
    gated_shape = (*leading, geometry.local_hidden_dim // GALAXY_COLUMNS)
    stats_shape = (*token_leading, TILE)

    def dram(shape: tuple[int, ...]) -> GalaxyTensorSpec:
        return _spec(shape, ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=replicate)

    attention_qkv = GalaxyCollectivePlan(
        key=GalaxyResourceKey("all_reduce", 1, qkv_shape, _sequence_key(qkv_shape)),
        topology=ttnn.Topology.Linear,
        num_links=1,
        persistent_output_specs=(dram(qkv_shape),),
    )
    attention_output = GalaxyCollectivePlan(
        key=GalaxyResourceKey("all_reduce", 0, output_shape, _sequence_key(output_shape)),
        topology=ttnn.Topology.Ring,
        num_links=4,
        persistent_output_specs=(dram(output_shape),),
    )
    mlp_reduce_scatters = tuple(
        GalaxyCollectivePlan(
            key=GalaxyResourceKey("reduce_scatter", 1, hidden_shape, _sequence_key(hidden_shape, stage)),
            topology=ttnn.Topology.Ring,
            num_links=4,
            semaphores_per_slot=3,
            persistent_output_specs=(dram(gated_shape),),
            intermediate_output_specs=(dram(hidden_shape),),
        )
        for stage in ("w1", "w3")
    )
    mlp_all_gather = GalaxyCollectivePlan(
        key=GalaxyResourceKey("all_gather", 1, gated_shape, _sequence_key(gated_shape, "gated")),
        topology=ttnn.Topology.Ring,
        num_links=4,
        persistent_output_specs=(dram(hidden_shape),),
    )
    mlp_final_reduce_scatter = GalaxyCollectivePlan(
        key=GalaxyResourceKey("reduce_scatter", 0, output_shape, _sequence_key(output_shape, "final")),
        topology=ttnn.Topology.Ring,
        num_links=4,
        semaphores_per_slot=3,
        persistent_output_specs=(dram(scattered_output_shape),),
        intermediate_output_specs=(dram(output_shape),),
    )
    mlp_final_all_gather = GalaxyCollectivePlan(
        key=GalaxyResourceKey("all_gather", 0, scattered_output_shape, _sequence_key(scattered_output_shape, "final")),
        topology=ttnn.Topology.Ring,
        num_links=4,
        persistent_output_specs=(dram(output_shape),),
    )
    norm_stats = GalaxyCollectivePlan(
        key=GalaxyResourceKey("all_gather", 1, stats_shape, _sequence_key(stats_shape)),
        topology=ttnn.Topology.Linear,
        num_links=1,
        semaphores_per_slot=1,
        persistent_output_specs=(
            _spec(
                (*token_leading, TILE * GALAXY_COLUMNS),
                ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                mesh_mapper=replicate,
            ),
        ),
    )
    return (
        attention_qkv,
        attention_output,
        *mlp_reduce_scatters,
        mlp_all_gather,
        mlp_final_reduce_scatter,
        mlp_final_all_gather,
        norm_stats,
    )


def galaxy_prefill_mode_plan(mesh_device: Any, collectives: tuple[GalaxyCollectivePlan, ...]) -> GalaxyModePlan:
    """Return the single-subdevice prefill envelope covering the full grid."""

    cores = galaxy_prefill_mode_plan_cores(mesh_device)
    worker_id = ttnn.SubDeviceId(0)
    return GalaxyModePlan(
        mode="prefill",
        sub_devices=(ttnn.SubDevice([cores]),),
        worker_sub_device_id=worker_id,
        stall_group=(worker_id,),
        semaphore_cores=cores,
        worker_cores=cores,
        collectives=collectives,
    )


def galaxy_decode_mode_plan(collectives: tuple[GalaxyCollectivePlan, ...]) -> GalaxyModePlan:
    """Return the canonical prefetch sender/worker decode subdevice partition."""

    senders = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in prefetch_sender_cores()])
    workers = worker_cores()
    worker_id = ttnn.SubDeviceId(1)
    return GalaxyModePlan(
        mode="decode",
        sub_devices=(ttnn.SubDevice([senders]), ttnn.SubDevice([workers])),
        worker_sub_device_id=worker_id,
        stall_group=(worker_id,),
        semaphore_cores=workers,
        worker_cores=workers,
        collectives=collectives,
    )


def build_galaxy_resources_config(
    mesh_device: Any,
    geometry: GalaxyDenseGeometry,
    placements: GalaxyDecodePlacements,
) -> GalaxyResourcesConfig:
    """Resolve the complete production CCL policy for one Galaxy transformer."""

    validate_galaxy_mesh("Galaxy resources", mesh_device)
    prefill_collectives: list[GalaxyCollectivePlan] = []
    # Concatenated physical-batch-32 prefill issues exactly the collectives of a
    # single-row prefill over its total token count, so one loop covers both.
    for length in geometry.collective_prefill_token_counts:
        prefill_collectives.extend(build_galaxy_prefill_collectives(mesh_device, geometry, length))
    keys = [plan.key for plan in prefill_collectives]
    if len(keys) != len(set(keys)):
        raise ValueError("prefill sequence lengths produced duplicate Galaxy resource keys")
    return GalaxyResourcesConfig(
        architecture=mesh_device.arch(),
        prefill=galaxy_prefill_mode_plan(mesh_device, tuple(prefill_collectives)),
        decode=galaxy_decode_mode_plan(build_galaxy_decode_collectives(mesh_device, geometry, placements)),
    )


def select_galaxy_resource(
    context: Any,
    operation: str,
    cluster_axis: int,
    tensor: Any,
    stage_key: Any = None,
) -> Any:
    """Return the resource keyed by the geometry the TTNN operation observes.

    ``MLP2D`` and ``RMSNorm2D`` accept this as their
    ``collective_resource_selector``; the Attention2D collectives call it
    directly. ``tensor`` may be a TTNN tensor or an explicit shape tuple.
    """

    shape = tuple(int(value) for value in (tensor if isinstance(tensor, tuple) else tensor.shape))
    return context.resources(operation, cluster_axis, shape, _sequence_key(shape, stage_key))


def galaxy_mode_names() -> tuple[GalaxyMode, ...]:
    return ("prefill", "decode")
