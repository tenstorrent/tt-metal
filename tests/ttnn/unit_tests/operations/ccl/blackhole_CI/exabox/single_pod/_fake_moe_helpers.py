# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers for the single-pod fake-MoE tests.

Surfaces:
  - sub-device worker setup/teardown (CCL ops require it).
  - `make_reduce_to_one_b1_inputs` + `step_reduce_to_one_b1` for the
    real `ReduceToOneB1` op the demo's MoE uses.
  - `make_fake_moe_decoder_stage_factory` / `make_fake_lm_head_stage_factory`
    used by the pipeline framework smoke test (`test_single_pod_pipeline_fake_moe`).
"""

from __future__ import annotations

import torch

import ttnn


# ---------------------------------------------------------------------------
# Sub-device worker setup (required by CCL ops)
# ---------------------------------------------------------------------------


def _setup_sub_devices(mesh_device):
    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)
    return worker_sub_device_id, sub_device_stall_group, sub_device_manager


def _teardown_sub_devices(mesh_device, sub_device_manager):
    mesh_device.reset_sub_device_stall_group()
    mesh_device.clear_loaded_sub_device_manager()
    mesh_device.remove_sub_device_manager(sub_device_manager)


# ---------------------------------------------------------------------------
# ReduceToOneB1 input bundle + step
# ---------------------------------------------------------------------------


def make_reduce_to_one_b1_inputs(sender_tensor: torch.Tensor, mesh_device, *, shard_grid, is_torus: bool = True):
    """Build (input, intermediate, output, semaphores) for `ReduceToOneB1.op(...)`.

    `sender_tensor` is `[1, hidden]` and gets replicated to every device of
    the (4, 2) submesh. Width-sharded layout matches the demo's MoE end-of-chain
    state (see test_reduce_to_one_b1.py for the canonical setup).
    """
    assert (
        sender_tensor.ndim == 2 and sender_tensor.shape[0] == 1
    ), f"sender_tensor must be [1, hidden]; got {tuple(sender_tensor.shape)}"
    mesh_rows, mesh_cols = mesh_device.shape[0], mesh_device.shape[1]
    assert mesh_rows == 4 and mesh_cols == 2, f"ReduceToOneB1 requires a 4×2 mesh; got {mesh_rows}×{mesh_cols}"

    hidden = sender_tensor.shape[1]
    shard_cores = ttnn.corerange_to_cores(shard_grid, row_wise=True)
    num_shard_cores = len(shard_cores)
    assert hidden % num_shard_cores == 0, f"hidden ({hidden}) must be divisible by num_shard_cores ({num_shard_cores})"

    shard_shape = [1, hidden // num_shard_cores]
    layout = ttnn.TILE_LAYOUT
    tile = ttnn.Tile((1, 32))
    dtype = ttnn.bfloat16

    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)

    replicated = sender_tensor.unsqueeze(0).unsqueeze(0).expand(mesh_rows, mesh_cols, 1, hidden).contiguous()
    mesh_mapper = ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)], mesh_device.shape),
    )
    input_tensor = ttnn.from_torch(
        replicated,
        device=mesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config,
        mesh_mapper=mesh_mapper,
    )

    # Intermediate: 3× shard width, same layout (backs a 3-page CB inside the kernel).
    intermediate_shard_shape = [1, shard_shape[1] * 3]
    intermediate_shard_spec = ttnn.ShardSpec(shard_grid, intermediate_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    intermediate_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, intermediate_shard_spec
    )
    intermediate_tensor = ttnn.from_torch(
        torch.zeros([mesh_rows, mesh_cols, 1, hidden * 3], dtype=torch.bfloat16),
        device=mesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=intermediate_mem_config,
        mesh_mapper=mesh_mapper,
    )

    # Output: single-core sharded on the aggregator core (first worker core
    # in shard grid), holds the full reduced [1, hidden] result on the root.
    aggregator_core = shard_cores[0]
    output_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(aggregator_core, aggregator_core)})
    output_shard_spec = ttnn.ShardSpec(output_shard_grid, [1, hidden], ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec)
    output_tensor = ttnn.from_torch(
        torch.zeros([mesh_rows, mesh_cols, 1, hidden], dtype=torch.bfloat16),
        device=mesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=output_mem_config,
        mesh_mapper=mesh_mapper,
    )

    # 4 semaphores: round1, round2, round3, exit.
    compute_grid = mesh_device.compute_with_storage_grid_size()
    available_cores = ttnn.num_cores_to_corerangeset(
        compute_grid.x * compute_grid.y,
        compute_grid,
        row_wise=True,
    )
    ttnn.synchronize_device(mesh_device)
    semaphores = [ttnn.create_global_semaphore(mesh_device, available_cores, 0) for _ in range(4)]
    ttnn.synchronize_device(mesh_device)

    return {
        "input_tensor": input_tensor,
        "intermediate_tensor": intermediate_tensor,
        "output_tensor": output_tensor,
        "semaphores": semaphores,
    }


def step_reduce_to_one_b1(
    tt_input,
    intermediate_tensor,
    output_tensor,
    semaphores,
    root_coord,
    exit_coord=None,
    *,
    is_torus: bool = True,
):
    """Run the demo's actual `ReduceToOneB1` op (3-level tree, 8 → 1 device)."""
    from models.demos.deepseek_v3_b1.micro_ops.reduce_to_one_b1.op import ReduceToOneB1

    if isinstance(root_coord, tuple):
        root_coord = ttnn.MeshCoordinate(root_coord[0], root_coord[1])
    if exit_coord is None:
        exit_coord = root_coord
    elif isinstance(exit_coord, tuple):
        exit_coord = ttnn.MeshCoordinate(exit_coord[0], exit_coord[1])

    return ReduceToOneB1.op(
        tt_input,
        intermediate_tensor,
        output_tensor,
        semaphores,
        root_coord,
        exit_coord,
        is_torus=is_torus,
    )


# ---------------------------------------------------------------------------
# Fake stages for the pipeline framework smoke test
# ---------------------------------------------------------------------------


def make_synthetic_embedding_weights(mesh_device, *, vocab_size: int = 129280, hidden_size: int = 7168):
    """Minimal stand-in for the demo's `SyntheticWeightProvider.load_embedding`.

    Returns a `DeepSeekV3EmbeddingLayerWeights` whose `.embedding` is a zero
    tensor uploaded as DRAM-interleaved + replicated across the mesh. The
    test never drives any tokens through the pipeline, so the actual content
    is irrelevant — only the shape and on-device residency matter.
    """
    from models.demos.deepseek_v3_b1.demo.stage import DeepSeekV3EmbeddingLayerWeights

    emb = ttnn.from_torch(
        torch.zeros((vocab_size, hidden_size), dtype=torch.bfloat16),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return DeepSeekV3EmbeddingLayerWeights(embedding=emb)


def make_fake_moe_decoder_stage_factory():
    """Stage factory producing a `PassthroughStage(ACTIVATION)` — used as
    a no-compute drop-in for `MoEDecoderStage` in the pipeline test."""
    from models.demos.deepseek_v3_b1.demo.stage import PassthroughPayload, PassthroughStage

    return lambda mesh_device: PassthroughStage(PassthroughPayload.ACTIVATION)


def make_fake_lm_head_stage_factory():
    """Stage factory producing a no-compute LMHead stub.

    Same upstream/downstream socket sizes as `LMHeadStage` (ACTIVATION in →
    TOKEN out, so adjacent stages 13 → 14 → 15 still match their FIFO
    contracts) but empty `setup` and empty `launch_compute`. Avoids the
    LMHead persistent kernel that strands rank 14 in teardown when no
    token is ever driven through the pipeline.
    """
    from models.demos.deepseek_v3_b1.demo.stage import (
        ACTIVATION_FIFO_SIZE,
        ACTIVATION_PAGE_SIZE_BYTES,
        PIPELINE_CORE_COORD,
        TOKEN_FIFO_SIZE,
        TOKEN_PAGE_SIZE_BYTES,
        StageKind,
    )
    from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import PipelineBlock

    class _FakeLMHeadStage(StageKind):
        def create_pipeline_block(self, ctx):
            return PipelineBlock(
                ctx.mesh_device,
                PIPELINE_CORE_COORD,
                upstream_d2d_socket_fifo_size=ACTIVATION_FIFO_SIZE,
                downstream_d2d_socket_fifo_size=TOKEN_FIFO_SIZE,
                upstream_d2d_socket_page_size=ACTIVATION_PAGE_SIZE_BYTES,
                downstream_d2d_socket_page_size=TOKEN_PAGE_SIZE_BYTES,
            )

    return lambda mesh_device: _FakeLMHeadStage()
