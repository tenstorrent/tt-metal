# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import nearest_y
from models.demos.deepseek_v3.tests.fused_op_unit_tests.mla.test_rope_deepseek import (
    apply_rotary_pos_emb_torch,
    create_rope_tensors,
)
from models.demos.deepseek_v3.utils.config_dataclass import SliceConfig
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW
from models.perf.benchmarking_utils import BenchmarkProfiler
from tests.ttnn.utils_for_testing import assert_with_pcc


def run_fwd_decode_q_rope_nope_with_trace(
    mesh_device,
    input_tensor_mesh,
    q_weight_tensor,
    q_nope_weight_tensor,
    rope_tensors,
    num_iter=100,
    warmup_iters=10,
    profiler=BenchmarkProfiler(),
    num_heads_local=16,
    qk_head_dim=128,
    qk_nope_head_dim=192,
    qk_rope_head_dim=64,
    bsz=32,
):
    """Run _fwd_decode_q_rope_nope with trace mode for performance measurement."""
    # Compile Run
    logger.info("Compiling _fwd_decode_q_rope_nope")

    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    q_rope_shape = (1, USERS_PER_ROW, num_heads_local, qk_rope_head_dim)
    q_rope_shard_height = nearest_y(q_rope_shape[2], ttnn.TILE_SIZE)
    q_rope_shard_width = q_rope_shape[3]
    q_rope_num_cores = q_rope_shape[1]
    q_rope_core_grid = ttnn.num_cores_to_corerangeset(q_rope_num_cores, compute_grid_size, row_wise=True)

    # Linear projection
    tt_q = ttnn.linear(
        input_tensor_mesh,
        q_weight_tensor,
        bias=None,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    # Reshape
    tt_q = ttnn.reshape(
        tt_q,
        (bsz, 1, num_heads_local, qk_head_dim),
    )
    tt_q_nope = ttnn.slice(
        tt_q,
        [0, 0, 0, 0],
        [bsz, 1, num_heads_local, qk_nope_head_dim],
    )
    # 32,1,16,192 L1 interleaved

    q_rope_slice_config = SliceConfig(
        memory_config=ttnn.create_sharded_memory_config(
            shape=(q_rope_shard_height, q_rope_shard_width),
            core_grid=q_rope_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
    )
    tt_q_rope = ttnn.slice(
        tt_q,
        [0, 0, 0, qk_nope_head_dim],
        [bsz, 1, num_heads_local, qk_head_dim],
        **q_rope_slice_config,  # Unpack SliceConfig to get memory_config
    )
    tt_q_nope = ttnn.permute(
        tt_q_nope,
        (1, 2, 0, 3),
    )
    tt_q_nope = ttnn.linear(
        tt_q_nope,
        q_nope_weight_tensor,
        bias=None,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_q_nope = ttnn.permute(tt_q_nope, (0, 2, 1, 3))
    tt_q_rope = ttnn.permute(
        tt_q_rope,
        (1, 0, 2, 3),
    )
    tt_q_rope = ttnn.experimental.rotary_embedding_llama(
        tt_q_rope,
        rope_tensors["cos_matrix"],
        rope_tensors["sin_matrix"],
        rope_tensors["trans_matrix"],
        is_decode_mode=True,
    )
    tt_q_rope = ttnn.to_memory_config(
        tt_q_rope,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_q = ttnn.concat(
        [tt_q_nope, tt_q_rope],
        dim=-1,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    ttnn.synchronize_device(mesh_device)

    # Capture warmup trace
    logger.info(f"Capturing warmup trace with {warmup_iters} iterations")
    trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for i in range(warmup_iters):
        # Linear projection
        tt_q = ttnn.linear(
            input_tensor_mesh,
            q_weight_tensor,
            bias=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Reshape
        tt_q = ttnn.reshape(
            tt_q,
            (bsz, 1, num_heads_local, qk_head_dim),
        )

        tt_q_nope = ttnn.slice(
            tt_q,
            [0, 0, 0, 0],
            [bsz, 1, num_heads_local, qk_nope_head_dim],
        )
        # 32,1,16,192 L1 interleaved

        q_rope_slice_config = SliceConfig(
            memory_config=ttnn.create_sharded_memory_config(
                shape=(q_rope_shard_height, q_rope_shard_width),
                core_grid=q_rope_core_grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                use_height_and_width_as_shard_shape=True,
            )
        )
        tt_q_rope = ttnn.slice(
            tt_q,
            [0, 0, 0, qk_nope_head_dim],
            [bsz, 1, num_heads_local, qk_head_dim],
            **q_rope_slice_config,  # Unpack SliceConfig to get memory_config
        )
        tt_q_nope = ttnn.permute(
            tt_q_nope,
            (1, 2, 0, 3),
        )
        tt_q_nope = ttnn.linear(
            tt_q_nope,
            q_nope_weight_tensor,
            bias=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        tt_q_nope = ttnn.permute(tt_q_nope, (0, 2, 1, 3))
        tt_q_rope = ttnn.permute(
            tt_q_rope,
            (1, 0, 2, 3),
        )
        tt_q_rope = ttnn.experimental.rotary_embedding_llama(
            tt_q_rope,
            rope_tensors["cos_matrix"],
            rope_tensors["sin_matrix"],
            rope_tensors["trans_matrix"],
            is_decode_mode=True,
        )
        tt_q_rope = ttnn.to_memory_config(
            tt_q_rope,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        tt_q = ttnn.concat(
            [tt_q_nope, tt_q_rope],
            dim=-1,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        tt_q.deallocate(True)
        tt_q_nope.deallocate(True)
        tt_q_rope.deallocate(True)
    ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Capture main trace
    logger.info(f"Capturing main trace with {num_iter} iterations")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for i in range(num_iter):
        # Linear projection
        tt_q = ttnn.linear(
            input_tensor_mesh,
            q_weight_tensor,
            bias=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Reshape
        tt_q = ttnn.reshape(
            tt_q,
            (bsz, 1, num_heads_local, qk_head_dim),
        )

        tt_q_nope = ttnn.slice(
            tt_q,
            [0, 0, 0, 0],
            [bsz, 1, num_heads_local, qk_nope_head_dim],
        )
        # 32,1,16,192 L1 interleaved

        q_rope_slice_config = SliceConfig(
            memory_config=ttnn.create_sharded_memory_config(
                shape=(q_rope_shard_height, q_rope_shard_width),
                core_grid=q_rope_core_grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                use_height_and_width_as_shard_shape=True,
            )
        )
        tt_q_rope = ttnn.slice(
            tt_q,
            [0, 0, 0, qk_nope_head_dim],
            [bsz, 1, num_heads_local, qk_head_dim],
            **q_rope_slice_config,  # Unpack SliceConfig to get memory_config
        )
        tt_q_nope = ttnn.permute(
            tt_q_nope,
            (1, 2, 0, 3),
        )
        tt_q_nope = ttnn.linear(
            tt_q_nope,
            q_nope_weight_tensor,
            bias=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        tt_q_nope = ttnn.permute(tt_q_nope, (0, 2, 1, 3))
        tt_q_rope = ttnn.permute(
            tt_q_rope,
            (1, 0, 2, 3),
        )
        tt_q_rope = ttnn.experimental.rotary_embedding_llama(
            tt_q_rope,
            rope_tensors["cos_matrix"],
            rope_tensors["sin_matrix"],
            rope_tensors["trans_matrix"],
            is_decode_mode=True,
        )
        tt_q_rope = ttnn.to_memory_config(
            tt_q_rope,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        tt_q = ttnn.concat(
            [tt_q_nope, tt_q_rope],
            dim=-1,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        if i != num_iter - 1:
            tt_q.deallocate(True)
            tt_q_nope.deallocate(True)
            tt_q_rope.deallocate(True)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Execute warmup trace
    logger.info("Executing warmup trace")
    profiler.start("fwd-decode-q-rope-nope-warmup")
    ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
    ttnn.release_trace(mesh_device, trace_id_warmup)
    profiler.end("fwd-decode-q-rope-nope-warmup")

    # Execute main trace with signposts
    logger.info("Executing main trace")
    signpost("start")
    profiler.start("fwd-decode-q-rope-nope")
    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
    profiler.end("fwd-decode-q-rope-nope")
    signpost("stop")

    return tt_q


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize(
    "op_name, input_shape, q_weight_shape, q_nope_weight_shape, bsz, num_heads_local, qk_head_dim, qk_nope_head_dim, qk_rope_head_dim",
    [
        (
            "fwd_decode_q_rope_nope",
            [1, 1, 32, 1536],  # Input: [1, 1, bsz, hidden_size]
            [1536, 3072],
            [128, 512],
            32,
            16,
            192,
            128,
            64,
        ),
    ],
    ids=["fwd_decode_q_rope_nope"],
)
@pytest.mark.parametrize("warmup_iters, num_iters", [(10, 100)])
@pytest.mark.parametrize("trace_mode", [True])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 4752000,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
def test_deepseek_v3_mla_fwd_decode_q_rope_nope_trace_mode(
    mesh_device,
    op_name,
    batch_size,
    input_shape,
    q_weight_shape,
    q_nope_weight_shape,
    bsz,
    num_heads_local,
    qk_head_dim,
    qk_nope_head_dim,
    qk_rope_head_dim,
    trace_mode,
    warmup_iters,
    num_iters,
    function_level_defaults,
):
    """
    Test the complete _fwd_decode_q_rope_nope sequence from mla1d.py (lines 1234-1295).

    This test captures the entire operation sequence:
    1. Linear projection: [1, 1, 32, 1536] x [1536, 3072] -> [1, 1, 32, 3072]
    2. Reshape: [1, 1, 32, 3072] -> [1, 32, 16, 192]
    3. Slice q_nope: [1, 32, 16, 192] -> [1, 32, 16, 128]
    4. Slice q_rope: [1, 32, 16, 192] -> [1, 32, 16, 64]
    5. Permute q_nope: [1, 32, 16, 192] -> [1, 16, 32, 192]
    6. Linear q_nope: [1, 16, 32, 192] x [192, 512] -> [1, 16, 32, 512]
    7. Permute q_nope: [1, 16, 32, 512] -> [1, 132, 16, 512]
    8. Permute q_rope: [1, 32, 16, 64] -> [32, 1, 16, 64]
    9. Rotary embedding q_rope: [32, 1, 16, 64] -> [32, 1, 16, 64]
    10. To memory config: [32, 1, 16, 64] -> [32, 1, 16, 64] L1 interleaved
    11. Concat q_nope and q_rope: [1, 32, 16, 512] + [32, 1, 16, 64] -> [1, 32, 16, 576]

    Configuration:
    - Warmup iterations: 10
    - Test iterations: 100
    - Trace mode: Enabled
    - Input memory: WIDTH_SHARDED 8x4 grid, shard [32, 32]
    - CCL topology: Linear
    """
    torch.manual_seed(0)

    if num_iters < 1:
        pytest.fail("num_iters must be >= 1")

    # Set up sub-devices and semaphores for async operation
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

    # Create input and weight tensors
    logger.info(f"Running wq_kv_a sequence test: {op_name}")
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"Q weight shape: {q_weight_shape}")
    logger.info(f"Q nope weight shape: {q_nope_weight_shape}")
    logger.info(f"Q output shape: {bsz, 1, num_heads_local, qk_head_dim}")
    logger.info(f"Q nope output shape: {bsz, 1, num_heads_local, qk_nope_head_dim}")
    logger.info(f"Q rope output shape: {bsz, 1, num_heads_local, qk_rope_head_dim}")

    # pytorch version
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_q_weight = torch.randn(q_weight_shape, dtype=torch.bfloat16)
    torch_q_nope_weight = torch.randn(q_nope_weight_shape, dtype=torch.bfloat16)
    torch_linear_out = torch_input @ torch_q_weight  # [1, 1, 32, 3072]
    torch_q = torch_linear_out.reshape(bsz, 1, num_heads_local, qk_head_dim)
    torch_q_nope = torch_q[:, :, :, :qk_nope_head_dim]
    torch_q_rope = torch_q[:, :, :, qk_nope_head_dim:]
    torch_q_nope = torch_q_nope.permute(1, 2, 0, 3)
    torch_q_nope = torch_q_nope @ torch_q_nope_weight
    torch_q_nope = torch_q_nope.permute(0, 2, 1, 3)
    torch_q_rope = torch_q_rope.permute(1, 0, 2, 3)
    rope_tensors = create_rope_tensors(mesh_device, qk_rope_head_dim, bsz)
    # Extract the 2D transformation matrix (first 32x32 block) like in test_rope_deepseek.py
    torch_trans_mat = rope_tensors["torch_trans"]  # [1, 1, batch_size*32, 32] or [1, 1, batch_size, 32]
    torch_trans_mat_2d = torch_trans_mat[:, :, 0:32, :]  # [1, 1, 32, 32]
    torch_q_rope = apply_rotary_pos_emb_torch(
        torch_q_rope, rope_tensors["torch_cos"], rope_tensors["torch_sin"], torch_trans_mat_2d
    )
    torch_q = torch.cat([torch_q_nope, torch_q_rope], dim=-1)

    grid_size = mesh_device.compute_with_storage_grid_size()
    shard_grid = ttnn.num_cores_to_corerangeset(16, grid_size, row_wise=True)
    shard_spec = ttnn.ShardSpec(shard_grid, [32, 96], ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)

    input_tensor_mesh = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh_device,
            ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementReplicate()], ttnn.MeshShape(1, 8)),
        ),
    )
    input_tensor_mesh = ttnn.to_memory_config(input_tensor_mesh, mem_config)

    # Convert weight to ttnn
    q_weight_tensor = ttnn.from_torch(
        torch_q_weight,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh_device,
            ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementReplicate()], ttnn.MeshShape(1, 8)),
        ),
    )
    q_nope_weight_tensor = ttnn.from_torch(
        torch_q_nope_weight,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh_device,
            ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementReplicate()], ttnn.MeshShape(1, 8)),
        ),
    )

    profiler = BenchmarkProfiler()

    try:
        if trace_mode:
            # Run sequence with trace
            tt_q = run_fwd_decode_q_rope_nope_with_trace(
                mesh_device,
                input_tensor_mesh,
                q_weight_tensor,
                q_nope_weight_tensor,
                rope_tensors,
                num_iter=num_iters,
                warmup_iters=warmup_iters,
                profiler=profiler,
                num_heads_local=num_heads_local,
                qk_head_dim=qk_head_dim,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                bsz=bsz,
            )
        else:
            pytest.skip("Non-trace mode not implemented for this test")

        # Verify correctness for all three outputs
        logger.info("Verifying correctness")
        passed = True

        # Verify tt_q
        logger.info("Verifying tt_q output")
        for i, t in enumerate(ttnn.get_device_tensors(tt_q)):
            tt_output_tensor = t.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
            logger.info(f"Checking tt_q for device {t.device().id()}")

            passed_q, output_q = assert_with_pcc(tt_output_tensor, torch_q, pcc=0.99)
            if not passed_q:
                logger.error(f"tt_q output mismatch for device {i}: {output_q}")
                passed = False

        assert passed, "Output verification failed for one or more tensors"

    finally:
        # Clean up sub-device configuration
        mesh_device.reset_sub_device_stall_group()
