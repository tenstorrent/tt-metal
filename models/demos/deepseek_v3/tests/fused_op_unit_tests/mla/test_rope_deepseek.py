# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler
from tests.ttnn.utils_for_testing import assert_with_pcc


def apply_rotary_pos_emb_torch(x, cos, sin, trans_mat):
    """
    Apply rotary position embedding in PyTorch for reference.

    This exactly replicates the ttnn.experimental.rotary_embedding_llama kernel:
    1. rotated = x @ trans_mat (matrix multiplication)
    2. sin_term = rotated * sin (element-wise)
    3. cos_term = x * cos (element-wise)
    4. out = cos_term + sin_term

    Args:
        x: Input tensor of shape [batch, seq_len, num_heads, head_dim]
        cos: Cosine tensor of shape [batch, seq_len, 1, head_dim]
        sin: Sine tensor of shape [batch, seq_len, 1, head_dim]
        trans_mat: Transformation matrix of shape [1, 1, 32, 32]

    Returns:
        Output tensor with rotary embeddings applied
    """
    # Extract the 32x32 transformation matrix and convert to same dtype as input
    trans_mat_2d = trans_mat[0, 0, :, :].to(x.dtype)  # [32, 32]

    # Apply transformation to the input in chunks of 32
    # x has shape [..., head_dim], we want to rotate in chunks of 32
    head_dim = x.shape[-1]
    rotated_chunks = []

    for i in range(0, head_dim, 32):
        chunk = x[..., i : i + 32]  # [..., 32]
        # rotated = chunk @ trans_mat.T (matmul on last dimension)
        rotated_chunk = chunk @ trans_mat_2d  # [..., 32] @ [32, 32] = [..., 32]
        rotated_chunks.append(rotated_chunk)

    rotated = torch.cat(rotated_chunks, dim=-1)

    # Apply RoPE formula from kernel
    cos_term = x * cos
    sin_term = rotated * sin
    out = cos_term + sin_term

    return out


def create_rope_tensors(device, head_dim, batch_size=32):
    """Create RoPE cos/sin/trans matrices for testing, matching the model's configuration."""
    # Create cos and sin matrices for RoPE using simplified frequency generation
    # Shape: [1, batch_size, 1, head_dim] after transformation
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(batch_size).float()
    freqs = torch.outer(t, inv_freq)  # [batch_size, head_dim // 2]

    # Stack to create [t1, t1, t2, t2, ..., td/2, td/2] format (meta-style)
    # This matches rope.py lines 57-61
    emb = torch.stack((freqs, freqs), dim=-1).flatten(-2)  # [batch_size, head_dim]

    cos_matrix = emb.cos().reshape(1, batch_size, 1, head_dim)
    sin_matrix = emb.sin().reshape(1, batch_size, 1, head_dim)

    # Create transformation matrix for RoPE (per rope.py line 94-100)
    # Shape: [1, 1, batch_size, 32] - repeated across batch for HEIGHT sharding
    dhead = 32
    rot_emb_matrix = torch.zeros(1, 1, dhead, dhead)
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1
    trans_matrix = rot_emb_matrix.repeat(1, 1, batch_size, 1)

    # Setup sharded memory config for cos/sin (per rope.py line 241-247)
    grid_size = device.compute_with_storage_grid_size()
    num_cores = 32  # batch_size for decode
    batch_grid = ttnn.num_cores_to_corerangeset(num_cores, grid_size, row_wise=True)

    cos_sin_mem_config = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, head_dim),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Setup sharded memory config for transformation matrix (per rope.py line 101-107)
    trans_mat_mem_config = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Convert to ttnn tensors with HEIGHT_SHARDED memory directly
    tt_cos_matrix = ttnn.from_torch(
        cos_matrix,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=cos_sin_mem_config,
    )

    tt_sin_matrix = ttnn.from_torch(
        sin_matrix,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=cos_sin_mem_config,
    )

    tt_trans_matrix = ttnn.from_torch(
        trans_matrix,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=trans_mat_mem_config,
    )

    return {
        "cos_matrix": tt_cos_matrix,
        "sin_matrix": tt_sin_matrix,
        "trans_matrix": tt_trans_matrix,
        "torch_cos": cos_matrix,
        "torch_sin": sin_matrix,
        "torch_trans": trans_matrix,
    }


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize(
    "op_name, input_shape, head_dim, shard_shape",
    [
        (
            "kv_rope_decode",
            [1, 32, 1, 64],
            64,
            [32, 64],  # HEIGHT_SHARDED: kv_rope_shard_height=32, kv_rope_shard_width=64
        ),
        (
            "q_rope_decode",
            [1, 32, 16, 64],
            64,
            [32, 64],  # HEIGHT_SHARDED: q_rope_shard_height=32, q_rope_shard_width=64
        ),
    ],
    ids=["kv_rope_decode", "q_rope_decode"],
)
@pytest.mark.parametrize("warmup_iters", [10])
@pytest.mark.parametrize("num_iters", [100])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 550912,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        }
    ],
    indirect=True,
)
def test_deepseek_v3_mla_rope_trace_mode(
    device,
    batch_size,
    op_name,
    input_shape,
    head_dim,
    shard_shape,
    warmup_iters,
    num_iters,
):
    """
    Test the rotary_embedding_llama operations from mla1d.py with trace mode.

    These operations apply rotary position embeddings in decode mode:
    1. kv_rope_decode (line 1159): [1, 32, 1, 64] with HEIGHT_SHARDED memory
       - kv_rope_mem_cfg from line 457-462: HEIGHT_SHARDED with 32 cores, shard shape [32, 64]
    2. q_rope_decode (line 1251): [1, 32, 16, 64] with HEIGHT_SHARDED memory
       - q_rope_mem_cfg from line 431-436: HEIGHT_SHARDED with 32 cores, shard shape [32, 64]

    Configuration:
    - Warmup iterations: 10
    - Test iterations: 100
    - Trace mode: Enabled
    - Both use HEIGHT_SHARDED with 32 cores (4x8 grid)
    - is_decode_mode: True
    """
    torch.manual_seed(0)

    # Create random tensor for input
    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)

    # Create RoPE tensors (cos, sin, trans matrices)
    rope_tensors = create_rope_tensors(device, head_dim, batch_size)

    torch_cos_single = rope_tensors["torch_cos"]
    torch_sin_single = rope_tensors["torch_sin"]

    # Get transformation matrix from rope_tensors
    # It's shaped [1, 1, batch_size*32, 32] where the 32x32 matrix is repeated vertically
    torch_trans_mat = rope_tensors["torch_trans"]  # [1, 1, 1024, 32]
    # Extract the first 32x32 block
    torch_trans_mat_2d = torch_trans_mat[:, :, 0:32, :]  # [1, 1, 32, 32]

    # Compute golden reference using torch
    torch_output_tensor = apply_rotary_pos_emb_torch(
        torch_input_tensor, torch_cos_single, torch_sin_single, torch_trans_mat_2d
    )

    # Create ttnn tensor with HEIGHT_SHARDED memory config
    # Both kv_rope and q_rope use HEIGHT_SHARDED with 32 cores
    num_cores = 32  # 4x8 grid
    grid_size = device.compute_with_storage_grid_size()
    shard_grid_set = ttnn.num_cores_to_corerangeset(num_cores, grid_size, row_wise=True)

    shard_spec = ttnn.ShardSpec(
        shard_grid_set,
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )

    # Create tensor with L1 interleaved first, then convert to sharded
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_mem_config,
    )

    # Compile run
    logger.info(f"Compiling rotary_embedding_llama operation: {op_name}")
    logger.info(f"  Input shape: {input_shape}")
    logger.info(f"  Head dim: {head_dim}")
    logger.info(f"  Shard shape: {shard_shape}")
    logger.info(f"  Memory config: HEIGHT_SHARDED with 32 cores")

    tt_output_tensor = ttnn.experimental.rotary_embedding_llama(
        tt_input_tensor,
        rope_tensors["cos_matrix"],
        rope_tensors["sin_matrix"],
        rope_tensors["trans_matrix"],
        is_decode_mode=True,
    )
    ttnn.synchronize_device(device)

    # Capture warmup trace
    logger.info(f"Capturing warmup trace with {warmup_iters} iterations")
    trace_id_warmup = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(warmup_iters):
        tt_output_tensor = ttnn.experimental.rotary_embedding_llama(
            tt_input_tensor,
            rope_tensors["cos_matrix"],
            rope_tensors["sin_matrix"],
            rope_tensors["trans_matrix"],
            is_decode_mode=True,
        )
    ttnn.end_trace_capture(device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(device)

    # Capture main trace
    logger.info(f"Capturing main trace with {num_iters} iterations")
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(num_iters):
        tt_output_tensor = ttnn.experimental.rotary_embedding_llama(
            tt_input_tensor,
            rope_tensors["cos_matrix"],
            rope_tensors["sin_matrix"],
            rope_tensors["trans_matrix"],
            is_decode_mode=True,
        )
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)

    # Execute warmup trace
    logger.info("Executing warmup trace")
    profiler = BenchmarkProfiler()
    profiler.start("warmup")
    ttnn.execute_trace(device, trace_id_warmup, blocking=False)
    ttnn.release_trace(device, trace_id_warmup)
    profiler.end("warmup")
    ttnn.synchronize_device(device)

    # Execute main trace with signposts
    logger.info("Executing main trace")
    signpost("start")
    profiler.start("main")
    ttnn.execute_trace(device, trace_id, blocking=False)
    ttnn.release_trace(device, trace_id)
    profiler.end("main")
    signpost("stop")
    ttnn.synchronize_device(device)

    # Verify correctness
    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    torch_output_from_tt = ttnn.to_torch(tt_output_tensor)

    assert (
        torch_output_from_tt.shape == torch_output_tensor.shape
    ), f"Shape mismatch: {torch_output_from_tt.shape} != {torch_output_tensor.shape}"

    # Compare with torch reference implementation
    assert_with_pcc(torch_output_tensor, torch_output_from_tt, 0.9999)

    logger.info(f"✓ Trace mode {op_name} test passed with correct output")
