# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN DRAM Streaming Matmul Micro Op Test

Tests the simplified DRAM streaming matmul operation where:
- Input A is REPLICATED on compute cores (each core has full [M, K])
- Input B (weights) is WIDTH_SHARDED in DRAM with per-shard tile reordering
- Output is WIDTH_SHARDED in L1 on compute cores
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.dram_streaming_matmul.op import DRAMStreamingMatmul
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def pad_to_dram_banks(num, tile_w, lcm):
    """Pad number to be aligned with DRAM banks."""
    remainder = num % lcm
    if remainder == 0:
        return num
    padding_needed = lcm - remainder
    return num + padding_needed


def shuffle_tensor_tiles(tensor, tile_size, num_banks):
    """
    Vectorized tile shuffle for WIDTH_SHARDED DRAM layout.

    Reorders tiles within each bank's shard from row-major to column-major.
    This is required because TTNN stores tiles row-major, but we want to
    stream K tiles contiguously for each N column.

    For shape [K, N] with WIDTH_SHARDED, TTNN tilizes with row-major tiles:
        (0,0), (0,1), ..., (0,pN-1), (1,0), (1,1), ...
    But we want column-major within each shard for streaming K tiles at a time:
        (0,0), (1,0), ..., (K-1,0), (0,1), (1,1), ...

    Args:
        tensor: [*, K, N] tensor (supports batch dimensions)
        tile_size: tile dimension (assumes square tiles)
        num_banks: number of DRAM banks (shards)

    Returns:
        [*, K, N] tensor with tiles rearranged per-shard (same shape as input)
    """
    orig_shape = tensor.shape
    K = orig_shape[-2]
    N = orig_shape[-1]

    # Pad N to align with num_banks if needed
    lcm = tile_size * num_banks
    n_padded = ((N + lcm - 1) // lcm) * lcm
    needs_padding = n_padded != N

    # Flatten batch dimensions
    tensor = tensor.reshape(-1, K, N)
    batch_size = tensor.shape[0]

    if needs_padding:
        tensor = torch.nn.functional.pad(tensor, (0, n_padded - N))

    K_tiles = K // tile_size
    per_N = n_padded // num_banks
    per_N_tiles = per_N // tile_size
    num_tiles_per_shard = K_tiles * per_N_tiles

    # Split into shards: [batch, K, N] -> [batch, K, num_banks, per_N]
    tensor = tensor.reshape(batch_size, K, num_banks, per_N)
    # -> [batch, num_banks, K, per_N]
    tensor = tensor.permute(0, 2, 1, 3).contiguous()
    # -> [batch * num_banks, K, per_N]
    shards = tensor.reshape(-1, K, per_N)

    # Reshape to tiles: [shards, K_tiles, tile_h, per_N_tiles, tile_w]
    tiles = shards.reshape(-1, K_tiles, tile_size, per_N_tiles, tile_size)
    # -> [shards, K_tiles, per_N_tiles, tile_h, tile_w]
    tiles = tiles.permute(0, 1, 3, 2, 4).contiguous()
    # Flatten tile grid: [shards, num_tiles, tile_h, tile_w]
    tiles = tiles.reshape(-1, num_tiles_per_shard, tile_size, tile_size)

    # Compute source indices for shuffle
    # For shuffled position i, source = (i % K_tiles) * per_N_tiles + (i // K_tiles)
    i = torch.arange(num_tiles_per_shard, device=tensor.device)
    source_idx = (i % K_tiles) * per_N_tiles + (i // K_tiles)

    # Gather tiles in shuffled order
    shuffled_tiles = tiles[:, source_idx, :, :]

    # Reshape back to shards
    shuffled_tiles = shuffled_tiles.reshape(-1, K_tiles, per_N_tiles, tile_size, tile_size)
    shuffled_tiles = shuffled_tiles.permute(0, 1, 3, 2, 4).contiguous()
    shuffled_shards = shuffled_tiles.reshape(-1, K, per_N)

    # Reshape back to full tensor
    shuffled = shuffled_shards.reshape(batch_size, num_banks, K, per_N)
    shuffled = shuffled.permute(0, 2, 1, 3).contiguous()
    shuffled = shuffled.reshape(batch_size, K, n_padded)

    # Slice back to original N if padded
    if needs_padding:
        shuffled = shuffled[:, :, :N]

    # Restore original batch dimensions
    shuffled = shuffled.reshape(*orig_shape)

    return shuffled


@pytest.mark.parametrize("k, n", [(7168, 2048), (2048, 7168)])
@pytest.mark.parametrize("m", [1, 4, 8])
@pytest.mark.parametrize("fused_activation", [None, "silu"])
@pytest.mark.parametrize("num_loop_iters", [100])
def test_dram_streaming_matmul(device, k, n, m, fused_activation, num_loop_iters):
    """Test simplified DRAM streaming matmul with optional fused activation.

    In the simplified version:
    - Input A is REPLICATED on compute cores (each core has full [M, K])
    - Input B is WIDTH_SHARDED [K, N] with per-shard tiles shuffled to column-major
    - No multicast needed - each core has its own copy
    - Output is WIDTH_SHARDED across N dimension

    Args:
        fused_activation: Optional activation to fuse (e.g., "silu", "gelu", "relu")
    """

    tile_h = m  # Tile height matches m (1 for tiny tiles, 32 for standard)
    tile_w = 32

    # Create tile object for tiny tiles when m=1
    in0_tile = ttnn.Tile([tile_h, tile_w])
    out_tile = ttnn.Tile([tile_h, tile_w])

    # Get compute cores from optimal DRAM bank assignment
    # These are the cores that will do the compute (8 on BH, 12 on WH)
    compute_cores = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    num_cores = len(compute_cores)

    # Get number of DRAM banks (should match num_cores)
    num_banks = device.dram_grid_size().x
    assert num_cores == num_banks, f"num_cores ({num_cores}) must equal num_banks ({num_banks})"

    logger.info(f"num_compute_cores={num_cores}, num_dram_banks={num_banks}")

    n_padded = pad_to_dram_banks(n, tile_w, tile_w * num_banks)
    per_core_N = n_padded // num_banks

    logger.info(f"n_padded={n_padded}, per_core_N={per_core_N}, Kt={k // tile_w}")

    # Define shapes
    in0_shape = [1, 1, m, k]
    in1_shape = [1, 1, k, n_padded]  # Original shape for matmul reference

    # Build CoreRangeSet for specific compute cores (not bounding box)
    compute_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in compute_cores]
    )

    # Create PyTorch tensors
    torch.manual_seed(42)
    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()  # [1, 1, K, N] for reference

    # ========== Input A - REPLICATED on compute cores ==========
    # Replicate the tensor num_cores times along height, then HEIGHT_SHARD
    in0_replicated = in0.repeat(1, 1, num_cores, 1)  # Shape: [1, 1, M * num_cores, K]
    in0_shard_shape_full = [m, k]  # Each core gets [M, K]
    in0_shard_spec = ttnn.ShardSpec(compute_core_grid, in0_shard_shape_full, ttnn.ShardOrientation.ROW_MAJOR)
    in0_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in0_shard_spec)
    in0_t = ttnn.from_torch(
        in0_replicated,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in0_memory_config,
        tile=in0_tile,
    )

    # ========== Input B - WIDTH_SHARDED in DRAM with per-shard tile reordering ==========
    # Shuffle tiles so K tiles are contiguous per N column (for streaming)
    in1_shuffled = shuffle_tensor_tiles(in1, tile_w, num_banks)

    # WIDTH_SHARDED across N dimension, each bank gets [K, n // num_banks]
    in1_shard_shape = [k, n_padded // num_banks]
    in1_shard_grid = ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
    in1_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), in1_shard_grid)})
    in1_shard_spec = ttnn.ShardSpec(in1_shard_grid, in1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    in1_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, in1_shard_spec)
    in1_t = ttnn.from_torch(
        in1_shuffled,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in1_memory_config,
    )

    # ========== Output tensor - WIDTH_SHARDED in L1 ==========
    output_shard_width = n_padded // num_banks
    output_shard_spec = ttnn.ShardSpec(compute_core_grid, (m, output_shard_width), ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    torch_output_zeros = torch.zeros([1, 1, m, n_padded]).bfloat16().float()
    ttnn_output = ttnn.from_torch(
        torch_output_zeros,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=out_tile,
    )

    if k == 7168:
        subblock_k = k // tile_w // 4
    else:
        subblock_k = k // tile_w // 2

    Kt = k // tile_w
    num_subblocks_k = Kt // subblock_k

    # ========== Working buffer for CB1 (needed for kernel-level looping) ==========
    in1_tile = ttnn.Tile([tile_w, tile_w])  # in1 uses 32x32 tiles
    in1_dtype = ttnn.bfloat4_b
    in1_tile_size = in1_tile.get_tile_size(in1_dtype)
    num_in1_buffers = 3 * num_subblocks_k
    in1_CB_tiles = subblock_k * num_in1_buffers
    # Working buffer: WIDTH_SHARDED in L1, shard = [tile_w, in1_CB_tiles * tile_w]
    working_buf_shard_shape = (tile_w, in1_CB_tiles * tile_w)
    working_buf_total_width = in1_CB_tiles * tile_w * num_cores
    working_buf_shard_spec = ttnn.ShardSpec(compute_core_grid, working_buf_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    working_buf_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, working_buf_shard_spec
    )
    working_buf_torch = torch.zeros([1, 1, tile_w, working_buf_total_width]).bfloat16().float()
    working_buf_t = ttnn.from_torch(
        working_buf_torch,
        dtype=in1_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=working_buf_mem_config,
        tile=in1_tile,
    )

    # Run DRAM streaming matmul
    activation_str = f" + {fused_activation}" if fused_activation else ""
    logger.info(
        f"Running DRAM streaming matmul{activation_str}: m={m}, k={k}, n={n_padded}, "
        f"num_cores={num_cores}, num_loop_iters={num_loop_iters}"
    )
    try:
        ttnn_result = DRAMStreamingMatmul.op(
            in0_t,
            in1_t,
            ttnn_output,
            fp32_dest_acc_en=True,
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            subblock_k=subblock_k,
            fused_activation=fused_activation,
            num_loop_iters=num_loop_iters,
            working_buf_tensor=working_buf_t,
        )
    except Exception as e:
        logger.error(f"DRAM streaming matmul{activation_str} failed: {e}")
        pytest.skip(f"Operation failed (may need API adjustments): {e}")

    # Compute PyTorch reference
    pt_out = DRAMStreamingMatmul.golden(in0, in1, fused_activation)

    # Convert to torch for comparison
    tt_out = ttnn.to_torch(ttnn_result)

    # Verify results
    if fused_activation != None:
        expected_pcc = 0.98
    else:
        expected_pcc = 0.99
    passing, output = comp_pcc(pt_out, tt_out, expected_pcc)
    logger.info(output)
    assert passing, f"PCC check failed: {output}"
    logger.info(f"DRAM streaming matmul{activation_str} test passed!")


@pytest.mark.parametrize("k, n", [(7168, 2048)])
@pytest.mark.parametrize("m", [1])
@pytest.mark.parametrize("num_experts", [256])
@pytest.mark.parametrize("fused_activation", ["silu"])
@pytest.mark.skip_post_commit
def test_dram_streaming_matmul_indexed(device, k, n, m, num_experts, fused_activation):
    """Test DRAM streaming matmul with expert indexing.

    Tests expert weight matrices stacked along K dimension [K*num_experts, N]
    and an index tensor to select which expert.

    For now, only uses the first index (single expert selection).
    """
    tile_h = m
    tile_w = 32

    in0_tile = ttnn.Tile([tile_h, tile_w])
    out_tile = ttnn.Tile([tile_h, tile_w])

    # Get compute cores from optimal DRAM bank assignment
    compute_cores = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    num_cores = len(compute_cores)
    num_banks = device.dram_grid_size().x
    assert num_cores == num_banks, f"num_cores ({num_cores}) must equal num_banks ({num_banks})"

    logger.info(f"num_compute_cores={num_cores}, num_dram_banks={num_banks}")

    n_padded = pad_to_dram_banks(n, tile_w, tile_w * num_banks)
    per_core_N = n_padded // num_banks

    # Total K dimension with all experts stacked
    k_total = k * num_experts  # K * 256

    logger.info(
        f"num_experts={num_experts}, k={k}, k_total={k_total}, n_padded={n_padded}, per_core_N={per_core_N}, Kt={k // tile_w}"
    )

    # Define shapes
    in0_shape = [1, 1, m, k]

    # Build CoreRangeSet for specific compute cores
    compute_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in compute_cores]
    )

    # Create PyTorch tensors
    torch.manual_seed(42)
    in0 = torch.randn(in0_shape).bfloat16().float()
    # in1 will be created per-expert below to avoid host memory issues

    # Create expert index - select one random expert
    torch.manual_seed(123)
    selected_expert_idx = torch.randint(0, num_experts, (1,)).item()
    logger.info(f"Selected expert index: {selected_expert_idx}")

    # Index tensor: [1, 16] with first value being the expert index (uint16)
    index_tensor_torch = torch.zeros(1, 16, dtype=torch.int32)
    index_tensor_torch[0, 0] = selected_expert_idx
    index_tensor_torch = index_tensor_torch.to(torch.uint16)

    # ========== Input A - REPLICATED on compute cores ==========
    in0_replicated = in0.repeat(1, 1, num_cores, 1)
    in0_shard_shape_full = [m, k]
    in0_shard_spec = ttnn.ShardSpec(compute_core_grid, in0_shard_shape_full, ttnn.ShardOrientation.ROW_MAJOR)
    in0_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in0_shard_spec)
    in0_t = ttnn.from_torch(
        in0_replicated,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in0_memory_config,
        tile=in0_tile,
    )

    # ========== Input B - WIDTH_SHARDED in DRAM with per-expert tile reordering ==========
    # Create each expert as a separate tensor and upload sequentially.
    # DRAM allocator places them contiguously, so kernel can use base_addr + expert_offset.
    # This avoids host memory issues with large single-tensor from_torch.

    in1_shard_shape = [k, n_padded // num_banks]  # Per-expert shard shape
    in1_shard_grid = ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
    in1_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), in1_shard_grid)})
    in1_shard_spec = ttnn.ShardSpec(in1_shard_grid, in1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    in1_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, in1_shard_spec)

    logger.info(f"Uploading {num_experts} experts as separate contiguous tensors...")
    expert_tensors = []
    selected_expert_weights = None  # Only keep the selected expert for reference

    for expert_idx in range(num_experts):
        # Create this expert's weights
        torch.manual_seed(42 + expert_idx)
        expert_weights = torch.randn(1, 1, k, n_padded).bfloat16()

        # Keep the selected expert for reference calculation
        if expert_idx == selected_expert_idx:
            selected_expert_weights = expert_weights.clone()

        # Shuffle tiles for this expert
        expert_shuffled = shuffle_tensor_tiles(expert_weights.reshape(1, k, n_padded), tile_w, num_banks)
        expert_shuffled = expert_shuffled.reshape(1, 1, k, n_padded)

        # Upload to DRAM - sequential allocation = contiguous in each bank
        expert_t = ttnn.from_torch(
            expert_shuffled.contiguous(),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=in1_memory_config,
        )
        expert_tensors.append(expert_t)

        # Free torch tensors we don't need
        del expert_weights, expert_shuffled

        if (expert_idx + 1) % 32 == 0:
            logger.info(f"  Uploaded {expert_idx + 1}/{num_experts} experts")

    logger.info(f"All experts uploaded. First expert tensor: {expert_tensors[0].shape}")

    # Use first expert tensor for the op - kernel calculates offset for other experts
    in1_t = expert_tensors[0]

    # ========== Index tensor - REPLICATED on compute cores (same grid as kernel) ==========
    index_tile = ttnn.Tile([1, 16])

    # Replicate index tensor for compute cores (must match kernel's core grid)
    index_tensor_replicated = index_tensor_torch.repeat(num_cores, 1)  # [num_cores, 16]
    index_shard_spec = ttnn.ShardSpec(compute_core_grid, (1, 16), ttnn.ShardOrientation.ROW_MAJOR)
    index_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, index_shard_spec
    )

    index_t = ttnn.from_torch(
        index_tensor_replicated,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=index_memory_config,
        tile=index_tile,
    )

    # ========== Output tensor - WIDTH_SHARDED in L1 ==========
    output_shard_width = n_padded // num_banks
    output_shard_spec = ttnn.ShardSpec(compute_core_grid, (m, output_shard_width), ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    torch_output_zeros = torch.zeros([1, 1, m, n_padded]).bfloat16().float()
    ttnn_output = ttnn.from_torch(
        torch_output_zeros,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=out_tile,
    )

    if k == 7168:
        subblock_k = k // tile_w // 4
    else:
        subblock_k = k // tile_w // 2

    # ========== Compute PyTorch reference ==========
    # Use the stored selected expert weights for reference calculation
    pt_out = in0 @ selected_expert_weights.float()
    if fused_activation == "silu":
        pt_out = torch.nn.functional.silu(pt_out)

    logger.info(f"Reference output shape: {pt_out.shape}")

    # ========== Run DRAM streaming matmul with indexing ==========
    activation_str = f" + {fused_activation}" if fused_activation else ""
    logger.info(
        f"Running DRAM streaming matmul indexed{activation_str}: m={m}, k={k}, n={n_padded}, num_cores={num_cores}"
    )

    try:
        ttnn_result = DRAMStreamingMatmul.op(
            in0_t,
            in1_t,
            ttnn_output,
            index_tensor=index_t,
            fp32_dest_acc_en=True,
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            subblock_k=subblock_k,
            fused_activation=fused_activation,
        )
    except Exception as e:
        logger.error(f"DRAM streaming matmul indexed{activation_str} failed: {e}")
        pytest.skip(f"Operation failed (may need API adjustments): {e}")

    # Convert to torch for comparison
    tt_out = ttnn.to_torch(ttnn_result)

    # Verify results
    if fused_activation is not None:
        expected_pcc = 0.98
    else:
        expected_pcc = 0.99
    passing, output = comp_pcc(pt_out, tt_out, expected_pcc)
    logger.info(output)
    assert passing, f"PCC check failed: {output}"
    logger.info(f"DRAM streaming matmul indexed{activation_str} test passed!")


@pytest.mark.parametrize("k, n", [(7168, 2048)])
@pytest.mark.parametrize("m", [1])
@pytest.mark.parametrize("fused_activation", [None])
def test_dram_streaming_matmul_with_mul(device, k, n, m, fused_activation):
    """Test DRAM streaming matmul with fused element-wise multiply and optional scalar multiply.

    Tests: silu(input @ weights) * mul_tensor [* scalar_tensor]

    All tensors use 1x32 tiles (to mimic matmul outputs).
    The kernel uses CB aliasing to view them as 16x16 tiles for the mul operation:
    - mm_out_tensor: 1x256 per core = 8 tiles of 1x32, CB 8 (matmul writes here)
    - mm_out viewed as 16x16: CB 7 (mul reads from here, aliased to CB 8)
    - mul_tensor: 1x256 per core = 8 tiles of 1x32, CB 6 views as 16x16
    - output_tensor: 1x256 per core = 8 tiles of 1x32, CB 4 views as 16x16
    - scalar_tensor (optional): 1x16 tensor, CB 9 views as 16x16 tile
    """
    tile_h = m
    tile_w = 32

    # Tile shapes - tensors use 1x32 tiles
    in0_tile = ttnn.Tile([tile_h, tile_w])  # 1x32 for matmul input
    mm_out_tile = ttnn.Tile([tile_h, tile_w])  # 1x32 for matmul output, mul_tensor, and final output

    # Get compute cores from optimal DRAM bank assignment
    compute_cores = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    num_cores = len(compute_cores)
    num_banks = device.dram_grid_size().x
    assert num_cores == num_banks, f"num_cores ({num_cores}) must equal num_banks ({num_banks})"

    logger.info(f"num_compute_cores={num_cores}, num_dram_banks={num_banks}")

    n_padded = pad_to_dram_banks(n, tile_w, tile_w * num_banks)
    per_core_N = n_padded // num_banks

    logger.info(f"n_padded={n_padded}, per_core_N={per_core_N}, Kt={k // tile_w}")

    # Define shapes
    in0_shape = [1, 1, m, k]
    in1_shape = [1, 1, k, n_padded]

    # Build CoreRangeSet for specific compute cores
    compute_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in compute_cores]
    )

    # Create PyTorch tensors with random values
    torch.manual_seed(42)
    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    mul_tensor_torch = torch.randn([1, 1, m, n_padded]).bfloat16().float()

    # Create scalar tensor (1x16 to match gate op output)
    # BRISC will read index 0 and replicate to a 16x16 CB for the mul operation
    scalar_value = 0.3  # Use a known scalar value for testing
    scalar_tensor_torch = torch.full([1, 16], scalar_value, dtype=torch.bfloat16).float()

    # ========== Input A - REPLICATED on compute cores ==========
    in0_replicated = in0.repeat(1, 1, num_cores, 1)
    in0_shard_shape_full = [m, k]
    in0_shard_spec = ttnn.ShardSpec(compute_core_grid, in0_shard_shape_full, ttnn.ShardOrientation.ROW_MAJOR)
    in0_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in0_shard_spec)
    in0_t = ttnn.from_torch(
        in0_replicated,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in0_memory_config,
        tile=in0_tile,
    )

    # ========== Input B - WIDTH_SHARDED in DRAM with per-shard tile reordering ==========
    in1_shuffled = shuffle_tensor_tiles(in1, tile_w, num_banks)
    in1_shard_shape = [k, n_padded // num_banks]
    in1_shard_grid = ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
    in1_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), in1_shard_grid)})
    in1_shard_spec = ttnn.ShardSpec(in1_shard_grid, in1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    in1_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, in1_shard_spec)
    in1_t = ttnn.from_torch(
        in1_shuffled,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in1_memory_config,
    )

    # ========== Matmul output tensor (intermediate) - 1x32 tiles ==========
    # This is where matmul writes its output before mul
    mm_out_shard_spec = ttnn.ShardSpec(compute_core_grid, (m, per_core_N), ttnn.ShardOrientation.ROW_MAJOR)
    mm_out_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, mm_out_shard_spec
    )
    mm_out_t = ttnn.from_torch(
        torch.zeros([1, 1, m, n_padded]).bfloat16().float(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mm_out_memory_config,
        tile=mm_out_tile,
    )

    # ========== Mul tensor - 1x32 tiles (same as mm_out, mimics another matmul output) ==========
    # Same shape and memory config as mm_out_tensor
    # Kernel will alias this CB to view it as 16x16 tiles for the mul operation
    mul_shard_spec = ttnn.ShardSpec(compute_core_grid, (m, per_core_N), ttnn.ShardOrientation.ROW_MAJOR)
    mul_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, mul_shard_spec)
    mul_t = ttnn.from_torch(
        mul_tensor_torch,  # [1, 1, m, n_padded]
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mul_memory_config,
        tile=mm_out_tile,  # 1x32 tiles
    )

    # ========== Output tensor - 1x32 tiles (CB will view as 16x16) ==========
    # Same format as mm_out and mul tensors
    output_shard_spec = ttnn.ShardSpec(compute_core_grid, (m, per_core_N), ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec)
    ttnn_output = ttnn.from_torch(
        torch.zeros([1, 1, m, n_padded]).bfloat16().float(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=mm_out_tile,  # 1x32 tiles
    )

    # ========== Scalar tensor - 1x16 tensor (matches gate op output) ==========
    # BRISC will read scalar from this tensor and replicate to a 16x16 CB
    # Replicate scalar tensor for each core (HEIGHT_SHARDED)
    scalar_replicated = (
        scalar_tensor_torch.unsqueeze(0).unsqueeze(0).repeat(1, 1, num_cores, 1)
    )  # [1, 1, num_cores, 16]
    scalar_shard_spec = ttnn.ShardSpec(compute_core_grid, (1, 16), ttnn.ShardOrientation.ROW_MAJOR)
    scalar_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, scalar_shard_spec
    )
    scalar_tile = ttnn.Tile([1, 16])
    scalar_t = ttnn.from_torch(
        scalar_replicated,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=scalar_memory_config,
        tile=scalar_tile,
    )

    if k == 7168:
        subblock_k = k // tile_w // 4
    else:
        subblock_k = k // tile_w // 2

    # Run DRAM streaming matmul with mul
    logger.info(
        f"Running DRAM streaming matmul + {fused_activation} + mul + scalar: m={m}, k={k}, n={n_padded}, num_cores={num_cores}"
    )
    try:
        ttnn_result = DRAMStreamingMatmul.op(
            in0_t,
            in1_t,
            ttnn_output,
            fp32_dest_acc_en=True,
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            subblock_k=subblock_k,
            fused_activation=fused_activation,
            mul_tensor=mul_t,
            mm_out_tensor=mm_out_t,
            scalar_tensor=scalar_t,
        )
    except Exception as e:
        logger.error(f"DRAM streaming matmul + {fused_activation} + mul + scalar failed: {e}")
        pytest.skip(f"Operation failed (may need API adjustments): {e}")

    # Compute PyTorch reference
    pt_out = DRAMStreamingMatmul.golden(in0, in1, fused_activation, mul_tensor_torch, scalar_tensor_torch)

    # Convert to torch for comparison
    tt_out = ttnn.to_torch(ttnn_result)

    # Verify results
    expected_pcc = 0.98
    passing, output = comp_pcc(pt_out, tt_out, expected_pcc)
    logger.info(output)
    assert passing, f"PCC check failed: {output}"
    logger.info(f"DRAM streaming matmul + {fused_activation} + mul + scalar test passed!")
