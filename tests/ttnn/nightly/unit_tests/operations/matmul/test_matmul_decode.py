# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import math
import torch

torch.set_printoptions(sci_mode=False)

import ttnn
from tracy import signpost
from tests.ttnn.utils_for_testing import assert_with_pcc

valid_tile_heights = [1, 2, 4, 8, 16, 32]


def get_tile_height(m):
    for tile_height in valid_tile_heights:
        if m <= tile_height:
            return tile_height
    return 32


def num_cores_to_rectangle_grid(num_cores, device):
    """Largest x that divides num_cores and fits the device grid; returns (x, y) or None."""
    x = device.compute_with_storage_grid_size().x
    while x > 0 and num_cores % x != 0:
        x -= 1
    if x == 0:
        return None
    return (x, num_cores // x)


def num_cores_to_rectangle_core_range_set(num_cores, device):
    """A single rectangular ``CoreRangeSet`` of exactly ``num_cores`` cores.

    Mirrors ``LinearDecode``'s ``_num_cores_to_rectangle_core_range_set`` in
    deepseek_v4_flash: finds the widest ``x`` dividing ``num_cores`` that fits the
    device grid, giving an ``(x, num_cores // x)`` rectangle.
    """
    grid = device.compute_with_storage_grid_size()
    x = grid.x
    while x > 0 and num_cores % x != 0:
        x -= 1
    y = num_cores // x if x > 0 else 0
    if x == 0 or y > grid.y:
        raise ValueError(f"cannot form a rectangular grid of {num_cores} cores within a {grid.x}x{grid.y} device grid")
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x - 1, y - 1))})


def find_subblock(per_core_m, per_core_n):
    """Pick (out_subblock_h, out_subblock_w) dividing the block dims with h*w <= 8."""
    for h in range(per_core_m, 0, -1):
        if per_core_m % h != 0:
            continue
        for w in range(per_core_n, 0, -1):
            if per_core_n % w == 0 and h * w <= 8:
                return h, w
    return 1, 1


@pytest.mark.parametrize(
    "m, k, n",
    [
        (1, 1024, 4096),
        (4, 1024, 4096),
        (8, 1024, 4096),
        (16, 1024, 4096),
        (32, 1024, 4096),
    ],
)
@pytest.mark.parametrize(
    "num_inputA_cores",
    [
        (32),
    ],
)
def test_matmul_decode(device, m, k, n, num_inputA_cores):
    torch.manual_seed(0)
    num_inputB_cores = n // 64
    if device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y < num_inputB_cores:
        pytest.skip(f"Skipping test as device doesn't have {num_inputB_cores} cores")
    tile_height = get_tile_height(m)
    inputA_tile_size = ttnn.Tile((tile_height, 32))
    print(f"num_inputA_cores: {num_inputA_cores}, num_inputB_cores: {num_inputB_cores}")
    torch_input_tensor_a = torch.randn((m, k), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k, n), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a.to(torch.float32) @ torch_input_tensor_b.to(torch.float32)

    input_a_core_range_set = num_cores_to_rectangle_core_range_set(num_inputA_cores, device)
    input_b_core_range_set = num_cores_to_rectangle_core_range_set(num_inputB_cores, device)
    in0_memory_config = ttnn.create_sharded_memory_config(
        (m, k // num_inputA_cores),
        core_grid=input_a_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    in1_memory_config = ttnn.create_sharded_memory_config(
        (k, n // num_inputB_cores),
        core_grid=input_b_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        tile=inputA_tile_size,
        device=device,
        memory_config=in0_memory_config,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in1_memory_config
    )

    # ---- ttnn.matmul (gather_in0) baseline for perf comparison ----
    # Mirror matmul_decode's residency: in0 (activations) and in1 (weights) are both L1
    # WIDTH_SHARDED across the same core grid, output is L1 WIDTH_SHARDED. gather_in0
    # gathers the activation across the ring, exactly like the decode op. gather_in0 needs
    # both operands on one grid with tile-aligned shards, so the core count must divide both
    # k/32 and n/32; use the largest such count.
    mm_num_cores = math.gcd(k // 32, n // 32)
    mm_storage_grid = num_cores_to_rectangle_grid(mm_num_cores, device)
    if mm_storage_grid is None:
        pytest.skip(f"Cannot form a rectangular grid from {mm_num_cores} cores")
    mm_num_cores = mm_storage_grid[0] * mm_storage_grid[1]
    mm_core_range_set = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(mm_storage_grid[0] - 1, mm_storage_grid[1] - 1))}
    )
    k_per_shard = k // mm_num_cores
    n_per_shard = n // mm_num_cores
    # Shard heights must be tile-aligned; pad M up to a full tile (decode has M < 32).
    m_padded = ((m + 31) // 32) * 32
    mm_in0_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(mm_core_range_set, [m_padded, k_per_shard], ttnn.ShardOrientation.ROW_MAJOR),
    )
    mm_in1_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(mm_core_range_set, [k, n_per_shard], ttnn.ShardOrientation.ROW_MAJOR),
    )
    mm_out_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(mm_core_range_set, [m_padded, n_per_shard], ttnn.ShardOrientation.ROW_MAJOR),
    )
    mm_input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=mm_in0_mem_config,
    )
    mm_input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=mm_in1_mem_config,
    )
    per_core_M = (m + 31) // 32
    per_core_N = n_per_shard // 32
    out_subblock_h, out_subblock_w = find_subblock(per_core_M, per_core_N)
    mm_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=mm_storage_grid,
        in0_block_w=k_per_shard // 32,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
        gather_in0=True,
    )

    # Run both ops back-to-back (twice) so a profiler trace captures each for comparison.
    signpost("matmul_decode")
    for _ in range(2):
        output_tensor = ttnn.experimental.matmul_decode(input_tensor_a, input_tensor_b)
    signpost("ttnn_matmul_gather_in0")
    for _ in range(2):
        mm_output_tensor = ttnn.matmul(
            mm_input_tensor_a,
            mm_input_tensor_b,
            program_config=mm_program_config,
            memory_config=mm_out_mem_config,
        )
    signpost("stop")

    assert output_tensor.shape == (m, n)
    assert mm_output_tensor.shape == (m, n)
    assert_with_pcc(torch_output_tensor, ttnn.to_torch(output_tensor), 0.99)


@pytest.mark.parametrize(
    "m, k, n, k_blocks, n_blocks",
    [
        (1, 4096, 1024, 2, 32),
        (4, 4096, 1024, 2, 32),
        (8, 4096, 1024, 2, 32),
        (16, 4096, 1024, 2, 32),
        (32, 4096, 1024, 2, 32),
        (64, 4096, 1024, 2, 32),
    ],
)
@pytest.mark.parametrize(
    "num_inputA_cores",
    [
        (32),
    ],
)
def test_matmul_decode_partial_width_sharded(device, m, k, n, k_blocks, n_blocks, num_inputA_cores):
    torch.manual_seed(0)
    tile_height = get_tile_height(m)
    inputA_tile_size = ttnn.Tile((tile_height, 32))
    kc = k // k_blocks
    nc = n // n_blocks
    num_inputB_cores = k_blocks * n_blocks
    print(
        f"num_inputA_cores: {num_inputA_cores}, num_inputB_cores: {num_inputB_cores}, "
        f"kc: {kc}, nc: {nc}, k_blocks: {k_blocks}, n_blocks: {n_blocks}"
    )
    if device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y < num_inputB_cores:
        pytest.skip(f"Skipping test as device doesn't have {num_inputB_cores} cores")

    torch_input_tensor_a = torch.randn((m, k), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k, n), dtype=torch.bfloat16)

    ref = torch_input_tensor_a.to(torch.float32) @ torch_input_tensor_b.to(torch.float32)
    m_padded = ((m + 31) // 32) * 32

    # Reshape + permute B so that a width-sharded tensor distributes a 2D (K x N)
    # block grid across cores: core c (row-major) holds B[kb*kc:(kb+1)*kc, nb*nc:(nb+1)*nc]
    # with c = kb * n_blocks + nb.
    torch_input_tensor_b_reshaped = torch_input_tensor_b.reshape(k_blocks, kc, n)
    torch_input_tensor_b_reshaped = torch.permute(torch_input_tensor_b_reshaped, (1, 0, 2))
    print("torch_input_tensor_b_reshaped.shape:", torch_input_tensor_b_reshaped.shape)
    torch_input_tensor_b_reshaped = torch_input_tensor_b_reshaped.reshape(kc, n * k_blocks)

    input_a_core_range_set = num_cores_to_rectangle_core_range_set(num_inputA_cores, device)
    input_b_core_range_set = num_cores_to_rectangle_core_range_set(num_inputB_cores, device)
    in0_memory_config = ttnn.create_sharded_memory_config(
        (m_padded, k // num_inputA_cores),
        core_grid=input_a_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    in1_memory_config = ttnn.create_sharded_memory_config(
        (kc, nc),
        core_grid=input_b_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in0_memory_config,
        dtype=ttnn.bfloat16,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b_reshaped,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in1_memory_config,
        dtype=ttnn.bfloat16,
    )
    print("input_tensor_a.shape:", input_tensor_a.shape)
    print("input_tensor_b.shape:", input_tensor_b.shape)

    # Mirror LinearDecode.forward (deepseek_v4_flash q_a_proj): the partial layout reduces the
    # K-partials onto n_blocks output cores, so shard the output WIDTH_SHARDED across n_blocks
    # cores (shard [padded_m, n / n_blocks]).
    output_core_range_set = num_cores_to_rectangle_core_range_set(n_blocks, device)
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(output_core_range_set, [m_padded, n // n_blocks], ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_tensor = ttnn.experimental.matmul_decode(
        input_tensor_a, input_tensor_b, partial_width_sharded=True, output_mem_config=output_mem_config
    )

    assert output_tensor.shape == (m, n)

    out = ttnn.to_torch(output_tensor).float()
    assert_with_pcc(ref, out, 0.99)


def _rectangle_core_range_set(width, height, device):
    """A single ``width`` x ``height`` rectangular ``CoreRangeSet`` anchored at (0, 0)."""
    grid = device.compute_with_storage_grid_size()
    if width > grid.x or height > grid.y:
        raise ValueError(f"cannot fit a {width}x{height} core rectangle within a {grid.x}x{grid.y} device grid")
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(width - 1, height - 1))})


@pytest.mark.parametrize(
    "d0, d1, m, k, n, b_blocks, n_blocks",
    [
        #     (2, 4, 1, 1024, 1024, 8, 4),  # batch = 8, Bc = 1 (one batch per core)
        #     (2, 4, 4, 1024, 1024, 8, 4),
        #     (2, 4, 4, 1024, 1024, 4, 4),  # Bc = 2 (a block spans two batches)
        #     (1, 8, 16, 1024, 1024, 4, 8),
        (1, 8, 32, 4096, 1024, 8, 8),  # larger K/N; previously overflowed L1
    ],
)
@pytest.mark.parametrize(
    "num_inputA_cores",
    [
        (32),
    ],
)
def test_matmul_decode_batched_width_sharded(device, d0, d1, m, k, n, b_blocks, n_blocks, num_inputA_cores):
    """Batched matmul C[b] = A[b] @ B[b] with the weights folded along BOTH batch and N.

    A is rank-4 ([d0, d1, M, K]); the batch is the product of the two leading dims (batch = d0*d1).
    The weights ([batch, K, N]) are reshaped/permuted so a 2D (b_blocks x n_blocks) grid of
    [Bc, K, Nc] blocks maps across b_blocks * n_blocks cores (Bc = batch / b_blocks,
    Nc = N / n_blocks) and are passed as a rank-4 width-sharded tensor [1, 1, Bc*K, b_blocks*N].
    The block-diagonal matmul needs no cross-core reduction: each core owns a distinct
    (batch-block, N-block) and produces its own [Bc, M, Nc] output block. For this initial
    implementation the output is DRAM-interleaved with shape [d0, d1, M, N] (the torch reference).
    """
    torch.manual_seed(0)
    batch = d0 * d1
    tile_height = get_tile_height(m)
    inputA_tile_size = ttnn.Tile((tile_height, 32))
    bc = batch // b_blocks
    nc = n // n_blocks
    num_inputB_cores = b_blocks * n_blocks
    print(
        f"d0: {d0}, d1: {d1}, batch: {batch}, num_inputA_cores: {num_inputA_cores}, "
        f"num_inputB_cores: {num_inputB_cores}, bc: {bc}, nc: {nc}, b_blocks: {b_blocks}, n_blocks: {n_blocks}"
    )
    if device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y < num_inputB_cores:
        pytest.skip(f"Skipping test as device doesn't have {num_inputB_cores} cores")

    torch_input_tensor_a = torch.randn((batch, m, k), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((batch, k, n), dtype=torch.bfloat16)

    # Reference: independent per-batch matmul -> [batch, m, n].
    ref = torch.matmul(torch_input_tensor_a.to(torch.float32), torch_input_tensor_b.to(torch.float32))
    m_padded = ((m + tile_height - 1) // tile_height) * tile_height

    # Fold the weights so a width-sharded tensor distributes a 2D (batch x N) block grid across
    # cores: core c (row-major) holds B[b_idx*bc:(b_idx+1)*bc, :, n_idx*nc:(n_idx+1)*nc] with
    # c = b_idx * n_blocks + n_idx. Build T[bc_i*k + kk, b_idx*n + nn] = B[b_idx*bc + bc_i, kk, nn]:
    #   [batch, k, n] -> [b_blocks, bc, k, n] -> permute -> [bc, k, b_blocks, n] -> [bc*k, b_blocks*n]
    # and pack it as rank-4 [1, 1, bc*k, b_blocks*n] (the batch is carried by the folded width).
    torch_input_tensor_b_folded = torch_input_tensor_b.reshape(b_blocks, bc, k, n)
    torch_input_tensor_b_folded = torch.permute(torch_input_tensor_b_folded, (1, 2, 0, 3))
    torch_input_tensor_b_folded = torch_input_tensor_b_folded.reshape(1, 1, bc * k, b_blocks * n)
    print("torch_input_tensor_b_folded.shape:", torch_input_tensor_b_folded.shape)

    # A is rank-4 [d0, d1, m, k]; batch = d0 * d1.
    torch_input_tensor_a_4d = torch_input_tensor_a.reshape(d0, d1, m, k)

    input_a_core_range_set = num_cores_to_rectangle_core_range_set(num_inputA_cores, device)
    # Weights: width-sharded across a (n_blocks wide x b_blocks tall) rectangle so the row-major
    # core index equals b_idx * n_blocks + n_idx.
    input_b_core_range_set = _rectangle_core_range_set(n_blocks, b_blocks, device)
    in0_memory_config = ttnn.create_sharded_memory_config(
        (batch * m_padded, k // num_inputA_cores),
        core_grid=input_a_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    in1_memory_config = ttnn.create_sharded_memory_config(
        (bc * k, nc),
        core_grid=input_b_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a_4d,
        layout=ttnn.TILE_LAYOUT,
        tile=inputA_tile_size,
        device=device,
        memory_config=in0_memory_config,
        dtype=ttnn.bfloat16,
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b_folded,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat4_b,
    )
    input_tensor_b_l1 = ttnn.to_memory_config(input_tensor_b, in1_memory_config)
    print("input_tensor_a.shape:", input_tensor_a.shape)
    print("input_tensor_b.shape:", input_tensor_b.shape)

    # Output is DRAM-interleaved with shape [d0, d1, m, n] (matches the torch reference directly;
    # b_blocks / n_blocks are inferred from the operand shapes).
    output_tensor = ttnn.experimental.matmul_decode(input_tensor_a, input_tensor_b_l1)

    assert tuple(output_tensor.shape) == (d0, d1, m, n)

    out = ttnn.to_torch(output_tensor).float().reshape(batch, m, n)
    assert_with_pcc(ref, out, 0.99)
