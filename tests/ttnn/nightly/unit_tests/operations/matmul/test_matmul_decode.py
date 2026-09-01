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
@pytest.mark.parametrize("ring_gather", [False, True])
def test_matmul_decode(device, m, k, n, num_inputA_cores, ring_gather):
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
        output_tensor = ttnn.experimental.matmul_decode(input_tensor_a, input_tensor_b, ring_gather=ring_gather)
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


@pytest.mark.parametrize("k, n", [(1024, 4096)])
@pytest.mark.parametrize("num_inputA_cores", [32])
@pytest.mark.parametrize("ring_gather", [False, True])
def test_matmul_decode_row_major_m1(device, k, n, num_inputA_cores, ring_gather):
    """ROW_MAJOR bfloat16 A with M=1 is consumed as a 1x32 tile."""
    torch.manual_seed(0)
    m = 1
    num_inputB_cores = n // 64
    if device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y < num_inputB_cores:
        pytest.skip(f"Skipping test as device doesn't have {num_inputB_cores} cores")

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
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=in0_memory_config,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in1_memory_config
    )

    output_tensor = ttnn.experimental.matmul_decode(input_tensor_a, input_tensor_b, ring_gather=ring_gather)
    assert output_tensor.shape == (m, n)
    assert output_tensor.layout == ttnn.TILE_LAYOUT
    assert output_tensor.tile == ttnn.Tile((1, 32))
    assert_with_pcc(torch_output_tensor, ttnn.to_torch(output_tensor), 0.99)


@pytest.mark.parametrize("k, n, k_blocks, n_blocks", [(4096, 1024, 2, 32)])
@pytest.mark.parametrize("num_inputA_cores", [32])
@pytest.mark.parametrize("ring_gather", [False, True])
def test_matmul_decode_partial_row_major_m1(device, k, n, k_blocks, n_blocks, num_inputA_cores, ring_gather):
    """ROW_MAJOR bfloat16 A with M=1 on the partial-width-sharded path."""
    torch.manual_seed(0)
    m = 1
    kc = k // k_blocks
    nc = n // n_blocks
    num_inputB_cores = k_blocks * n_blocks
    if device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y < num_inputB_cores:
        pytest.skip(f"Skipping test as device doesn't have {num_inputB_cores} cores")

    torch_input_tensor_a = torch.randn((m, k), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k, n), dtype=torch.bfloat16)
    ref = torch_input_tensor_a.to(torch.float32) @ torch_input_tensor_b.to(torch.float32)

    torch_input_tensor_b_reshaped = torch_input_tensor_b.reshape(k_blocks, kc, n)
    torch_input_tensor_b_reshaped = torch.permute(torch_input_tensor_b_reshaped, (1, 0, 2))
    torch_input_tensor_b_reshaped = torch_input_tensor_b_reshaped.reshape(kc, n * k_blocks)

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
        (kc, nc),
        core_grid=input_b_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        memory_config=in0_memory_config,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b_reshaped,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in1_memory_config,
        dtype=ttnn.bfloat16,
    )

    output_tensor = ttnn.experimental.matmul_decode(
        input_tensor_a, input_tensor_b, partial_width_sharded=True, ring_gather=ring_gather
    )
    assert output_tensor.shape == (m, n)
    assert output_tensor.layout == ttnn.TILE_LAYOUT
    assert output_tensor.tile == ttnn.Tile((1, 32))
    assert_with_pcc(ref, ttnn.to_torch(output_tensor), 0.99)


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
@pytest.mark.parametrize("ring_gather", [False, True])
def test_matmul_decode_partial_width_sharded(device, m, k, n, k_blocks, n_blocks, num_inputA_cores, ring_gather):
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
        input_tensor_a,
        input_tensor_b,
        partial_width_sharded=True,
        output_mem_config=output_mem_config,
        ring_gather=ring_gather,
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


def _row_major_run_core_range_set(start, count, grid_w):
    """Cores ``[start, start + count)`` of the row-major core index as a ``CoreRangeSet``.

    Built one row segment at a time, in index order, so ROW_MAJOR shard placement and
    core enumeration over the returned set both follow the run's core-index order.
    """
    ranges = []
    c = start
    end = start + count - 1
    while c <= end:
        x, y = c % grid_w, c // grid_w
        row_end = min(end, y * grid_w + grid_w - 1)
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(row_end % grid_w, y)))
        c = row_end + 1
    return ttnn.CoreRangeSet(ranges)


def _tile_stream(slab):
    """``[R, C]`` -> ``[R*C/32, 32]``: the slab's 32x32 tiles, row-major, stacked one tile wide.

    Tilizing the result reproduces the slab's tiles byte-for-byte in the order a dedicated
    width-sharded weight shard stores them, which is the order matmul_decode consumes.
    """
    r, c = slab.shape
    rt, ct = r // 32, c // 32
    return slab.reshape(rt, 32, ct, 32).permute(0, 2, 1, 3).reshape(rt * ct * 32, 32)


def _core_slab(w, pos, k_blocks, n_blocks):
    """The [Kc, Nc] block that zone-relative core ``pos`` holds: core = kb * n_blocks + nb."""
    K, N = w.shape
    kc, nc = K // k_blocks, N // n_blocks
    kb, nb = pos // n_blocks, pos % n_blocks
    return w[kb * kc : (kb + 1) * kc, nb * nc : (nb + 1) * nc]


@pytest.mark.parametrize("m", [32])
def test_matmul_decode_packed_weights(device, m):
    """Four DeepSeek-V4-Flash decode matmuls served out of ONE fused L1 weight tensor.

    The model keeps its resident decode weights in a single bfloat4_b tensor, HEIGHT sharded
    across the chip with one equal-sized, one-tile-wide shard per core (see
    models/experimental/deepseek_v4_flash/tt/l1_weights.py). Every core's shard is the
    concatenation of the tile streams of the weight slabs that core owns, zero-padded so all
    shards match; each matmul then receives the fused tensor plus a
    ``MatmulDecodePackedWeightSpec`` locating its region (tile offset, [K, N], cores, cut).

    This test packs four of the model's projections into two zones of a 48-core pack:

        zone A (cores 0-31):  q_a_proj [4096, 1024]  partial k_blocks=2, n_blocks=16  @ tile 0
                              q_b_proj [1024, 32768] full n_blocks=32                 @ tile 128
        zone B (cores 32-47): kv_proj  [4096, 512]   full n_blocks=16                 @ tile 0
                              shared_gate_proj [4096, 2048] full n_blocks=16          @ tile 128

    Zone A fills its 1152-tile shard exactly; zone B uses 640 tiles and is zero-padded, so the
    equal-shard padding path is exercised too. Each matmul runs with ``packed_weight`` and is
    verified against torch. The activation grid moves per matmul (row-major runs starting at
    cores 8, 16, 24, 32) so nothing assumes the input is anchored at (0, 0) or shared between
    the packed matmuls.
    """
    torch.manual_seed(0)
    grid = device.compute_with_storage_grid_size()
    grid_w = grid.x

    # name: (K, N, k_blocks, n_blocks, zone) -- zone: (first row-major core, core count)
    ZONES = {"A": (0, 32), "B": (32, 16)}
    WEIGHTS = {
        "q_a_proj": (4096, 1024, 2, 16, "A"),
        "q_b_proj": (1024, 32768, 1, 32, "A"),
        "kv_proj": (4096, 512, 1, 16, "B"),
        "shared_gate_proj": (4096, 2048, 1, 16, "B"),
    }
    num_pack_cores = sum(count for _, count in ZONES.values())
    num_inputA_cores = 32
    # Each matmul's activation lives on its own 32-core run, starting at 8, 16, 24, 32.
    inputA_starts = {name: 8 * (i + 1) for i, name in enumerate(WEIGHTS)}
    num_cores_needed = max(num_pack_cores, max(inputA_starts.values()) + num_inputA_cores)
    if grid.x * grid.y < num_cores_needed:
        pytest.skip(f"Skipping test as device doesn't have {num_cores_needed} cores")

    torch_weights = {name: torch.randn((w[0], w[1]), dtype=torch.bfloat16).float() for name, w in WEIGHTS.items()}

    # ---- pack: per-zone tile offsets, then one [num_pack_cores * shard_rows, 32] host tensor ----
    tile_offsets = {}
    zone_fill = {zone: 0 for zone in ZONES}
    for name, (K, N, k_blocks, n_blocks, zone) in WEIGHTS.items():
        slab_tiles = (K // k_blocks // 32) * (N // n_blocks // 32)
        tile_offsets[name] = zone_fill[zone]
        zone_fill[zone] += slab_tiles
    shard_tiles = max(zone_fill.values())
    shard_rows = shard_tiles * 32
    print(f"pack: {num_pack_cores} cores, shard {shard_tiles} tiles, zone fill {zone_fill}, offsets {tile_offsets}")

    fused = torch.zeros((num_pack_cores * shard_rows, 32), dtype=torch.float32)
    for name, (K, N, k_blocks, n_blocks, zone) in WEIGHTS.items():
        start, count = ZONES[zone]
        assert k_blocks * n_blocks == count
        for pos in range(count):
            stream = _tile_stream(_core_slab(torch_weights[name], pos, k_blocks, n_blocks))
            row0 = (start + pos) * shard_rows + tile_offsets[name] * 32
            fused[row0 : row0 + stream.shape[0]] = stream

    pack_core_range_set = _row_major_run_core_range_set(0, num_pack_cores, grid_w)
    fused_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(pack_core_range_set, [shard_rows, 32], ttnn.ShardOrientation.ROW_MAJOR),
    )
    fused_tensor = ttnn.from_torch(
        fused,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=fused_memory_config,
    )

    # ---- run every packed matmul against the same fused tensor ----
    for name, (K, N, k_blocks, n_blocks, zone) in WEIGHTS.items():
        start, count = ZONES[zone]
        spec = ttnn._ttnn.operations.experimental.MatmulDecodePackedWeightSpec(
            tile_offset=tile_offsets[name],
            K=K,
            N=N,
            cores=_row_major_run_core_range_set(start, count, grid_w),
            k_blocks=k_blocks,
        )

        torch_input = torch.randn((m, K), dtype=torch.bfloat16)
        ref = torch_input.float() @ torch_weights[name]
        input_a_core_range_set = _row_major_run_core_range_set(inputA_starts[name], num_inputA_cores, grid_w)
        in0_memory_config = ttnn.create_sharded_memory_config(
            (m, K // num_inputA_cores),
            core_grid=input_a_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        input_tensor_a = ttnn.from_torch(
            torch_input,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=in0_memory_config,
            dtype=ttnn.bfloat16,
        )

        output_tensor = ttnn.experimental.matmul_decode(input_tensor_a, fused_tensor, packed_weight=spec)

        assert output_tensor.shape == (m, N)
        out = ttnn.to_torch(output_tensor).float()
        assert_with_pcc(ref, out, 0.99)
        print(
            f"{name}: [{K}, {N}] @ tile {tile_offsets[name]} on {count} cores, "
            f"in0 on cores {inputA_starts[name]}-{inputA_starts[name] + num_inputA_cores - 1} ok"
        )


def test_matmul_decode_deepseek_layer_packed_weights(device):
    """All 13 non-expert linear matmuls in one DeepSeek-V4 CSA layer, packed into one BF4 L1 tensor.

    This mirrors the one-layer subset of deepseek_v4_flash/tt/l1_placement.py:

      Z0 cores 0-63:    q_a, q_b, compressor.gate, o_b
      Z1 cores 64-95:   compressor.kv, grouped o_a, attn_hc.fn, ffn_hc.fn
      Z2 cores 96-111:  kv, shared gate, shared up
      Z3 cores 112-119: shared down, router gate

    Every core receives one equally sized shard. Each zone concatenates its local weight slabs
    and pads to the largest zone (1184 BF4 tiles/core). Inputs use varying, non-zero 32-core
    ranges. The test covers full, partial-K, and grouped-batch packed matmul_decode paths.
    """
    torch.manual_seed(0)
    m = 1
    input_tile = ttnn.Tile((get_tile_height(m), 32))
    num_cores = 120
    num_input_cores = 32
    grid = device.compute_with_storage_grid_size()
    grid_w = grid.x
    if grid.x * grid.y < num_cores:
        pytest.skip("Complete DeepSeek layer placement requires 120 worker cores")

    zones = {
        "Z0": (0, 64),
        "Z1": (64, 32),
        "Z2": (96, 16),
        "Z3": (112, 8),
    }
    # name, K, N, zone, k_blocks, batch, b_blocks
    weights = [
        ("q_a_proj", 4096, 1024, "Z0", 2, 1, 1),
        ("q_b_proj", 1024, 32768, "Z0", 1, 1, 1),
        ("kv_proj", 4096, 512, "Z2", 1, 1, 1),
        ("compressor.gate_proj", 4096, 1024, "Z0", 2, 1, 1),
        ("compressor.kv_proj", 4096, 1024, "Z1", 1, 1, 1),
        ("o_a_proj", 4096, 1024, "Z1", 1, 8, 8),
        ("o_b_proj", 8192, 4096, "Z0", 1, 1, 1),
        ("shared_gate_proj", 4096, 2048, "Z2", 1, 1, 1),
        ("shared_up_proj", 4096, 2048, "Z2", 1, 1, 1),
        ("shared_down_proj", 2048, 4096, "Z3", 1, 1, 1),
        ("router_gate", 4096, 256, "Z3", 1, 1, 1),
        ("attn_hc.fn", 16384, 32, "Z1", 32, 1, 1),
        ("ffn_hc.fn", 16384, 32, "Z1", 32, 1, 1),
    ]

    # Keep source weights in bf16 to limit host memory. o_a is [batch, K, N];
    # all other weights are ordinary [K, N].
    torch_weights = {
        name: torch.randn((batch, K, N) if batch > 1 else (K, N), dtype=torch.bfloat16)
        for name, K, N, _, _, batch, _ in weights
    }

    tile_offsets = {}
    zone_fill = {zone: 0 for zone in zones}
    for name, K, N, zone, k_blocks, batch, b_blocks in weights:
        count = zones[zone][1]
        n_blocks = count // (b_blocks if batch > 1 else k_blocks)
        slab_rows = (batch // b_blocks) * K if batch > 1 else K // k_blocks
        slab_tiles = (slab_rows // 32) * (N // n_blocks // 32)
        tile_offsets[name] = zone_fill[zone]
        zone_fill[zone] += slab_tiles

    shard_tiles = max(zone_fill.values())
    shard_rows = shard_tiles * 32
    assert zone_fill == {"Z0": 1152, "Z1": 1184, "Z2": 1152, "Z3": 1152}
    fused = torch.zeros((num_cores * shard_rows, 32), dtype=torch.bfloat16)

    for name, K, N, zone, k_blocks, batch, b_blocks in weights:
        start, count = zones[zone]
        n_blocks = count // (b_blocks if batch > 1 else k_blocks)
        weight = torch_weights[name]
        for pos in range(count):
            nb = pos % n_blocks
            nc = N // n_blocks
            if batch > 1:
                bb = pos // n_blocks
                bc = batch // b_blocks
                slab = weight[bb * bc : (bb + 1) * bc, :, nb * nc : (nb + 1) * nc].reshape(bc * K, nc)
            else:
                kb = pos // n_blocks
                kc = K // k_blocks
                slab = weight[kb * kc : (kb + 1) * kc, nb * nc : (nb + 1) * nc]
            stream = _tile_stream(slab)
            row0 = (start + pos) * shard_rows + tile_offsets[name] * 32
            fused[row0 : row0 + stream.shape[0]] = stream

    all_cores = _row_major_run_core_range_set(0, num_cores, grid_w)
    fused_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(all_cores, [shard_rows, 32], ttnn.ShardOrientation.ROW_MAJOR),
    )
    fused_tensor = ttnn.from_torch(
        fused,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=fused_memory_config,
    )
    del fused

    packed_spec_type = ttnn._ttnn.operations.experimental.MatmulDecodePackedWeightSpec
    for index, (name, K, N, zone, k_blocks, batch, b_blocks) in enumerate(weights):
        start, count = zones[zone]
        input_start = 4 * (index + 1)
        input_cores = _row_major_run_core_range_set(input_start, num_input_cores, grid_w)
        spec = packed_spec_type(
            tile_offset=tile_offsets[name],
            K=K,
            N=N,
            cores=_row_major_run_core_range_set(start, count, grid_w),
            k_blocks=k_blocks,
            batch=batch,
            b_blocks=b_blocks,
        )

        if batch > 1:
            torch_input = torch.randn((1, batch, m, K), dtype=torch.bfloat16)
            ref = torch.matmul(torch_input.reshape(batch, m, K).float(), torch_weights[name].float())
            input_shape = (batch * m, K // num_input_cores)
        else:
            torch_input = torch.randn((m, K), dtype=torch.bfloat16)
            ref = torch_input.float() @ torch_weights[name].float()
            input_shape = (m, K // num_input_cores)

        input_memory_config = ttnn.create_sharded_memory_config(
            input_shape,
            core_grid=input_cores,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        input_tensor = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            tile=input_tile,
            device=device,
            memory_config=input_memory_config,
        )
        output = ttnn.experimental.matmul_decode(input_tensor, fused_tensor, packed_weight=spec)
        actual = ttnn.to_torch(output).float()
        if batch > 1:
            actual = actual.reshape(batch, m, N)
            assert tuple(output.shape) == (1, batch, m, N)
        else:
            assert tuple(output.shape) == (m, N)
        assert_with_pcc(ref, actual, 0.99)
        print(
            f"{name}: [{K}, {N}], {zone} @ tile {tile_offsets[name]}, "
            f"in0 cores {input_start}-{input_start + num_input_cores - 1} ok"
        )


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


# Unique (K, N, layout) used by deepseek_v4_flash decode matmuls. Duplicates are
# omitted: CSA compressor.gate == q_a_proj, HCA compressor.gate == kv_proj,
# shared_up_proj == shared_gate_proj, attn_hc.fn == ffn_hc.fn. Packed-L1 cuts that
# keep the same (K, N) as DECODE_LAYOUTS (e.g. full-width kv / shared_gate) are
# not listed separately; router_gate and hc.fn exist only on the packed path.
# See decode_prefetch.DECODE_LAYOUTS and l1_placement._COMMON / _COMPRESSOR.
_DEEPSEEK_V4_FLASH_MATMUL_DECODE_SHAPES = [
    pytest.param(1, 4096, 1024, 2, 32, 1, 1, id="q_a_proj"),
    pytest.param(1, 1024, 32768, 1, 64, 1, 1, id="q_b_proj"),
    pytest.param(1, 4096, 512, 4, 16, 1, 1, id="kv_proj"),
    pytest.param(1, 4096, 1536, 4, 16, 1, 1, id="qa_kv_proj"),
    pytest.param(1, 8192, 4096, 1, 64, 1, 1, id="o_b_proj"),
    pytest.param(1, 4096, 1024, 1, 8, 8, 8, id="o_a_proj"),
    pytest.param(1, 4096, 2048, 2, 32, 1, 1, id="shared_gate_up"),
    pytest.param(1, 2048, 4096, 1, 64, 1, 1, id="shared_down"),
    pytest.param(1, 4096, 256, 1, 8, 1, 1, id="router_gate"),
    pytest.param(1, 16384, 32, 32, 1, 1, 1, id="hc_fn"),
]

# Column-parallel TP4 of the same projections: local N = TP1 N / 4, then
# all-gather restores N_full so a profiler capture can compare TP1 compute
# against TP4 compute + collective. n_blocks follows LinearDecode's 64-col
# shards (capped at 64 cores for q_b). Balanced q_a/kv keep the galaxy32
# partial-K layout (64 B cores: q_a 8x8, kv 16x4); fused all-gather only covers full-width.
# Omitted: o_a (batched factory has no all-gather), hc.fn (N=32 is one tile).
_TP4 = 4
_DEEPSEEK_V4_FLASH_MATMUL_DECODE_TP4_SHAPES = [
    pytest.param(1, 4096, 256, 8, 8, 1, 1, id="q_a_proj_tp4"),
    pytest.param(1, 1024, 8192, 1, 64, 1, 1, id="q_b_proj_tp4"),
    pytest.param(1, 4096, 128, 16, 4, 1, 1, id="kv_proj_tp4"),
    pytest.param(1, 4096, 384, 1, 6, 1, 1, id="qa_kv_proj_tp4"),
    pytest.param(1, 8192, 1024, 1, 16, 1, 1, id="o_b_proj_tp4"),
    pytest.param(1, 4096, 512, 1, 8, 1, 1, id="shared_gate_up_tp4"),
    pytest.param(1, 2048, 1024, 1, 16, 1, 1, id="shared_down_tp4"),
    pytest.param(1, 4096, 64, 1, 1, 1, 1, id="router_gate_tp4"),
]


@pytest.mark.parametrize("m, k, n, k_blocks, n_blocks, batch, b_blocks", _DEEPSEEK_V4_FLASH_MATMUL_DECODE_SHAPES)
def test_matmul_decode_deepseek_v4_flash_shapes(device, m, k, n, k_blocks, n_blocks, batch, b_blocks, request):
    """Single-device matmul_decode on every unique DeepSeek-V4-Flash decode shape.

    Activations are bf16 with the decode tile height; weights are bf4, matching the
    model. Full-width, partial-K, and batched (o_a) layouts each follow the same
    sharding the corresponding LinearDecode / BatchedLinearDecode path uses.
    """
    torch.manual_seed(0)
    num_inputA_cores = 32
    if k % (num_inputA_cores * 32) != 0:
        num_inputA_cores = k // 32
        assert k % (num_inputA_cores * 32) == 0

    tile_height = get_tile_height(m)
    inputA_tile_size = ttnn.Tile((tile_height, 32))
    m_padded = ((m + tile_height - 1) // tile_height) * tile_height
    is_batched = batch > 1
    is_partial = (not is_batched) and k_blocks > 1

    if is_batched:
        num_inputB_cores = b_blocks * n_blocks
        bc, nc = batch // b_blocks, n // n_blocks
    elif is_partial:
        num_inputB_cores = k_blocks * n_blocks
        kc, nc = k // k_blocks, n // n_blocks
    else:
        num_inputB_cores = n_blocks
        nc = n // n_blocks

    grid = device.compute_with_storage_grid_size()
    if grid.x * grid.y < max(num_inputA_cores, num_inputB_cores):
        pytest.skip(
            f"Skipping test as device grid {grid.x}x{grid.y} is smaller than "
            f"{max(num_inputA_cores, num_inputB_cores)} cores"
        )

    input_a_core_range_set = num_cores_to_rectangle_core_range_set(num_inputA_cores, device)
    if is_batched:
        try:
            input_b_core_range_set = _rectangle_core_range_set(n_blocks, b_blocks, device)
        except ValueError as exc:
            pytest.skip(str(exc))
        torch_a = torch.randn((batch, m, k), dtype=torch.bfloat16)
        torch_b = torch.randn((batch, k, n), dtype=torch.bfloat16)
        ref = torch.matmul(torch_a.to(torch.float32), torch_b.to(torch.float32))
        torch_a_device = torch_a.reshape(1, batch, m, k)
        torch_b_device = torch_b.reshape(b_blocks, bc, k, n).permute(1, 2, 0, 3).reshape(1, 1, bc * k, b_blocks * n)
        in0_shard_shape = (batch * m_padded, k // num_inputA_cores)
        in1_shard_shape = (bc * k, nc)
        expected_shape = (1, batch, m, n)
    else:
        input_b_core_range_set = num_cores_to_rectangle_core_range_set(num_inputB_cores, device)
        torch_a = torch.randn((m, k), dtype=torch.bfloat16)
        torch_b = torch.randn((k, n), dtype=torch.bfloat16)
        ref = torch_a.to(torch.float32) @ torch_b.to(torch.float32)
        torch_a_device = torch_a
        if is_partial:
            torch_b_device = torch.permute(torch_b.reshape(k_blocks, kc, n), (1, 0, 2)).reshape(kc, n * k_blocks)
            in1_shard_shape = (kc, nc)
        else:
            torch_b_device = torch_b
            in1_shard_shape = (k, nc)
        in0_shard_shape = (m_padded, k // num_inputA_cores)
        expected_shape = (m, n)

    in0_memory_config = ttnn.create_sharded_memory_config(
        in0_shard_shape,
        core_grid=input_a_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    in1_memory_config = ttnn.create_sharded_memory_config(
        in1_shard_shape,
        core_grid=input_b_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    input_tensor_a = ttnn.from_torch(
        torch_a_device,
        layout=ttnn.TILE_LAYOUT,
        tile=inputA_tile_size,
        device=device,
        memory_config=in0_memory_config,
        dtype=ttnn.bfloat16,
    )
    # DRAM then L1: bf4 tilize into a sharded L1 config can fail to allocate on the
    # largest decode weights (q_b / o_b) if the host path stages a second copy.
    input_tensor_b = ttnn.to_memory_config(
        ttnn.from_torch(
            torch_b_device,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat4_b,
        ),
        in1_memory_config,
    )

    decode_kwargs = {}
    if is_partial:
        output_core_range_set = num_cores_to_rectangle_core_range_set(n_blocks, device)
        decode_kwargs["partial_width_sharded"] = True
        decode_kwargs["output_mem_config"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(output_core_range_set, [m_padded, n // n_blocks], ttnn.ShardOrientation.ROW_MAJOR),
        )
    elif not is_batched:
        decode_kwargs["output_mem_config"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(input_b_core_range_set, [m_padded, nc], ttnn.ShardOrientation.ROW_MAJOR),
        )

    # Twice so a profiler trace captures a warm run; signposts split the
    # parametrized cases when they share one capture. Use idlist[0] (the
    # pytest.param id) — the full callspec.id also includes auto-generated
    # mesh/fabric fragments that Tracy truncates.
    signpost(request.node.callspec._idlist[0])
    for _ in range(2):
        output_tensor = ttnn.experimental.matmul_decode(input_tensor_a, input_tensor_b, **decode_kwargs)
    signpost("stop")
    assert tuple(output_tensor.shape) == expected_shape
    out = ttnn.to_torch(output_tensor).float()
    if is_batched:
        out = out.reshape(batch, m, n)
    assert_with_pcc(ref, out, 0.988)


def _1x4_line_submesh(mesh_device):
    """Carve a 1x4 ethernet line out of the opened 8x4 galaxy mesh."""
    return mesh_device.create_submesh(ttnn.MeshShape(1, 4))


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("m, k, n, k_blocks, n_blocks, batch, b_blocks", _DEEPSEEK_V4_FLASH_MATMUL_DECODE_TP4_SHAPES)
def test_matmul_decode_deepseek_v4_flash_shapes_tp4(mesh_device, m, k, n, k_blocks, n_blocks, batch, b_blocks, request):
    """TP4 column-parallel matmul_decode + all-gather on DeepSeek-V4-Flash shapes.

    Each rank owns N columns of the TP1 weight; a fused (full-width) or explicit
    (partial-K) all-gather concatenates them back to N_full. Same dtypes and
    signposts as ``test_matmul_decode_deepseek_v4_flash_shapes`` so a profiler
    capture can compare TP1 vs TP4.
    """
    torch.manual_seed(0)
    assert batch == 1 and b_blocks == 1, "TP4 all-gather path is not batched"
    mesh_device = _1x4_line_submesh(mesh_device)
    ring_size = mesh_device.get_num_devices()
    assert ring_size == _TP4
    n_local = n
    n_full = n_local * ring_size
    is_partial = k_blocks > 1
    fused_all_gather = not is_partial

    num_inputA_cores = 32
    if k % (num_inputA_cores * 32) != 0:
        num_inputA_cores = k // 32
        assert k % (num_inputA_cores * 32) == 0

    tile_height = get_tile_height(m)
    inputA_tile_size = ttnn.Tile((tile_height, 32))
    m_padded = ((m + tile_height - 1) // tile_height) * tile_height

    if is_partial:
        num_inputB_cores = k_blocks * n_blocks
        kc, nc = k // k_blocks, n_local // n_blocks
        output_num_cores = n_blocks
    else:
        num_inputB_cores = n_blocks
        nc = n_local // n_blocks
        output_num_cores = n_blocks

    grid = mesh_device.compute_with_storage_grid_size()
    if grid.x * grid.y < max(num_inputA_cores, num_inputB_cores):
        pytest.skip(
            f"Skipping test as device grid {grid.x}x{grid.y} is smaller than "
            f"{max(num_inputA_cores, num_inputB_cores)} cores"
        )

    torch_a = torch.randn((m, k), dtype=torch.bfloat16)
    torch_b_full = torch.randn((k, n_full), dtype=torch.bfloat16)
    ref = torch_a.to(torch.float32) @ torch_b_full.to(torch.float32)

    input_a_core_range_set = num_cores_to_rectangle_core_range_set(num_inputA_cores, mesh_device)
    input_b_core_range_set = num_cores_to_rectangle_core_range_set(num_inputB_cores, mesh_device)
    output_core_range_set = num_cores_to_rectangle_core_range_set(output_num_cores, mesh_device)

    if is_partial:
        torch_b_device = (
            torch_b_full.reshape(k_blocks, kc, ring_size, n_local)
            .permute(1, 2, 0, 3)
            .reshape(kc, ring_size * k_blocks * n_local)
        )
        in1_shard_shape = (kc, nc)
    else:
        torch_b_device = torch_b_full
        in1_shard_shape = (k, nc)

    in0_memory_config = ttnn.create_sharded_memory_config(
        (m_padded, k // num_inputA_cores),
        core_grid=input_a_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    in1_memory_config = ttnn.create_sharded_memory_config(
        in1_shard_shape,
        core_grid=input_b_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    input_tensor_a = ttnn.from_torch(
        torch_a,
        layout=ttnn.TILE_LAYOUT,
        tile=inputA_tile_size,
        device=mesh_device,
        memory_config=in0_memory_config,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    input_tensor_b = ttnn.to_memory_config(
        ttnn.from_torch(
            torch_b_device,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat4_b,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        ),
        in1_memory_config,
    )

    decode_kwargs = {"all_gather": fused_all_gather}
    if is_partial:
        decode_kwargs["partial_width_sharded"] = True
        decode_kwargs["output_mem_config"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(output_core_range_set, [m_padded, nc], ttnn.ShardOrientation.ROW_MAJOR),
        )

    signpost(request.node.callspec._idlist[0])
    for _ in range(2):
        output_tensor = ttnn.experimental.matmul_decode(input_tensor_a, input_tensor_b, **decode_kwargs)
        if not fused_all_gather:
            all_gather_memory_config = ttnn.create_sharded_memory_config(
                (m_padded, n_full // output_num_cores),
                core_grid=output_core_range_set,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            output_tensor = ttnn.all_gather(
                output_tensor,
                dim=-1,
                memory_config=all_gather_memory_config,
            )
    signpost("stop")

    assert tuple(output_tensor.shape)[-2:] == (m, n_full)
    device_outputs = ttnn.get_device_tensors(output_tensor)
    assert len(device_outputs) == ring_size
    for device_output in device_outputs:
        actual = ttnn.to_torch(device_output).float()
        assert_with_pcc(ref, actual, 0.988)


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("fused_all_gather", [True, False], ids=["fused", "explicit"])
@pytest.mark.parametrize(
    "m,k,n_full,num_inputA_cores,num_inputB_cores",
    [
        # Original mux/remapping regression with two output tiles per compute core.
        pytest.param(32, 1024, 16384, 4, 64, id="mux-remap-m32"),
        # DeepSeek-V4-Flash decode projections. N is the post-all-gather global width;
        # every device owns N/4 columns before the fused collective. Use the model's
        # usual 64-column B shards where the device core count permits it.
        pytest.param(1, 4096, 1024, 32, 4, id="deepseek-q-a-compressor"),
        pytest.param(1, 1024, 32768, 32, 64, id="deepseek-q-b"),
        pytest.param(1, 4096, 512, 32, 2, id="deepseek-kv-hca"),
        pytest.param(1, 4096, 1536, 32, 6, id="deepseek-fused-qa-kv"),
        pytest.param(1, 8192, 4096, 32, 16, id="deepseek-o-b"),
        pytest.param(1, 4096, 2048, 32, 8, id="deepseek-shared-gate-up"),
        pytest.param(1, 2048, 4096, 32, 16, id="deepseek-shared-down"),
        pytest.param(1, 4096, 256, 32, 1, id="deepseek-router-gate-direct"),
    ],
)
def test_matmul_decode_all_gather_full_width(
    mesh_device, fused_all_gather, m, k, n_full, num_inputA_cores, num_inputB_cores
):
    """Compare fused and explicit all-gather on DeepSeek-V4-Flash decode shapes."""
    torch.manual_seed(0)
    mesh_device = _1x4_line_submesh(mesh_device)

    ring_size = mesh_device.get_num_devices()
    assert n_full % ring_size == 0
    n_local = n_full // ring_size
    assert k % (num_inputA_cores * 32) == 0
    assert n_local % (num_inputB_cores * 32) == 0
    grid = mesh_device.compute_with_storage_grid_size()
    if grid.x * grid.y < max(num_inputA_cores, num_inputB_cores):
        pytest.skip(f"device grid {grid.x}x{grid.y} is too small")

    torch_a = torch.randn((m, k), dtype=torch.bfloat16)
    torch_b_full = torch.randn((k, n_full), dtype=torch.bfloat16)
    ref = torch_a.to(torch.float32) @ torch_b_full.to(torch.float32)

    input_a_core_range_set = num_cores_to_rectangle_core_range_set(num_inputA_cores, mesh_device)
    input_b_core_range_set = num_cores_to_rectangle_core_range_set(num_inputB_cores, mesh_device)
    in0_memory_config = ttnn.create_sharded_memory_config(
        (m, k // num_inputA_cores),
        core_grid=input_a_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    in1_memory_config = ttnn.create_sharded_memory_config(
        (k, n_local // num_inputB_cores),
        core_grid=input_b_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    input_tensor_a = ttnn.from_torch(
        torch_a,
        layout=ttnn.TILE_LAYOUT,
        tile=ttnn.Tile((get_tile_height(m), 32)),
        device=mesh_device,
        memory_config=in0_memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    input_tensor_b = ttnn.from_torch(
        torch_b_full,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=in1_memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
    )

    output_tensor = ttnn.experimental.matmul_decode(input_tensor_a, input_tensor_b, all_gather=fused_all_gather)
    if not fused_all_gather:
        all_gather_memory_config = ttnn.create_sharded_memory_config(
            (get_tile_height(m), n_full // num_inputB_cores),
            core_grid=input_b_core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        output_tensor = ttnn.all_gather(
            output_tensor,
            dim=-1,
            memory_config=all_gather_memory_config,
        )
    assert tuple(output_tensor.shape)[-2:] == (m, n_full)

    device_outputs = ttnn.get_device_tensors(output_tensor)
    assert len(device_outputs) == ring_size
    for device_output in device_outputs:
        actual = ttnn.to_torch(device_output).float()
        assert_with_pcc(ref, actual, 0.99)
