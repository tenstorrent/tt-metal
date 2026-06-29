# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Comprehensive tests for ttnn.pixel_unshuffle — dedicated NCHW direct-gather kernel.
#
# PyTorch reference:
#   F.pixel_unshuffle(input [N,C,H,W], r) -> [N, C*r², H/r, W/r]
#   output[n, c*r²+rh*r+rw, h', w'] = input[n, c, h'*r+rh, w'*r+rw]
#
# Kernel properties:
#   - Reads NCHW ROW_MAJOR input directly (no permute needed)
#   - Writes NCHW output with any MemoryConfig via TensorAccessor (sharded output supported)
#   - Sharded input is accepted: TensorAccessor resolves page_id across cores via NOC
#   - TILE input is untilized to ROW_MAJOR before the kernel
#   - output_layout=TILE_LAYOUT tilizes the ROW_MAJOR kernel output

import pytest
import torch
import torch.nn.functional as F
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sharded_memory_config(device, N, C, H, W, shard_strategy, n_cores_x=8, n_cores_y=1):
    """Build a HEIGHT_SHARDED or BLOCK_SHARDED MemoryConfig for an NCHW RM output."""
    total_rows = N * C * (H) * 1  # after pixel_unshuffle the spatial becomes H/r × W/r
    # For output shape [N, C*r², H/r, W/r]: pass the already-divided dims
    n_cores = n_cores_x * n_cores_y
    shard_h = max(1, (N * C * H + n_cores - 1) // n_cores)
    shard_w = W

    if shard_strategy == ttnn.ShardStrategy.HEIGHT:
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n_cores_x - 1, n_cores_y - 1))})
        shard_spec = ttnn.ShardSpec(grid, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR)
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    else:
        # BLOCK_SHARDED — split both H-sticks and W elements across 2D grid
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n_cores_x - 1, n_cores_y - 1))})
        bh = max(1, (N * C * H + n_cores_y - 1) // n_cores_y)
        bw = max(1, (W + n_cores_x - 1) // n_cores_x)
        shard_spec = ttnn.ShardSpec(grid, [bh, bw], ttnn.ShardOrientation.ROW_MAJOR)
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard_spec)


# ---------------------------------------------------------------------------
# Section A — Basic correctness: shapes × layouts × dtypes × memory configs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "N,C,H,W",
    [
        (1, 1, 1536, 1536),  # BEV Y path
        (1, 32, 48, 768),  # BEV UV path
        (1, 3, 64, 64),  # typical RGB conv
        (1, 16, 32, 32),  # C=16
        (2, 4, 32, 32),  # batch > 1
        (1, 2, 48, 768),  # non-square, r=2
        (1, 1, 32, 32),  # small, C=1
        (4, 8, 24, 24),  # larger batch, r=2 (H=24, W=24 — divisible by 2 not 4)
    ],
)
@pytest.mark.parametrize("downscale_factor", [2, 4])
@pytest.mark.parametrize("input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("input_mem_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize("output_mem_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_pixel_unshuffle_correctness(
    device, N, C, H, W, downscale_factor, input_layout, input_dtype, input_mem_config, output_mem_config
):
    """Correctness for all shape/layout/dtype/memory-config combinations."""
    r = downscale_factor

    # Skip invalid combinations
    if H % r != 0 or W % r != 0:
        pytest.skip(f"H={H} or W={W} not divisible by r={r}")
    if input_layout == ttnn.TILE_LAYOUT and (H < 32 or W < 32):
        pytest.skip("TILE layout requires H,W >= 32")
    if input_dtype == ttnn.float32 and input_layout == ttnn.TILE_LAYOUT:
        pytest.skip("float32 TILE not always supported in this environment")

    torch.manual_seed(42)
    torch_dtype = torch.float32 if input_dtype == ttnn.float32 else torch.bfloat16
    x = torch.randn(N, C, H, W, dtype=torch_dtype)
    golden = F.pixel_unshuffle(x.float(), r)

    tt_in = ttnn.from_torch(
        x,
        dtype=input_dtype,
        layout=input_layout,
        device=device,
        memory_config=input_mem_config,
    )
    tt_out = ttnn.pixel_unshuffle(tt_in, downscale_factor=r, memory_config=output_mem_config)
    result = ttnn.to_torch(ttnn.to_layout(tt_out, ttnn.ROW_MAJOR_LAYOUT))

    assert list(result.shape) == [N, C * r * r, H // r, W // r], f"Shape mismatch: got {list(result.shape)}"
    assert_with_pcc(golden, result.float(), pcc=0.99)


# ---------------------------------------------------------------------------
# Section B — Output layout: TILE output via output_layout parameter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "N,C,H,W,r",
    [
        (1, 1, 1536, 1536, 4),  # Y-path BEV: C=1 → 16 channels
        (1, 32, 48, 768, 2),  # UV-path BEV
        (1, 3, 64, 64, 2),  # general: C=3 → 12 channels
        (2, 4, 32, 32, 2),  # batch=2
    ],
)
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_pixel_unshuffle_tile_output(device, N, C, H, W, r, input_layout):
    """output_layout=TILE_LAYOUT → output tensor is tilized after the kernel."""
    torch.manual_seed(1)
    x = torch.randn(N, C, H, W, dtype=torch.bfloat16)
    golden = F.pixel_unshuffle(x, r)

    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=input_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out = ttnn.pixel_unshuffle(tt_in, downscale_factor=r, output_layout=ttnn.TILE_LAYOUT)

    assert tt_out.layout == ttnn.TILE_LAYOUT, f"Expected TILE, got {tt_out.layout}"
    result = ttnn.to_torch(ttnn.to_layout(tt_out, ttnn.ROW_MAJOR_LAYOUT))
    assert list(result.shape) == [N, C * r * r, H // r, W // r]
    assert_with_pcc(golden, result, pcc=0.99)


# ---------------------------------------------------------------------------
# Section C — Sharded output (HEIGHT_SHARDED L1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "N,C,H,W,r",
    [
        (1, 1, 32, 32, 2),  # small: output [1,4,16,16], 64 sticks
        (1, 4, 32, 32, 2),  # output [1,16,16,16], 256 sticks
        (1, 1, 64, 64, 2),  # output [1,4,32,32], 128 sticks
    ],
)
def test_pixel_unshuffle_height_sharded_output(device, N, C, H, W, r):
    """Output in HEIGHT_SHARDED L1.  Kernel writes directly to sharded output."""
    torch.manual_seed(3)
    x = torch.randn(N, C, H, W, dtype=torch.bfloat16)
    golden = F.pixel_unshuffle(x, r)

    C_out = C * r * r
    Ho = H // r
    Wo = W // r

    # Total output sticks (NCHW RM): N*C_out*Ho rows, each Wo elements wide
    total_sticks = N * C_out * Ho
    n_cores = min(total_sticks, 8)  # use up to 8 cores
    shard_h = (total_sticks + n_cores - 1) // n_cores
    shard_w = Wo

    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n_cores - 1, 0))})
    shard_spec = ttnn.ShardSpec(grid, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR)
    sharded_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out = ttnn.pixel_unshuffle(tt_in, downscale_factor=r, memory_config=sharded_cfg)

    assert tt_out.memory_config().is_sharded(), "Output should be sharded"
    result = ttnn.to_torch(tt_out)
    assert list(result.shape) == [N, C_out, Ho, Wo]
    assert_with_pcc(golden, result, pcc=0.99)


# ---------------------------------------------------------------------------
# Section D — Sharded input is handled (converted to interleaved internally)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "N,C,H,W,r",
    [
        (1, 1, 32, 32, 2),
        (1, 4, 32, 32, 2),
    ],
)
def test_pixel_unshuffle_sharded_input(device, N, C, H, W, r):
    """Sharded input tensor is accepted natively via TensorAccessor (no DRAM copy)."""
    torch.manual_seed(5)
    x = torch.randn(N, C, H, W, dtype=torch.bfloat16)
    golden = F.pixel_unshuffle(x, r)

    # Create a HEIGHT_SHARDED L1 input
    total_input_sticks = N * C * H
    n_cores = min(total_input_sticks, 4)
    shard_h = (total_input_sticks + n_cores - 1) // n_cores
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n_cores - 1, 0))})
    shard_spec = ttnn.ShardSpec(grid, [shard_h, W], ttnn.ShardOrientation.ROW_MAJOR)
    sharded_in_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_in_sharded = ttnn.to_memory_config(tt_in, sharded_in_cfg)

    # pixel_unshuffle reads sharded input natively via NOC TensorAccessor reads
    tt_out = ttnn.pixel_unshuffle(tt_in_sharded, downscale_factor=r)
    result = ttnn.to_torch(tt_out)

    assert list(result.shape) == [N, C * r * r, H // r, W // r]
    assert_with_pcc(golden, result, pcc=0.99)


# ---------------------------------------------------------------------------
# Section E — downscale_factor sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("r", [1, 2, 3, 4, 8])
def test_pixel_unshuffle_downscale_factors(device, r):
    """Various downscale factors including r=1 (identity) and r=3 (non-power-of-2)."""
    N, C, H, W = 1, 1, r * 8, r * 8  # H and W always divisible by r
    torch.manual_seed(r)
    x = torch.randn(N, C, H, W, dtype=torch.bfloat16)
    golden = F.pixel_unshuffle(x, r)

    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out = ttnn.pixel_unshuffle(tt_in, downscale_factor=r)
    result = ttnn.to_torch(tt_out)

    assert list(result.shape) == [N, C * r * r, H // r, W // r]
    assert_with_pcc(golden, result, pcc=0.99)


# ---------------------------------------------------------------------------
# Section F — dtype sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,torch_dtype",
    [
        (ttnn.bfloat16, torch.bfloat16),
        (ttnn.float32, torch.float32),
        (ttnn.uint16, torch.int16),
    ],
)
def test_pixel_unshuffle_dtypes(device, dtype, torch_dtype):
    """bfloat16, float32, uint16 dtypes — output dtype must match input dtype."""
    torch.manual_seed(10)
    if torch_dtype == torch.int16:
        # uint16 in TTNN: use non-negative values so signed/unsigned interpretation agrees
        x = torch.randint(0, 1000, (1, 4, 32, 32), dtype=torch.int16)
    else:
        x = torch.randn(1, 4, 32, 32, dtype=torch_dtype)
    golden = F.pixel_unshuffle(x.float(), 2)

    tt_in = ttnn.from_torch(
        x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out = ttnn.pixel_unshuffle(tt_in, downscale_factor=2)

    # Output dtype must be preserved — not silently downcast
    assert tt_out.dtype == dtype, f"Output dtype {tt_out.dtype} != input dtype {dtype}"
    result = ttnn.to_torch(tt_out)
    assert_with_pcc(golden, result.float(), pcc=0.99)


# ---------------------------------------------------------------------------
# Section G — output memory config: DRAM, L1 interleaved, custom
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "out_mem",
    [
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ],
)
def test_pixel_unshuffle_output_memory_configs(device, out_mem):
    """DRAM and L1-interleaved output memory configs."""
    torch.manual_seed(20)
    x = torch.randn(1, 2, 32, 32, dtype=torch.bfloat16)
    golden = F.pixel_unshuffle(x, 2)

    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out = ttnn.pixel_unshuffle(tt_in, downscale_factor=2, memory_config=out_mem)

    assert tt_out.memory_config().buffer_type == out_mem.buffer_type
    result = ttnn.to_torch(tt_out)
    assert_with_pcc(golden, result, pcc=0.99)


# ---------------------------------------------------------------------------
# Section H — combined TILE output + sharded output
# ---------------------------------------------------------------------------


def test_pixel_unshuffle_tile_output_l1(device):
    """TILE output in L1 memory: output_layout=TILE + memory_config=L1."""
    torch.manual_seed(30)
    x = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
    golden = F.pixel_unshuffle(x, 2)

    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out = ttnn.pixel_unshuffle(
        tt_in,
        downscale_factor=2,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_layout=ttnn.TILE_LAYOUT,
    )

    assert tt_out.layout == ttnn.TILE_LAYOUT
    assert tt_out.memory_config().buffer_type == ttnn.BufferType.L1
    result = ttnn.to_torch(ttnn.to_layout(tt_out, ttnn.ROW_MAJOR_LAYOUT))
    assert_with_pcc(golden, result, pcc=0.99)


# ---------------------------------------------------------------------------
# Section I — batch size sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", [1, 2, 4, 8])
def test_pixel_unshuffle_batch_sizes(device, N):
    torch.manual_seed(N)
    x = torch.randn(N, 2, 32, 32, dtype=torch.bfloat16)
    golden = F.pixel_unshuffle(x, 2)

    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out = ttnn.pixel_unshuffle(tt_in, downscale_factor=2)
    result = ttnn.to_torch(tt_out)

    assert list(result.shape) == [N, 8, 16, 16]
    assert_with_pcc(golden, result, pcc=0.99)


# ---------------------------------------------------------------------------
# Section J — large C sweep (channel-heavy inputs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("C", [1, 3, 16, 32, 64, 128])
def test_pixel_unshuffle_channel_sweep(device, C):
    r = 2
    H, W = 32, 32
    torch.manual_seed(C)
    x = torch.randn(1, C, H, W, dtype=torch.bfloat16)
    golden = F.pixel_unshuffle(x, r)

    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out = ttnn.pixel_unshuffle(tt_in, downscale_factor=r)
    result = ttnn.to_torch(tt_out)

    assert list(result.shape) == [1, C * 4, 16, 16]
    assert_with_pcc(golden, result, pcc=0.99)


# ---------------------------------------------------------------------------
# Section K — BEV model shapes (primary use case, both layouts)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,N,C,H,W,r",
    [
        ("Y_path", 1, 1, 1536, 1536, 4),  # replaces baseline R1 bottleneck (2.24 ms)
        ("UV_path", 1, 32, 48, 768, 2),  # replaces baseline R2 bottleneck (1.71 ms)
    ],
)
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.slow
def test_pixel_unshuffle_bev_shapes(device, name, N, C, H, W, r, input_layout):
    """BEV model shapes — correctness at full model scale for Y and UV paths."""
    torch.manual_seed(7)
    x = torch.randn(N, C, H, W, dtype=torch.bfloat16)
    golden = F.pixel_unshuffle(x, r)

    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=input_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out = ttnn.pixel_unshuffle(tt_in, downscale_factor=r)
    result = ttnn.to_torch(tt_out)

    assert list(result.shape) == [N, C * r * r, H // r, W // r]
    pcc_passed, pcc_msg = assert_with_pcc(golden, result, pcc=0.99)
    print(f"\n{name} [{input_layout}]: {pcc_msg}")


# ---------------------------------------------------------------------------
# Section L — program cache: same op twice (verifies cache hit validation path)
# ---------------------------------------------------------------------------


def test_pixel_unshuffle_program_cache(device):
    """Run the same kernel twice to exercise the program-cache-hit validation path."""
    x = torch.randn(1, 4, 32, 32, dtype=torch.bfloat16)
    golden = F.pixel_unshuffle(x, 2)

    for _ in range(2):
        tt_in = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tt_out = ttnn.pixel_unshuffle(tt_in, downscale_factor=2)
        result = ttnn.to_torch(tt_out)
        assert_with_pcc(golden, result, pcc=0.99)


# ---------------------------------------------------------------------------
# Section M — output dtype preservation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_pixel_unshuffle_output_dtype_preserved(device, dtype):
    """Output dtype must exactly match input dtype — no silent downcast."""
    x = torch.randn(1, 2, 32, 32, dtype=torch.float32)
    tt_in = ttnn.from_torch(
        x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out = ttnn.pixel_unshuffle(tt_in, downscale_factor=2)
    assert tt_out.dtype == dtype, f"dtype mismatch: got {tt_out.dtype}, expected {dtype}"


# ---------------------------------------------------------------------------
# Section N — error-path / validation tests
# ---------------------------------------------------------------------------


def test_pixel_unshuffle_invalid_rank(device):
    """3D input must raise — pixel_unshuffle requires exactly 4D [N,C,H,W]."""
    x = torch.randn(3, 32, 32, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    with pytest.raises(RuntimeError):
        ttnn.pixel_unshuffle(tt_in, downscale_factor=2)


def test_pixel_unshuffle_h_not_divisible(device):
    """H not divisible by r must raise."""
    x = torch.randn(1, 1, 33, 32, dtype=torch.bfloat16)  # H=33, r=2 → not divisible
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    with pytest.raises(RuntimeError):
        ttnn.pixel_unshuffle(tt_in, downscale_factor=2)


def test_pixel_unshuffle_w_not_divisible(device):
    """W not divisible by r must raise."""
    x = torch.randn(1, 1, 32, 33, dtype=torch.bfloat16)  # W=33, r=2 → not divisible
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    with pytest.raises(RuntimeError):
        ttnn.pixel_unshuffle(tt_in, downscale_factor=2)


def test_pixel_unshuffle_zero_r(device):
    """downscale_factor=0 must raise."""
    x = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    with pytest.raises((RuntimeError, ValueError)):
        ttnn.pixel_unshuffle(tt_in, downscale_factor=0)
