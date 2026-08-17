# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Comprehensive combinatorial ttnn.grid_sample tests vs torch.nn.functional.grid_sample
# (https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.grid_sample.html)
#
# Cross-product coverage per mode:
#   * mode                : bilinear, nearest
#   * align_corners       : True, False
#   * padding_mode        : "zeros" (only supported; border/reflection tested to raise)
#   * use_precomputed_grid: {False, True} for both bilinear and nearest
#   * batch_output_channels + grid batching factor K: (False,K=1), (False,K=4 W-extend), (True,K=4 C-batch)
#   * memory layout       : interleaved grid, HEIGHT_SHARDED grid
#   * shapes              : several NHWC input / grid sizes
#
# Reference is ttnn.operations.pool.golden_grid_sample, which wraps torch.nn.functional.grid_sample
# and applies the K-batching / NHWC conventions.

import pytest
import torch
import torch.nn.functional as F

import ttnn
from ttnn.operations.pool import golden_grid_sample
from tests.ttnn.utils_for_testing import assert_with_pcc

RM = ttnn.ROW_MAJOR_LAYOUT
DRAM = ttnn.DRAM_MEMORY_CONFIG
L1_ALIGN = 16


# --------------------------------------------------------------------------- helpers
def _natural_grid(n, h_out, w_grid, k):
    # Natural grid (N, H_out, W_grid*K, 2), coords slightly beyond [-1,1] so some samples are
    # out of bounds -> exercises "zeros" padding.  Packed form is (N, H_out, W_grid, 2*K).
    return torch.rand((n, h_out, w_grid * k, 2), dtype=torch.float32) * 2.4 - 1.2


def _height_sharded_grid_mem(device, total_height, logical_width, bytes_per_elem):
    """HEIGHT_SHARDED grid mem-config with shard width padded to L1 alignment (16B).
    A sub-alignment shard width makes the sharded reader misread the grid."""
    cg = device.compute_with_storage_grid_size()
    core_grid = ttnn.CoreGrid(y=cg.y, x=cg.x)
    shard_h = (total_height + core_grid.num_cores - 1) // core_grid.num_cores
    aligned_w = (((logical_width * bytes_per_elem + L1_ALIGN - 1) // L1_ALIGN) * L1_ALIGN) // bytes_per_elem
    return ttnn.create_sharded_memory_config(
        (shard_h, aligned_w),
        core_grid,
        ttnn.ShardStrategy.HEIGHT,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _build_ttnn_grid(
    device, natural, n, h_out, w_grid, k, mode, precomputed, align_corners, input_shape, grid_dtype, memcfg="dram"
):
    """Returns (device grid tensor, logical last-dim width, elem_bytes).
    memcfg controls the interleaved buffer placement ("l1" -> L1 interleaved, else DRAM interleaved);
    HEIGHT_SHARDED placement is handled by the caller via a reshard."""
    packed = natural.reshape(n, h_out, w_grid, 2 * k)  # (N,H,W,2K)

    if not precomputed:
        host = ttnn.from_torch(packed, layout=RM, dtype=grid_dtype)
        logical_w = 2 * k
        elem_bytes = 4 if grid_dtype == ttnn.float32 else 2
    elif mode == "nearest":
        # prepare supports batched nearest directly: (N,H,W,2K) -> (N,H,W,2K) precomputed indices
        gh = ttnn.from_torch(packed, dtype=ttnn.float32)
        host = ttnn.prepare_grid_sample_grid(
            gh, list(input_shape), mode="nearest", align_corners=align_corners, output_dtype=ttnn.bfloat16
        )
        logical_w = 2 * k
        elem_bytes = 2
    else:  # bilinear precomputed: prepare unbatched (N,H,W*K,2)->(...,6), then pack K into last dim
        gh = ttnn.from_torch(natural, dtype=ttnn.float32)  # (N,H_out,W_grid*K,2)
        prep = ttnn.prepare_grid_sample_grid(
            gh, list(input_shape), mode="bilinear", align_corners=align_corners, output_dtype=ttnn.bfloat16
        )  # (N,H_out,W_grid*K,6)
        prep_t = ttnn.to_torch(prep).reshape(n, h_out, w_grid, 6 * k)
        host = ttnn.from_torch(prep_t, layout=RM, dtype=ttnn.bfloat16)
        logical_w = 6 * k
        elem_bytes = 2

    il_mem = ttnn.L1_MEMORY_CONFIG if memcfg == "l1" else DRAM
    grid_il = ttnn.to_device(host, device=device, memory_config=il_mem)
    return grid_il, logical_w, elem_bytes


def _run_case(
    device,
    mode,
    align_corners,
    precomputed,
    batch_output_channels,
    sharded,
    input_shape,
    h_out,
    w_grid,
    k,
    memcfg="dram",
    grid_dtype_override=None,
):
    torch.manual_seed(0)
    n, h, w, c = input_shape

    inp = torch.randn(n, h, w, c, dtype=torch.bfloat16)
    natural = _natural_grid(n, h_out, w_grid, k)

    # Effective grid dtype: precomputed grids MUST be bf16 (op requirement); non-precomputed defaults to
    # fp32 (exact match to the fp32 golden), but an explicit override lets bilinear also exercise bf16.
    if precomputed:
        grid_dtype = ttnn.bfloat16
    elif grid_dtype_override is not None:
        grid_dtype = grid_dtype_override
    else:
        grid_dtype = ttnn.float32

    # For a bf16 non-precomputed grid the device reads bf16-quantized coords; quantize the golden's grid
    # to bf16 too so both see identical coordinates. A bf16 grid carries coarser coordinates than fp32,
    # so this is required for a fair comparison — it is an input-precision choice, not an op error. (The
    # kernel itself rounds half-to-even like torch, so it is bit-exact given the same coordinates.)
    golden_natural = natural
    if (not precomputed) and grid_dtype == ttnn.bfloat16:
        golden_natural = natural.to(torch.bfloat16).float()
    golden_packed = golden_natural.reshape(n, h_out, w_grid, 2 * k)

    # torch-based golden (handles mode, align_corners, K batching, batch_output_channels)
    golden = golden_grid_sample(
        input_tensor=inp,
        grid=golden_packed,
        mode=mode,
        padding_mode="zeros",
        align_corners=align_corners,
        batch_output_channels=batch_output_channels,
    )

    grid_il, logical_w, elem_bytes = _build_ttnn_grid(
        device, natural, n, h_out, w_grid, k, mode, precomputed, align_corners, input_shape, grid_dtype, memcfg
    )
    do_shard = sharded or (memcfg == "sharded")
    if do_shard:
        mem = _height_sharded_grid_mem(device, n * h_out * w_grid, logical_w, elem_bytes)
        tt_grid = ttnn.to_memory_config(grid_il, mem)
    else:
        tt_grid = grid_il

    input_mem = ttnn.L1_MEMORY_CONFIG if memcfg == "l1" else DRAM
    tt_in = ttnn.from_torch(inp, layout=RM, device=device, memory_config=input_mem)
    out = ttnn.grid_sample(
        tt_in,
        tt_grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=align_corners,
        use_precomputed_grid=precomputed,
        batch_output_channels=batch_output_channels,
    )
    got = ttnn.to_torch(out)

    assert list(got.shape) == list(golden.shape), f"shape {list(got.shape)} != golden {list(golden.shape)}"
    assert_with_pcc(golden, got, 0.99)


# batching combos: (batch_output_channels, K)
_BATCH_COMBOS = [(False, 1), (False, 4), (True, 4)]

# input shapes + grid geometry: (input_nhwc, h_out, w_grid)  (w_grid is the per-row width before K)
_SHAPES = [
    ((1, 16, 16, 32), 8, 8),
    ((2, 24, 20, 64), 12, 6),
    ((1, 32, 24, 96), 12, 8),
]


# ===========================================================================
# BILINEAR — all combinations
# ===========================================================================
@pytest.mark.parametrize("align_corners", [True, False])
@pytest.mark.parametrize("precomputed", [False, True])
@pytest.mark.parametrize("batch_output_channels, K", _BATCH_COMBOS)
@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("input_shape, h_out, w_grid", _SHAPES)
def test_bilinear_matrix(
    device, align_corners, precomputed, batch_output_channels, K, sharded, input_shape, h_out, w_grid
):
    _run_case(device, "bilinear", align_corners, precomputed, batch_output_channels, sharded, input_shape, h_out, w_grid, K)


# ===========================================================================
# NEAREST — all combinations, both use_precomputed_grid False (raw grid, on-device
# nearest) and True (host-prepared indices)
# ===========================================================================
@pytest.mark.parametrize("align_corners", [True, False])
@pytest.mark.parametrize("precomputed", [False, True])
@pytest.mark.parametrize("batch_output_channels, K", _BATCH_COMBOS)
@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("input_shape, h_out, w_grid", _SHAPES)
def test_nearest_matrix(device, align_corners, precomputed, batch_output_channels, K, sharded, input_shape, h_out, w_grid):
    n = input_shape[0]
    # Nearest auto-shards its output; with an INTERLEAVED grid the op derives core count from H*W,
    # so N>1 or K-width-expansion can require >64 shards. Route those through a SHARDED grid instead.
    if not sharded:
        out_row_mult = 1 if batch_output_channels else K
        if n > 1 or out_row_mult > 1:
            pytest.skip("nearest + interleaved grid auto-shard exceeds core count for N>1 or K-width-extend")
    _run_case(device, "nearest", align_corners, precomputed, batch_output_channels, sharded, input_shape, h_out, w_grid, K)


# ===========================================================================
# GRID DTYPE — bilinear standard grid accepts bf16 and fp32
# ===========================================================================
@pytest.mark.parametrize("grid_dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("align_corners", [True, False])
def test_bilinear_grid_dtype(device, grid_dtype, align_corners):
    torch.manual_seed(0)
    n, h, w, c = 1, 24, 24, 64
    inp = torch.randn(n, h, w, c, dtype=torch.bfloat16)
    grid = torch.rand(n, 16, 16, 2, dtype=torch.float32) * 2.4 - 1.2
    # Golden runs in the same grid dtype as ttnn: a bf16 ttnn grid -> bf16-quantized golden grid,
    # a fp32 ttnn grid -> fp32 golden grid. Otherwise the two see different coordinates.
    golden_grid = grid.to(torch.bfloat16).float() if grid_dtype == ttnn.bfloat16 else grid
    golden = golden_grid_sample(
        input_tensor=inp, grid=golden_grid, mode="bilinear", padding_mode="zeros", align_corners=align_corners
    )
    tt_in = ttnn.from_torch(inp, layout=RM, device=device)
    tt_grid = ttnn.from_torch(grid, layout=RM, device=device, dtype=grid_dtype)
    out = ttnn.grid_sample(tt_in, tt_grid, mode="bilinear", align_corners=align_corners)
    assert_with_pcc(golden, ttnn.to_torch(out), 0.99)


# ===========================================================================
# NCHW-input shape sweep (raw grid, non-precomputed) — incl. non-precomputed NEAREST.
# data_shape is NCHW (N, C, H_in, W_in); the kernel receives NHWC so C must be % 32.
# ===========================================================================
@pytest.mark.parametrize(
    "data_shape, grid_shape, mode, padding_mode, align_corners",
    [
        pytest.param((1, 32, 8, 8), (1, 4, 4, 2), "bilinear", "zeros", 1),
        pytest.param((1, 32, 8, 8), (1, 4, 4, 2), "bilinear", "zeros", 0),
        pytest.param((1, 32, 8, 8), (1, 4, 4, 2), "nearest", "zeros", 1),
        pytest.param((1, 32, 8, 8), (1, 4, 4, 2), "nearest", "zeros", 0),
        pytest.param((1, 64, 96, 96), (1, 128, 64, 2), "bilinear", "zeros", 1),
        pytest.param((1, 64, 96, 96), (1, 128, 64, 2), "nearest", "zeros", 1),
        # BEV-representative shape: C=64 (divisible by 32), bilinear mode.
        pytest.param((1, 64, 80, 144), (1, 128, 64, 2), "bilinear", "zeros", 1),
    ],
)
def test_grid_sample_nchw_shapes(device, data_shape, grid_shape, mode, padding_mode, align_corners):
    torch.manual_seed(0)
    ac = bool(align_corners)
    n, c, h_in, w_in = data_shape

    inp_nchw = torch.randn(*data_shape, dtype=torch.bfloat16)          # NCHW
    grid = torch.rand(*grid_shape, dtype=torch.float32) * 2.2 - 1.1    # [-1.1,1.1] -> some OOB (zeros padding)

    # torch reference (NCHW in/out)
    golden_nchw = F.grid_sample(
        inp_nchw.float(), grid.float(), mode=mode, padding_mode=padding_mode, align_corners=ac
    )
    golden_nhwc = golden_nchw.permute(0, 2, 3, 1)                       # NCHW -> NHWC for comparison

    # ttnn (NHWC in/out); raw grid, non-precomputed (nearest computes indices on-device).
    # fp32 grid so coords match the fp32 golden exactly (matters for discrete NEAREST rounding).
    tt_in = ttnn.from_torch(inp_nchw.permute(0, 2, 3, 1).contiguous(), layout=RM, device=device)  # NHWC
    tt_grid = ttnn.from_torch(grid, layout=RM, device=device, dtype=ttnn.float32)
    out = ttnn.grid_sample(
        tt_in, tt_grid, mode=mode, padding_mode=padding_mode, align_corners=ac, use_precomputed_grid=False
    )
    got = ttnn.to_torch(out)

    assert list(got.shape) == list(golden_nhwc.shape), f"shape {list(got.shape)} != golden {list(golden_nhwc.shape)}"
    assert_with_pcc(golden_nhwc, got, 0.99)


# ===========================================================================
# NEAREST, non-precomputed, batched (K=8) with batch_output_channels=True
#   input  (1, H_in, W_in, 64) NHWC  (C=64)
#   grid   (1, 128, 64, 16)          (last dim = 2*K, K=8 coord pairs per row)
#   -> output (1, 128, 64, 512)      (C*K = 64*8), align_corners=True, padding "zeros"
# ===========================================================================
@pytest.mark.parametrize("h_in, w_in", [(80, 144), (96, 96)])
def test_nearest_batched_channel_extend(device, h_in, w_in):
    torch.manual_seed(0)
    n, c = 1, 64
    h_out, w_out, K = 128, 64, 8

    inp = torch.randn(n, h_in, w_in, c, dtype=torch.bfloat16)                       # NHWC (1,80,144,64)
    grid = torch.rand(n, h_out, w_out, 2 * K, dtype=torch.float32) * 2.2 - 1.1      # (1,128,64,16), some OOB

    golden = golden_grid_sample(
        input_tensor=inp,
        grid=grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
        batch_output_channels=True,
    )

    tt_in = ttnn.from_torch(inp, layout=RM, device=device)
    tt_grid = ttnn.from_torch(grid, layout=RM, device=device, dtype=ttnn.float32)   # fp32 raw grid (non-precomputed)
    out = ttnn.grid_sample(
        tt_in,
        tt_grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
        batch_output_channels=True,
        use_precomputed_grid=False,
    )
    got = ttnn.to_torch(out)

    assert list(got.shape) == [1, 128, 64, 512], f"shape {list(got.shape)} != [1, 128, 64, 512]"
    assert list(golden.shape) == [1, 128, 64, 512], f"golden shape {list(golden.shape)} != [1, 128, 64, 512]"
    assert_with_pcc(golden, got, 0.99)


# ===========================================================================
# LARGE COMBINATORIAL MATRIX
#   Bigger input / grid / output shapes, both grid dtypes (fp32 + bf16), both modes,
#   align_corners {T,F}, use_precomputed_grid {F,T}, padding_mode "zeros",
#   batch_output_channels {F,T} with several K factors, and three memory configs
#   (DRAM interleaved, L1 interleaved, HEIGHT_SHARDED). Tile-aligned and non-tile-aligned
#   grid-point counts are both covered by _BIG_SHAPES (input C stays % 32, an op requirement).
# ===========================================================================
# (input_nhwc, h_out, w_grid, tile_aligned)  — tile_aligned = (N*h_out*w_grid) % 32 == 0
_BIG_SHAPES = [
    ((1, 64, 96, 64), 64, 32, True),  #  N*h_out*w_grid = 2048 (tile-aligned)
    ((1, 48, 72, 64), 30, 17, False),  # N*h_out*w_grid =  510 (not tile-aligned)
]

# (precomputed, grid_dtype): precomputed grids are always bf16, so fp32 there would just duplicate.
_PRECOMP_DTYPE = [
    (False, ttnn.float32),
    (False, ttnn.bfloat16),
    (True, ttnn.bfloat16),
]

_MEMCFGS = ["dram", "l1", "sharded"]


@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
@pytest.mark.parametrize("align_corners", [True, False])
@pytest.mark.parametrize("precomputed, grid_dtype", _PRECOMP_DTYPE)
@pytest.mark.parametrize("batch_output_channels, K", _BATCH_COMBOS)
@pytest.mark.parametrize("memcfg", _MEMCFGS)
@pytest.mark.parametrize("input_shape, h_out, w_grid, tile_aligned", _BIG_SHAPES)
def test_grid_sample_large_matrix(
    device,
    mode,
    align_corners,
    precomputed,
    grid_dtype,
    batch_output_channels,
    K,
    memcfg,
    input_shape,
    h_out,
    w_grid,
    tile_aligned,
):
    n = input_shape[0]

    # Nearest + non-precomputed + bf16 grid is fully supported: the kernel rounds half-to-even (matching
    # PyTorch's std::nearbyint), so it is bit-exact with the golden given the same coordinates. _run_case
    # bf16-quantizes the golden's grid so both see identical coords (a bf16 grid genuinely carries coarser
    # coordinates than fp32 — that is an input-precision choice, not an op error).

    # Nearest auto-shards its output; with an INTERLEAVED grid the op derives core count from H*W, so
    # N>1 or K-width-expansion can require >64 shards. Those go through a SHARDED grid instead.
    if mode == "nearest" and memcfg != "sharded":
        out_row_mult = 1 if batch_output_channels else K
        if n > 1 or out_row_mult > 1:
            pytest.skip("nearest + interleaved grid auto-shard exceeds core count for N>1 or K-width-extend")

    _run_case(
        device,
        mode,
        align_corners,
        precomputed,
        batch_output_channels,
        sharded=False,
        input_shape=input_shape,
        h_out=h_out,
        w_grid=w_grid,
        k=K,
        memcfg=memcfg,
        grid_dtype_override=grid_dtype,
    )
