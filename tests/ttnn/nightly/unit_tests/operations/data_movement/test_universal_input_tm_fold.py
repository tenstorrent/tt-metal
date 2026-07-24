# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Universal input/output tests for ttnn.fold — one case per routing path (RM/TILE × interleaved/H/W/B sharded × L1/DRAM × override × collapse_output × padding × legacy transpose). Default output is folded_4d (N, H/sh, W/sw, C*sh*sw); collapse_output=True emits (1, 1, N*H'*W', C*sh*sw) directly via compute_output_specs."""

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


_TTNN_TO_TORCH_DTYPE = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
}

L1_INTERLEAVED = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
DRAM_INTERLEAVED = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)


def _fold_golden(x_nhwc, stride_h, stride_w):
    """NHWC pixel_unshuffle — matches ttnn.fold's (N, H/sh, W/sw, C*sh*sw) output."""
    N, H, W, C = x_nhwc.shape
    r = x_nhwc.reshape(N, H // stride_h, stride_h, W // stride_w, stride_w, C)
    return r.permute(0, 1, 3, 2, 4, 5).reshape(N, H // stride_h, W // stride_w, C * stride_h * stride_w)


def _height_shard_rm(
    shape, device, num_cores=4, buffer_type=ttnn.BufferType.L1, orientation=ttnn.ShardOrientation.ROW_MAJOR
):
    """HEIGHT-sharded RM config with shard_h a multiple of (W * stride_h) — matches fold's fast-path pre-condition."""
    N, H, W, C = shape
    total_h = N * H * W
    grid = device.compute_with_storage_grid_size()
    num_cores = min(num_cores, grid.x * grid.y)
    while num_cores > 1 and total_h % num_cores != 0:
        num_cores -= 1
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, True)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type,
        ttnn.ShardSpec(shard_grid, [total_h // num_cores, C], orientation),
    )


def _width_shard_rm(shape, device, num_cores=4, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    """WIDTH-sharded L1 RM config over the channel dim."""
    N, H, W, C = shape
    grid = device.compute_with_storage_grid_size()
    num_cores = min(num_cores, grid.x * grid.y)
    while num_cores > 1 and C % num_cores != 0:
        num_cores -= 1
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, True)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(shard_grid, [N * H * W, C // num_cores], orientation),
    )


def _block_shard_rm(shape, device, grid_y=2, grid_x=2, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    """BLOCK-sharded L1 RM config."""
    N, H, W, C = shape
    grid = device.compute_with_storage_grid_size()
    grid_y = min(grid_y, grid.y)
    grid_x = min(grid_x, grid.x)
    while grid_y > 1 and (N * H * W) % grid_y != 0:
        grid_y -= 1
    while grid_x > 1 and C % grid_x != 0:
        grid_x -= 1
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}),
            [(N * H * W) // grid_y, C // grid_x],
            orientation,
        ),
    )


def _tile_width_shard(shape, device, num_cores=4):
    """Tile-aligned WIDTH-sharded L1 config. Requires C divisible by num_cores * 32."""
    N, H, W, C = shape
    grid = device.compute_with_storage_grid_size()
    num_cores = min(num_cores, grid.x * grid.y)
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, True)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(shard_grid, [N * H * W, C // num_cores], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _tile_block_shard(shape, device, grid_y=2, grid_x=2):
    """Tile-aligned BLOCK-sharded L1 config."""
    N, H, W, C = shape
    grid = device.compute_with_storage_grid_size()
    grid_y = min(grid_y, grid.y)
    grid_x = min(grid_x, grid.x)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}),
            [(N * H * W) // grid_y, C // grid_x],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def _run_fold(
    shape,
    stride_h,
    stride_w,
    layout,
    input_mem_config,
    output_mem_config,
    device,
    pcc=0.9999,
    dtype=ttnn.bfloat16,
    expected_shard_orientation=None,
    use_transpose_as_fold=False,
    padding=None,
    collapse_output=False,
):
    """Run fold; assert output shape, memory config (layout + orientation), and PCC vs golden — collapse_output=True flattens ref via reshape for rank-matched PCC."""
    torch.manual_seed(12345)
    torch_dtype = _TTNN_TO_TORCH_DTYPE[dtype]
    x = torch.rand(shape, dtype=torch_dtype)

    ttnn_in = ttnn.from_torch(x, layout=layout, dtype=dtype, device=device, memory_config=input_mem_config)
    kwargs = {}
    if output_mem_config is not None:
        kwargs["override_memory_config"] = output_mem_config
    if use_transpose_as_fold:
        kwargs["use_transpose_as_fold"] = True
    if padding is not None:
        kwargs["padding"] = padding
    if collapse_output:
        kwargs["collapse_output"] = True
    result = ttnn.fold(ttnn_in, stride_h, stride_w, **kwargs)

    N, H, W, C = shape
    ph_top = ph_bot = pw_l = pw_r = pc_f = pc_b = 0
    if padding is not None:
        p = list(padding)
        if len(p) == 2:
            ph_top = ph_bot = p[0]
            pw_l = pw_r = p[1]
        elif len(p) == 4:
            ph_top, ph_bot, pw_l, pw_r = p
        else:
            ph_top, ph_bot, pw_l, pw_r, pc_f, pc_b = p
    H_eff = H + ph_top + ph_bot
    W_eff = W + pw_l + pw_r
    C_eff = C + pc_f + pc_b
    Hp = H_eff // stride_h
    Wp = W_eff // stride_w
    Cp = C_eff * stride_h * stride_w
    expected_shape = (1, 1, N * Hp * Wp, Cp) if collapse_output else (N, Hp, Wp, Cp)
    assert tuple(result.shape) == expected_shape, f"Expected shape {expected_shape}, got {tuple(result.shape)}"

    if output_mem_config is not None:
        actual = result.memory_config()
        assert (
            actual.memory_layout == output_mem_config.memory_layout
        ), f"Expected memory_layout {output_mem_config.memory_layout}, got {actual.memory_layout}"
        assert (
            actual.buffer_type == output_mem_config.buffer_type
        ), f"Expected buffer_type {output_mem_config.buffer_type}, got {actual.buffer_type}"
        if output_mem_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            assert actual.shard_spec is not None, "Sharded output requested but result has no shard_spec"
            if expected_shard_orientation is not None:
                assert (
                    actual.shard_spec.orientation == expected_shard_orientation
                ), f"Expected shard orientation {expected_shard_orientation}, got {actual.shard_spec.orientation}"

    x_padded = torch.nn.functional.pad(x, (pc_f, pc_b, pw_l, pw_r, ph_top, ph_bot)) if padding is not None else x
    ref = _fold_golden(x_padded, stride_h, stride_w)
    got = ttnn.to_torch(result.cpu().to(ttnn.ROW_MAJOR_LAYOUT))
    if collapse_output:
        ref = ref.reshape(1, 1, N * Hp * Wp, Cp)
    assert_with_pcc(ref.float(), got.float(), pcc)


# Interleaved routing — RM goes through MultiCoreDRAMFold RM branch; TILE untilizes first.


@pytest.mark.parametrize(
    "layout, shape, input_mc, output_mc",
    [
        pytest.param(ttnn.ROW_MAJOR_LAYOUT, (1, 16, 16, 8), L1_INTERLEAVED, L1_INTERLEAVED, id="rm_L1_to_L1"),
        pytest.param(ttnn.ROW_MAJOR_LAYOUT, (1, 16, 16, 8), DRAM_INTERLEAVED, DRAM_INTERLEAVED, id="rm_DRAM_to_DRAM"),
        pytest.param(ttnn.ROW_MAJOR_LAYOUT, (1, 16, 16, 8), L1_INTERLEAVED, DRAM_INTERLEAVED, id="rm_L1_to_DRAM"),
        pytest.param(ttnn.ROW_MAJOR_LAYOUT, (1, 16, 16, 8), DRAM_INTERLEAVED, L1_INTERLEAVED, id="rm_DRAM_to_L1"),
        pytest.param(ttnn.TILE_LAYOUT, (1, 32, 32, 32), L1_INTERLEAVED, L1_INTERLEAVED, id="tile_L1_to_L1"),
        pytest.param(ttnn.TILE_LAYOUT, (1, 32, 32, 32), DRAM_INTERLEAVED, DRAM_INTERLEAVED, id="tile_DRAM_to_DRAM"),
        pytest.param(ttnn.TILE_LAYOUT, (1, 32, 32, 32), L1_INTERLEAVED, DRAM_INTERLEAVED, id="tile_L1_to_DRAM"),
    ],
)
def test_fold_interleaved(layout, shape, input_mc, output_mc, device):
    _run_fold(shape, 2, 2, layout, input_mc, output_mc, device)


# RM + HEIGHT_SHARDED L1 — zero-NOC fast path; parametrized over shape/stride/ncores + col_major + multi-batch.


@pytest.mark.parametrize(
    "shape, stride, ncores, orientation",
    [
        pytest.param((1, 16, 16, 8), (2, 2), 4, ttnn.ShardOrientation.ROW_MAJOR, id="16x16x8_2x2_4c"),
        pytest.param((1, 32, 32, 16), (2, 2), 8, ttnn.ShardOrientation.ROW_MAJOR, id="32x32x16_2x2_8c"),
        pytest.param((2, 16, 16, 8), (4, 4), 4, ttnn.ShardOrientation.ROW_MAJOR, id="N2_16x16x8_4x4_4c_multi_batch"),
        pytest.param((2, 16, 16, 8), (2, 2), 4, ttnn.ShardOrientation.ROW_MAJOR, id="N2_16x16x8_2x2_4c_multi_batch"),
        pytest.param((4, 16, 16, 8), (2, 2), 4, ttnn.ShardOrientation.ROW_MAJOR, id="N4_16x16x8_2x2_4c_multi_batch"),
        pytest.param((1, 16, 16, 8), (2, 2), 4, ttnn.ShardOrientation.COL_MAJOR, id="16x16x8_2x2_4c_col_major"),
    ],
)
def test_fold_height_sharded_fastpath(shape, stride, ncores, orientation, device):
    """Default output — HEIGHT_SHARDED L1 propagates through the fast path with rescaled shard; orientation preserved end-to-end."""
    in_mc = _height_shard_rm(shape, device, num_cores=ncores, orientation=orientation)
    _run_fold(shape, stride[0], stride[1], ttnn.ROW_MAJOR_LAYOUT, in_mc, None, device)


# HEIGHT_SHARDED input × (L1/DRAM buffer) × (native / interleaved override) — fast path + optional tail to_memory_config.


@pytest.mark.parametrize(
    "input_buffer_type, output_mc",
    [
        pytest.param(ttnn.BufferType.L1, L1_INTERLEAVED, id="L1_in_to_L1"),
        pytest.param(ttnn.BufferType.L1, DRAM_INTERLEAVED, id="L1_in_to_DRAM"),
        pytest.param(ttnn.BufferType.DRAM, None, id="DRAM_in_native"),
        pytest.param(ttnn.BufferType.DRAM, L1_INTERLEAVED, id="DRAM_in_to_L1"),
    ],
)
def test_fold_height_sharded_input_variants(input_buffer_type, output_mc, device):
    """H-sharded RM input × {L1, DRAM} × {native output preserves sharding, override → interleaved}."""
    shape = (1, 16, 16, 8)
    in_mc = _height_shard_rm(shape, device, num_cores=4, buffer_type=input_buffer_type)
    _run_fold(shape, 2, 2, ttnn.ROW_MAJOR_LAYOUT, in_mc, output_mc, device)


# RM + WIDTH/BLOCK-sharded input × (sub-align / aligned / col_major) → interleaved output.


@pytest.mark.parametrize(
    "shard_factory, shape, orientation",
    [
        pytest.param(_width_shard_rm, (1, 16, 16, 8), ttnn.ShardOrientation.ROW_MAJOR, id="width_sub_align"),
        pytest.param(_block_shard_rm, (1, 16, 16, 8), ttnn.ShardOrientation.ROW_MAJOR, id="block_sub_align"),
        pytest.param(_width_shard_rm, (1, 16, 16, 64), ttnn.ShardOrientation.ROW_MAJOR, id="width_aligned_native"),
        pytest.param(_block_shard_rm, (1, 16, 16, 64), ttnn.ShardOrientation.ROW_MAJOR, id="block_aligned_native"),
        pytest.param(_width_shard_rm, (1, 16, 16, 8), ttnn.ShardOrientation.COL_MAJOR, id="width_col_major"),
        pytest.param(_block_shard_rm, (1, 16, 16, 8), ttnn.ShardOrientation.COL_MAJOR, id="block_col_major"),
    ],
)
def test_fold_wb_sharded_input(shard_factory, shape, orientation, device):
    """C=8 (sub-align): stages via L1 interleaved. C=64 (aligned): native RM factory. COL_MAJOR: orientation not carried into interleaved output."""
    in_mc = shard_factory(shape, device, orientation=orientation)
    _run_fold(shape, 2, 2, ttnn.ROW_MAJOR_LAYOUT, in_mc, L1_INTERLEAVED, device)


# Interleaved input → sharded output — prim + tail to_memory_config.


@pytest.mark.parametrize(
    "output_mc_factory",
    [
        pytest.param(_height_shard_rm, id="to_height_sh"),
        pytest.param(_width_shard_rm, id="to_width_sh"),
        pytest.param(_block_shard_rm, id="to_block_sh"),
    ],
)
@pytest.mark.parametrize(
    "input_mc",
    [
        pytest.param(L1_INTERLEAVED, id="L1_in"),
        pytest.param(DRAM_INTERLEAVED, id="DRAM_in"),
    ],
)
def test_fold_rm_interleaved_to_sharded(input_mc, output_mc_factory, device):
    shape = (1, 16, 16, 8)
    out_shape = (1, 8, 8, 32)
    out_mc = output_mc_factory(out_shape, device)
    _run_fold(shape, 2, 2, ttnn.ROW_MAJOR_LAYOUT, input_mc, out_mc, device)


# TILE + W/B-sharded input — composite untilize + stage L1 interleaved.


@pytest.mark.parametrize(
    "input_mc_factory",
    [
        pytest.param(_tile_width_shard, id="width_sh_in"),
        pytest.param(_tile_block_shard, id="block_sh_in"),
    ],
)
def test_fold_tile_wb_sharded_input(input_mc_factory, device):
    """TILE input in W/B-sharded config → interleaved output — exercises composite staging path."""
    shape = (1, 32, 32, 128)
    in_mc = input_mc_factory(shape, device)
    _run_fold(shape, 2, 2, ttnn.TILE_LAYOUT, in_mc, L1_INTERLEAVED, device)


# Sharded → sharded cross-layout — composite reshard on output.


@pytest.mark.parametrize(
    "in_factory, out_factory",
    [
        pytest.param(_height_shard_rm, _block_shard_rm, id="height_to_block"),
        pytest.param(_width_shard_rm, _height_shard_rm, id="width_to_height"),
    ],
)
def test_fold_rm_sharded_cross_layout(in_factory, out_factory, device):
    shape = (1, 16, 16, 8)
    out_shape = (1, 8, 8, 32)
    in_mc = in_factory(shape, device, num_cores=4) if in_factory is _height_shard_rm else in_factory(shape, device)
    out_mc = (
        out_factory(out_shape, device, num_cores=4)
        if out_factory is _height_shard_rm
        else out_factory(out_shape, device)
    )
    _run_fold(shape, 2, 2, ttnn.ROW_MAJOR_LAYOUT, in_mc, out_mc, device)


def test_fold_rm_interleaved_to_height_sharded_col_major_output(device):
    """Interleaved in → HEIGHT-sh COL_MAJOR out — tail to_memory_config sets orientation."""
    shape = (1, 16, 16, 8)
    out_shape = (1, 8, 8, 32)
    out_mc = _height_shard_rm(out_shape, device, num_cores=4, orientation=ttnn.ShardOrientation.COL_MAJOR)
    _run_fold(
        shape,
        2,
        2,
        ttnn.ROW_MAJOR_LAYOUT,
        L1_INTERLEAVED,
        out_mc,
        device,
        expected_shard_orientation=ttnn.ShardOrientation.COL_MAJOR,
    )


# f32 dtype coverage — representative paths.


@pytest.mark.parametrize(
    "shape, layout, in_mc_factory",
    [
        pytest.param((1, 16, 16, 8), ttnn.ROW_MAJOR_LAYOUT, lambda _s, _d: L1_INTERLEAVED, id="f32_rm_l1"),
        pytest.param((1, 16, 16, 8), ttnn.ROW_MAJOR_LAYOUT, lambda _s, _d: DRAM_INTERLEAVED, id="f32_rm_dram"),
        pytest.param(
            (1, 16, 16, 8),
            ttnn.ROW_MAJOR_LAYOUT,
            lambda s, d: _height_shard_rm(s, d, num_cores=4),
            id="f32_rm_hsharded_fastpath",
        ),
    ],
)
def test_fold_f32_dtype(shape, layout, in_mc_factory, device):
    _run_fold(shape, 2, 2, layout, in_mc_factory(shape, device), None, device, dtype=ttnn.float32)


# Non-standard strides — asymmetric, non-power-of-two (e.g. 3x5, 2x3).


@pytest.mark.parametrize(
    "shape, stride",
    [
        pytest.param((1, 24, 30, 8), (3, 5), id="stride_3x5_asym"),
        pytest.param((1, 12, 24, 8), (2, 3), id="stride_2x3_asym"),
    ],
)
def test_fold_asymmetric_stride(shape, stride, device):
    _run_fold(shape, stride[0], stride[1], ttnn.ROW_MAJOR_LAYOUT, L1_INTERLEAVED, L1_INTERLEAVED, device)


# Explicit padding — 2/4/6-elem variants; composite pads before prim::fold.


@pytest.mark.parametrize(
    "shape, stride, padding",
    [
        pytest.param((1, 14, 14, 8), (2, 2), [1, 1], id="sym_h1_w1"),  # 2-elem: symmetric [pad_h, pad_w].
        pytest.param((1, 14, 14, 8), (2, 2), [1, 1, 1, 1], id="asym_h1_w1"),  # 4-elem: [top, bot, l, r].
        pytest.param((1, 16, 16, 8), (2, 2), [0, 0, 0, 0, 0, 8], id="c_pad_only"),  # 6-elem: with C padding.
    ],
)
def test_fold_padding(shape, stride, padding, device):
    _run_fold(
        shape,
        stride[0],
        stride[1],
        ttnn.ROW_MAJOR_LAYOUT,
        L1_INTERLEAVED,
        L1_INTERLEAVED,
        device,
        padding=padding,
    )


# use_transpose_as_fold=True — legacy pad+transpose+reshape+slice smoke (deep coverage lives in test_fold_op.py).


def test_fold_use_transpose_as_fold_smoke(device):
    """Legacy pad+transpose+reshape+slice branch still runs (retirement blocked on #29514)."""
    shape = (1, 16, 16, 4)
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=torch.bfloat16)
    ttnn_in = ttnn.from_torch(
        x, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=L1_INTERLEAVED
    )
    result = ttnn.fold(ttnn_in, 2, 2, use_transpose_as_fold=True)
    assert result is not None and len(result.shape) == 4


# Negative test — invalid shard shape must FATAL, not silently truncate.


def test_fold_invalid_shard_shape_fatals(device):
    """Fast path requires shard_h % (input_W * stride_h) == 0; non-divisible height must be repaired by reshard_if_needed or FATAL cleanly (no SIGFPE)."""
    shape = (1, 16, 16, 8)
    total_h = 1 * 16 * 16
    grid = device.compute_with_storage_grid_size()
    num_cores = 16
    if grid.x * grid.y < num_cores:
        pytest.skip("Not enough cores for this negative test")
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, True)
    bad_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(shard_grid, [total_h // num_cores, 8], ttnn.ShardOrientation.ROW_MAJOR),
    )
    x = torch.rand(shape, dtype=torch.bfloat16)
    ttnn_in = ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=bad_mc)
    try:
        result = ttnn.fold(ttnn_in, 2, 2)
        # reshard_if_needed may repair the shard height; if so, verify correctness.
        ref = _fold_golden(x, 2, 2)
        got = ttnn.to_torch(result.cpu().to(ttnn.ROW_MAJOR_LAYOUT))
        assert_with_pcc(ref.float(), got.float(), 0.9999)
    except RuntimeError as e:
        # Clean FATAL — must not be a SIGFPE / segfault.
        assert "shard height" in str(e).lower() or "stride" in str(e).lower(), f"Unexpected FATAL message: {e}"


# collapse_output=True — device op emits (1,1,N*H'*W',C·sh·sw) via compute_output_specs across every routing path.


@pytest.mark.parametrize(
    "layout, shape, in_mc_factory, out_mc",
    [
        # RM interleaved.
        pytest.param(
            ttnn.ROW_MAJOR_LAYOUT, (1, 16, 16, 8), lambda s, d: L1_INTERLEAVED, L1_INTERLEAVED, id="rm_l1_interleaved"
        ),
        pytest.param(
            ttnn.ROW_MAJOR_LAYOUT,
            (1, 16, 16, 8),
            lambda s, d: DRAM_INTERLEAVED,
            DRAM_INTERLEAVED,
            id="rm_dram_interleaved",
        ),
        # RM sharded (aligned, native).
        pytest.param(
            ttnn.ROW_MAJOR_LAYOUT, (1, 16, 16, 64), _height_shard_rm, None, id="rm_l1_height_sharded_fastpath"
        ),
        pytest.param(ttnn.ROW_MAJOR_LAYOUT, (1, 16, 16, 64), _width_shard_rm, None, id="rm_l1_width_sharded_native"),
        pytest.param(ttnn.ROW_MAJOR_LAYOUT, (1, 16, 16, 64), _block_shard_rm, None, id="rm_l1_block_sharded_native"),
        # RM sub-alignment W/B (composite L1-interleaved staging hop).
        pytest.param(ttnn.ROW_MAJOR_LAYOUT, (1, 16, 16, 8), _width_shard_rm, None, id="rm_width_sh_sub_align"),
        pytest.param(ttnn.ROW_MAJOR_LAYOUT, (1, 16, 16, 8), _block_shard_rm, None, id="rm_block_sh_sub_align"),
        # DRAM H-sharded RM (native RM factory, not fast-path).
        pytest.param(
            ttnn.ROW_MAJOR_LAYOUT,
            (1, 16, 16, 8),
            lambda s, d: _height_shard_rm(s, d, num_cores=4, buffer_type=ttnn.BufferType.DRAM),
            None,
            id="rm_dram_height_sharded",
        ),
        # TILE interleaved (composite untilize hop).
        pytest.param(ttnn.TILE_LAYOUT, (1, 16, 16, 8), lambda s, d: L1_INTERLEAVED, None, id="tile_l1_interleaved"),
        pytest.param(ttnn.TILE_LAYOUT, (1, 16, 16, 8), lambda s, d: DRAM_INTERLEAVED, None, id="tile_dram_interleaved"),
        # TILE W/B-sharded (composite untilize + L1-interleaved staging).
        pytest.param(ttnn.TILE_LAYOUT, (1, 32, 32, 128), _tile_width_shard, None, id="tile_width_sh"),
        pytest.param(ttnn.TILE_LAYOUT, (1, 32, 32, 128), _tile_block_shard, None, id="tile_block_sh"),
    ],
)
def test_fold_collapse_output(layout, shape, in_mc_factory, out_mc, device):
    """collapse_output cross-product over every input routing path — shape asserted collapsed, bytes equal to folded_4d + reshape reference."""
    in_mc = in_mc_factory(shape, device)
    _run_fold(shape, 2, 2, layout, in_mc, out_mc, device, collapse_output=True)


def test_fold_collapse_output_shape_bytes_invariance(device):
    """folded_4d and collapsed outputs are byte-identical (metadata-only difference); verified via torch.equal."""
    torch.manual_seed(0)
    shape = (1, 16, 16, 8)
    stride_h, stride_w = 2, 2
    x = torch.rand(shape, dtype=torch.bfloat16)

    ttnn_in = ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device)

    folded_4d = ttnn.fold(ttnn_in, stride_h, stride_w)
    collapsed = ttnn.fold(ttnn_in, stride_h, stride_w, collapse_output=True)

    N, H, W, C = shape
    Hp = H // stride_h
    Wp = W // stride_w
    Cp = C * stride_h * stride_w

    assert tuple(folded_4d.shape) == (N, Hp, Wp, Cp)
    assert tuple(collapsed.shape) == (1, 1, N * Hp * Wp, Cp)

    t_folded = ttnn.to_torch(folded_4d.cpu().to(ttnn.ROW_MAJOR_LAYOUT))
    t_collapsed = ttnn.to_torch(collapsed.cpu().to(ttnn.ROW_MAJOR_LAYOUT))
    assert torch.equal(t_folded.reshape(1, 1, N * Hp * Wp, Cp), t_collapsed), "Byte equality broken between shapes"


def test_fold_collapse_output_preserves_sharded_memory_config(device):
    """H-sharded input + collapse_output: shard spec identical to folded_4d (rows_per_core = N·H'·W'/n_cores, orientation preserved)."""
    shape = (1, 16, 16, 8)
    in_mc = _height_shard_rm(shape, device, num_cores=4)
    x = torch.rand(shape, dtype=torch.bfloat16)
    ttnn_in = ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=in_mc)

    r_folded = ttnn.fold(ttnn_in, 2, 2)
    r_collapsed = ttnn.fold(ttnn_in, 2, 2, collapse_output=True)

    assert r_folded.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    assert r_collapsed.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    assert (
        r_folded.memory_config().shard_spec.shape == r_collapsed.memory_config().shard_spec.shape
    ), "Shard spec must be invariant under folded_4d ↔ collapsed rewrite"
    assert (
        r_folded.memory_config().shard_spec.orientation == r_collapsed.memory_config().shard_spec.orientation
    ), "Shard orientation must be preserved"


def test_fold_collapse_output_override_memory_config(device):
    """collapse_output + override_memory_config compose: shape reflects collapse, memconfig reflects override."""
    shape = (1, 16, 16, 8)
    in_mc = _height_shard_rm(shape, device, num_cores=4)
    _run_fold(
        shape,
        2,
        2,
        ttnn.ROW_MAJOR_LAYOUT,
        in_mc,
        L1_INTERLEAVED,
        device,
        collapse_output=True,
    )


def test_fold_collapse_output_rejects_use_transpose_as_fold(device, expect_error):
    """Legacy composite bypasses prim::fold → collapse_output has no plug-in point; must FATAL, not silently ignore."""
    shape = (1, 16, 16, 8)
    x = torch.rand(shape, dtype=torch.bfloat16)
    ttnn_in = ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device)
    with expect_error(RuntimeError, "collapse_output.*use_transpose_as_fold"):
        ttnn.fold(ttnn_in, 2, 2, use_transpose_as_fold=True, collapse_output=True)


# Specless sharded override — helper either reuses input's rescaled spec (matching layout) or synthesises fresh.


def _l1_interleaved_factory(_shape, _device):
    return L1_INTERLEAVED


@pytest.mark.parametrize(
    "layout_in, shape, in_mc_factory, override_layout, collapse_output",
    [
        # Interleaved input + specless H/W/B override → fresh synthesis.
        pytest.param(
            ttnn.ROW_MAJOR_LAYOUT,
            (1, 16, 16, 64),
            _l1_interleaved_factory,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            False,
            id="rm_l1_in_H_override",
        ),
        pytest.param(
            ttnn.ROW_MAJOR_LAYOUT,
            (1, 16, 16, 64),
            _l1_interleaved_factory,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            False,
            id="rm_l1_in_W_override",
        ),
        pytest.param(
            ttnn.ROW_MAJOR_LAYOUT,
            (1, 16, 16, 64),
            _l1_interleaved_factory,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            False,
            id="rm_l1_in_B_override",
        ),
        # Sharded input + specless override (matching-layout → reuse; cross-layout → fresh).
        pytest.param(
            ttnn.ROW_MAJOR_LAYOUT,
            (1, 16, 16, 64),
            _height_shard_rm,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            False,
            id="H_in_H_override_reuse",
        ),
        pytest.param(
            ttnn.ROW_MAJOR_LAYOUT,
            (1, 16, 16, 64),
            _height_shard_rm,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            False,
            id="H_in_W_override_fresh",
        ),
        pytest.param(
            ttnn.ROW_MAJOR_LAYOUT,
            (1, 16, 16, 64),
            _height_shard_rm,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            False,
            id="H_in_B_override_fresh",
        ),
        # TILE input + specless sharded override (untilize hop + tail synth).
        pytest.param(
            ttnn.TILE_LAYOUT,
            (1, 16, 16, 64),
            _l1_interleaved_factory,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            False,
            id="tile_in_H_override",
        ),
        # Sub-align W/B → H-sharded override (composite stage + tail synth).
        pytest.param(
            ttnn.ROW_MAJOR_LAYOUT,
            (1, 16, 16, 8),
            _width_shard_rm,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            False,
            id="W_sub_align_H_override",
        ),
        pytest.param(
            ttnn.ROW_MAJOR_LAYOUT,
            (1, 16, 16, 8),
            _block_shard_rm,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            False,
            id="B_sub_align_H_override",
        ),
        # collapse_output + specless override.
        pytest.param(
            ttnn.ROW_MAJOR_LAYOUT,
            (1, 16, 16, 64),
            _l1_interleaved_factory,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            True,
            id="collapse_l1_in_H_override",
        ),
    ],
)
def test_fold_specless_sharded_override_e2e(layout_in, shape, in_mc_factory, override_layout, collapse_output, device):
    """E2E specless override cross-product — PCC vs golden covers both reuse and fresh-synthesis paths in the helper."""
    in_mc = in_mc_factory(shape, device)
    out_mc = ttnn.MemoryConfig(override_layout, ttnn.BufferType.L1)
    _run_fold(shape, 2, 2, layout_in, in_mc, out_mc, device, collapse_output=collapse_output)


# Matching-layout specless override — reuse branch must preserve input grid + orientation across aligned/sub-align/COL_MAJOR.


@pytest.mark.parametrize(
    "shard_factory, override_layout, orientation, shape",
    [
        pytest.param(
            _height_shard_rm,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
            (1, 16, 16, 64),
            id="H_aligned_row_major",
        ),
        pytest.param(
            _height_shard_rm,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.COL_MAJOR,
            (1, 16, 16, 64),
            id="H_aligned_col_major",
        ),
        pytest.param(
            _width_shard_rm,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
            (1, 16, 16, 8),
            id="W_sub_align_row_major",
        ),
        pytest.param(
            _width_shard_rm,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.ShardOrientation.COL_MAJOR,
            (1, 16, 16, 8),
            id="W_sub_align_col_major",
        ),
        pytest.param(
            _block_shard_rm,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
            (1, 16, 16, 8),
            id="B_sub_align_row_major",
        ),
    ],
)
def test_fold_specless_matching_layout_preserves_input_geometry(
    shard_factory, override_layout, orientation, shape, device
):
    """Matching-layout specless override must reuse input's grid + orientation (not synthesise full compute grid)."""
    in_mc = shard_factory(shape, device, orientation=orientation)
    out_mc = ttnn.MemoryConfig(override_layout, ttnn.BufferType.L1)
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=torch.bfloat16)
    ttnn_in = ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=in_mc)
    result = ttnn.fold(ttnn_in, 2, 2, override_memory_config=out_mc)
    mc = result.memory_config()
    assert mc.memory_layout == override_layout
    assert mc.shard_spec is not None
    assert (
        mc.shard_spec.grid == in_mc.shard_spec.grid
    ), f"Matching-layout reuse must preserve input grid; got {mc.shard_spec.grid} vs input {in_mc.shard_spec.grid}"
    assert (
        mc.shard_spec.orientation == orientation
    ), f"Matching-layout reuse must preserve input orientation; got {mc.shard_spec.orientation}"
    ref = _fold_golden(x, 2, 2)
    got = ttnn.to_torch(result.cpu().to(ttnn.ROW_MAJOR_LAYOUT))
    assert_with_pcc(ref.float(), got.float(), 0.9999)


def test_fold_from_torch_rejects_specless_sharded_input(device, expect_error):
    """Doc: from_torch refuses specless sharded MC — fold's has_value() input guards are defensive-only (mirror transpose)."""
    x = torch.rand(1, 16, 16, 8, dtype=torch.bfloat16)
    specless_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)
    with expect_error(RuntimeError, "[Ss]hard spec must not be None"):
        ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=specless_mc)


@pytest.mark.parametrize(
    "shape,override_layout",
    [
        pytest.param(
            (1, 8, 8, 64), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, id="H_shrinks"
        ),  # post-fold rows=16 → 16 cores (not 64).
        pytest.param(
            (1, 4, 4, 64), ttnn.TensorMemoryLayout.BLOCK_SHARDED, id="B_shrinks"
        ),  # post-fold 4×256 → 4x8=32 cores (not 64).
    ],
)
def test_fold_specless_sharded_override_grid_shrinks_when_rows_lt_max_cores(shape, override_layout, device):
    """rows < max_cores after fold → synthesised grid must size to actual populated shards, not full compute grid."""
    in_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    out_mc = ttnn.MemoryConfig(override_layout, ttnn.BufferType.L1)
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=torch.bfloat16)
    ttnn_in = ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=in_mc)
    result = ttnn.fold(ttnn_in, 2, 2, override_memory_config=out_mc)
    max_cores = device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y
    mc = result.memory_config()
    assert mc.shard_spec is not None
    assert (
        mc.shard_spec.grid.num_cores() < max_cores
    ), f"Synthesised grid over-declared cores: {mc.shard_spec.grid.num_cores()} == max {max_cores}"
    ref = _fold_golden(x, 2, 2)
    got = ttnn.to_torch(result.cpu().to(ttnn.ROW_MAJOR_LAYOUT))
    assert_with_pcc(ref.float(), got.float(), 0.9999)


# Coverage-gap tests aligned with transpose/slice — TILE H-sharded input, fast-path reshard trigger.


def test_fold_tile_height_sharded_input(device):
    """TILE + L1 H-sharded input → composite untilizes to RM H-sharded and then falls into the zero-NOC fast path (mirrors transpose's tile-then-untilize pattern)."""
    shape = (1, 32, 32, 32)
    total_h = shape[0] * shape[1] * shape[2]
    grid = device.compute_with_storage_grid_size()
    num_cores = 4
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, True)
    in_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(shard_grid, [total_h // num_cores, shape[3]], ttnn.ShardOrientation.ROW_MAJOR),
    )
    _run_fold(shape, 2, 2, ttnn.TILE_LAYOUT, in_mc, L1_INTERLEAVED, device)


def test_fold_fast_path_reshard_trigger_specless_matching_layout_override(device):
    """Fast-path input misaligned to (W*stride_h) → reshard_if_needed fires; tail helper must rescale post-reshard geometry, not original."""
    # total_H=24, W*stride_h=8, patch=4; shard_h=6 trips reshard AND 6%4=2 (would under-cover if pre-reshard geometry were used).
    shape = (1, 6, 4, 64)
    in_mc = _height_shard_rm(shape, device, num_cores=4)
    assert in_mc.shard_spec.shape[0] % (shape[2] * 2) != 0, "test setup must trip reshard_if_needed"
    assert in_mc.shard_spec.shape[0] % (2 * 2) != 0, "test setup must expose the integer-truncation bug on pre-fix path"
    out_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)  # specless matching-layout.
    torch.manual_seed(0)
    x = torch.rand(shape, dtype=torch.bfloat16)
    ttnn_in = ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=in_mc)
    result = ttnn.fold(ttnn_in, 2, 2, override_memory_config=out_mc)
    ref = _fold_golden(x, 2, 2)
    got = ttnn.to_torch(result.cpu().to(ttnn.ROW_MAJOR_LAYOUT))
    assert_with_pcc(ref.float(), got.float(), 0.9999)
