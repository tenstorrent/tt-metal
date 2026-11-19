# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""conv2d_3x3 on an L1 BLOCK_SHARDED input (regime R4): C_in is split over the grid's columns, each core
computes fp32 partials over its channel slice, and every grid row reduce-scatters them so column y packs
output N-slice y. Output BLOCK_SHARDED on the same grid.

- layout matrix: ROW_MAJOR bf16, TILE bf16 and TILE bf8_b; halo several shards away (d = 8), stride 2,
  a short last channel slice, a short last N-slice, columns that own no output, and a single shard
  column (no split).
- ReLU after the combine only: per-column partials that are large and cancel across the row.
- placement: output shard spec, COL_MAJOR orientation on a grid away from (0, 0), and the program-cache
  hit on fresh shards.
- multi-M-block shard: one combine round per M-block, with state crossing the rounds.
- extended rows: the K split's M split runs down grid rows past the shard grid (to the device edge), and the
  owners store their finished tiles to output shards on other cores.
- spare-column groups: more M-groups than grid rows, transposed into the device columns past the shard grid, their
  weight rounds forwarded from the column streams.
- host-planned fill: each core's slab reads planned on the host (split cores' writer part issued at kernel start, the
  spare members' upper half on NoC1, a short last slice's padding tiles zeroed), bit-identical to the device plan.
"""

import importlib

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.conv2d_3x3 import conv2d_3x3, prepare_conv2d_bias, prepare_conv2d_weights

_op_module = importlib.import_module("ttnn.operations.conv2d_3x3.conv2d_3x3")
_pd = importlib.import_module("ttnn.operations.conv2d_3x3.conv2d_3x3_program_descriptor")

PCC = {ttnn.bfloat16: 0.997, ttnn.bfloat8_b: 0.99}

LAYOUTS = [
    pytest.param(ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, id="rm_bf16"),
    pytest.param(ttnn.bfloat16, ttnn.TILE_LAYOUT, id="tile_bf16"),
    pytest.param(ttnn.bfloat8_b, ttnn.TILE_LAYOUT, id="tile_bf8b"),
]

# ((H, W, C_in), C_out, padding, stride, dilation), {layout: (shard rows, shard width)}
RM, TL = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
SHAPES = [
    # 4 / 2 channel slices; RM shards do not start on an image row
    (((16, 16, 64), 128, 1, 1, 1), {RM: (40, 16), TL: (64, 32)}),
    # stem-like, one shard column: no K split
    (((32, 32, 8), 32, 1, 2, 1), {RM: (110, 8), TL: (128, 32)}),
    # d = 8 halo 8 shards away; C_out = 32: only column 0 owns output, the others only send
    (((32, 64, 64), 32, 8, 1, 8), {RM: (256, 16), TL: (256, 32)}),
    # short last channel slice (16 of 32), short last N-slice (1 of 2 tiles)
    (((16, 34, 48), 96, 0, 1, 1), {RM: (68, 32), TL: (96, 32)}),
    # res4-like: 8 channel slices, one output tile column each
    (((8, 16, 256), 256, 1, 1, 1), {RM: (32, 32), TL: (32, 32)}),
]


def _case_id(case):
    ((H, W, c_in), c_out, p, s, d), _ = case
    return f"H{H}_W{W}_Ci{c_in}_Co{c_out}_p{p}_s{s}_d{d}"


def _block_config(num_px, c_in, layout, shard, *, start=(0, 0), orientation=ttnn.ShardOrientation.ROW_MAJOR):
    shard_rows, shard_width = shard
    if layout == ttnn.TILE_LAYOUT:
        num_px = -(-num_px // 32) * 32
    rows, cols = -(-num_px // shard_rows), -(-c_in // shard_width)
    if orientation == ttnn.ShardOrientation.COL_MAJOR:
        rows, cols = cols, rows
    x0, y0 = start
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x0 + cols - 1, y0 + rows - 1))])
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [shard_rows, shard_width], orientation),
    )


def _reference(torch_input, torch_weight, torch_bias, H, W, p, s, d, relu):
    c_in = torch_input.shape[-1]
    x = torch_input.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    out = torch.nn.functional.conv2d(
        x, torch_weight.float(), None if torch_bias is None else torch_bias.float(), stride=s, padding=p, dilation=d
    )
    if relu:
        out = torch.relu(out)
    return out.permute(0, 2, 3, 1).reshape(1, 1, -1, out.shape[1])


def _run(
    case_hw,
    dtype,
    layout,
    memory_config,
    device,
    *,
    with_bias,
    relu,
    dispatches=1,
    weight_dtype=ttnn.bfloat8_b,
    make_tensors=None,
    pcc=None,
):
    (H, W, c_in), c_out, p, s, d = case_hw
    if make_tensors is None:
        torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
        make_input = lambda: torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    else:
        torch_weight, make_input = make_tensors()
    torch_bias = torch.randn(c_out).to(torch.bfloat16) if with_bias else None
    kwargs = dict(
        stride=s,
        padding=p,
        dilation=d,
        input_dtype=dtype,
        input_layout=layout,
        weight_dtype=weight_dtype,
        input_memory_config=memory_config,
        device=device,
    )
    tt_weight = prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = prepare_conv2d_bias(torch_bias, **kwargs) if with_bias else None
    h_out = (H + 2 * p - 2 * d - 1) // s + 1
    w_out = (W + 2 * p - 2 * d - 1) // s + 1
    m = h_out * w_out
    rows, cols = _pd.block_grid_shape(memory_config.shard_spec)

    live = []  # keep every dispatch's tensors alive, so later dispatches get fresh addresses
    for _ in range(dispatches):
        torch_input = make_input()
        tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device, memory_config=memory_config)
        tt_output = conv2d_3x3(
            tt_input,
            tt_weight,
            input_height=H,
            input_width=W,
            bias=tt_bias,
            padding=p,
            stride=s,
            dilation=d,
            activation="relu" if relu else None,
        )
        live.append((tt_input, tt_output))

        out_config = tt_output.memory_config()
        assert out_config.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
        assert out_config.buffer_type == ttnn.BufferType.L1
        out_spec = out_config.shard_spec
        assert out_spec.grid == memory_config.shard_spec.grid
        assert out_spec.orientation == memory_config.shard_spec.orientation
        out_rows = -(-(-(-m // rows)) // 32) * 32  # roundup32(ceil(M / n_rows))
        out_width = -(-(-(-c_out // cols)) // 32) * 32  # roundup32(ceil(C_out / n_cols))
        assert list(out_spec.shape) == [out_rows, out_width]
        assert tt_output.layout == ttnn.TILE_LAYOUT and tt_output.dtype == dtype
        assert list(tt_output.shape) == [1, 1, m, c_out]

        actual = ttnn.to_torch(tt_output).float()
        expected = _reference(torch_input, torch_weight, torch_bias, H, W, p, s, d, relu)
        assert torch.isfinite(actual).all()
        assert_with_pcc(expected, actual, pcc or PCC[ttnn.bfloat8_b])
        if relu:
            assert (actual >= 0).all()
    return live


@pytest.mark.parametrize("with_bias,relu", [(True, True), (False, False)], ids=["bias_relu", "plain"])
@pytest.mark.parametrize("dtype,layout", LAYOUTS)
@pytest.mark.parametrize("case", SHAPES, ids=[_case_id(c) for c in SHAPES])
def test_conv2d_3x3_block_sharded(case, dtype, layout, with_bias, relu, device):
    torch.manual_seed(42)
    case_hw, shards = case
    H, W, c_in = case_hw[0]
    config = _block_config(H * W, c_in, layout, shards[layout])
    _run(case_hw, dtype, layout, config, device, with_bias=with_bias, relu=relu)


@pytest.mark.parametrize("with_bias", [True, False], ids=["bias", "no_bias"])
@pytest.mark.parametrize("dtype,layout", [LAYOUTS[0], LAYOUTS[1]])
def test_conv2d_3x3_block_sharded_relu_after_combine(dtype, layout, with_bias, device):
    """Each channel slice's partial is large and the slices cancel: the input repeats its first 32
    channels in the next 32, and the weights of the second slice are minus the first plus a small
    term. The output is the small term's convolution; ReLU applied to any partial would destroy it."""
    torch.manual_seed(5)
    (H, W, c_in), c_out = (16, 16, 64), 64

    def make_tensors():
        big = 8.0 * torch.randn(c_out, 32, 3, 3)
        small = torch.randn(c_out, 32, 3, 3)
        torch_weight = torch.cat([big, -big + small], dim=1).to(torch.bfloat16)

        def make_input():
            half = torch.randn(1, 1, H * W, 32)
            return torch.cat([half, half], dim=-1).to(torch.bfloat16)

        return torch_weight, make_input

    config = _block_config(H * W, c_in, layout, (64, 32))
    assert _pd.block_grid_shape(config.shard_spec)[1] == 2
    _run(
        ((H, W, c_in), c_out, 1, 1, 1),
        dtype,
        layout,
        config,
        device,
        with_bias=with_bias,
        relu=True,
        weight_dtype=ttnn.bfloat16,
        make_tensors=make_tensors,
        pcc=0.99,
    )


@pytest.mark.parametrize("dtype,layout", LAYOUTS)
def test_conv2d_3x3_block_sharded_col_major_offset_grid(dtype, layout, device):
    """COL_MAJOR orientation (height shards along x, channel slices along y) on a grid starting at
    (1, 1), dispatched again on fresh shards: the re-dispatch must hit the program cache."""
    torch.manual_seed(7)
    (H, W, c_in), c_out = (16, 32, 64), 64
    shard = (128, 32) if layout == ttnn.TILE_LAYOUT else (100, 16)
    config = _block_config(H * W, c_in, layout, shard, start=(1, 1), orientation=ttnn.ShardOrientation.COL_MAJOR)
    case_hw = ((H, W, c_in), c_out, 1, 1, 1)
    _run(case_hw, dtype, layout, config, device, with_bias=True, relu=True)
    n_programs = device.num_program_cache_entries()
    _run(case_hw, dtype, layout, config, device, with_bias=True, relu=True, dispatches=2)
    assert device.num_program_cache_entries() == n_programs, "re-dispatch on fresh shards missed the program cache"


MULTI_BLOCK_CASES = [
    pytest.param(ttnn.bfloat8_b, ttnn.TILE_LAYOUT, True, id="tile_bf8b_bias_relu"),
    pytest.param(ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, False, id="rm_bf16_relu"),
]


@pytest.mark.parametrize("dtype,layout,with_bias", MULTI_BLOCK_CASES)
def test_conv2d_3x3_block_sharded_multi_m_block(dtype, layout, with_bias, device, monkeypatch):
    """One-tile-row M-blocks inside 4-tile-row output shards: every grid row runs four combine rounds,
    so the credit / arrival counters, the slot CB and the PACK_RELU / L1-acc state cross rounds."""
    # The test pins the grid-row groups' structure (rows, ragged row, M-blocks): no spare-column groups.
    monkeypatch.setattr(_pd, "SPARE_MGROUPS", False)
    torch.manual_seed(42)
    (H, W, c_in), c_out = (16, 64, 64), 96
    shard = (128, 32) if layout == ttnn.TILE_LAYOUT else (128, 16)
    config = _block_config(H * W, c_in, layout, shard)
    real_select = _pd.select_blocking

    def one_tile_row_blocks(g, grid, l1_budget):
        b = real_select(g, grid, l1_budget)
        assert g.k_split and b.core_m_tiles == 4, "the test expects a K split and a 4-tile-row output shard"
        fit = _pd._footprint(g, b.core_m_tiles, 1, b.block_k_tiles, b.block_n_tiles, nslice=b.nslice)
        return _pd.Conv3x3Blocking(
            grid_x=b.grid_x,
            core_m_tiles=b.core_m_tiles,
            num_m_groups=b.num_m_groups,
            block_m_tiles=1,
            block_k_tiles=b.block_k_tiles,
            block_n_tiles=b.block_n_tiles,
            out_depth_row_groups=b.out_depth_row_groups,
            **fit,
        )

    monkeypatch.setattr(_op_module, "select_blocking", one_tile_row_blocks)
    _run(((H, W, c_in), c_out, 1, 1, 1), dtype, layout, config, device, with_bias=with_bias, relu=True)


@pytest.mark.parametrize("dtype,layout", LAYOUTS)
def test_conv2d_3x3_block_sharded_column_weight_mcast_ragged_row(dtype, layout, device, monkeypatch):
    """One-tile-row M-blocks over 20 output tile-rows on 3 grid rows of 7-tile-row shards: rows 0 and 1 run 7
    M-blocks (21 weight multicast rounds per column, interleaved with 7 combine rounds), the last row runs 6 and
    must stream its own weight slice. W_out = 40, so a tile-row's taps span two image rows. The grid sits on the
    device's last 3 rows, so the M split stays on them (block_compute_rows), and it keeps the uniform split's ragged
    last row (no work-balanced ranges, which may move the short row off the grid's edge)."""
    # The test pins the grid-row groups' structure (rows, ragged row, M-blocks): no spare-column groups.
    monkeypatch.setattr(_pd, "SPARE_MGROUPS", False)
    torch.manual_seed(5)
    (H, W, c_in), c_out = (16, 40, 64), 64
    shard = (224, 32) if layout == ttnn.TILE_LAYOUT else (224, 16)
    config = _block_config(H * W, c_in, layout, shard, start=(0, device.compute_with_storage_grid_size().y - 3))
    real_select = _pd.select_blocking
    real_column_mcast = _pd._column_weight_mcast
    channels = []

    def one_tile_row_blocks(g, grid, l1_budget):
        b = real_select(g, grid, l1_budget)
        assert g.k_split and b.core_m_tiles == 7 and b.num_m_groups == 3
        fit = _pd._footprint(g, b.core_m_tiles, 1, b.block_k_tiles, b.block_n_tiles, nslice=b.nslice)
        return _pd.Conv3x3Blocking(
            grid_x=b.grid_x,
            core_m_tiles=b.core_m_tiles,
            num_m_groups=b.num_m_groups,
            block_m_tiles=1,
            block_k_tiles=b.block_k_tiles,
            block_n_tiles=b.block_n_tiles,
            out_depth_row_groups=b.out_depth_row_groups,
            **fit,
        )

    def recording_column_mcast(input_tensor, work, m_blocks_per_index):
        channel = real_column_mcast(input_tensor, work, m_blocks_per_index)
        channels.append((channel, list(work)))
        return channel

    monkeypatch.setattr(_op_module, "select_blocking", one_tile_row_blocks)
    monkeypatch.setattr(_op_module, "balanced_block_ranges", lambda *args: None)
    monkeypatch.setattr(_pd, "_column_weight_mcast", recording_column_mcast)
    _run(((H, W, c_in), c_out, 1, 1, 1), dtype, layout, config, device, with_bias=True, relu=True)

    ((channel, work),) = channels
    assert channel is not None, "the column weight multicast channel was not built"
    receivers, group_size, _ = channel
    k_cols = _pd.block_grid_shape(config.shard_spec)[1]
    assert group_size == 2 and receivers.num_cores() == 2 * k_cols
    assert not any(receivers.contains(core) for core, row, _ in work if row == 2), "the ragged row streams its own"


@pytest.mark.parametrize("dtype,layout,with_bias", MULTI_BLOCK_CASES)
def test_conv2d_3x3_block_sharded_multi_m_block_extended_rows(dtype, layout, with_bias, device, monkeypatch):
    """One-tile-row M-blocks on extended rows: 20 output tile-rows on 3 shard rows of 7 run on 10 grid rows of 2, so
    every core runs two combine rounds and stores two blocks to output shards it does not sit on."""
    torch.manual_seed(42)
    (H, W, c_in), c_out = (16, 40, 64), 96
    shard = (224, 32) if layout == ttnn.TILE_LAYOUT else (224, 16)
    config = _block_config(H * W, c_in, layout, shard)
    if device.compute_with_storage_grid_size().y < 10:
        pytest.skip("needs 10 grid rows")
    real_select = _pd.select_blocking

    def one_tile_row_blocks(g, grid, l1_budget):
        b = real_select(g, grid, l1_budget)
        assert g.k_split and g.remote_output and grid[0] == 10 and b.core_m_tiles == 2, (grid, b)
        fit = _pd._footprint(g, b.core_m_tiles, 1, b.block_k_tiles, b.block_n_tiles, nslice=b.nslice)
        return _pd.Conv3x3Blocking(
            grid_x=b.grid_x,
            core_m_tiles=b.core_m_tiles,
            num_m_groups=b.num_m_groups,
            block_m_tiles=1,
            block_k_tiles=b.block_k_tiles,
            block_n_tiles=b.block_n_tiles,
            out_depth_row_groups=b.out_depth_row_groups,
            **fit,
        )

    monkeypatch.setattr(_op_module, "select_blocking", one_tile_row_blocks)
    _run(((H, W, c_in), c_out, 1, 1, 1), dtype, layout, config, device, with_bias=with_bias, relu=True)


@pytest.mark.parametrize("dtype,layout", LAYOUTS)
def test_conv2d_3x3_block_sharded_column_weight_mcast_ragged_extended_row(dtype, layout, device, monkeypatch):
    """The ragged-row case on extended rows: the 3-row shard grid starts at y = 3, so 7 grid rows reach a 10-row
    device's edge; 20 output tile-rows split 3 per row, the last row runs 2 one-tile-row M-blocks (6 x 3 + 2) and
    streams its own weights while the 6 full rows share each column's multicast."""
    # The test pins the grid-row groups' structure (rows, ragged row, M-blocks): no spare-column groups.
    monkeypatch.setattr(_pd, "SPARE_MGROUPS", False)
    torch.manual_seed(5)
    (H, W, c_in), c_out = (16, 40, 64), 64
    shard = (224, 32) if layout == ttnn.TILE_LAYOUT else (224, 16)
    if device.compute_with_storage_grid_size().y != 10:
        pytest.skip("pinned to a 10-row grid")
    config = _block_config(H * W, c_in, layout, shard, start=(0, 3))
    real_select = _pd.select_blocking
    real_column_mcast = _pd._column_weight_mcast
    channels = []

    def one_tile_row_blocks(g, grid, l1_budget):
        b = real_select(g, grid, l1_budget)
        assert g.k_split and g.remote_output and grid[0] == 7 and b.core_m_tiles == 3 and b.num_m_groups == 7
        fit = _pd._footprint(g, b.core_m_tiles, 1, b.block_k_tiles, b.block_n_tiles, nslice=b.nslice)
        return _pd.Conv3x3Blocking(
            grid_x=b.grid_x,
            core_m_tiles=b.core_m_tiles,
            num_m_groups=b.num_m_groups,
            block_m_tiles=1,
            block_k_tiles=b.block_k_tiles,
            block_n_tiles=b.block_n_tiles,
            out_depth_row_groups=b.out_depth_row_groups,
            **fit,
        )

    def recording_column_mcast(input_tensor, work, m_blocks_per_index):
        channel = real_column_mcast(input_tensor, work, m_blocks_per_index)
        channels.append((channel, list(work)))
        return channel

    monkeypatch.setattr(_op_module, "select_blocking", one_tile_row_blocks)
    monkeypatch.setattr(_pd, "_column_weight_mcast", recording_column_mcast)
    _run(((H, W, c_in), c_out, 1, 1, 1), dtype, layout, config, device, with_bias=True, relu=True)

    ((channel, work),) = channels
    assert channel is not None, "the column weight multicast channel was not built"
    receivers, group_size, _ = channel
    k_cols = _pd.block_grid_shape(config.shard_spec)[1]
    assert group_size == 6 and receivers.num_cores() == 6 * k_cols
    assert not any(receivers.contains(core) for core, row, _ in work if row == 6), "the ragged row streams its own"


def test_conv2d_3x3_block_sharded_grid_column_mismatch(device, expect_error):
    """A shard grid with more columns than C_in has channel slices is refused before dispatch."""
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))])
    config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [64, 32], ttnn.ShardOrientation.ROW_MAJOR),
    )
    tt_input = ttnn.from_torch(
        torch.randn(1, 1, 256, 64).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=config,
    )
    weight = torch.zeros(32, 64, 3, 3)
    weight_config = _block_config(256, 64, ttnn.TILE_LAYOUT, (64, 32))
    tt_weight = prepare_conv2d_weights(
        weight,
        padding=1,
        input_dtype=ttnn.bfloat16,
        input_layout=ttnn.TILE_LAYOUT,
        weight_dtype=ttnn.bfloat16,
        input_memory_config=weight_config,
        device=device,
    )
    with expect_error(ValueError, "columns"):
        conv2d_3x3(tt_input, tt_weight, input_height=16, input_width=16, padding=1)


@pytest.mark.parametrize("dtype,layout", [LAYOUTS[0], LAYOUTS[2]])
def test_conv2d_3x3_block_sharded_writer_only_scatter(dtype, layout, device, monkeypatch):
    """COMBINE_SPLIT_NOCS off: the writer scatters every N-slice on NoC1 and the reader sends nothing."""
    torch.manual_seed(11)
    monkeypatch.setattr(_pd, "COMBINE_SPLIT_NOCS", False)
    case_hw, shards = SHAPES[4]
    H, W, c_in = case_hw[0]
    config = _block_config(H * W, c_in, layout, shards[layout])
    _run(case_hw, dtype, layout, config, device, with_bias=True, relu=True)


SPARE_CASES = [
    # res4.x: 8 slices, 13 groups of 5 tile-rows (10 grid rows + 3 spare columns), 1-tile N-slices.
    pytest.param(((32, 64, 256), 256, 1, 1, 1), (256, 32), ttnn.ShardOrientation.ROW_MAJOR, id="res4_8x8"),
    # res5.0: 2-tile N-slices, 5-row groups (4 + 1: the wide short group), the padded-slab prelude.
    pytest.param(((32, 64, 512), 512, 2, 1, 2), (224, 64), ttnn.ShardOrientation.ROW_MAJOR, id="res5_0_8x10"),
    # 2 slices, d = 8: 16 groups of 4 (6 spare columns), C_out 96: a short last N-slice.
    pytest.param(((32, 64, 64), 96, 8, 1, 8), (256, 32), ttnn.ShardOrientation.ROW_MAJOR, id="k2_d8_co96"),
    # COL_MAJOR: the slices run down y, the spare groups along the rows below the shard grid.
    pytest.param(((32, 64, 64), 64, 1, 1, 1), (256, 32), ttnn.ShardOrientation.COL_MAJOR, id="k2_col_major"),
]


@pytest.mark.parametrize("dtype,layout", [LAYOUTS[1], LAYOUTS[2]])
@pytest.mark.parametrize("case_hw,shard,orientation", SPARE_CASES)
def test_conv2d_3x3_block_sharded_spare_groups(case_hw, shard, orientation, dtype, layout, device, monkeypatch):
    """Spare-column M-groups: more M-groups than grid rows, transposed into the device columns past the shard grid
    (rows below it on COL_MAJOR), their weights forwarded from the column streams; owners store remotely. The output
    keeps its shard spec; the second dispatch on fresh shards must hit the program cache."""
    torch.manual_seed(3)
    grid = device.compute_with_storage_grid_size()
    if (grid.x, grid.y) != (11, 10):
        pytest.skip("pinned to the 11 x 10 grid")
    (H, W, c_in), c_out, p, s, d = case_hw
    config = _block_config(H * W, c_in, layout, shard, orientation=orientation)
    real_select = _op_module.select_blocking
    seen = []

    def recording(g, work_grid, l1_budget):
        b = real_select(g, work_grid, l1_budget)
        seen.append((g.block_rows, b.num_m_groups, b.nslice, b.max_m_blocks_per_core))
        return b

    monkeypatch.setattr(_op_module, "select_blocking", recording)
    _run(case_hw, dtype, layout, config, device, with_bias=True, relu=True)
    block_rows, groups, nslice, m_blocks = seen[-1]
    assert nslice and m_blocks == 1 and 0 < block_rows < groups, seen[-1]
    n_programs = device.num_program_cache_entries()
    _run(case_hw, dtype, layout, config, device, with_bias=True, relu=True, dispatches=2)
    assert device.num_program_cache_entries() == n_programs, "re-dispatch on fresh shards missed the program cache"


WEIGHT_PLACEMENT_CASES = [
    # res4.x: 1-tile N-slices, 9-tile rounds (spare groups and their forwarders).
    pytest.param(((32, 64, 256), 256, 1, 1, 1), (256, 32), id="res4_8x8"),
    # res5.0: 2-tile N-slices, 36-tile rounds.
    pytest.param(((32, 64, 512), 512, 2, 1, 2), (224, 64), id="res5_0_8x10"),
    # C_out 96 over 2 slices: 3 tiles do not split into whole N-slices, so the weights stay DRAM interleaved.
    pytest.param(((32, 64, 64), 96, 8, 1, 8), (256, 32), id="k2_d8_co96"),
]


@pytest.mark.parametrize("case_hw,shard", WEIGHT_PLACEMENT_CASES)
def test_conv2d_3x3_block_sharded_weight_page_runs(case_hw, shard, device, monkeypatch):
    """BLOCK prepared weights sit one ND shard per N-slice weight round (all K of one channel slice x one column's
    N-slice), so the column weight stream reads a round as one run (the writer's NSLICE_WEIGHT_PAGE_RUNS). The same
    weights placed DRAM interleaved (one-tile reads) must give a bit-identical output."""
    torch.manual_seed(5)
    grid = device.compute_with_storage_grid_size()
    if (grid.x, grid.y) != (11, 10):
        pytest.skip("pinned to the 11 x 10 grid")
    (H, W, c_in), c_out, p, s, d = case_hw
    dtype, layout = ttnn.bfloat8_b, ttnn.TILE_LAYOUT
    config = _block_config(H * W, c_in, layout, shard)
    torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
    torch_bias = torch.randn(c_out).to(torch.bfloat16)
    torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    kwargs = dict(
        stride=s,
        padding=p,
        dilation=d,
        input_dtype=dtype,
        input_layout=layout,
        weight_dtype=ttnn.bfloat8_b,
        input_memory_config=config,
        device=device,
    )

    def run(tt_weight):
        tt_bias = prepare_conv2d_bias(torch_bias, **kwargs)
        tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device, memory_config=config)
        tt_output = conv2d_3x3(
            tt_input,
            tt_weight,
            input_height=H,
            input_width=W,
            bias=tt_bias,
            padding=p,
            stride=s,
            dilation=d,
            activation="relu",
            compute_kernel_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False
            ),
        )
        return ttnn.to_torch(tt_output).float()

    placed = prepare_conv2d_weights(torch_weight, **kwargs)
    spec = placed.memory_config().nd_shard_spec
    n_tiles = -(-c_out // 32)
    out_n = -(-n_tiles // -(-c_in // shard[1]))
    if n_tiles % out_n:
        assert spec is None, spec
    else:
        assert spec is not None and list(spec.shard_shape)[-2:] == [3 * (-(-3 * shard[1] // 32) * 32), out_n * 32]
    out_placed = run(placed)
    monkeypatch.setattr(_op_module, "_block_weight_memory_config", lambda *args, **kwargs: ttnn.DRAM_MEMORY_CONFIG)
    interleaved = prepare_conv2d_weights(torch_weight, **kwargs)
    assert interleaved.memory_config().nd_shard_spec is None
    out_interleaved = run(interleaved)
    assert torch.equal(out_placed, out_interleaved)
    expected = _reference(torch_input, torch_weight, torch_bias, H, W, p, s, d, True)
    assert_with_pcc(expected, out_placed, PCC[ttnn.bfloat8_b])


HOST_FILL_CASES = [
    # res4.0: stride 2 (the S2_TAPS prelude's own ring), 13 groups of 5 (3 spare columns): spare members split on NoC1.
    pytest.param(((64, 128, 256), 256, 1, 2, 1), (1024, 32), ttnn.ShardOrientation.ROW_MAJOR, id="res4_0_s2"),
    # res5.1: grid rows that read >= 8 KB from higher shards split as before, and the spare members split too.
    pytest.param(((32, 64, 512), 512, 4, 1, 4), (224, 64), ttnn.ShardOrientation.ROW_MAJOR, id="res5_1"),
    # 2 slices, d = 8: 6 spare columns of 2 members, C_out 96 (a short last N-slice).
    pytest.param(((32, 64, 64), 96, 8, 1, 8), (256, 32), ttnn.ShardOrientation.ROW_MAJOR, id="k2_d8_co96"),
    # COL_MAJOR: the owner shards run down x, the spare groups along the rows below the shard grid.
    pytest.param(((32, 64, 64), 64, 1, 1, 1), (256, 32), ttnn.ShardOrientation.COL_MAJOR, id="k2_col_major"),
    # aspp.convs.3: 11 slices, the last one short (2 of 3 tiles): both RISCs read its real tiles and zero the rest.
    pytest.param(((32, 64, 1024), 256, 18, 1, 18), (256, 96), ttnn.ShardOrientation.ROW_MAJOR, id="aspp3_short_slice"),
]


@pytest.mark.parametrize("case_hw,shard,orientation", HOST_FILL_CASES)
def test_conv2d_3x3_block_sharded_host_fill(case_hw, shard, orientation, device, monkeypatch):
    """BLOCK_HOST_FILL: the host plans every core's slab reads (owner core, shard offset, slab row, rows); a split
    core's writer (grid rows reading >= 8 KB from higher shards, spare-column members: the upper half of their slab)
    reads its part on NoC1 from kernel start. Bit-identical to the device-planned fill (BLOCK_HOST_FILL = False), and
    the plan is taken, with writer reads on some core."""
    torch.manual_seed(13)
    grid = device.compute_with_storage_grid_size()
    if (grid.x, grid.y) != (11, 10):
        pytest.skip("pinned to the 11 x 10 grid")
    (H, W, c_in), c_out, p, s, d = case_hw
    dtype, layout = ttnn.bfloat8_b, ttnn.TILE_LAYOUT
    config = _block_config(H * W, c_in, layout, shard, orientation=orientation)
    torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
    torch_bias = torch.randn(c_out).to(torch.bfloat16)
    torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    real_args = _pd._host_fill_args
    writer_batches = []

    def recording(*args, **kwargs):
        reader, writer = real_args(*args, **kwargs)
        writer_batches.append(writer[0])
        return reader, writer

    monkeypatch.setattr(_pd, "_host_fill_args", recording)
    outputs = {}
    for host_fill in (True, False):
        monkeypatch.setattr(_pd, "BLOCK_HOST_FILL", host_fill)
        device.disable_and_clear_program_cache()
        writer_batches.clear()
        kwargs = dict(
            stride=s,
            padding=p,
            dilation=d,
            input_dtype=dtype,
            input_layout=layout,
            weight_dtype=ttnn.bfloat8_b,
            input_memory_config=config,
            device=device,
        )
        tt_weight = prepare_conv2d_weights(torch_weight, **kwargs)
        tt_bias = prepare_conv2d_bias(torch_bias, **kwargs)
        tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device, memory_config=config)
        tt_output = conv2d_3x3(
            tt_input,
            tt_weight,
            input_height=H,
            input_width=W,
            bias=tt_bias,
            padding=p,
            stride=s,
            dilation=d,
            activation="relu",
            compute_kernel_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False
            ),
        )
        assert tt_output.memory_config().shard_spec.grid == config.shard_spec.grid
        outputs[host_fill] = ttnn.to_torch(tt_output).float()
        if host_fill:
            assert writer_batches and max(writer_batches) > 0, "host-planned fill not taken, or no core split"
        else:
            assert not writer_batches
    assert torch.equal(outputs[True], outputs[False]), "the host-planned fill differs from the device-planned one"
    expected = _reference(torch_input, torch_weight, torch_bias, H, W, p, s, d, True)
    assert_with_pcc(expected, outputs[True], PCC[ttnn.bfloat8_b])


ROUND0_CASES = [
    # res5.0: 36-tile rounds: the round-0 readers sit on the rows the NoC1 model picks (Blackhole).
    pytest.param(((32, 64, 512), 512, 2, 1, 2), (224, 64), id="res5_0_8x10"),
    # res4.x: 9-tile rounds: the column senders read round 0.
    pytest.param(((32, 64, 256), 256, 1, 1, 1), (256, 32), id="res4_8x8"),
]


@pytest.mark.parametrize("case_hw,shard", ROUND0_CASES)
def test_conv2d_3x3_block_sharded_round0_reader(case_hw, shard, device, monkeypatch):
    """N-slice round 0 (NSLICE_ROUND0): one round-0 reader per column multicasts round 0 without a handshake, the
    column sender relays it to the spare members. The readers on the model's rows and on the sender rows must give a
    bit-identical output."""
    torch.manual_seed(13)
    grid = device.compute_with_storage_grid_size()
    if (grid.x, grid.y) != (11, 10):
        pytest.skip("pinned to the 11 x 10 grid")
    (H, W, c_in), c_out, p, s, d = case_hw
    dtype, layout = ttnn.bfloat8_b, ttnn.TILE_LAYOUT
    config = _block_config(H * W, c_in, layout, shard)
    torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
    torch_bias = torch.randn(c_out).to(torch.bfloat16)
    torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    kwargs = dict(
        stride=s,
        padding=p,
        dilation=d,
        input_dtype=dtype,
        input_layout=layout,
        weight_dtype=ttnn.bfloat8_b,
        input_memory_config=config,
        device=device,
    )
    tt_weight = prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = prepare_conv2d_bias(torch_bias, **kwargs)
    real_args = _pd._round0_args
    readers = []

    def recording(*args):
        out = real_args(*args)
        readers.append(sorted(xy for xy, a in out.items() if a[0]))
        return out

    monkeypatch.setattr(_pd, "_round0_args", recording)

    def run():
        device.disable_and_clear_program_cache()
        tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device, memory_config=config)
        tt_output = conv2d_3x3(
            tt_input,
            tt_weight,
            input_height=H,
            input_width=W,
            bias=tt_bias,
            padding=p,
            stride=s,
            dilation=d,
            activation="relu",
            compute_kernel_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False
            ),
        )
        return ttnn.to_torch(tt_output).float()

    out_model = run()
    monkeypatch.setattr(_pd, "NSLICE_ROUND0_MODEL", False)
    out_senders = run()
    device.enable_program_cache()
    assert len(readers) == 2 and len(readers[0]) == len(readers[1]) > 0, readers
    # One reader per column; the column senders sit on the staggered diagonal (column c on grid row c).
    assert readers[1] == sorted((c, c) for c in range(len(readers[1]))), readers[1]
    model = (
        ttnn.device.is_blackhole(device)
        and device.dram_core_from_logical_core(ttnn.CoreCoord(0, 0)).y in (12, 14)
        and c_out * 9 * shard[1] // (c_in // shard[1]) // 1024 >= _pd.NSLICE_ROUND0_MODEL_MIN_BYTES // 1024
    )
    if model:
        # The model moves the readers whose DRAM read paths the senders' rows share (res5.0: five of eight).
        assert readers[0] != readers[1], readers
    assert torch.equal(out_model, out_senders)
    expected = _reference(torch_input, torch_weight, torch_bias, H, W, p, s, d, True)
    assert_with_pcc(expected, out_model, PCC[ttnn.bfloat8_b])
