# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for quasar TM ops."""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp
from tests.ttnn.utils_for_testing import assert_reshape as _assert_reshape

_LAYOUTS = [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]
_LAYOUT_IDS = ["TILE", "RM"]


L1_INTERLEAVED = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)


def _explicit_height_shard_config(device, ncores, sh, sw, buffer_type=ttnn.BufferType.L1):
    compute_grid = device.compute_with_storage_grid_size()
    if ncores > compute_grid.x * compute_grid.y:
        pytest.skip(f"Device has {compute_grid.x * compute_grid.y} cores, test needs {ncores}")
    spec = ttnn.ShardSpec(
        ttnn.num_cores_to_corerangeset(ncores, compute_grid, True),
        (sh, sw),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type, spec)


def _run_quasar_slice(shape, begins, ends, step, imc, omc, device):
    torch.manual_seed(12345)
    x = torch.rand(shape, dtype=torch.bfloat16)
    ttnn_in = ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=imc)
    result = ttnn.experimental.quasar.slice(ttnn_in, list(begins), list(ends), list(step), memory_config=omc)

    actual = result.memory_config()
    assert actual.memory_layout == omc.memory_layout
    if omc.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        assert actual.shard_spec is not None

    slices = tuple(slice(b, e, s) for b, e, s in zip(begins, ends, step))
    ref = x[slices]
    got = ttnn.to_torch(result.cpu().to(ttnn.ROW_MAJOR_LAYOUT))
    assert_with_ulp(expected_result=ref, actual_result=got, ulp_threshold=0)


@pytest.mark.parametrize(
    "shape, begins, ends, step, in_shard, out_shard",
    [
        pytest.param(
            (1, 1, 52, 64), (0, 0, 0, 0), (1, 1, 26, 64), (1, 1, 1, 1), (4, 13, 64), (2, 13, 64), id="A_coalesced"
        ),
        pytest.param(
            (1, 1, 32, 64), (0, 0, 0, 16), (1, 1, 32, 32), (1, 1, 1, 1), (4, 8, 64), (4, 8, 16), id="F_w_begin_aligned"
        ),
        pytest.param(
            (1, 1, 32, 52),
            (0, 0, 0, 0),
            (1, 1, 26, 52),
            (1, 1, 1, 1),
            (4, 8, 52),
            (2, 13, 52),
            id="H_w_unaligned_stride",
        ),
        # HS→L1 fallback with misaligned W-begin. HS→HS output triggers an unrelated Quasar
        # SliceRmProgramFactory bug in the misalignment+HS-output path; interleaved output still
        # exercises the misaligned-begin fallback route through the same predicate.
        pytest.param(
            (1, 1, 32, 64),
            (0, 0, 0, 1),
            (1, 1, 32, 32),
            (1, 1, 1, 1),
            (4, 8, 64),
            None,
            id="G_w_begin_misaligned_routes_to_rm_fallback",
        ),
    ],
)
def test_quasar_slice_row_major_height_sharded_nontile_aligned(shape, begins, ends, step, in_shard, out_shard, device):
    imc = _explicit_height_shard_config(device, *in_shard)
    omc = _explicit_height_shard_config(device, *out_shard) if out_shard is not None else L1_INTERLEAVED
    _run_quasar_slice(shape, begins, ends, step, imc, omc, device)


def test_quasar_slice_tile_tensor_args(device):
    shape = (1, 1, 64, 64)
    starts, ends = [0, 0, 32, 0], [1, 1, 64, 64]
    torch.manual_seed(12345)
    x = torch.rand(shape, dtype=torch.bfloat16)
    ttnn_in = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    start_t = ttnn.from_torch(torch.tensor(starts), device=device)
    end_t = ttnn.from_torch(torch.tensor(ends), device=device)
    result = ttnn.experimental.quasar.slice(ttnn_in, start_t, end_t, slice_dim=2, num_devices=2)
    got = ttnn.to_torch(result.cpu().to(ttnn.ROW_MAJOR_LAYOUT))
    assert_with_ulp(expected_result=x[:, :, 32:64, :], actual_result=got, ulp_threshold=0)


# ---------------------------------------------------------------------------
# ND shard-spec reshape tests — quasar entry point
# Mirrors test_universal_input_tm_reshape.py's test_reshape_nd_* suite so that
# both ttnn.reshape and ttnn.experimental.quasar.reshape are covered.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", _LAYOUTS, ids=_LAYOUT_IDS)
@pytest.mark.parametrize(
    "nd_shard_shape, strategy_name",
    [
        ([1, 1, 32, 128], "height"),
        ([1, 1, 64, 64], "width"),
    ],
    ids=["nd_height", "nd_width"],
)
def test_quasar_reshape_nd_shard_spec_normalized_to_2d_input(device, layout, nd_shard_shape, strategy_name):
    """An input whose MemoryConfig was built from a rank-4 NdShardSpec that normalizes to a 2D
    layout must reshape to a lower-rank shape without tripping the shard-rank check, using the
    quasar reshape entry point."""
    input_shape = [1, 1, 64, 128]
    output_shape = [1, 128, 64]

    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=ttnn.Shape(nd_shard_shape), grid=grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    in_memcfg = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec)

    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output = torch_input.reshape(output_shape)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=in_memcfg)

    in_mc = tt_input.memory_config()
    expected_layout = (
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED if strategy_name == "height" else ttnn.TensorMemoryLayout.WIDTH_SHARDED
    )
    assert in_mc.memory_layout == expected_layout, f"expected {expected_layout}, got {in_mc.memory_layout}"
    assert in_mc.nd_shard_spec is not None, "input should retain the higher-rank nd_shard_spec"
    assert len(in_mc.nd_shard_spec.shard_shape) == 4, "retained nd_shard_spec should be rank 4"

    tt_output = ttnn.experimental.quasar.reshape(tt_input, output_shape, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)
    _assert_reshape(torch_output, actual, ttnn.bfloat16)


@pytest.mark.parametrize("layout", _LAYOUTS, ids=_LAYOUT_IDS)
@pytest.mark.parametrize(
    "input_shape, output_shape, nd_shard_shape, is_genuine_nd",
    [
        ([1, 1, 64, 128], [1, 1, 128, 64], [1, 1, 64, 64], False),
        ([2, 2, 64, 64], [2, 2, 32, 128], [1, 1, 64, 64], True),
        ([1, 1, 64, 128], [1, 1, 64, 128], [1, 1, 64, 64], False),
    ],
    ids=["normalizes_2d", "genuine_nd", "same_shape"],
)
def test_quasar_reshape_nd_sharded_output(device, layout, input_shape, output_shape, nd_shard_shape, is_genuine_nd):
    """Reshape with an ND-sharded output memory_config must succeed and preserve values via the
    quasar reshape entry point, covering normalizes-to-2D, genuinely-ND, and same-shape cases."""
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=ttnn.Shape(nd_shard_shape), grid=grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    out_memcfg = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec)

    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output = torch_input.reshape(output_shape)

    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_output = ttnn.experimental.quasar.reshape(tt_input, output_shape, memory_config=out_memcfg)

    out_mc = tt_output.memory_config()
    assert out_mc.is_sharded()
    if is_genuine_nd:
        assert (
            out_mc.memory_layout == ttnn.TensorMemoryLayout.ND_SHARDED
        ), f"expected ND_SHARDED, got {out_mc.memory_layout}"
        assert out_mc.nd_shard_spec is not None
        assert list(out_mc.nd_shard_spec.shard_shape) == list(out_memcfg.nd_shard_spec.shard_shape), (
            f"output nd shard shape {list(out_mc.nd_shard_spec.shard_shape)} != "
            f"requested {list(out_memcfg.nd_shard_spec.shard_shape)}"
        )
    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)
    _assert_reshape(torch_output, actual, ttnn.bfloat16)


@pytest.mark.parametrize("layout", _LAYOUTS, ids=_LAYOUT_IDS)
def test_quasar_reshape_nd_sharded_input_rank_change_inherited(device, layout):
    """A genuinely ND-sharded input reshaped to a different rank with an inherited output config
    must adapt the shard shape to the new rank via the quasar reshape entry point."""
    input_shape = [2, 2, 64, 64]
    output_shape = [4, 64, 64]

    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=ttnn.Shape([1, 1, 64, 64]), grid=grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    in_memcfg = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec)

    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output = torch_input.reshape(output_shape)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=in_memcfg)
    assert tt_input.memory_config().memory_layout == ttnn.TensorMemoryLayout.ND_SHARDED

    tt_output = ttnn.experimental.quasar.reshape(tt_input, output_shape)

    out_mc = tt_output.memory_config()
    assert (
        out_mc.memory_layout == ttnn.TensorMemoryLayout.ND_SHARDED
    ), f"expected ND_SHARDED output, got {out_mc.memory_layout}"
    assert out_mc.nd_shard_spec is not None
    assert out_mc.nd_shard_spec.shard_shape.rank == len(output_shape)

    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)
    _assert_reshape(torch_output, actual, ttnn.bfloat16)


@pytest.mark.parametrize("layout", _LAYOUTS, ids=_LAYOUT_IDS)
@pytest.mark.parametrize("explicit_output", [False, True], ids=["default_output", "explicit_output"])
def test_quasar_reshape_nd_sharded_input(device, layout, explicit_output):
    """A genuinely ND-sharded input must reshape and preserve values whether the output
    memory_config is inherited (default) or passed explicitly, via the quasar reshape entry point."""
    input_shape = [2, 2, 64, 64]
    output_shape = [2, 2, 32, 128]

    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=ttnn.Shape([1, 1, 64, 64]), grid=grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    in_memcfg = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec)

    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output = torch_input.reshape(output_shape)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=in_memcfg)
    assert tt_input.memory_config().memory_layout == ttnn.TensorMemoryLayout.ND_SHARDED

    if explicit_output:
        tt_output = ttnn.experimental.quasar.reshape(tt_input, output_shape, memory_config=in_memcfg)
    else:
        tt_output = ttnn.experimental.quasar.reshape(tt_input, output_shape)

    out_mc = tt_output.memory_config()
    assert (
        out_mc.memory_layout == ttnn.TensorMemoryLayout.ND_SHARDED
    ), f"expected ND_SHARDED output, got {out_mc.memory_layout}"
    assert out_mc.nd_shard_spec is not None
    if explicit_output:
        assert list(out_mc.nd_shard_spec.shard_shape) == list(in_memcfg.nd_shard_spec.shard_shape), (
            f"explicit output shard shape {list(out_mc.nd_shard_spec.shard_shape)} != "
            f"requested {list(in_memcfg.nd_shard_spec.shard_shape)}"
        )
    else:
        assert list(out_mc.nd_shard_spec.shard_shape) == [
            1,
            1,
            32,
            128,
        ], f"derived output shard shape {list(out_mc.nd_shard_spec.shard_shape)} != expected [1, 1, 32, 128]"

    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)
    _assert_reshape(torch_output, actual, ttnn.bfloat16)


# Mirrors test_reshape_explicit_normalized_nd_memory_config in
# tests/ttnn/nightly/unit_tests/operations/data_movement/test_universal_input_tm_reshape.py, so the
# quasar copy of the nd_shard_spec strip does not drift.
@pytest.mark.parametrize("layout", _LAYOUTS, ids=_LAYOUT_IDS)
def test_quasar_reshape_explicit_normalized_nd_memory_config(device, layout):
    """A tensor-derived config that normalized from an NdShardSpec to BLOCK_SHARDED carries both
    specs; handing it to a rank-changing reshape must not carry the higher-rank nd_shard_spec into
    the output's spec."""
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
    # Default ROUND_ROBIN_1D: GRID_2D rejects a rank-4 shard shape, so the donor would not allocate.
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=ttnn.Shape([1, 1, 32, 32]), grid=grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    donor = ttnn.from_torch(
        torch.zeros([1, 1, 64, 64], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=layout,
        device=device,
        memory_config=ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec),
    )
    explicit_mc = donor.memory_config()
    assert explicit_mc.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED, explicit_mc.memory_layout
    assert explicit_mc.shard_spec is not None, "normalized config should carry a 2D shard_spec"
    assert explicit_mc.nd_shard_spec is not None, "normalized config should retain the nd_shard_spec"
    assert len(explicit_mc.nd_shard_spec.shard_shape) == 4, "retained nd_shard_spec should be rank 4"

    input_shape = [1, 1, 64, 64]
    output_shape = [1, 2, 32, 64]

    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_output = torch_input.reshape(output_shape)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    tt_output = ttnn.experimental.quasar.reshape(tt_input, output_shape, memory_config=explicit_mc)

    out_mc = tt_output.memory_config()
    assert out_mc.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED, out_mc.memory_layout
    assert out_mc.nd_shard_spec is None or len(out_mc.nd_shard_spec.shard_shape) <= len(tt_output.shape)

    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)
    _assert_reshape(torch_output, actual, ttnn.bfloat16)


@pytest.mark.parametrize("layout", _LAYOUTS, ids=_LAYOUT_IDS)
def test_quasar_reshape_same_shape_functionally_identical_config_is_noop(device, layout):
    """A same-shape reshape asked for the input's own distribution, expressed as a plain 2D config
    rather than the input's normalized-from-ND one, must stay a no-op."""
    shape = [1, 1, 64, 128]
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=ttnn.Shape([1, 1, 32, 128]), grid=grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )

    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=layout,
        device=device,
        memory_config=ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec),
    )

    in_mc = tt_input.memory_config()
    assert in_mc.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED, in_mc.memory_layout
    assert in_mc.nd_shard_spec is not None, "normalized config should retain the nd_shard_spec"

    equivalent_mc = ttnn.MemoryConfig(in_mc.memory_layout, ttnn.BufferType.L1, in_mc.shard_spec)
    tt_output = ttnn.experimental.quasar.reshape(tt_input, shape, memory_config=equivalent_mc)

    assert (
        tt_output.buffer_address() == tt_input.buffer_address()
    ), "a functionally identical config must not allocate a new buffer"

    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)
    _assert_reshape(torch_input.reshape(shape), actual, ttnn.bfloat16)


# Mirrors test_reshape_nd_input_logical_only_update_is_view /
# test_reshape_nd_input_last_dim_change_is_not_a_view in
# tests/ttnn/nightly/unit_tests/operations/data_movement/test_universal_input_tm_reshape.py.
# The quasar copy carries the identical zero-copy gate, so it needs the same coverage.
def test_quasar_reshape_nd_input_logical_only_update_is_view(device):
    """A genuinely ND-sharded input reshaped to a different logical shape with the same padded shape
    stays a metadata-only view (same buffer), and returns the cropped values."""
    padded_shape = [2, 2, 64, 64]
    logical_shape = [2, 2, 60, 64]

    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=ttnn.Shape([1, 1, 64, 64]), grid=grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )

    torch.manual_seed(0)
    torch_input = torch.randn(padded_shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec),
    )
    in_mc = tt_input.memory_config()
    assert (
        in_mc.memory_layout == ttnn.TensorMemoryLayout.ND_SHARDED
    ), f"input should stay ND_SHARDED, got {in_mc.memory_layout}"
    assert in_mc.shard_spec is None, "a genuinely ND config should have no 2D shard_spec"

    tt_output = ttnn.experimental.quasar.reshape(tt_input, logical_shape, padded_shape)

    assert list(tt_output.shape) == logical_shape, f"got {list(tt_output.shape)}"
    assert (
        tt_output.buffer_address() == tt_input.buffer_address()
    ), "a logical-only update on an ND tensor must stay a view, not round-trip through DRAM"

    expected = torch_input[:, :, : logical_shape[-2], :]
    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)
    _assert_reshape(expected, actual, ttnn.bfloat16)


def test_quasar_reshape_nd_input_last_dim_change_is_not_a_view(device):
    """Same padded shape but a changed logical last dim must go through the staging path and obey
    torch-reshape semantics, not return the input's bytes in place."""
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})  # 1 core
    # Shard must be tile-aligned for TILE layout; [1, 32, 32] over [2, 62, 32] gives 4 shards on
    # 1 core, so normalization is rejected and the config stays genuinely ND_SHARDED.
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=ttnn.Shape([1, 32, 32]), grid=grid, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )

    torch.manual_seed(0)
    # [2, 62, 32] and [2, 64, 31] both pad to [2, 64, 32] and have volume 3968, so the padded shape
    # and volume match while the last dim changes 32 -> 31.
    torch_input = torch.randn([2, 62, 32], dtype=torch.bfloat16)
    torch_output = torch_input.reshape([2, 64, 31])

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec),
    )
    assert tt_input.memory_config().memory_layout == ttnn.TensorMemoryLayout.ND_SHARDED

    tt_output = ttnn.experimental.quasar.reshape(tt_input, [2, 64, 31], [2, 64, 32])

    assert list(tt_output.shape) == [2, 64, 31], f"got {list(tt_output.shape)}"
    actual = ttnn.to_torch(tt_output).to(torch.bfloat16)
    _assert_reshape(torch_output, actual, ttnn.bfloat16)
