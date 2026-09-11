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
