# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_interleaved_to_sharded.

The op moves a DRAM-interleaved bfloat16 TILE tensor into the L1 height or
width shards that memory_config names. The result must equal the input bit for
bit and carry exactly the requested placement.

Cases: the two production placements, plus the cases the spec's rules require
(orientation, partial last shard, multi-shard height map, padding
independence, special values, cache key, plan report, malformed requests). A
case outside the op's SUPPORTED is xfail-strict on its support refusal and
becomes a real test when the refinement lands.
"""

from __future__ import annotations

import copy
import math

import pytest
import torch
import ttnn

from eval.feature_matrix import apply_input_taggers, unsupported_reason
from eval.golden_tests.g6_interleaved_to_sharded.feature_spec import FLUX_57, FLUX_58
from ttnn.operations.g6_interleaved_to_sharded import g6_interleaved_to_sharded
from ttnn.operations.g6_interleaved_to_sharded import EXCLUSIONS, INPUT_TAGGERS, SUPPORTED, describe_plan


def pytorch_g6_interleaved_to_sharded(x):
    """Reference: the identity on the logical values; the placement never enters them."""
    return x.clone()


_SCHEME = {
    "height_sharded": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    "width_sharded": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
}
_ORIENTATION = {"row_major": ttnn.ShardOrientation.ROW_MAJOR, "col_major": ttnn.ShardOrientation.COL_MAJOR}
_BUFFER = {"l1": ttnn.BufferType.L1, "dram": ttnn.BufferType.DRAM}


def _variant(base, shape=None, **out_memory):
    scenario = copy.deepcopy(base)
    if shape is not None:
        scenario["shape"] = list(shape)
    scenario["out_memory"].update(out_memory)
    return scenario


FLUX_58_COL_MAJOR = _variant(FLUX_58, orientation="col_major")
# H = 33: tile row 0 is whole, tile row 1 holds one valid row. 27 width shards
# of 6 tile columns on 32 cores: the last shard holds 4 columns, cores 27..31 none.
RAGGED_WIDTH = _variant(FLUX_58, shape=(33, 5120), shard_shape=[64, 192])
# Two images of two tile rows each, the last row of each image partial; one tile
# row per height shard on 4 cores.
HEIGHT_MULTI_SHARD = _variant(
    FLUX_58, shape=(2, 1, 33, 128), scheme="height_sharded", grid=[[0, 0], [3, 0]], shard_shape=[32, 128]
)


def _grid(memory):
    (x0, y0), (x1, y1) = memory["grid"]
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))})


def _output_config(memory, device):
    _, (x1, y1) = memory["grid"]
    grid_size = device.compute_with_storage_grid_size()
    if x1 >= grid_size.x or y1 >= grid_size.y:
        pytest.skip(f"shard grid end ({x1},{y1}) exceeds the compute grid ({grid_size.x},{grid_size.y})")
    spec = ttnn.ShardSpec(_grid(memory), tuple(memory["shard_shape"]), _ORIENTATION[memory["orientation"]])
    return ttnn.MemoryConfig(_SCHEME[memory["scheme"]], _BUFFER[memory["buffer"]], spec)


def _gated(*scenarios, name):
    reasons = [
        unsupported_reason(apply_input_taggers(INPUT_TAGGERS, (scenario,)), SUPPORTED, EXCLUSIONS)
        for scenario in scenarios
    ]
    reasons = [reason for reason in reasons if reason is not None]
    marks = [pytest.mark.xfail(reason="; ".join(reasons), strict=True, raises=NotImplementedError)] if reasons else []
    return pytest.param(*scenarios, marks=marks, id=name)


# bfloat16 bit patterns: NaN payloads 0x7F81 and 0xFFA5, +inf, -inf, -0.0, the smallest denormal.
_SPECIAL_BITS = (0x7F81, 0xFFA5 - 0x10000, 0x7F80, 0xFF80 - 0x10000, 0x8000 - 0x10000, 0x0001)


def _torch_input(shape, values="randn"):
    x = torch.randn(tuple(shape), dtype=torch.float32).to(torch.bfloat16)
    if values == "special":
        bits = x.view(torch.int16).flatten().clone()
        picks = torch.arange(0, bits.numel(), 7)
        pattern = torch.tensor(_SPECIAL_BITS, dtype=torch.int16)
        bits[picks] = pattern[torch.arange(picks.numel()) % len(_SPECIAL_BITS)]
        x = bits.reshape(tuple(shape)).view(torch.bfloat16)
    return x


def _device_input(torch_input, device, *, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=memory_config)


def _bit_mismatch(actual, expected):
    got = actual.contiguous().view(torch.int16).flatten()
    want = expected.contiguous().view(torch.int16).flatten()
    wrong = got != want
    if not bool(wrong.any()):
        return None
    first = int(wrong.nonzero()[0])
    return (
        f"{int(wrong.sum())} of {wrong.numel()} values differ in their bit pattern; first at flat index "
        f"{first}: got 0x{int(got[first]) & 0xFFFF:04X}, expected 0x{int(want[first]) & 0xFFFF:04X}"
    )


def _check(output, expected, shape, out_config):
    """Shape, dtype, layout, the exact requested placement, then bit equality."""
    assert list(output.shape) == list(shape)
    assert output.dtype == ttnn.bfloat16
    assert output.layout == ttnn.TILE_LAYOUT
    got = output.memory_config()
    assert got.buffer_type == out_config.buffer_type
    assert got.memory_layout == out_config.memory_layout
    assert got.shard_spec is not None, "result has no shard spec"
    assert got.shard_spec.grid == out_config.shard_spec.grid
    assert list(got.shard_spec.shape) == list(out_config.shard_spec.shape)
    assert got.shard_spec.orientation == out_config.shard_spec.orientation
    actual = ttnn.to_torch(output)
    assert list(actual.shape) == list(shape)
    mismatch = _bit_mismatch(actual, expected)
    assert mismatch is None, mismatch


def _move_and_check(scenario, device, *, values="randn", poison=None):
    out_config = _output_config(scenario["out_memory"], device)
    ttnn_input = _device_input(_torch_input(scenario["shape"], values), device)
    if poison is not None:
        ttnn_input = ttnn.fill_implicit_tile_padding(ttnn_input, poison, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    # Graded against the values the device holds: a host conversion may rewrite a NaN payload.
    expected = pytorch_g6_interleaved_to_sharded(ttnn.to_torch(ttnn_input))
    output = g6_interleaved_to_sharded(ttnn_input, out_config)
    _check(output, expected, scenario["shape"], out_config)


@pytest.mark.parametrize(
    "scenario",
    [
        _gated(FLUX_58, name="flux58_1x5120_width_32cores"),
        _gated(FLUX_57, name="flux57_1x1x1x128_height_1core"),
        _gated(FLUX_58_COL_MAJOR, name="flux58_col_major"),
        _gated(RAGGED_WIDTH, name="33x5120_width_ragged_27of32"),
        _gated(HEIGHT_MULTI_SHARD, name="2x1x33x128_height_4cores"),
    ],
)
def test_g6_interleaved_to_sharded(scenario, device):
    torch.manual_seed(42)
    _move_and_check(scenario, device)


@pytest.mark.parametrize("scenario", [_gated(FLUX_58, name="flux58")])
def test_special_values_bit_exact(scenario, device):
    """NaN payloads, infinities, -0.0 and a denormal come back bit for bit."""
    torch.manual_seed(42)
    _move_and_check(scenario, device, values="special")


@pytest.mark.parametrize(
    "scenario", [_gated(FLUX_58, name="flux58"), _gated(RAGGED_WIDTH, name="33x5120_width_ragged")]
)
def test_result_ignores_tile_padding(scenario, device):
    """NaN in the input's tile padding must not reach the logical values."""
    torch.manual_seed(42)
    _move_and_check(scenario, device, poison=float("nan"))


@pytest.mark.parametrize(
    "base,col_major,ragged",
    [_gated(FLUX_58, FLUX_58_COL_MAJOR, RAGGED_WIDTH, name="flux58-colmajor-ragged-flux58")],
)
def test_repeat_and_cache_key(base, col_major, ragged, device):
    """Repeat calls agree bit for bit; the shard spec, orientation and input
    address all reach a program-cache hit."""
    torch.manual_seed(42)
    row_major_config = _output_config(base["out_memory"], device)
    col_major_config = _output_config(col_major["out_memory"], device)
    ragged_config = _output_config(ragged["out_memory"], device)

    first = _device_input(_torch_input(base["shape"]), device)
    first_values = ttnn.to_torch(first)
    _check(g6_interleaved_to_sharded(first, row_major_config), first_values, base["shape"], row_major_config)
    _check(g6_interleaved_to_sharded(first, row_major_config), first_values, base["shape"], row_major_config)
    _check(g6_interleaved_to_sharded(first, col_major_config), first_values, base["shape"], col_major_config)

    # Allocated while `first` is alive, so it sits at another DRAM address.
    second = _device_input(_torch_input(base["shape"]), device)
    _check(g6_interleaved_to_sharded(second, row_major_config), ttnn.to_torch(second), base["shape"], row_major_config)

    third = _device_input(_torch_input(ragged["shape"]), device)
    _check(g6_interleaved_to_sharded(third, ragged_config), ttnn.to_torch(third), ragged["shape"], ragged_config)

    _check(g6_interleaved_to_sharded(first, row_major_config), first_values, base["shape"], row_major_config)


def _expected_plan(scenario):
    """(num_shards, [(core, role, shard, block_rows, block_cols)] in shard-grid order)."""
    shape = scenario["shape"]
    out = scenario["out_memory"]
    tensor_rows_tiles = math.prod(shape[:-2]) * math.ceil(shape[-2] / 32)
    tensor_cols_tiles = math.ceil(shape[-1] / 32)
    shard_rows_tiles, shard_cols_tiles = out["shard_shape"][0] // 32, out["shard_shape"][1] // 32
    height = out["scheme"] == "height_sharded"
    if height:
        num_shards = math.ceil(tensor_rows_tiles / shard_rows_tiles)
    else:
        num_shards = math.ceil(tensor_cols_tiles / shard_cols_tiles)
    cores = ttnn.corerange_to_cores(_grid(out), row_wise=(out["orientation"] == "row_major"))
    entries = []
    for s, core in enumerate(cores):
        if s >= num_shards:
            entries.append(((core.x, core.y), "idle", None, 0, 0))
        elif height:
            rows = min(shard_rows_tiles, tensor_rows_tiles - s * shard_rows_tiles)
            entries.append(((core.x, core.y), "owner", s, rows, tensor_cols_tiles))
        else:
            cols = min(shard_cols_tiles, tensor_cols_tiles - s * shard_cols_tiles)
            entries.append(((core.x, core.y), "owner", s, tensor_rows_tiles, cols))
    return num_shards, entries


@pytest.mark.parametrize(
    "scenario",
    [
        _gated(FLUX_58, name="flux58"),
        _gated(FLUX_58_COL_MAJOR, name="flux58_col_major"),
        _gated(RAGGED_WIDTH, name="33x5120_width_ragged"),
    ],
)
def test_plan_reported(scenario, device):
    """The plan a call selects is reportable (BB-08) and is the owner-fetch
    map with trimmed last-row reads; the core order follows the orientation."""
    torch.manual_seed(42)
    out_config = _output_config(scenario["out_memory"], device)
    ttnn_input = _device_input(_torch_input(scenario["shape"]), device)
    plan = describe_plan(ttnn_input, out_config)
    assert plan["regime"] == "owner_fetch"
    # H mod 32 = 1: one 32 B face row per top face, rounded up to the DRAM read alignment.
    assert plan["last_row_form"] == "trimmed"
    dram_alignment = ttnn.get_dram_alignment()
    assert plan["trim_span_bytes"] == math.ceil(32 / dram_alignment) * dram_alignment
    num_shards, expected = _expected_plan(scenario)
    assert plan["num_shards"] == num_shards
    got = [(tuple(c["core"]), c["role"], c["shard"], c["block_rows"], c["block_cols"]) for c in plan["cores"]]
    assert got == expected


_MALFORMED = [
    "too_few_cores",
    "shard_not_tile_multiple",
    "width_shard_shorter_than_tensor",
    "height_shard_narrower_than_tensor",
    "output_interleaved",
    "output_in_dram",
    "input_in_l1",
    "input_row_major",
    "input_float32",
    "input_rank_3",
    "input_on_host",
]


def _malformed_request(name, device):
    def tile_input(shape, **kwargs):
        return _device_input(_torch_input(shape), device, **kwargs)

    def sharded(scenario):
        return _output_config(scenario["out_memory"], device)

    shape = FLUX_58["shape"]
    if name == "too_few_cores":  # 40 width shards of 4 tile columns, 32 cores
        return tile_input(shape), sharded(_variant(FLUX_58, shard_shape=[32, 128]))
    if name == "shard_not_tile_multiple":  # 176 = 5.5 tiles
        return tile_input(shape), sharded(_variant(FLUX_58, shard_shape=[32, 176]))
    if name == "width_shard_shorter_than_tensor":  # padded height 64, shard height 32
        return tile_input((33, 5120)), sharded(_variant(FLUX_58, shard_shape=[32, 192]))
    if name == "height_shard_narrower_than_tensor":  # malformed, so refused before the support check
        return tile_input(FLUX_57["shape"]), sharded(_variant(FLUX_57, grid=[[0, 0], [1, 0]], shard_shape=[32, 64]))
    if name == "output_interleaved":
        return tile_input(shape), ttnn.L1_MEMORY_CONFIG
    if name == "output_in_dram":
        return tile_input(shape), sharded(_variant(FLUX_58, buffer="dram"))
    if name == "input_in_l1":
        return tile_input(shape, memory_config=ttnn.L1_MEMORY_CONFIG), sharded(FLUX_58)
    if name == "input_row_major":
        return tile_input(shape, layout=ttnn.ROW_MAJOR_LAYOUT), sharded(FLUX_58)
    if name == "input_float32":
        x = ttnn.from_torch(
            torch.randn(shape, dtype=torch.float32),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return x, sharded(FLUX_58)
    if name == "input_rank_3":
        return tile_input((1, 1, 5120)), sharded(FLUX_58)
    if name == "input_on_host":
        return ttnn.from_torch(_torch_input(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), sharded(FLUX_58)
    raise ValueError(f"unknown malformed request {name!r}")


@pytest.mark.parametrize("name", _MALFORMED)
def test_malformed_request_refused(name, device, expect_error):
    """Malformed requests raise ValueError or RuntimeError, never a support refusal."""
    torch.manual_seed(42)
    input_tensor, memory_config = _malformed_request(name, device)
    # The refusal wording is the op's own; only the exception class is the contract.
    with expect_error((ValueError, RuntimeError), "") as refused:
        g6_interleaved_to_sharded(input_tensor, memory_config)
    # SupportRefusal derives from NotImplementedError, itself a RuntimeError.
    assert not isinstance(
        refused.value, NotImplementedError
    ), f"malformed request {name!r} was answered with a support refusal: {refused.value!r}"
