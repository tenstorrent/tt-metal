# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_to_layout, Refinement 1: bfloat8_b TILE input, interleaved or sharded.

Shapes and placements are the production bfloat8_b cases (eval/golden_tests/g6_to_layout/
feature_spec.py: 20, 29, 37, 43, 48). The output is bfloat16 ROW_MAJOR and must equal the host's
bfloat16 rendering of the packed bfloat8_b input bit for bit.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_to_layout import g6_to_layout, select_plan

L1 = ttnn.BufferType.L1
RM = ttnn.ROW_MAJOR_LAYOUT
ROW_MAJOR_ORIENTATION = ttnn.ShardOrientation.ROW_MAJOR

# -0.0, subnormals, +inf, -inf, NaN with payload
SPECIAL_BITS = (0x8000, 0x0001, 0x8001, 0x007F, 0x7F80, 0xFF80, 0x7FC1)

# Cases 37/48: 11x7 + 9x1 cores, 96-row shards; the last shard holds 32 live rows of 96.
_HEIGHT_GRID = [((0, 0), (10, 6)), ((0, 7), (8, 7))]


def _grid(ranges):
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for (x0, y0), (x1, y1) in ranges]
    )


def _sharded(memory_layout, ranges, shard_shape):
    return ttnn.MemoryConfig(memory_layout, L1, ttnn.ShardSpec(_grid(ranges), list(shard_shape), ROW_MAJOR_ORIENTATION))


# (case id, shape, input memory config, output memory_config keyword or None)
CASES = [
    (29, (1, 32, 64, 256), ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    (43, (1, 128, 256, 32), ttnn.L1_MEMORY_CONFIG, None),
    (
        20,
        (1, 1, 1, 256),
        _sharded(ttnn.TensorMemoryLayout.BLOCK_SHARDED, [((0, 0), (7, 0))], (32, 32)),
        ttnn.L1_MEMORY_CONFIG,
    ),
    (
        37,
        (1, 64, 128, 256),
        _sharded(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, _HEIGHT_GRID, (96, 256)),
        ttnn.L1_MEMORY_CONFIG,
    ),
    (
        48,
        (1, 64, 128, 128),
        _sharded(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, _HEIGHT_GRID, (96, 128)),
        ttnn.L1_MEMORY_CONFIG,
    ),
]


def _case_id(case):
    case_id, shape, memory_config, _ = case
    return f"c{case_id}-{'x'.join(str(d) for d in shape)}-{str(memory_config.memory_layout).split('.')[-1].lower()}"


def _bits(t):
    return t.contiguous().view(torch.int16)


def _randn(shape):
    torch.manual_seed(42)
    return torch.randn(shape, dtype=torch.bfloat16)


def _with_specials(shape):
    x = _randn(shape)
    flat = x.view(-1).view(torch.int16)
    table = torch.tensor([b - 0x10000 if b >= 0x8000 else b for b in SPECIAL_BITS], dtype=torch.int16)
    count = max(len(SPECIAL_BITS), flat.numel() // 100)
    positions = torch.randint(0, flat.numel(), (count,))
    flat[positions] = table[torch.randint(0, len(SPECIAL_BITS), (count,))]
    return x


def _host_bf8_rendering(torch_input):
    packed = ttnn.from_torch(torch_input, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    return ttnn.to_torch(packed).to(torch.bfloat16).reshape(torch_input.shape)


def _run_and_check(device, torch_input, memory_config, out_memory_config):
    ttnn_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    kwargs = {} if out_memory_config is None else {"memory_config": out_memory_config}
    output = g6_to_layout(ttnn_input, RM, **kwargs)
    assert output.dtype == ttnn.bfloat16
    assert output.layout == RM
    assert output.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    actual = ttnn.to_torch(output).reshape(torch_input.shape)
    expected = _host_bf8_rendering(torch_input)
    differ = _bits(actual) != _bits(expected)
    count = int(differ.sum())
    assert count == 0, (
        f"{count} of {differ.numel()} values differ bitwise, first at {tuple(differ.nonzero()[0].tolist())}: "
        f"got 0x{int(_bits(actual)[differ][0]) & 0xFFFF:04X} expected 0x{int(_bits(expected)[differ][0]) & 0xFFFF:04X}"
    )


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_bf8b_to_row_major(device, case):
    _, shape, memory_config, out_memory_config = case
    _run_and_check(device, _randn(shape), memory_config, out_memory_config)


# The packer renders a bfp8 NaN datum as +-inf; the writer must restore the host's NaN bits. One
# interleaved case with column-chunked blocks and one sharded case with a ragged last shard.
@pytest.mark.parametrize("case", [CASES[0], CASES[3]], ids=_case_id)
def test_bf8b_special_bit_patterns(device, case):
    _, shape, memory_config, out_memory_config = case
    _run_and_check(device, _with_specials(shape), memory_config, out_memory_config)


@pytest.mark.parametrize(
    "case,regime",
    [(CASES[0], "packer_untilize"), (CASES[2], "sharded_in_place"), (CASES[3], "sharded_in_place")],
    ids=lambda v: _case_id(v) if isinstance(v, tuple) else v,
)
def test_regime_selection(device, case, regime):
    _, shape, memory_config, out_memory_config = case
    ttnn_input = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, device, memory_config
    )
    kwargs = {} if out_memory_config is None else {"memory_config": out_memory_config}
    plan = select_plan(ttnn_input, RM, **kwargs)
    assert plan["regime"] == regime
    if regime == "sharded_in_place":
        # The shard is the block: every live shard is consumed in one pass, and a shard's CB
        # spans only its live tile-rows.
        assert plan["shard"] is not None
        live = [core for core in plan["cores"] if core["rows"] > 0]
        assert len(live) == plan["num_cores"]
        assert plan["block_rows_tiles"] == plan["shard"]["shard_shape_tiles"][0]


def test_rejects_bf16_sharded_input(device, expect_error):
    case_id, shape, memory_config, out_memory_config = CASES[3]
    ttnn_input = ttnn.from_torch(
        _randn(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    with expect_error(Exception, "g6_to_layout"):
        g6_to_layout(ttnn_input, RM, memory_config=out_memory_config)
