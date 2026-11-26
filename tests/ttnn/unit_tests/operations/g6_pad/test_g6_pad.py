# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_pad (Phase 0: bfloat16, TILE, DRAM interleaved).

Immutable spec for the implementer. Production shapes only (BB-22). The gate is bit equality of
every logical output position (helpers.TOLERANCES is ("exact", None) for every dtype); the
output's own tile padding is not compared.

Covers both built regimes of op_design.md:
  E  tile_padding_extension: in place, write-only new rows of the last tile row of each slice,
     output is a metadata-only view of the input's buffer
  N  new_storage: one pass, boundary tiles assembled in L1, fill-only tiles never read
Every call must dispatch exactly one device program (BB-01).
"""

import pytest
import torch
import ttnn

from eval.sharding import auto_shard_config
from ttnn.operations.g6_pad import g6_pad

TILE = 32


def torch_reference(x, padding, value):
    """torch.nn.functional.pad with the contract's pair order (leading -> trailing, aligned to the
    trailing dims) mapped to torch's flat list (last dim first). Computed in x's dtype."""
    flat = []
    for low, high in reversed([tuple(p) for p in padding]):
        flat += [int(low), int(high)]
    return torch.nn.functional.pad(x, flat, mode="constant", value=value)


def highs_of(shape, padding):
    return [0] * (len(shape) - len(padding)) + [p[1] for p in padding]


def is_extension(shape, padding):
    highs = highs_of(shape, padding)
    out_h = shape[-2] + highs[-2]
    only_h = highs[-2] > 0 and all(h == 0 for i, h in enumerate(highs) if i != len(shape) - 2)
    return only_h and -(-shape[-2] // TILE) == -(-out_h // TILE)


def assert_bit_equal(actual, expected, what="output"):
    assert list(actual.shape) == list(expected.shape), f"{what}: shape {list(actual.shape)} != {list(expected.shape)}"
    assert actual.dtype == expected.dtype, f"{what}: dtype {actual.dtype} != {expected.dtype}"
    a = actual.contiguous().view(torch.int16)
    e = expected.contiguous().view(torch.int16)
    mismatches = int((a != e).sum())
    assert mismatches == 0, f"{what}: {mismatches} of {a.numel()} positions differ bitwise"


def call_one_program(fn):
    """Run fn under graph capture and require exactly one device program (the op's own)."""
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        out = fn()
    finally:
        graph = ttnn.graph.end_graph_capture()
    programs = [
        node["params"].get("name", "?")
        for node in graph
        if node.get("node_type") == "function_start" and "program_factory_type" in node.get("params", {})
    ]
    assert len(programs) == 1, f"the call dispatched {len(programs)} device programs {programs}, expected 1"
    return out


# (shape, padding, value): Phase 0 production cases (feature_spec.py SCENARIOS ids in comments).
CASES = [
    # E, single tile (one tile row, one tile column)                     -- case 269
    ((1, 31, 12), ((0, 0), (0, 1), (0, 0)), 0.0),
    # E, multi-batch: 16 slices of one tile row                          -- case 254
    ((16, 1, 1536), ((0, 0), (0, 31), (0, 0)), 0.0),
    # E, multi-tile-row slice (storage takeover required)                -- case 292
    ((1, 100, 1536), ((0, 0), (0, 28), (0, 0)), 0.0),
    # E, tall: 121 tile rows, only the last one may be touched           -- case 247
    ((1, 3844, 1536), ((0, 0), (0, 28), (0, 0)), 0.0),
    # E, wide short (p2, p3, p4): new rows end at 32 / at 16 / at 32     -- cases 258, 253, 291
    ((1, 1, 2, 62080), ((0, 0), (0, 0), (0, 30), (0, 0)), 0.0),
    ((1, 1, 8, 62080), ((0, 0), (0, 0), (0, 8), (0, 0)), 0),
    ((1, 1, 8, 62080), ((0, 0), (0, 0), (0, 24), (0, 0)), 0.0),
    # E, new rows are exactly faces 2 and 3                              -- case 257
    ((1, 1, 16, 62080), ((0, 0), (0, 0), (0, 16), (0, 0)), 0.0),
    # N, dim -2 padded across tile rows (fill-only tile rows)            -- case 68
    ((1, 185, 128), ((0, 71), (0, 0)), 0),
    # N, both dims padded, boundary rows/cols/corner, c = -2^127         -- cases 65, 72
    ((1, 1, 196, 196), ((0, 60), (0, 60)), -1.7014118346046923e38),
    ((1, 1, 207, 207), ((0, 49), (0, 49)), -1.7014118346046923e38),
]


def _ids(case):
    shape, padding, value = case
    regime = "E" if is_extension(shape, padding) else "N"
    return f"{regime}-{'x'.join(map(str, shape))}-pad{'.'.join(str(h) for h in highs_of(shape, padding))}-v{value}"


def _make_input(shape, device, poison=None, torch_input=None, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    if torch_input is None:
        torch.manual_seed(42)
        torch_input = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    if poison is not None:
        tt_input = ttnn.fill_implicit_tile_padding(tt_input, poison)
    return torch_input, tt_input


def _raw_rows(tt_tensor):
    """The tensor's raw padded buffer as (slices, padded H, padded W)."""
    raw = tt_tensor.cpu().to_torch_with_padded_shape()
    return raw.reshape(-1, raw.shape[-2], raw.shape[-1])


def _check_call(shape, padding, value, torch_input, tt_input, tt_output, raw_before):
    expected = torch_reference(torch_input, padding, value)
    assert tt_output.dtype == ttnn.bfloat16
    assert tt_output.layout == ttnn.TILE_LAYOUT
    assert tt_output.memory_config().buffer_type == ttnn.BufferType.DRAM
    assert_bit_equal(ttnn.to_torch(tt_output), expected)

    if is_extension(shape, padding):
        # Storage takeover: out is x's buffer at x's padded shape; x's logical values unchanged.
        assert tt_output is not tt_input
        assert tt_output.buffer_address() == tt_input.buffer_address()
        assert list(tt_output.padded_shape) == list(tt_input.padded_shape)
        assert list(tt_input.shape) == list(shape)
        assert_bit_equal(ttnn.to_torch(tt_input), torch_input, "input after the call")
        # No tile row other than the last one of each slice may change (raw padded buffer).
        after = _raw_rows(tt_input)
        keep_rows = after.shape[-2] - TILE
        assert_bit_equal(after[:, :keep_rows, :], raw_before[:, :keep_rows, :], "rows above the last tile row")
    else:
        assert tt_output.buffer_address() != tt_input.buffer_address()
        assert_bit_equal(ttnn.to_torch(tt_input), torch_input, "input after the call")


@pytest.mark.parametrize("case", CASES, ids=[_ids(c) for c in CASES])
@pytest.mark.parametrize("poison", [None, 1000.0], ids=["zero_padded", "poisoned_padding"])
def test_g6_pad(device, case, poison):
    shape, padding, value = case
    torch_input, tt_input = _make_input(shape, device, poison)
    raw_before = _raw_rows(tt_input).clone()

    tt_output = call_one_program(lambda: g6_pad(tt_input, padding, value=value))

    _check_call(shape, padding, value, torch_input, tt_input, tt_output, raw_before)


def test_g6_pad_value_positional(device):
    """`value` passed positionally (production call pattern)."""
    shape, padding, value = (1, 77, 1536), ((0, 0), (0, 19), (0, 0)), 0.0  # case 276
    torch_input, tt_input = _make_input(shape, device, poison=1000.0)
    raw_before = _raw_rows(tt_input).clone()
    tt_output = call_one_program(lambda: g6_pad(tt_input, padding, value))
    _check_call(shape, padding, value, torch_input, tt_input, tt_output, raw_before)


def test_g6_pad_keywords_and_default_value(device):
    """`padding` as keyword; `value` omitted (default 0.0)."""
    shape, padding = (1, 196, 128), ((0, 60), (0, 0))  # case 64, N
    torch_input, tt_input = _make_input(shape, device, poison=1000.0)
    raw_before = _raw_rows(tt_input).clone()
    tt_output = call_one_program(lambda: g6_pad(tt_input, padding=padding))
    _check_call(shape, padding, 0.0, torch_input, tt_input, tt_output, raw_before)


def test_g6_pad_same_shape_values_and_paddings(device):
    """Program cache: same shape, two fill values, then another padding (BB-07). Earlier tensors
    stay alive."""
    shape = (1, 1, 8, 62080)
    keep = []
    for padding, value in (
        (((0, 0), (0, 0), (0, 8), (0, 0)), 0),
        (((0, 0), (0, 0), (0, 8), (0, 0)), 7.0),
        (((0, 0), (0, 0), (0, 24), (0, 0)), 0.1),
    ):
        torch_input, tt_input = _make_input(shape, device, poison=1000.0)
        raw_before = _raw_rows(tt_input).clone()
        tt_output = call_one_program(lambda: g6_pad(tt_input, padding, value=value))
        _check_call(shape, padding, value, torch_input, tt_input, tt_output, raw_before)
        keep.extend([tt_input, tt_output])


@pytest.mark.parametrize(
    "case",
    [CASES[5], CASES[9]],  # E case 253, N case 65
    ids=lambda c: _ids(c),
)
def test_g6_pad_special_values(device, case):
    """Signed zero, +-inf and subnormals in the logical region are moved bit for bit."""
    shape, padding, value = case
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    tiny = torch.finfo(torch.bfloat16).tiny
    specials = torch.tensor([-0.0, float("inf"), float("-inf"), tiny, tiny / 4, -tiny / 8]).to(torch.bfloat16)
    flat = torch_input.reshape(-1)
    flat[: specials.numel()] = specials
    torch_input = flat.reshape(shape)
    torch_input, tt_input = _make_input(shape, device, poison=1000.0, torch_input=torch_input)
    raw_before = _raw_rows(tt_input).clone()
    tt_output = call_one_program(lambda: g6_pad(tt_input, padding, value=value))
    _check_call(shape, padding, value, torch_input, tt_input, tt_output, raw_before)


@pytest.mark.parametrize(
    "padding, value",
    [
        (((0, 0), (1, 28), (0, 0)), 0.0),  # low != 0
        (((0, 0), (0, 28), (0, -4)), 0.0),  # high < 0
        (((0, 0), (0, 0), (0, 0), (0, 28)), 0.0),  # more pairs than dims
        ((), 0.0),  # no pair
        (((0, 0), (0, 28, 0), (0, 0)), 0.0),  # pair of three
        (((0, 0), (0, 28.0), (0, 0)), 0.0),  # high not an int
        (((0, 0), (0, 28), (0, 0)), "0"),  # value not a number
    ],
    ids=[
        "low_nonzero",
        "high_negative",
        "too_many_pairs",
        "no_pairs",
        "pair_of_three",
        "high_not_int",
        "value_not_number",
    ],
)
def test_g6_pad_refusals(device, expect_error, padding, value):
    _, tt_input = _make_input((1, 100, 1536), device)
    with expect_error((ValueError, RuntimeError), ""):
        g6_pad(tt_input, padding, value=value)


def test_g6_pad_refuses_sharded_input(device, expect_error):
    shape = (1, 196, 128)  # case 64
    memory_config = auto_shard_config(
        list(shape),
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )
    _, tt_input = _make_input(shape, device, memory_config=memory_config)
    with expect_error((ValueError, RuntimeError), ""):
        g6_pad(tt_input, ((0, 60), (0, 0)), value=0.0)
