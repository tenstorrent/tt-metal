# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_pad (Phase 0: bfloat16, TILE, DRAM interleaved, rank 3/4).

Immutable spec for the implementer. Only production cases of the contract (BB-22).
Gate: bit equality of every logical output position (helpers.TOLERANCES is ("exact", None)).
Regimes are pinned by observable storage:
  extension_band    -> output shares the input's buffer and padded shape, input unchanged
  new_storage_tiles -> output lives in a new buffer
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_pad import g6_pad

TILE = 32


def to_torch_pad(padding):
    flat = []
    for low, high in reversed([tuple(p) for p in padding]):
        flat += [int(low), int(high)]
    return flat


def torch_g6_pad(x, padding, value):
    """Reference: torch.nn.functional.pad in the input dtype (no float detour)."""
    return torch.nn.functional.pad(x, to_torch_pad(padding), mode="constant", value=value)


def highs(shape, padding):
    return [0] * (len(shape) - len(padding)) + [p[1] for p in padding]


def is_extension(shape, padding):
    h = highs(shape, padding)
    only_h = h[-2] > 0 and all(v == 0 for i, v in enumerate(h) if i != len(h) - 2)
    out_h = shape[-2] + h[-2]
    return only_h and -(-shape[-2] // TILE) == -(-out_h // TILE)


def assert_bits_equal(actual, expected, what="output"):
    assert list(actual.shape) == list(expected.shape), f"{what}: shape {list(actual.shape)} != {list(expected.shape)}"
    a = actual.contiguous().view(torch.int16)
    e = expected.contiguous().view(torch.int16)
    diff = a != e
    if bool(diff.any()):
        idx = tuple(int(i) for i in diff.nonzero()[0])
        pytest.fail(
            f"{what}: {int(diff.sum())} of {diff.numel()} positions differ, first {idx}: "
            f"got {actual[idx]} expected {expected[idx]}"
        )


def make_input(shape):
    return (torch.randn(shape, dtype=torch.float32) * 100).to(torch.bfloat16)


def to_device(x, device):
    return ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def call_one_program(x_tt, padding, value, positional):
    """The call must dispatch exactly one device program (BB-01; the view launches none)."""
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        out = g6_pad(x_tt, padding, value) if positional else g6_pad(x_tt, padding, value=value)
    finally:
        graph = ttnn.graph.end_graph_capture()
    programs = [
        n["params"].get("name", "?")
        for n in graph
        if n.get("node_type") == "function_start" and "program_factory_type" in n.get("params", {})
    ]
    assert len(programs) == 1, f"expected one device program, got {programs}"
    return out


def run_case(device, shape, padding, value, *, positional=False, poison=None, untouched_rows=False):
    x = make_input(shape)
    expected = torch_g6_pad(x, padding, value)

    x_tt = to_device(x, device)
    if poison is not None:
        x_tt = ttnn.fill_implicit_tile_padding(x_tt, poison)
    raw_before = None
    if untouched_rows:
        raw = x_tt.cpu().to_torch_with_padded_shape()
        raw_before = raw.reshape(-1, raw.shape[-2], raw.shape[-1]).clone()
    in_addr = x_tt.buffer_address()
    in_padded = list(x_tt.padded_shape)

    out_tt = call_one_program(x_tt, padding, value, positional)

    assert out_tt.dtype == ttnn.bfloat16
    assert out_tt.layout == ttnn.TILE_LAYOUT
    assert out_tt.memory_config().buffer_type == ttnn.BufferType.DRAM
    assert_bits_equal(ttnn.to_torch(out_tt), expected)

    if is_extension(shape, padding):
        # Regime extension_band: metadata-only view of the input's storage.
        assert out_tt.buffer_address() == in_addr, "extension call must reuse the input's storage"
        assert list(out_tt.padded_shape) == in_padded, "extension view must keep the input's padded shape"
        assert list(x_tt.shape) == list(shape), "input logical shape changed"
        assert_bits_equal(ttnn.to_torch(x_tt), x, what="input after the call")
        if untouched_rows:
            raw = x_tt.cpu().to_torch_with_padded_shape()
            raw_after = raw.reshape(-1, raw.shape[-2], raw.shape[-1])
            keep = raw_after.shape[-2] - TILE
            assert_bits_equal(raw_after[:, :keep, :], raw_before[:, :keep, :], what="input tile rows before the last")
    else:
        # Regime new_storage_tiles: a new tensor; the input stays.
        assert out_tt.buffer_address() != in_addr, "new_storage call must write a new tensor"
        assert_bits_equal(ttnn.to_torch(x_tt), x, what="input after the call")
    return out_tt


NEG_2_127 = -1.7014118346046923e38

# (shape, padding, value): Phase 0 production cases (bfloat16 TILE DRAM).
EXTENSION_CASES = [
    ((1, 1, 2, 62080), ((0, 0), (0, 0), (0, 30), (0, 0)), 0.0),  # p2, one tile row
    ((1, 1, 8, 62080), ((0, 0), (0, 0), (0, 8), (0, 0)), 0),  # p3
    ((1, 1, 8, 62080), ((0, 0), (0, 0), (0, 24), (0, 0)), 0.0),  # p4
    ((1, 1, 1, 62080), ((0, 0), (0, 0), (0, 7), (0, 0)), 0),
    ((1, 3844, 1536), ((0, 0), (0, 28), (0, 0)), 0.0),  # 121 tile rows, 48 columns
    ((1, 3844, 12), ((0, 0), (0, 28), (0, 0)), 0.0),  # one band tile
    ((16, 1, 1536), ((0, 0), (0, 31), (0, 0)), 0.0),  # 16 slices
    ((8, 1, 12), ((0, 0), (0, 31), (0, 0)), 0.0),
    ((1, 33, 2560), ((0, 0), (0, 31), (0, 0)), 0.0),  # band row 1..31 of tile row 1
    ((1, 77, 1536), ((0, 0), (0, 19), (0, 0)), 0.0),
]

NEW_STORAGE_CASES = [
    ((1, 196, 128), ((0, 60), (0, 0)), 0),  # new tile rows
    ((1, 185, 128), ((0, 71), (0, 0)), 0),
    ((1, 1, 196, 196), ((0, 60), (0, 60)), NEG_2_127),  # hw, c = -2^127
    ((1, 1, 207, 207), ((0, 49), (0, 49)), NEG_2_127),
]


def _id(case):
    shape, padding, value = case
    return f"{'x'.join(map(str, shape))}-pad{'.'.join(str(p[1]) for p in padding)}-v{value}"


@pytest.mark.parametrize("case", EXTENSION_CASES, ids=[_id(c) for c in EXTENSION_CASES])
def test_g6_pad_extension(device, case):
    torch.manual_seed(42)
    shape, padding, value = case
    run_case(device, shape, padding, value)


@pytest.mark.parametrize("case", NEW_STORAGE_CASES, ids=[_id(c) for c in NEW_STORAGE_CASES])
def test_g6_pad_new_storage(device, case):
    torch.manual_seed(42)
    shape, padding, value = case
    run_case(device, shape, padding, value)


@pytest.mark.parametrize(
    "case",
    [EXTENSION_CASES[1], EXTENSION_CASES[4], NEW_STORAGE_CASES[1], NEW_STORAGE_CASES[2]],
    ids=lambda c: _id(c),
)
def test_g6_pad_poisoned_input_padding(device, case):
    """The input's tile padding holds a loud value: no logical output position may take it."""
    torch.manual_seed(42)
    shape, padding, value = case
    run_case(device, shape, padding, value, poison=1000.0, positional=True)


@pytest.mark.parametrize("case", [EXTENSION_CASES[4], EXTENSION_CASES[7]], ids=lambda c: _id(c))
def test_g6_pad_extension_touches_only_last_tile_row(device, case):
    torch.manual_seed(42)
    shape, padding, value = case
    run_case(device, shape, padding, value, poison=1000.0, untouched_rows=True)


def test_g6_pad_same_shape_new_value(device):
    """Program cache on: one shape with two fill values, then another padding of the same shape."""
    torch.manual_seed(42)
    keep = []
    keep.append(run_case(device, (1, 1, 8, 62080), ((0, 0), (0, 0), (0, 8), (0, 0)), 0))
    keep.append(run_case(device, (1, 1, 8, 62080), ((0, 0), (0, 0), (0, 8), (0, 0)), 7.0))
    keep.append(run_case(device, (1, 1, 8, 62080), ((0, 0), (0, 0), (0, 24), (0, 0)), 0.0, poison=1000.0))
    keep.append(run_case(device, (1, 1, 196, 196), ((0, 60), (0, 60)), 0.0))
    keep.append(run_case(device, (1, 1, 196, 196), ((0, 60), (0, 60)), NEG_2_127))


@pytest.mark.parametrize(
    "padding",
    [
        ((0, 0), (1, 28), (0, 0)),  # low != 0
        ((0, 0), (0, 28), (0, -4)),  # high < 0
        ((0, 0), (0, 0), (0, 0), (0, 8), (0, 0)),  # more pairs than dims
        (),  # no pair
    ],
)
def test_g6_pad_refuses_bad_padding(device, padding, expect_error):
    torch.manual_seed(42)
    x_tt = to_device(make_input((1, 100, 1536)), device)
    with expect_error((ValueError, RuntimeError), ""):
        g6_pad(x_tt, padding, value=0.0)


def test_g6_pad_refuses_bad_value(device, expect_error):
    torch.manual_seed(42)
    x_tt = to_device(make_input((1, 100, 1536)), device)
    with expect_error((ValueError, RuntimeError), ""):
        g6_pad(x_tt, ((0, 0), (0, 28), (0, 0)), value="0")
