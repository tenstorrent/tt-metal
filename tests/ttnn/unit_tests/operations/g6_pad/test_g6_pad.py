# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_pad, Phase 0: bfloat16, TILE_LAYOUT, DRAM interleaved, rank 3 and 4.

Every shape is a production case of the contract. The gate is bit equality of the logical
values, since the op moves values and computes nothing. The input's tile padding is poisoned
before every call, because a logical output position that lies there must hold the fill value,
not the producer's leftovers.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_pad import describe_plan, g6_pad

TILE = 32
POISON = 1000.0
NEG_2_POW_127 = -1.7014118346046923e38


def torch_reference(x, padding, value):
    flat = []
    for low, high in reversed([tuple(pair) for pair in padding]):
        flat += [int(low), int(high)]
    return torch.nn.functional.pad(x, flat, mode="constant", value=value)


def highs_of(shape, padding):
    return [0] * (len(shape) - len(padding)) + [pair[1] for pair in padding]


def tile_rows(h):
    return -(-h // TILE)


def is_extension(shape, padding):
    highs = highs_of(shape, padding)
    only_h = highs[-2] > 0 and all(h == 0 for i, h in enumerate(highs) if i != len(highs) - 2)
    return only_h and tile_rows(shape[-2]) == tile_rows(shape[-2] + highs[-2])


def assert_bit_equal(actual, expected, what="output"):
    assert list(actual.shape) == list(expected.shape), f"{what}: shape {list(actual.shape)} != {list(expected.shape)}"
    assert actual.dtype == expected.dtype, f"{what}: dtype {actual.dtype} != {expected.dtype}"
    diff = actual.contiguous().view(torch.int16) != expected.contiguous().view(torch.int16)
    if bool(diff.any()):
        index = tuple(int(i) for i in diff.nonzero()[0])
        raise AssertionError(
            f"{what}: {int(diff.sum())} of {diff.numel()} positions differ, first {index}: "
            f"got {actual[index]} expected {expected[index]}"
        )


def make_input(shape):
    torch.manual_seed(42)
    return (torch.randn(shape, dtype=torch.float32) * 100).to(torch.bfloat16)


def to_device(torch_input, device, poison=POISON):
    tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if poison is not None:
        tensor = ttnn.fill_implicit_tile_padding(tensor, poison)
    return tensor


def call_one_program(x, padding, value, value_positional=False):
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        out = g6_pad(x, padding, value) if value_positional else g6_pad(x, padding, value=value)
    finally:
        graph = ttnn.graph.end_graph_capture()
    programs = [
        node["params"].get("name", "?")
        for node in graph
        if node.get("node_type") == "function_start" and "program_factory_type" in node.get("params", {})
    ]
    assert len(programs) == 1, f"the call dispatched {len(programs)} device programs {programs}, expected exactly one"
    return out


def padded_rows(tensor):
    raw = tensor.cpu().to_torch_with_padded_shape()
    return raw.reshape(-1, raw.shape[-2], raw.shape[-1])


def run_case(device, shape, padding, value, *, value_positional=False, torch_input=None):
    """One graded call: values, placement, storage class, and the input's untouched state."""
    if torch_input is None:
        torch_input = make_input(shape)
    expected = torch_reference(torch_input, padding, value)
    extension = is_extension(shape, padding)

    x = to_device(torch_input, device)
    assert describe_plan(x, padding)["regime"] == ("extension_band" if extension else "new_storage_tiles")
    raw_before = padded_rows(x) if extension else None

    out = call_one_program(x, padding, value, value_positional)

    assert out.dtype == ttnn.bfloat16
    assert out.layout == ttnn.TILE_LAYOUT
    assert out.memory_config().buffer_type == ttnn.BufferType.DRAM
    assert_bit_equal(ttnn.to_torch(out), expected)

    if extension:
        assert out.buffer_address() == x.buffer_address(), "extension output must be a view of the input's storage"
        assert list(out.padded_shape) == list(x.padded_shape)
        assert list(x.shape) == list(shape)
        assert_bit_equal(ttnn.to_torch(x), torch_input, what="input after the call")
        keep = raw_before.shape[-2] - TILE
        assert_bit_equal(padded_rows(x)[:, :keep, :], raw_before[:, :keep, :], what="input tile rows before the last")
    else:
        assert out.buffer_address() != x.buffer_address(), "new_storage output must be a new tensor"
        assert_bit_equal(ttnn.to_torch(x), torch_input, what="input after the call")
    return out


CASES = [
    # extension_band, one tile row per slice
    pytest.param([1, 1, 2, 62080], [[0, 0], [0, 0], [0, 30], [0, 0]], 0.0, id="p2_1x1x2x62080_to32"),
    pytest.param([1, 1, 8, 62080], [[0, 0], [0, 0], [0, 8], [0, 0]], 0, id="p3_1x1x8x62080_to16"),
    pytest.param([1, 1, 8, 62080], [[0, 0], [0, 0], [0, 24], [0, 0]], 0.0, id="p4_1x1x8x62080_to32"),
    pytest.param([16, 1, 1536], [[0, 0], [0, 31], [0, 0]], 0.0, id="16x1x1536_to32"),
    pytest.param([8, 1, 12], [[0, 0], [0, 31], [0, 0]], 0.0, id="8x1x12_to32"),
    # extension_band, several tile rows per slice
    pytest.param([1, 3844, 1536], [[0, 0], [0, 28], [0, 0]], 0.0, id="1x3844x1536_to3872"),
    pytest.param([1, 3844, 12], [[0, 0], [0, 28], [0, 0]], 0.0, id="1x3844x12_to3872"),
    pytest.param([1, 33, 2560], [[0, 0], [0, 31], [0, 0]], 0.0, id="1x33x2560_to64"),
    pytest.param([1, 1000, 1536], [[0, 0], [0, 24], [0, 0]], 0.0, id="1x1000x1536_to1024"),
    # new_storage_tiles
    pytest.param([1, 196, 128], [[0, 60], [0, 0]], 0, id="1x196x128_to256"),
    pytest.param([1, 185, 128], [[0, 71], [0, 0]], 0, id="1x185x128_to256"),
    pytest.param([1, 1, 196, 196], [[0, 60], [0, 60]], NEG_2_POW_127, id="1x1x196x196_to256"),
    pytest.param([1, 1, 207, 207], [[0, 49], [0, 49]], NEG_2_POW_127, id="1x1x207x207_to256"),
]


@pytest.mark.parametrize("shape,padding,value", CASES)
def test_g6_pad(device, shape, padding, value):
    run_case(device, shape, padding, value)


def test_g6_pad_value_positional(device):
    run_case(device, [1, 3844, 1536], ((0, 0), (0, 28), (0, 0)), 0.0, value_positional=True)


def test_g6_pad_same_shape_new_value(device):
    # A program cache keyed without the fill value would replay the first value.
    run_case(device, [1, 1, 8, 62080], [[0, 0], [0, 0], [0, 8], [0, 0]], 0)
    run_case(device, [1, 1, 8, 62080], [[0, 0], [0, 0], [0, 8], [0, 0]], 7.0)
    run_case(device, [1, 1, 8, 62080], [[0, 0], [0, 0], [0, 24], [0, 0]], 0.0)
    run_case(device, [1, 1, 196, 196], [[0, 60], [0, 60]], NEG_2_POW_127)
    run_case(device, [1, 1, 196, 196], [[0, 60], [0, 60]], 3.5)


def test_g6_pad_inexact_value(device):
    # 0.1 is not exact in bfloat16; the fill must be torch's conversion of it.
    run_case(device, [1, 1, 8, 62080], [[0, 0], [0, 0], [0, 8], [0, 0]], 0.1)
    run_case(device, [1, 196, 128], [[0, 60], [0, 0]], 0.1)


@pytest.mark.parametrize(
    "shape,padding,value",
    [
        pytest.param([1, 1, 8, 62080], [[0, 0], [0, 0], [0, 8], [0, 0]], 0, id="extension"),
        pytest.param([1, 1, 196, 196], [[0, 60], [0, 60]], NEG_2_POW_127, id="new_storage"),
    ],
)
def test_g6_pad_special_values(device, shape, padding, value):
    torch_input = make_input(shape)
    tiny = torch.finfo(torch.bfloat16).tiny
    specials = torch.tensor([-0.0, float("inf"), float("-inf"), tiny, tiny / 4, -tiny / 8], dtype=torch.float32)
    flat = torch_input.reshape(-1)
    flat[: specials.numel()] = specials.to(torch.bfloat16)
    run_case(device, shape, padding, value, torch_input=flat.reshape(shape))


@pytest.mark.parametrize(
    "padding",
    [
        pytest.param([[0, 0], [1, 28], [0, 0]], id="low_nonzero"),
        pytest.param([[0, 0], [0, 28], [0, -4]], id="high_negative"),
        pytest.param([[0, 0], [0, 0], [0, 0], [0, 28]], id="more_pairs_than_dims"),
        pytest.param([], id="no_pairs"),
        pytest.param([[0, 0, 0], [0, 28], [0, 0]], id="pair_of_three"),
        pytest.param([[0, 0], [0, 2.5], [0, 0]], id="non_int_high"),
    ],
)
def test_g6_pad_rejects_bad_padding(device, expect_error, padding):
    x = to_device(make_input([1, 3844, 1536]), device, poison=None)
    with expect_error((ValueError, RuntimeError), ""):
        g6_pad(x, padding, value=0.0)


def test_g6_pad_rejects_bad_value(device, expect_error):
    x = to_device(make_input([1, 196, 128]), device, poison=None)
    with expect_error((ValueError, RuntimeError), ""):
        g6_pad(x, [[0, 60], [0, 0]], value="0")


def test_g6_pad_rejects_sharded_input(device, expect_error):
    memory_config = ttnn.create_sharded_memory_config(
        (1, 224, 128), core_grid=ttnn.CoreGrid(y=1, x=7), strategy=ttnn.ShardStrategy.HEIGHT
    )
    x = ttnn.from_torch(
        make_input([1, 196, 128]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    with expect_error((ValueError, RuntimeError), ""):
        g6_pad(x, [[0, 60], [0, 0]], value=0)
