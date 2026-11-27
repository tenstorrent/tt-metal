# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_pad, Phase 0: bfloat16, TILE_LAYOUT, DRAM interleaved.

Cases are production cases of the contract plus the cases its rules require (BB-22).
The gate is bit equality against torch.nn.functional.pad: helpers.TOLERANCES is
("exact", None) for every dtype.
"""

import pytest
import torch

import ttnn
from ttnn.operations._op_contract import SupportRefusal
from ttnn.operations.g6_pad import g6_pad, g6_pad_plan

TILE = 32
MASK_FILL = -1.7014118346046923e38  # -2**127, finite in bfloat16

# Device is module-scoped by conftest.py (use_module_device hook).


def to_torch_pad(padding):
    flat = []
    for low, high in reversed([tuple(pair) for pair in padding]):
        flat += [int(low), int(high)]
    return flat


def pytorch_reference(torch_input, padding, value):
    return torch.nn.functional.pad(torch_input, to_torch_pad(padding), mode="constant", value=value)


def _highs(shape, padding):
    return [0] * (len(shape) - len(padding)) + [pair[1] for pair in padding]


def _is_extension(shape, padding):
    highs = _highs(shape, padding)
    only_h = highs[-2] > 0 and all(h == 0 for i, h in enumerate(highs) if i != len(highs) - 2)
    same_tile_rows = -(-shape[-2] // TILE) == -(-(shape[-2] + highs[-2]) // TILE)
    return only_h and same_tile_rows


def _work_units(shape, padding):
    """R1: last-row tiles of every slice. R2: max(staged pages, fill pages)."""
    highs = _highs(shape, padding)
    out_shape = [d + h for d, h in zip(shape, highs)]
    slices_in = 1
    for d in shape[:-2]:
        slices_in *= d
    slices_out = 1
    for d in out_shape[:-2]:
        slices_out *= d
    wt_in = -(-shape[-1] // TILE)
    if _is_extension(shape, padding):
        return slices_in * wt_in
    staged = slices_in * -(-shape[-2] // TILE) * wt_in
    total = slices_out * -(-out_shape[-2] // TILE) * -(-out_shape[-1] // TILE)
    return max(staged, total - staged)


def _assert_bit_exact(actual, expected, what="output"):
    assert list(actual.shape) == list(expected.shape), f"{what} shape {list(actual.shape)} != {list(expected.shape)}"
    assert actual.dtype == torch.bfloat16, f"{what} dtype {actual.dtype}"
    diff = actual.contiguous().view(torch.int16) != expected.contiguous().view(torch.int16)
    if bool(diff.any()):
        first = tuple(int(i) for i in diff.nonzero()[0])
        raise AssertionError(
            f"{what}: {int(diff.sum())} of {diff.numel()} positions differ in bits, first {first}: "
            f"got {actual[first]} expected {expected[first]}"
        )


def _padded_rows(tensor):
    raw = tensor.cpu().to_torch_with_padded_shape()
    return raw.reshape(-1, raw.shape[-2], raw.shape[-1])


def _make_input(shape, device, poison=None):
    torch_input = (torch.randn(shape, dtype=torch.float32) * 100).to(torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if poison is not None:
        ttnn_input = ttnn.fill_implicit_tile_padding(ttnn_input, poison)
    return torch_input, ttnn_input


def _call_one_program(ttnn_input, padding, value, positional):
    """BB-01: the call dispatches exactly one device program (a view must dispatch none)."""
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        if positional:
            out = g6_pad(ttnn_input, padding, value)
        else:
            out = g6_pad(ttnn_input, padding, value=value)
    finally:
        graph = ttnn.graph.end_graph_capture()
    programs = [
        node["params"].get("name", "?")
        for node in graph
        if node.get("node_type") == "function_start" and "program_factory_type" in node.get("params", {})
    ]
    assert len(programs) == 1, f"the call dispatched {len(programs)} device programs: {programs}"
    return out


def _run_case(device, shape, padding, value, *, positional=False, poison=None, check_untouched=False, keep=None):
    torch_input, ttnn_input = _make_input(shape, device, poison)
    expected = pytorch_reference(torch_input, padding, value)
    extension = _is_extension(shape, padding)

    plan = g6_pad_plan(ttnn_input, padding, value)
    assert plan.regime == ("extension_band" if extension else "new_storage_tiles"), plan.regime
    if _work_units(shape, padding) > 1:
        assert plan.num_cores > 1, f"{plan.num_cores} core for {_work_units(shape, padding)} work units"

    raw_before = _padded_rows(ttnn_input) if check_untouched else None

    ttnn_output = _call_one_program(ttnn_input, padding, value, positional)
    if keep is not None:
        keep.extend([ttnn_input, ttnn_output])

    assert ttnn_output.dtype == ttnn.bfloat16
    assert ttnn_output.layout == ttnn.TILE_LAYOUT
    assert ttnn_output.memory_config().buffer_type == ttnn.BufferType.DRAM
    _assert_bit_exact(ttnn.to_torch(ttnn_output), expected)

    if extension:
        # The output takes over the input's storage: same buffer, same tile-padded shape.
        assert ttnn_output.buffer_address() == ttnn_input.buffer_address()
        assert list(ttnn_output.padded_shape) == list(ttnn_input.padded_shape)
        assert list(ttnn_input.shape) == list(shape)
    else:
        assert ttnn_output.buffer_address() != ttnn_input.buffer_address()
    _assert_bit_exact(ttnn.to_torch(ttnn_input), torch_input, what="input after the call")

    if check_untouched:
        raw_after = _padded_rows(ttnn_input)
        keep_rows = raw_after.shape[-2] - TILE
        _assert_bit_exact(
            raw_after[:, :keep_rows, :], raw_before[:, :keep_rows, :], what="input tile rows before the last"
        )
    return ttnn_output


EXTENSION_CASES = [
    # (shape, padding, value, value passed positionally)
    pytest.param((1, 31, 12), [[0, 0], [0, 1], [0, 0]], 0.0, False, id="single_tile_1x31x12_to_32"),
    pytest.param((16, 1, 1536), [[0, 0], [0, 31], [0, 0]], 0.0, False, id="multi_batch_16x1x1536_to_32"),
    pytest.param((1, 3844, 1536), [[0, 0], [0, 28], [0, 0]], 0.0, True, id="multi_tile_row_1x3844x1536_to_3872"),
    pytest.param((1, 1000, 12), [[0, 0], [0, 24], [0, 0]], 0.0, False, id="narrow_1x1000x12_to_1024"),
    pytest.param((1, 1, 2, 62080), [[0, 0], [0, 0], [0, 30], [0, 0]], 0.0, False, id="p2_1x1x2x62080_to_32"),
    pytest.param((1, 1, 8, 62080), [[0, 0], [0, 0], [0, 8], [0, 0]], 0, False, id="p3_1x1x8x62080_to_16"),
    pytest.param((1, 1, 8, 62080), [[0, 0], [0, 0], [0, 24], [0, 0]], 0.0, False, id="p4_1x1x8x62080_to_32"),
    pytest.param((1, 1, 16, 62080), [[0, 0], [0, 0], [0, 16], [0, 0]], 0.0, False, id="lower_half_1x1x16x62080_to_32"),
    pytest.param((1, 1, 1, 62080), [[0, 0], [0, 0], [0, 7], [0, 0]], 0, False, id="one_row_1x1x1x62080_to_8"),
]

NEW_STORAGE_CASES = [
    pytest.param((1, 196, 128), [[0, 60], [0, 0]], 0, False, id="non_square_1x196x128_to_256"),
    pytest.param((1, 207, 128), [[0, 49], [0, 0]], 0, False, id="non_square_1x207x128_to_256"),
    pytest.param((1, 1, 196, 196), [[0, 60], [0, 60]], MASK_FILL, False, id="hw_1x1x196x196_to_256_mask_fill"),
    pytest.param((1, 1, 185, 185), [[0, 71], [0, 71]], MASK_FILL, False, id="hw_1x1x185x185_to_256_mask_fill"),
]


@pytest.mark.parametrize("shape, padding, value, positional", EXTENSION_CASES + NEW_STORAGE_CASES)
def test_g6_pad(device, shape, padding, value, positional):
    torch.manual_seed(42)
    _run_case(device, shape, padding, value, positional=positional)


@pytest.mark.parametrize(
    "shape, padding, value",
    [
        pytest.param((1, 1, 8, 62080), [[0, 0], [0, 0], [0, 8], [0, 0]], 0, id="extension_p3"),
        pytest.param((1, 196, 128), [[0, 60], [0, 0]], 0, id="new_storage_rows"),
        pytest.param((1, 1, 196, 196), [[0, 60], [0, 60]], MASK_FILL, id="new_storage_hw"),
    ],
)
def test_poisoned_input_padding(device, shape, padding, value):
    """Logical output positions inside the input's tile padding hold c, never the producer's leftovers."""
    torch.manual_seed(42)
    _run_case(device, shape, padding, value, poison=1000.0)


@pytest.mark.parametrize(
    "shape, padding",
    [
        pytest.param((1, 3844, 1536), [[0, 0], [0, 28], [0, 0]], id="1x3844x1536"),
        pytest.param((1, 1000, 12), [[0, 0], [0, 24], [0, 0]], id="1x1000x12"),
    ],
)
def test_extension_untouched_rows(device, shape, padding):
    """An extension call reads and writes only the last tile row of each slice."""
    torch.manual_seed(42)
    _run_case(device, shape, padding, 0.0, positional=True, poison=1000.0, check_untouched=True)


def test_same_shape_two_values(device):
    """BB-07: padding and value are per-call data; a cached program must not replay a stale c."""
    torch.manual_seed(42)
    keep = []
    shape, padding = (1, 1, 8, 62080), [[0, 0], [0, 0], [0, 8], [0, 0]]
    _run_case(device, shape, padding, 0, keep=keep)
    _run_case(device, shape, padding, 7.0, keep=keep)
    _run_case(device, shape, [[0, 0], [0, 0], [0, 24], [0, 0]], 0.0, poison=1000.0, keep=keep)


def test_inexact_value(device):
    """c is torch's conversion of the value: 0.1 rounds to nearest even in bfloat16."""
    torch.manual_seed(42)
    _run_case(device, (1, 1, 8, 62080), [[0, 0], [0, 0], [0, 8], [0, 0]], 0.1)


def _assert_refused(expect_error, call):
    # The contract fixes the exception types, not the message text.
    with expect_error((ValueError, RuntimeError), "") as info:
        call()
    assert not isinstance(info.value, SupportRefusal), f"argument refusal raised a support refusal: {info.value!r}"


@pytest.mark.parametrize(
    "shape, padding, value",
    [
        pytest.param((1, 3844, 1536), [[0, 0], [1, 28], [0, 0]], 0.0, id="low_not_zero"),
        pytest.param((1, 3844, 1536), [[0, 0], [0, 28], [0, -4]], 0.0, id="negative_high"),
        pytest.param((1, 1, 8, 62080), [[0, 0], [0, 0], [0, 0], [0, 8], [0, 0]], 0, id="more_pairs_than_dims"),
        pytest.param((1, 1, 8, 62080), [[0, 0], [0, 0], [0, 8], [0, 0]], "0", id="value_not_a_number"),
    ],
)
def test_argument_refusals(device, expect_error, shape, padding, value):
    torch.manual_seed(42)
    _, ttnn_input = _make_input(shape, device)
    _assert_refused(expect_error, lambda: g6_pad(ttnn_input, padding, value=value))


def test_sharded_input_refused(device, expect_error):
    from eval.sharding import auto_shard_config

    torch.manual_seed(42)
    shape = (1, 196, 128)
    memory_config = auto_shard_config(
        list(shape), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
    )
    ttnn_input = ttnn.from_torch(
        torch.randn(shape, dtype=torch.float32).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    _assert_refused(expect_error, lambda: g6_pad(ttnn_input, [[0, 60], [0, 0]], value=0))
