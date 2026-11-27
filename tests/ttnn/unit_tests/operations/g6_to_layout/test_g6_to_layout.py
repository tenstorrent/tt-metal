# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_to_layout, Phase 0: TILE bfloat16 -> ROW_MAJOR bfloat16, interleaved.

Shapes are production cases of the Phase 0 class (eval/golden_tests/g6_to_layout/feature_spec.py);
the remaining tests are the cases the spec's rules require. The op moves values and computes
nothing, so every value is graded bit for bit (helpers.TOLERANCES[bfloat16] == ("exact", None)).
No production case of this class is a single tile: the smallest, [1, 185], is six.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_to_layout import g6_to_layout

DRAM = ttnn.BufferType.DRAM
L1 = ttnn.BufferType.L1
RM = ttnn.ROW_MAJOR_LAYOUT

# (case id of the production case list, shape, input buffer)
PHASE0_CASES = [
    (309, (1, 185), DRAM),  # rank 2, H = 1, last tile column 25 wide
    (290, (1, 196), DRAM),  # last tile column 4 wide: left faces only
    (292, (32, 5120), DRAM),  # one tile-row of 160 tiles: split across columns
    (294, (1, 131072), DRAM),  # one row wider than a core's L1
    (2003, (1, 32, 2560), DRAM),
    (2038, (1, 32, 2560), L1),  # L1 input, the output keeps the L1 placement
    (1993, (16, 1, 2560), DRAM),  # multi-batch, H = 1
    (2028, (8, 128, 2560), DRAM),  # multi-batch, whole tile-rows per core
    (271, (1, 2048, 128), DRAM),
    (273, (1, 64, 128, 32), DRAM),  # one tile column, W = 32: staged_reorder
    (286, (1, 1024, 1024, 3), DRAM),  # one tile column, W = 3: same_bank_slot_writes
]

# -0.0, subnormals, +inf, -inf, NaN with payload
SPECIAL_BITS = (0x8000, 0x0001, 0x8001, 0x007F, 0x7F80, 0xFF80, 0x7FC1)

MODEL_L1_FREE_BYTES = 1347776
_FILLER_PAGE_BYTES = 2048
_GUARD_BITS = 0x5A5A


def _case_id(case):
    case_id, shape, buffer = case
    return f"c{case_id}-{'x'.join(str(d) for d in shape)}-{'l1' if buffer == L1 else 'dram'}"


def _param_id(value):
    if isinstance(value, tuple):
        return "x".join(str(d) for d in value)
    return str(value)


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
    flat[: len(SPECIAL_BITS)] = table
    return x


def _interleaved(buffer):
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer)


def _to_device(torch_input, device, buffer=DRAM):
    return ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_interleaved(buffer),
        preserve_nan_values=True,
    )


def pytorch_g6_to_layout(torch_input):
    return torch_input.clone()


def _bits(t):
    return t.contiguous().view(torch.int16)


def _to_torch(output, shape):
    return ttnn.to_torch(output).reshape(shape)


def _check(output, expected, buffer):
    assert list(output.shape) == list(expected.shape)
    assert output.dtype == ttnn.bfloat16
    assert output.layout == RM
    memory = output.memory_config()
    assert memory.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert memory.buffer_type == buffer
    actual = _to_torch(output, expected.shape)
    assert actual.dtype == torch.bfloat16
    differ = _bits(actual) != _bits(expected)
    count = int(differ.sum())
    assert (
        count == 0
    ), f"{count} of {differ.numel()} values differ bitwise, first at {tuple(differ.nonzero()[0].tolist())}"


@pytest.mark.parametrize("case", PHASE0_CASES, ids=_case_id)
def test_g6_to_layout(device, case):
    _, shape, buffer = case
    torch_input = _randn(shape)
    output = g6_to_layout(_to_device(torch_input, device, buffer), RM)
    _check(output, pytorch_g6_to_layout(torch_input), buffer)


@pytest.mark.parametrize("shape", [(1, 1024, 1024, 3), (16, 1, 2560), (1, 2048, 128)], ids=_param_id)
def test_special_bit_patterns(device, shape):
    torch_input = _with_specials(shape)
    output = g6_to_layout(_to_device(torch_input, device), RM)
    _check(output, pytorch_g6_to_layout(torch_input), DRAM)


@pytest.mark.parametrize("poison", [1000.0, float("nan")], ids=["poison_1000", "poison_nan"])
@pytest.mark.parametrize("shape", [(1, 185), (1, 1024, 1024, 3), (16, 1, 2560)], ids=_param_id)
def test_input_padding_never_reaches_output(device, shape, poison):
    torch_input = _randn(shape)
    ttnn_input = ttnn.fill_implicit_tile_padding(_to_device(torch_input, device), poison)
    output = g6_to_layout(ttnn_input, RM)
    _check(output, pytorch_g6_to_layout(torch_input), DRAM)


def test_repeat_is_bit_identical(device):
    shape = (1, 1024, 1024, 3)
    torch_input = _randn(shape)
    ttnn_input = _to_device(torch_input, device)
    first = _to_torch(g6_to_layout(ttnn_input, RM), shape)
    assert torch.equal(_bits(first), _bits(pytorch_g6_to_layout(torch_input)))
    for _ in range(2):
        again = _to_torch(g6_to_layout(ttnn_input, RM), shape)
        assert torch.equal(_bits(again), _bits(first))


@pytest.mark.parametrize(
    "shape,regime",
    [
        ((1, 1024, 1024, 3), "same_bank_slot_writes"),
        ((1, 2048, 2048, 3), "same_bank_slot_writes"),
        ((1, 64, 128, 32), "staged_reorder"),
        ((1, 185), "staged_reorder"),
        ((16, 4096, 2560), "staged_reorder"),
    ],
    ids=_param_id,
)
def test_regime_selection(device, shape, regime):
    from ttnn.operations.g6_to_layout import select_plan

    # The selection is shape-only: a device tensor of the shape is all it needs, so the largest
    # production shapes are pinned without moving their data.
    ttnn_input = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    assert select_plan(ttnn_input, RM)["regime"] == regime


def test_l1_output_within_model_budget(device):
    # Case 2038: the L1 input keeps its placement, so the output lands in L1 beside the model's
    # tensors. The input already lives outside the budget; the filler leaves exactly the model's
    # free bytes per bank, and guard pages next to them must survive the call.
    shape = (1, 32, 2560)
    torch_input = _randn(shape)
    ttnn_input = _to_device(torch_input, device, L1)
    view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    banks = view.num_banks
    filler_pages = (view.largest_contiguous_bytes_free_per_bank - MODEL_L1_FREE_BYTES - _FILLER_PAGE_BYTES) // (
        _FILLER_PAGE_BYTES
    )
    held = []
    if filler_pages > 0:
        held.append(
            ttnn.allocate_tensor_on_device(
                ttnn.Shape([filler_pages * banks, _FILLER_PAGE_BYTES // 2]),
                ttnn.bfloat16,
                RM,
                device,
                ttnn.L1_MEMORY_CONFIG,
            )
        )
    pattern = torch.full((banks, _FILLER_PAGE_BYTES // 2), _GUARD_BITS, dtype=torch.int16).view(torch.bfloat16)
    guard = ttnn.from_torch(pattern, dtype=ttnn.bfloat16, layout=RM, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

    output = g6_to_layout(ttnn_input, RM)

    assert int((_bits(ttnn.to_torch(guard)) != _GUARD_BITS).sum()) == 0, "guard pages beside the op were overwritten"
    _check(output, pytorch_g6_to_layout(torch_input), L1)
    del held


def test_rejects_layout_equal_to_input(device, expect_error):
    ttnn_input = _to_device(_randn((1, 32, 2560)), device)
    with expect_error((ValueError, RuntimeError), "g6_to_layout"):
        g6_to_layout(ttnn_input, ttnn.TILE_LAYOUT)


def test_rejects_bfloat8_b_row_major(device, expect_error):
    ttnn_input = _to_device(_randn((1, 32, 2560)), device)
    with expect_error((ValueError, RuntimeError), "g6_to_layout"):
        g6_to_layout(ttnn_input, RM, dtype=ttnn.bfloat8_b)


def test_rejects_rank_outside_2_to_4(device, expect_error):
    ttnn_input = _to_device(_randn((1, 1, 1, 32, 32)), device)
    with expect_error((ValueError, RuntimeError), "g6_to_layout"):
        g6_to_layout(ttnn_input, RM)


def test_rejects_sharded_memory_config(device, expect_error):
    ttnn_input = _to_device(_randn((1, 32, 2560)), device)
    sharded = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
            [32, 2560],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    with expect_error((ValueError, RuntimeError), "g6_to_layout"):
        g6_to_layout(ttnn_input, RM, memory_config=sharded)
