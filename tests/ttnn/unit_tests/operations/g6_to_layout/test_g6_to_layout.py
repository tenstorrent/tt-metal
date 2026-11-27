# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_to_layout, Phase 0: TILE -> ROW_MAJOR, bfloat16, interleaved.

Shapes are production cases of eval/golden_tests/g6_to_layout/feature_spec.py only
(model rule BB-22); none of them is a single tile, the smallest is (1, 185).
Gate: helpers.TOLERANCES[bfloat16] is ("exact", None), so every logical value is
compared bit for bit, NaN payloads included.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_to_layout import g6_to_layout

DRAM = ttnn.BufferType.DRAM
L1 = ttnn.BufferType.L1

L1_OP_BUDGET_BYTES = 1347776
_L1_PAGE_BYTES = 2048
_GUARD_BITS = 0x5A5A
SPECIAL_BITS = (0x8000, 0x0001, 0x8001, 0x007F, 0x7F80, 0xFF80, 0x7FC1)

# (production case id, shape, input buffer)
VALUE_CASES = [
    (309, (1, 185), DRAM),  # rank 2, H = 1, W = 185
    (2025, (1, 32, 512), DRAM),  # one tile-row of 16 tiles
    (273, (1, 64, 128, 32), DRAM),  # rank 4, one tile column, many tile-rows
    (1993, (16, 1, 2560), DRAM),  # multi-batch, one live row per tile-row
    (2028, (8, 128, 2560), DRAM),  # multi-batch, tile-aligned, split by columns
    (292, (32, 5120), DRAM),  # rank 2, one tile-row of 160 tiles
    (293, (1, 1, 131072), DRAM),  # one 256 KiB row
    (271, (1, 2048, 128), DRAM),  # tall, four tile columns
    (2038, (1, 32, 2560), L1),  # L1 interleaved in, L1 out
    (286, (1, 1024, 1024, 3), DRAM),  # W = 3, blocks at the L1 cap
]


def _case_id(case):
    return f"c{case[0]}-{'x'.join(str(d) for d in case[1])}-{'L1' if case[2] == L1 else 'DRAM'}"


def _bits(t):
    return t.contiguous().view(torch.int16)


def pytorch_g6_to_layout(x):
    return x.clone()


def _to_device(torch_input, device, buffer):
    return ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buffer),
        preserve_nan_values=True,
    )


def _check_contract(output, shape, buffer):
    assert list(output.shape) == list(shape)
    assert output.dtype == ttnn.bfloat16
    assert output.layout == ttnn.ROW_MAJOR_LAYOUT
    memory = output.memory_config()
    assert memory.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert memory.buffer_type == buffer


def _assert_bit_identical(output, expected):
    actual = ttnn.to_torch(output).reshape(expected.shape).to(torch.bfloat16)
    differ = _bits(actual) != _bits(expected)
    count = int(differ.sum())
    if count:
        index = tuple(int(i) for i in differ.nonzero()[0])
        got = int(_bits(actual)[index]) & 0xFFFF
        want = int(_bits(expected)[index]) & 0xFFFF
        pytest.fail(f"{count} of {differ.numel()} values differ; first at {index}: got 0x{got:04X} want 0x{want:04X}")


def _with_specials(shape, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    x = torch.randn(shape, dtype=torch.bfloat16, generator=generator)
    flat = x.reshape(-1).view(torch.int16)
    count = max(len(SPECIAL_BITS), flat.numel() // 100)
    positions = torch.randperm(flat.numel(), generator=generator)[:count]
    table = torch.tensor([b - 0x10000 if b >= 0x8000 else b for b in SPECIAL_BITS], dtype=torch.int16)
    # Every pattern appears at least once, the rest are drawn at random.
    extra = torch.randint(0, len(SPECIAL_BITS), (count - len(SPECIAL_BITS),), generator=generator)
    picks = torch.cat([torch.arange(len(SPECIAL_BITS)), extra])
    flat[positions] = table[picks]
    return x


@pytest.mark.parametrize("case", VALUE_CASES, ids=_case_id)
def test_g6_to_layout_values(case, device):
    torch.manual_seed(42)
    _, shape, buffer = case
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    tt_input = _to_device(torch_input, device, buffer)

    output = g6_to_layout(tt_input, ttnn.ROW_MAJOR_LAYOUT)

    _check_contract(output, shape, buffer)
    _assert_bit_identical(output, pytorch_g6_to_layout(torch_input))


@pytest.mark.parametrize(
    "case",
    [
        (309, (1, 185), DRAM),
        (273, (1, 64, 128, 32), DRAM),
        (2028, (8, 128, 2560), DRAM),
        (286, (1, 1024, 1024, 3), DRAM),
    ],
    ids=_case_id,
)
def test_g6_to_layout_special_bit_patterns(case, device):
    """-0.0, subnormals, +-inf and a NaN payload must come out bit for bit."""
    _, shape, buffer = case
    torch_input = _with_specials(shape, seed=1234)
    tt_input = _to_device(torch_input, device, buffer)

    output = g6_to_layout(tt_input, ttnn.ROW_MAJOR_LAYOUT)

    _check_contract(output, shape, buffer)
    _assert_bit_identical(output, pytorch_g6_to_layout(torch_input))


@pytest.mark.parametrize("poison", [1000.0, float("nan")], ids=["1000", "nan"])
@pytest.mark.parametrize(
    "case",
    [(309, (1, 185), DRAM), (1993, (16, 1, 2560), DRAM), (286, (1, 1024, 1024, 3), DRAM)],
    ids=_case_id,
)
def test_g6_to_layout_ignores_input_padding(case, poison, device):
    """Tile padding of the input never reaches a logical output value."""
    torch.manual_seed(42)
    _, shape, buffer = case
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    tt_input = ttnn.fill_implicit_tile_padding(_to_device(torch_input, device, buffer), poison)

    output = g6_to_layout(tt_input, ttnn.ROW_MAJOR_LAYOUT)

    _check_contract(output, shape, buffer)
    _assert_bit_identical(output, pytorch_g6_to_layout(torch_input))


def _squeeze_l1(device, free_bytes):
    """Leave `free_bytes` per L1 bank free, then place one 0x5A5A guard page per
    bank right below, so an op that overruns its share overwrites the guard."""
    view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    banks = view.num_banks
    filler_pages = (view.largest_contiguous_bytes_free_per_bank - free_bytes - _L1_PAGE_BYTES) // _L1_PAGE_BYTES
    held = []
    if filler_pages > 0:
        held.append(
            ttnn.allocate_tensor_on_device(
                ttnn.Shape([filler_pages * banks, _L1_PAGE_BYTES // 2]),
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
                device,
                ttnn.L1_MEMORY_CONFIG,
            )
        )
    pattern = torch.full((banks, _L1_PAGE_BYTES // 2), _GUARD_BITS, dtype=torch.int16).view(torch.bfloat16)
    guard = ttnn.from_torch(
        pattern,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        preserve_nan_values=True,
    )
    return held, guard


@pytest.mark.parametrize("case", [(286, (1, 1024, 1024, 3), DRAM), (2038, (1, 32, 2560), L1)], ids=_case_id)
def test_g6_to_layout_fits_model_l1(case, device):
    """BB-05: CBs plus an L1 output's pages fit in the 1347776 B the model leaves."""
    torch.manual_seed(42)
    _, shape, buffer = case
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    tt_input = _to_device(torch_input, device, buffer)
    held, guard = _squeeze_l1(device, L1_OP_BUDGET_BYTES)

    output = g6_to_layout(tt_input, ttnn.ROW_MAJOR_LAYOUT)

    changed = int((_bits(ttnn.to_torch(guard)) != _GUARD_BITS).sum())
    assert changed == 0, f"{changed} guard values overwritten"
    _check_contract(output, shape, buffer)
    _assert_bit_identical(output, pytorch_g6_to_layout(torch_input))
    del held


def test_g6_to_layout_repeat_is_bit_identical(device):
    """BB-06: the same call twice gives the same bits."""
    torch.manual_seed(42)
    shape = (8, 128, 2560)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    tt_input = _to_device(torch_input, device, DRAM)

    first = ttnn.to_torch(g6_to_layout(tt_input, ttnn.ROW_MAJOR_LAYOUT))
    second = ttnn.to_torch(g6_to_layout(tt_input, ttnn.ROW_MAJOR_LAYOUT))

    assert torch.equal(_bits(first), _bits(second))
    assert torch.equal(_bits(first.reshape(shape).to(torch.bfloat16)), _bits(torch_input))


def test_g6_to_layout_rejects_same_layout(device, expect_error):
    tt_input = _to_device(torch.randn((1, 32, 512), dtype=torch.bfloat16), device, DRAM)
    with expect_error((ValueError, RuntimeError), r"(?i)layout"):
        g6_to_layout(tt_input, ttnn.TILE_LAYOUT)


def test_g6_to_layout_rejects_bf8b_row_major(device, expect_error):
    tt_input = _to_device(torch.randn((1, 32, 512), dtype=torch.bfloat16), device, DRAM)
    with expect_error((ValueError, RuntimeError), r"(?i)bfloat8_b|dtype"):
        g6_to_layout(tt_input, ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat8_b)


def test_g6_to_layout_rejects_sharded_output(device, expect_error):
    tt_input = _to_device(torch.randn((1, 32, 512), dtype=torch.bfloat16), device, DRAM)
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    sharded = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core, [32, 512], ttnn.ShardOrientation.ROW_MAJOR),
    )
    with expect_error((ValueError, RuntimeError), r"(?i)shard"):
        g6_to_layout(tt_input, ttnn.ROW_MAJOR_LAYOUT, memory_config=sharded)


def test_g6_to_layout_rejects_rank_5(device, expect_error):
    tt_input = _to_device(torch.randn((1, 1, 1, 32, 32), dtype=torch.bfloat16), device, DRAM)
    with expect_error((ValueError, RuntimeError), r"(?i)rank"):
        g6_to_layout(tt_input, ttnn.ROW_MAJOR_LAYOUT)
