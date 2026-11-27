# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_embedding.

Every op call uses a production case of eval/golden_tests/g6_embedding/feature_spec.py (BB-22).
The op moves values and computes none, so results are graded by bit equality of the bfloat16
patterns (helpers.TOLERANCES is ("exact", None)), never by PCC.
"""

import functools

import pytest
import torch
import torch.nn.functional as F
import ttnn

from ttnn.operations._op_contract import UnsupportedAxisValue
from ttnn.operations.g6_embedding import g6_embedding, select_plan

SEED = 42
L1_BOUND_BYTES = 1_347_776

QWEN_SMALL = (262144, 64)
QWEN_LARGE = (248320, 1280)
FLUX_LARGE = (131072, 2560)
FLUX_NARROW_VOCAB = (32, 5120)
FLUX_RANK3 = (1, 256, 128)

# bfloat16 bit patterns that a float route rewrites: -0.0, two subnormals, +inf, -inf,
# quiet NaN, NaN with sign and payload set, signalling NaN.
SPECIAL_BITS = (0x8000, 0x0001, 0x8008, 0x7F80, 0xFF80, 0x7FC0, 0xFFFF, 0x7F81)


@functools.lru_cache(maxsize=2)
def host_table(weight_shape):
    """Shared across tests (a large table is 0.6 GB); callers must not mutate it."""
    generator = torch.Generator().manual_seed(SEED)
    return torch.randn(weight_shape, generator=generator, dtype=torch.bfloat16)


def table_of(weight):
    return weight[0] if weight.dim() == 3 else weight


def pytorch_reference(index, weight):
    return F.embedding(index.to(torch.int64), table_of(weight))


def to_device_index(index, device):
    return ttnn.from_torch(
        index,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def to_device_weight(weight, device):
    return ttnn.from_torch(
        weight,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def op_kwargs(memory_config):
    kwargs = {"layout": ttnn.TILE_LAYOUT, "runtime_rows": False}
    if memory_config == "dram":
        kwargs["memory_config"] = ttnn.DRAM_MEMORY_CONFIG
    return kwargs


def check_exact(output, expected):
    assert output.layout == ttnn.TILE_LAYOUT, f"layout {output.layout}"
    assert output.dtype == ttnn.bfloat16, f"dtype {output.dtype}"
    placement = output.memory_config()
    assert placement.buffer_type == ttnn.BufferType.DRAM, f"placement {placement}"
    assert placement.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, f"placement {placement}"
    actual = ttnn.to_torch(output)
    assert tuple(actual.shape) == tuple(expected.shape), f"shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    assert actual.dtype == torch.bfloat16
    # int16 views: value comparison would equate -0.0 with 0.0 and fail on NaN.
    got = actual.contiguous().view(torch.int16)
    want = expected.contiguous().view(torch.int16)
    differs = got != want
    if bool(differs.any()):
        first = tuple(int(i) for i in differs.nonzero()[0])
        raise AssertionError(
            f"{int(differs.sum())} of {differs.numel()} values differ; first at {first}: "
            f"got {int(got[first]) & 0xFFFF:#06x} expected {int(want[first]) & 0xFFFF:#06x}"
        )


def check_plan(plan, device, expected_regime):
    grid = device.compute_with_storage_grid_size()
    assert plan.l1_bytes_per_core <= L1_BOUND_BYTES, f"L1 {plan.l1_bytes_per_core} B > {L1_BOUND_BYTES} B"
    assert 1 < plan.num_cores <= grid.x * grid.y, f"num_cores {plan.num_cores}"
    assert plan.regime in ("whole_row", "column_blocked")
    if expected_regime is not None:
        assert plan.regime == expected_regime, f"regime {plan.regime}, expected {expected_regime}"


# (index_shape, weight_shape, memory_config as the model passes it, regime fixed for every real grid or None).
# Grouped by table: host_table keeps two tables.
CASES = [
    pytest.param((1, 1), FLUX_RANK3, None, "column_blocked", id="1x1_flux_rank3_weight"),
    pytest.param((1, 1), FLUX_NARROW_VOCAB, None, "column_blocked", id="1x1_flux5120"),
    pytest.param((1, 1), QWEN_SMALL, "dram", "column_blocked", id="1x1_qwen64"),
    pytest.param((2, 128), QWEN_SMALL, "dram", None, id="2x128_qwen64"),
    pytest.param((1, 4096), QWEN_SMALL, "dram", None, id="1x4096_qwen64"),
    pytest.param((1, 8), QWEN_LARGE, "dram", "column_blocked", id="1x8_qwen1280"),
    pytest.param((1, 77), QWEN_LARGE, "dram", "column_blocked", id="1x77_qwen1280"),
    pytest.param((16, 128), QWEN_LARGE, "dram", None, id="16x128_qwen1280"),
    pytest.param((1, 3844), QWEN_LARGE, "dram", "whole_row", id="1x3844_qwen1280_p4"),
    pytest.param((16, 1024), QWEN_LARGE, "dram", "whole_row", id="16x1024_qwen1280_p3"),
    pytest.param((1, 512), FLUX_LARGE, None, None, id="1x512_flux2560"),
]


@pytest.mark.parametrize("index_shape, weight_shape, memory_config, expected_regime", CASES)
def test_g6_embedding(device, index_shape, weight_shape, memory_config, expected_regime):
    torch.manual_seed(SEED)
    vocab = weight_shape[-2]
    index = torch.randint(0, vocab, index_shape, dtype=torch.int32)
    weight = host_table(weight_shape)

    tt_index = to_device_index(index, device)
    tt_weight = to_device_weight(weight, device)
    try:
        check_plan(select_plan(tt_index, tt_weight), device, expected_regime)
        output = g6_embedding(tt_index, tt_weight, **op_kwargs(memory_config))
        check_exact(output, pytorch_reference(index, weight))
    finally:
        ttnn.deallocate(tt_weight)


def place_special_rows(table, index):
    """A copy of `table` whose rows named by `index` hold SPECIAL_BITS, shifted per row so every
    column sees every pattern; written through an int16 view so no float conversion touches them."""
    special = table.clone()
    rows = torch.unique(index.to(torch.int64))
    bits = torch.tensor([b - 0x10000 if b >= 0x8000 else b for b in SPECIAL_BITS], dtype=torch.int16)
    width = special.shape[-1]
    pattern = bits[(torch.arange(width).unsqueeze(0) + rows.unsqueeze(1)) % len(bits)]
    special.view(-1, width).view(torch.int16)[rows] = pattern
    return special


@pytest.mark.parametrize("weight_shape", [QWEN_SMALL, QWEN_LARGE], ids=["qwen64", "qwen1280"])
def test_special_values_bit_exact(device, weight_shape):
    torch.manual_seed(SEED)
    index = torch.randint(0, weight_shape[-2], (1, 128), dtype=torch.int32)
    weight = place_special_rows(host_table(weight_shape), index)

    tt_weight = to_device_weight(weight, device)
    try:
        back = ttnn.to_torch(tt_weight)
        if not torch.equal(back.contiguous().view(torch.int16), weight.contiguous().view(torch.int16)):
            raise RuntimeError("the special bit patterns did not survive the host-to-device copy")
        output = g6_embedding(to_device_index(index, device), tt_weight, **op_kwargs("dram"))
        check_exact(output, pytorch_reference(index, weight))
    finally:
        ttnn.deallocate(tt_weight)


def test_new_index_values_same_program(device):
    """Rows are chosen on the device from the values of each call, never from the first call's."""
    torch.manual_seed(SEED)
    weight = host_table(QWEN_SMALL)
    first = torch.randint(0, QWEN_SMALL[0], (1, 4096), dtype=torch.int32)
    second = torch.randint(0, QWEN_SMALL[0], (1, 4096), dtype=torch.int32)

    tt_weight = to_device_weight(weight, device)
    try:
        for index in (first, second, first):
            output = g6_embedding(to_device_index(index, device), tt_weight, **op_kwargs("dram"))
            check_exact(output, pytorch_reference(index, weight))
    finally:
        ttnn.deallocate(tt_weight)


def test_runtime_rows_true_refused(device, expect_error):
    index = torch.zeros((1, 1), dtype=torch.int32)
    weight = host_table(FLUX_RANK3)
    with expect_error(UnsupportedAxisValue, "runtime_rows"):
        g6_embedding(
            to_device_index(index, device),
            to_device_weight(weight, device),
            layout=ttnn.TILE_LAYOUT,
            runtime_rows=True,
        )


def test_rank3_weight_leading_extent_refused(device, expect_error):
    index = torch.zeros((1, 1), dtype=torch.int32)
    weight = torch.zeros((2, 256, 128), dtype=torch.bfloat16)
    with expect_error((ValueError, RuntimeError), "weight"):
        g6_embedding(to_device_index(index, device), to_device_weight(weight, device), layout=ttnn.TILE_LAYOUT)
