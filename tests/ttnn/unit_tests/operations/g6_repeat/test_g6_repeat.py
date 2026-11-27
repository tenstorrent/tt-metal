# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_repeat: out = x.repeat(reps), graded bit for bit.

The op moves values and computes nothing, so every case compares the int16 bit
patterns of the bfloat16 readback (helpers.TOLERANCES is ("exact", None)).
Shapes stay inside the Phase 0 contract, x (1, 1, 1, W) with W % 32 == 0 and
reps [1, 1, R, 1] with 1 <= R < 32; each extra shape reaches a kernel branch
the production case does not (see op_design.md, "Acceptance test scope").
"""

import importlib

import pytest
import torch
import ttnn

from ttnn.operations.g6_repeat import g6_repeat, INPUT_TAGGERS

# NaN with payloads (quiet and signalling, both signs), +-inf, -0.0, denormals,
# the largest finite values.
# Every refusal raised by validate() starts with this prefix (op_design.md, host contract).
REFUSAL_MESSAGE = "^g6_repeat:"

BF16_SPECIAL_BITS = [0x7FC0, 0x7FA1, 0xFFC5, 0x7F81, 0x7F80, 0xFF80, 0x8000, 0x0001, 0x0040, 0x7F7F, 0xFF7F]


def pytorch_g6_repeat(x, reps):
    return x.repeat(*[int(r) for r in reps])


def _bits(t):
    return t.contiguous().view(torch.int16)


def _to_device(torch_x, device, memory_config=None):
    return ttnn.from_torch(
        torch_x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
    )


def _randn_row(w):
    return torch.randn((1, 1, 1, w), dtype=torch.float32).to(torch.bfloat16)


def _special_values_row(w):
    x = _randn_row(w)
    flat = _bits(x).reshape(-1).clone()
    patterns = [p - 0x10000 if p >= 0x8000 else p for p in BF16_SPECIAL_BITS]
    flat[: len(patterns)] = torch.tensor(patterns, dtype=torch.int16)
    return flat.view(torch.bfloat16).reshape(x.shape)


def _assert_bit_exact(ttnn_out, expected, ttnn_in):
    assert list(ttnn_out.shape) == list(expected.shape), f"shape {list(ttnn_out.shape)} != {list(expected.shape)}"
    assert ttnn_out.dtype == ttnn.bfloat16
    assert ttnn_out.layout == ttnn.TILE_LAYOUT
    mc = ttnn_out.memory_config()
    assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert mc.buffer_type == ttnn.BufferType.DRAM
    assert ttnn_out.buffer_address() != ttnn_in.buffer_address(), "out must be a new tensor, not a view of x"

    actual = ttnn.to_torch(ttnn_out)
    assert actual.dtype == torch.bfloat16
    got, want = _bits(actual), _bits(expected)
    if not torch.equal(got, want):
        differ = got != want
        first = tuple(int(i) for i in differ.nonzero()[0])
        pytest.fail(
            f"{int(differ.sum())} of {differ.numel()} logical elements differ; first at {first}: "
            f"got 0x{int(got[first]) & 0xFFFF:04X} expected 0x{int(want[first]) & 0xFFFF:04X}"
        )
    return actual


def _assert_uploaded_bits(ttnn_x, torch_x):
    readback = ttnn.to_torch(ttnn_x)
    assert torch.equal(_bits(readback), _bits(torch_x)), "harness precondition: x on device lost its host bit patterns"


@pytest.mark.parametrize(
    "w, r",
    [
        (256, 16),  # production: FLUX.2-dev scenario 75, 8 tiles, rows fill faces 0-1
        (32, 1),  # one tile, nothing to replicate
        (64, 17),  # first row in faces 2-3
        (96, 31),  # largest R of the contract
        (9600, 5),  # 300 tile columns: more than one per core, two core groups
    ],
    ids=["p1_w256_r16", "w32_r1", "w64_r17", "w96_r31", "w9600_r5"],
)
def test_g6_repeat(device, w, r):
    torch.manual_seed(42)
    torch_x = _randn_row(w)
    reps = [1, 1, r, 1]
    expected = pytorch_g6_repeat(torch_x, reps)

    ttnn_x = _to_device(torch_x, device)
    ttnn_out = g6_repeat(ttnn_x, reps)
    _assert_bit_exact(ttnn_out, expected, ttnn_x)


def test_g6_repeat_reps_tuple(device):
    torch.manual_seed(42)
    torch_x = _randn_row(256)
    reps = (1, 1, 16, 1)
    ttnn_x = _to_device(torch_x, device)
    _assert_bit_exact(g6_repeat(ttnn_x, reps), pytorch_g6_repeat(torch_x, reps), ttnn_x)


def test_g6_repeat_special_values(device):
    torch.manual_seed(42)
    torch_x = _special_values_row(256)
    reps = [1, 1, 16, 1]
    ttnn_x = _to_device(torch_x, device)
    _assert_uploaded_bits(ttnn_x, torch_x)
    _assert_bit_exact(g6_repeat(ttnn_x, reps), pytorch_g6_repeat(torch_x, reps), ttnn_x)


def _poisoned_padding_row(device, w, poison_bits):
    """x (1, 1, 1, W) whose tile padding (rows 1..31 of every tile) holds `poison_bits`.

    A (1, 1, 32, W) tile tensor with the poison in rows 1..31 is viewed as (1, 1, 1, W) over
    the same buffer. ttnn.fill_implicit_tile_padding cannot poison with NaN: it packs the fill
    through DEST and lands +inf (0x7F80) instead of the payload.
    """
    full = _bits(torch.randn((1, 1, 32, w), dtype=torch.float32).to(torch.bfloat16)).clone()
    signed = poison_bits - 0x10000 if poison_bits >= 0x8000 else poison_bits
    full[..., 1:, :] = signed
    full = full.view(torch.bfloat16)
    t32 = _to_device(full, device)
    x = ttnn.reshape(t32, ttnn.Shape([1, 1, 1, w]), ttnn.Shape([1, 1, 32, w]))
    assert x.buffer_address() == t32.buffer_address(), "harness precondition: the view must share the buffer"
    padding = _bits(x.cpu().to_torch_with_padded_shape()[..., 1:, :].contiguous())
    assert bool((padding == signed).all()), "harness precondition: the tile padding of x does not hold the poison"
    row = full[..., :1, :].contiguous()
    _assert_uploaded_bits(x, row)
    return row, x


# 0x447A = 1000.0 (fails a form that combines the 32 rows of a tile); NaN payloads of both
# signs (also fail a form that multiplies the padding rows by 0).
@pytest.mark.parametrize("poison_bits", [0x447A, 0x7FC1, 0xFFC5], ids=["p1000", "qnan_payload", "neg_qnan_payload"])
@pytest.mark.parametrize("w, r", [(256, 16), (64, 17), (96, 31)], ids=["p1_w256_r16", "w64_r17", "w96_r31"])
def test_g6_repeat_poisoned_input_padding(device, w, r, poison_bits):
    torch.manual_seed(42)
    row, ttnn_x = _poisoned_padding_row(device, w, poison_bits)
    reps = [1, 1, r, 1]
    _assert_bit_exact(g6_repeat(ttnn_x, reps), pytorch_g6_repeat(row, reps), ttnn_x)


def test_g6_repeat_determinism(device):
    torch.manual_seed(42)
    torch_x = _randn_row(256)
    reps = [1, 1, 16, 1]
    expected = pytorch_g6_repeat(torch_x, reps)
    ttnn_x = _to_device(torch_x, device)

    first = _assert_bit_exact(g6_repeat(ttnn_x, reps), expected, ttnn_x)
    second = _assert_bit_exact(g6_repeat(ttnn_x, reps), expected, ttnn_x)
    assert torch.equal(_bits(first), _bits(second)), "two calls on the same input differ"

    # Allocated while x and both outputs live, so it sits at another address:
    # a program that kept the first call's addresses fails here.
    other = _randn_row(256)
    ttnn_other = _to_device(other, device)
    _assert_bit_exact(g6_repeat(ttnn_other, reps), pytorch_g6_repeat(other, reps), ttnn_other)


def test_g6_repeat_multi_block(device, monkeypatch):
    """Several blocks per core and a ragged last block (slot reuse after the writes flush)."""
    knobs = importlib.import_module("ttnn.operations.g6_repeat.g6_repeat_program_descriptor")
    monkeypatch.setattr(knobs, "MAX_NUM_CORES", 2)
    monkeypatch.setattr(knobs, "MAX_BLOCK_WIDTH_TILES", 3)

    torch.manual_seed(42)
    torch_x = _randn_row(224)  # 7 tile columns on 2 cores: 4 (blocks 3 + 1) and 3 (one block)
    reps = [1, 1, 16, 1]
    ttnn_x = _to_device(torch_x, device)
    _assert_bit_exact(g6_repeat(ttnn_x, reps), pytorch_g6_repeat(torch_x, reps), ttnn_x)


@pytest.mark.parametrize(
    "shape, reps",
    [
        ((1, 1, 1, 256), [1, 16, 1]),  # reps rank differs from the input's
        ((1, 1, 256), [1, 16, 1]),  # rank 3
        ((1, 1, 1, 256), [1, 1, 0, 1]),  # zero count
        ((1, 1, 1, 256), [1, 1, -2, 1]),  # negative count
        ((1, 1, 1, 256), [1, 1, 16, 2]),  # dim 3 repeated
        ((1, 1, 1, 256), [2, 1, 16, 1]),  # dim 0 repeated
        ((1, 1, 1, 256), [1, 2, 16, 1]),  # dim 1 repeated
        ((1, 1, 1, 256), [1, 1, 32, 1]),  # out spans a whole tile row
        ((1, 1, 2, 256), [1, 1, 16, 1]),  # x has more than one row
        ((2, 1, 1, 256), [1, 1, 16, 1]),  # multi-batch x
    ],
    ids=[
        "reps_rank_mismatch",
        "rank3",
        "zero_rep",
        "negative_rep",
        "dim3_repeated",
        "dim0_repeated",
        "dim1_repeated",
        "r32",
        "x_two_rows",
        "multi_batch",
    ],
)
def test_g6_repeat_rejects_outside_contract(device, expect_error, shape, reps):
    torch.manual_seed(42)
    ttnn_x = _to_device(torch.randn(shape, dtype=torch.float32).to(torch.bfloat16), device)
    with expect_error((ValueError, RuntimeError), REFUSAL_MESSAGE):
        g6_repeat(ttnn_x, reps)


def test_g6_repeat_rejects_host_tensor(expect_error):
    torch.manual_seed(42)
    host_x = ttnn.from_torch(_randn_row(256), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    with expect_error((ValueError, RuntimeError), REFUSAL_MESSAGE):
        g6_repeat(host_x, [1, 1, 16, 1])


def test_g6_repeat_rejects_l1_interleaved(device, expect_error):
    torch.manual_seed(42)
    ttnn_x = _to_device(_randn_row(256), device, memory_config=ttnn.L1_MEMORY_CONFIG)
    with expect_error((ValueError, RuntimeError), REFUSAL_MESSAGE):
        g6_repeat(ttnn_x, [1, 1, 16, 1])


@pytest.mark.parametrize(
    "shape, reps, expected",
    [
        ([1, 1, 1, 256], [1, 1, 16, 1], "h_non_aligned"),
        ([1, 1, 1, 256], [1, 1, 32, 1], "tile_aligned"),
        ([1, 1, 1, 48], [1, 1, 16, 1], "w_non_aligned"),
        ([1, 1, 1, 48], [1, 1, 32, 1], "w_non_aligned"),
    ],
    ids=["production", "aligned", "w_and_h_non_aligned", "w_non_aligned_h_aligned"],
)
def test_tag_alignment(shape, reps, expected):
    tag = INPUT_TAGGERS["alignment"](({"shape": shape, "reps": reps},), {})
    assert tag == expected
