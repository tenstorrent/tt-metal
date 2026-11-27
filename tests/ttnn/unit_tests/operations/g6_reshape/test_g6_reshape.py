# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_reshape, Phase 0: bfloat16 and float32, TILE_LAYOUT, DRAM interleaved.

g6_reshape moves values and never computes on them, so every check is bit-exact
(the golden suite's TOLERANCES are ("exact", None) for every dtype).
"""

import importlib

import pytest
import torch

import ttnn
from ttnn.operations.g6_reshape import describe_plan, g6_reshape

_op_module = importlib.import_module("ttnn.operations.g6_reshape.g6_reshape")

# BB-05: L1 the model leaves to the op on every core.
L1_OP_BUDGET = 1347776

# The bfloat16 TILE production cases of feature_spec.INPUTS: (input shape, target shape).
PRODUCTION_CASES = [
    ((1, 16, 6, 256), (1, 1, 16, 1536)),
    ((16, 1, 64), (1, 1, 16, 64)),
    ((16, 1, 1536), (1, 1, 16, 1536)),
    ((1, 1, 16, 4608), (16, 1, 4608)),
    ((1, 16, 64), (16, 1, 64)),
    ((16, 1, 5120), (1, 1, 16, 5120)),
    ((2, 1, 5120), (1, 1, 2, 5120)),
    ((1, 8, 6, 256), (1, 1, 8, 1536)),
    ((8, 1, 64), (1, 1, 8, 64)),
    ((8, 1, 1536), (1, 1, 8, 1536)),
    ((1, 1, 8, 4608), (8, 1, 4608)),
    ((1, 8, 64), (8, 1, 64)),
    ((8, 1, 5120), (1, 1, 8, 5120)),
    ((1, 1, 6, 256), (1, 1, 1536)),
]

# The FLUX row-to-column geometry at the Phase 0 dtype. A last dim that is not a multiple
# of 16 pins the element-run regime, which Phase 0 claims for every bfloat16 TILE shape.
ELEMENT_RUN_CASES = [
    ((1, 185), (1, 1, 185, 1)),
    ((1, 512), (1, 1, 512, 1)),
]

# The FLUX float32 production cases: one valid row of each input tile becomes one valid column of an
# output tile. None of the last dims is a multiple of 16 on both sides, so all run the element-run regime.
FLOAT32_CASES = [
    ((1, 512), (1, 1, 512, 1)),
    ((1, 256), (1, 1, 256, 1)),
    ((1, 196), (1, 1, 196, 1)),
    ((1, 185), (1, 1, 185, 1)),
    ((1, 207), (1, 1, 207, 1)),
]

# NaN, infinities, -0.0, denormals and a large finite value; -1e-45 is a float32-only denormal.
_BF16_SPECIALS = [float("nan"), float("inf"), float("-inf"), -0.0, 9.18e-41, 3.0e38]
_F32_SPECIALS = [float("nan"), float("inf"), float("-inf"), -0.0, 1e-40, -1e-45, 3.0e38]

_TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}
_TTNN_DTYPE = {v: k for k, v in _TORCH_DTYPE.items()}
_BITS_DTYPE = {torch.bfloat16: torch.int16, torch.float32: torch.int32}


def _case_id(case):
    src, dst = case
    return "x".join(map(str, src)) + "-to-" + "x".join(map(str, dst))


def _random_input(shape, dtype=ttnn.bfloat16):
    torch.manual_seed(42)
    return torch.randn(shape).to(_TORCH_DTYPE[dtype])


def _to_device(torch_tensor, device):
    return ttnn.from_torch(
        torch_tensor,
        dtype=_TTNN_DTYPE[torch_tensor.dtype],
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        preserve_nan_values=True,
    )


def _bits(t):
    return t.contiguous().view(_BITS_DTYPE[t.dtype])


def _check_exact(ttnn_output, expected):
    assert tuple(ttnn_output.shape) == tuple(expected.shape)
    assert ttnn_output.dtype == _TTNN_DTYPE[expected.dtype]
    assert ttnn_output.layout == ttnn.TILE_LAYOUT
    memory_config = ttnn_output.memory_config()
    assert memory_config.buffer_type == ttnn.BufferType.DRAM
    assert memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    got = ttnn.to_torch(ttnn_output)
    assert got.dtype == expected.dtype
    assert tuple(got.shape) == tuple(expected.shape)
    mismatch = _bits(got) != _bits(expected)
    assert not mismatch.any(), f"{int(mismatch.sum())} of {mismatch.numel()} values differ"
    return got


def _run_and_check(torch_input, target, device, *, poison=None):
    ttnn_input = _to_device(torch_input, device)
    if poison is not None:
        ttnn_input = ttnn.fill_implicit_tile_padding(ttnn_input, poison)
    output = g6_reshape(ttnn_input, target)
    return _check_exact(output, torch.reshape(torch_input, tuple(target)))


def _check_plan(torch_input, target, device, regime):
    plan = describe_plan(_to_device(torch_input, device), target)
    assert plan["regime"] == regime
    assert plan["num_cores"] > 1
    assert plan["l1_bytes_per_core"] <= L1_OP_BUDGET
    return plan


@pytest.mark.parametrize("case", PRODUCTION_CASES, ids=_case_id)
def test_production_cases(case, device):
    src, dst = case
    torch_input = _random_input(src)
    _run_and_check(torch_input, list(dst), device)
    _check_plan(torch_input, list(dst), device, "tile_face_row")


@pytest.mark.parametrize("case", ELEMENT_RUN_CASES, ids=_case_id)
def test_element_run_cases(case, device):
    src, dst = case
    torch_input = _random_input(src)
    _run_and_check(torch_input, list(dst), device)
    _check_plan(torch_input, list(dst), device, "tile_element_run")


# NaN poison in the input tile padding: every FLUX input tile holds one valid row of 31 padding rows.
@pytest.mark.parametrize("case", FLOAT32_CASES, ids=_case_id)
def test_float32_cases(case, device):
    src, dst = case
    torch_input = _random_input(src, ttnn.float32)
    _run_and_check(torch_input, list(dst), device, poison=float("nan"))
    _check_plan(torch_input, list(dst), device, "tile_element_run")


def test_float32_special_values(device):
    src, dst = (1, 512), [1, 1, 512, 1]
    flat = _random_input(src, ttnn.float32).reshape(-1)
    # An odd mantissa bit everywhere: a path through a narrower or rounding format changes it.
    flat = (flat.view(torch.int32) | 1).view(torch.float32).clone()
    stride = flat.numel() // len(_F32_SPECIALS)
    for i, value in enumerate(_F32_SPECIALS):
        flat[i * stride] = value
    _run_and_check(flat.reshape(src), dst, device)


def test_tuple_shape(device):
    torch_input = _random_input((1, 1, 16, 4608))
    _run_and_check(torch_input, (16, 1, 4608), device)


@pytest.mark.parametrize(
    "src, target, resolved",
    [
        ((16, 1, 1536), [1, 1, -1, 1536], (1, 1, 16, 1536)),
        ((1, 16, 6, 256), [-1, 1536], (16, 1536)),
    ],
)
def test_inferred_entry(src, target, resolved, device):
    torch_input = _random_input(src)
    output = g6_reshape(_to_device(torch_input, device), target)
    assert tuple(output.shape) == resolved
    _check_exact(output, torch.reshape(torch_input, resolved))


# Input tile padding filled with NaN must never reach a logical output position; from_torch
# pads with zeros and would hide such a leak.
@pytest.mark.parametrize(
    "case",
    [
        ((16, 1, 64), (1, 1, 16, 64)),
        ((1, 16, 6, 256), (1, 1, 16, 1536)),
        ((1, 1, 16, 4608), (16, 1, 4608)),
        ((1, 185), (1, 1, 185, 1)),
    ],
    ids=_case_id,
)
def test_poisoned_padding(case, device):
    src, dst = case
    _run_and_check(_random_input(src), list(dst), device, poison=float("nan"))


def test_special_values(device):
    src, dst = (1, 16, 6, 256), [1, 1, 16, 1536]
    flat = _random_input(src).reshape(-1)
    # An odd mantissa bit everywhere: a path through a narrower or rounding format changes it.
    flat = (flat.view(torch.int16) | 1).view(torch.bfloat16).clone()
    stride = flat.numel() // len(_BF16_SPECIALS)
    for i, value in enumerate(_BF16_SPECIALS):
        flat[i * stride] = value
    _run_and_check(flat.reshape(src), dst, device)


@pytest.mark.parametrize(
    "case",
    [((1, 16, 6, 256), (1, 1, 16, 1536)), ((1, 185), (1, 1, 185, 1))],
    ids=_case_id,
)
def test_round_trip(case, device):
    src, dst = case
    torch_input = _random_input(src)
    forward = g6_reshape(_to_device(torch_input, device), list(dst))
    back = g6_reshape(forward, list(src))
    _check_exact(back, torch_input)


# One input, two targets with different page maps, then the first again: a program cache key
# that misses the target shape replays the wrong program; repeats must be bit-identical.
def test_retarget_and_repeat(device):
    torch_input = _random_input((16, 1, 64))
    ttnn_input = _to_device(torch_input, device)
    first = _check_exact(g6_reshape(ttnn_input, [1, 1, 16, 64]), torch.reshape(torch_input, (1, 1, 16, 64)))
    _check_exact(g6_reshape(ttnn_input, [1, 1, 1, 1024]), torch.reshape(torch_input, (1, 1, 1, 1024)))
    again = _check_exact(g6_reshape(ttnn_input, [1, 1, 16, 64]), torch.reshape(torch_input, (1, 1, 16, 64)))
    assert torch.equal(_bits(first), _bits(again))


# (16,1,64)->(1,1,16,64) produces 2 output tiles from 32 input tiles: the core count must come
# from the blocking, not from the output tile count.
def test_core_count_not_capped_by_output_tiles(device):
    plan = describe_plan(_to_device(_random_input((16, 1, 64)), device), [1, 1, 16, 64])
    assert plan["num_cores"] > 2


@pytest.mark.parametrize(
    "target",
    [
        [1, 1, 16, 65],  # product differs from the element count
        [0, 1024],  # zero entry
        [-2, -512],  # entry below -1, product still equals the element count
        [-1, -1, 1024],  # more than one -1
        [1, 1, 1, 16, 64],  # output rank 5
        [],  # output rank 0
    ],
)
def test_invalid_shape(target, device, expect_error):
    ttnn_input = _to_device(_random_input((16, 1, 64)), device)
    with expect_error((ValueError, RuntimeError), ""):
        g6_reshape(ttnn_input, target)


def test_invalid_input_rank(device, expect_error):
    ttnn_input = _to_device(_random_input((1, 1, 1, 16, 64)), device)
    with expect_error((ValueError, RuntimeError), ""):
        g6_reshape(ttnn_input, [16, 64])


# Every golden shape is one block per mover on the full grid, so the block loop, the depth-2 slot
# rotation, a single mover and a single core only run when the knobs force them. Each setting gives
# every case more than one block per mover; NaN padding catches a block boundary that leaks padding.
KNOB_SETTINGS = {
    "block1": {"MAX_CORES": 8, "MAX_BLOCK_ATOMS": 1},
    "block3_depth2": {"MAX_CORES": 8, "MAX_BLOCK_ATOMS": 3, "STAGE_DEPTH": 2, "IMAGE_DEPTH": 2},
    "one_mover": {"MAX_CORES": 8, "MOVERS_PER_CORE": 1, "MAX_BLOCK_ATOMS": 4},
    "one_core": {"MAX_CORES": 1, "MAX_BLOCK_ATOMS": 16},
}


# float32 has one-row atoms (64-B face-rows fill a DRAM word), so its block boundaries fall on every row.
@pytest.mark.parametrize("knobs", KNOB_SETTINGS.values(), ids=KNOB_SETTINGS.keys())
@pytest.mark.parametrize(
    "case, dtype",
    [
        (((16, 1, 5120), (1, 1, 16, 5120)), ttnn.bfloat16),
        (((1, 16, 6, 256), (1, 1, 16, 1536)), ttnn.bfloat16),
        (((1, 185), (1, 1, 185, 1)), ttnn.bfloat16),
        (((1, 185), (1, 1, 185, 1)), ttnn.float32),
    ],
    ids=["16x1x5120-bf16", "1x16x6x256-bf16", "1x185-bf16", "1x185-fp32"],
)
def test_block_knobs(case, dtype, knobs, device, monkeypatch):
    for name, value in knobs.items():
        monkeypatch.setattr(_op_module, name, value)
    src, dst = case
    torch_input = _random_input(src, dtype)
    plan = describe_plan(_to_device(torch_input, device), list(dst))
    assert plan["max_blocks_per_mover"] > 1
    assert plan["l1_bytes_per_core"] <= L1_OP_BUDGET
    _run_and_check(torch_input, list(dst), device, poison=float("nan"))
