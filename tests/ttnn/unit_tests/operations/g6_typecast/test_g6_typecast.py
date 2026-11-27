# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_typecast, Phase 0: bfloat16 TILE inputs converted to float32,
bfloat8_b and int32, DRAM and L1 interleaved.

The shapes are production cases of eval/golden_tests/g6_typecast/feature_spec.py; the
remaining tests are the cases the spec's rules require (argument validation, repeat-call
determinism, program-cache keying). Grading follows helpers.TOLERANCES: bit-exact against
torch.Tensor.to, bfloat8_b within the listed number of quantization steps of each 16-value
group of the host rendering.
"""

import math

import pytest
import torch

import ttnn
from ttnn.operations.g6_typecast import g6_typecast
from eval.golden_tests.g6_typecast.helpers import TOLERANCES

F32, I32, BF8B = ttnn.float32, ttnn.int32, ttnn.bfloat8_b
BFP8_GROUP = 16
MEMORY_CONFIG = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}

# (production case, shape, destination, buffer, call form)
PHASE0_CASES = [
    ("Q2227", (1, 1, 12), F32, "l1", "positional"),  # one partial tile, L1
    ("Q2191", (16, 1, 12), F32, "dram", "positional"),  # 16 images, one partial tile each
    ("Q2201", (1, 33, 12), F32, "dram", "positional"),  # H and W both non-aligned
    ("Q2185", (16, 4096, 12), F32, "dram", "positional"),  # 2048 tiles
    ("Q2197", (1, 1, 31, 256), BF8B, "dram", "positional"),  # one partial tile row
    ("Q2187", (1, 1, 3844, 256), BF8B, "dram", "positional"),  # 968 tiles, non-square
    ("P22", (1, 32, 64, 256), BF8B, "l1", "positional"),  # multi-batch, L1
    ("P39", (1, 128, 256, 256), BF8B, "l1", "positional"),  # 8192 tiles, L1
    ("Q2229", (1, 1, 32, 128), I32, "dram", "sub_core_grids"),
    ("Q2229", (1, 1, 32, 128), I32, "dram", "keyword"),
]


def _make_input(shape, output_dtype):
    torch.manual_seed(42)
    if output_dtype == I32:
        # Magnitudes up to 2^30 with fractional parts: a conversion that is right only for
        # small integers, or that rounds instead of truncating, fails here.
        n = math.prod(shape)
        z = torch.randn(n)
        exponent = torch.randint(-2, 30, (n,)).to(torch.float32)
        x = torch.sign(z) * (1.0 + z.abs().frac()) * torch.pow(2.0, exponent)
        boundary = torch.tensor([-2.75, -0.5, 0.5, 2.75, 127.5, -(2.0**31 - 2.0**23), 2.0**30, -1.0])
        k = min(n, boundary.numel())
        x[:k] = boundary[:k]
        return x.to(torch.bfloat16).reshape(shape)
    return torch.randn(shape).to(torch.bfloat16)


def _to_device(torch_input, buffer, device):
    return ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=MEMORY_CONFIG[buffer],
    )


def _call(x, output_dtype, form):
    if form == "positional":
        return g6_typecast(x, output_dtype)
    if form == "sub_core_grids":
        return g6_typecast(x, output_dtype, sub_core_grids=None)
    return g6_typecast(x, dtype=output_dtype, sub_core_grids=None)


def _assert_contract(out, x, shape, output_dtype):
    assert list(out.shape) == list(shape)
    assert out.dtype == output_dtype
    assert out.layout == x.layout
    assert out.memory_config() == x.memory_config()


def _assert_no_mismatch(bad, what):
    n_bad = int(bad.sum().item())
    assert n_bad == 0, f"{n_bad}/{bad.numel()} values {what}"


def _groups(t):
    t = t.to(torch.float64)
    pad = (-t.shape[-1]) % BFP8_GROUP
    if pad:
        t = torch.nn.functional.pad(t, (0, pad))
    return t.reshape(-1, BFP8_GROUP)


def _assert_values(actual, torch_input, output_dtype):
    rule, steps = TOLERANCES[output_dtype]
    if rule == "exact":
        if output_dtype == F32:
            got = actual.to(torch.float32).contiguous().view(torch.int32)
            want = torch_input.to(torch.float32).contiguous().view(torch.int32)
        else:
            got = actual.to(torch.int64)
            want = torch_input.to(torch.int32).to(torch.int64)
        _assert_no_mismatch(got != want, "differ from torch.Tensor.to")
        return

    assert rule == "bfp8_group", f"unexpected grading rule {rule!r}"
    exact = torch_input.to(torch.float32)
    host = ttnn.to_torch(ttnn.from_torch(exact, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT))
    a, h, x = _groups(actual), _groups(host), _groups(exact)
    group_max = x.abs().amax(dim=1, keepdim=True)
    _, exponent = torch.frexp(group_max)
    # One quantization step is 2^(e_max - 6); frexp's exponent is e_max + 1.
    step = torch.where(group_max > 0, torch.pow(2.0, (exponent - 7).to(torch.float64)), torch.zeros_like(group_max))
    _assert_no_mismatch((a - h).abs() > steps * step, f"beyond {steps} step(s) of the host bfloat8_b rendering")


@pytest.mark.parametrize(
    "case,shape,output_dtype,buffer,form",
    PHASE0_CASES,
    ids=[f"{c[0]}-{c[4]}" for c in PHASE0_CASES],
)
def test_g6_typecast(device, case, shape, output_dtype, buffer, form):
    torch_input = _make_input(shape, output_dtype)
    x = _to_device(torch_input, buffer, device)

    out = _call(x, output_dtype, form)

    _assert_contract(out, x, shape, output_dtype)
    actual = ttnn.to_torch(out)
    assert list(actual.shape) == list(shape)
    _assert_values(actual, torch_input, output_dtype)


VALIDATION_SHAPE = (1, 1, 32, 128)


@pytest.mark.parametrize("bad_dtype", [torch.float32, "float32", None], ids=["torch_dtype", "string", "none"])
def test_rejects_dtype_that_is_not_a_ttnn_datatype(device, expect_error, bad_dtype):
    x = _to_device(_make_input(VALIDATION_SHAPE, F32), "dram", device)
    with expect_error((ValueError, RuntimeError), "dtype"):
        g6_typecast(x, bad_dtype)


def test_rejects_sub_core_grids_other_than_none(device, expect_error):
    x = _to_device(_make_input(VALIDATION_SHAPE, I32), "dram", device)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    with expect_error((ValueError, RuntimeError), "sub_core_grids"):
        g6_typecast(x, I32, sub_core_grids=grid)


def test_rejects_rank_above_four(device, expect_error):
    torch.manual_seed(42)
    torch_input = torch.randn(1, 1, 1, 32, 128).to(torch.bfloat16)
    x = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error((ValueError, RuntimeError), "(?i)rank"):
        g6_typecast(x, F32)


def test_repeat_call_is_bit_identical(device):
    shape = (1, 1, 3844, 256)  # Q2187
    x = _to_device(_make_input(shape, BF8B), "dram", device)

    first = ttnn.to_torch(g6_typecast(x, BF8B))
    second = ttnn.to_torch(g6_typecast(x, BF8B))

    assert torch.equal(first, second)


def test_program_cache_keys_on_destination(device):
    shape = (1, 1, 32, 128)  # Q2229
    torch_input = _make_input(shape, I32)
    x = _to_device(torch_input, "dram", device)

    g6_typecast(x, I32)
    entries = device.num_program_cache_entries()
    out_int = g6_typecast(x, I32)
    assert device.num_program_cache_entries() == entries, "a repeated call must not build a new program"
    _assert_values(ttnn.to_torch(out_int), torch_input, I32)

    # Same shape, different destination: a key without the output dtype would replay the int32 program.
    out_float = g6_typecast(x, F32)
    _assert_values(ttnn.to_torch(out_float), torch_input, F32)
