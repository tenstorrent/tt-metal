# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_typecast, Phase 0: bfloat16 TILE -> float32 / bfloat8_b / int32,
DRAM and L1 interleaved, ranks 3-4. Every value case is a production case of
eval/golden_tests/g6_typecast/feature_spec.py; grading follows helpers.TOLERANCES."""

import pytest
import torch
import ttnn

from ttnn.operations.g6_typecast import g6_typecast, describe_plan
from eval.golden_tests.g6_typecast.helpers import TOLERANCES

_MEMORY_CONFIG = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}
_BFP8_GROUP = 16

# Exact in bfloat16; covers truncation toward zero on both signs, a fraction at 2^7 and the
# largest magnitudes below 2^31 that bfloat16 can hold.
_INT32_WITNESSES = [-2.75, -0.5, 0.5, 2.75, 127.5, -(2.0**31 - 2.0**23), 2.0**30]

# (case id, input shape, destination dtype, buffer, call form)
CASES = [
    ("Q2227", (1, 1, 12), ttnn.float32, "l1", "positional"),  # single tile, H and W partial
    ("Q2191", (16, 1, 12), ttnn.float32, "dram", "positional"),  # multi-batch
    ("Q2198", (1, 31, 12), ttnn.float32, "dram", "positional"),
    ("Q2201", (1, 33, 12), ttnn.float32, "dram", "positional"),
    ("Q2185", (16, 4096, 12), ttnn.float32, "dram", "positional"),  # 2048 tiles: several blocks per core
    ("Q2229", (1, 1, 32, 128), ttnn.int32, "dram", "sub_core_grids"),
    ("Q2229-kw", (1, 1, 32, 128), ttnn.int32, "dram", "keyword"),
    ("Q2197", (1, 1, 31, 256), ttnn.bfloat8_b, "dram", "positional"),
    ("Q2203", (1, 1, 256, 256), ttnn.bfloat8_b, "dram", "positional"),  # multi-tile, square
    ("Q2187", (1, 1, 3844, 256), ttnn.bfloat8_b, "dram", "positional"),  # non-square, partial tile rows
    ("P22", (1, 32, 64, 256), ttnn.bfloat8_b, "l1", "positional"),
    ("P39", (1, 128, 256, 256), ttnn.bfloat8_b, "l1", "positional"),  # 8192 tiles, ragged last block
]


def torch_reference(x, dtype):
    """torch.Tensor.to; bfloat8_b has no torch dtype, so its exact float32 values are returned
    and rendered in the block format by the check."""
    if dtype == ttnn.bfloat8_b:
        return x.to(torch.float32)
    return x.to({ttnn.float32: torch.float32, ttnn.int32: torch.int32}[dtype])


def make_input(shape, dtype):
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.float32)
    if dtype == ttnn.int32:
        # bfloat16 holds 8 significant bits, so N(0,1) alone never reaches the large-exponent
        # path of the float -> int32 conversion; scale each value by 2^k, k < 30.
        exponent = torch.randint(0, 30, shape, dtype=torch.int32).to(torch.float32)
        x = x.clamp(-1.9, 1.9) * torch.pow(2.0, exponent)
        flat = x.view(-1)
        k = min(len(_INT32_WITNESSES), flat.numel())
        flat[:k] = torch.tensor(_INT32_WITNESSES[:k], dtype=torch.float32)
    return x.to(torch.bfloat16)


def call_op(x, dtype, call):
    if call == "positional":
        return g6_typecast(x, dtype)
    if call == "sub_core_grids":
        return g6_typecast(x, dtype, sub_core_grids=None)
    if call == "keyword":
        return g6_typecast(x, dtype=dtype, sub_core_grids=None)
    raise ValueError(call)


def assert_output_contract(out, x, shape, dtype):
    assert list(out.shape) == list(shape), f"shape {list(out.shape)} != {list(shape)}"
    assert out.dtype == dtype, f"dtype {out.dtype} != {dtype}"
    assert out.layout == x.layout, f"layout {out.layout} != {x.layout}"
    assert out.memory_config() == x.memory_config(), f"memory_config {out.memory_config()} != {x.memory_config()}"


def check_exact(actual, expected, dtype):
    if dtype == ttnn.float32:
        a = actual.to(torch.float32).contiguous().view(torch.int32)
        e = expected.to(torch.float32).contiguous().view(torch.int32)
    else:
        a = actual.to(torch.int64)
        e = expected.to(torch.int64)
    bad = a != e
    n_bad = int(bad.sum())
    if n_bad:
        idx = torch.nonzero(bad.reshape(-1)).reshape(-1)[:5].tolist()
        got, want = actual.reshape(-1), expected.reshape(-1)
        detail = ", ".join(f"[{i}] got {got[i].item()!r} expected {want[i].item()!r}" for i in idx)
        pytest.fail(f"{n_bad}/{bad.numel()} values differ from torch.Tensor.to: {detail}")


def _groups(t):
    t = t.to(torch.float64)
    pad = (-t.shape[-1]) % _BFP8_GROUP
    if pad:
        t = torch.nn.functional.pad(t, (0, pad))
    return t.reshape(-1, _BFP8_GROUP)


def check_bfp8_group(actual, exact, steps):
    """|device - host bfloat8_b rendering| <= steps * 2^(e_max - 6) per 16-value group,
    e_max the exponent of the group's largest logical magnitude."""
    host = ttnn.to_torch(ttnn.from_torch(exact.to(torch.float32), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT))
    a, h, x = _groups(actual), _groups(host), _groups(exact)
    gmax = x.abs().amax(dim=1, keepdim=True)
    _, exp = torch.frexp(gmax)
    step = torch.where(gmax > 0, torch.pow(2.0, (exp - 7).to(torch.float64)), torch.zeros_like(gmax))
    bad = (a - h).abs() > steps * step
    n_bad = int(bad.sum())
    assert n_bad == 0, f"{n_bad}/{bad.numel()} values beyond {steps} step(s) of the host bfloat8_b rendering"


def grade(actual, expected, dtype):
    rule, param = TOLERANCES[dtype]
    if rule == "exact":
        check_exact(actual, expected, dtype)
    elif rule == "bfp8_group":
        check_bfp8_group(actual, expected, param)
    else:
        raise AssertionError(f"unknown grading rule {rule!r}")


def to_device(x, buffer, device):
    return ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_MEMORY_CONFIG[buffer]
    )


@pytest.mark.parametrize("case_id, shape, dtype, buffer, call", CASES, ids=[c[0] for c in CASES])
def test_g6_typecast(device, case_id, shape, dtype, buffer, call):
    torch_input = make_input(shape, dtype)
    expected = torch_reference(torch_input, dtype)

    ttnn_input = to_device(torch_input, buffer, device)
    ttnn_output = call_op(ttnn_input, dtype, call)

    assert_output_contract(ttnn_output, ttnn_input, shape, dtype)
    actual = ttnn.to_torch(ttnn_output)
    assert list(actual.shape) == list(shape)
    grade(actual, expected, dtype)


def test_g6_typecast_plan_is_reported(device):
    x = to_device(make_input((16, 128, 12), ttnn.float32), "dram", device)
    plan = describe_plan(x, ttnn.float32)
    assert plan["regime"] == "tile_convert"
    assert plan["num_cores"] >= 1
    assert plan["block_tiles"] >= 1
    assert len(plan["core_roles"]) == plan["num_cores"]
    assert set(plan["core_roles"].values()) == {"reader+compute+writer"}


def test_g6_typecast_repeat_is_bit_identical_and_cached(device):
    shape = (16, 4096, 12)
    x = to_device(make_input(shape, ttnn.float32), "dram", device)

    first = ttnn.to_torch(g6_typecast(x, ttnn.float32))
    entries = device.num_program_cache_entries()
    second = ttnn.to_torch(g6_typecast(x, ttnn.float32))

    assert device.num_program_cache_entries() == entries, "a repeated call built a new program"
    assert torch.equal(first.contiguous().view(torch.int32), second.contiguous().view(torch.int32))


def test_g6_typecast_rejects_non_datatype_dtype(device, expect_error):
    x = to_device(make_input((1, 1, 32, 128), ttnn.float32), "dram", device)
    with expect_error((ValueError, RuntimeError), "dtype"):
        g6_typecast(x, "float32")


def test_g6_typecast_rejects_sub_core_grids(device, expect_error):
    x = to_device(make_input((1, 1, 32, 128), ttnn.float32), "dram", device)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    with expect_error((ValueError, RuntimeError), "sub_core_grids"):
        g6_typecast(x, ttnn.float32, sub_core_grids=grid)


def test_g6_typecast_rejects_rank_5(device, expect_error):
    x = to_device(make_input((1, 1, 1, 32, 32), ttnn.float32), "dram", device)
    with expect_error((ValueError, RuntimeError), "rank"):
        g6_typecast(x, ttnn.float32)
