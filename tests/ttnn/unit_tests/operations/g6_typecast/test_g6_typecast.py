# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_typecast, Phase 0: bfloat16 TILE -> float32 / bfloat8_b / int32.

Every case is a production case of eval/golden_tests/g6_typecast/feature_spec.py (case ids
kept). The grading follows helpers.TOLERANCES of the golden suite: float32 and int32 must be
bit-identical to torch.Tensor.to; bfloat8_b must lie within one quantization step
2^(e_max - 6) of each 16-value group against the host bfloat8_b rendering.
"""

import pytest
import torch
import ttnn

from ttnn.operations._op_contract import SupportRefusal
from ttnn.operations.g6_typecast import g6_typecast

_MEMORY_CONFIG = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}
_BFP8_GROUP = 16

# (case, input shape, output dtype, buffer, call form)
CASES = [
    ("Q2227", (1, 1, 12), ttnn.float32, "l1", "positional"),  # one partial tile, L1
    ("Q2229", (1, 1, 32, 128), ttnn.int32, "dram", "sub_core_grids"),  # truncation, full 32-bit range
    ("Q2197", (1, 1, 31, 256), ttnn.bfloat8_b, "dram", "positional"),  # h_non_aligned
    ("Q2192", (16, 128, 12), ttnn.float32, "dram", "positional"),  # multi-batch rank 3, w_non_aligned
    ("Q2187", (1, 1, 3844, 256), ttnn.bfloat8_b, "dram", "positional"),  # 968 tiles: ragged split
    ("Q2185", (16, 4096, 12), ttnn.float32, "dram", "positional"),  # 2048 tiles: several blocks per core
    ("P39", (1, 128, 256, 256), ttnn.bfloat8_b, "l1", "positional"),  # 8192 tiles in L1
]


def _make_input(shape, output_dtype, seed=42):
    torch.manual_seed(seed)
    x = torch.randn(shape, dtype=torch.float32)
    if output_dtype == ttnn.int32:
        # Magnitudes 2^-2 .. 2^30 with fractional bits, so truncation is visible at every scale
        # and the result uses all 32 bits; the leading witnesses are trunc-vs-round boundaries.
        sign = torch.where(x < 0, -1.0, 1.0)
        x = sign * torch.pow(2.0, torch.rand(shape) * 32 - 2)
        witnesses = torch.tensor([-2.75, -0.5, 0.5, 2.75, 127.5, -(2.0**31 - 2.0**23), 2.0**30])
        flat = x.view(-1)
        k = min(flat.numel(), witnesses.numel())
        flat[:k] = witnesses[:k]
    return x.to(torch.bfloat16)


def _reference(torch_input, output_dtype):
    if output_dtype == ttnn.int32:
        return torch_input.to(torch.int32)
    # float32 and bfloat8_b: the exact values; bfloat8_b is rendered by the group check
    return torch_input.to(torch.float32)


def _call(ttnn_input, output_dtype, call):
    if call == "positional":
        return g6_typecast(ttnn_input, output_dtype)
    if call == "sub_core_grids":
        return g6_typecast(ttnn_input, output_dtype, sub_core_grids=None)
    if call == "keyword":
        return g6_typecast(ttnn_input, dtype=output_dtype, sub_core_grids=None)
    raise ValueError(f"unknown call form {call!r}")


def _check_contract(ttnn_output, ttnn_input, shape, output_dtype):
    assert list(ttnn_output.shape) == list(shape)
    assert ttnn_output.dtype == output_dtype
    assert ttnn_output.layout == ttnn_input.layout
    assert ttnn_output.memory_config() == ttnn_input.memory_config()


def _check_exact(actual, expected, output_dtype):
    if output_dtype == ttnn.float32:
        a = actual.to(torch.float32).contiguous().view(torch.int32)
        e = expected.to(torch.float32).contiguous().view(torch.int32)
    else:
        a = actual.to(torch.int64)
        e = expected.to(torch.int64)
    bad = a != e
    n_bad = int(bad.sum().item())
    if n_bad:
        idx = torch.nonzero(bad.reshape(-1)).reshape(-1)[:5].tolist()
        detail = ", ".join(
            f"[{i}] got {actual.reshape(-1)[i].item()!r} expected {expected.reshape(-1)[i].item()!r}" for i in idx
        )
        pytest.fail(f"{n_bad}/{bad.numel()} values differ from torch.Tensor.to: {detail}")


def _groups(t):
    # Zero-padded to whole groups: tile padding never sets a group's exponent.
    t = t.to(torch.float64)
    pad = (-t.shape[-1]) % _BFP8_GROUP
    if pad:
        t = torch.nn.functional.pad(t, (0, pad))
    return t.reshape(-1, _BFP8_GROUP)


def _check_bfp8_group(actual, exact):
    host = ttnn.to_torch(ttnn.from_torch(exact.to(torch.float32), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT))
    a, h, x = _groups(actual), _groups(host), _groups(exact)
    gmax = x.abs().amax(dim=1, keepdim=True)
    _, exp = torch.frexp(gmax)  # gmax = m * 2^exp, m in [0.5, 1) -> e_max = exp - 1
    step = torch.where(gmax > 0, torch.pow(2.0, (exp - 7).to(torch.float64)), torch.zeros_like(gmax))
    bad = (a - h).abs() > step
    n_bad = int(bad.sum().item())
    assert n_bad == 0, f"{n_bad}/{bad.numel()} values beyond one step of their group vs the host bfloat8_b rendering"


def _run_and_check(device, shape, output_dtype, buffer, call, seed=42):
    torch_input = _make_input(shape, output_dtype, seed=seed)
    expected = _reference(torch_input, output_dtype)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_MEMORY_CONFIG[buffer],
    )
    ttnn_output = _call(ttnn_input, output_dtype, call)
    _check_contract(ttnn_output, ttnn_input, shape, output_dtype)
    actual = ttnn.to_torch(ttnn_output)
    assert list(actual.shape) == list(shape)
    if output_dtype == ttnn.bfloat8_b:
        _check_bfp8_group(actual, expected)
    else:
        _check_exact(actual, expected, output_dtype)
    return actual


@pytest.mark.parametrize("case,shape,output_dtype,buffer,call", CASES, ids=[c[0] for c in CASES])
def test_g6_typecast(device, case, shape, output_dtype, buffer, call):
    _run_and_check(device, shape, output_dtype, buffer, call)


@pytest.mark.parametrize("call", ["positional", "sub_core_grids", "keyword"])
def test_call_forms(device, call):
    _run_and_check(device, (1, 1, 32, 128), ttnn.int32, "dram", call)


def test_repeat_calls(device):
    """Q2186 shape, three calls: a second input on the cached program must land in its own
    output (fresh buffer addresses), and repeating the first input must be bit-identical."""
    shape = (8, 4096, 12)
    first = _run_and_check(device, shape, ttnn.float32, "dram", "positional", seed=42)
    _run_and_check(device, shape, ttnn.float32, "dram", "positional", seed=43)
    again = _run_and_check(device, shape, ttnn.float32, "dram", "positional", seed=42)
    assert torch.equal(first.contiguous().view(torch.int32), again.contiguous().view(torch.int32))


def _assert_invalid_argument(expect_error, fn):
    # The refusal text is the implementation's; any message matches.
    with expect_error((ValueError, RuntimeError), "") as excinfo:
        fn()
    # An invalid argument is not a support cell a refinement could enable.
    assert not isinstance(excinfo.value, SupportRefusal)


def test_validation_errors(device, expect_error):
    x = ttnn.from_torch(
        torch.randn((1, 1, 32, 32)).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    _assert_invalid_argument(expect_error, lambda: g6_typecast(x, "float32"))
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    _assert_invalid_argument(expect_error, lambda: g6_typecast(x, ttnn.float32, sub_core_grids=grid))

    x5 = ttnn.from_torch(
        torch.randn((1, 1, 1, 32, 32)).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    _assert_invalid_argument(expect_error, lambda: g6_typecast(x5, ttnn.float32))
