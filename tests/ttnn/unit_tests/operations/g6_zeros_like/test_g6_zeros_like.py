# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_zeros_like.

Limited to the production cases (float32 (1, 12, 128, 128) in both call forms,
bfloat8_b (5, 1, 32, 256)) and to the cases the spec's rules require: inputs a
value path through x would break, buffer identity across program-cache hits,
the plan covering every page on more than one core, and the validation errors.
Graded exactly: float32 on bit patterns (+0.0 only), bfloat8_b on decoded values.
"""

import math

import pytest
import torch
import ttnn

from ttnn.operations.g6_zeros_like import SUPPORTED, describe_plan, g6_zeros_like
from ttnn.operations.g6_zeros_like import g6_zeros_like_program_descriptor as pd

FP32_SHAPE = (1, 12, 128, 128)
BF8_SHAPE = (5, 1, 32, 256)
SEED = 42

# -0.0, +inf, -inf, NaN with payload, smallest subnormal, largest negative subnormal, largest finite.
_SPECIAL_BITS = [0x80000000, 0x7F800000, 0xFF800000, 0x7FC00001, 0x00000001, 0x807FFFFF, 0x7F7FFFFF]


def pytorch_reference(x, optional=None):
    if optional is None:
        return torch.zeros_like(x)
    return torch.ops.aten.zeros_like.out(x, out=optional)


def _random_bits(shape):
    bits = torch.randint(-(2**31), 2**31, shape, dtype=torch.int64).to(torch.int32)
    return bits.view(torch.float32)


def _special_bits(shape):
    numel = math.prod(shape)
    signed = [v - (1 << 32) if v >= (1 << 31) else v for v in _SPECIAL_BITS]
    pattern = torch.tensor(signed, dtype=torch.int32)
    return pattern.repeat(-(-numel // pattern.numel()))[:numel].reshape(shape).view(torch.float32)


def _fp32_input(kind, shape=FP32_SHAPE):
    torch.manual_seed(SEED)
    if kind == "randn":
        return torch.randn(shape, dtype=torch.float32)
    if kind == "random_bits":
        return _random_bits(shape)
    if kind == "special_bits":
        return _special_bits(shape)
    raise ValueError(kind)


def _to_device(t, device, dtype, *, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=memory_config)


def _nan_buffer(device, shape=FP32_SHAPE):
    return _to_device(torch.full(shape, float("nan"), dtype=torch.float32), device, ttnn.float32)


def _bits(t):
    return t.contiguous().view(torch.int32)


def _assert_zeros(out, x, expected, dtype):
    """Output contract of `out` against input `x`, then the exact grade against `expected`."""
    assert list(out.shape) == list(x.shape), f"shape {list(out.shape)} != {list(x.shape)}"
    assert out.dtype == dtype, f"dtype {out.dtype} != {dtype}"
    assert out.layout == ttnn.TILE_LAYOUT, f"layout {out.layout}"
    assert out.memory_config() == x.memory_config(), f"memory config {out.memory_config()} != {x.memory_config()}"
    actual = ttnn.to_torch(out)
    assert list(actual.shape) == list(x.shape)
    if dtype == ttnn.float32:
        assert actual.dtype == torch.float32
        wrong = _bits(actual) != _bits(expected.to(torch.float32))
    else:
        wrong = actual.to(torch.float32) != expected.to(torch.float32)
    n_wrong = int(wrong.sum().item())
    if n_wrong:
        first = tuple(int(i) for i in wrong.nonzero()[0].tolist())
        pytest.fail(f"{n_wrong} of {wrong.numel()} elements are not +0.0 (first at {first}: {actual[first].item()!r})")


def _assert_unchanged(x, before):
    changed = int((_bits(ttnn.to_torch(x)) != _bits(before)).sum().item())
    assert changed == 0, f"input_tensor changed in {changed} elements"


@pytest.mark.parametrize("input_kind", ["randn", "random_bits", "special_bits"])
def test_new_output_fp32(device, input_kind):
    torch_x = _fp32_input(input_kind)
    x = _to_device(torch_x, device, ttnn.float32)
    before = ttnn.to_torch(x)

    out = g6_zeros_like(x)

    assert out.buffer_address() != x.buffer_address(), "output shares the input's buffer"
    _assert_zeros(out, x, pytorch_reference(torch_x), ttnn.float32)
    _assert_unchanged(x, before)


def test_into_optional_fp32(device):
    torch_x = _fp32_input("random_bits")
    x = _to_device(torch_x, device, ttnn.float32)
    before = ttnn.to_torch(x)
    torch_o = torch.full(FP32_SHAPE, float("nan"), dtype=torch.float32)
    o = _to_device(torch_o, device, ttnn.float32)

    out = g6_zeros_like(x, optional_tensor=o)

    assert out.buffer_address() == o.buffer_address(), "returned tensor is not optional_tensor"
    _assert_zeros(o, x, pytorch_reference(torch_x, optional=torch_o), ttnn.float32)
    _assert_unchanged(x, before)


def test_optional_is_input_fp32(device):
    torch_x = _fp32_input("special_bits")
    x = _to_device(torch_x, device, ttnn.float32)

    out = g6_zeros_like(x, optional_tensor=x)

    assert out.buffer_address() == x.buffer_address(), "returned tensor is not optional_tensor"
    _assert_zeros(x, x, pytorch_reference(torch_x), ttnn.float32)


def test_each_call_writes_its_own_buffer_fp32(device):
    """Repeat calls with the same shape and dtype hit the program cache and must
    still write the buffer each call names."""
    torch_x = _fp32_input("random_bits")
    x = _to_device(torch_x, device, ttnn.float32)
    expected = pytorch_reference(torch_x)

    for _ in range(2):
        o = _nan_buffer(device)
        out = g6_zeros_like(x, optional_tensor=o)
        assert out.buffer_address() == o.buffer_address(), "returned tensor is not optional_tensor"
        _assert_zeros(o, x, expected, ttnn.float32)

    # A freed NaN block of the output's size is where the next allocation is likely to land.
    ttnn.deallocate(_nan_buffer(device))
    first = g6_zeros_like(x)
    second = g6_zeros_like(x)
    assert first.buffer_address() != second.buffer_address(), "two live outputs share one buffer"
    _assert_zeros(first, x, expected, ttnn.float32)
    _assert_zeros(second, x, expected, ttnn.float32)


def test_bfloat8_b(device, expect_error):
    torch.manual_seed(SEED)
    torch_x = torch.randn(BF8_SHAPE, dtype=torch.float32).to(torch.bfloat16)
    x = _to_device(torch_x, device, ttnn.bfloat8_b)

    if ttnn.bfloat8_b not in SUPPORTED["dtype"]:
        # Any refusal text: the registry contract types the refusal, not its wording.
        with expect_error(NotImplementedError, ""):
            g6_zeros_like(x)
        return

    # Ones in a freed block of the output's size: a page left unwritten decodes to 1, not 0.
    ttnn.deallocate(_to_device(torch.ones(BF8_SHAPE, dtype=torch.bfloat16), device, ttnn.bfloat8_b))
    out = g6_zeros_like(x)

    assert out.buffer_address() != x.buffer_address(), "output shares the input's buffer"
    _assert_zeros(out, x, pytorch_reference(torch_x), ttnn.bfloat8_b)


@pytest.mark.parametrize(
    "shape,dtype",
    [pytest.param(FP32_SHAPE, ttnn.float32, id="float32"), pytest.param(BF8_SHAPE, ttnn.bfloat8_b, id="bfloat8_b")],
)
def test_plan_writes_every_page_once_on_many_cores(device, shape, dtype):
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    x = _to_device(torch.zeros(shape, dtype=torch_dtype), device, dtype)

    plan = describe_plan(x)

    cores = [tuple(entry["core"]) for entry in plan]
    assert len(cores) > 1, f"single-core plan: {plan}"
    assert len(set(cores)) == len(cores), "a core appears twice in the plan"
    grid = device.compute_with_storage_grid_size()
    assert all(0 <= cx < grid.x and 0 <= cy < grid.y for cx, cy in cores), f"core outside the worker grid: {cores}"
    assert all(entry["num_pages"] > 0 for entry in plan), "a planned core has no page"
    pages = sorted(
        p
        for entry in plan
        for p in range(
            entry["start_page"], entry["start_page"] + entry["num_pages"] * entry["page_stride"], entry["page_stride"]
        )
    )
    assert pages == list(range(x.buffer_num_pages())), "pages not covered exactly once"


@pytest.mark.parametrize("head_pages", [0, 1, 2, None], ids=lambda h: f"head{h}")
@pytest.mark.parametrize(
    "shape,dtype,regime,scratch_pages",
    [
        pytest.param(FP32_SHAPE, ttnn.float32, "bank_run_coalesced", 1, id="float32-bank_run"),
        pytest.param(FP32_SHAPE, ttnn.float32, "bank_run_coalesced", 2, id="float32-bank_run-scratch2"),
        pytest.param(FP32_SHAPE, ttnn.float32, "page_split_local_zero", 1, id="float32-page_split"),
        pytest.param(BF8_SHAPE, ttnn.bfloat8_b, "bank_run_coalesced", 1, id="bfloat8_b"),
    ],
)
def test_mem_zeros_head_knob(device, monkeypatch, shape, dtype, regime, scratch_pages, head_pages):
    """Every split of a core's range between the MEM_ZEROS head and the scratch writes covers every byte."""
    monkeypatch.setattr(pd, "REGIME", regime)
    monkeypatch.setattr(pd, "ZERO_SCRATCH_PAGES", scratch_pages)
    monkeypatch.setattr(pd, "MEM_ZEROS_HEAD_PAGES", head_pages)
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    x = _to_device(torch.zeros(shape, dtype=torch_dtype), device, dtype)
    # NaN (float32) / ones (bfloat8_b) in the target: any byte left unwritten fails the exact grade.
    fill = float("nan") if dtype == ttnn.float32 else 1.0
    o = _to_device(torch.full(shape, fill, dtype=torch_dtype), device, dtype)

    out = g6_zeros_like(x, optional_tensor=o)

    assert out.buffer_address() == o.buffer_address(), "returned tensor is not optional_tensor"
    _assert_zeros(o, x, torch.zeros(shape, dtype=torch_dtype), dtype)


def _fp32(device, shape=FP32_SHAPE, **kwargs):
    return _to_device(torch.zeros(shape, dtype=torch.float32), device, ttnn.float32, **kwargs)


def _fp32_host(shape=FP32_SHAPE):
    return ttnn.from_torch(torch.zeros(shape, dtype=torch.float32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)


def _height_sharded_l1():
    return ttnn.create_sharded_memory_config(
        FP32_SHAPE, core_grid=ttnn.CoreGrid(y=1, x=8), strategy=ttnn.ShardStrategy.HEIGHT
    )


_INVALID_CALLS = {
    "input_on_host": lambda d: (_fp32_host(), None),
    "input_row_major": lambda d: (_fp32(d, layout=ttnn.ROW_MAJOR_LAYOUT), None),
    "input_rank_3": lambda d: (_fp32(d, shape=(12, 128, 128)), None),
    "input_height_not_multiple_of_32": lambda d: (_fp32(d, shape=(1, 12, 120, 128)), None),
    "input_width_not_multiple_of_32": lambda d: (_fp32(d, shape=(1, 12, 128, 100)), None),
    "input_l1_interleaved": lambda d: (_fp32(d, memory_config=ttnn.L1_MEMORY_CONFIG), None),
    "input_height_sharded": lambda d: (_fp32(d, memory_config=_height_sharded_l1()), None),
    "optional_shape": lambda d: (_fp32(d), _fp32(d, shape=(1, 12, 128, 64))),
    "optional_dtype": lambda d: (_fp32(d), _to_device(torch.zeros(FP32_SHAPE), d, ttnn.bfloat16)),
    "optional_layout": lambda d: (_fp32(d), _fp32(d, layout=ttnn.ROW_MAJOR_LAYOUT)),
    "optional_l1_interleaved": lambda d: (_fp32(d), _fp32(d, memory_config=ttnn.L1_MEMORY_CONFIG)),
    "optional_on_host": lambda d: (_fp32(d), _fp32_host()),
}


@pytest.mark.parametrize("case", list(_INVALID_CALLS))
def test_validation_errors(device, expect_error, case):
    input_tensor, optional_tensor = _INVALID_CALLS[case](device)
    with expect_error((ValueError, RuntimeError), ""):
        g6_zeros_like(input_tensor, optional_tensor=optional_tensor)
