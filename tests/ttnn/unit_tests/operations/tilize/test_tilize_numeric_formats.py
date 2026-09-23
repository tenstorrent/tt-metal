# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Numeric formats for tilize (Refinement 7): every legal (input dtype, output dtype) pair.

tilize does no arithmetic. The value-preserving cast happens at pack (cb_output_tiles is the
output dtype), so a pair whose output format represents the input exactly must come back
bit-identical; the lossy pairs (fp32 -> bf16, anything -> bfloat8_b / bfloat4_b) are held to
PCC floors. Metrics (PCC, allclose deltas, ULP stats, median / p99 abs error, relative RMS) are
printed for every case.

test_tilize_precision_matrix is the precision characterization (numeric-formats skill section
10). math_fidelity / math_approx_mode are no-ops for tilize (there is no FPU arithmetic), so the
matrix pins HiFi4 and test_tilize_fidelity_is_noop checks every fidelity once; fp32_dest_acc_en
is crossed (the op forces it on wherever a 32-bit / uint8 page needs it).
"""
import os

import pytest
import torch
import ttnn

from models.common.utility_functions import calculate_detailed_ulp_stats, comp_allclose, comp_pcc
from ttnn.operations.tilize import tilize

_KERNEL_CONFIG = getattr(ttnn, "ComputeKernelConfig", None) or ttnn.WormholeComputeKernelConfig

_FLOATS_IN = [ttnn.bfloat16, ttnn.float32]
_FLOATS_OUT = [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b, ttnn.bfloat4_b]
PAIRS = [(i, o) for i in _FLOATS_IN for o in _FLOATS_OUT] + [
    (ttnn.fp8_e4m3, ttnn.float32),
    (ttnn.fp8_e4m3, ttnn.bfloat16),
    (ttnn.fp8_e4m3, ttnn.bfloat8_b),
    (ttnn.fp8_e4m3, ttnn.bfloat4_b),
    (ttnn.uint32, ttnn.uint32),
    (ttnn.uint32, ttnn.int32),
    (ttnn.int32, ttnn.int32),
    (ttnn.int32, ttnn.uint32),
    (ttnn.uint16, ttnn.uint16),
    (ttnn.uint8, ttnn.uint8),
]
_NAME = {
    ttnn.bfloat16: "bf16",
    ttnn.float32: "fp32",
    ttnn.fp8_e4m3: "fp8",
    ttnn.bfloat8_b: "bfp8",
    ttnn.bfloat4_b: "bfp4",
    ttnn.uint32: "u32",
    ttnn.int32: "i32",
    ttnn.uint16: "u16",
    ttnn.uint8: "u8",
}
_PAIR_PARAMS = [pytest.param(i, o, id=f"{_NAME[i]}_to_{_NAME[o]}") for i, o in PAIRS]
_INTEGERS = (ttnn.uint32, ttnn.int32, ttnn.uint16, ttnn.uint8)
# torch dtype of the readback / expected tensor (block-float and fp8 have no torch form).
_TORCH = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
    ttnn.bfloat8_b: torch.bfloat16,
    ttnn.bfloat4_b: torch.bfloat16,
    ttnn.uint32: torch.int32,
    ttnn.int32: torch.int32,
    ttnn.uint16: torch.int32,
    ttnn.uint8: torch.uint8,
}


def _readback(y):
    """to_torch, with torch's unsigned dtypes mapped to the signed view the expected tensor uses."""
    got = ttnn.to_torch(y)
    if got.dtype == torch.uint32:
        return got.view(torch.int32)
    if got.dtype == torch.uint16:
        return got.to(torch.int32)
    return got


def _tolerance(i, o):
    """("exact", None) or ("pcc", floor) — the golden suite's floors."""
    if i == o:
        return ("exact", None)
    if i == ttnn.fp8_e4m3:
        return ("pcc", {ttnn.float32: 0.9999, ttnn.bfloat16: 0.9999, ttnn.bfloat8_b: 0.999, ttnn.bfloat4_b: 0.98}[o])
    if o == ttnn.bfloat4_b:
        return ("pcc", 0.98)
    if o == ttnn.bfloat8_b:
        return ("pcc", 0.99)
    if i == ttnn.float32 and o == ttnn.bfloat16:
        return ("pcc", 0.999)
    return ("exact", None)  # bf16 -> fp32, same-width integer bit_casts


def _skip_if_unconstructible(dtype, device):
    if dtype == ttnn.fp8_e4m3 and device.arch() != ttnn.device.Arch.BLACKHOLE:
        pytest.skip("fp8_e4m3 tensors are constructible only on Blackhole")


def _input(dtype, shape, dist, seed=0):
    g = torch.Generator().manual_seed(seed)
    if dtype == ttnn.uint8:
        return torch.randint(0, 256, shape, generator=g, dtype=torch.int32).to(torch.uint8)
    if dtype == ttnn.uint16:
        return torch.randint(0, 65536, shape, generator=g, dtype=torch.int32)
    if dtype == ttnn.uint32:
        # uint32 reads back through int32: the full 32-bit pattern space, signed view.
        return torch.randint(-(2**31), 2**31 - 1, shape, generator=g, dtype=torch.int32)
    if dtype == ttnn.int32:
        return torch.randint(-(2**31), 2**31 - 1, shape, generator=g, dtype=torch.int32)
    x = torch.rand(shape, generator=g) if dist == "rand" else torch.randn(shape, generator=g)
    if dtype == ttnn.fp8_e4m3:
        return x.to(torch.float8_e4m3fn).to(torch.float32)
    return x.to(torch.float32 if dtype == ttnn.float32 else torch.bfloat16)


def _metrics(expected, got):
    # comp_pcc / comp_allclose may modify their arguments: every metric gets its own copy.
    e, a = expected.double(), got.double()
    abs_err = (a - e).abs()
    out = {
        "median_abs": abs_err.median().item(),
        "p99_abs": torch.quantile(abs_err.flatten().float(), 0.99).item(),
        "rel_rms": (abs_err.pow(2).mean().sqrt() / e.pow(2).mean().sqrt().clamp(min=1e-10)).item(),
        "pcc": comp_pcc(e.clone(), a.clone(), 0.0)[1],
        "allclose": comp_allclose(e.clone(), a.clone())[1],
    }
    if expected.dtype in (torch.bfloat16, torch.float32) and got.dtype == expected.dtype:
        out["ulp"] = calculate_detailed_ulp_stats(expected.clone(), got.clone())
    return out


def _run(device, shape, in_dtype, out_dtype, *, dist="randn", compute_kernel_config=None, seed=0):
    x = _input(in_dtype, shape, dist, seed)
    aligned = shape[-2] % 32 == 0 and shape[-1] % 32 == 0
    t = ttnn.from_torch(
        x, dtype=in_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    kwargs = {} if aligned else {"pad_value": 0}
    y = tilize(t, ttnn.DRAM_MEMORY_CONFIG, dtype=out_dtype, compute_kernel_config=compute_kernel_config, **kwargs)
    assert y.layout == ttnn.TILE_LAYOUT and y.dtype == out_dtype
    got = _readback(y)
    expected = x.to(_TORCH[out_dtype])
    assert list(got.shape) == list(shape)
    return expected, got


def _check(expected, got, in_dtype, out_dtype, label):
    m = _metrics(expected, got)
    print(f"\n[{label}] {m}")
    mode, floor = _tolerance(in_dtype, out_dtype)
    if mode == "exact":
        assert torch.equal(got, expected), f"{label}: not bit-identical; {m}"
    else:
        ok, msg = comp_pcc(expected, got, floor)
        assert ok, f"{label}: {msg}"


SHAPES = [
    pytest.param((32, 32), id="32x32_small"),
    pytest.param((32, 64), id="32x64"),
    pytest.param((64, 128), id="64x128"),
    pytest.param((128, 512), id="128x512"),
    pytest.param((256, 2048), id="256x2048_large"),
    pytest.param((32, 48), id="32x48_W_non_aligned"),
    pytest.param((48, 64), id="48x64_H_non_aligned"),
    pytest.param((48, 80), id="48x80_both_non_aligned"),
]


@pytest.mark.parametrize("distribution", ["rand", "randn"])
@pytest.mark.parametrize("fp32_acc", [pytest.param(True, id="fp32_acc"), pytest.param(False, id="bf16_acc")])
@pytest.mark.parametrize("in_dtype, out_dtype", _PAIR_PARAMS)
@pytest.mark.parametrize("shape", SHAPES)
def test_tilize_precision_matrix(device, shape, in_dtype, out_dtype, fp32_acc, distribution):
    _skip_if_unconstructible(in_dtype, device)
    if in_dtype in _INTEGERS and distribution == "rand":
        pytest.skip("integer inputs are uniform over their bit patterns; one distribution covers them")
    config = _KERNEL_CONFIG(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32_acc)
    expected, got = _run(device, shape, in_dtype, out_dtype, dist=distribution, compute_kernel_config=config)
    _check(expected, got, in_dtype, out_dtype, f"{shape} {_NAME[in_dtype]}->{_NAME[out_dtype]} acc={fp32_acc}")


@pytest.mark.parametrize(
    "fidelity", [ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi3, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.LoFi]
)
@pytest.mark.parametrize(
    "in_dtype, out_dtype",
    [
        pytest.param(ttnn.bfloat16, ttnn.bfloat16, id="bf16_to_bf16"),
        pytest.param(ttnn.float32, ttnn.float32, id="fp32_to_fp32"),
        pytest.param(ttnn.float32, ttnn.bfloat8_b, id="fp32_to_bfp8"),
    ],
)
def test_tilize_fidelity_is_noop(device, in_dtype, out_dtype, fidelity):
    """tilize does no arithmetic: every math_fidelity gives the default call's bits."""
    shape = (64, 128)
    _, reference = _run(device, shape, in_dtype, out_dtype)
    config = _KERNEL_CONFIG(math_fidelity=fidelity, math_approx_mode=True)
    _, got = _run(device, shape, in_dtype, out_dtype, compute_kernel_config=config)
    assert torch.equal(got, reference)


@pytest.mark.parametrize(
    "dtype, values",
    [
        pytest.param(ttnn.int32, [-(2**31), -1, 0, 1, 2**31 - 1], id="i32_extremes"),
        pytest.param(ttnn.uint32, [-(2**31), -1, 0, 1, 2**31 - 1], id="u32_all_bit_patterns"),
        pytest.param(ttnn.uint16, [0, 1, 32767, 32768, 65535], id="u16_extremes"),
        pytest.param(ttnn.uint8, [0, 1, 127, 128, 255], id="u8_extremes"),
    ],
)
def test_tilize_integer_extremes(device, dtype, values):
    """Integer patterns that a float datapath (tf32 SrcA, 16-bit DEST) would corrupt."""
    shape = (1, 1, 64, 96)
    n = shape[-2] * shape[-1]
    v = torch.tensor(values, dtype=torch.int64).repeat(n // len(values) + 1)[:n].reshape(shape)
    x = v.to(torch.uint8) if dtype == ttnn.uint8 else v.to(torch.int32)
    t = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    got = _readback(tilize(t, dtype=dtype))
    assert torch.equal(got, x)


@pytest.mark.parametrize(
    "in_dtype, out_dtype, pad_value",
    [
        pytest.param(ttnn.int32, ttnn.int32, -7, id="i32_negative_fill"),
        pytest.param(ttnn.uint32, ttnn.uint32, -7, id="u32_negative_fill_bitcast"),
        pytest.param(ttnn.uint8, ttnn.uint8, 200, id="u8_fill"),
        pytest.param(ttnn.uint16, ttnn.uint16, 60000, id="u16_fill"),
        pytest.param(ttnn.float32, ttnn.bfloat16, -2.5, id="fp32_to_bf16_fill"),
        pytest.param(ttnn.bfloat16, ttnn.float32, 3.0, id="bf16_to_fp32_fill"),
    ],
)
def test_tilize_numeric_pad_fill(device, in_dtype, out_dtype, pad_value):
    """The pad fill is encoded in the input dtype (signed -> unsigned bit_cast for integers)."""
    shape = (1, 1, 50, 70)
    x = _input(in_dtype, shape, "randn", seed=3)
    t = ttnn.from_torch(x, dtype=in_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    y = tilize(t, dtype=out_dtype, pad_value=pad_value)
    padded = y.cpu().to_torch_with_padded_shape()
    if padded.dtype == torch.uint32:
        padded = padded.view(torch.int32)
    elif padded.dtype == torch.uint16:
        padded = padded.to(torch.int32)
    expected = torch.nn.functional.pad(x.to(_TORCH[out_dtype]), (0, 96 - 70, 0, 64 - 50), value=pad_value)
    assert torch.equal(padded, expected)


@pytest.mark.parametrize(
    "in_dtype, out_dtype, shape",
    [
        pytest.param(ttnn.bfloat16, ttnn.bfloat16, (1, 1, 16384, 64), id="bf16_perf_focus"),
        pytest.param(ttnn.float32, ttnn.float32, (1, 1, 8192, 32), id="fp32_loose_ref"),
        pytest.param(ttnn.float32, ttnn.float32, (1, 1, 16384, 64), id="fp32_16384x64"),
        pytest.param(ttnn.bfloat16, ttnn.bfloat8_b, (1, 1, 16384, 64), id="bf16_to_bfp8_16384x64"),
        pytest.param(ttnn.int32, ttnn.int32, (1, 1, 16384, 64), id="i32_16384x64"),
        pytest.param(ttnn.uint8, ttnn.uint8, (1, 1, 16384, 64), id="u8_16384x64"),
    ],
)
def test_tilize_numeric_perf_shape(device, in_dtype, out_dtype, shape):
    """Profile with run_safe_pytest.sh --profile; correctness still asserted."""
    expected, got = _run(device, shape, in_dtype, out_dtype)
    if os.environ.get("TILIZE_ABLATION") != "1":
        _check(expected, got, in_dtype, out_dtype, f"perf {shape}")
