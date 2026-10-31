# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ttnn.SfpuMode: one keyword selects one of four SFPU instruction sequences.

| Op         | Approximate vs Precise                          | Strict adds                                      | Legacy call  |
|------------|-------------------------------------------------|--------------------------------------------------|--------------|
| softsign   | reciprocal of 1 + |x|: BH none / WH 1 vs 2 Newton | +-1 for finite |x| >= 2**126, -0 -> -0          | PreciseRelaxed |
| exp2       | fp32: degree-4 vs degree-6 polynomial (bf16 alias) | NaN/overflow/underflow guards (bf16: the clamp)  | PreciseStrict  |
| reciprocal | BH 7-bit SFPARECIP vs Newton; WH 1 vs 2 Newton   | NaN -> NaN (Relaxed returns 0)                   | PreciseRelaxed |
"""

import pytest
import torch
import ttnn

ALL_MODES = [
    ttnn.SfpuMode.PreciseStrict,
    ttnn.SfpuMode.ApproximateStrict,
    ttnn.SfpuMode.PreciseRelaxed,
    ttnn.SfpuMode.ApproximateRelaxed,
]
STRICT_MODES = [ttnn.SfpuMode.PreciseStrict, ttnn.SfpuMode.ApproximateStrict]
APPROXIMATE_MODES = {ttnn.SfpuMode.ApproximateStrict, ttnn.SfpuMode.ApproximateRelaxed}

TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}

# op name -> (ttnn function, torch reference, UnaryOpType, the mode the mode-less call runs)
OPS = {
    "softsign": (ttnn.softsign, torch.nn.functional.softsign, ttnn.UnaryOpType.SOFTSIGN, ttnn.SfpuMode.PreciseRelaxed),
    "exp2": (ttnn.exp2, torch.exp2, ttnn.UnaryOpType.EXP2, ttnn.SfpuMode.PreciseStrict),
    "reciprocal": (ttnn.reciprocal, torch.reciprocal, ttnn.UnaryOpType.RECIP, ttnn.SfpuMode.PreciseRelaxed),
}

# Max relative error on each op's interior domain, per (dtype, approximate). bf16 is dominated by output rounding.
# Approximate: softsign/reciprocal on Blackhole use the 7-bit SFPARECIP seed alone; exp2 fp32 a degree-4 polynomial.
REL_TOL = {
    "softsign": {
        (ttnn.bfloat16, False): 1e-2,
        (ttnn.bfloat16, True): 3e-2,
        (ttnn.float32, False): 1e-5,
        (ttnn.float32, True): 3e-2,
    },
    "exp2": {
        (ttnn.bfloat16, False): 1e-2,
        (ttnn.bfloat16, True): 1e-2,
        (ttnn.float32, False): 2e-6,
        (ttnn.float32, True): 1e-5,
    },
    "reciprocal": {
        (ttnn.bfloat16, False): 1e-2,
        (ttnn.bfloat16, True): 3e-2,
        (ttnn.float32, False): 2e-6,
        (ttnn.float32, True): 3e-2,
    },
}

FLUSH_BAND = [2.0**126, 1.5 * 2.0**126, 2.0**127, 3.0e38]


def _to_device(torch_tensor, dtype, device):
    return ttnn.from_torch(torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _interior_input(op, dtype, shape=(1, 1, 64, 64)):
    """Values where all four modes are defined: softsign [-5, 5], exp2 [-20, 20], reciprocal +-[0.1, 10]."""
    generator = torch.Generator().manual_seed(0)
    u = torch.rand(shape, generator=generator)
    if op == "softsign":
        x = u * 10.0 - 5.0
    elif op == "exp2":
        x = u * 40.0 - 20.0
    else:
        sign = torch.where(torch.rand(shape, generator=generator) < 0.5, -1.0, 1.0)
        x = sign * (0.1 + u * 9.9)
    return x.to(TORCH_DTYPE[dtype])


def _tile_of(values, dtype):
    flat = torch.tensor(values * (1024 // len(values)) + values[: 1024 % len(values)], dtype=torch.float32)
    return flat.reshape(1, 1, 32, 32).to(TORCH_DTYPE[dtype])


def _bits(tensor):
    return tensor.contiguous().view(torch.int32 if tensor.dtype == torch.float32 else torch.int16)


def _max_rel_error(result, golden):
    return ((result.float() - golden).abs() / golden.abs().clamp_min(1e-30)).max().item()


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("sfpu_mode", ALL_MODES)
@pytest.mark.parametrize("op", list(OPS))
def test_mode_interior_accuracy(device, op, sfpu_mode, dtype):
    """Every mode meets its relative-error bound on the op's interior domain, where all four are defined."""
    ttnn_fn, torch_fn, _, _ = OPS[op]
    x = _interior_input(op, dtype)
    result = ttnn.to_torch(ttnn_fn(_to_device(x, dtype, device), sfpu_mode=sfpu_mode))
    golden = torch_fn(x.double()).float()
    rel = _max_rel_error(result, golden)
    bound = REL_TOL[op][(dtype, sfpu_mode in APPROXIMATE_MODES)]
    assert rel <= bound, f"{op} {sfpu_mode} {dtype}: max relative error {rel:.3g} exceeds {bound:.3g}"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("sfpu_mode", STRICT_MODES)
def test_softsign_strict_flush_band(device, dtype, sfpu_mode):
    """Strict softsign returns exactly +-1.0 for finite |x| >= 2**126 (the legacy sequence returns +-0)."""
    x = _tile_of(
        FLUSH_BAND + [-v for v in FLUSH_BAND] + [2.0**125, -(2.0**125), 1.0e30, 1.0e6, 3.0, -3.0, 1.0, 0.0], dtype
    )
    result = ttnn.to_torch(ttnn.softsign(_to_device(x, dtype, device), sfpu_mode=sfpu_mode))
    xf = x.float()
    in_band = torch.isfinite(xf) & (xf.abs() >= 2.0**126)
    expected = torch.sign(xf[in_band]).to(result.dtype)
    assert torch.equal(result[in_band], expected), f"flush-band lanes not exactly +-1.0: {result[in_band][:8].tolist()}"
    golden = torch.nn.functional.softsign(xf[~in_band].double()).float()
    assert _max_rel_error(result[~in_band], golden) <= REL_TOL["softsign"][(dtype, sfpu_mode in APPROXIMATE_MODES)]


# fp32 input -> expected output (NaN means any NaN). Each table lists what Relaxed may get wrong.
STRICT_SPECIALS = {
    "softsign": [
        (0.0, 0.0),
        (-0.0, -0.0),
        (float("nan"), float("nan")),
        (float("inf"), float("nan")),
        (float("-inf"), float("nan")),
        (2.0**127, 1.0),
    ],
    "exp2": [
        (float("nan"), float("nan")),
        (float("inf"), float("inf")),
        (float("-inf"), 0.0),
        (200.0, float("inf")),
        (128.0, float("inf")),
        (-200.0, 0.0),
        (0.0, 1.0),
        (-0.0, 1.0),
    ],
    "reciprocal": [
        (float("nan"), float("nan")),
        (0.0, float("inf")),
        (-0.0, float("-inf")),
        (float("inf"), 0.0),
        (float("-inf"), -0.0),
    ],
}


@pytest.mark.parametrize("sfpu_mode", STRICT_MODES)
@pytest.mark.parametrize("op", list(OPS))
def test_mode_strict_specials_fp32(device, op, sfpu_mode):
    """Strict modes meet the op's special-value contract bit for bit (fp32 reaches the SFPU unaltered)."""
    ttnn_fn, _, _, _ = OPS[op]
    table = STRICT_SPECIALS[op]
    x = _tile_of([v for v, _ in table], ttnn.float32)
    expected = _tile_of([e for _, e in table], ttnn.float32)
    result = ttnn.to_torch(ttnn_fn(_to_device(x, ttnn.float32, device), sfpu_mode=sfpu_mode))
    nan = torch.isnan(expected)
    assert torch.isnan(result[nan]).all(), f"{op}: expected NaN, got {result[nan][:4].tolist()}"
    assert torch.equal(
        _bits(result[~nan]), _bits(expected[~nan])
    ), f"{op}: got {result[~nan][: len(table)].tolist()}, want {expected[~nan][: len(table)].tolist()}"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("op", list(OPS))
def test_legacy_call_is_unchanged_mode(device, op, dtype):
    """The mode-less call is bit-identical to the mode it has always run (see OPS)."""
    ttnn_fn, _, _, legacy_mode = OPS[op]
    x = torch.cat([_interior_input(op, dtype, (1, 1, 32, 32)), _tile_of(FLUSH_BAND + [0.0, -1.0, 7.5], dtype)], dim=-1)
    tt_x = _to_device(x, dtype, device)
    legacy = ttnn.to_torch(ttnn_fn(tt_x))
    moded = ttnn.to_torch(ttnn_fn(tt_x, sfpu_mode=legacy_mode))
    assert torch.equal(_bits(legacy), _bits(moded))


@pytest.mark.parametrize("sfpu_mode", ALL_MODES)
@pytest.mark.parametrize("op", list(OPS))
def test_mode_through_unary_chain(device, op, sfpu_mode):
    """UnaryWithParam(OP, mode) in an op chain runs the same sequence as ttnn.<op>(sfpu_mode=mode)."""
    ttnn_fn, _, op_type, _ = OPS[op]
    x = _interior_input(op, ttnn.float32, (1, 1, 32, 64))
    tt_x = _to_device(x, ttnn.float32, device)
    chained = ttnn.to_torch(ttnn.unary_chain(tt_x, [ttnn.UnaryWithParam(op_type, sfpu_mode)]))
    direct = ttnn.to_torch(ttnn_fn(tt_x, sfpu_mode=sfpu_mode))
    assert torch.equal(_bits(chained), _bits(direct))


@pytest.mark.parametrize("op", list(OPS))
def test_modes_are_distinct_programs(device, op):
    """Each mode compiles its own program; repeating a mode reuses its cache entry."""
    ttnn_fn, _, _, _ = OPS[op]
    tt_x = _to_device(_interior_input(op, ttnn.bfloat16, (1, 1, 32, 32)), ttnn.bfloat16, device)
    start = device.num_program_cache_entries()
    for mode in ALL_MODES:
        ttnn_fn(tt_x, sfpu_mode=mode)
    assert device.num_program_cache_entries() - start == len(ALL_MODES)
    for mode in ALL_MODES:
        ttnn_fn(tt_x, sfpu_mode=mode)
    assert device.num_program_cache_entries() - start == len(ALL_MODES)


@pytest.mark.parametrize("param", [False, True, 0, 1, 2.5])
def test_unary_with_param_numeric_ctor_unchanged(param):
    """A bool/int/float parameter still builds a numeric UnaryWithParam, never an SfpuMode one.

    nanobind converts bools and ints to enums, so without noconvert the SfpuMode overload
    would capture UnaryWithParam(EXP, False) and refuse it.
    """
    p = ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, param)
    assert p.op_type == ttnn.UnaryOpType.EXP
    assert list(p.params) == [float(param)]


def test_unary_with_param_rejects_mode_for_other_ops(expect_error):
    """An SfpuMode on an op without a mode-typed kernel API is refused when the param is built."""
    with expect_error(RuntimeError, "does not take an SfpuMode"):
        ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, ttnn.SfpuMode.PreciseStrict)
