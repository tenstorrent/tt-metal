# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ttnn.SfpuMode on softsign: one keyword selects one of four SFPU instruction sequences.

| Mode               | Reciprocal of 1 + |x|              | Special-value guard |
|--------------------|------------------------------------|---------------------|
| PreciseStrict      | 2 Newton steps                     | yes                 |
| ApproximateStrict  | Blackhole: none, Wormhole: 1 step  | yes                 |
| PreciseRelaxed     | 2 Newton steps                     | no                  |
| ApproximateRelaxed | Blackhole: none, Wormhole: 1 step  | no                  |

The guard covers the band the reciprocal flushes: for finite |x| >= 2**126 the unguarded
sequence returns +-0, where softsign (and torch) give +-1.
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

# Max relative error on the interior domain, per (dtype, approximate). bf16 is dominated by the
# output rounding; fp32 precise by two Newton steps; approximate by the hardware reciprocal seed.
REL_TOL = {
    (ttnn.bfloat16, False): 1e-2,
    (ttnn.bfloat16, True): 3e-2,
    (ttnn.float32, False): 1e-5,
    (ttnn.float32, True): 3e-2,
}
APPROXIMATE_MODES = {ttnn.SfpuMode.ApproximateStrict, ttnn.SfpuMode.ApproximateRelaxed}

TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}

FLUSH_BAND = [2.0**126, 1.5 * 2.0**126, 2.0**127, 3.0e38]


def _to_device(torch_tensor, dtype, device):
    return ttnn.from_torch(torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _interior_input(dtype, shape=(1, 1, 64, 64)):
    generator = torch.Generator().manual_seed(0)
    return (torch.rand(shape, generator=generator) * 10.0 - 5.0).to(TORCH_DTYPE[dtype])


def _band_input(dtype):
    """One tile: the flush band (both signs) and ordinary values around it."""
    values = FLUSH_BAND + [-x for x in FLUSH_BAND] + [2.0**125, -(2.0**125), 1.0e30, 1.0e6, 3.0, -3.0, 1.0, 0.0]
    flat = torch.tensor(values * (1024 // len(values)) + values[: 1024 % len(values)], dtype=torch.float32)
    return flat.reshape(1, 1, 32, 32).to(TORCH_DTYPE[dtype])


def _bits(tensor):
    return tensor.contiguous().view(torch.int32 if tensor.dtype == torch.float32 else torch.int16)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("sfpu_mode", ALL_MODES)
def test_softsign_mode_interior_accuracy(device, dtype, sfpu_mode):
    """Every mode meets its relative-error bound on [-5, 5], where all four are defined."""
    x = _interior_input(dtype)
    result = ttnn.to_torch(ttnn.softsign(_to_device(x, dtype, device), sfpu_mode=sfpu_mode)).float()
    golden = torch.nn.functional.softsign(x.double()).float()
    rel = ((result - golden).abs() / golden.abs().clamp_min(1e-30)).max().item()
    bound = REL_TOL[(dtype, sfpu_mode in APPROXIMATE_MODES)]
    assert rel <= bound, f"{sfpu_mode} {dtype}: max relative error {rel:.3g} exceeds {bound:.3g}"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("sfpu_mode", STRICT_MODES)
def test_softsign_strict_flush_band(device, dtype, sfpu_mode):
    """Strict modes return exactly +-1.0 for finite |x| >= 2**126 (the legacy sequence returns +-0)."""
    x = _band_input(dtype)
    result = ttnn.to_torch(ttnn.softsign(_to_device(x, dtype, device), sfpu_mode=sfpu_mode))
    xf = x.float()
    in_band = torch.isfinite(xf) & (xf.abs() >= 2.0**126)
    expected = torch.sign(xf[in_band]).to(result.dtype)
    assert torch.equal(result[in_band], expected), (
        f"{sfpu_mode} {dtype}: flush-band lanes not exactly +-1.0: " f"{result[in_band][:8].tolist()}"
    )
    # Outside the band, Strict still computes softsign.
    out_of_band = ~in_band
    golden = torch.nn.functional.softsign(xf[out_of_band].double()).float()
    rel = ((result[out_of_band].float() - golden).abs() / golden.abs().clamp_min(1e-30)).max().item()
    assert rel <= REL_TOL[(dtype, sfpu_mode in APPROXIMATE_MODES)]


@pytest.mark.parametrize("sfpu_mode", STRICT_MODES)
def test_softsign_strict_specials_fp32(device, sfpu_mode):
    """fp32 specials follow torch: +-0 -> +-0, NaN -> NaN, +-inf -> NaN (inf / inf)."""
    values = [0.0, -0.0, float("nan"), float("inf"), float("-inf"), 1.0, -1.0, 2.0**127]
    flat = torch.tensor(values * (1024 // len(values)), dtype=torch.float32)
    x = flat.reshape(1, 1, 32, 32)
    result = ttnn.to_torch(ttnn.softsign(_to_device(x, ttnn.float32, device), sfpu_mode=sfpu_mode))

    zeros = x == 0
    assert torch.equal(_bits(result[zeros]), _bits(x[zeros])), "softsign(+-0) must be +-0, sign included"
    nonfinite = ~torch.isfinite(x)
    assert torch.isnan(result[nonfinite]).all(), "softsign(NaN) and softsign(+-inf) must be NaN"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_softsign_legacy_call_is_precise_relaxed(device, dtype):
    """ttnn.softsign(x) without a mode is unchanged: bit-identical to PreciseRelaxed.

    The unary program runs with math_approx_mode=False, so the legacy APPROX-driven call
    selects the precise reciprocal with no guard -- including on the flush band.
    """
    x = torch.cat([_interior_input(dtype, (1, 1, 32, 32)), _band_input(dtype)], dim=-1)
    tt_x = _to_device(x, dtype, device)
    legacy = ttnn.to_torch(ttnn.softsign(tt_x))
    relaxed = ttnn.to_torch(ttnn.softsign(tt_x, sfpu_mode=ttnn.SfpuMode.PreciseRelaxed))
    assert torch.equal(_bits(legacy), _bits(relaxed))


@pytest.mark.parametrize("sfpu_mode", ALL_MODES)
def test_softsign_mode_through_unary_chain(device, sfpu_mode):
    """UnaryWithParam(SOFTSIGN, mode) in an op chain runs the same sequence as ttnn.softsign(sfpu_mode=mode)."""
    x = torch.cat([_interior_input(ttnn.float32, (1, 1, 32, 32)), _band_input(ttnn.float32)], dim=-1)
    tt_x = _to_device(x, ttnn.float32, device)
    chained = ttnn.to_torch(ttnn.unary_chain(tt_x, [ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTSIGN, sfpu_mode)]))
    direct = ttnn.to_torch(ttnn.softsign(tt_x, sfpu_mode=sfpu_mode))
    assert torch.equal(_bits(chained), _bits(direct))


def test_softsign_modes_are_distinct_programs(device):
    """Each mode compiles its own program; repeating a mode reuses its cache entry."""
    tt_x = _to_device(_interior_input(ttnn.bfloat16, (1, 1, 32, 32)), ttnn.bfloat16, device)
    start = device.num_program_cache_entries()
    for mode in ALL_MODES:
        ttnn.softsign(tt_x, sfpu_mode=mode)
    assert device.num_program_cache_entries() - start == len(ALL_MODES)
    for mode in ALL_MODES:
        ttnn.softsign(tt_x, sfpu_mode=mode)
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
    with expect_error(RuntimeError, "SOFTSIGN only"):
        ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, ttnn.SfpuMode.PreciseStrict)
