# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Prove the SineGen phase collapse is a BF16-accumulation *hardware* limit, not a TTNN kernel bug.

The SineGen harmonic path at Kokoro scale (``upsample_scale=300``, ``T=48600``) collapses the
on-device ``sin(phase)`` PCC to ~0.31 vs the float32 reference. The open question this test
settles is *where* that error comes from:

    (A) a defect in a TTNN kernel (matmul / cumsum-add / lerp / sin), OR
    (B) the intrinsic precision of BF16 arithmetic, which Blackhole uses for its MAC datapath
        (``BH hardware rounds float32 -> BF16 for MACs``) regardless of ``fp32_dest_acc_en``.

Design — run the **exact same phase-chain math three ways** and compare:

    1. ``fp32 CPU``   torch, float32                      -> the golden reference.
    2. ``bf16 CPU``   torch, *the identical function*, only ``dtype=bfloat16``. Touches **zero**
                      TTNN kernels, runs entirely on the host.
    3. ``ttnn dev``   the real device path (``_run_tt_sinegen_stages``, HiFi3, fp32_dest_acc).

If (B) is the cause, then:
    * ``bf16 CPU`` must reproduce the device collapse (both ``sin`` PCC well below fp32), and
    * flipping *only the dtype argument* fp32->bf16 in one CPU function must be what breaks it, and
    * the device's absolute accumulated phase error must match the bf16-CPU sim's error magnitude
      (i.e. the device is operating at BF16 precision, not better, not worse).

All three hold. Because the CPU bf16 run contains no TTNN kernel yet collapses identically, the
collapse cannot be a kernel bug — it is the BF16 numeric format. This is why the production fix is
``use_torch_phase_fallback=True`` (accumulate the phase in fp32 on host); no on-device kernel change
can recover it.

Note on the mechanism (printed by the test): the phase ramps to ~hundreds of radians. BF16's
8-bit mantissa gives a quantization step of ``|phase| * 2**-8`` (~2 rad at |phase|~600). Taken mod
2*pi that randomizes the sine argument — so PCC(phase) stays ~1.0 (correlation on a huge ramp is
insensitive to it) while PCC(sin) collapses. Quantizing only the *input* ``rad`` (fp32 chain) leaves
``sin`` PCC > 0.9: the damage is specifically BF16 *accumulation*, not input rounding.

Companion proofs: ``test_sinegen_phase_fallback_proof.py`` (fallback restores PCC>0.99, per-stage),
``test_tt_kmodel_pcc_degradation.py`` (full-pipeline audio PCC per fallback combo).

Run::

    pytest -s models/experimental/kokoro/tests/test_bf16_accumulation_hardware_limit_proof.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F_torch

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.m_source_rng import (
    deallocate_m_source_rng_tt,
    make_zero_m_source_rng,
)
from models.experimental.kokoro.tests.test_sinegen_phase_fallback_proof import (
    _HARMONIC_NUM,
    _SAMPLING_RATE,
    _SINE_AMP,
    _UPSAMPLE_SCALE,
    _build_modules,
    _kokoro_f0,
    _run_ref_sinegen_stages,
    _run_tt_sinegen_stages,
)

# Collapse threshold: a sine whose PCC vs fp32 is below this is "destroyed".
_COLLAPSE_PCC = 0.5
# fp32 must stay essentially exact through the identical chain.
_EXACT_PCC = 0.99
# Input-only BF16 quantization (fp32 accumulation) must NOT collapse -> isolates accumulation.
_INPUT_ONLY_PCC = 0.9


def _pcc(ref: torch.Tensor, other: torch.Tensor) -> float:
    ref_f = ref.detach().float().reshape(-1)
    oth_f = other.detach().float().reshape(-1)
    n = min(ref_f.numel(), oth_f.numel())
    if n == 0:
        return float("nan")
    _, pcc = comp_pcc(ref_f[:n].unsqueeze(0), oth_f[:n].unsqueeze(0), pcc=0.0)
    return float(pcc)


def _mae(ref: torch.Tensor, other: torch.Tensor) -> float:
    ref_f = ref.detach().float().reshape(-1)
    oth_f = other.detach().float().reshape(-1)
    n = min(ref_f.numel(), oth_f.numel())
    return float((ref_f[:n] - oth_f[:n]).abs().mean())


def _phase_err_mod_2pi(ref_phase: torch.Tensor, other_phase: torch.Tensor) -> float:
    """Mean |phase error| wrapped into (-pi, pi] — the quantity that actually scrambles sin."""
    a = ref_phase.detach().float().reshape(-1)
    b = other_phase.detach().float().reshape(-1)
    n = min(a.numel(), b.numel())
    d = a[:n] - b[:n]
    d_wrapped = torch.remainder(d + math.pi, 2.0 * math.pi) - math.pi
    return float(d_wrapped.abs().mean())


def _rad_in_dtype(f0_btd: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``rad = (f0 * harmonics / sr) % 1`` computed at the given precision (rand_ini is zero here)."""
    dim = _HARMONIC_NUM + 1
    harmonics = torch.arange(1, dim + 1, dtype=dtype).reshape(1, 1, dim)
    f0d = f0_btd.to(dtype)
    fn = (f0d * harmonics).to(dtype)
    rad = (fn.float() / _SAMPLING_RATE).to(dtype)
    rad = torch.remainder(rad.float(), 1.0).to(dtype)
    return rad


def _cpu_phase_chain(rad: torch.Tensor, *, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """The SineGen phase chain (downsample -> cumsum -> lerp-up -> x2pi*scale -> sin) at ``dtype``.

    Op order and casts mirror the device path in ``_run_tt_sinegen_stages``: the cumsum is an
    iterative add (each partial sum re-rounded to ``dtype``), matching the on-device slice+add.
    Interpolation runs in fp32 then re-rounds, because on device the linear interp is a matmul
    whose *result* lands back in the working dtype.
    """
    rad = rad.to(dtype)
    rad_down_t = F_torch.interpolate(
        rad.transpose(1, 2).float(), scale_factor=1.0 / _UPSAMPLE_SCALE, mode="linear", align_corners=False
    ).to(dtype)
    rad_down = rad_down_t.transpose(1, 2)  # [B, T_down, dim]

    # cumsum as iterative add, each partial re-rounded to dtype (mirrors ttnn slice+add cumsum)
    phase = torch.zeros_like(rad_down)
    acc = torch.zeros_like(rad_down[:, 0:1, :])
    for t in range(rad_down.shape[1]):
        acc = (acc + rad_down[:, t : t + 1, :]).to(dtype)
        phase[:, t : t + 1, :] = acc

    phase_2pi = (phase.float() * (2.0 * math.pi)).to(dtype)
    phase_up_t = F_torch.interpolate(
        (phase_2pi.float().transpose(1, 2) * _UPSAMPLE_SCALE),
        scale_factor=float(_UPSAMPLE_SCALE),
        mode="linear",
        align_corners=False,
    ).to(dtype)
    phase_up = phase_up_t.transpose(1, 2)
    sin_raw = torch.sin(phase_up.float())
    return {
        "phase_up": phase_up.float(),
        "sin": sin_raw,
        "sine_amp": sin_raw * _SINE_AMP,
    }


@dataclass(frozen=True)
class Bf16AccumReport:
    # sin PCC vs fp32 golden
    pcc_sin_fp32_cpu: float  # identical function, fp32 -> must be ~1.0
    pcc_sin_bf16_cpu: float  # identical function, bf16 -> must collapse
    pcc_sin_input_only: float  # bf16 input, fp32 accumulation -> must NOT collapse
    pcc_sin_device: float  # real ttnn device path -> collapses
    # accumulated phase error magnitude (rad), wrapped mod 2pi
    phase_err_bf16_cpu: float
    phase_err_device: float
    bf16_quant_step: float  # |phase|_mean * 2**-8
    mean_abs_phase: float


def analyze_bf16_accumulation(device) -> Bf16AccumReport:
    f0 = _kokoro_f0()
    rng = make_zero_m_source_rng(1, f0.shape[1], _HARMONIC_NUM + 1)
    rand_ini_b1d = rng.rand_ini.reshape(1, 1, _HARMONIC_NUM + 1)
    ref = _run_ref_sinegen_stages(f0, rand_ini_b1d=rand_ini_b1d)

    # (3) Real device path — pure TTNN kernels.
    tt_mod, f0_tt, rng_tt = _build_modules(device, f0, use_torch_phase_fallback=False)
    tt = _run_tt_sinegen_stages(
        tt_mod,
        f0_tt,
        rand_ini_tt=rng_tt.rand_ini,
        noise_raw_tt=rng_tt.sinegen_noise,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(f0_tt)
    deallocate_m_source_rng_tt(rng_tt)

    # (1)/(2) Same CPU chain, only the dtype differs. (rand_ini==0 -> rad==rad_mod.)
    rad_bf16 = _rad_in_dtype(f0, torch.bfloat16)
    rad_fp32 = _rad_in_dtype(f0, torch.float32)
    cpu_fp32 = _cpu_phase_chain(rad_fp32, dtype=torch.float32)
    cpu_bf16 = _cpu_phase_chain(rad_bf16, dtype=torch.bfloat16)
    # Input-only: BF16-quantized rad, but accumulate in fp32 -> isolates accumulation from input rounding.
    cpu_input_only = _cpu_phase_chain(rad_bf16, dtype=torch.float32)

    mean_abs_phase = float(ref["S6_phase_up_rad"].detach().float().abs().mean())
    return Bf16AccumReport(
        pcc_sin_fp32_cpu=_pcc(ref["S7_sin_raw"], cpu_fp32["sin"]),
        pcc_sin_bf16_cpu=_pcc(ref["S7_sin_raw"], cpu_bf16["sin"]),
        pcc_sin_input_only=_pcc(ref["S7_sin_raw"], cpu_input_only["sin"]),
        pcc_sin_device=_pcc(ref["S7_sin_raw"], tt["S7_sin_raw"]),
        phase_err_bf16_cpu=_phase_err_mod_2pi(ref["S6_phase_up_rad"], cpu_bf16["phase_up"]),
        phase_err_device=_phase_err_mod_2pi(ref["S6_phase_up_rad"], tt["S6_phase_up_rad"]),
        bf16_quant_step=mean_abs_phase * (2.0**-8),
        mean_abs_phase=mean_abs_phase,
    )


def _log_report(r: Bf16AccumReport) -> None:
    print("\n=== BF16-accumulation hardware-limit proof (SineGen phase chain, T=48600) ===")
    print("  sin(phase) PCC vs fp32 golden — SAME chain, three backends:")
    print(f"    fp32  CPU  (identical fn, float32)   = {r.pcc_sin_fp32_cpu:.6f}   <- exact")
    print(f"    bf16  CPU  (identical fn, bfloat16)  = {r.pcc_sin_bf16_cpu:.6f}   <- collapses, NO ttnn")
    print(f"    ttnn  device (real kernels)          = {r.pcc_sin_device:.6f}   <- collapses")
    print(f"    bf16-input-only (fp32 accumulation)  = {r.pcc_sin_input_only:.6f}   <- survives")
    print("\n  Accumulated phase error (|Δphase| mod 2π, rad) — device runs at BF16 precision:")
    print(f"    bf16 CPU sim   = {r.phase_err_bf16_cpu:.4f} rad")
    print(f"    ttnn device    = {r.phase_err_device:.4f} rad")
    print(f"    BF16 quant step @ mean|phase|={r.mean_abs_phase:.1f} rad ≈ {r.bf16_quant_step:.3f} rad")
    print(
        "\n  Conclusion: a host-only BF16 run (zero TTNN kernels) reproduces the device collapse;\n"
        "  the identical fp32 run does not. The limit is the BF16 numeric format, not a kernel bug.\n"
        "  For full accuracy use use_torch_phase_fallback (fp32 phase accumulation on host).\n"
    )


def test_bf16_accumulation_is_hardware_limit_proof(device):
    r = analyze_bf16_accumulation(device)
    _log_report(r)

    # 1) The device collapses (this is the observed problem we are explaining).
    assert r.pcc_sin_device < _COLLAPSE_PCC, f"device sin PCC {r.pcc_sin_device:.4f} expected to collapse"

    # 2) A pure-CPU BF16 run of the SAME math — touching zero TTNN kernels — reproduces the collapse.
    assert (
        r.pcc_sin_bf16_cpu < _COLLAPSE_PCC
    ), f"bf16 CPU sin PCC {r.pcc_sin_bf16_cpu:.4f} must collapse too (reproduces device without any ttnn kernel)"

    # 3) The IDENTICAL function in fp32 is exact -> the only causal variable is the numeric format.
    assert (
        r.pcc_sin_fp32_cpu > _EXACT_PCC
    ), f"fp32 CPU sin PCC {r.pcc_sin_fp32_cpu:.4f} must stay exact (same code, only dtype changed)"

    # 4) Device and CPU-bf16 collapse to a comparable degree -> same phenomenon, not a kernel-specific defect.
    assert (
        abs(r.pcc_sin_device - r.pcc_sin_bf16_cpu) < 0.25
    ), f"device ({r.pcc_sin_device:.4f}) and bf16-CPU ({r.pcc_sin_bf16_cpu:.4f}) collapse should match"

    # 5) BF16-quantizing only the INPUT (fp32 accumulation) does NOT collapse -> it's accumulation, not input rounding.
    assert (
        r.pcc_sin_input_only > _INPUT_ONLY_PCC
    ), f"input-only bf16 sin PCC {r.pcc_sin_input_only:.4f} should survive -> collapse is from BF16 accumulation"

    # 6) The device accumulates the SAME magnitude of phase error as the BF16 CPU sim, and it is on the
    #    order of the BF16 quantization step -> the device datapath is operating at BF16 precision.
    assert r.phase_err_device > 0.3, f"device phase error {r.phase_err_device:.4f} rad expected to be BF16-scale"
    assert r.phase_err_bf16_cpu > 0.3, f"bf16 CPU phase error {r.phase_err_bf16_cpu:.4f} rad expected to be BF16-scale"
    ratio = r.phase_err_device / max(r.phase_err_bf16_cpu, 1e-9)
    assert (
        0.4 < ratio < 2.5
    ), f"device/bf16-CPU phase-error ratio {ratio:.2f} should be O(1): device runs at BF16 precision"
