# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Prove STFT phase decorrelation is atan2 sensitivity, not broken conv or ``ttnn.atan2``.

On Kokoro harmonic input (n_fft=20, hop=5, sine amp=0.1) the strided-conv STFT bins agree
closely with ``torch.stft``:

    PCC(X_real_ref, X_real_tt) ≈ 1.0
    PCC(X_imag_ref, X_imag_tt) ≈ 1.0

yet ``transform`` phase / cos(phase) PCC is low (~0.6–0.8).  Degradation lives in
``_magnitude_phase_from_xy`` (``sqrt``/``atan2``), concentrated on near-zero-magnitude bins
where the angle is ill-conditioned.  The round-trip audio stays ~lossless because those bins
carry negligible energy.

Also checks ``ttnn.atan2`` tracks ``torch.atan2`` on the *same* fp32 (x, y) inputs — small
input differences are amplified near the origin; that is atan2 sensitivity, not a broken SFPU.

Run::

    pytest -s models/experimental/kokoro/tests/test_stft_atan2_sensitivity_proof.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference.istftnet import TorchSTFT
from models.experimental.kokoro.tt.tt_torch_stft import TTTorchSTFT, preprocess_tt_torch_stft

_N_FFT = 20
_HOP = 5
_WIN = 20
_INPUT_LEN = 1500
_F0 = 200.0
_AMPLITUDE = 0.1
_SR = 24_000
_NEAR_ORIGIN_RADIUS = 0.05


@dataclass(frozen=True)
class StageMetric:
    name: str
    pcc: float
    mae: float


@dataclass(frozen=True)
class Atan2SensitivityReport:
    x_real: StageMetric
    x_imag: StageMetric
    magnitude: StageMetric
    phase: StageMetric
    cos_phase: StageMetric
    atan2_same_inputs: StageMetric
    audio_roundtrip: StageMetric
    total_bins: int
    near_origin_count: int
    high_mag_count: int
    phase_pcc_high_mag: float
    phase_pcc_near_origin: float


def _metric(name: str, ref: torch.Tensor, tt: torch.Tensor) -> StageMetric:
    ref_f = ref.detach().float().reshape(-1)
    tt_f = tt.detach().float().reshape(-1)
    n = min(ref_f.numel(), tt_f.numel())
    ref_f, tt_f = ref_f[:n], tt_f[:n]
    _, pcc = comp_pcc(ref_f.unsqueeze(0), tt_f.unsqueeze(0), pcc=0.0)
    mae = float((ref_f - tt_f).abs().mean())
    return StageMetric(name=name, pcc=float(pcc), mae=mae)


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    n = min(af.numel(), bf.numel())
    if n == 0:
        return float("nan")
    _, pcc = comp_pcc(af[:n].unsqueeze(0), bf[:n].unsqueeze(0), pcc=0.0)
    return float(pcc)


def _harmonic_signal() -> torch.Tensor:
    t = torch.arange(_INPUT_LEN, dtype=torch.float32)
    return (_AMPLITUDE * torch.sin(2 * torch.pi * _F0 / _SR * t)).unsqueeze(0)


def _ref_xy(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ref = TorchSTFT(filter_length=_N_FFT, hop_length=_HOP, win_length=_WIN).eval()
    with torch.no_grad():
        z = torch.stft(
            x,
            _N_FFT,
            _HOP,
            _WIN,
            window=ref.window.to(x.device),
            return_complex=True,
        )
        x_real = z.real
        x_imag = z.imag
        mag_ref, phase_ref = torch.abs(z), torch.angle(z)
    return x_real, x_imag, mag_ref, phase_ref


def analyze_atan2_sensitivity(device) -> Atan2SensitivityReport:
    x = _harmonic_signal()
    x_real_ref, x_imag_ref, mag_ref, phase_ref = _ref_xy(x)

    params = preprocess_tt_torch_stft(
        filter_length=_N_FFT,
        hop_length=_HOP,
        win_length=_WIN,
        input_length=_INPUT_LEN,
        device=device,
    )
    tt_mod = TTTorchSTFT(device, params, use_torch_stft_fallback=False)
    x_tt = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    X_real_tt, X_imag_tt = tt_mod._forward_stft_conv(x_tt)
    mag_tt, phase_tt = tt_mod._magnitude_phase_from_xy(X_real_tt, X_imag_tt)
    y_tt = tt_mod.inverse(mag_tt, phase_tt)

    x_real_tt = ttnn.to_torch(X_real_tt).float()
    x_imag_tt = ttnn.to_torch(X_imag_tt).float()
    mag_tt_cpu = ttnn.to_torch(mag_tt).float()
    phase_tt_cpu = ttnn.to_torch(phase_tt).float()
    y_tt_cpu = ttnn.to_torch(y_tt).float()

    ttnn.deallocate(X_real_tt)
    ttnn.deallocate(X_imag_tt)
    ttnn.deallocate(mag_tt)
    ttnn.deallocate(phase_tt)
    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)

    while x_real_tt.dim() > x_real_ref.dim():
        x_real_tt = x_real_tt.squeeze(0)
    while x_imag_tt.dim() > x_imag_ref.dim():
        x_imag_tt = x_imag_tt.squeeze(0)
    while mag_tt_cpu.dim() > mag_ref.dim():
        mag_tt_cpu = mag_tt_cpu.squeeze(0)
    while phase_tt_cpu.dim() > phase_ref.dim():
        phase_tt_cpu = phase_tt_cpu.squeeze(0)

    ref = TorchSTFT(filter_length=_N_FFT, hop_length=_HOP, win_length=_WIN).eval()
    with torch.no_grad():
        y_ref = ref(x).squeeze(-2)

    while y_tt_cpu.dim() > y_ref.dim():
        y_tt_cpu = y_tt_cpu.squeeze(0)

    # ``ttnn.atan2`` fidelity on identical fp32 inputs (mix ref+tt xy to stress near-origin).
    mc = ttnn.DRAM_MEMORY_CONFIG
    x_mix = 0.5 * (x_real_ref + x_real_tt)
    y_mix = 0.5 * (x_imag_ref + x_imag_tt)
    x_t = ttnn.from_torch(x_mix, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    y_t = ttnn.from_torch(y_mix, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    phase_tt_on_mix = ttnn.atan2(y_t, x_t, memory_config=mc)
    phase_torch = torch.atan2(y_mix, x_mix)
    phase_tt_mix_cpu = ttnn.to_torch(phase_tt_on_mix).float()
    ttnn.deallocate(phase_tt_on_mix)
    ttnn.deallocate(x_t)
    ttnn.deallocate(y_t)

    mag_flat = mag_ref.reshape(-1)
    phase_ref_flat = phase_ref.reshape(-1)
    phase_tt_flat = phase_tt_cpu.reshape(-1)
    z_mag = torch.sqrt(x_real_ref.reshape(-1) ** 2 + x_imag_ref.reshape(-1) ** 2)
    near = z_mag < _NEAR_ORIGIN_RADIUS
    high_mag = ~near
    n_near = int(near.sum().item())
    n_high = int(high_mag.sum().item())
    n_total = int(mag_flat.numel())

    phase_pcc_high = _pearson(phase_ref_flat[high_mag], phase_tt_flat[high_mag]) if high_mag.any() else float("nan")
    phase_pcc_near = _pearson(phase_ref_flat[near], phase_tt_flat[near]) if near.any() else float("nan")

    return Atan2SensitivityReport(
        x_real=_metric("X_real", x_real_ref, x_real_tt),
        x_imag=_metric("X_imag", x_imag_ref, x_imag_tt),
        magnitude=_metric("magnitude", mag_ref, mag_tt_cpu),
        phase=_metric("phase", phase_ref, phase_tt_cpu),
        cos_phase=_metric("cos(phase)", torch.cos(phase_ref), torch.cos(phase_tt_cpu)),
        atan2_same_inputs=_metric("atan2(same xy)", phase_torch, phase_tt_mix_cpu),
        audio_roundtrip=_metric("audio", y_ref, y_tt_cpu),
        total_bins=n_total,
        near_origin_count=n_near,
        high_mag_count=n_high,
        phase_pcc_high_mag=phase_pcc_high,
        phase_pcc_near_origin=phase_pcc_near,
    )


def _log_report(r: Atan2SensitivityReport) -> None:
    print(f"\n=== STFT atan2 sensitivity proof (n_fft={_N_FFT}, hop={_HOP}, L={_INPUT_LEN}) ===")
    for m in (r.x_real, r.x_imag, r.magnitude, r.phase, r.cos_phase, r.atan2_same_inputs, r.audio_roundtrip):
        print(f"  {m.name:<22} PCC={m.pcc:10.6f}  MAE={m.mae:13.6e}")
    print(
        f"\n  Phase PCC by |z| region (radius={_NEAR_ORIGIN_RADIUS}, total={r.total_bins}):\n"
        f"    high-|z| bins (n={r.high_mag_count})              PCC = {r.phase_pcc_high_mag:.6f}\n"
        f"    near-origin bins (n={r.near_origin_count})     PCC = {r.phase_pcc_near_origin:.6f}"
    )
    print(
        "\n  Conclusion: xy PCC ≈ 1.0 but phase PCC is low → degradation is atan2 (near-zero bins), "
        "not conv. ttnn.atan2 on shared inputs is faithful; sensitivity is geometric."
    )


def test_stft_atan2_sensitivity_proof(device):
    """Conv bins match; phase decorrelates at atan2; round-trip audio stays ~lossless."""
    r = analyze_atan2_sensitivity(device)
    _log_report(r)

    assert r.x_real.pcc > 0.99, f"X_real PCC should be ~1.0 (got {r.x_real.pcc:.6f})"
    assert r.x_imag.pcc > 0.99, f"X_imag PCC should be ~1.0 (got {r.x_imag.pcc:.6f})"
    assert r.magnitude.pcc > 0.99, f"magnitude PCC should be high (got {r.magnitude.pcc:.6f})"

    assert (
        r.phase.pcc < r.x_real.pcc - 0.1
    ), f"phase PCC {r.phase.pcc:.6f} should be well below xy PCC {r.x_real.pcc:.6f}"
    assert r.cos_phase.pcc < 0.90, f"cos(phase) PCC should be low on harmonic input (got {r.cos_phase.pcc:.6f})"

    assert (
        r.atan2_same_inputs.pcc > 0.99
    ), f"ttnn.atan2 should match torch.atan2 on identical inputs (got {r.atan2_same_inputs.pcc:.6f})"

    assert r.phase_pcc_near_origin < r.phase_pcc_high_mag, (
        f"near-origin phase PCC {r.phase_pcc_near_origin:.4f} should be worse than "
        f"high-|z| {r.phase_pcc_high_mag:.4f}"
    )

    assert (
        r.audio_roundtrip.pcc > 0.999
    ), f"STFT round-trip should be ~lossless despite low phase PCC (got {r.audio_roundtrip.pcc:.6f})"
