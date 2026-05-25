# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Prove atan2 phase decorrelation is not a TTNN bug — STFT ``xref``/``yref`` vs ``xtt``/``ytt``.

Naming (x = real, y = imag):

- ``xref``, ``yref`` — reference :class:`~models.experimental.kokoro.reference.istftnet.TorchSTFT`
  (``torch.stft`` real/imag, tag ``ref_torch_stft``)
- ``xtt``, ``ytt`` — TT conv STFT bins before ``ttnn.atan2`` (tag ``magnitude_phase_fp32``)

On Kokoro harmonic input the two paths agree closely (PCC(xref, xtt) ≈ 1, PCC(yref, ytt) ≈ 1),
yet ``atan2`` outputs are much less correlated — and ``ttnn.atan2`` tracks ``torch.atan2`` on the
same inputs.  Small input differences are amplified near the origin; that is atan2 sensitivity,
not broken ``ttnn.atan2``.

Load dumps from the fixed default dir (overwritten each run)::

    KOKORO_DUMP_STFT_XY=1 pytest .../test_tt_torch_stft_pcc.py -k dump_xy -s
    pytest .../test_stft_atan2_correlation_proof.py -k from_existing -s

Default dir: ``/tmp/pytest-of-ubuntu/kokoro_stft_xy``

Flowchart: ``tests/STFT_ATAN2_SENSITIVITY.md``.
"""

from __future__ import annotations

import os
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
from models.experimental.kokoro.stft_xy_dump import reset_stft_xy_dump_counter, stft_xy_dump_dir, stft_xy_dump_paths
from models.experimental.kokoro.tt.tt_torch_stft import TTTorchSTFT, preprocess_tt_torch_stft

_N_FFT = 20
_HOP = 5
_WIN = 20
_REF_TAG = "ref_torch_stft"
_TT_TAG = "magnitude_phase_fp32"


@dataclass(frozen=True)
class Atan2StftCorrelationReport:
    pcc_xref_xtt: float
    pcc_yref_ytt: float
    pcc_torch_atan2_cross: float
    pcc_ttnn_atan2_cross: float
    pcc_ttnn_atan2_same_on_tt: float
    pcc_ttnn_atan2_cross_ref_torch: float
    mae_x: float
    mae_y: float
    mae_phase_torch: float
    mae_phase_ttnn: float
    near_origin_count: int
    near_origin_pcc_torch: float
    near_origin_pcc_ttnn: float


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.detach().float().reshape(-1)
    b_f = b.detach().float().reshape(-1)
    return torch.corrcoef(torch.stack([a_f, b_f]))[0, 1].item()


def load_ref_tt_xy(
    dump_dir: Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load ``xref``, ``yref``, ``xtt``, ``ytt`` from dump dir (ignores CustomSTFT files)."""
    d = dump_dir or stft_xy_dump_dir()
    ref_real, ref_imag, _ = stft_xy_dump_paths(_REF_TAG)
    tt_real, tt_imag, _ = stft_xy_dump_paths(_TT_TAG)
    for p in (ref_real, ref_imag, tt_real, tt_imag):
        if not p.exists():
            raise FileNotFoundError(f"missing dump {p} (dir={d})")
    xref = torch.load(ref_real, map_location="cpu", weights_only=True)
    yref = torch.load(ref_imag, map_location="cpu", weights_only=True)
    xtt = torch.load(tt_real, map_location="cpu", weights_only=True)
    ytt = torch.load(tt_imag, map_location="cpu", weights_only=True)
    return xref, yref, xtt, ytt


def _harmonic_signal(L: int = 1500, f0: float = 200.0, amplitude: float = 0.1, sr: int = 24_000) -> torch.Tensor:
    t = torch.arange(L, dtype=torch.float32)
    return (amplitude * torch.sin(2 * torch.pi * f0 / sr * t)).unsqueeze(0)


def _dump_ref_tt_xy(device, x: torch.Tensor) -> None:
    """Run ref TorchSTFT + TT transform; writes to ``stft_xy_dump_dir()`` when env enabled."""
    reset_stft_xy_dump_counter()
    ref = TorchSTFT(filter_length=_N_FFT, hop_length=_HOP, win_length=_WIN).eval()
    params = preprocess_tt_torch_stft(
        filter_length=_N_FFT, hop_length=_HOP, win_length=_WIN, input_length=x.shape[-1], device=device
    )
    tt_mod = TTTorchSTFT(device, params, use_torch_stft_fallback=False, use_torch_atan2_fallback=False)

    with torch.no_grad():
        ref.transform(x)

    x_tt = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    mag_tt, phase_tt = tt_mod.transform(x_tt)
    ttnn.deallocate(mag_tt)
    ttnn.deallocate(phase_tt)
    ttnn.deallocate(x_tt)


def _ttnn_atan2_phase(device, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mc = ttnn.DRAM_MEMORY_CONFIG
    x_tt = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    y_tt = ttnn.from_torch(y, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    phase_tt = ttnn.atan2(y_tt, x_tt, memory_config=mc)
    out = ttnn.to_torch(phase_tt).float().reshape(-1)
    ttnn.deallocate(x_tt)
    ttnn.deallocate(y_tt)
    ttnn.deallocate(phase_tt)
    return out


def analyze_atan2_stft_correlation(
    device,
    xref: torch.Tensor,
    yref: torch.Tensor,
    xtt: torch.Tensor,
    ytt: torch.Tensor,
    *,
    near_origin_radius: float = 0.05,
) -> Atan2StftCorrelationReport:
    """Compare input vs ``atan2`` output correlation for ref vs TT STFT bins."""
    pcc_x = _pearson(xref, xtt)
    pcc_y = _pearson(yref, ytt)

    phase_ref_torch = torch.atan2(yref, xref).reshape(-1)
    phase_tt_torch = torch.atan2(ytt, xtt).reshape(-1)
    phase_ref_ttnn = _ttnn_atan2_phase(device, xref, yref)
    phase_tt_ttnn = _ttnn_atan2_phase(device, xtt, ytt)

    _, pcc_torch_cross = comp_pcc(phase_ref_torch.unsqueeze(0), phase_tt_torch.unsqueeze(0), pcc=0.0)
    _, pcc_ttnn_cross = comp_pcc(phase_ref_ttnn.unsqueeze(0), phase_tt_ttnn.unsqueeze(0), pcc=0.0)
    _, pcc_ttnn_same_on_tt = comp_pcc(phase_tt_ttnn.unsqueeze(0), phase_tt_torch.unsqueeze(0), pcc=0.0)
    _, pcc_ttnn_cross_ref_torch = comp_pcc(phase_ref_torch.unsqueeze(0), phase_tt_ttnn.unsqueeze(0), pcc=0.0)

    mag_ref = torch.sqrt(xref**2 + yref**2)
    near = mag_ref.reshape(-1) < near_origin_radius
    n_near = int(near.sum().item())
    if n_near > 0:
        pcc_near_torch = _pearson(phase_ref_torch[near], phase_tt_torch[near])
        pcc_near_ttnn = _pearson(phase_ref_ttnn[near], phase_tt_ttnn[near])
    else:
        pcc_near_torch = float("nan")
        pcc_near_ttnn = float("nan")

    return Atan2StftCorrelationReport(
        pcc_xref_xtt=float(pcc_x),
        pcc_yref_ytt=float(pcc_y),
        pcc_torch_atan2_cross=float(pcc_torch_cross),
        pcc_ttnn_atan2_cross=float(pcc_ttnn_cross),
        pcc_ttnn_atan2_same_on_tt=float(pcc_ttnn_same_on_tt),
        pcc_ttnn_atan2_cross_ref_torch=float(pcc_ttnn_cross_ref_torch),
        mae_x=(xref - xtt).abs().mean().item(),
        mae_y=(yref - ytt).abs().mean().item(),
        mae_phase_torch=(phase_ref_torch - phase_tt_torch).abs().mean().item(),
        mae_phase_ttnn=(phase_ref_ttnn - phase_tt_ttnn).abs().mean().item(),
        near_origin_count=n_near,
        near_origin_pcc_torch=float(pcc_near_torch),
        near_origin_pcc_ttnn=float(pcc_near_ttnn),
    )


def _log_report(r: Atan2StftCorrelationReport, *, dump_dir: str) -> None:
    print(
        f"\n=== STFT atan2 correlation proof (dump_dir={dump_dir}) ===\n"
        f"  PCC(xref, xtt)                 = {r.pcc_xref_xtt:.6f}   (inputs highly correlated)\n"
        f"  PCC(yref, ytt)                 = {r.pcc_yref_ytt:.6f}\n"
        f"  MAE(xref-xtt)                  = {r.mae_x:.6e}\n"
        f"  MAE(yref-ytt)                  = {r.mae_y:.6e}\n"
        f"\n"
        f"  PCC(torch.atan2(yref,xref),\n"
        f"      torch.atan2(ytt,xtt))      = {r.pcc_torch_atan2_cross:.6f}   (outputs less correlated)\n"
        f"  PCC(ttnn.atan2(yref,xref),\n"
        f"      ttnn.atan2(ytt,xtt))       = {r.pcc_ttnn_atan2_cross:.6f}   (same cross, ttnn both sides)\n"
        f"  PCC(ttnn.atan2(ytt,xtt),\n"
        f"      torch.atan2(ytt,xtt))      = {r.pcc_ttnn_atan2_same_on_tt:.6f}   (same tt input — op OK)\n"
        f"  PCC(torch.atan2(yref,xref),\n"
        f"      ttnn.atan2(ytt,xtt))       = {r.pcc_ttnn_atan2_cross_ref_torch:.6f}   (mixed ref/tt inputs)\n"
        f"  MAE phase torch (ref vs tt)   = {r.mae_phase_torch:.6f}\n"
        f"  MAE phase ttnn (ref vs tt)     = {r.mae_phase_ttnn:.6f}\n"
        f"\n"
        f"  Near-origin (|z|<{0.05}, n={r.near_origin_count}):\n"
        f"    PCC torch atan2 cross        = {r.near_origin_pcc_torch:.6f}\n"
        f"    PCC ttnn atan2 cross         = {r.near_origin_pcc_ttnn:.6f}\n"
    )


def test_stft_atan2_correlation_from_dump(device, monkeypatch):
    """Dump ref+TT xy, then show atan2 decorrelates while ttnn matches torch on same (xtt,ytt)."""
    monkeypatch.setenv("KOKORO_DUMP_STFT_XY", "1")

    x = _harmonic_signal()
    _dump_ref_tt_xy(device, x)
    dump_dir = stft_xy_dump_dir()
    xref, yref, xtt, ytt = load_ref_tt_xy(dump_dir)

    r = analyze_atan2_stft_correlation(device, xref, yref, xtt, ytt)
    _log_report(r, dump_dir=str(dump_dir))

    assert r.pcc_xref_xtt > 0.99, "xref and xtt should be nearly identical STFT real parts"
    assert r.pcc_yref_ytt > 0.99, "yref and ytt should be nearly identical STFT imag parts"
    assert r.pcc_torch_atan2_cross < r.pcc_xref_xtt, "atan2 outputs less correlated than inputs"
    assert r.pcc_torch_atan2_cross < 0.90, f"phase cross PCC too high: {r.pcc_torch_atan2_cross:.4f}"
    assert r.pcc_ttnn_atan2_same_on_tt > 0.99, "ttnn.atan2 must match torch.atan2 on same (xtt,ytt)"
    assert r.pcc_ttnn_atan2_cross < r.pcc_xref_xtt, "ttnn atan2 cross also less correlated than inputs"
    gap = abs(r.pcc_ttnn_atan2_cross - r.pcc_torch_atan2_cross)
    assert gap < 0.05, f"ttnn cross PCC should match torch cross; gap={gap:.4f}"


def test_stft_atan2_correlation_from_existing_dump(device):
    """Re-analyze dumps from ``/tmp/pytest-of-ubuntu/kokoro_stft_xy`` (or ``STFT_XY_DUMP_DIR``)."""
    dump_dir = Path(os.getenv("STFT_XY_DUMP_DIR", str(stft_xy_dump_dir())))
    xref, yref, xtt, ytt = load_ref_tt_xy(dump_dir)
    r = analyze_atan2_stft_correlation(device, xref, yref, xtt, ytt)
    _log_report(r, dump_dir=str(dump_dir))

    assert r.pcc_xref_xtt > 0.99
    assert r.pcc_yref_ytt > 0.99
    assert r.pcc_torch_atan2_cross < r.pcc_xref_xtt
    assert r.pcc_ttnn_atan2_same_on_tt > 0.99
    assert r.pcc_ttnn_atan2_cross < r.pcc_xref_xtt
    gap = abs(r.pcc_ttnn_atan2_cross - r.pcc_torch_atan2_cross)
    assert gap < 0.05
