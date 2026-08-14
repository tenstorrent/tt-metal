# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Prove the low ``rad_frac`` / voicing PCC is the F0 **input** divergence, not any SineGen op.

In end-to-end diagnostics ``rad_frac = (fn / sr) % 1`` shows a low PCC (~0.54). The natural
suspicion is that an op amplifies a small F0 error. This test rules out **both** candidate ops
and pins the drop on the input:

  1. **Not the modulo.** ``% 1`` is discontinuous at integer boundaries, but at Kokoro scale
     ``fn / sr ≈ 200/24000 ≈ 0.008`` is nowhere near a wrap point, so the modulo is a near-identity.
  2. **Not the voicing threshold op.** The genuinely discontinuous op is the boolean
     ``uv = f0 > 0`` (it flips 0↔1 on frames near the voicing boundary), but running that exact
     threshold in **torch** instead of on device (``ttnn.gt``) gives a bit-for-bit identical PCC —
     so the op backend is irrelevant.

What's left is the **input**: a sub-Hz disagreement between the ref and TT prosody paths'
``f0_upsampled``. The discontinuous ``> 0`` turns that into mask flips regardless of backend, and
``rad_frac`` (≈0 on unvoiced frames, small-positive on voiced) merely inherits the flipped mask.

``test_sinegen_phase_fallback_proof.py`` feeds the **same** synthetic F0 to ref and TT, so
``rad_frac`` stays at PCC ≈ 1.0 and that test cannot show this behaviour on real kmodel input.

This test uses the real config-E kmodel ``f0_upsampled`` pair (ref ``f0u_ref`` vs on-device
``f0u_tt`` from ``"Hello from Tenstorrent."``) and runs three checks:

  A. **Path-faithful** — ref stages vs TT stages on each path's own upsampled F0.
     ``fn_harmonics`` stays tight (~1.0); ``uv_mask`` drops first (~0.57) where voicing flips;
     ``rad_frac`` inherits that drop (~0.54) without adding a new cliff at the modulo op.

  B. **Shared-input** — both paths fed ``f0u_ref``.
     ``rad_frac`` stays at PCC ≈ 1.0 (MAE ~1e-5 from BF16 ``ttnn.remainder`` only) — proving
     the modulo op itself is faithful.

  C. **Voicing threshold: torch vs device** — build ``uv = f0 > 0`` on the TT-path F0 both in
     torch and via ``ttnn.gt``. ``PCC(uv_torch, uv_device) ≈ 1.0`` and both score the *same*
     ~0.57 against ``uv_ref`` — running the threshold in torch recovers nothing, so the drop is
     the input, not the op.

Run::

    pytest -s models/experimental/kokoro/tests/test_sinegen_voicing_input_not_op_proof.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F_torch

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

import ttnn

from models.experimental.kokoro.m_source_rng import (
    deallocate_m_source_rng_tt,
    make_zero_m_source_rng,
    upload_m_source_rng,
)
from models.experimental.kokoro.tests.kokoro_checkpoint import STFT_PHASE_FALLBACK_KWARGS, _ref_prosody, _tokenize
from models.experimental.kokoro.tests.test_sinegen_phase_fallback_proof import (
    _SINEGEN_INPUT_STAGE,
    _SINEGEN_STAGE_EXEC_ORDER,
    _SINEGEN_STAGE_OPS,
    _run_tt_sinegen_stages,
    sinegen_stage_display,
)
from models.experimental.kokoro.tests.test_tt_kmodel_pcc import (
    _find_checkpoint,
    _setup,
    _zero_noise,
)
from models.experimental.kokoro.tt.tt_generator import _upsample_nearest_axis1
from models.experimental.kokoro.tt.tt_kmodel import TTKModel
from models.experimental.kokoro.tt.tt_sinegen import TTSineGen, preprocess_tt_sinegen

_TEST_TEXT = "Hello from Tenstorrent."
_SAMPLING_RATE = 24000.0
_SINE_AMP = 0.1
_NOISE_STD = 0.003
_VOICED_THRESHOLD = 0.0


@dataclass(frozen=True)
class ModuloStageRow:
    stage: str
    pcc: float
    mae: float
    ops: str
    note: str = ""


def _pcc_mae(ref: torch.Tensor, tt: torch.Tensor) -> tuple[float, float, str]:
    ref_f = ref.detach().float().reshape(-1)
    tt_f = tt.detach().float().reshape(-1)
    note = ""
    if ref_f.numel() != tt_f.numel():
        n = min(ref_f.numel(), tt_f.numel())
        note = f"truncated {ref_f.numel()} vs {tt_f.numel()} -> {n}"
        ref_f, tt_f = ref_f[:n], tt_f[:n]
    if ref_f.numel() == 0:
        return float("nan"), float("nan"), note or "empty"
    from models.common.utility_functions import comp_pcc

    _, pcc = comp_pcc(ref_f.unsqueeze(0), tt_f.unsqueeze(0), pcc=0.0)
    mae = float((ref_f - tt_f).abs().mean())
    return float(pcc), mae, note


def _run_ref_sinegen_stages_kmodel(
    f0u: torch.Tensor,
    *,
    harmonic_num: int,
    upsample_scale: int,
    rand_ini_b1d: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    dim = harmonic_num + 1
    harmonics = torch.arange(1, dim + 1, dtype=torch.float32).reshape(1, 1, dim)

    uv = (f0u > _VOICED_THRESHOLD).float()
    fn = f0u.float() * harmonics
    rad_mod = (fn / _SAMPLING_RATE) % 1.0
    rad = rad_mod.clone()
    if rand_ini_b1d is not None:
        rad[:, 0:1, :] = rad[:, 0:1, :] + rand_ini_b1d

    rad_down_t = F_torch.interpolate(
        rad.transpose(1, 2), scale_factor=1.0 / upsample_scale, mode="linear", align_corners=False
    )
    rad_down = rad_down_t.transpose(1, 2)
    phase_cumsum = torch.cumsum(rad_down, dim=1)
    phase_2pi = phase_cumsum * (2.0 * math.pi)
    phase_up_t = F_torch.interpolate(
        phase_2pi.transpose(1, 2) * upsample_scale,
        scale_factor=float(upsample_scale),
        mode="linear",
        align_corners=False,
    )
    phase_up = phase_up_t.transpose(1, 2)
    sin_raw = torch.sin(phase_up)
    sine_x_amp = sin_raw * _SINE_AMP
    noise_amp = uv * _NOISE_STD + (1.0 - uv) * (_SINE_AMP / 3.0)
    noise = noise_amp * torch.zeros_like(sine_x_amp)
    out = sine_x_amp * uv + noise

    return {
        "S0_fn": fn,
        "S1_uv": uv,
        "S2_rad_mod": rad_mod,
        "S2_rad": rad,
        "S3_rad_down": rad_down,
        "S4_phase_cumsum": phase_cumsum,
        "S6_phase_up_rad": phase_up,
        "S7_sin_raw": sin_raw,
        "S8_sine_x_amp": sine_x_amp,
        "S9_out_uv_noise": out,
    }


def _kmodel_f0_upsampled_pair(device) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    ckpt = _find_checkpoint()
    if ckpt is None:
        raise RuntimeError("Kokoro-82M checkpoint not found locally.")

    ref, params, phonemes, ref_s = _setup(ckpt, device, disable_complex=False)
    input_ids, _, input_lengths, lengths_list = _tokenize(ref.vocab, phonemes, ref.context_length)
    _, F0_ref, _, _, _ = _ref_prosody(ref, phonemes, ref_s)

    with _zero_noise():
        tt_model = TTKModel(device, ref, params, **STFT_PHASE_FALLBACK_KWARGS)
    mc = ttnn.DRAM_MEMORY_CONFIG
    ck = tt_model._predictor.compute_kernel_config
    _, F0_tt, _, _, _ = tt_model._device_forward_prosody_stages(
        input_ids, input_lengths, lengths_list, ref_s[:, params.style_dim :], 1.0, mc, ck
    )
    F0_tt_cpu = ttnn.to_torch(F0_tt).float().squeeze().cpu()
    ttnn.deallocate(F0_tt)

    gen_ref = ref.decoder.generator
    gen_tt = tt_model._get_decoder(int(F0_ref.shape[-1]))._generator
    scale = int(gen_tt.params.upsample_scale_full)
    harmonic_num = int(gen_ref.m_source.l_sin_gen.harmonic_num)

    with torch.no_grad(), _zero_noise():
        f0u_ref = gen_ref.f0_upsamp(F0_ref[:, None]).transpose(1, 2).contiguous().cpu()

    f0_tt = ttnn.from_torch(
        F0_tt_cpu.unsqueeze(0), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
    )
    f0_b_t_1 = ttnn.unsqueeze(f0_tt, 2)
    ttnn.deallocate(f0_tt)
    f0_har = _upsample_nearest_axis1(f0_b_t_1, scale=scale, memory_config=mc)
    ttnn.deallocate(f0_b_t_1)
    f0u_tt = ttnn.to_torch(f0_har).float().cpu()
    ttnn.deallocate(f0_har)

    return f0u_ref, f0u_tt, scale, harmonic_num


def analyze_modulo_pcc(
    f0u_ref: torch.Tensor,
    f0u_tt: torch.Tensor,
    scale: int,
    harmonic_num: int,
    device,
) -> tuple[list[ModuloStageRow], list[ModuloStageRow]]:
    dim = harmonic_num + 1
    rng_cpu = make_zero_m_source_rng(1, int(f0u_ref.shape[1]), dim)
    rand_ini = rng_cpu.rand_ini.reshape(1, 1, dim)

    ref_caps = _run_ref_sinegen_stages_kmodel(
        f0u_ref, harmonic_num=harmonic_num, upsample_scale=scale, rand_ini_b1d=rand_ini
    )

    mc = ttnn.DRAM_MEMORY_CONFIG
    params = preprocess_tt_sinegen(
        device=device,
        sampling_rate=_SAMPLING_RATE,
        upsample_scale=scale,
        harmonic_num=harmonic_num,
        sine_amp=_SINE_AMP,
        noise_std=_NOISE_STD,
        voiced_threshold=_VOICED_THRESHOLD,
        time_len=int(f0u_ref.shape[1]),
        weights_dtype=ttnn.bfloat16,
    )
    rng_tt = upload_m_source_rng(rng_cpu, device, memory_config=mc)
    with _zero_noise():
        tt_mod = TTSineGen(device, params, use_torch_phase_fallback=True)

    def _tt_caps(f0u: torch.Tensor) -> dict[str, torch.Tensor]:
        f0_tt = ttnn.from_torch(f0u, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
        caps = _run_tt_sinegen_stages(
            tt_mod,
            f0_tt,
            rand_ini_tt=rng_tt.rand_ini,
            noise_raw_tt=rng_tt.sinegen_noise,
            memory_config=mc,
        )
        ttnn.deallocate(f0_tt)
        return caps

    path_tt = _tt_caps(f0u_tt)
    shared_tt = _tt_caps(f0u_ref)
    deallocate_m_source_rng_tt(rng_tt)

    def _rows(tt_f0u: torch.Tensor, tt_caps: dict[str, torch.Tensor], *, note: str) -> list[ModuloStageRow]:
        out: list[ModuloStageRow] = []
        f0_pcc, f0_mae, f0_note = _pcc_mae(f0u_ref, tt_f0u)
        out.append(
            ModuloStageRow(
                stage=_SINEGEN_INPUT_STAGE,
                pcc=f0_pcc,
                mae=f0_mae,
                ops=_SINEGEN_STAGE_OPS[_SINEGEN_INPUT_STAGE],
                note=note or f0_note,
            )
        )
        for stage in _SINEGEN_STAGE_EXEC_ORDER[1:]:
            if stage not in ref_caps or stage not in tt_caps:
                continue
            pcc, mae, row_note = _pcc_mae(ref_caps[stage], tt_caps[stage])
            out.append(
                ModuloStageRow(
                    stage=stage,
                    pcc=pcc,
                    mae=mae,
                    ops=_SINEGEN_STAGE_OPS.get(stage, ""),
                    note=note or row_note,
                )
            )
        return out

    return (
        _rows(f0u_tt, path_tt, note="path-faithful f0u"),
        _rows(f0u_ref, shared_tt, note="shared ref f0u"),
    )


def _log_modulo_report(path_rows: list[ModuloStageRow], shared_rows: list[ModuloStageRow]) -> None:
    def _print_table(title: str, rows: list[ModuloStageRow]) -> None:
        print(f"\n{title}")
        print(f"  {'step':<28} {'PCC':>10}  {'MAE':>13}  ops / note")
        print("  " + "-" * 96)
        prev = 1.0
        for r in rows:
            flag = "  <-- drop" if prev > 0.95 and r.pcc < 0.95 else ""
            print(f"  {sinegen_stage_display(r.stage):<28} {r.pcc:10.6f}  {r.mae:13.6e}  " f"{r.ops}{flag}  {r.note}")
            prev = r.pcc

    print(f"\n=== SineGen modulo / pre-phase PCC (kmodel f0_upsampled, text={_TEST_TEXT!r}) ===")
    _print_table("  A. Path-faithful (ref f0u_ref vs TT f0u_tt):", path_rows)
    _print_table("  B. Shared-input (both fed ref f0u_ref — isolates op fidelity):", shared_rows)
    print(
        "\n  Interpretation: low path-faithful rad_frac PCC tracks uv_mask / f0 sign disagreements, "
        "not broken ttnn.remainder. Shared-input rad_frac PCC ≈ 1 proves modulo is faithful."
    )


# How close the torch-threshold PCC must be to the on-device-threshold PCC to count as "the same".
_SAME_PCC_TOL = 0.02


@dataclass(frozen=True)
class VoicingGtReport:
    pcc_ref_vs_device: float
    pcc_ref_vs_torch: float
    pcc_device_vs_torch: float
    mae_device_vs_torch: float


def analyze_voicing_gt(f0u_ref: torch.Tensor, f0u_tt: torch.Tensor, device) -> VoicingGtReport:
    """Build ``uv = f0 > 0`` on the TT-path F0 via torch and via ``ttnn.gt``; compare vs ``uv_ref``."""
    uv_ref = (f0u_ref > _VOICED_THRESHOLD).float()
    uv_torch = (f0u_tt > _VOICED_THRESHOLD).float()

    mc = ttnn.DRAM_MEMORY_CONFIG
    f0_tt = ttnn.from_torch(f0u_tt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    uv_bool = ttnn.gt(f0_tt, _VOICED_THRESHOLD, memory_config=mc)
    uv_dev_tt = ttnn.typecast(uv_bool, ttnn.float32, memory_config=mc)
    uv_device = ttnn.to_torch(uv_dev_tt).float().reshape(uv_torch.shape)
    ttnn.deallocate(uv_bool)
    ttnn.deallocate(uv_dev_tt)
    ttnn.deallocate(f0_tt)

    pcc_rd, _, _ = _pcc_mae(uv_ref, uv_device)
    pcc_rt, _, _ = _pcc_mae(uv_ref, uv_torch)
    pcc_dt, mae_dt, _ = _pcc_mae(uv_device, uv_torch)
    return VoicingGtReport(
        pcc_ref_vs_device=pcc_rd,
        pcc_ref_vs_torch=pcc_rt,
        pcc_device_vs_torch=pcc_dt,
        mae_device_vs_torch=mae_dt,
    )


def _log_voicing_gt_report(r: VoicingGtReport) -> None:
    print("\n  C. Voicing threshold (uv = f0 > 0) torch vs device:")
    print(f"     PCC(uv_device, uv_torch)  = {r.pcc_device_vs_torch:.6f}   MAE = {r.mae_device_vs_torch:.6e}")
    print(f"     PCC(uv_ref,    uv_device) = {r.pcc_ref_vs_device:.6f}   (threshold on device, ttnn.gt)")
    print(f"     PCC(uv_ref,    uv_torch)  = {r.pcc_ref_vs_torch:.6f}   (threshold in torch)")
    print(f"     change from moving gt to torch = {r.pcc_ref_vs_torch - r.pcc_ref_vs_device:+.6f}")
    print(
        "     -> running the threshold in torch does NOT change the PCC; the ~0.57 drop is the "
        "input F0 divergence, not ttnn.gt."
    )


@pytest.mark.timeout(600)
def test_sinegen_voicing_input_not_op_proof(device):
    """The low rad_frac / voicing PCC is the F0 input divergence, not the modulo OR the gt op.

    A (path-faithful): fn stays tight, uv_mask drops first, rad_frac inherits it — no new cliff
    at the modulo.  B (shared-input): modulo is faithful (rad_frac PCC ≈ 1.0).  C (gt torch vs
    device): the voicing threshold scores the same PCC on either backend — the drop is the input.
    """
    ckpt = _find_checkpoint()
    if ckpt is None:
        pytest.skip("Kokoro-82M checkpoint not found locally.")

    # Load the real kmodel f0_upsampled pair ONCE and feed it to all three checks.
    f0u_ref, f0u_tt, scale, harmonic_num = _kmodel_f0_upsampled_pair(device)

    path_rows, shared_rows = analyze_modulo_pcc(f0u_ref, f0u_tt, scale, harmonic_num, device)
    _log_modulo_report(path_rows, shared_rows)
    gt = analyze_voicing_gt(f0u_ref, f0u_tt, device)
    _log_voicing_gt_report(gt)

    path_by = {r.stage: r for r in path_rows}
    shared_by = {r.stage: r for r in shared_rows}

    # A/B — modulo. Matched f0u: modulo (and other pre-phase ops through rad_rand_ini) stay tight.
    assert (
        shared_by["S2_rad_mod"].pcc > 0.99
    ), f"shared-input rad_frac should be faithful (got {shared_by['S2_rad_mod'].pcc:.6f})"
    assert shared_by["S2_rad_mod"].mae < 1e-3, (
        f"shared-input rad_frac MAE should be tiny BF16 remainder error " f"(got {shared_by['S2_rad_mod'].mae:.6e})"
    )

    # Real kmodel path: fn stays tight; uv drops first; rad_frac inherits without a new cliff.
    assert path_by["S0_fn"].pcc > 0.99, f"fn should stay tight on path-faithful f0 (got {path_by['S0_fn'].pcc:.6f})"
    assert path_by["S1_uv"].pcc < 0.7, f"uv should drop on path-faithful f0 (got {path_by['S1_uv'].pcc:.6f})"
    assert (
        path_by["S2_rad_mod"].pcc < path_by["S0_fn"].pcc - 0.3
    ), f"rad_frac PCC {path_by['S2_rad_mod'].pcc:.4f} should inherit uv drop, not stay at fn level"
    assert abs(path_by["S2_rad_mod"].pcc - path_by["S1_uv"].pcc) < 0.05, (
        f"rad_frac PCC {path_by['S2_rad_mod'].pcc:.4f} should track uv_mask "
        f"{path_by['S1_uv'].pcc:.4f} — modulo does not add a separate cliff"
    )

    # C — voicing threshold op. ttnn.gt is bit-faithful on identical input...
    assert (
        gt.pcc_device_vs_torch > 0.999
    ), f"ttnn.gt should match torch '>' on identical input (got {gt.pcc_device_vs_torch:.6f})"
    # ...both thresholds drop against the reference...
    assert gt.pcc_ref_vs_device < 0.7, f"on-device voicing PCC should drop (got {gt.pcc_ref_vs_device:.6f})"
    assert gt.pcc_ref_vs_torch < 0.7, (
        f"torch threshold PCC {gt.pcc_ref_vs_torch:.6f} also dropped — "
        "running gt in torch should not recover the voicing mask"
    )
    # ...and they agree, so the op backend is irrelevant; the input F0 divergence is the cause.
    assert abs(gt.pcc_ref_vs_torch - gt.pcc_ref_vs_device) < _SAME_PCC_TOL, (
        f"torch ({gt.pcc_ref_vs_torch:.6f}) and device ({gt.pcc_ref_vs_device:.6f}) voicing PCC differ "
        f"by more than {_SAME_PCC_TOL} — the threshold op, not the input, would then be the cause"
    )
