# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Prove SineGen ``use_torch_phase_fallback`` is required at Kokoro scale.

At ``upsample_scale=300`` the TTNN phase chain (downsample matmul → cumsum → lerp upsample
→ ``× 2π×upsample_scale`` → ``sin``) runs MACs in BF16.  Small cumsum values (~3×10⁻⁵ cycles)
pick up ~3×10⁻⁵ absolute error that is amplified by ``2π × 300 ≈ 1885`` into ~0.06–0.25 rad
phase error — comparable to ``sine_amp=0.1`` and enough to destroy sine-wave PCC.

This test captures per-stage PCC/MAE in device execution order on the **same F0** and shows:

1. Pre-phase ops (``fn``, ``uv``, ``rad_frac`` modulo, ``rad_rand_ini``, ``rad_down``) stay tight on TTNN.
2. Phase-chain stages (``phase_cumsum``, ``phase_up``, ``sin``, ``sine×amp``, final ``out``) degrade on TTNN.
3. ``use_torch_phase_fallback=True`` restores phase-chain PCC > 0.99.

Modulo / ``rad_frac`` behaviour on **real kmodel** ``f0_upsampled`` (path-faithful vs shared-input)
is in ``test_sinegen_voicing_input_not_op_proof.py`` — this file uses synthetic F0 where ``rad_frac`` stays
≈ 1.0 and cannot demonstrate modulo sensitivity.

Full-pipeline audio PCC vs each fallback combination (none / stft-only / phase-only / stft+phase)
is in ``test_tt_kmodel_pcc_degradation.py``.

Run::

    pytest -s models/experimental/kokoro/tests/test_sinegen_phase_fallback_proof.py -k kokoro_scale
"""

from __future__ import annotations

import math
import sys
from contextlib import contextmanager
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
    MSourceRngTT,
    deallocate_m_source_rng_tt,
    make_zero_m_source_rng,
    upload_m_source_rng,
)
from models.experimental.kokoro.tt.tt_sinegen import TTSineGen, _to_fp32_if_needed, preprocess_tt_sinegen

# Kokoro harmonic-path scale: ``T_f0=162`` mel frames × ``upsample_scale=300`` → ``T_har=48600``.
_SAMPLING_RATE = 24000.0
_UPSAMPLE_SCALE = 300
_TIME_LEN = 48600
_HARMONIC_NUM = 0
_SINE_AMP = 0.1
_NOISE_STD = 0.003
_VOICED_THRESHOLD = 0.0

# Input tensor key (not produced by SineGen ops — compared separately).
_SINEGEN_INPUT_STAGE = "S0_f0_btd"

# Internal capture keys in device execution order (see ``_run_tt_sinegen_stages``).
_SINEGEN_STAGE_EXEC_ORDER = (
    _SINEGEN_INPUT_STAGE,
    "S1_uv",
    "S0_fn",
    "S2_rad_mod",
    "S2_rad",
    "S3_rad_down",
    "S4_phase_cumsum",
    "S6_phase_up_rad",
    "S7_sin_raw",
    "S8_sine_x_amp",
    "S9_out_uv_noise",
)

_SINEGEN_STAGE_LABELS: dict[str, str] = {
    _SINEGEN_INPUT_STAGE: "f0_input",
    "S1_uv": "uv_mask",
    "S0_fn": "fn_harmonics",
    "S2_rad_mod": "rad_frac",
    "S2_rad": "rad_rand_ini",
    "S3_rad_down": "rad_down",
    "S4_phase_cumsum": "phase_cumsum",
    "S6_phase_up_rad": "phase_up",
    "S7_sin_raw": "sin",
    "S8_sine_x_amp": "sine_amp",
    "S9_out_uv_noise": "sine_wavs",
}

_SINEGEN_STAGE_OPS: dict[str, str] = {
    _SINEGEN_INPUT_STAGE: "f0_btd upsampled F0 input [B,T,1]",
    "S1_uv": "uv = f0 > threshold  [ttnn.gt, typecast]",
    "S0_fn": "fn = f0 × harmonics  [ttnn.multiply]",
    "S2_rad_mod": "rad = (fn / sr) % 1  [× inv_sr, remainder]",
    "S2_rad": "rad += rand_ini @ t=0  [ttnn.add]",
    "S3_rad_down": "rad_down = interp(rad)  [permute, matmul]",
    "S4_phase_cumsum": "phase = cumsum(rad_down)  [slice+add]",
    "S6_phase_up_rad": "phase_up = lerp(phase) × 2π  [concat, multiply]",
    "S7_sin_raw": "sin(phase_up)  [ttnn.sin]",
    "S8_sine_x_amp": "sine × amp  [ttnn.multiply]",
    "S9_out_uv_noise": "uv × sine + noise  [multiply, add]",
}

_PRE_PHASE_STAGES = _SINEGEN_STAGE_EXEC_ORDER[1:6]
_PHASE_CHAIN_STAGES = _SINEGEN_STAGE_EXEC_ORDER[6:]
# In use_torch_phase_fallback, only the phase ACCUMULATION runs on CPU; sin/×amp/uv-mix stay on device.
_TORCH_FALLBACK_STAGES = ("S3_rad_down", "S4_phase_cumsum", "S6_phase_up_rad")


def sinegen_stage_display(stage_key: str) -> str:
    """Human-readable step label with execution index (00–10)."""
    idx = _SINEGEN_STAGE_EXEC_ORDER.index(stage_key)
    return f"{idx:02d} {_SINEGEN_STAGE_LABELS[stage_key]}"


@dataclass(frozen=True)
class SinegenPhaseStageRow:
    stage: str
    pcc: float
    mae: float
    ttnn_ops: str


@dataclass(frozen=True)
class SinegenPhaseFallbackReport:
    rows_ttnn: tuple[SinegenPhaseStageRow, ...]
    rows_fallback: tuple[SinegenPhaseStageRow, ...]
    pcc_sine_ttnn: float
    pcc_sine_fallback: float
    mae_sine_ttnn: float
    mae_sine_fallback: float


@contextmanager
def _deterministic_torch_random():
    real_rand = torch.rand
    real_randn_like = torch.randn_like

    def fake_rand(*size, **kwargs):
        return torch.zeros(*size, **kwargs)

    def fake_randn_like(t, **kwargs):
        return torch.zeros_like(t, **kwargs)

    torch.rand = fake_rand
    torch.randn_like = fake_randn_like
    try:
        yield
    finally:
        torch.rand = real_rand
        torch.randn_like = real_randn_like


def _pcc(ref: torch.Tensor, tt: torch.Tensor) -> float:
    ref_f = ref.detach().float().reshape(-1)
    tt_f = tt.detach().float().reshape(-1)
    n = min(ref_f.numel(), tt_f.numel())
    if n == 0:
        return float("nan")
    _, pcc = comp_pcc(ref_f[:n].unsqueeze(0), tt_f[:n].unsqueeze(0), pcc=0.0)
    return float(pcc)


def _mae(ref: torch.Tensor, tt: torch.Tensor) -> float:
    ref_f = ref.detach().float().reshape(-1)
    tt_f = tt.detach().float().reshape(-1)
    n = min(ref_f.numel(), tt_f.numel())
    if n == 0:
        return float("nan")
    return float((ref_f[:n] - tt_f[:n]).abs().mean())


def _kokoro_f0(seed: int = 4) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.relu(torch.randn(1, _TIME_LEN, 1) * 200.0)


def _run_ref_sinegen_stages(
    f0_btd: torch.Tensor,
    *,
    rand_ini_b1d: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Reference float32 stages aligned with ``TTSineGen`` naming."""
    dim = _HARMONIC_NUM + 1
    harmonics = torch.arange(1, dim + 1, dtype=torch.float32).reshape(1, 1, dim)

    uv = (f0_btd > _VOICED_THRESHOLD).float()
    fn = f0_btd * harmonics
    rad_mod = (fn / _SAMPLING_RATE) % 1.0
    rad = rad_mod.clone()
    if rand_ini_b1d is not None:
        rad[:, 0:1, :] = rad[:, 0:1, :] + rand_ini_b1d

    rad_down_t = F_torch.interpolate(
        rad.transpose(1, 2), scale_factor=1.0 / _UPSAMPLE_SCALE, mode="linear", align_corners=False
    )
    rad_down = rad_down_t.transpose(1, 2)
    phase_cumsum = torch.cumsum(rad_down, dim=1)
    phase_2pi = phase_cumsum * (2.0 * math.pi)
    phase_up_t = F_torch.interpolate(
        phase_2pi.transpose(1, 2) * _UPSAMPLE_SCALE,
        scale_factor=float(_UPSAMPLE_SCALE),
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


def _tt_to_cpu(t: ttnn.Tensor) -> torch.Tensor:
    out = ttnn.to_torch(t).float()
    while out.dim() > 3 and out.shape[0] == 1:
        out = out.squeeze(0)
    return out


def _run_torch_phase_stages(
    tt_mod: TTSineGen,
    f0_tt: ttnn.Tensor,
    *,
    rand_ini_tt: ttnn.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Stages produced by ``_torch_phase_fallback`` (CPU float32 phase chain)."""
    p = tt_mod.params
    B = int(f0_tt.shape[0])
    f0_cpu = ttnn.to_torch(f0_tt).float().reshape(B, p.time_len, 1)
    harmonics_cpu = ttnn.to_torch(p.harmonics).float().reshape(p.dim)

    fn = f0_cpu * harmonics_cpu
    rad_mod = (fn / p.sampling_rate) % 1.0
    rad = rad_mod.clone()
    if rand_ini_tt is not None:
        rand_ini_cpu = ttnn.to_torch(rand_ini_tt).float().reshape(B, 1, p.dim)
        rand_ini_cpu[..., 0] = 0.0
        rad[:, 0:1, :] = rad[:, 0:1, :] + rand_ini_cpu

    rad_down_t = F_torch.interpolate(
        rad.transpose(1, 2), scale_factor=1.0 / p.upsample_scale, mode="linear", align_corners=False
    )
    rad_down = rad_down_t.transpose(1, 2)
    phase_cumsum = torch.cumsum(rad_down, dim=1)
    phase_2pi = phase_cumsum * (2.0 * math.pi)
    phase_up_t = F_torch.interpolate(
        phase_2pi.transpose(1, 2) * p.upsample_scale,
        scale_factor=float(p.upsample_scale),
        mode="linear",
        align_corners=False,
    )
    phase_up = phase_up_t.transpose(1, 2)
    sin_raw = torch.sin(phase_up)
    sine_amp_float = float(ttnn.to_torch(p.sine_amp).flatten()[0].item())
    sine_x_amp = sin_raw * sine_amp_float

    return {
        "S0_fn": fn,
        "S2_rad_mod": rad_mod,
        "S3_rad_down": rad_down,
        "S4_phase_cumsum": phase_cumsum,
        "S6_phase_up_rad": phase_up,
        "S7_sin_raw": sin_raw,
        "S8_sine_x_amp": sine_x_amp,
    }


def _run_tt_sinegen_stages(
    tt_mod: TTSineGen,
    f0_tt: ttnn.Tensor,
    *,
    rand_ini_tt: ttnn.Tensor | None,
    noise_raw_tt: ttnn.Tensor | None,
    memory_config: ttnn.MemoryConfig,
) -> dict[str, torch.Tensor]:
    """Instrumented forward — per-op captures for pure TTNN or torch-phase fallback."""
    p = tt_mod.params
    B = int(f0_tt.shape[0])
    caps: dict[str, torch.Tensor] = {}

    uv_bool = ttnn.gt(f0_tt, p.voiced_threshold, memory_config=memory_config)
    uv = ttnn.typecast(uv_bool, p.activation_dtype, memory_config=memory_config)
    ttnn.deallocate(uv_bool)
    caps["S1_uv"] = _tt_to_cpu(uv)

    fn = ttnn.multiply(f0_tt, p.harmonics, memory_config=memory_config)
    caps["S0_fn"] = _tt_to_cpu(fn)
    rad = ttnn.multiply(fn, p.inv_sampling_rate, memory_config=memory_config)
    ttnn.deallocate(fn)
    rad = ttnn.remainder(rad, p.one, memory_config=memory_config)
    caps["S2_rad_mod"] = _tt_to_cpu(rad)

    if rand_ini_tt is not None:
        rand_masked = ttnn.multiply(rand_ini_tt, p.fundamental_zero_mask, memory_config=memory_config)
        if p.time_len > 1:
            tail = ttnn.zeros(
                [B, p.time_len - 1, p.dim],
                dtype=rad.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=tt_mod.device,
                memory_config=memory_config,
            )
            rand_pad = ttnn.concat([rand_masked, tail], dim=1, memory_config=memory_config)
            ttnn.deallocate(tail)
        else:
            rand_pad = rand_masked
        rad = ttnn.add(rad, rand_pad, memory_config=memory_config)
        ttnn.deallocate(rand_pad)
        if rand_pad is not rand_masked:
            ttnn.deallocate(rand_masked)

    caps["S2_rad"] = _tt_to_cpu(rad)

    if tt_mod.use_torch_phase_fallback:
        caps.update(_run_torch_phase_stages(tt_mod, f0_tt, rand_ini_tt=rand_ini_tt))
        ttnn.deallocate(rad)
        # New contract: ``_torch_phase_fallback`` returns the fp32 ``phase_up`` argument; the
        # nonlinear ``sin`` and ``× sine_amp`` run on device (fp32 sin matches torch.sin at PCC≈1.0).
        phase_up = tt_mod._torch_phase_fallback(f0_tt, rand_ini_tt)
        sine_amp_fp32, owns_sa = _to_fp32_if_needed(p.sine_amp, memory_config)
        sines = ttnn.sin(phase_up, memory_config=memory_config)
        ttnn.deallocate(phase_up)
        caps["S7_sin_raw"] = _tt_to_cpu(sines)
        sine_waves_unmasked = ttnn.multiply(sines, sine_amp_fp32, memory_config=memory_config)
        ttnn.deallocate(sines)
        if owns_sa:
            ttnn.deallocate(sine_amp_fp32)
        caps["S8_sine_x_amp"] = _tt_to_cpu(sine_waves_unmasked)
    else:
        rad_fp32, owns_rad = _to_fp32_if_needed(rad, memory_config)
        if owns_rad:
            ttnn.deallocate(rad)
        rad_bdt = ttnn.permute(rad_fp32, (0, 2, 1), memory_config=memory_config)
        ttnn.deallocate(rad_fp32)

        rad_down = ttnn.matmul(
            rad_bdt,
            p.interp_down,
            memory_config=memory_config,
            compute_kernel_config=tt_mod.compute_kernel_config,
        )
        ttnn.deallocate(rad_bdt)
        rad_down, owns_rd = _to_fp32_if_needed(rad_down, memory_config)
        caps["S3_rad_down"] = _tt_to_cpu(rad_down).permute(0, 2, 1).contiguous()

        r_slices = [
            ttnn.slice(rad_down, [0, 0, t], [B, p.dim, t + 1], [1, 1, 1], memory_config=memory_config)
            for t in range(p.time_len_down)
        ]
        if owns_rd:
            ttnn.deallocate(rad_down)
        prefix = [r_slices[0]]
        for t in range(1, p.time_len_down):
            prefix.append(ttnn.add(prefix[-1], r_slices[t], memory_config=memory_config))
        for t in range(1, p.time_len_down):
            ttnn.deallocate(r_slices[t])
        phase = ttnn.concat(prefix, dim=2, memory_config=memory_config)
        for t in range(p.time_len_down):
            ttnn.deallocate(prefix[t])
        caps["S4_phase_cumsum"] = _tt_to_cpu(phase).permute(0, 2, 1).contiguous()

        phase_btd = ttnn.permute(phase, (0, 2, 1), memory_config=memory_config)
        ttnn.deallocate(phase)

        lerp_alpha, owns_la = _to_fp32_if_needed(p.lerp_alpha, memory_config)
        lerp_one_minus, owns_lom = _to_fp32_if_needed(p.lerp_one_minus_alpha, memory_config)
        lerp_clamp, owns_lc = _to_fp32_if_needed(p.lerp_clamp_ones, memory_config)
        two_pi_scale, owns_tps = _to_fp32_if_needed(p.two_pi_times_scale, memory_config)
        sine_amp_fp32, owns_sa = _to_fp32_if_needed(p.sine_amp, memory_config)

        p0 = ttnn.slice(phase_btd, [0, 0, 0], [B, 1, p.dim], [1, 1, 1], memory_config=memory_config)
        start_seg = ttnn.multiply(p0, lerp_clamp, memory_config=memory_config)
        ttnn.deallocate(p0)

        lerp_segs = [start_seg]
        for s in range(p.time_len_down - 1):
            p_s = ttnn.slice(phase_btd, [0, s, 0], [B, s + 1, p.dim], [1, 1, 1], memory_config=memory_config)
            p_s1 = ttnn.slice(phase_btd, [0, s + 1, 0], [B, s + 2, p.dim], [1, 1, 1], memory_config=memory_config)
            seg = ttnn.add(
                ttnn.multiply(p_s, lerp_one_minus, memory_config=memory_config),
                ttnn.multiply(p_s1, lerp_alpha, memory_config=memory_config),
                memory_config=memory_config,
            )
            ttnn.deallocate(p_s)
            ttnn.deallocate(p_s1)
            lerp_segs.append(seg)

        p_last = ttnn.slice(
            phase_btd,
            [0, p.time_len_down - 1, 0],
            [B, p.time_len_down, p.dim],
            [1, 1, 1],
            memory_config=memory_config,
        )
        end_seg = ttnn.multiply(p_last, lerp_clamp, memory_config=memory_config)
        ttnn.deallocate(p_last)
        ttnn.deallocate(phase_btd)
        lerp_segs.append(end_seg)

        phase_up = ttnn.concat(lerp_segs, dim=1, memory_config=memory_config)
        for seg in lerp_segs:
            ttnn.deallocate(seg)

        phase_up = ttnn.multiply(phase_up, two_pi_scale, memory_config=memory_config)
        caps["S6_phase_up_rad"] = _tt_to_cpu(phase_up)

        sines = ttnn.sin(phase_up, memory_config=memory_config)
        ttnn.deallocate(phase_up)
        caps["S7_sin_raw"] = _tt_to_cpu(sines)

        sine_waves_unmasked = ttnn.multiply(sines, sine_amp_fp32, memory_config=memory_config)
        ttnn.deallocate(sines)
        caps["S8_sine_x_amp"] = _tt_to_cpu(sine_waves_unmasked)

        if owns_la:
            ttnn.deallocate(lerp_alpha)
        if owns_lom:
            ttnn.deallocate(lerp_one_minus)
        if owns_lc:
            ttnn.deallocate(lerp_clamp)
        if owns_tps:
            ttnn.deallocate(two_pi_scale)
        if owns_sa:
            ttnn.deallocate(sine_amp_fp32)

    one_minus_uv = ttnn.subtract(p.one, uv, memory_config=memory_config)
    noise_amp = ttnn.add(
        ttnn.multiply(uv, p.noise_std, memory_config=memory_config),
        ttnn.multiply(one_minus_uv, p.sine_amp_over_three, memory_config=memory_config),
        memory_config=memory_config,
    )
    ttnn.deallocate(one_minus_uv)

    if noise_raw_tt is None:
        if B == 1:
            noise_raw_local = tt_mod._noise_raw
            owns_noise = False
        else:
            noise_raw_local = ttnn.concat([tt_mod._noise_raw] * B, dim=0, memory_config=memory_config)
            owns_noise = True
    else:
        noise_raw_local = noise_raw_tt
        owns_noise = False
    noise = ttnn.multiply(noise_amp, noise_raw_local, memory_config=memory_config)
    ttnn.deallocate(noise_amp)
    if owns_noise:
        ttnn.deallocate(noise_raw_local)

    sine_waves_masked = ttnn.multiply(sine_waves_unmasked, uv, memory_config=memory_config)
    ttnn.deallocate(sine_waves_unmasked)
    out = ttnn.add(sine_waves_masked, noise, memory_config=memory_config)
    ttnn.deallocate(sine_waves_masked)
    ttnn.deallocate(noise)
    caps["S9_out_uv_noise"] = _tt_to_cpu(out)
    ttnn.deallocate(out)
    ttnn.deallocate(uv)

    return caps


def _build_modules(
    device,
    f0: torch.Tensor,
    *,
    use_torch_phase_fallback: bool,
) -> tuple[TTSineGen, ttnn.Tensor, MSourceRngTT]:
    params = preprocess_tt_sinegen(
        device=device,
        sampling_rate=_SAMPLING_RATE,
        upsample_scale=_UPSAMPLE_SCALE,
        harmonic_num=_HARMONIC_NUM,
        sine_amp=_SINE_AMP,
        noise_std=_NOISE_STD,
        voiced_threshold=_VOICED_THRESHOLD,
        time_len=_TIME_LEN,
        weights_dtype=ttnn.bfloat16,
    )
    rng = make_zero_m_source_rng(1, _TIME_LEN, _HARMONIC_NUM + 1)
    rng_tt = upload_m_source_rng(rng, device, dtype=params.activation_dtype)
    with _deterministic_torch_random():
        tt_mod = TTSineGen(device, params, use_torch_phase_fallback=use_torch_phase_fallback)
    f0_tt = ttnn.from_torch(f0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return tt_mod, f0_tt, rng_tt


def analyze_sinegen_phase_fallback(device) -> SinegenPhaseFallbackReport:
    """Stage-wise PCC: ref float32 vs TTNN path and vs torch-phase fallback."""
    f0 = _kokoro_f0()
    rng = make_zero_m_source_rng(1, _TIME_LEN, _HARMONIC_NUM + 1)
    rand_ini_b1d = rng.rand_ini.reshape(1, 1, _HARMONIC_NUM + 1)
    ref_caps = _run_ref_sinegen_stages(f0, rand_ini_b1d=rand_ini_b1d)

    rows_ttnn: list[SinegenPhaseStageRow] = []
    rows_fallback: list[SinegenPhaseStageRow] = []

    for use_fb in (False, True):
        tt_mod, f0_tt, rng_tt = _build_modules(device, f0, use_torch_phase_fallback=use_fb)
        mc = ttnn.DRAM_MEMORY_CONFIG
        f0_btd_tt = _tt_to_cpu(f0_tt)
        tt_caps = _run_tt_sinegen_stages(
            tt_mod,
            f0_tt,
            rand_ini_tt=rng_tt.rand_ini,
            noise_raw_tt=rng_tt.sinegen_noise,
            memory_config=mc,
        )
        ttnn.deallocate(f0_tt)
        deallocate_m_source_rng_tt(rng_tt)

        target = rows_ttnn if not use_fb else rows_fallback
        target.append(
            SinegenPhaseStageRow(
                stage=_SINEGEN_INPUT_STAGE,
                pcc=_pcc(f0, f0_btd_tt),
                mae=_mae(f0, f0_btd_tt),
                ttnn_ops=_SINEGEN_STAGE_OPS[_SINEGEN_INPUT_STAGE],
            )
        )
        for stage in _PRE_PHASE_STAGES + _PHASE_CHAIN_STAGES:
            if stage not in ref_caps or stage not in tt_caps:
                continue
            target.append(
                SinegenPhaseStageRow(
                    stage=stage,
                    pcc=_pcc(ref_caps[stage], tt_caps[stage]),
                    mae=_mae(ref_caps[stage], tt_caps[stage]),
                    ttnn_ops=_SINEGEN_STAGE_OPS[stage],
                )
            )

    pcc_sine_ttnn = next(r.pcc for r in rows_ttnn if r.stage == "S8_sine_x_amp")
    pcc_sine_fallback = next(r.pcc for r in rows_fallback if r.stage == "S8_sine_x_amp")
    mae_sine_ttnn = next(r.mae for r in rows_ttnn if r.stage == "S8_sine_x_amp")
    mae_sine_fallback = next(r.mae for r in rows_fallback if r.stage == "S8_sine_x_amp")
    return SinegenPhaseFallbackReport(
        rows_ttnn=tuple(rows_ttnn),
        rows_fallback=tuple(rows_fallback),
        pcc_sine_ttnn=pcc_sine_ttnn,
        pcc_sine_fallback=pcc_sine_fallback,
        mae_sine_ttnn=mae_sine_ttnn,
        mae_sine_fallback=mae_sine_fallback,
    )


def _log_report(r: SinegenPhaseFallbackReport) -> None:
    print(f"\n=== SineGen phase fallback proof (upsample_scale={_UPSAMPLE_SCALE}, T={_TIME_LEN}) ===")
    print(f"  sine×amp PCC  pure TTNN = {r.pcc_sine_ttnn:.6f}   MAE = {r.mae_sine_ttnn:.6e}")
    print(f"  sine×amp PCC  torch fb  = {r.pcc_sine_fallback:.6f}   MAE = {r.mae_sine_fallback:.6e}")
    print(
        f"  recovery delta  PCC={r.pcc_sine_fallback - r.pcc_sine_ttnn:+.6f}  MAE={r.mae_sine_fallback - r.mae_sine_ttnn:+.6e}\n"
    )

    print("  Pure TTNN path (ref vs device, execution order):")
    print(f"  {'step':<28} {'PCC':>10} {'MAE':>13}  ops")
    print("  " + "-" * 88)
    for row in r.rows_ttnn:
        print(f"  {sinegen_stage_display(row.stage):<28} {row.pcc:10.6f} {row.mae:13.6e}  {row.ttnn_ops}")

    print("\n  With use_torch_phase_fallback=True (execution order):")
    print(f"  {'step':<28} {'PCC':>10} {'MAE':>13}  backend")
    print("  " + "-" * 88)
    for row in r.rows_fallback:
        backend = "torch.cpu" if row.stage in _TORCH_FALLBACK_STAGES else "ttnn"
        print(f"  {sinegen_stage_display(row.stage):<28} {row.pcc:10.6f} {row.mae:13.6e}  {backend}")


def test_sinegen_phase_fallback_kokoro_scale_proof(device):
    """Per-stage proof: TTNN phase chain degrades PCC; torch fallback restores it."""
    report = analyze_sinegen_phase_fallback(device)
    _log_report(report)

    pre_ttnn = {r.stage: r.pcc for r in report.rows_ttnn if r.stage in _PRE_PHASE_STAGES}
    phase_ttnn = {r.stage: r.pcc for r in report.rows_ttnn if r.stage in _PHASE_CHAIN_STAGES}
    phase_fb = {r.stage: r.pcc for r in report.rows_fallback if r.stage in _PHASE_CHAIN_STAGES}

    for stage, pcc in pre_ttnn.items():
        assert pcc > 0.99, f"pre-phase {stage} should stay tight on TTNN (got {pcc:.6f})"

    # At T=48600 the TTNN phase chain collapses; sin is the stage that exposes accumulated error.
    assert (
        phase_ttnn["S7_sin_raw"] < 0.99
    ), f"sin PCC {phase_ttnn['S7_sin_raw']:.6f} should degrade on TTNN at full Kokoro length"
    assert (
        phase_ttnn["S8_sine_x_amp"] < 0.99
    ), f"sine×amp PCC {phase_ttnn['S8_sine_x_amp']:.6f} should degrade on TTNN at full Kokoro length"
    assert report.pcc_sine_ttnn < 0.99

    for stage, pcc in phase_fb.items():
        assert pcc > 0.99, f"torch phase fallback must restore {stage} (got {pcc:.6f})"

    assert report.pcc_sine_fallback > 0.99
    assert (
        report.pcc_sine_fallback - report.pcc_sine_ttnn > 0.5
    ), "fallback should materially recover sine PCC vs pure TTNN at T=48600"

    # MAE corroborates PCC: the fallback drives the sine-stage absolute error down toward zero.
    assert report.mae_sine_fallback < report.mae_sine_ttnn, (
        f"fallback should reduce sine MAE (got fb {report.mae_sine_fallback:.6e} "
        f"vs TTNN {report.mae_sine_ttnn:.6e})"
    )

    # ``sin`` nonlinearly amplifies small phase error from cumsum/lerp/×2π×scale.
    assert (
        phase_ttnn["S6_phase_up_rad"] > phase_ttnn["S7_sin_raw"]
    ), "sin PCC must be worse than phase_up PCC (nonlinear amplification)"
    assert (
        pre_ttnn["S3_rad_down"] > phase_ttnn["S7_sin_raw"]
    ), "sin PCC must be worse than rad_down PCC (error grows through phase chain)"
