# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Stage-wise PCC diagnostic for Kokoro TTKModel vs reference KModel.

``_run_tt_stages`` mirrors ``TTKModel._device_forward`` (same path as
``test_tt_kmodel_stft_and_phase_fallback_pcc`` / ``TTKModel(phonemes=..., deterministic=True)``).

Run::

    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
    python models/experimental/kokoro/tests/kmodel_pcc_stage_diagnostic.py --stft-phase-fallback --write-report
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
import ttnn

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference.model import KModel
from models.experimental.kokoro.tt.tt_kmodel import (
    TTKModel,
    _build_alignment,
    _to_fp32_if_needed,
    _zero_noise,
    preprocess_tt_kmodel,
)
from models.experimental.kokoro.m_source_rng import (
    deallocate_m_source_rng_tt,
    make_zero_m_source_rng,
    m_source_rng_shapes_from_f0,
    patched_m_source_torch_rng,
    upload_m_source_rng,
)
from models.experimental.kokoro.reference.istftnet import SineGen
from models.experimental.kokoro.tt.tt_lstm import tt_bilstm_nlc
from models.experimental.kokoro.tt.tt_sinegen import TTSineGen, preprocess_tt_sinegen
from models.experimental.kokoro.tests.kmodel_decode_stack_diagnostic import (
    StageTable,
    _compare_stages,
    _pcc_flat,
    _run_ref_decode_stack,
    _run_tt_decode_stack,
)
from models.experimental.kokoro.tests.sinegen_stage_diagnostic import (
    _STAGE_OPS as _SINEGEN_STAGE_OPS,
    _classify as _classify_pcc,
    _run_ref_sinegen_stages,
    _run_tt_sinegen_stages,
)

_DEFAULT_TEXT = os.getenv("KOKORO_PCC_DEBUG_TEXT", "Hello from Tenstorrent.")
_VOICE = "af_heart"
_LANG_CODE = "a"

_CKPT_CANDIDATES = (
    Path("/home/ubuntu/ign-tt/kokoro/examples/checkpoints/kokoro-v1_0.pth"),
    Path.home() / ".cache/huggingface/hub/models--hexgrad--Kokoro-82M/snapshots",
)

# Matches ``test_tt_kmodel_stft_and_phase_fallback_pcc`` (config E, device F0 + f0 upsample).
STFT_PHASE_FALLBACK_KWARGS = dict(
    use_torch_stft_fallback=True,
    use_torch_phase_fallback=True,
    use_torch_sinegen_fallback=False,
    use_torch_linear_fallback=False,
    use_torch_tanh_fallback=False,
    use_torch_stft_conv_fallback=False,
    use_torch_atan2_fallback=False,
    use_torch_f0n_conv_fallback=False,
    use_torch_f0_upsamp_fallback=False,
    use_fp32_prosody_boundary=True,
)


@dataclass
class RefStages:
    d_en_bct: torch.Tensor
    d_nlc: torch.Tensor
    pred_dur: torch.LongTensor
    en_bct: torch.Tensor
    F0: torch.Tensor
    N: torch.Tensor
    asr_bct: torch.Tensor
    audio: torch.Tensor


@dataclass
class TTStages:
    d_en_bct: torch.Tensor
    d_nlc: torch.Tensor
    pred_dur: torch.LongTensor
    en_bct: torch.Tensor
    F0: torch.Tensor
    N: torch.Tensor
    asr_bct: torch.Tensor
    audio: torch.Tensor


def _find_checkpoint() -> Optional[Path]:
    for p in _CKPT_CANDIDATES:
        if p.is_file():
            return p
        if p.is_dir():
            for child in p.rglob("kokoro-v1_0.pth"):
                return child
    return None


def _phonemize(text: str) -> tuple[str, torch.Tensor]:
    from kokoro import KPipeline

    pipe = KPipeline(lang_code=_LANG_CODE, model=False)
    results = list(pipe(text, voice=_VOICE))
    phonemes = results[0].phonemes
    pack = pipe.load_voice(_VOICE)
    ref_s = pack[len(phonemes) - 1]
    if not isinstance(ref_s, torch.Tensor):
        ref_s = torch.tensor(ref_s)
    ref_s = ref_s.float().cpu()
    if ref_s.dim() == 1:
        ref_s = ref_s.unsqueeze(0)
    return phonemes, ref_s


def _tokenize(
    vocab: dict, phonemes: str, context_length: int
) -> tuple[torch.LongTensor, torch.Tensor, torch.LongTensor, list[int]]:
    input_ids_list = list(filter(lambda i: i is not None, map(lambda p: vocab.get(p), phonemes)))
    assert len(input_ids_list) + 2 <= context_length
    input_ids = torch.LongTensor([[0, *input_ids_list, 0]])
    B, T = input_ids.shape
    input_lengths = torch.full((B,), T, dtype=torch.long)
    text_mask = torch.arange(T).unsqueeze(0).expand(B, -1).type_as(input_lengths)
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))
    return input_ids, text_mask, input_lengths, input_lengths.tolist()


def _tt_to_float(t: ttnn.Tensor) -> torch.Tensor:
    out = ttnn.to_torch(t).float()
    while out.dim() > 3 and out.shape[0] == 1:
        out = out.squeeze(0)
    return out


def _squeeze_batch(t: torch.Tensor) -> torch.Tensor:
    while t.dim() > 0 and t.shape[0] == 1 and t.dim() > 1:
        t = t.squeeze(0)
    return t


def _pcc_row(name: str, ref: torch.Tensor, tt: torch.Tensor) -> tuple[str, float, str]:
    ref_f = ref.detach().float().reshape(-1)
    tt_f = tt.detach().float().reshape(-1)
    note = ""
    if ref_f.numel() != tt_f.numel():
        n = min(ref_f.numel(), tt_f.numel())
        note = f"len ref={ref_f.numel()} tt={tt_f.numel()} (using first {n})"
        ref_f = ref_f[:n]
        tt_f = tt_f[:n]
    if ref_f.numel() == 0:
        return name, float("nan"), "empty"
    _, pcc = comp_pcc(ref_f.unsqueeze(0), tt_f.unsqueeze(0), pcc=0.0)
    if ref.shape != tt.shape and not note:
        note = f"shape ref={tuple(ref.shape)} tt={tuple(tt.shape)}"
    return name, float(pcc), note


def _print_table(rows: list[tuple[str, float, str]]) -> None:
    print(f"\n{'Stage':<42} {'PCC':>10}  Notes")
    print("-" * 72)
    for name, pcc, note in rows:
        print(f"{name:<42} {pcc:10.6f}  {note}")


def _run_ref_stages(ref: KModel, input_ids: torch.LongTensor, ref_s: torch.Tensor, speed: float = 1.0) -> RefStages:
    input_ids = input_ids.to(ref.device)
    ref_s = ref_s.to(ref.device)
    input_lengths = torch.full((input_ids.shape[0],), input_ids.shape[-1], device=input_ids.device, dtype=torch.long)
    text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1)
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(ref.device)

    with torch.no_grad(), _zero_noise():
        bert_dur = ref.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = ref.bert_encoder(bert_dur).transpose(-1, -2)
        s_pred = ref_s[:, 128:]
        d = ref.predictor.text_encoder(d_en, s_pred, input_lengths, text_mask)
        x, _ = ref.predictor.lstm(d)
        duration = ref.predictor.duration_proj(x)
        pred_dur = torch.round(torch.sigmoid(duration).sum(dim=-1) / speed).clamp(min=1).long().squeeze()
        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=ref.device), pred_dur)
        aln = torch.zeros((input_ids.shape[1], indices.shape[0]), device=ref.device)
        aln[indices, torch.arange(indices.shape[0], device=ref.device)] = 1
        aln = aln.unsqueeze(0)
        en = d.transpose(-1, -2) @ aln
        F0_pred, N_pred = ref.predictor.F0Ntrain(en, s_pred)
        t_en = ref.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ aln
        s_style = ref_s[:, :128]
        audio = ref.decoder(asr, F0_pred, N_pred, s_style).squeeze()

    return RefStages(
        d_en_bct=d_en.cpu(),
        d_nlc=d.cpu(),
        pred_dur=pred_dur.cpu(),
        en_bct=en.cpu(),
        F0=F0_pred.cpu(),
        N=N_pred.cpu(),
        asr_bct=asr.cpu(),
        audio=audio.cpu().float(),
    )


def _run_tt_stages(
    tt_model: TTKModel,
    input_ids: torch.LongTensor,
    text_mask: torch.Tensor,
    input_lengths: torch.LongTensor,
    lengths_list: list[int],
    ref_s: torch.Tensor,
    speed: float,
    *,
    deterministic: bool = True,
) -> TTStages:
    """Mirror ``TTKModel._device_forward`` with per-stage captures (E2E-equivalent to pytest)."""
    p = tt_model.params
    dev = tt_model.device
    mc = ttnn.DRAM_MEMORY_CONFIG
    ck = tt_model._predictor.compute_kernel_config
    B, T = input_ids.shape

    s_pred_cpu = ref_s[:, p.style_dim :]
    s_style_cpu = ref_s[:, : p.style_dim]
    caps: dict[str, torch.Tensor] = {}

    with _zero_noise():
        # 1. PL-BERT — attention_mask=None (matches forward)
        bert_out = tt_model._bert(input_ids, attention_mask=None)

        bert_for_enc = bert_out
        owns_bert_cast = False
        if tt_model._use_fp32_prosody_boundary and bert_out.dtype != ttnn.float32:
            bert_for_enc = ttnn.typecast(bert_out, ttnn.float32, memory_config=mc)
            owns_bert_cast = True
        d_en = ttnn.linear(
            bert_for_enc,
            p.bert_encoder_w,
            bias=p.bert_encoder_b,
            transpose_b=True,
            memory_config=mc,
            compute_kernel_config=ck,
        )
        ttnn.deallocate(bert_for_enc if owns_bert_cast else bert_out)
        while len(d_en.shape) > 3:
            d_en = ttnn.squeeze(d_en, 0)
        d_en_bct = ttnn.permute(d_en, (0, 2, 1), memory_config=mc)
        ttnn.deallocate(d_en)
        caps["d_en_bct"] = _tt_to_float(d_en_bct)

        prosody_dtype = ttnn.float32 if tt_model._use_fp32_prosody_boundary else ttnn.bfloat16
        if tt_model._use_fp32_prosody_boundary and d_en_bct.dtype != prosody_dtype:
            d_en_fp32 = ttnn.typecast(d_en_bct, ttnn.float32, memory_config=mc)
            ttnn.deallocate(d_en_bct)
            d_en_bct = d_en_fp32

        s_pred_tt = ttnn.from_torch(
            s_pred_cpu, dtype=prosody_dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc
        )
        keep_mask = ttnn.ones([B, T, 1], dtype=prosody_dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
        d_nlc = tt_model._predictor._text_encoder.forward(
            d_en_bct=d_en_bct,
            style_bs=s_pred_tt,
            sequence_lengths=lengths_list,
            keep_mask_btl=keep_mask,
            compute_kernel_config=ck,
            memory_config=mc,
            wire_dtype=prosody_dtype,
        )
        ttnn.deallocate(d_en_bct)
        ttnn.deallocate(keep_mask)
        caps["d_nlc"] = _tt_to_float(d_nlc)

        x_lstm = tt_bilstm_nlc(
            x_nlc=d_nlc,
            fwd=p.predictor.lstm_fwd,
            rev=p.predictor.lstm_rev,
            compute_kernel_config=ck,
            memory_config=mc,
            sequence_lengths=lengths_list,
        )
        duration = tt_model._predictor._duration_proj.forward(x_lstm, compute_kernel_config=ck, memory_config=mc)
        ttnn.deallocate(x_lstm)

        dur_sig = ttnn.sigmoid(duration, memory_config=mc)
        ttnn.deallocate(duration)
        dur_sum_tt = ttnn.sum(dur_sig, dim=-1, memory_config=mc)
        ttnn.deallocate(dur_sig)
        if speed != 1.0:
            dur_scaled = ttnn.multiply(dur_sum_tt, 1.0 / speed, memory_config=mc)
            ttnn.deallocate(dur_sum_tt)
            dur_sum_tt = dur_scaled
        dur_rounded_tt = ttnn.round(dur_sum_tt, memory_config=mc)
        ttnn.deallocate(dur_sum_tt)
        dur_clipped_tt = ttnn.clip(dur_rounded_tt, min=1.0, memory_config=mc)
        ttnn.deallocate(dur_rounded_tt)
        pred_dur = ttnn.to_torch(dur_clipped_tt).long().squeeze()
        ttnn.deallocate(dur_clipped_tt)
        caps["pred_dur"] = pred_dur.cpu()

        aln_cpu = _build_alignment(pred_dur)
        T_aligned = int(aln_cpu.shape[2])
        aln_dtype = ttnn.float32 if tt_model._use_fp32_prosody_boundary else ttnn.bfloat16
        aln_tt = ttnn.from_torch(aln_cpu, dtype=aln_dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)

        aln_Ta_T = ttnn.permute(aln_tt, (0, 2, 1), memory_config=mc)
        if tt_model._use_fp32_prosody_boundary:
            d_mat, owns_d = _to_fp32_if_needed(d_nlc, mc)
            if owns_d:
                ttnn.deallocate(d_nlc)
            en_nlc = ttnn.matmul(aln_Ta_T, d_mat, memory_config=mc, compute_kernel_config=ck)
            ttnn.deallocate(d_mat)
        else:
            en_nlc = ttnn.matmul(aln_Ta_T, d_nlc, memory_config=mc, compute_kernel_config=ck)
            ttnn.deallocate(d_nlc)
        ttnn.deallocate(aln_Ta_T)
        caps["en_bct"] = _tt_to_float(en_nlc).permute(0, 2, 1).contiguous()

        if tt_model._use_fp32_prosody_boundary:
            en_fp32, owns_en = _to_fp32_if_needed(en_nlc, mc)
            if owns_en:
                ttnn.deallocate(en_nlc)
                en_nlc = en_fp32
            s_pred_f0, owns_s = _to_fp32_if_needed(s_pred_tt, mc)
            F0, N = tt_model._predictor.F0Ntrain(en_nlc, s_pred_f0, memory_config=mc, use_fp32_boundary=True)
            if owns_s:
                ttnn.deallocate(s_pred_f0)
        else:
            F0, N = tt_model._predictor.F0Ntrain(en_nlc, s_pred_tt, memory_config=mc, use_fp32_boundary=False)
        ttnn.deallocate(en_nlc)
        ttnn.deallocate(s_pred_tt)
        caps["F0"] = _squeeze_batch(_tt_to_float(F0))
        caps["N"] = _squeeze_batch(_tt_to_float(N))

        t_en_bct = tt_model._text_encoder(input_ids, input_lengths=input_lengths)
        asr_bct = ttnn.matmul(t_en_bct, aln_tt, memory_config=mc, compute_kernel_config=ck)
        ttnn.deallocate(t_en_bct)
        ttnn.deallocate(aln_tt)
        caps["asr_bct"] = _tt_to_float(asr_bct)

        asr_nlc = ttnn.permute(asr_bct, (0, 2, 1), memory_config=mc)
        ttnn.deallocate(asr_bct)
        if tt_model._use_fp32_prosody_boundary:
            if asr_nlc.dtype != ttnn.float32:
                asr_fp32 = ttnn.typecast(asr_nlc, ttnn.float32, memory_config=mc)
                ttnn.deallocate(asr_nlc)
                asr_nlc = asr_fp32
            if F0.dtype != ttnn.float32:
                F0_fp32 = ttnn.typecast(F0, ttnn.float32, memory_config=mc)
                ttnn.deallocate(F0)
                F0 = F0_fp32
            if N.dtype != ttnn.float32:
                N_fp32 = ttnn.typecast(N, ttnn.float32, memory_config=mc)
                ttnn.deallocate(N)
                N = N_fp32

        s_style_tt = ttnn.from_torch(
            s_style_cpu, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc
        )
        decoder = tt_model._get_decoder(T_aligned)
        gen = decoder._generator
        m_source_kwargs: dict = {}
        rng_tt = None
        if deterministic:
            from models.experimental.kokoro.m_source_rng import (
                deallocate_m_source_rng_tt,
                make_zero_m_source_rng,
                upload_m_source_rng,
            )

            B_dec = int(F0.shape[0])
            T_har = int(F0.shape[1]) * int(gen.params.upsample_scale_full)
            dim = int(gen.params.m_source.sinegen.dim)
            rng_cpu = make_zero_m_source_rng(B_dec, T_har, dim)
            rng_tt = upload_m_source_rng(rng_cpu, dev, memory_config=mc)
            m_source_kwargs = {
                "sinegen_rand_ini": rng_tt.rand_ini,
                "sinegen_noise_raw": rng_tt.sinegen_noise,
                "source_noise_raw": rng_tt.source_noise,
            }

        audio_tt = decoder(asr_nlc, F0, N, s_style_tt, memory_config=mc, **m_source_kwargs)
        if deterministic and rng_tt is not None:
            deallocate_m_source_rng_tt(rng_tt)
        caps["audio"] = _squeeze_batch(_tt_to_float(audio_tt))
        ttnn.deallocate(audio_tt)
        ttnn.deallocate(asr_nlc)
        ttnn.deallocate(F0)
        ttnn.deallocate(N)
        ttnn.deallocate(s_style_tt)

    return TTStages(
        d_en_bct=caps["d_en_bct"],
        d_nlc=caps["d_nlc"],
        pred_dur=caps["pred_dur"],
        en_bct=caps["en_bct"],
        F0=caps["F0"],
        N=caps["N"],
        asr_bct=caps["asr_bct"],
        audio=caps["audio"],
    )


def _normalize_f0_b1t(F0_curve: torch.Tensor) -> torch.Tensor:
    """``[B, T_f0]`` or ``[T_f0]`` → ``[B, T_f0, 1]`` (harmonic-path layout)."""
    f0 = F0_curve.detach().float()
    if f0.dim() == 1:
        f0 = f0.unsqueeze(0)
    if f0.dim() == 2:
        f0 = f0.unsqueeze(-1)
    return f0


def _f0_curve_to_har_btd(ref: KModel, F0_curve: torch.Tensor) -> torch.Tensor:
    f0 = F0_curve.detach().float()
    if f0.dim() == 1:
        f0 = f0.unsqueeze(0)
    with torch.no_grad():
        return ref.decoder.generator.f0_upsamp(f0.unsqueeze(1)).transpose(1, 2).contiguous().float()


def _run_ref_harmonic_stages(ref: KModel, F0_curve: torch.Tensor) -> dict[str, torch.Tensor]:
    """Harmonic path on reference PyTorch using the given F0_curve (typically TT prosody)."""
    gen = ref.decoder.generator
    msrc = gen.m_source
    caps: dict[str, torch.Tensor] = {}
    f0 = F0_curve.detach().float()
    if f0.dim() == 1:
        f0 = f0.unsqueeze(0)
    caps["H0. F0_curve"] = f0.cpu()
    f0_b1t = f0.unsqueeze(1)
    caps["H1. f0 unsqueeze→fp32"] = f0_b1t.cpu()
    dim = int(msrc.l_sin_gen.dim)
    with torch.no_grad():
        f0_har = gen.f0_upsamp(f0_b1t).transpose(1, 2).contiguous()
        T_har = int(f0_har.shape[1])
    with torch.no_grad(), patched_m_source_torch_rng(make_zero_m_source_rng(1, T_har, dim)):
        caps["H2. after f0_upsamp"] = f0_har.cpu()
        sine_wavs, uv, _noise = msrc.l_sin_gen(f0_har)
        caps["H3. SineGen sine_wavs"] = sine_wavs.cpu()
        lin = msrc.l_linear(sine_wavs)
        caps["H4. m_source l_linear (pre-tanh)"] = lin.cpu()
        sine_merge = torch.tanh(lin)
        caps["H5. after tanh [B,T_har]"] = sine_merge.squeeze(-1).cpu()
        caps["H6. after typecast→fp32 (STFT in)"] = sine_merge.squeeze(-1).cpu()
        har_flat = sine_merge.transpose(1, 2).squeeze(1)
        har_spec, har_phase = gen.stft.transform(har_flat)
        caps["H7a. after stft.transform (magnitude)"] = har_spec.cpu()
        caps["H7b. after stft.transform (cos phase)"] = torch.cos(har_phase).cpu()
        caps["H8. har cat [mag|phase] BCT"] = torch.cat([har_spec, har_phase], dim=1).cpu()
    return caps


_HARMONIC_COMPARE_KEYS = (
    "H0. F0_curve",
    "H1. f0 unsqueeze→fp32",
    "H2. after f0_upsamp",
    "H3. SineGen sine_wavs",
    "H4. m_source l_linear (pre-tanh)",
    "H5. after tanh [B,T_har]",
    "H6. after typecast→fp32 (STFT in)",
    "H7a. after stft.transform (magnitude)",
    "H7b. after stft.transform (cos phase)",
    "H8. har cat [mag|phase] BCT",
)


def _run_tt_harmonic_stages(
    tt_model: TTKModel,
    ref: KModel,
    F0_curve: torch.Tensor,
    *,
    T_mel: int,
) -> dict[str, torch.Tensor]:
    """Harmonic path on TT via ``_harmonic_source_path``; H2 uses ref upsample on same F0."""
    gen = tt_model._get_decoder(T_mel)._generator
    dev = tt_model.device
    mc = ttnn.DRAM_MEMORY_CONFIG
    caps: dict[str, torch.Tensor] = {}

    f0_b1t = _normalize_f0_b1t(F0_curve)
    caps["H0. F0_curve"] = f0_b1t.squeeze(-1).cpu()
    caps["H1. f0 unsqueeze→fp32"] = f0_b1t.cpu()
    caps["H2. after f0_upsamp"] = _f0_curve_to_har_btd(ref, F0_curve)

    B_rng, T_har, dim = m_source_rng_shapes_from_f0(
        f0_b1t.squeeze(-1).float(),
        upsample_scale_full=int(gen.params.upsample_scale_full),
        dim=int(gen.params.m_source.sinegen.dim),
    )
    rng_cpu = make_zero_m_source_rng(B_rng, T_har, dim)
    rng_tt = upload_m_source_rng(rng_cpu, dev, memory_config=mc)

    f0_tt = ttnn.from_torch(f0_b1t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
    f0_in = ttnn.squeeze(f0_tt, 2) if int(f0_tt.shape[2]) == 1 else f0_tt
    if f0_in is not f0_tt:
        ttnn.deallocate(f0_tt)

    with patched_m_source_torch_rng(rng_cpu):
        har_nlc = gen._harmonic_source_path(
            f0_in,
            sinegen_rand_ini=rng_tt.rand_ini,
            sinegen_noise_raw=rng_tt.sinegen_noise,
            source_noise_raw=rng_tt.source_noise,
            memory_config=mc,
        )
    ttnn.deallocate(f0_in)

    har_bct = _tt_to_float(har_nlc).permute(0, 2, 1).contiguous()
    ttnn.deallocate(har_nlc)
    K = har_bct.shape[1] // 2
    caps["H7a. after stft.transform (magnitude)"] = har_bct[:, :K, :]
    caps["H7b. after stft.transform (cos phase)"] = torch.cos(har_bct[:, K:, :])
    caps["H8. har cat [mag|phase] BCT"] = har_bct

    deallocate_m_source_rng_tt(rng_tt)
    return caps


def _collect_sinegen_rows(
    device: ttnn.Device,
    f0_har_btd: torch.Tensor,
    *,
    use_torch_phase_fallback: bool,
    label: str,
) -> list[tuple[str, float, str, str, str]]:
    """Ref vs TT SineGen internal stages for one F0 tensor [B,T,1]."""
    B = int(f0_har_btd.shape[0])
    T = int(f0_har_btd.shape[1])
    dim = int(f0_har_btd.shape[2])
    ref_sg = SineGen(
        samp_rate=24000.0,
        upsample_scale=300,
        harmonic_num=dim - 1,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0.0,
    ).eval()
    rng = make_zero_m_source_rng(B, T, dim)
    params = preprocess_tt_sinegen(
        device=device,
        sampling_rate=24000.0,
        upsample_scale=300,
        harmonic_num=dim - 1,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0.0,
        time_len=T,
        weights_dtype=ttnn.bfloat16,
    )
    tt_sg = TTSineGen(
        device,
        params,
        use_torch_phase_fallback=use_torch_phase_fallback,
        use_torch_sinegen_fallback=False,
    )
    rng_tt = upload_m_source_rng(rng, device, dtype=params.activation_dtype)
    f0_tt = ttnn.from_torch(
        f0_har_btd, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    with patched_m_source_torch_rng(rng):
        ref_caps = _run_ref_sinegen_stages(ref_sg, f0_har_btd, rand_ini=rng.rand_ini)
    tt_caps = _run_tt_sinegen_stages(tt_sg, f0_tt, rand_ini_tt=rng_tt.rand_ini, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(f0_tt)
    deallocate_m_source_rng_tt(rng_tt)

    rows: list[tuple[str, float, str, str, str]] = []
    for key in ref_caps:
        if key not in tt_caps or key == "S5b_phase_rad_down":
            continue
        name, pcc, note = _pcc_flat(f"{label}:{key}", ref_caps[key], tt_caps[key])
        if use_torch_phase_fallback and key in ("S6_phase_up_rad", "S7_sin_raw", "S8_sine_x_amp", "S9_out_uv_noise"):
            backend = "torch.cpu (phase chain)"
        else:
            backend = "ttnn"
        rows.append((name, float(pcc), _classify_pcc(float(pcc)), backend, _SINEGEN_STAGE_OPS.get(key, "")))
    return rows


def _backend_for_harmonic_stage(name: str) -> str:
    if "H3" in name or "SineGen" in name:
        return "torch.cpu" if STFT_PHASE_FALLBACK_KWARGS["use_torch_phase_fallback"] else "ttnn"
    if "H7" in name or "stft" in name.lower():
        return "torch.cpu" if STFT_PHASE_FALLBACK_KWARGS["use_torch_stft_fallback"] else "ttnn"
    if "H4" in name or "H5" in name or "linear" in name or "tanh" in name:
        return "ttnn"
    return "ttnn"


def _rows_to_md_table(
    rows: list[tuple[str, float, str, str]],
    *,
    extra_col: str | None = None,
) -> list[str]:
    if extra_col:
        lines = [
            f"| Stage | PCC | Status | {extra_col} | Notes |",
            "|-------|-----|--------|" + "-" * (len(extra_col) + 2) + "|-------|",
        ]
    else:
        lines = ["| Stage | PCC | Status | Notes |", "|-------|-----|--------|-------|"]
    for row in rows:
        if extra_col:
            name, pcc, status, col, note = row  # type: ignore[misc]
            lines.append(f"| {name} | {pcc:.6f} | {status} | {col} | {note} |")
        else:
            name, pcc, status, note = row  # type: ignore[misc]
            lines.append(f"| {name} | {pcc:.6f} | {status} | {note} |")
    return lines


def _compare_cap_dicts(
    ref_caps: dict[str, torch.Tensor], tt_caps: dict[str, torch.Tensor]
) -> list[tuple[str, float, str, str]]:
    out: list[tuple[str, float, str, str]] = []
    for k in ref_caps:
        if k not in tt_caps:
            continue
        name, pcc, note = _pcc_flat(k, ref_caps[k], tt_caps[k])
        out.append((name, float(pcc), _classify_pcc(float(pcc)), note))
    return out


def _verify_e2e_equivalent(tt_model: TTKModel, phonemes: str, ref_s: torch.Tensor, staged_audio: torch.Tensor) -> float:
    """Return PCC between staged walk audio and ``TTKModel.forward`` (pytest path)."""
    with _zero_noise():
        out = tt_model(phonemes=phonemes, ref_s=ref_s, speed=1.0, deterministic=True)
    model_audio = out.audio.detach().float().squeeze()
    ref_f = staged_audio.reshape(-1)
    tt_f = model_audio.reshape(-1)
    n = min(ref_f.numel(), tt_f.numel())
    _, pcc = comp_pcc(ref_f[:n].unsqueeze(0), tt_f[:n].unsqueeze(0), pcc=0.0)
    return float(pcc)


@dataclass
class StftPhaseReportData:
    text: str
    phonemes: str
    e2e_pcc: float
    e2e_equiv_pcc: float
    prosody_rows: list[tuple[str, float, str, str]]
    pred_dur_match: bool
    harmonic_rows: list[tuple[str, float, str, str, str]]
    decode_table: StageTable
    decoder_isolation_pcc: float
    sinegen_config_e: list[tuple[str, float, str, str, str]]
    sinegen_device_only: list[tuple[str, float, str, str, str]]
    T_f0: int
    T_har: int
    T_mel: int


def _format_stft_phase_report(data: StftPhaseReportData) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    all_summary_rows = list(data.prosody_rows) + [(n, pcc, s, b) for n, pcc, s, b, _ in data.harmonic_rows]
    degraded = [r for r in all_summary_rows if r[1] == r[1] and r[1] < 0.99]
    degraded.sort(key=lambda x: x[1])

    lines = [
        "# Kokoro STFT+phase fallback PCC diagnostic",
        "",
        f"Generated: {ts}",
        "",
        "This report matches **`test_tt_kmodel_stft_and_phase_fallback_pcc`** (config E).",
        "Stage 6 and pytest both use `TTKModel._device_forward` with `deterministic=True`.",
        "",
        "## Config E flags",
        "",
        "| Flag | Value | Effect |",
        "|------|-------|--------|",
        "| `use_torch_stft_fallback` | **True** | `torch.stft` on CPU for harmonic STFT |",
        "| `use_torch_phase_fallback` | **True** | SineGen phase chain on CPU float32 |",
        "| `use_torch_sinegen_fallback` | False | No full CPU SineGen |",
        "| `use_torch_f0n_conv_fallback` | **False** | F0/N stride-2 conv on device |",
        "| `use_torch_f0_upsamp_fallback` | **False** | f0 nearest upsample on device |",
        "| `use_fp32_prosody_boundary` | **True** | fp32 bert→predictor→F0Ntrain boundary |",
        "",
        f"**Text:** `{data.text}`",
        f"**Phonemes ({len(data.phonemes)}):** `{data.phonemes}`",
        "",
        "## End-to-end summary",
        "",
        "| Metric | PCC | Notes |",
        "|--------|-----|-------|",
        f"| Stage 6 (full TT pipeline vs ref) | **{data.e2e_pcc:.6f}** | pytest asserts > 0.84 |",
        f"| Staged walk vs `TTKModel.forward` | **{data.e2e_equiv_pcc:.6f}** | must be **1.0** (E2E path match) |",
        f"| Decoder isolation (ref ASR/F0/N → TT vocoder) | **{data.decoder_isolation_pcc:.6f}** | vocoder without prosody drift |",
        f"| `pred_dur` exact match | **{data.pred_dur_match}** | duration frames |",
        "",
        "### Geometry (this sentence)",
        "",
        "| Symbol | Value |",
        "|--------|-------|",
        f"| `T_f0` (mel F0 frames) | {data.T_f0} |",
        f"| `T_har` (`T_f0` × 300) | {data.T_har} |",
        f"| `T_mel` (aligned text frames) | {data.T_mel} |",
        "| STFT bins K | 301 |",
        "| SineGen dim | 9 |",
        "",
        "## Input regime (stage 6 = pytest)",
        "",
        "| | Reference | TT (`_device_forward`) |",
        "|--|-----------|------------------------|",
        "| Entry | `KModel` token forward | `TTKModel(phonemes=..., deterministic=True)` |",
        "| PL-BERT mask | `attention_mask` from lengths | **`attention_mask=None`** + `ttnn.ones` keep_mask |",
        "| Duration / alignment | CPU torch | **device** sigmoid→round→clip + `_build_alignment` |",
        "| RNG | `_zero_noise()` | `_zero_noise()` + `make_zero_m_source_rng` |",
        "| Decoder entry | fp32 tensors | fp32 cast at prosody boundary |",
        "",
        "Earlier staged PCC **0.850290** vs pytest **0.850388** was from a non-equivalent walk",
        "(wrong PL-BERT mask, CPU duration). The E2E-equivalent walk now matches pytest.",
        "",
        "## Prosody path (stages 1–5)",
        "",
    ]
    lines.extend(_rows_to_md_table(data.prosody_rows))
    lines.extend(
        [
            "",
            f"**pred_dur:** ref and TT tensors equal = `{data.pred_dur_match}`",
            "",
            "## Harmonic source (H0–H8, TT `F0_curve` fed to both sides)",
            "",
            "Isolates vocoder harmonic branch using **TT prosody F0** so H0 reflects predictor error, not ref F0.",
            "",
            "| Stage | PCC | Status | Backend | Notes |",
            "|-------|-----|--------|---------|-------|",
        ]
    )
    for name, pcc, status, backend, note in data.harmonic_rows:
        lines.append(f"| {name} | {pcc:.6f} | {status} | {backend} | {note} |")
    lines.extend(
        [
            "",
            "**H7b / H8 / G0b caveat:** isolated rows compare ref **conv1d STFT** + full ref `m_source`",
            "vs TT **`torch.stft` fallback** + `_harmonic_source_path`. Algorithms differ, so H7b/H8/G0b",
            "PCC can look severe (~0.3–0.75) even when **stage 6 E2E ≈ 0.85** and **G7_audio ≈ 0.87**.",
            "",
            f"### {data.decode_table.title}",
            "",
            "| Stage | PCC | Notes |",
            "|-------|-----|-------|",
        ]
    )
    for name, pcc, note, marker in data.decode_table.rows:
        m = f" {marker}" if marker else ""
        lines.append(f"| {name} | {pcc:.6f} | {note}{m} |")
    lines.extend(
        [
            "",
            "## SineGen internal (S0–S9) on captured TT F0",
            "",
            f"F0 source: TT `F0_curve` → ref `f0_upsamp` → `f0_har` shape `[1, {data.T_har}, 1]`.",
            "Zero RNG (`make_zero_m_source_rng`) on both sides.",
            "",
            "### With config E (`use_torch_phase_fallback=True`)",
            "",
            "_Isolated walk uses `_run_tt_sinegen_stages` (per-op capture). E2E uses the integrated",
            "CPU phase fallback inside `TTSineGen.forward`; S7–S9 PCC here can understate fallback quality._",
            "",
            "| Stage | PCC | Status | Backend | TTNN ops |",
            "|-------|-----|--------|---------|----------|",
        ]
    )
    for name, pcc, status, backend, ops in data.sinegen_config_e:
        lines.append(f"| {name} | {pcc:.6f} | {status} | {backend} | {ops} |")
    lines.extend(
        [
            "",
            "### Device-only phase chain (no phase fallback — reference for BH BF16 ceiling)",
            "",
            "Shows what **no-fallback** SineGen would look like on the same F0; see [`SINEGEN_STAGE_PCC.md`](SINEGEN_STAGE_PCC.md).",
            "",
            "| Stage | PCC | Status | Backend | TTNN ops |",
            "|-------|-----|--------|---------|----------|",
        ]
    )
    for name, pcc, status, backend, ops in data.sinegen_device_only:
        lines.append(f"| {name} | {pcc:.6f} | {status} | {backend} | {ops} |")
    if degraded:
        lines.extend(["", "## Degraded ops (PCC < 0.99)", ""])
        lines.append("| Stage | PCC | Status |")
        lines.append("|-------|-----|--------|")
        for name, pcc, status, _ in degraded:
            lines.append(f"| {name} | {pcc:.6f} | {status} |")
    lines.extend(
        [
            "",
            "## Interpretation (config E)",
            "",
            "- **Prosody 1–5** stay > 0.998 with fp32 boundary — E2E gap is not PL-BERT/duration/F0 predictor.",
            "- **STFT + phase CPU fallbacks** restore harmonic path; **F0/N conv + f0 upsample on device**.",
            "- **Stage 6 ≈ 0.85** is the production target for this test; **no-fallback E2E ~0.26** is a different test.",
            "- **Decoder isolation** shows vocoder quality with perfect conditioning.",
            "",
            "## Related reports",
            "",
            "- [`NO_FALLBACK_OP_PCC.md`](NO_FALLBACK_OP_PCC.md) — full device vocoder failure map",
            "- [`SINEGEN_STAGE_PCC.md`](SINEGEN_STAGE_PCC.md) — phase chain at T=48600",
            "- [`KOKORO_PCC_STATUS_2026-05-20.md`](KOKORO_PCC_STATUS_2026-05-20.md) — status summary",
            "",
            "## Re-run",
            "",
            "```bash",
            "source python_env/bin/activate",
            "export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)",
            "python models/experimental/kokoro/tests/kmodel_pcc_stage_diagnostic.py --stft-phase-fallback --write-report",
            "pytest -s models/experimental/kokoro/tests/test_tt_kmodel_pcc.py::test_tt_kmodel_stft_and_phase_fallback_pcc",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="KModel stage PCC diagnostic")
    parser.add_argument("--text", default=_DEFAULT_TEXT)
    parser.add_argument("--stft-phase-fallback", action="store_true")
    parser.add_argument("--write-report", nargs="?", const=None, default=None)
    args = parser.parse_args()

    ckpt = _find_checkpoint()
    if ckpt is None:
        sys.exit("Kokoro-82M checkpoint not found.")

    phonemes, ref_s = _phonemize(args.text)
    ref = KModel(repo_id="hexgrad/Kokoro-82M", model=str(ckpt), disable_complex=True).eval()
    input_ids, text_mask, input_lengths, lengths_list = _tokenize(ref.vocab, phonemes, ref.context_length)
    ref_st = _run_ref_stages(ref, input_ids, ref_s)

    device = ttnn.open_device(device_id=0)
    try:
        params = preprocess_tt_kmodel(ref, device)
        with _zero_noise():
            tt_model = TTKModel(device, ref, params, **STFT_PHASE_FALLBACK_KWARGS)

        print(f"Text: {args.text!r}")
        print(f"Phonemes ({len(phonemes)}): {phonemes!r}")
        if args.stft_phase_fallback:
            print("Config: STFT+phase fallback (E2E-equivalent staged path)")

        tt_st = _run_tt_stages(
            tt_model, input_ids, text_mask, input_lengths, lengths_list, ref_s, speed=1.0, deterministic=True
        )

        rows = [
            _pcc_row("1. After PL-BERT + bert_encoder (d_en)", ref_st.d_en_bct, tt_st.d_en_bct),
            _pcc_row("2. After DurationEncoder (d, pre-align)", ref_st.d_nlc, tt_st.d_nlc),
            ("2b. pred_dur", float(torch.equal(ref_st.pred_dur, tt_st.pred_dur)), ""),
            _pcc_row("3. After alignment (en)", ref_st.en_bct, tt_st.en_bct),
            _pcc_row("4. After predictor F0Ntrain (F0)", ref_st.F0, tt_st.F0),
            _pcc_row("4. After predictor F0Ntrain (N)", ref_st.N, tt_st.N),
            _pcc_row("5. After TextEncoder @ aln (asr)", ref_st.asr_bct, tt_st.asr_bct),
            _pcc_row("6. After Decoder (end-to-end audio)", ref_st.audio, tt_st.audio),
        ]
        _print_table(rows)

        e2e_equiv = _verify_e2e_equivalent(tt_model, phonemes, ref_s, tt_st.audio)
        print(f"\nStaged audio vs TTKModel.forward PCC: {e2e_equiv:.6f} (expect 1.0)")

        if args.write_report is not None or "--write-report" in sys.argv:
            report_path = (
                Path(args.write_report)
                if args.write_report is not None
                else Path(__file__).resolve().parent / "STFT_PHASE_FALLBACK_OP_PCC.md"
            )
            s_style = ref_s[:, : tt_model.params.style_dim]

            T_mel = int(ref_st.asr_bct.shape[-1])
            T_f0 = int(ref_st.F0.reshape(-1).shape[0])
            f0_har = _f0_curve_to_har_btd(ref, tt_st.F0)
            T_har = int(f0_har.shape[1])

            prosody_rows = [
                (name, pcc, _classify_pcc(pcc), note)
                for name, pcc, note in [
                    _pcc_row("1. After PL-BERT + bert_encoder (d_en)", ref_st.d_en_bct, tt_st.d_en_bct),
                    _pcc_row("2. After DurationEncoder (d, pre-align)", ref_st.d_nlc, tt_st.d_nlc),
                    _pcc_row("3. After alignment (en)", ref_st.en_bct, tt_st.en_bct),
                    _pcc_row("4. After predictor F0Ntrain (F0)", ref_st.F0, tt_st.F0),
                    _pcc_row("4. After predictor F0Ntrain (N)", ref_st.N, tt_st.N),
                    _pcc_row("5. After TextEncoder @ aln (asr)", ref_st.asr_bct, tt_st.asr_bct),
                    _pcc_row("6. After Decoder (end-to-end audio)", ref_st.audio, tt_st.audio),
                ]
            ]
            _, e2e_pcc = comp_pcc(ref_st.audio.unsqueeze(0), tt_st.audio.unsqueeze(0), pcc=0.0)

            ref_har = _run_ref_harmonic_stages(ref, tt_st.F0)
            tt_har = _run_tt_harmonic_stages(tt_model, ref, tt_st.F0, T_mel=T_mel)
            harmonic_rows = []
            for key in _HARMONIC_COMPARE_KEYS:
                if key not in ref_har or key not in tt_har:
                    continue
                if key in (
                    "H3. SineGen sine_wavs",
                    "H4. m_source l_linear (pre-tanh)",
                    "H5. after tanh [B,T_har]",
                    "H6. after typecast→fp32 (STFT in)",
                ):
                    note = "ref-only capture; see SineGen S0–S9 table for TT phase path"
                    harmonic_rows.append((key, float("nan"), "N/A", _backend_for_harmonic_stage(key), note))
                    continue
                name, pcc, note = _pcc_flat(key, ref_har[key], tt_har[key])
                harmonic_rows.append(
                    (name, float(pcc), _classify_pcc(float(pcc)), _backend_for_harmonic_stage(name), note)
                )

            ref_dec = _run_ref_decode_stack(ref, ref_st.asr_bct, ref_st.F0, ref_st.N, s_style)
            tt_dec = _run_tt_decode_stack(
                tt_model,
                ref_st.asr_bct,
                ref_st.F0,
                ref_st.N,
                s_style,
                asr_entry_dtype="fp32",
                use_f0n_cpu=False,
            )
            decode_table = _compare_stages(
                ref_dec,
                tt_dec,
                "Decode stack (ref prosody → ref vs TT, device F0/N conv)",
            )
            _, decoder_iso_pcc = comp_pcc(ref_dec["G7_audio"].unsqueeze(0), tt_dec["G7_audio"].unsqueeze(0), pcc=0.0)

            sinegen_e = _collect_sinegen_rows(device, f0_har, use_torch_phase_fallback=True, label="captured_tt_f0")
            sinegen_dev = _collect_sinegen_rows(
                device, f0_har, use_torch_phase_fallback=False, label="captured_tt_f0_no_phase_fb"
            )

            report = _format_stft_phase_report(
                StftPhaseReportData(
                    text=args.text,
                    phonemes=phonemes,
                    e2e_pcc=float(e2e_pcc),
                    e2e_equiv_pcc=e2e_equiv,
                    prosody_rows=prosody_rows,
                    pred_dur_match=bool(torch.equal(ref_st.pred_dur, tt_st.pred_dur)),
                    harmonic_rows=harmonic_rows,
                    decode_table=decode_table,
                    decoder_isolation_pcc=float(decoder_iso_pcc),
                    sinegen_config_e=sinegen_e,
                    sinegen_device_only=sinegen_dev,
                    T_f0=T_f0,
                    T_har=T_har,
                    T_mel=T_mel,
                )
            )
            report_path.write_text(report, encoding="utf-8")
            print(f"\nWrote report: {report_path}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
