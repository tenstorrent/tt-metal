# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Decode-stack PCC diagnostic for Kokoro KModel on Blackhole.

With baseline E (STFT + phase + sinegen) the full-pipeline audio PCC tops out at ~0.41 on
BH BF16 even with ref ASR/F0/N injected. **The remaining bottleneck is in the decode stack**:

    asr → F0_conv → N_conv → concat → encode → 4× decode → generator(ups/resblocks/noise_res)

This script captures PCC at every AdainResBlk boundary inside Decoder + Generator and
prints a table so we can identify *which* on-device op gives the biggest BF16 drift.
Run with the recommended fallback config (STFT + phase + sinegen) so we measure decode-
stack precision in isolation, not stacked on top of STFT noise.

Run (from repo root, with kokoro package + checkpoint + TT device):

    # Default: ref prosody inputs → ref vs TT decode-stack stages
    python models/experimental/kokoro/tests/kmodel_decode_stack_diagnostic.py

    # Full plan: ref vs TT prosody, bf16 ASR boundary, write DECODE_STACK_FINDINGS.md
    python models/experimental/kokoro/tests/kmodel_decode_stack_diagnostic.py --full-plan
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn.functional as F_torch
import ttnn

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference.model import KModel
from models.experimental.kokoro.tt.tt_conv import (
    tt_conv1d_nlc,
    tt_conv1d_nlc_cpu,
    tt_conv1d_stride2_k3_1ch_nlc,
    tt_conv_transpose1d_nlc,
)
from models.experimental.kokoro.tt.tt_decoder import _to_interleaved
from models.experimental.kokoro.tt.tt_generator import _reflection_pad_left_1_nlc
from models.experimental.kokoro.tt.tt_kmodel import (
    TTKModel,
    _build_alignment,
    _to_fp32_if_needed,
    preprocess_tt_kmodel,
)
from models.experimental.kokoro.tt.tt_lstm import tt_bilstm_nlc
from models.experimental.kokoro.m_source_rng import (
    deallocate_m_source_rng_tt,
    make_zero_m_source_rng,
    m_source_rng_shapes_from_f0,
    patched_m_source_torch_rng,
    upload_m_source_rng,
)

_TEST_TEXT = os.getenv("KOKORO_PCC_DEBUG_TEXT", "Hello from Tenstorrent.")
_VOICE = "af_heart"
_LANG_CODE = "a"

_CKPT_CANDIDATES = (
    Path("/home/ubuntu/ign-tt/kokoro/examples/checkpoints/kokoro-v1_0.pth"),
    Path.home() / ".cache/huggingface/hub/models--hexgrad--Kokoro-82M/snapshots",
)


@contextmanager
def _zero_noise():
    real_rand = torch.rand
    real_randn_like = torch.randn_like
    torch.rand = lambda *size, **kwargs: torch.zeros(*size, **kwargs)
    torch.randn_like = lambda t, **kwargs: torch.zeros_like(t, **kwargs)
    try:
        yield
    finally:
        torch.rand = real_rand
        torch.randn_like = real_randn_like


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
    return phonemes, ref_s.float().cpu().unsqueeze(0) if ref_s.dim() == 1 else ref_s.float().cpu()


def _ref_prosody(ref: KModel, phonemes: str, ref_s: torch.Tensor, speed: float = 1.0):
    """Compute reference ASR/F0/N for decoder input."""
    vocab = ref.vocab
    input_ids_list = list(filter(lambda i: i is not None, map(lambda p: vocab.get(p), phonemes)))
    input_ids = torch.LongTensor([[0, *input_ids_list, 0]]).to(ref.device)
    B, T = input_ids.shape
    input_lengths = torch.full((B,), T, dtype=torch.long, device=ref.device)
    text_mask = torch.arange(T, device=ref.device).unsqueeze(0).expand(B, -1).type_as(input_lengths)
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))

    with torch.no_grad(), _zero_noise():
        bert_dur = ref.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = ref.bert_encoder(bert_dur).transpose(-1, -2)
        s_pred = ref_s[:, 128:]
        s_style = ref_s[:, :128]
        d = ref.predictor.text_encoder(d_en, s_pred, input_lengths, text_mask)
        x_lstm, _ = ref.predictor.lstm(d)
        duration = ref.predictor.duration_proj(x_lstm)
        dur_sum = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(dur_sum).clamp(min=1).long().squeeze()
        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=ref.device), pred_dur)
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=ref.device)
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)
        en = d.transpose(-1, -2) @ pred_aln_trg
        F0, N = ref.predictor.F0Ntrain(en, s_pred)
        t_en = ref.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
    return asr.cpu(), F0.cpu(), N.cpu(), s_style.cpu(), pred_dur.cpu()


def _pcc_flat(name: str, ref: torch.Tensor, tt: torch.Tensor) -> tuple[str, float, str]:
    r = ref.detach().float().reshape(-1)
    t = tt.detach().float().reshape(-1)
    if r.numel() != t.numel():
        n = min(r.numel(), t.numel())
        r, t = r[:n], t[:n]
        note = f"trimmed to {n}"
    else:
        note = ""
    if r.numel() == 0:
        return name, float("nan"), "empty"
    _, pcc = comp_pcc(r.unsqueeze(0), t.unsqueeze(0), pcc=0.0)
    return name, pcc, note


def _tt_to_torch(t: ttnn.Tensor) -> torch.Tensor:
    out = ttnn.to_torch(t).float()
    while out.dim() > 3 and out.shape[0] == 1:
        out = out.squeeze(0)
    return out


def _run_ref_decode_stack(
    ref: KModel, asr: torch.Tensor, F0_curve: torch.Tensor, N_curve: torch.Tensor, s_style: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Walk the reference decoder + generator and capture intermediates."""
    dec = ref.decoder
    gen = dec.generator
    caps: dict[str, torch.Tensor] = {}

    with torch.no_grad(), _zero_noise():
        F0 = dec.F0_conv(F0_curve.unsqueeze(1))
        N_d = dec.N_conv(N_curve.unsqueeze(1))
        caps["D0_F0_conv"] = F0.cpu()
        caps["D1_N_conv"] = N_d.cpu()
        x = torch.cat([asr, F0, N_d], dim=1)
        caps["D2_cat_in"] = x.cpu()
        x = dec.encode(x, s_style)
        caps["D3_encode"] = x.cpu()
        asr_res = dec.asr_res(asr)
        caps["D4_asr_res"] = asr_res.cpu()
        res = True
        for i, block in enumerate(dec.decode):
            if res:
                x = torch.cat([x, asr_res, F0, N_d], dim=1)
            x = block(x, s_style)
            caps[f"D5+{i}_decode[{i}]"] = x.cpu()
            if block.upsample_type != "none":
                res = False
        caps["G0_gen_in"] = x.cpu()

        # Generator: harmonic source + ups stack + conv_post + istft
        f0_b1t = F0_curve.unsqueeze(1)
        har_source, _noi_source, _uv = gen.m_source(gen.f0_upsamp(f0_b1t).transpose(1, 2))
        har_source = har_source.transpose(1, 2).squeeze(1)
        har_spec, har_phase = gen.stft.transform(har_source)
        har = torch.cat([har_spec, har_phase], dim=1)
        caps["G0b_har"] = har.cpu()

        for i in range(gen.num_upsamples):
            x = F_torch.leaky_relu(x, negative_slope=0.1)
            caps[f"G{i+1}a_lrelu"] = x.cpu()
            x_source = gen.noise_convs[i](har)
            caps[f"G{i+1}b_noise_conv"] = x_source.cpu()
            x_source = gen.noise_res[i](x_source, s_style)
            caps[f"G{i+1}c_noise_res"] = x_source.cpu()
            x = gen.ups[i](x)
            caps[f"G{i+1}d_ups"] = x.cpu()
            if i == gen.num_upsamples - 1:
                x = gen.reflection_pad(x)
            x = x + x_source
            caps[f"G{i+1}e_add"] = x.cpu()
            xs = None
            for j in range(gen.num_kernels):
                r = gen.resblocks[i * gen.num_kernels + j](x, s_style)
                if xs is None:
                    xs = r
                else:
                    xs = xs + r
            x = xs / gen.num_kernels
            caps[f"G{i+1}f_resblk_mean"] = x.cpu()

        x = F_torch.leaky_relu(x)
        caps["G3_post_lrelu"] = x.cpu()
        x_post = gen.conv_post(x)
        caps["G4_conv_post"] = x_post.cpu()
        K = gen.post_n_fft // 2 + 1
        spec = torch.exp(x_post[:, :K, :])
        phase = torch.sin(x_post[:, K:, :])
        caps["G5_spec"] = spec.cpu()
        caps["G6_phase"] = phase.cpu()
        audio = gen.stft.inverse(spec, phase).squeeze()
        caps["G7_audio"] = audio.cpu().float()
    return caps


def _tokenize(vocab: dict, phonemes: str, context_length: int):
    input_ids_list = list(filter(lambda i: i is not None, map(lambda p: vocab.get(p), phonemes)))
    assert len(input_ids_list) + 2 <= context_length
    input_ids = torch.LongTensor([[0, *input_ids_list, 0]])
    B, T = input_ids.shape
    input_lengths = torch.full((B,), T, dtype=torch.long)
    text_mask = torch.arange(T).unsqueeze(0).expand(B, -1).type_as(input_lengths)
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))
    return input_ids, text_mask, input_lengths, input_lengths.tolist()


@dataclass
class ProsodyTensors:
    asr_bct: torch.Tensor
    F0_curve: torch.Tensor
    N_curve: torch.Tensor
    s_style: torch.Tensor
    T_mel: int


def _run_tt_prosody(
    tt_model: TTKModel,
    input_ids: torch.LongTensor,
    text_mask: torch.Tensor,
    input_lengths: torch.LongTensor,
    lengths_list: list[int],
    ref_s: torch.Tensor,
    speed: float = 1.0,
) -> ProsodyTensors:
    """TT prosody stages 1–9 only (ASR/F0/N), matching TTKModel.forward before decoder."""
    p = tt_model.params
    dev = tt_model.device
    mc = ttnn.DRAM_MEMORY_CONFIG
    ck = tt_model._predictor.compute_kernel_config
    s_pred_cpu = ref_s[:, p.style_dim :]
    s_style_cpu = ref_s[:, : p.style_dim]
    attention_mask = (~text_mask).int()

    with _zero_noise():
        bert_out = tt_model._bert(input_ids, attention_mask)
        d_en = ttnn.linear(
            bert_out,
            p.bert_encoder_w,
            bias=p.bert_encoder_b,
            transpose_b=True,
            memory_config=mc,
            compute_kernel_config=ck,
        )
        ttnn.deallocate(bert_out)
        while len(d_en.shape) > 3:
            d_en = ttnn.squeeze(d_en, 0)
        d_en_bct = ttnn.permute(d_en, (0, 2, 1), memory_config=mc)
        ttnn.deallocate(d_en)

        s_pred_tt = ttnn.from_torch(
            s_pred_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc
        )
        keep_mask = ttnn.from_torch(
            (~text_mask).to(torch.float32).unsqueeze(-1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=mc,
        )
        d_nlc_tt = tt_model._predictor._text_encoder.forward(
            d_en_bct=d_en_bct,
            style_bs=s_pred_tt,
            sequence_lengths=lengths_list,
            keep_mask_btl=keep_mask,
            compute_kernel_config=ck,
            memory_config=mc,
        )
        ttnn.deallocate(keep_mask)
        ttnn.deallocate(d_en_bct)

        x_lstm = tt_bilstm_nlc(
            x_nlc=d_nlc_tt,
            fwd=p.predictor.lstm_fwd,
            rev=p.predictor.lstm_rev,
            compute_kernel_config=ck,
            memory_config=mc,
            sequence_lengths=lengths_list,
        )
        duration_tt = tt_model._predictor._duration_proj.forward(x_lstm, compute_kernel_config=ck, memory_config=mc)
        ttnn.deallocate(x_lstm)
        dur_cpu = ttnn.to_torch(duration_tt).float()
        ttnn.deallocate(duration_tt)
        dur_sum = torch.sigmoid(dur_cpu).sum(dim=-1) / speed
        pred_dur = torch.round(dur_sum).clamp(min=1).long().squeeze()
        aln_cpu = _build_alignment(pred_dur)
        T_aligned = int(aln_cpu.shape[2])
        aln_tt = ttnn.from_torch(aln_cpu, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)

        d_nlc_fp32, owns_d = _to_fp32_if_needed(d_nlc_tt, mc)
        if owns_d:
            ttnn.deallocate(d_nlc_tt)
        aln_Ta_T = ttnn.permute(aln_tt, (0, 2, 1), memory_config=mc)
        en_nlc_tt = ttnn.matmul(aln_Ta_T, d_nlc_fp32, memory_config=mc, compute_kernel_config=ck)
        ttnn.deallocate(aln_Ta_T)
        ttnn.deallocate(d_nlc_fp32)

        s_pred_f0, owns_s = _to_fp32_if_needed(s_pred_tt, mc)
        F0_tt, N_tt = tt_model._predictor.F0Ntrain(en_nlc_tt, s_pred_f0, memory_config=mc)
        if owns_s:
            ttnn.deallocate(s_pred_f0)
        ttnn.deallocate(en_nlc_tt)

        t_en_bct = tt_model._text_encoder(input_ids, input_lengths=input_lengths, text_mask=text_mask)
        asr_bct_tt = ttnn.matmul(t_en_bct, aln_tt, memory_config=mc, compute_kernel_config=ck)
        ttnn.deallocate(t_en_bct)
        ttnn.deallocate(s_pred_tt)
        ttnn.deallocate(aln_tt)

    asr_bct = _tt_to_torch(asr_bct_tt)
    while asr_bct.dim() > 3:
        asr_bct = asr_bct.squeeze(0)
    F0_curve = _tt_to_torch(F0_tt)
    N_curve = _tt_to_torch(N_tt)
    while F0_curve.dim() > 2:
        F0_curve = F0_curve.squeeze(0)
    while N_curve.dim() > 2:
        N_curve = N_curve.squeeze(0)
    ttnn.deallocate(asr_bct_tt)
    ttnn.deallocate(F0_tt)
    ttnn.deallocate(N_tt)
    return ProsodyTensors(
        asr_bct=asr_bct.cpu(),
        F0_curve=F0_curve.cpu(),
        N_curve=N_curve.cpu(),
        s_style=s_style_cpu.cpu(),
        T_mel=T_aligned,
    )


def _run_tt_decode_stack(
    tt_model: TTKModel,
    asr: torch.Tensor,
    F0_curve: torch.Tensor,
    N_curve: torch.Tensor,
    s_style: torch.Tensor,
    *,
    asr_entry_dtype: Literal["fp32", "bf16"] = "fp32",
    use_f0n_cpu: bool = True,
) -> dict[str, torch.Tensor]:
    """Walk TTDecoder + TTGenerator forward, capturing intermediates."""
    T_mel = int(asr.shape[-1])
    dec = tt_model._get_decoder(T_mel)
    gen = dec._generator
    p = gen.params
    dev = tt_model.device
    mc = ttnn.DRAM_MEMORY_CONFIG
    ck = gen.compute_kernel_config

    asr_nlc_cpu = asr.permute(0, 2, 1).contiguous()
    caps: dict[str, torch.Tensor] = {}

    B_rng, T_har, dim = m_source_rng_shapes_from_f0(
        F0_curve.float(),
        upsample_scale_full=int(p.upsample_scale_full),
        dim=int(p.m_source.sinegen.dim),
    )
    rng_cpu = make_zero_m_source_rng(B_rng, T_har, dim)
    rng_tt = upload_m_source_rng(rng_cpu, dev, memory_config=mc)

    with patched_m_source_torch_rng(rng_cpu):
        asr_dtype = ttnn.bfloat16 if asr_entry_dtype == "bf16" else ttnn.float32
        asr_tt = ttnn.from_torch(asr_nlc_cpu, dtype=asr_dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
        if asr_entry_dtype == "bf16":
            asr_fp32 = ttnn.typecast(asr_tt, ttnn.float32, memory_config=mc)
            ttnn.deallocate(asr_tt)
            asr_tt = asr_fp32
        F0_tt = ttnn.from_torch(
            F0_curve.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc
        )
        N_tt = ttnn.from_torch(
            N_curve.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc
        )
        s_tt = ttnn.from_torch(
            s_style.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc
        )

        # Decoder forward (mirroring TTDecoder.forward)
        f0_nlc = ttnn.unsqueeze(F0_tt, 2)
        n_nlc = ttnn.unsqueeze(N_tt, 2)
        _f0n_conv = tt_conv1d_nlc_cpu if use_f0n_cpu else tt_conv1d_stride2_k3_1ch_nlc
        F0_down = _to_interleaved(_f0n_conv(x_nlc=f0_nlc, params=dec.params.F0_conv, device=dev, memory_config=mc), mc)
        N_down = _to_interleaved(_f0n_conv(x_nlc=n_nlc, params=dec.params.N_conv, device=dev, memory_config=mc), mc)
        caps["D0_F0_conv"] = (
            _tt_to_torch(F0_down).permute(0, 2, 1).contiguous()
            if _tt_to_torch(F0_down).dim() == 3
            else _tt_to_torch(F0_down)
        )
        caps["D1_N_conv"] = (
            _tt_to_torch(N_down).permute(0, 2, 1).contiguous()
            if _tt_to_torch(N_down).dim() == 3
            else _tt_to_torch(N_down)
        )
        ttnn.deallocate(f0_nlc)
        ttnn.deallocate(n_nlc)

        target_dtype = ttnn.float32
        x = ttnn.concat([asr_tt, F0_down, N_down], dim=2, memory_config=mc)
        caps["D2_cat_in"] = _tt_to_torch(x).permute(0, 2, 1).contiguous()
        x = dec._encode.forward(x, s_tt, memory_config=mc)
        if x.dtype != target_dtype:
            x = ttnn.typecast(x, target_dtype, memory_config=mc)
        caps["D3_encode"] = _tt_to_torch(x).permute(0, 2, 1).contiguous()

        asr_res = _to_interleaved(
            tt_conv1d_nlc(
                x_nlc=asr_tt,
                params=dec.params.asr_res,
                device=dev,
                compute_config=ck,
                memory_config=mc,
                preserve_input_dtype=True,
            ),
            mc,
        )
        caps["D4_asr_res"] = _tt_to_torch(asr_res).permute(0, 2, 1).contiguous()
        res = True
        for i, blk in enumerate(dec._decode):
            if res:
                x_cat = ttnn.concat([x, asr_res, F0_down, N_down], dim=2, memory_config=mc)
                ttnn.deallocate(x)
                x = x_cat
            x = blk.forward(x, s_tt, memory_config=mc)
            if x.dtype != target_dtype:
                x_cast = ttnn.typecast(x, target_dtype, memory_config=mc)
                ttnn.deallocate(x)
                x = x_cast
            caps[f"D5+{i}_decode[{i}]"] = _tt_to_torch(x).permute(0, 2, 1).contiguous()
            if blk._params.layer_type != "none":
                res = False
        ttnn.deallocate(asr_res)
        ttnn.deallocate(F0_down)
        ttnn.deallocate(N_down)
        ttnn.deallocate(asr_tt)

        caps["G0_gen_in"] = _tt_to_torch(x).permute(0, 2, 1).contiguous()

        # Harmonic source path (use generator's _harmonic_source_path)
        har_nlc = gen._harmonic_source_path(
            F0_tt,
            sinegen_rand_ini=rng_tt.rand_ini,
            sinegen_noise_raw=rng_tt.sinegen_noise,
            source_noise_raw=rng_tt.source_noise,
            memory_config=mc,
        )
        caps["G0b_har"] = _tt_to_torch(har_nlc).permute(0, 2, 1).contiguous()

        # Generator ups + resblocks + conv_post + istft
        har_dtype = har_nlc.dtype
        if x.dtype != har_dtype:
            x_cast = ttnn.typecast(x, har_dtype, memory_config=mc)
            ttnn.deallocate(x)
            x = x_cast
        for i, stage in enumerate(p.stages):
            x_act = ttnn.leaky_relu(x, negative_slope=0.1, memory_config=mc)
            if x_act is not x:
                ttnn.deallocate(x)
            x = x_act
            caps[f"G{i+1}a_lrelu"] = _tt_to_torch(x).permute(0, 2, 1).contiguous()
            x_source = tt_conv1d_nlc(
                x_nlc=har_nlc,
                params=stage.noise_conv,
                device=dev,
                compute_config=ck,
                memory_config=mc,
                preserve_input_dtype=True,
            )
            caps[f"G{i+1}b_noise_conv"] = _tt_to_torch(x_source).permute(0, 2, 1).contiguous()
            x_source = gen._noise_res[i].forward(x_source, s_tt, memory_config=mc)
            caps[f"G{i+1}c_noise_res"] = _tt_to_torch(x_source).permute(0, 2, 1).contiguous()
            x_up = tt_conv_transpose1d_nlc(x_nlc=x, params=stage.ups, device=dev, compute_config=ck, memory_config=mc)
            ttnn.deallocate(x)
            x = x_up
            caps[f"G{i+1}d_ups"] = _tt_to_torch(x).permute(0, 2, 1).contiguous()
            if i == p.num_upsamples - 1:
                x_pad = _reflection_pad_left_1_nlc(x, memory_config=mc)
                ttnn.deallocate(x)
                x = x_pad
            x_sum = ttnn.add(x, x_source, memory_config=mc)
            ttnn.deallocate(x)
            ttnn.deallocate(x_source)
            x = x_sum
            caps[f"G{i+1}e_add"] = _tt_to_torch(x).permute(0, 2, 1).contiguous()
            xs: Optional[ttnn.Tensor] = None
            for resblk in gen._resblocks[i]:
                r = resblk.forward(x, s_tt, memory_config=mc)
                if xs is None:
                    xs = r
                else:
                    new_xs = ttnn.add(xs, r, memory_config=mc)
                    ttnn.deallocate(xs)
                    ttnn.deallocate(r)
                    xs = new_xs
            ttnn.deallocate(x)
            x = ttnn.multiply(xs, 1.0 / p.num_kernels, memory_config=mc)
            ttnn.deallocate(xs)
            caps[f"G{i+1}f_resblk_mean"] = _tt_to_torch(x).permute(0, 2, 1).contiguous()

        ttnn.deallocate(har_nlc)
        x_act = ttnn.leaky_relu(x, negative_slope=0.01, memory_config=mc)
        ttnn.deallocate(x)
        x = x_act
        caps["G3_post_lrelu"] = _tt_to_torch(x).permute(0, 2, 1).contiguous()
        x_post = tt_conv1d_nlc(
            x_nlc=x,
            params=p.conv_post,
            device=dev,
            compute_config=ck,
            memory_config=mc,
            preserve_input_dtype=True,
        )
        ttnn.deallocate(x)
        caps["G4_conv_post"] = _tt_to_torch(x_post).permute(0, 2, 1).contiguous()
        K = p.post_n_fft // 2 + 1
        B = int(x_post.shape[0])
        T_post = int(x_post.shape[1])
        spec_nlc = ttnn.slice(x_post, [0, 0, 0], [B, T_post, K], [1, 1, 1], memory_config=mc)
        phase_nlc = ttnn.slice(x_post, [0, 0, K], [B, T_post, 2 * K], [1, 1, 1], memory_config=mc)
        ttnn.deallocate(x_post)
        spec_nlc = ttnn.exp(spec_nlc, memory_config=mc)
        phase_nlc = ttnn.sin(phase_nlc, memory_config=mc)
        spec_bct = ttnn.permute(spec_nlc, (0, 2, 1), memory_config=mc)
        phase_bct = ttnn.permute(phase_nlc, (0, 2, 1), memory_config=mc)
        caps["G5_spec"] = _tt_to_torch(spec_bct)
        caps["G6_phase"] = _tt_to_torch(phase_bct)
        ttnn.deallocate(spec_nlc)
        ttnn.deallocate(phase_nlc)
        audio_tt = gen._stft.inverse(spec_bct, phase_bct)
        caps["G7_audio"] = _tt_to_torch(audio_tt).squeeze()
        ttnn.deallocate(spec_bct)
        ttnn.deallocate(phase_bct)
        ttnn.deallocate(audio_tt)
        ttnn.deallocate(F0_tt)
        ttnn.deallocate(N_tt)
        ttnn.deallocate(s_tt)

    deallocate_m_source_rng_tt(rng_tt)
    return caps


@dataclass
class StageTable:
    title: str
    rows: list[tuple[str, float, str, str]]  # name, pcc, note, drop_marker


def _compare_stages(ref_caps: dict[str, torch.Tensor], tt_caps: dict[str, torch.Tensor], title: str) -> StageTable:
    rows: list[tuple[str, float, str, str]] = []
    prev = 1.0
    for k in ref_caps:
        if k not in tt_caps:
            continue
        name, pcc, note = _pcc_flat(k, ref_caps[k], tt_caps[k])
        marker = ""
        if pcc == pcc:
            drop = prev - pcc
            if drop > 0.05:
                marker = f"drop {drop:+.3f}"
            prev = pcc
        rows.append((name, pcc, note, marker))
    return StageTable(title=title, rows=rows)


def _print_table(table: StageTable) -> None:
    print(f"\n=== {table.title} ===")
    print(f"{'Stage':<28} {'PCC':>10}  Notes")
    print("-" * 60)
    for name, pcc, note, marker in table.rows:
        extra = f"  *** {marker} ***" if marker else ""
        print(f"{name:<28} {pcc:10.6f}  {note}{extra}")


def _audio_pcc(ref_audio: torch.Tensor, tt_audio: torch.Tensor) -> float:
    _, pcc = comp_pcc(ref_audio.detach().float().reshape(1, -1), tt_audio.detach().float().reshape(1, -1), pcc=0.0)
    return float(pcc)


def _format_report(
    *,
    text: str,
    phonemes: str,
    input_rows: list[tuple[str, float]],
    tables: list[StageTable],
    summaries: list[str],
) -> str:
    lines = [
        "# Kokoro decode-stack diagnostic findings",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Text: `{text}`",
        f"Phonemes ({len(phonemes)}): `{phonemes}`",
        "",
        "TT fallbacks during decode walk: config E — STFT + SineGen + Phase + linear + tanh CPU fallbacks.",
        "",
        "## Prosody inputs (ref vs TT, before decoder)",
        "",
        "| Tensor | PCC (TT vs ref) |",
        "|--------|-----------------|",
    ]
    for name, pcc in input_rows:
        lines.append(f"| {name} | {pcc:.6f} |")
    lines.extend(["", "## Staged decode-stack PCC (ref PyTorch vs TT walk)", ""])
    for table in tables:
        lines.append(f"### {table.title}")
        lines.append("")
        lines.append("| Stage | PCC | Notes |")
        lines.append("|-------|-----|-------|")
        for name, pcc, note, marker in table.rows:
            m = f" {marker}" if marker else ""
            lines.append(f"| {name} | {pcc:.6f} | {note}{m} |")
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    for s in summaries:
        lines.append(f"- {s}")
    lines.append("")
    return "\n".join(lines)


def run_full_plan(*, write_report: Path | None = None) -> None:
    ckpt = _find_checkpoint()
    if ckpt is None:
        sys.exit("Kokoro-82M checkpoint not found.")
    phonemes, ref_s = _phonemize(_TEST_TEXT)
    print(f"Text: {_TEST_TEXT!r}")
    print(f"Phonemes ({len(phonemes)}): {phonemes!r}")

    ref = KModel(repo_id="hexgrad/Kokoro-82M", model=str(ckpt), disable_complex=True).eval()
    ref_pros = _ref_prosody(ref, phonemes, ref_s)
    ref_asr, ref_F0, ref_N, ref_style = ref_pros[0], ref_pros[1], ref_pros[2], ref_pros[3]

    input_ids, text_mask, input_lengths, lengths_list = _tokenize(ref.vocab, phonemes, ref.context_length)

    device = ttnn.open_device(device_id=0)
    tables: list[StageTable] = []
    summaries: list[str] = []
    try:
        params = preprocess_tt_kmodel(ref, device)
        with _zero_noise():
            tt_model = TTKModel(
                device,
                ref,
                params,
                use_torch_stft_fallback=True,
                use_torch_phase_fallback=True,
                use_torch_sinegen_fallback=True,
                use_torch_linear_fallback=True,
                use_torch_tanh_fallback=True,
            )
        tt_pros = _run_tt_prosody(tt_model, input_ids, text_mask, input_lengths, lengths_list, ref_s)

        input_rows = [
            ("ASR (BCT)", _pcc_flat("asr", ref_asr, tt_pros.asr_bct)[1]),
            ("F0_curve", _pcc_flat("F0", ref_F0, tt_pros.F0_curve)[1]),
            ("N_curve", _pcc_flat("N", ref_N, tt_pros.N_curve)[1]),
            ("s_style", _pcc_flat("style", ref_style, tt_pros.s_style)[1]),
        ]
        print("\n=== Prosody boundary (TT vs ref) ===")
        for name, pcc in input_rows:
            print(f"  {name:<12} PCC = {pcc:.6f}")

        ref_caps_ref_in = _run_ref_decode_stack(ref, ref_asr, ref_F0, ref_N, ref_style)

        # A: ref prosody → TT decode (isolates decode+generator TT error)
        tt_caps_ref_in_fp32 = _run_tt_decode_stack(tt_model, ref_asr, ref_F0, ref_N, ref_style, asr_entry_dtype="fp32")
        tables.append(
            _compare_stages(
                ref_caps_ref_in,
                tt_caps_ref_in_fp32,
                "A. Ref prosody inputs → ref decode vs TT decode (ASR fp32 at boundary)",
            )
        )
        audio_a = _audio_pcc(ref_caps_ref_in["G7_audio"], tt_caps_ref_in_fp32["G7_audio"])
        print(f"\n  End audio PCC (A): {audio_a:.6f}")

        # B: ref prosody → TT decode with bf16 ASR (matches TTKModel matmul output dtype)
        tt_caps_ref_in_bf16 = _run_tt_decode_stack(tt_model, ref_asr, ref_F0, ref_N, ref_style, asr_entry_dtype="bf16")
        tables.append(
            _compare_stages(
                tt_caps_ref_in_fp32,
                tt_caps_ref_in_bf16,
                "B. Same inputs: TT decode ASR fp32 vs bf16 entry (KModel boundary)",
            )
        )
        audio_bf16 = _audio_pcc(tt_caps_ref_in_fp32["G7_audio"], tt_caps_ref_in_bf16["G7_audio"])
        print(f"  End audio PCC (B fp32 vs bf16 ASR): {audio_bf16:.6f}")

        # C: TT prosody → TT decode
        tt_caps_tt_in = _run_tt_decode_stack(
            tt_model,
            tt_pros.asr_bct,
            tt_pros.F0_curve,
            tt_pros.N_curve,
            tt_pros.s_style,
            asr_entry_dtype="bf16",
        )
        tables.append(
            _compare_stages(
                ref_caps_ref_in,
                tt_caps_tt_in,
                "C. Ref decode vs TT decode with TT prosody inputs (bf16 ASR)",
            )
        )
        audio_c = _audio_pcc(ref_caps_ref_in["G7_audio"], tt_caps_tt_in["G7_audio"])
        print(f"  End audio PCC (C ref decode vs TT+TT prosody): {audio_c:.6f}")

        # D: TT prosody → ref decode (how much prosody alone hurts)
        ref_caps_tt_in = _run_ref_decode_stack(ref, tt_pros.asr_bct, tt_pros.F0_curve, tt_pros.N_curve, tt_pros.s_style)
        tables.append(
            _compare_stages(
                ref_caps_ref_in,
                ref_caps_tt_in,
                "D. Ref prosody vs TT prosody through ref PyTorch decode",
            )
        )
        audio_d = _audio_pcc(ref_caps_ref_in["G7_audio"], ref_caps_tt_in["G7_audio"])
        print(f"  End audio PCC (D ref decode, ref vs TT inputs): {audio_d:.6f}")

        summaries = [
            f"Prosody ASR PCC (TT vs ref): {input_rows[0][1]:.6f}; F0: {input_rows[1][1]:.6f}; N: {input_rows[2][1]:.6f}.",
            f"A (ref inputs, decode only): staged walk ends G7 audio PCC {audio_a:.6f} — TT decode+gen vs ref decode+gen on identical conditioning.",
            f"B (bf16 vs fp32 ASR at TT boundary): G7 audio PCC {audio_bf16:.6f} — quantifies KModel bf16 ASR upload effect.",
            f"C (TT prosody + TT decode): G7 audio PCC {audio_c:.6f} vs ref decode on ref inputs.",
            f"D (prosody only, ref decode): G7 audio PCC {audio_d:.6f} — ref decoder with TT ASR/F0/N.",
        ]
        if input_rows[0][1] > 0.99 and audio_a >= 0.85:
            summaries.append(
                "With ref prosody + config E, TT decode+generator matches ref (A G7 > 0.85) — vocoder path is in good shape."
            )
        elif input_rows[0][1] > 0.99 and audio_a < 0.85:
            summaries.append(
                "Prosody PCC high but decode walk still low → check harmonic G0b and config E fallbacks (see DECODE_STACK_NOTES.md)."
            )
        if input_rows[1][1] > 0.99 and audio_c < 0.35:
            summaries.append(
                "Full pipeline capped by TT F0_curve (often PCC>0.999 but max_abs ~8+ Hz) — improve F0 predictor, not G0b upsample/SineGen."
            )
        elif input_rows[0][1] < 0.99 and audio_d < 0.85:
            summaries.append("Prosody inputs already diverge → improve stages 1–9 before decode-stack op work.")
        if audio_bf16 > 0.999:
            summaries.append("bf16 ASR entry at decoder boundary has negligible effect once TTDecoder recasts to fp32.")
        else:
            summaries.append("bf16 ASR entry measurably hurts — keep explicit fp32 cast before decoder in TTKModel.")

        for t in tables:
            _print_table(t)

    finally:
        ttnn.close_device(device)

    report = _format_report(
        text=_TEST_TEXT,
        phonemes=phonemes,
        input_rows=input_rows,
        tables=tables,
        summaries=summaries,
    )
    if write_report is not None:
        write_report.parent.mkdir(parents=True, exist_ok=True)
        write_report.write_text(report, encoding="utf-8")
        print(f"\nWrote report: {write_report}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode-stack PCC diagnostic")
    parser.add_argument(
        "--full-plan",
        action="store_true",
        help="Run ref vs TT prosody, bf16 ASR boundary, and write findings markdown",
    )
    parser.add_argument(
        "--write-report",
        type=Path,
        default=None,
        help="Markdown path for findings (default: DECODE_STACK_FINDINGS.md with --full-plan)",
    )
    args = parser.parse_args()
    report_path = args.write_report
    if args.full_plan and report_path is None:
        report_path = Path(__file__).resolve().parent / "DECODE_STACK_FINDINGS.md"
    if args.full_plan:
        run_full_plan(write_report=report_path)
        return

    ckpt = _find_checkpoint()
    if ckpt is None:
        sys.exit("Kokoro-82M checkpoint not found.")
    phonemes, ref_s = _phonemize(_TEST_TEXT)
    print(f"Text: {_TEST_TEXT!r}")
    print(f"Phonemes ({len(phonemes)}): {phonemes!r}")

    ref = KModel(repo_id="hexgrad/Kokoro-82M", model=str(ckpt), disable_complex=True).eval()
    asr, F0, N, s_style, _pred_dur = _ref_prosody(ref, phonemes, ref_s)
    ref_caps = _run_ref_decode_stack(ref, asr, F0, N, s_style)

    device = ttnn.open_device(device_id=0)
    try:
        params = preprocess_tt_kmodel(ref, device)
        with _zero_noise():
            tt_model = TTKModel(
                device,
                ref,
                params,
                use_torch_stft_fallback=True,
                use_torch_phase_fallback=True,
                use_torch_sinegen_fallback=True,
                use_torch_linear_fallback=True,
                use_torch_tanh_fallback=True,
            )
        tt_caps = _run_tt_decode_stack(tt_model, asr, F0, N, s_style)
        _print_table(_compare_stages(ref_caps, tt_caps, "Ref prosody → ref vs TT decode"))
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
