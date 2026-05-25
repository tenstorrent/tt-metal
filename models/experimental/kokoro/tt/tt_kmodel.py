# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of Kokoro-82M KModel: full on-device phonemes → audio forward pass."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch

import ttnn

from .tt_custom_albert import TTCustomAlbert, TTCustomAlbertParams, preprocess_tt_custom_albert
from .tt_decoder import TTDecoder, preprocess_tt_decoder
from .tt_lstm import tt_bilstm_nlc
from .tt_prosody_predictor import TTProsodyPredictor, TTProsodyPredictorParams, preprocess_tt_prosody_predictor
from .tt_text_encoder import TTTextEncoder, TTTextEncoderParams, preprocess_tt_text_encoder


# ---------------------------------------------------------------------------
# Shared config constants
# ---------------------------------------------------------------------------


class KokoroConfig:
    repo_id: str = "hexgrad/Kokoro-82M"
    sample_rate_hz: int = 24_000


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TTKModelParams:
    bert: TTCustomAlbertParams
    bert_encoder_w: ttnn.Tensor  # stored transposed for transpose_b=True in ttnn.linear
    bert_encoder_b: ttnn.Tensor
    predictor: TTProsodyPredictorParams
    text_encoder: TTTextEncoderParams
    hidden_dim: int
    style_dim: int


# ---------------------------------------------------------------------------
# Preprocess
# ---------------------------------------------------------------------------


def preprocess_tt_kmodel(
    ref,  # KModel instance
    device: ttnn.Device,
    *,
    weights_dtype=ttnn.bfloat16,
    conv_weights_dtype=ttnn.float32,
) -> TTKModelParams:
    """Upload KModel weights to device. TTDecoder is handled lazily by TTKModel._get_decoder."""
    bert_params = preprocess_tt_custom_albert(ref.bert, device, weights_dtype=weights_dtype)

    enc_w = ttnn.from_torch(
        ref.bert_encoder.weight.detach().cpu(),
        dtype=weights_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    enc_b = ttnn.from_torch(
        ref.bert_encoder.bias.detach().cpu().reshape(1, 1, 1, -1),
        dtype=weights_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    pred_params = preprocess_tt_prosody_predictor(
        ref.predictor,
        device,
        weights_dtype=weights_dtype,
        conv_weights_dtype=conv_weights_dtype,
    )
    text_enc_params = preprocess_tt_text_encoder(ref.text_encoder, device, weights_dtype=weights_dtype)

    hidden_dim = int(ref.bert_encoder.out_features)
    style_dim = int(ref.predictor.text_encoder.sty_dim)

    return TTKModelParams(
        bert=bert_params,
        bert_encoder_w=enc_w,
        bert_encoder_b=enc_b,
        predictor=pred_params,
        text_encoder=text_enc_params,
        hidden_dim=hidden_dim,
        style_dim=style_dim,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _build_alignment(pred_dur: torch.LongTensor) -> torch.Tensor:
    """Build the duration-alignment matrix from per-phoneme frame counts.

    Args:
        pred_dur: ``[T_tokens]`` integer frame counts (each ≥ 1).

    Returns:
        ``[1, T_tokens, T_aligned]`` float32 one-hot alignment on CPU.

    Note: this runs on CPU because T_aligned = sum(pred_dur) is only known after
    reading back from device, and TTNN requires static shapes at allocation time.
    """
    T_tokens = int(pred_dur.shape[0])
    T_aligned = int(pred_dur.sum().item())
    indices = torch.repeat_interleave(torch.arange(T_tokens), pred_dur)
    aln = torch.zeros(T_tokens, T_aligned)
    aln[indices, torch.arange(T_aligned)] = 1.0
    return aln.unsqueeze(0)  # [1, T_tokens, T_aligned]


def _to_fp32_if_needed(x: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> tuple[ttnn.Tensor, bool]:
    """Cast to fp32 when needed; return ``(tensor, owns_dealloc)``."""
    if x.dtype == ttnn.float32:
        return x, False
    out = ttnn.typecast(x, ttnn.float32, memory_config=memory_config)
    return out, True


# ---------------------------------------------------------------------------
# TTKModel
# ---------------------------------------------------------------------------


class TTKModel:
    """Full TTNN port of KModel. Alignment matrix read-back is the only host-side step."""

    @dataclass
    class Output:
        audio: torch.FloatTensor
        pred_dur: Optional[torch.LongTensor] = None

    def __init__(
        self,
        device: ttnn.Device,
        ref,  # KModel
        params: TTKModelParams,
        *,
        use_torch_stft_fallback: bool = False,
        use_torch_stft_conv_fallback: bool = False,
        use_torch_atan2_fallback: bool = False,
        use_torch_phase_fallback: bool = False,
        use_torch_sinegen_fallback: bool = False,
        use_torch_f0n_conv_fallback: bool = False,
        use_torch_f0_upsamp_fallback: Optional[bool] = None,
        use_torch_linear_fallback: bool = False,
        use_torch_tanh_fallback: bool = False,
        use_fp32_prosody_boundary: bool = True,
    ) -> None:
        self.device = device
        self.vocab = ref.vocab
        self.context_length = ref.context_length
        self.params = params
        self._ref_decoder = ref.decoder  # kept for lazy preprocess_tt_decoder calls
        self._use_stft_fallback = use_torch_stft_fallback
        self._use_stft_conv_fallback = use_torch_stft_conv_fallback
        self._use_atan2_fallback = use_torch_atan2_fallback
        self._use_phase_fallback = use_torch_phase_fallback
        self._use_sinegen_fallback = use_torch_sinegen_fallback
        self._use_f0n_conv_fallback = use_torch_f0n_conv_fallback
        self._use_f0_upsamp_fallback = use_torch_f0_upsamp_fallback
        self._use_linear_fallback = use_torch_linear_fallback
        self._use_tanh_fallback = use_torch_tanh_fallback
        self._use_fp32_prosody_boundary = use_fp32_prosody_boundary

        self._bert = TTCustomAlbert(device, params.bert)
        self._predictor = TTProsodyPredictor(device, params.predictor)
        self._text_encoder = TTTextEncoder(device, params.text_encoder)

        # Decoder params cached per T_mel (STFT precomputation depends on sequence length).
        self._decoder_cache: dict[int, TTDecoder] = {}

    # ------------------------------------------------------------------
    # Decoder cache
    # ------------------------------------------------------------------

    def _get_decoder(self, t_mel: int) -> TTDecoder:
        if t_mel not in self._decoder_cache:
            dec_params = preprocess_tt_decoder(
                self._ref_decoder,
                self.device,
                time_len_asr=t_mel,
            )
            self._decoder_cache[t_mel] = TTDecoder(
                self.device,
                dec_params,
                use_torch_stft_fallback=self._use_stft_fallback,
                use_torch_stft_conv_fallback=self._use_stft_conv_fallback,
                use_torch_atan2_fallback=self._use_atan2_fallback,
                use_torch_phase_fallback=self._use_phase_fallback,
                use_torch_sinegen_fallback=self._use_sinegen_fallback,
                use_torch_f0n_conv_fallback=self._use_f0n_conv_fallback,
                use_torch_f0_upsamp_fallback=self._use_f0_upsamp_fallback,
                use_torch_linear_fallback=self._use_linear_fallback,
                use_torch_tanh_fallback=self._use_tanh_fallback,
            )
        return self._decoder_cache[t_mel]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        phonemes: str,
        ref_s: torch.FloatTensor,
        speed: float = 1.0,
        deterministic: bool = False,
    ) -> "TTKModel.Output":
        input_ids_list = list(filter(lambda i: i is not None, map(lambda p: self.vocab.get(p), phonemes)))
        assert len(input_ids_list) + 2 <= self.context_length, (
            len(input_ids_list) + 2,
            self.context_length,
        )
        input_ids = torch.LongTensor([[0, *input_ids_list, 0]])

        ref_s = ref_s.cpu()
        if ref_s.dim() == 1:
            ref_s = ref_s.unsqueeze(0)

        B, T = input_ids.shape
        input_lengths = torch.full((B,), T, dtype=torch.long)
        lengths_list: list[int] = input_lengths.tolist()

        s_pred_cpu = ref_s[:, self.params.style_dim :]
        s_style_cpu = ref_s[:, : self.params.style_dim]

        mc = ttnn.DRAM_MEMORY_CONFIG
        ck = self._predictor.compute_kernel_config

        ctx = _zero_noise() if deterministic else _noop()
        with ctx:
            audio_tt = self._device_forward(
                input_ids,
                input_lengths,
                lengths_list,
                s_pred_cpu,
                s_style_cpu,
                speed,
                mc,
                ck,
                deterministic=deterministic,
            )

        audio = ttnn.to_torch(audio_tt).float().squeeze()
        ttnn.deallocate(audio_tt)

        return self.Output(audio=audio)

    __call__ = forward

    # ------------------------------------------------------------------
    # Internal on-device forward
    # ------------------------------------------------------------------

    def _device_forward(
        self,
        input_ids: torch.LongTensor,
        input_lengths: torch.LongTensor,
        lengths_list: list,
        s_pred_cpu: torch.Tensor,
        s_style_cpu: torch.Tensor,
        speed: float,
        mc: ttnn.MemoryConfig,
        ck,
        *,
        deterministic: bool = False,
    ) -> ttnn.Tensor:
        p = self.params
        dev = self.device
        B, T = input_ids.shape

        # ------ 1. PL-BERT (attention_mask=None → all-attend, no torch mask ops) ------
        bert_out = self._bert(input_ids, attention_mask=None)

        # ------ 2. bert_encoder: Linear(hidden_size → hidden_dim) --------
        bert_for_enc = bert_out
        owns_bert_cast = False
        if self._use_fp32_prosody_boundary and bert_out.dtype != ttnn.float32:
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

        # ttnn.linear may prepend a singleton batch dim
        while len(d_en.shape) > 3:
            d_en = ttnn.squeeze(d_en, 0)

        d_en_bct = ttnn.permute(d_en, (0, 2, 1), memory_config=mc)
        ttnn.deallocate(d_en)

        prosody_dtype = ttnn.float32 if self._use_fp32_prosody_boundary else ttnn.bfloat16
        if self._use_fp32_prosody_boundary and d_en_bct.dtype != prosody_dtype:
            d_en_fp32 = ttnn.typecast(d_en_bct, prosody_dtype, memory_config=mc)
            ttnn.deallocate(d_en_bct)
            d_en_bct = d_en_fp32

        # ------ 3. Style on device ----------------------------------------
        s_pred_tt = ttnn.from_torch(
            s_pred_cpu, dtype=prosody_dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc
        )

        # ------ 4. DurationEncoder (P6: wire_dtype keeps concat dtypes unified in fp32) ------
        # Full-length sequence (no padding): keep_mask is all-ones, created directly on device.
        keep_mask = ttnn.ones([B, T, 1], dtype=prosody_dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
        d_nlc = self._predictor._text_encoder.forward(
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

        # ------ 5. BiLSTM + duration_proj ---------------------------------
        x_lstm = tt_bilstm_nlc(
            x_nlc=d_nlc,
            fwd=p.predictor.lstm_fwd,
            rev=p.predictor.lstm_rev,
            compute_kernel_config=ck,
            memory_config=mc,
            sequence_lengths=lengths_list,
        )
        duration = self._predictor._duration_proj.forward(x_lstm, compute_kernel_config=ck, memory_config=mc)
        ttnn.deallocate(x_lstm)

        # ------ 6. Alignment (host step: read pred_dur, build matrix) -----
        # sigmoid → sum → scale → round → clip all run on device.
        # The single CPU readback is unavoidable: T_aligned = sum(pred_dur) determines
        # the alignment matrix shape, which TTNN must know before allocating tensors.
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

        aln_cpu = _build_alignment(pred_dur)
        T_aligned = int(aln_cpu.shape[2])

        aln_dtype = ttnn.float32 if self._use_fp32_prosody_boundary else ttnn.bfloat16
        aln_tt = ttnn.from_torch(aln_cpu, dtype=aln_dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)

        # ------ 7. en_nlc = aln^T @ d_nlc --------------------------------
        aln_Ta_T = ttnn.permute(aln_tt, (0, 2, 1), memory_config=mc)
        if self._use_fp32_prosody_boundary:
            d_mat, owns_d = _to_fp32_if_needed(d_nlc, mc)
            if owns_d:
                ttnn.deallocate(d_nlc)
            en_nlc = ttnn.matmul(aln_Ta_T, d_mat, memory_config=mc, compute_kernel_config=ck)
            ttnn.deallocate(d_mat)
        else:
            en_nlc = ttnn.matmul(aln_Ta_T, d_nlc, memory_config=mc, compute_kernel_config=ck)
            ttnn.deallocate(d_nlc)
        ttnn.deallocate(aln_Ta_T)

        # ------ 8. F0 / N from predictor ---------------------------------
        if self._use_fp32_prosody_boundary:
            en_fp32, owns_en = _to_fp32_if_needed(en_nlc, mc)
            if owns_en:
                ttnn.deallocate(en_nlc)
                en_nlc = en_fp32
            s_pred_f0, owns_s = _to_fp32_if_needed(s_pred_tt, mc)
            F0, N = self._predictor.F0Ntrain(en_nlc, s_pred_f0, memory_config=mc, use_fp32_boundary=True)
            if owns_s:
                ttnn.deallocate(s_pred_f0)
        else:
            F0, N = self._predictor.F0Ntrain(en_nlc, s_pred_tt, memory_config=mc, use_fp32_boundary=False)
        ttnn.deallocate(en_nlc)
        ttnn.deallocate(s_pred_tt)

        # ------ 9. TextEncoder + asr = t_en @ aln -------------------------
        # No mask arguments: text encoder creates all-ones keep_mask on device (full-length sequence).
        t_en_bct = self._text_encoder(input_ids, input_lengths=input_lengths)

        asr_bct = ttnn.matmul(t_en_bct, aln_tt, memory_config=mc, compute_kernel_config=ck)
        ttnn.deallocate(t_en_bct)
        ttnn.deallocate(aln_tt)

        asr_nlc = ttnn.permute(asr_bct, (0, 2, 1), memory_config=mc)
        ttnn.deallocate(asr_bct)
        if self._use_fp32_prosody_boundary:
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

        # ------ 10. Decoder (vocoder) ------------------------------------
        s_style_tt = ttnn.from_torch(
            s_style_cpu, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc
        )
        decoder = self._get_decoder(T_aligned)
        gen = decoder._generator
        m_source_kwargs: dict = {}
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
        else:
            rng_tt = None

        audio = decoder(asr_nlc, F0, N, s_style_tt, memory_config=mc, **m_source_kwargs)
        if deterministic:
            deallocate_m_source_rng_tt(rng_tt)
        ttnn.deallocate(asr_nlc)
        ttnn.deallocate(F0)
        ttnn.deallocate(N)
        ttnn.deallocate(s_style_tt)

        return audio


@contextmanager
def _noop():
    yield
