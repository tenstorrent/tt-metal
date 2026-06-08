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
from .tt_matmul_memory import en_matmul_plan, maybe_reshard_to_caller
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


def _en_matmul_nlc(
    alignment_TaT: ttnn.Tensor,
    d_nlc: ttnn.Tensor,
    *,
    memory_config: ttnn.MemoryConfig,
    compute_kernel_config,
) -> ttnn.Tensor:
    """``en_nlc = alignment^T @ d`` with swept L1-width-sharded output when feasible."""
    en_pc, en_out_mc, en_reshard = en_matmul_plan(alignment_TaT, d_nlc)
    en_matmul_mc = en_out_mc if en_out_mc is not None else memory_config
    en_nlc = ttnn.matmul(
        alignment_TaT,
        d_nlc,
        program_config=en_pc,
        memory_config=en_matmul_mc,
        compute_kernel_config=compute_kernel_config,
    )
    if en_reshard:
        return maybe_reshard_to_caller(en_nlc, memory_config)
    if en_matmul_mc.buffer_type != memory_config.buffer_type:
        out = ttnn.to_memory_config(en_nlc, memory_config)
        if out is not en_nlc:
            ttnn.deallocate(en_nlc)
        return out
    return en_nlc


def _to_fp32_if_needed(x: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> tuple[ttnn.Tensor, bool]:
    """Cast to fp32 when needed; return ``(tensor, owns_dealloc)``."""
    if x.dtype == ttnn.float32:
        return x, False
    out = ttnn.typecast(x, ttnn.float32, memory_config=memory_config)
    return out, True


def _batch_is_full_length(input_lengths: torch.LongTensor, seq_len: int) -> bool:
    """True when every row uses all ``seq_len`` tokens (no padding; common in ``TTKModel.forward``)."""
    return bool((input_lengths == seq_len).all().item())


def _attention_keep_mask_bt(input_lengths: torch.LongTensor, seq_len: int) -> torch.Tensor:
    """``[B, T]`` bool; ``True`` on real tokens (PL-BERT ``attention_mask`` uses ``.int()`` of this)."""
    B = int(input_lengths.shape[0])
    positions = torch.arange(seq_len, device=input_lengths.device).unsqueeze(0).expand(B, -1)
    return (positions + 1) <= input_lengths.unsqueeze(1)


def _text_mask_from_input_lengths(input_lengths: torch.LongTensor, seq_len: int) -> torch.Tensor:
    """``[B, T]`` bool mask; ``True`` marks padded positions (same as reference ``KModel``)."""
    return ~_attention_keep_mask_bt(input_lengths, seq_len)


def _keep_mask_btl_tt(
    input_lengths: torch.LongTensor,
    seq_len: int,
    *,
    device: ttnn.Device,
    dtype: ttnn.DataType,
    memory_config: ttnn.MemoryConfig,
) -> ttnn.Tensor:
    """Upload ``[B, T, 1]`` keep mask (``1`` = real token) for DurationEncoder / AdaIN paths.

    Built on CPU then uploaded: ``input_lengths`` is a CPU long tensor and TTNN has no
    ``arange``+compare shortcut cheaper than a single ``from_torch`` for ``T <= 512``.
    """
    keep = _attention_keep_mask_bt(input_lengths, seq_len).to(torch.float32).unsqueeze(-1)
    return ttnn.from_torch(
        keep,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


# BH BF16 MACs accumulate rounding error over 512-token sequences; PLBERT PCC ~0.94 at T=512
# causes DurationEncoder to drift pred_dur by ~9%, making audio PCCs collapse to ~0.
# Below this token count PLBERT PCC is >0.998 and TTNN prosody is fully accurate.
_LONG_SEQUENCE_CPU_PROSODY_THRESHOLD = 480


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
        use_torch_phase_fallback: bool = False,
    ) -> None:
        self.device = device
        self.vocab = ref.vocab
        self.context_length = ref.context_length
        self.params = params
        self._ref_decoder = ref.decoder  # kept for lazy preprocess_tt_decoder calls
        self._use_stft_fallback = use_torch_stft_fallback
        self._use_phase_fallback = use_torch_phase_fallback

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
                use_torch_phase_fallback=self._use_phase_fallback,
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
            audio_tt, pred_dur = self._device_forward(
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

        return self.Output(audio=audio, pred_dur=pred_dur)

    __call__ = forward

    # ------------------------------------------------------------------
    # Internal on-device forward
    # ------------------------------------------------------------------

    def _device_forward_prosody_stages(
        self,
        input_ids: torch.LongTensor,
        input_lengths: torch.LongTensor,
        lengths_list: list,
        s_pred_cpu: torch.Tensor,
        speed: float,
        mc: "ttnn.MemoryConfig",
        ck,
    ) -> "tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, int, torch.LongTensor]":
        """Steps 1–9 of the forward pass: BERT → alignment → F0/N → TextEncoder → asr_nlc.

        Returns ``(asr_nlc, F0, N, T_aligned, pred_dur_cpu)``.  All returned device
        tensors are owned by the caller; the caller must deallocate them.

        Used by :meth:`_device_forward` and by
        :class:`~models.experimental.kokoro.runner.performant_runner.KokoroPerformantRunner`
        to pre-compute stable decoder inputs for trace capture.
        """
        p = self.params
        dev = self.device
        B, T = input_ids.shape
        prosody_dtype = ttnn.float32
        full_length = _batch_is_full_length(input_lengths, T)
        text_mask: torch.Tensor | None = None if full_length else _text_mask_from_input_lengths(input_lengths, T)

        # 1. PL-BERT
        if full_length:
            bert_out = self._bert(input_ids, attention_mask=None)
        else:
            bert_out = self._bert(input_ids, attention_mask=_attention_keep_mask_bt(input_lengths, T).int())

        # 2. bert_encoder
        bert_for_enc = bert_out
        owns_bert_cast = False
        if bert_out.dtype != ttnn.float32:
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

        if d_en_bct.dtype != prosody_dtype:
            d_en_fp32 = ttnn.typecast(d_en_bct, prosody_dtype, memory_config=mc)
            ttnn.deallocate(d_en_bct)
            d_en_bct = d_en_fp32

        # 3. Style
        s_pred_tt = ttnn.from_torch(
            s_pred_cpu, dtype=prosody_dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc
        )

        # 4. DurationEncoder
        if full_length:
            keep_mask = ttnn.ones([B, T, 1], dtype=prosody_dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
        else:
            keep_mask = _keep_mask_btl_tt(input_lengths, T, device=dev, dtype=prosody_dtype, memory_config=mc)
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

        # 5. BiLSTM + duration_proj
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

        # 6. Alignment host step (unavoidable: T_aligned = sum(pred_dur) sets tensor shapes)
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
        pred_dur_cpu = pred_dur.clone()

        aln_cpu = _build_alignment(pred_dur)
        T_aligned = int(aln_cpu.shape[2])
        aln_tt = ttnn.from_torch(aln_cpu, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)

        return self._device_forward_prosody_stages_from_aln(
            input_ids,
            input_lengths,
            text_mask,
            d_nlc,
            s_pred_tt,
            aln_tt,
            T_aligned,
            pred_dur_cpu,
            mc,
            ck,
        )

    def _device_forward_prosody_stages_from_aln(
        self,
        input_ids: torch.LongTensor,
        input_lengths: torch.LongTensor,
        text_mask: torch.Tensor | None,
        d_nlc: ttnn.Tensor,
        s_pred_tt: ttnn.Tensor,
        aln_tt: ttnn.Tensor,
        T_aligned: int,
        pred_dur_cpu: torch.LongTensor,
        mc: ttnn.MemoryConfig,
        ck,
    ) -> "tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, int, torch.LongTensor]":
        """Steps 7–9 given ``d_nlc``, alignment, and style (shared by on-device and CPU prosody paths)."""
        # 7. en_nlc = aln^T @ d_nlc
        aln_Ta_T = ttnn.permute(aln_tt, (0, 2, 1), memory_config=mc)
        d_mat, owns_d = _to_fp32_if_needed(d_nlc, mc)
        if owns_d:
            ttnn.deallocate(d_nlc)
        en_nlc = _en_matmul_nlc(
            aln_Ta_T,
            d_mat,
            memory_config=mc,
            compute_kernel_config=ck,
        )
        ttnn.deallocate(d_mat)
        ttnn.deallocate(aln_Ta_T)

        # 8. F0 / N
        F0, N = self._predictor.F0Ntrain(en_nlc, s_pred_tt, memory_config=mc)
        ttnn.deallocate(en_nlc)
        ttnn.deallocate(s_pred_tt)

        # 9. TextEncoder + asr = t_en @ aln
        t_en_bct = self._text_encoder(input_ids, input_lengths=input_lengths, text_mask=text_mask)
        asr_bct = ttnn.matmul(t_en_bct, aln_tt, memory_config=mc, compute_kernel_config=ck)
        ttnn.deallocate(t_en_bct)
        ttnn.deallocate(aln_tt)
        asr_nlc = ttnn.permute(asr_bct, (0, 2, 1), memory_config=mc)
        ttnn.deallocate(asr_bct)

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

        return asr_nlc, F0, N, T_aligned, pred_dur_cpu

    def _device_forward(
        self,
        input_ids: torch.LongTensor,
        input_lengths: torch.LongTensor,
        lengths_list: list,
        s_pred_cpu: torch.Tensor,
        s_style_cpu: torch.Tensor,
        speed: float,
        mc: "ttnn.MemoryConfig",
        ck,
        *,
        deterministic: bool = False,
    ) -> "tuple[ttnn.Tensor, torch.LongTensor]":
        """Full forward steps 1–10. Returns ``(audio_tt, pred_dur_cpu)``."""
        # Steps 1-9
        asr_nlc, F0, N, T_aligned, pred_dur_cpu = self._device_forward_prosody_stages(
            input_ids, input_lengths, lengths_list, s_pred_cpu, speed, mc, ck
        )

        # ------ 10. Decoder (vocoder) ------------------------------------
        dev = self.device
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

        return audio, pred_dur_cpu

    def _device_forward_fixed_aln(
        self,
        input_ids: torch.LongTensor,
        input_lengths: torch.LongTensor,
        lengths_list: list,
        s_pred_tt: "ttnn.Tensor",
        s_style_tt: "ttnn.Tensor",
        aln_tt: "ttnn.Tensor",
        T_aligned: int,
        mc: "ttnn.MemoryConfig",
        ck,
    ) -> "ttnn.Tensor":
        """Traceable forward that uses pre-allocated alignment and style tensors.

        Identical to ``_device_forward`` except:
        - ``s_pred_tt``, ``s_style_tt``, ``aln_tt`` are **borrowed** (pre-allocated by the
          caller and must not be deallocated here).
        - Steps 5-6 (BiLSTM → pred_dur readback → aln_cpu build) are skipped; the caller
          provides ``aln_tt`` directly so the entire path is traceable on a single CQ.
        - Returns the raw ``audio`` device tensor (no ``to_torch``).

        Used by :class:`~models.experimental.kokoro.runner.performant_runner.KokoroPerformantRunner`
        for ``ttnn.begin_trace_capture`` / ``execute_trace`` perf runs.
        """
        p = self.params
        dev = self.device
        B, T = input_ids.shape

        # 1. PL-BERT
        bert_out = self._bert(input_ids, attention_mask=None)

        # 2. bert_encoder: Linear(hidden_size → hidden_dim)
        bert_for_enc = bert_out
        owns_bert_cast = False
        if bert_out.dtype != ttnn.float32:
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

        prosody_dtype = ttnn.float32
        if d_en_bct.dtype != prosody_dtype:
            d_en_fp32 = ttnn.typecast(d_en_bct, prosody_dtype, memory_config=mc)
            ttnn.deallocate(d_en_bct)
            d_en_bct = d_en_fp32

        # 3+4. DurationEncoder (s_pred_tt is borrowed — not deallocated here)
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

        # Steps 5-6 skipped: BiLSTM, pred_dur readback, aln_cpu build omitted.
        # aln_tt is provided pre-allocated by the caller.

        # 7. en_nlc = aln^T @ d_nlc
        aln_Ta_T = ttnn.permute(aln_tt, (0, 2, 1), memory_config=mc)
        d_mat, owns_d = _to_fp32_if_needed(d_nlc, mc)
        if owns_d:
            ttnn.deallocate(d_nlc)
        en_nlc = _en_matmul_nlc(
            aln_Ta_T,
            d_mat,
            memory_config=mc,
            compute_kernel_config=ck,
        )
        ttnn.deallocate(d_mat)
        ttnn.deallocate(aln_Ta_T)

        # 8. F0/N
        F0, N = self._predictor.F0Ntrain(en_nlc, s_pred_tt, memory_config=mc)
        ttnn.deallocate(en_nlc)
        # s_pred_tt is borrowed — not deallocated here

        # 9. TextEncoder + asr = t_en @ aln
        t_en_bct = self._text_encoder(input_ids, input_lengths=input_lengths)
        asr_bct = ttnn.matmul(t_en_bct, aln_tt, memory_config=mc, compute_kernel_config=ck)
        # aln_tt is borrowed — not deallocated here
        ttnn.deallocate(t_en_bct)
        asr_nlc = ttnn.permute(asr_bct, (0, 2, 1), memory_config=mc)
        ttnn.deallocate(asr_bct)

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

        # 10. Decoder (s_style_tt is borrowed — not deallocated here)
        decoder = self._get_decoder(T_aligned)
        audio = decoder(asr_nlc, F0, N, s_style_tt, memory_config=mc)
        ttnn.deallocate(asr_nlc)
        ttnn.deallocate(F0)
        ttnn.deallocate(N)
        # s_style_tt is borrowed — not deallocated here

        return audio


@contextmanager
def _noop():
    yield
