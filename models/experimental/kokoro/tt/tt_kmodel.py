# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of Kokoro-82M KModel: full on-device phonemes → audio forward pass."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch
from loguru import logger

import ttnn

from .tt_custom_albert import TTCustomAlbert, TTCustomAlbertParams, preprocess_tt_custom_albert
from .tt_decoder import TTDecoder, preprocess_tt_decoder
from .tt_lstm import tt_bilstm_nlc
from .tt_matmul_memory import en_matmul_plan, maybe_reshard_to_caller
from .tt_prosody_predictor import TTProsodyPredictor, TTProsodyPredictorParams, preprocess_tt_prosody_predictor
from .tt_text_encoder import TTTextEncoder, TTTextEncoderParams, preprocess_tt_text_encoder
from .tt_trace_manager import TraceManager


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


# Trace-decoder T_aligned bucketing: round the aligned mel-length up to a fixed grid so the decoder
# trace (keyed by the bucketed length) could be REUSED across texts of similar length, not just exact
# repeats. Padded frames are zeros at the tail (trimmed from the audio).
#
# DEFAULT 1 = exact keying (no padding/trim) — the proven, bit-identical capture/replay path. Coarser
# steps are NOT yet usable: step 128 (T_aligned 131 -> 256) overflows L1 (a decoder conv's static CBs
# clash with L1 buffers at the ~2x-padded length, independent of activations_in_l1); step 32 fits L1
# but the replay then diverges from the capture (root cause TBD — the exact-key path is bit-identical,
# so bucketing introduces the nondeterminism). Left at 1 until both are resolved; the pad/trim
# machinery below is retained for that work. See docs/generator_perf_optimizations.md.
_TRACE_ALIGN_STEP = 1


def _bucket_t_aligned(t: int, step: int = _TRACE_ALIGN_STEP) -> int:
    return ((int(t) + step - 1) // step) * step


def _pad_len_dim1(x: "ttnn.Tensor", target: int, mc: "ttnn.MemoryConfig") -> "ttnn.Tensor":
    """Zero-pad ``x`` along dim 1 (time) up to ``target`` (back padding, TILE). No-op if already >=."""
    cur = int(x.shape[1])
    if cur >= target:
        return x
    if x.layout != ttnn.TILE_LAYOUT:
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=mc)
    padding = [(0, 0)] * len(x.shape)
    padding[1] = (0, target - cur)
    return ttnn.pad(x, padding=padding, value=0.0, memory_config=mc)


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
        activations_in_l1: bool = False,
        disable_complex: bool = False,
        trace: bool = False,
    ) -> None:
        self.device = device
        self.vocab = ref.vocab
        self.context_length = ref.context_length
        self.params = params
        self._ref_decoder = ref.decoder  # kept for lazy preprocess_tt_decoder calls
        self._use_stft_fallback = use_torch_stft_fallback
        self._use_phase_fallback = use_torch_phase_fallback
        self._activations_in_l1 = activations_in_l1
        self._disable_complex = disable_complex

        self._bert = TTCustomAlbert(device, params.bert)
        self._predictor = TTProsodyPredictor(device, params.predictor)
        self._text_encoder = TTTextEncoder(device, params.text_encoder)

        # Decoder params cached per T_mel (STFT precomputation depends on sequence length).
        self._decoder_cache: dict[int, TTDecoder] = {}

        # Optional metal-trace of the decoder (asr/F0/N/s -> audio, the bulk of device compute),
        # captured once per T_aligned and replayed on repeat lengths. Requires ``trace_region_size``
        # on the device and the checkpoint's deterministic RNG path. Upstream (BERT/prosody) stays
        # eager — the duration readback splits the pipeline (see docs/generator_perf_optimizations.md).
        self._trace = trace
        self._trace_mgr = TraceManager(device) if trace else None
        # Trace A (prosody→duration + ASR TextEncoder): the fixed-shape T_tokens device region that
        # runs BEFORE the duration readback splits the pipeline. Captured once per identical input
        # (keyed by exact ids/speed/style — the token uploads are baked into the graph) and replayed
        # bit-for-bit on a repeat call, so BERT + prosody + the ASR encoder's ~480 BiLSTM dispatches
        # stop running eagerly. Full-length path only (the padded path uploads masks mid-graph).
        self._trace_mgr_a = TraceManager(device) if trace else None

    # ------------------------------------------------------------------
    # Decoder cache
    # ------------------------------------------------------------------

    def _get_decoder(self, t_mel: int) -> TTDecoder:
        if t_mel not in self._decoder_cache:
            _t0 = time.perf_counter()
            dec_params = preprocess_tt_decoder(
                self._ref_decoder,
                self.device,
                time_len_asr=t_mel,
                disable_complex=self._disable_complex,
            )
            self._decoder_cache[t_mel] = TTDecoder(
                self.device,
                dec_params,
                use_torch_stft_fallback=self._use_stft_fallback,
                use_torch_phase_fallback=self._use_phase_fallback,
                activations_in_l1=self._activations_in_l1,
            )
            logger.info(
                f"_get_decoder MISS t_mel={t_mel} preprocess={time.perf_counter() - _t0:.3f}s "
                f"(cache size now {len(self._decoder_cache)})"
            )
        else:
            logger.info(f"_get_decoder HIT  t_mel={t_mel} (cache size {len(self._decoder_cache)})")
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

    def release_traces(self) -> None:
        """Release captured metal traces + persistent buffers. Call before closing the device."""
        if self._trace_mgr is not None:
            self._trace_mgr.release()
        if self._trace_mgr_a is not None:
            self._trace_mgr_a.release()

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

        Used by :meth:`_device_forward` and PCC tests that need prosody outputs without
        running the vocoder.
        """
        p = self.params
        dev = self.device
        B, T = input_ids.shape
        prosody_dtype = ttnn.float32
        full_length = _batch_is_full_length(input_lengths, T)
        text_mask: torch.Tensor | None = None if full_length else _text_mask_from_input_lengths(input_lengths, T)

        # Style + keep-mask are the Trace A region's device inputs; created here (outside the captured
        # region) and treated as read-only inside it. Under a trace they become persistent buffers.
        s_pred_tt = ttnn.from_torch(
            s_pred_cpu, dtype=prosody_dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc
        )
        if full_length:
            keep_mask = ttnn.ones([B, T, 1], dtype=prosody_dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
        else:
            keep_mask = _keep_mask_btl_tt(input_lengths, T, device=dev, dtype=prosody_dtype, memory_config=mc)

        def _trace_a_region(pers: dict) -> "tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]":
            """Fixed-shape T_tokens device graph (steps 1–5 + the ASR TextEncoder):
            BERT → bert_encoder → DurationEncoder → duration BiLSTM → duration_proj → dur_clipped,
            and the ASR ``text_encoder`` → ``t_en_bct``. Returns ``(dur_clipped, d_nlc, t_en_bct)``.
            Treats ``pers`` tensors (style, keep-mask) as read-only so they stay valid across replays.
            """
            s_pred = pers["s_pred"]
            keep = pers["keep_mask"]
            if full_length:
                bert_out = self._bert(input_ids, attention_mask=None)
            else:
                bert_out = self._bert(input_ids, attention_mask=_attention_keep_mask_bt(input_lengths, T).int())
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
            d_nlc_r = self._predictor._text_encoder.forward(
                d_en_bct=d_en_bct,
                style_bs=s_pred,
                sequence_lengths=lengths_list,
                keep_mask_btl=keep,
                compute_kernel_config=ck,
                memory_config=mc,
                wire_dtype=prosody_dtype,
            )
            ttnn.deallocate(d_en_bct)  # local temp; keep/s_pred are the caller's, left intact
            x_lstm = tt_bilstm_nlc(
                x_nlc=d_nlc_r,
                fwd=p.predictor.lstm_fwd,
                rev=p.predictor.lstm_rev,
                compute_kernel_config=ck,
                memory_config=mc,
                sequence_lengths=lengths_list,
            )
            duration = self._predictor._duration_proj.forward(x_lstm, compute_kernel_config=ck, memory_config=mc)
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
            dur_clipped_r = ttnn.clip(dur_rounded_tt, min=1.0, memory_config=mc)
            ttnn.deallocate(dur_rounded_tt)
            # ASR TextEncoder — same T_tokens regime, depends only on ids/lengths (not the alignment).
            t_en_r = self._text_encoder(input_ids, input_lengths=input_lengths, text_mask=text_mask)
            return dur_clipped_r, d_nlc_r, t_en_r

        # Trace A: capture once per identical input (ids baked into the graph), replay bit-for-bit on
        # a repeat call so BERT + prosody + the ASR encoder stop running eagerly. Full-length only (the
        # padded path uploads masks/ids mid-graph, which trace capture forbids).
        if self._trace_mgr_a is not None and full_length:
            key = (
                "traceA",
                int(T),
                tuple(int(v) for v in input_ids.reshape(-1).tolist()),
                round(float(speed), 6),
                hash(s_pred_cpu.detach().cpu().contiguous().numpy().tobytes()),
            )
            dur_c, d_nlc_p, t_en_p = self._trace_mgr_a.run(
                key, {"s_pred": s_pred_tt, "keep_mask": keep_mask}, _trace_a_region
            )
            # Manager owns the persistent outputs; clone so downstream can consume/free them freely.
            dur_clipped_tt = ttnn.clone(dur_c)
            d_nlc = ttnn.clone(d_nlc_p)
            t_en_bct = ttnn.clone(t_en_p)
        else:
            dur_clipped_tt, d_nlc, t_en_bct = _trace_a_region({"s_pred": s_pred_tt, "keep_mask": keep_mask})

        ttnn.deallocate(keep_mask)

        # 6. Alignment host step (unavoidable: T_aligned = sum(pred_dur) sets tensor shapes)
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
            t_en_bct=t_en_bct,
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
        *,
        t_en_bct: "ttnn.Tensor | None" = None,
    ) -> "tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, int, torch.LongTensor]":
        """Steps 7–9 given ``d_nlc``, alignment, and style (shared by on-device and CPU prosody paths).

        ``t_en_bct`` (optional): the ASR ``TextEncoder`` output ``[B, C, T_tokens]``, precomputed in the
        Trace A region. When ``None`` the encoder is run here (the original eager order); when provided
        (traced path) it is used directly and this method owns it (deallocates after the alignment matmul).
        """
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

        F0, N = self._predictor.F0Ntrain(en_nlc, s_pred_tt, memory_config=mc)
        ttnn.deallocate(en_nlc)
        ttnn.deallocate(s_pred_tt)

        if t_en_bct is None:
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
        asr_nlc, F0, N, T_aligned, pred_dur_cpu = self._device_forward_prosody_stages(
            input_ids, input_lengths, lengths_list, s_pred_cpu, speed, mc, ck
        )

        dev = self.device
        s_style_tt = ttnn.from_torch(
            s_style_cpu, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc
        )

        # When tracing, run the decoder at a BUCKETED mel-length so the trace is keyed by (and reused
        # across) the bucket, not the exact T_aligned. Inputs are zero-padded to the bucket and the
        # audio is trimmed back. Eager (no trace) runs at the exact length (dec_len == t_mel), so the
        # original path is byte-for-byte unchanged.
        t_mel = int(T_aligned)
        dec_len = _bucket_t_aligned(t_mel) if self._trace_mgr is not None else t_mel
        decoder = self._get_decoder(dec_len)
        gen = decoder._generator

        if dec_len != t_mel:
            asr_in = _pad_len_dim1(asr_nlc, dec_len, mc)  # [B, dec_len, C]
            F0_in = _pad_len_dim1(F0, 2 * dec_len, mc)  # T_f0 == 2 * T_mel
            N_in = _pad_len_dim1(N, 2 * dec_len, mc)
            ttnn.deallocate(asr_nlc)
            ttnn.deallocate(F0)
            ttnn.deallocate(N)
        else:
            asr_in, F0_in, N_in = asr_nlc, F0, N

        m_source_kwargs: dict = {}
        if deterministic:
            from models.experimental.kokoro.m_source_rng import (
                deallocate_m_source_rng_tt,
                make_zero_m_source_rng,
                upload_m_source_rng,
            )

            B_dec = int(F0_in.shape[0])
            T_har = int(F0_in.shape[1]) * int(gen.params.upsample_scale_full)
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

        if self._trace_mgr is not None:
            # Trace B (decoder): capture once per bucket, replay on repeats. Inputs are copied into the
            # manager's persistent buffers; the graph clones them (the decoder consumes inputs).
            inputs = {"asr": asr_in, "F0": F0_in, "N": N_in, "s_style": s_style_tt}
            if deterministic:
                inputs["rand_ini"] = rng_tt.rand_ini
                inputs["sinegen_noise"] = rng_tt.sinegen_noise
                inputs["source_noise"] = rng_tt.source_noise

            def _decoder_fwd(p: dict) -> ttnn.Tensor:
                kwargs: dict = {}
                if deterministic:
                    kwargs = {
                        "sinegen_rand_ini": ttnn.clone(p["rand_ini"]),
                        "sinegen_noise_raw": ttnn.clone(p["sinegen_noise"]),
                        "source_noise_raw": ttnn.clone(p["source_noise"]),
                    }
                return decoder(
                    ttnn.clone(p["asr"]),
                    ttnn.clone(p["F0"]),
                    ttnn.clone(p["N"]),
                    ttnn.clone(p["s_style"]),
                    memory_config=mc,
                    **kwargs,
                )

            # Manager owns the persistent output; clone so the caller frees it independently.
            audio_full = ttnn.clone(self._trace_mgr.run(dec_len, inputs, _decoder_fwd))
            if dec_len != t_mel:
                # Trim the bucket audio back to the real length (audio_len is ~linear in mel frames;
                # the iSTFT's small constant offset leaves at most a few inaudible boundary samples).
                bl = int(audio_full.shape[-1])
                real_len = bl * t_mel // dec_len
                audio = ttnn.slice(
                    audio_full, [0, 0, 0], [int(audio_full.shape[0]), 1, real_len], [1, 1, 1], memory_config=mc
                )
                ttnn.deallocate(audio_full)
            else:
                audio = audio_full
        else:
            audio = decoder(asr_in, F0_in, N_in, s_style_tt, memory_config=mc, **m_source_kwargs)

        if deterministic:
            deallocate_m_source_rng_tt(rng_tt)
        ttnn.deallocate(asr_in)
        ttnn.deallocate(F0_in)
        ttnn.deallocate(N_in)
        ttnn.deallocate(s_style_tt)

        return audio, pred_dur_cpu


@contextmanager
def _noop():
    yield
