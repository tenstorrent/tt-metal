# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of Kokoro-82M :class:`~models.experimental.kokoro.reference.model.KModel`.

Full on-device forward pass composing:

    TTCustomAlbert   (PL-BERT)
    bert_encoder     (ttnn.linear 768 → 512)
    TTDurationEncoder + BiLSTM + TTLinearNorm   (duration prediction)
    alignment matrix   (built on host from integer pred_dur; pushed to device)
    TTAdainResBlk1d branches × 3   (F0 / N from TTProsodyPredictor.F0Ntrain)
    TTTextEncoder   (CNN + BiLSTM text features)
    matmul alignment   (asr = t_en @ pred_aln_trg)
    TTDecoder → TTGenerator   (iSTFTNet vocoder)

The only host-side step is reading predicted durations (a single small integer
tensor) to construct the alignment matrix, which is then immediately pushed back to
device.  This mirrors the existing precedent in :class:`TTProsodyPredictor` and
:class:`TTTextEncoder` which already read ``sequence_lengths`` from the host.

Generator fallbacks are propagated to :class:`TTDecoder` and are intended for
testing / hardware-limitation workarounds only.
"""

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
    """Shared constants for Kokoro-82M (hexgrad/Kokoro-82M on HuggingFace)."""

    repo_id: str = "hexgrad/Kokoro-82M"
    sample_rate_hz: int = 24_000


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TTKModelParams:
    """Device-resident weights for :class:`TTKModel` (excluding the decoder)."""

    bert: TTCustomAlbertParams
    bert_encoder_w: ttnn.Tensor  # [hidden_dim, hidden_size_bert] — used with transpose_b=True
    bert_encoder_b: ttnn.Tensor  # [1, 1, 1, hidden_dim]
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
    """Upload a reference :class:`~models.experimental.kokoro.reference.model.KModel`
    to device for :class:`TTKModel`.

    The vocoder (:class:`TTDecoder`) is **not** preprocessed here because its STFT
    matrices depend on ``time_len_asr``, which is determined at inference time.
    :meth:`TTKModel._get_decoder` handles that lazily and caches by ``T_mel``.
    """
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
    """Context that returns zeros from all stochastic torch ops (deterministic mode)."""
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
    """
    T_tokens = int(pred_dur.shape[0])
    T_aligned = int(pred_dur.sum().item())
    indices = torch.repeat_interleave(torch.arange(T_tokens), pred_dur)
    aln = torch.zeros(T_tokens, T_aligned)
    aln[indices, torch.arange(T_aligned)] = 1.0
    return aln.unsqueeze(0)  # [1, T_tokens, T_aligned]


# ---------------------------------------------------------------------------
# TTKModel
# ---------------------------------------------------------------------------


class TTKModel:
    """Full TTNN port of ``KModel``.

    Composes :class:`TTCustomAlbert`, :class:`TTProsodyPredictor`,
    :class:`TTTextEncoder`, and :class:`TTDecoder` entirely on-device.

    The alignment matrix (built from predicted durations) is the only step that
    touches the host — ``pred_dur`` values are read back as integers, the
    one-hot matrix is constructed on CPU in float32, and immediately pushed
    to device.

    Args:
        device: Open TT device.
        ref: Reference :class:`~models.experimental.kokoro.reference.model.KModel`.
            Used for ``vocab``, ``context_length``, and the ``decoder`` module
            (for dynamic :meth:`_get_decoder` preprocessing).
        params: Preprocessed device weights from :func:`preprocess_tt_kmodel`.
        use_torch_stft_fallback: Route ``torch.stft`` transform through CPU float32.
            Required together with ``use_torch_phase_fallback`` to reach PCC > 0.99
            on BH hardware (see :class:`TTGenerator` docstring).
        use_torch_phase_fallback: Route SineGen phase chain through CPU float32.
        use_torch_linear_fallback: Route source-module linear through CPU float32.
        use_torch_tanh_fallback: Route source-module tanh through CPU float32.
    """

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
        use_torch_linear_fallback: bool = False,
        use_torch_tanh_fallback: bool = False,
    ) -> None:
        self.device = device
        self.vocab = ref.vocab
        self.context_length = ref.context_length
        self.params = params
        self._ref_decoder = ref.decoder  # kept for lazy preprocess_tt_decoder calls
        self._use_stft_fallback = use_torch_stft_fallback
        self._use_phase_fallback = use_torch_phase_fallback
        self._use_linear_fallback = use_torch_linear_fallback
        self._use_tanh_fallback = use_torch_tanh_fallback

        self._bert = TTCustomAlbert(device, params.bert)
        self._predictor = TTProsodyPredictor(device, params.predictor)
        self._text_encoder = TTTextEncoder(device, params.text_encoder)

        # Decoder params cached per T_mel (STFT precomputation depends on sequence length).
        self._decoder_cache: dict[int, TTDecoder] = {}

    # ------------------------------------------------------------------
    # Decoder cache
    # ------------------------------------------------------------------

    def _get_decoder(self, t_mel: int) -> TTDecoder:
        """Return cached :class:`TTDecoder` for the given mel-frame count, uploading on first access."""
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
        """End-to-end TT inference: phonemes → audio waveform.

        Args:
            phonemes: IPA phoneme string (e.g. from ``KPipeline``).
            ref_s: ``[1, style_dim*2]`` voice style embedding (CPU).
            speed: Speaking-rate multiplier.
            deterministic: Zero all stochastic noise (torch.rand / torch.randn_like)
                for reproducible output when generator fallbacks are enabled.

        Returns:
            :class:`Output` with ``.audio`` (1-D float32 waveform on CPU) and
            ``.pred_dur`` (per-phoneme frame count on CPU).
        """
        # --- Tokenise --------------------------------------------------
        input_ids_list = list(filter(lambda i: i is not None, map(lambda p: self.vocab.get(p), phonemes)))
        assert len(input_ids_list) + 2 <= self.context_length, (
            len(input_ids_list) + 2,
            self.context_length,
        )
        input_ids = torch.LongTensor([[0, *input_ids_list, 0]])  # [1, T]

        ref_s = ref_s.cpu()
        if ref_s.dim() == 1:
            ref_s = ref_s.unsqueeze(0)

        B, T = input_ids.shape
        input_lengths = torch.full((B,), T, dtype=torch.long)
        # text_mask[b, t] = True where position is padding (beyond valid length)
        text_mask = torch.arange(T).unsqueeze(0).expand(B, -1).type_as(input_lengths)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))  # [B, T] bool
        attention_mask = (~text_mask).int()  # [B, T]  1=keep, 0=pad

        # Style slices (CPU, will be pushed to device when needed)
        s_pred_cpu = ref_s[:, self.params.style_dim :]  # [B, style_dim] for predictor
        s_style_cpu = ref_s[:, : self.params.style_dim]  # [B, style_dim] for decoder

        mc = ttnn.DRAM_MEMORY_CONFIG
        ck = self._predictor.compute_kernel_config
        lengths_list: list[int] = input_lengths.tolist()

        ctx = _zero_noise() if deterministic else _noop()
        with ctx:
            audio_tt = self._device_forward(
                input_ids,
                attention_mask,
                text_mask,
                input_lengths,
                lengths_list,
                s_pred_cpu,
                s_style_cpu,
                speed,
                mc,
                ck,
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
        attention_mask: torch.Tensor,
        text_mask: torch.Tensor,
        input_lengths: torch.LongTensor,
        lengths_list: list,
        s_pred_cpu: torch.Tensor,
        s_style_cpu: torch.Tensor,
        speed: float,
        mc: ttnn.MemoryConfig,
        ck,
    ) -> ttnn.Tensor:
        B, T = input_ids.shape
        p = self.params
        dev = self.device

        # ------ 1. PL-BERT -----------------------------------------------
        bert_out = self._bert(input_ids, attention_mask)  # [B, T, hidden_size] NLC

        # ------ 2. bert_encoder: Linear(hidden_size → hidden_dim) --------
        d_en = ttnn.linear(
            bert_out,
            p.bert_encoder_w,
            bias=p.bert_encoder_b,
            transpose_b=True,
            memory_config=mc,
            compute_kernel_config=ck,
        )  # [B, T, hidden_dim] NLC
        ttnn.deallocate(bert_out)

        # Squeeze any leading singleton dims that ttnn.linear may add
        while len(d_en.shape) > 3:
            d_en = ttnn.squeeze(d_en, 0)

        d_en_bct = ttnn.permute(d_en, (0, 2, 1), memory_config=mc)  # [B, hidden_dim, T]
        ttnn.deallocate(d_en)

        # ------ 3. Style on device ----------------------------------------
        s_pred_tt = ttnn.from_torch(
            s_pred_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc
        )

        # ------ 4. DurationEncoder ----------------------------------------
        keep_mask = ttnn.from_torch(
            (~text_mask).to(torch.float32).unsqueeze(-1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=mc,
        )
        d_nlc = self._predictor._text_encoder.forward(
            d_en_bct=d_en_bct,
            style_bs=s_pred_tt,
            sequence_lengths=lengths_list,
            keep_mask_btl=keep_mask,
            compute_kernel_config=ck,
            memory_config=mc,
        )  # [B, T, d_hid + style_dim]
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
        )  # [B, T, d_hid]
        duration = self._predictor._duration_proj.forward(x_lstm, compute_kernel_config=ck, memory_config=mc)
        ttnn.deallocate(x_lstm)
        # duration: [B, T, max_dur]

        # ------ 6. Alignment (host step: read pred_dur, build matrix) -----
        dur_cpu = ttnn.to_torch(duration).float()  # [B, T, max_dur]
        ttnn.deallocate(duration)
        dur_sum = torch.sigmoid(dur_cpu).sum(dim=-1) / speed  # [B, T]
        pred_dur = torch.round(dur_sum).clamp(min=1).long().squeeze()  # [T]

        aln_cpu = _build_alignment(pred_dur)  # [1, T, T_aligned]
        T_aligned = int(aln_cpu.shape[2])

        aln_tt = ttnn.from_torch(aln_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)

        # ------ 7. en_nlc = aln^T @ d_nlc --------------------------------
        aln_Ta_T = ttnn.permute(aln_tt, (0, 2, 1), memory_config=mc)  # [1, T_aligned, T]
        en_nlc = ttnn.matmul(aln_Ta_T, d_nlc, memory_config=mc, compute_kernel_config=ck)
        # [1, T_aligned, d_hid + style_dim]
        ttnn.deallocate(aln_Ta_T)
        ttnn.deallocate(d_nlc)

        # ------ 8. F0 / N from predictor ---------------------------------
        F0, N = self._predictor.F0Ntrain(en_nlc, s_pred_tt, memory_config=mc)
        # F0, N: [B, T_f0] where T_f0 = 2 * T_aligned
        ttnn.deallocate(en_nlc)
        ttnn.deallocate(s_pred_tt)

        # ------ 9. TextEncoder + asr = t_en @ aln -------------------------
        t_en_bct = self._text_encoder(input_ids, input_lengths=input_lengths, text_mask=text_mask)
        # [B, hidden_dim, T] BCT

        asr_bct = ttnn.matmul(t_en_bct, aln_tt, memory_config=mc, compute_kernel_config=ck)
        # [B, hidden_dim, T_aligned] BCT
        ttnn.deallocate(t_en_bct)
        ttnn.deallocate(aln_tt)

        asr_nlc = ttnn.permute(asr_bct, (0, 2, 1), memory_config=mc)  # [B, T_aligned, hidden_dim]
        ttnn.deallocate(asr_bct)

        # ------ 10. Decoder (vocoder) ------------------------------------
        s_style_tt = ttnn.from_torch(
            s_style_cpu, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc
        )
        decoder = self._get_decoder(T_aligned)
        audio = decoder(asr_nlc, F0, N, s_style_tt, memory_config=mc)
        ttnn.deallocate(asr_nlc)
        ttnn.deallocate(F0)
        ttnn.deallocate(N)
        ttnn.deallocate(s_style_tt)

        return audio  # [B, 1, audio_len]


# ---------------------------------------------------------------------------
# Null context (Python 3.6 compatible via explicit class, avoids nullcontext)
# ---------------------------------------------------------------------------


@contextmanager
def _noop():
    yield
