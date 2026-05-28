# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of ``SeamlessM4Tv2CodeHifiGan`` (code HiFi-GAN wrapper).

The PyTorch reference is
``models/demos/facebook_seamless_m4t_v2_large/reference/functional.py::code_hifigan_vocoder_forward``
which reproduces the forward of HuggingFace
``transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2.SeamlessM4Tv2CodeHifiGan``.

Op sequence::

    hidden = unit_embedding(input_ids).transpose(1, 2)        # (B, C_u, T)
    spkr   = speaker_embedding(speaker_id).transpose(1, 2)    # (B, C_s, 1)
    lang   = language_embedding(lang_id).transpose(1, 2)      # (B, C_l, 1)
    log_dur = dur_predictor(hidden.transpose(1, 2))           # (B, T)
    dur_out = clamp(round(expm1(log_dur)).long(), min=1)      # (B, T)
    hidden = repeat_interleave(hidden, dur_out.view(-1), dim=2)  # (B, C_u, T_up)
    spkr = spkr.repeat(1, 1, T_up)
    lang = lang.repeat(1, 1, T_up)
    x = cat([lang, hidden, spkr], dim=1)                      # (B, C_l+C_u+C_s, T_up)
    waveform = hifigan_vocoder(x)                             # (B, T_out)

Implementation notes (TTNN port):

* ``ttnn.embedding`` is used for the dynamic ``input_ids -> unit_embedding``
  lookup. For ``speaker_id`` / ``lang_id`` (single rows on the default S2ST
  path) we gather the row host-side, broadcast it to the (B, 1) shape and
  upload as a tiny TTNN tensor. This matches the t2u_decoder host-resident
  pattern documented in
  ``models/demos/facebook_seamless_m4t_v2_large/tt/t2u_decoder.py``.
* The duration-driven hard-upsample (``repeat_interleave`` by predicted
  per-unit duration) is an acknowledged host-resident op: ``dur_out`` is an
  integer tensor whose values determine the upsampled length. This is the
  same pattern the t2u_decoder uses (host-side ``_hard_upsample`` step #3).
* Speaker/language broadcast + final concat happen on device in TILE layout
  using ``ttnn.repeat`` + ``ttnn.concat`` along the channel dim.
* The duration predictor and HiFi-GAN core are reused via the existing
  ``VariancePredictor`` and ``HifiGanVocoder`` TT modules.

The forward returns a single ttnn TILE_LAYOUT tensor of shape ``[B, T_out]``
to match the contract of the standalone ``hifigan_vocoder`` block. Per-batch
output lengths (an integer tensor) are computed host-side from ``dur_out``
and exposed via ``self.last_lengths`` for callers that need them (the trim
to ``lengths[b]`` is a host concern that lives outside this block).
"""

from __future__ import annotations

from typing import Sequence

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.facebook_seamless_m4t_v2_large.tt.hifigan_vocoder import HifiGanVocoder
from models.demos.facebook_seamless_m4t_v2_large.tt.variance_predictor import VariancePredictor


class CodeHifiGanVocoder(LightweightModule):
    """SeamlessM4T-v2 code HiFi-GAN wrapper around ``HifiGanVocoder``.

    Args:
        device: ttnn device.
        state_dict: nested dict matching ``code_hifigan_vocoder_forward``:
            - ``"unit_embedding"``    = ``{"weight": [vocab, C_u]}``
            - ``"speaker_embedding"`` = ``{"weight": [num_spkrs, C_s]}``
            - ``"language_embedding"``= ``{"weight": [num_langs, C_l]}``
            - ``"dur_predictor"``     = state dict for ``VariancePredictor``
            - ``"hifi_gan"``          = state dict for ``HifiGanVocoder``
        pad_token_id: ``config.t2u_pad_token_id`` (used to compute lengths).
        variance_predictor_kernel_size: kernel size of duration predictor.
        variance_predictor_eps: LayerNorm eps in the duration predictor.
        upsample_rates / upsample_kernel_sizes / resblock_kernel_sizes /
        resblock_dilation_sizes / leaky_relu_slope: forwarded to
            ``HifiGanVocoder`` (defaults match SeamlessM4T-v2-Large).
        weight_dtype: storage dtype for weights.
    """

    def __init__(
        self,
        device,
        state_dict: dict,
        pad_token_id: int = 1,
        variance_predictor_kernel_size: int = 3,
        variance_predictor_eps: float = 1e-5,
        upsample_rates: Sequence[int] = (5, 4, 4, 2, 2),
        upsample_kernel_sizes: Sequence[int] = (11, 8, 8, 4, 4),
        resblock_kernel_sizes: Sequence[int] = (3, 7, 11),
        resblock_dilation_sizes: Sequence[Sequence[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        leaky_relu_slope: float = 0.1,
        weight_dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.pad_token_id = int(pad_token_id)
        self.weight_dtype = weight_dtype
        self.upsample_rates = tuple(int(s) for s in upsample_rates)
        self.upsample_kernel_sizes = tuple(int(k) for k in upsample_kernel_sizes)
        self.resblock_kernel_sizes = tuple(int(k) for k in resblock_kernel_sizes)
        self.resblock_dilation_sizes = tuple(tuple(int(d) for d in row) for row in resblock_dilation_sizes)
        self.leaky_relu_slope = float(leaky_relu_slope)

        # --- Embedding tables ------------------------------------------------
        # Unit embedding: dynamic gather via ttnn.embedding. Table is kept in
        # ROW_MAJOR DRAM as required by the gather op (mirrors scaled_word_embedding).
        unit_emb_w = state_dict["unit_embedding"]["weight"]
        self.unit_vocab_size, self.unit_embed_dim = int(unit_emb_w.shape[0]), int(unit_emb_w.shape[1])
        self.unit_embedding_weight = ttnn.from_torch(
            unit_emb_w,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Speaker / language embeddings: a single row is gathered host-side
        # (a tiny B*C transfer to device). This is consistent with the
        # t2u_decoder pattern of keeping data-dependent integer ops on host.
        spkr_emb_w = state_dict["speaker_embedding"]["weight"]
        lang_emb_w = state_dict["language_embedding"]["weight"]
        self.spkr_embed_dim = int(spkr_emb_w.shape[1])
        self.lang_embed_dim = int(lang_emb_w.shape[1])
        # Stash as host tensors -- the gather happens on every forward call.
        self._spkr_embedding_host = spkr_emb_w.contiguous()
        self._lang_embedding_host = lang_emb_w.contiguous()

        # --- Duration predictor (VariancePredictor) --------------------------
        dur_sd = state_dict["dur_predictor"]
        self.dur_predictor = VariancePredictor(
            device=device,
            conv1_weight=dur_sd["conv1"]["weight"],
            conv1_bias=dur_sd["conv1"]["bias"],
            ln1_weight=dur_sd["ln1"]["weight"],
            ln1_bias=dur_sd["ln1"]["bias"],
            conv2_weight=dur_sd["conv2"]["weight"],
            conv2_bias=dur_sd["conv2"]["bias"],
            ln2_weight=dur_sd["ln2"]["weight"],
            ln2_bias=dur_sd["ln2"]["bias"],
            proj_weight=dur_sd["proj"]["weight"],
            proj_bias=dur_sd["proj"]["bias"],
            kernel_size=int(variance_predictor_kernel_size),
            eps=float(variance_predictor_eps),
            weight_dtype=weight_dtype,
        )

        # --- HiFi-GAN vocoder core -------------------------------------------
        self.hifi_gan = HifiGanVocoder(
            device=device,
            state_dict=state_dict["hifi_gan"],
            upsample_rates=self.upsample_rates,
            upsample_kernel_sizes=self.upsample_kernel_sizes,
            resblock_kernel_sizes=self.resblock_kernel_sizes,
            resblock_dilation_sizes=self.resblock_dilation_sizes,
            leaky_relu_slope=self.leaky_relu_slope,
            weight_dtype=weight_dtype,
        )

        # Side-channel for callers that want the per-batch valid sample
        # counts. Populated on every forward.
        self.last_dur_out: torch.Tensor | None = None
        self.last_lengths: torch.Tensor | None = None

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _hard_upsample_host(hidden_bct: torch.Tensor, dur_out: torch.Tensor) -> torch.Tensor:
        """Host-side ``repeat_interleave`` over time, matching HF.

        ``hidden_bct`` is ``[B, C, T]``. For ``B == 1`` HF uses a single
        ``torch.repeat_interleave``; for ``B > 1`` it interleaves per-sample
        and ``pad_sequence``-pads. We mirror that exactly.
        """
        batch = hidden_bct.shape[0]
        if batch == 1:
            return torch.repeat_interleave(hidden_bct, dur_out.view(-1), dim=2)
        pieces = [torch.repeat_interleave(h, d, dim=-1).transpose(0, 1) for (h, d) in zip(hidden_bct, dur_out)]
        return torch.nn.utils.rnn.pad_sequence(pieces, batch_first=True).transpose(1, 2)

    @staticmethod
    def _get_dur_output_lengths(input_ids: torch.Tensor, dur_out: torch.Tensor, pad_token_id: int) -> torch.Tensor:
        """Port of HF ``SeamlessM4Tv2CodeHifiGan._get_dur_output_lengths``."""
        unit_lengths = (input_ids != pad_token_id).sum(1)
        unit_lengths = torch.clamp(unit_lengths, 0, dur_out.shape[1] - 1)
        cumulative_dur_out = torch.cumsum(dur_out, dim=1)
        unit_lengths = cumulative_dur_out.gather(dim=1, index=unit_lengths.unsqueeze(1)).squeeze()
        return unit_lengths

    @staticmethod
    def _get_output_hifigan_lengths(
        input_lengths,
        upsample_rates: Sequence[int],
        upsample_kernel_sizes: Sequence[int],
        resblock_kernel_sizes: Sequence[int],
        resblock_dilation_sizes: Sequence[Sequence[int]],
    ):
        """Port of HF ``SeamlessM4Tv2CodeHifiGan._get_output_hifigan_lengths``.

        Walks the vocoder conv stack to compute per-batch valid output sample
        counts. Pure integer host op.
        """

        def _conv_out_length(input_length, kernel_size, stride, pad, dilation=1):
            if isinstance(input_length, torch.Tensor):
                return (
                    torch.div(
                        input_length + 2 * pad - dilation * (kernel_size - 1) - 1,
                        stride,
                        rounding_mode="floor",
                    )
                    + 1
                )
            return (input_length + 2 * pad - dilation * (kernel_size - 1) - 1) // stride + 1

        def _transpose_conv_out_length(input_length, kernel_size, stride, pad, dilation=1):
            return (input_length - 1) * stride - 2 * pad + dilation * (kernel_size - 1) + 1

        # conv_pre: kernel=7, stride=1, pad=3.
        input_lengths = _conv_out_length(input_lengths, 7, 1, 3)
        # upsampler ConvTranspose1d's
        for i, (up_rate, up_kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            pad = (up_kernel - up_rate) // 2
            input_lengths = _transpose_conv_out_length(input_lengths, up_kernel, up_rate, pad)
            # MRF resblocks: each kernel size, dilations applied as kernel=k, stride=1, pad=(k*d - d)//2
            for k, dilations in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                # convs1 (dilated) + convs2 (dilation=1) -- each preserves length.
                for d in dilations:
                    pad = ((k - 1) * d) // 2
                    input_lengths = _conv_out_length(input_lengths, k, 1, pad, dilation=d)
                    # convs2 has kernel=k, dilation=1, pad=(k-1)//2.
                    pad2 = (k - 1) // 2
                    input_lengths = _conv_out_length(input_lengths, k, 1, pad2, dilation=1)
        # conv_post: kernel=7, stride=1, pad=3.
        input_lengths = _conv_out_length(input_lengths, 7, 1, 3)
        return input_lengths

    # ----------------------------------------------------------------- forward

    def forward(
        self,
        input_ids: torch.Tensor,
        speaker_id: torch.Tensor,
        lang_id: torch.Tensor,
    ) -> ttnn.Tensor:
        """Run the code HiFi-GAN vocoder.

        Args:
            input_ids: host-side ``[B, T]`` long tensor of RVQ unit ids.
            speaker_id: host-side ``[B]`` or ``[B, 1]`` long tensor of speaker
                indices.
            lang_id: host-side ``[B]`` or ``[B, 1]`` long tensor of language
                indices.

        Returns:
            ttnn TILE_LAYOUT tensor of shape ``[B, T_out]`` in [-1, 1]
            representing the synthesised waveform. ``self.last_dur_out`` and
            ``self.last_lengths`` are also populated for callers that need
            per-batch valid sample counts.
        """
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be [B, T], got shape {tuple(input_ids.shape)}")
        batch = int(input_ids.shape[0])
        seq_in = int(input_ids.shape[1])

        # ------------------------------------------------------------------
        # 1. Unit embedding lookup on device (ttnn.embedding).
        #    Speaker/language are single-row gathers done host-side.
        # ------------------------------------------------------------------
        unit_ids_tt = ttnn.from_torch(
            input_ids.to(torch.int32),
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Returns [B, T, C_u] TILE_LAYOUT.
        unit_embeds_btC = ttnn.embedding(
            unit_ids_tt,
            self.unit_embedding_weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(unit_ids_tt)

        # ------------------------------------------------------------------
        # 2. Duration predictor. Expects [B, T, C_u] TILE; returns [B, T].
        # ------------------------------------------------------------------
        log_dur_tt = self.dur_predictor(unit_embeds_btC)
        log_dur = ttnn.to_torch(log_dur_tt).to(torch.float32).reshape(batch, seq_in)
        ttnn.deallocate(log_dur_tt)
        dur_out = torch.clamp(torch.round(torch.expm1(log_dur)).long(), min=1)  # [B, T]
        self.last_dur_out = dur_out

        # ------------------------------------------------------------------
        # 3. Hard-upsample unit embeddings by ``dur_out`` on host.
        #    NOTE: this is an acknowledged host-resident op -- ``dur_out`` is
        #    integer and controls the resulting tensor length. Mirrors the
        #    t2u_decoder._hard_upsample pattern.
        # ------------------------------------------------------------------
        unit_embeds_host = ttnn.to_torch(unit_embeds_btC).to(torch.float32).reshape(batch, seq_in, self.unit_embed_dim)
        ttnn.deallocate(unit_embeds_btC)
        # [B, T, C_u] -> [B, C_u, T] then repeat_interleave on time.
        hidden_bct = unit_embeds_host.transpose(1, 2).contiguous()
        hidden_bct = self._hard_upsample_host(hidden_bct, dur_out)  # [B, C_u, T_up]
        t_up = int(hidden_bct.shape[-1])

        # ------------------------------------------------------------------
        # 4. Speaker / language row lookups + broadcast to T_up.
        #    Done on host because speaker_id/lang_id are single-row gathers.
        # ------------------------------------------------------------------
        # Normalise speaker_id/lang_id to [B]; HF allows [B] or [B, 1].
        spkr_idx = speaker_id.long().view(batch, -1)[:, 0]
        lang_idx = lang_id.long().view(batch, -1)[:, 0]
        # Gather rows: [B, C_s], [B, C_l].
        spkr_rows = self._spkr_embedding_host.index_select(0, spkr_idx)  # [B, C_s]
        lang_rows = self._lang_embedding_host.index_select(0, lang_idx)  # [B, C_l]
        # Broadcast to time: [B, C, T_up].
        spkr_bct = spkr_rows.unsqueeze(-1).expand(batch, self.spkr_embed_dim, t_up).contiguous()
        lang_bct = lang_rows.unsqueeze(-1).expand(batch, self.lang_embed_dim, t_up).contiguous()

        # ------------------------------------------------------------------
        # 5. Concatenate [lang, hidden, spkr] along the channel dim on host
        #    (single contiguous upload) and push to device for the vocoder.
        # ------------------------------------------------------------------
        cat_features_host = torch.cat([lang_bct, hidden_bct, spkr_bct], dim=1)  # [B, C_l+C_u+C_s, T_up]
        cat_features_tt = ttnn.from_torch(
            cat_features_host.to(torch.bfloat16) if self.weight_dtype == ttnn.bfloat16 else cat_features_host,
            device=self.device,
            dtype=self.weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # ------------------------------------------------------------------
        # 6. HiFi-GAN vocoder core -> waveform [B, T_out].
        # ------------------------------------------------------------------
        waveform_tt = self.hifi_gan(cat_features_tt)
        ttnn.deallocate(cat_features_tt)

        # ------------------------------------------------------------------
        # 7. Compute per-batch valid output sample counts (host-side).
        # ------------------------------------------------------------------
        unit_lengths = self._get_dur_output_lengths(input_ids, dur_out, self.pad_token_id)
        self.last_lengths = self._get_output_hifigan_lengths(
            unit_lengths,
            upsample_rates=self.upsample_rates,
            upsample_kernel_sizes=self.upsample_kernel_sizes,
            resblock_kernel_sizes=self.resblock_kernel_sizes,
            resblock_dilation_sizes=self.resblock_dilation_sizes,
        )

        return waveform_tt
