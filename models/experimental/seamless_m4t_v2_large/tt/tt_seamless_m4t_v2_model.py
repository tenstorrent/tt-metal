# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN [`SeamlessM4Tv2Model`]: composes text/speech encoders, text decoder, T2U, vocoder, and ``lm_head``."""

from __future__ import annotations

from typing import Any, Optional

import ttnn

from models.experimental.seamless_m4t_v2_large.tt.tt_code_hifigan import TTSeamlessM4Tv2CodeHifiGan
from models.experimental.seamless_m4t_v2_large.tt.tt_speech_encoder import TTSeamlessM4Tv2SpeechEncoder
from models.experimental.seamless_m4t_v2_large.tt.tt_text_decoder import TTSeamlessM4Tv2Decoder
from models.experimental.seamless_m4t_v2_large.tt.tt_text_encoder import TTSeamlessM4Tv2Encoder
from models.experimental.seamless_m4t_v2_large.tt.tt_text_to_unit import (
    TTSeamlessM4Tv2TextToUnitForConditionalGeneration,
)


def _core_grid(device: ttnn.Device) -> ttnn.CoreGrid:
    grid = device.compute_with_storage_grid_size()
    return ttnn.CoreGrid(y=grid.y, x=grid.x)


class TTSeamlessM4Tv2Model:
    """
    TTNN port of Hugging Face ``SeamlessM4Tv2Model``.

    Holds the same submodules as the PyTorch model (``text_encoder``, ``speech_encoder``, ``text_decoder``,
    ``t2u_model``, ``vocoder``, main ``lm_head``). ``forward_text`` matches the HF text-modality
    ``forward`` path (encoder → decoder → ``lm_head``).
    """

    def __init__(
        self,
        device: ttnn.Device,
        parameters: Any,
        *,
        # Parent (main) config scalars
        layer_norm_eps: float,
        encoder_layers: int,
        encoder_attention_heads: int,
        decoder_layers: int,
        decoder_attention_heads: int,
        hidden_size: int,
        feature_projection_input_dim: int,
        speech_encoder_attention_heads: int,
        speech_encoder_intermediate_size: int,
        speech_encoder_layers: int,
        speech_encoder_chunk_size: Optional[int],
        speech_encoder_left_chunk_num: int,
        # t2u submodule config (``model.t2u_model.config``)
        t2u_layer_norm_eps: float,
        t2u_encoder_layers: int,
        t2u_encoder_attention_heads: int,
        t2u_decoder_layers: int,
        t2u_decoder_attention_heads: int,
        t2u_pad_token_id: int,
        variance_predictor_embed_dim: int,
        variance_predictor_hidden_dim: int,
        variance_predictor_kernel_size: int,
        # Object passed to TT vocoder (expects HF-style ``config`` attributes)
        vocoder_config: Any,
    ):
        self.device = device
        self.parameters = parameters
        self.vocoder_config = vocoder_config

        self.text_encoder = TTSeamlessM4Tv2Encoder(
            device,
            parameters.text_encoder,
            layer_norm_eps=layer_norm_eps,
            num_hidden_layers=encoder_layers,
            num_attention_heads=encoder_attention_heads,
            hidden_size=hidden_size,
        )
        self.text_decoder = TTSeamlessM4Tv2Decoder(
            device,
            parameters.text_decoder,
            layer_norm_eps=layer_norm_eps,
            num_hidden_layers=decoder_layers,
            num_attention_heads=decoder_attention_heads,
            hidden_size=hidden_size,
        )
        self.speech_encoder = TTSeamlessM4Tv2SpeechEncoder(
            device,
            parameters.speech_encoder,
            hidden_size=hidden_size,
            feature_projection_input_dim=feature_projection_input_dim,
            speech_encoder_attention_heads=speech_encoder_attention_heads,
            speech_encoder_intermediate_size=speech_encoder_intermediate_size,
            speech_encoder_layers=speech_encoder_layers,
            layer_norm_eps=layer_norm_eps,
            speech_encoder_chunk_size=speech_encoder_chunk_size,
            speech_encoder_left_chunk_num=speech_encoder_left_chunk_num,
        )
        self.t2u = TTSeamlessM4Tv2TextToUnitForConditionalGeneration(
            device,
            parameters.t2u,
            layer_norm_eps=t2u_layer_norm_eps,
            encoder_layers=t2u_encoder_layers,
            encoder_attention_heads=t2u_encoder_attention_heads,
            decoder_layers=t2u_decoder_layers,
            decoder_attention_heads=t2u_decoder_attention_heads,
            hidden_size=hidden_size,
            pad_token_id=t2u_pad_token_id,
            variance_predictor_embed_dim=variance_predictor_embed_dim,
            variance_predictor_hidden_dim=variance_predictor_hidden_dim,
            variance_predictor_kernel_size=variance_predictor_kernel_size,
        )
        self.vocoder = TTSeamlessM4Tv2CodeHifiGan(device, parameters.vocoder, vocoder_config)

        self._linear_ln_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward_text(
        self,
        encoder_input_ids: ttnn.Tensor,
        encoder_position_ids: ttnn.Tensor,
        encoder_attention_mask_4d: ttnn.Tensor,
        decoder_input_ids: ttnn.Tensor,
        decoder_position_ids: ttnn.Tensor,
        decoder_causal_mask_4d: ttnn.Tensor,
        decoder_cross_mask_4d: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Text modality: same dataflow as HF ``SeamlessM4Tv2Model.forward`` with ``input_ids`` set.

        Args:
            encoder_input_ids: ``uint32`` ``[B, enc_seq]``.
            encoder_position_ids: ``uint32`` ``[B, enc_seq]``.
            encoder_attention_mask_4d: additive self-attention mask ``[B, 1, enc_seq, enc_seq]`` bf16.
            decoder_input_ids: ``uint32`` ``[B, dec_seq]``.
            decoder_position_ids: ``uint32`` ``[B, dec_seq]``.
            decoder_causal_mask_4d: additive causal mask ``[B, 1, dec_seq, dec_seq]`` bf16.
            decoder_cross_mask_4d: additive cross mask ``[B, 1, dec_seq, enc_seq]`` bf16.

        Returns:
            LM logits ``[B, dec_seq, vocab_size]`` bf16 on device.
        """
        enc_out = self.text_encoder.forward(
            encoder_input_ids,
            encoder_position_ids,
            encoder_attention_mask_4d,
        )
        dec_out = self.text_decoder.forward(
            decoder_input_ids,
            decoder_position_ids,
            enc_out,
            decoder_causal_mask_4d,
            decoder_cross_mask_4d,
        )
        ttnn.deallocate(enc_out)
        logits = ttnn.linear(
            dec_out,
            self.parameters.lm_head.weight,
            bias=None,
            core_grid=_core_grid(self.device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._linear_ln_compute_cfg,
        )
        ttnn.deallocate(dec_out)
        return logits
