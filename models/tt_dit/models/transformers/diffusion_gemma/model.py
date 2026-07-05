# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Top-level DiffusionGemma model composition.

Mirrors ``DiffusionGemmaModel`` and ``DiffusionGemmaForBlockDiffusion`` from
``transformers.models.diffusion_gemma.modeling_diffusion_gemma``:

  - ``DiffusionGemmaModel`` runs the (multimodal) encoder to populate a per-layer K/V
    cache, then runs the decoder canvas through ``DiffusionGemmaDecoderModel``,
    cross-attending to that cache. From the decoder's perspective the cache is read-only.
  - ``DiffusionGemmaForBlockDiffusion`` applies ``lm_head`` + tanh logit-softcap on top
    of the decoder's last hidden states to produce ``[B, canvas_length, vocab_size]``
    logits ready for the diffusion sampler.
"""

from __future__ import annotations

import torch

import ttnn

from ....layers.linear import ColParallelLinear, Linear
from ....layers.module import Module
from ....parallel.config import DiTParallelConfig
from .encoder_model import DiffusionGemmaEncoderModel
from .text_decoder import DiffusionGemmaDecoderModel


class DiffusionGemmaModel(Module):
    """Encoder + decoder composition."""

    def __init__(
        self,
        *,
        encoder: DiffusionGemmaEncoderModel,
        decoder: DiffusionGemmaDecoderModel,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        *,
        input_ids: ttnn.Tensor,
        position_ids: torch.Tensor,
        encoder_attention_masks: dict[str, ttnn.Tensor | None],
        decoder_input_ids: ttnn.Tensor,
        decoder_position_ids: torch.Tensor,
        decoder_attention_masks: dict[str, ttnn.Tensor | None],
        pixel_values: ttnn.Tensor | None = None,
        pixel_position_ids: torch.Tensor | None = None,
        padding_positions: torch.Tensor | None = None,
        input_ids_host: torch.Tensor | None = None,
        self_conditioning_signal: ttnn.Tensor | None = None,
    ) -> tuple[ttnn.Tensor, list[tuple[ttnn.Tensor, ttnn.Tensor]]]:
        """Run encoder to populate the cache, then decoder with that cache."""
        encoder_h, encoder_kv = self.encoder(
            input_ids,
            position_ids,
            encoder_attention_masks,
            pixel_values=pixel_values,
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
            input_ids_host=input_ids_host,
        )
        # Encoder hidden state is currently unused by the decoder (per HF, it only consumes the KV cache),
        # but we return it so a caller could use it for an auxiliary encoder loss / inspection.
        decoder_h = self.decoder(
            decoder_input_ids,
            decoder_position_ids,
            encoder_kv_cache=encoder_kv,
            decoder_attention_masks=decoder_attention_masks,
            self_conditioning_signal=self_conditioning_signal,
        )
        return decoder_h, encoder_kv


class DiffusionGemmaForBlockDiffusion(Module):
    """LM-head wrapper that produces softcapped logits over the canvas."""

    def __init__(
        self,
        *,
        model: DiffusionGemmaModel,
        text_hidden_size: int,
        vocab_size: int,
        final_logit_softcapping: float,
        mesh_device: ttnn.MeshDevice,
        parallel_config: DiTParallelConfig | None = None,
        ccl_manager=None,
    ) -> None:
        super().__init__()
        self.model = model
        self.final_logit_softcapping = final_logit_softcapping
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        # lm_head: hidden_size → vocab_size. TP-shard on the vocab axis when a parallel_config
        # is provided (vocab 262144 × hidden 2816 at bf16 is ~1.4 GB per device replicated —
        # halved with TP=2, quartered with TP=4). Halving alone is what makes the full-config
        # model fit on WH T3K DRAM alongside the layer stack. Falls back to a plain replicated
        # Linear when no parallel_config is passed (existing standalone tests).
        if parallel_config is not None and parallel_config.tensor_parallel.factor > 1:
            self.lm_head = ColParallelLinear(
                text_hidden_size,
                vocab_size,
                bias=False,
                mesh_device=mesh_device,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                ccl_manager=ccl_manager,
            )
            self._lm_head_tp = True
        else:
            self.lm_head = Linear(text_hidden_size, vocab_size, bias=False, mesh_device=mesh_device)
            self._lm_head_tp = False

    def _prepare_torch_state(self, state) -> None:
        """HF ties ``lm_head.weight`` to ``model.decoder.embed_tokens.weight``. If the loaded
        state-dict already populates both keys (untied) we leave them; if only one is present
        we duplicate it under the other name so the loader finds both."""
        if "lm_head.weight" not in state and "model.decoder.embed_tokens.weight" in state:
            state["lm_head.weight"] = state["model.decoder.embed_tokens.weight"]

    def tie_shared_embeddings(self) -> None:
        """Alias ``encoder.embed_tokens.weight`` to ``decoder.embed_tokens.weight`` on device.

        HF ties ``encoder.embed ↔ decoder.embed ↔ lm_head`` via ``tie_word_embeddings=True``.
        In our TT model each is a separate ``Parameter`` with its own device allocation, so
        we pay 3x the vocab-embedding storage (~1.4 GB each at bf16 for vocab=262144,
        hidden=2816). This method aliases encoder → decoder to save one copy (~1.4 GB per
        device). ``lm_head.weight`` stores its data transposed (Linear semantics: [in, out])
        while embedding tables are [vocab, hidden], so lm_head can't be trivially aliased —
        it stays separate.

        Call after ``load_state_dict`` so both parameters have their loaded (independent)
        tensors first; we deallocate encoder's copy and re-point it at the decoder's.
        """
        encoder_embed = self.model.encoder.language_model.embed_tokens
        decoder_embed = self.model.decoder.embed_tokens

        # Free the encoder's independent allocation before aliasing.
        if encoder_embed.weight._data is not None:
            ttnn.deallocate(encoder_embed.weight._data)
        encoder_embed.weight._data = decoder_embed.weight._data

        # embed_scale is a scalar buffer (value = sqrt(hidden_size)) — safe to leave as two
        # separate tiny [1, 1] tensors; not worth the aliasing complexity.

    def forward(
        self,
        **kwargs,
    ) -> ttnn.Tensor:
        """Run the full model (encoder + decoder + lm_head + softcap). Returns logits ``[B, canvas, vocab]``.

        For the diffusion sampling loop the encoder runs *once* per canvas and the decoder
        runs many times — use :meth:`encoder_step` + :meth:`decoder_step` directly to avoid
        re-running the encoder per denoising step.
        """
        decoder_h, _encoder_kv = self.model(**kwargs)
        return self._apply_lm_head(decoder_h)

    def encoder_step(
        self,
        input_ids: "ttnn.Tensor",
        position_ids,
        encoder_attention_masks: dict,
        *,
        pixel_values=None,
        pixel_position_ids=None,
        padding_positions=None,
        input_ids_host=None,
    ) -> tuple["ttnn.Tensor", list]:
        """Run only the encoder (multimodal). Returns (encoder_hidden, per_layer_kv)."""
        return self.model.encoder(
            input_ids,
            position_ids,
            encoder_attention_masks,
            pixel_values=pixel_values,
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
            input_ids_host=input_ids_host,
        )

    def decoder_step(
        self,
        decoder_input_ids: "ttnn.Tensor",
        decoder_position_ids,
        encoder_kv_cache: list,
        decoder_attention_masks: dict,
        self_conditioning_signal: "ttnn.Tensor | None" = None,
    ) -> "ttnn.Tensor":
        """Run one decoder forward + lm_head + softcap. Used per diffusion step.

        The encoder cache is read-only here; the decoder does NOT modify it.
        """
        decoder_h = self.model.decoder(
            decoder_input_ids,
            decoder_position_ids,
            encoder_kv_cache=encoder_kv_cache,
            decoder_attention_masks=decoder_attention_masks,
            self_conditioning_signal=self_conditioning_signal,
        )
        return self._apply_lm_head(decoder_h)

    def _apply_lm_head(self, decoder_h: "ttnn.Tensor") -> "ttnn.Tensor":
        """Project decoder hidden states to vocab logits and apply tanh softcap on-device."""
        logits = self.lm_head(decoder_h)
        ttnn.deallocate(decoder_h)
        # When lm_head is ColParallelLinear, its output is TP-fractured on the vocab axis
        # ([1, B, S, vocab/tp]). Gather it back so downstream (host-side sampler) sees the
        # full vocab. When lm_head is plain Linear, this is a no-op.
        if getattr(self, "_lm_head_tp", False) and self.parallel_config.tensor_parallel.factor > 1:
            # ``use_persistent_buffer=False`` because the layer stack's all_gathers use the
            # same input shape and share the ping-pong cache — leaving that state around
            # manifests as "Input Tensor is not allocated" on the *next* denoising step's
            # first all_gather. lm_head runs once per denoising step so the fresh allocation
            # cost is negligible (same fix pattern as ``self_conditioning``'s all_gather).
            logits = self.ccl_manager.all_gather(
                logits,
                dim=3,
                mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
                use_hyperparams=False,
                use_persistent_buffer=False,
            )
        cap = self.final_logit_softcapping
        scaled = ttnn.multiply(logits, 1.0 / cap)
        ttnn.deallocate(logits)
        capped = ttnn.tanh(scaled)
        ttnn.deallocate(scaled)
        out = ttnn.multiply(capped, cap)
        ttnn.deallocate(capped)
        return out
