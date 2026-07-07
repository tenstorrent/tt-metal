# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full DiffusionGemma decoder text model.

Mirrors ``DiffusionGemmaDecoderModel`` from
``transformers.models.diffusion_gemma.modeling_diffusion_gemma``. Differs from the
encoder text model in two ways:

  1. Has a ``self_conditioning`` block that combines the input canvas embeddings
     with soft-embeddings derived from the previous denoising step's logits.
  2. Each layer's self-attention cross-attends to the encoder's read-only KV cache,
     concatenated with the canvas's own K/V along the seq axis.

Decoder positions continue *after* the encoder sequence, so ``decoder_position_ids``
are offset by the encoder's cache length.
"""

from __future__ import annotations

import torch

import ttnn

from ....encoders.gemma4.rope import Gemma4RotaryEmbedding
from ....layers.module import Module, ModuleList
from ....layers.normalization import RMSNorm
from ....parallel.config import DiTParallelConfig
from .embedding import DiffusionGemmaScaledWordEmbedding
from .layer import DiffusionGemmaLayer
from .self_conditioning import DiffusionGemmaSelfConditioning


class DiffusionGemmaDecoderModel(Module):
    """Embed → self_conditioning → 30 layers (with encoder cache cross-attn) → final norm.

    Optionally accepts pre-constructed ``shared_layers`` / ``shared_norm`` /
    ``shared_embed_tokens`` from a companion ``DiffusionGemmaEncoderTextModel``. When
    provided, this decoder reuses those Python objects instead of constructing new ones
    — the way HF's ``DiffusionGemmaModel._tied_weights_keys`` intends. Without sharing
    the decoder would double-allocate every layer weight, which OOMs the full 30-layer
    model on WH T3K DRAM (even at bfp8 experts). Standalone tests skip these kwargs and
    still work — they just pay the extra allocation.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        layer_types: list[str],
        num_attention_heads: int,
        num_key_value_heads: int,
        num_global_key_value_heads: int,
        head_dim: int,
        global_head_dim: int,
        sliding_window: int,
        num_experts: int,
        top_k_experts: int,
        moe_intermediate_size: int,
        rms_norm_eps: float,
        max_position_embeddings: int,
        sliding_rope_theta: float,
        full_rope_theta: float,
        full_partial_rotary_factor: float,
        pad_token_id: int,
        moe_state_dicts: list[dict] | None,
        mesh_device: ttnn.MeshDevice,
        ccl_manager,
        parallel_config: DiTParallelConfig,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        expert_dtype: ttnn.DataType = ttnn.bfloat16,
        router_dtype: ttnn.DataType = ttnn.bfloat16,
        tensor_cache_path: str | None = None,
        shared_layers: "ModuleList | None" = None,
        shared_norm: "RMSNorm | None" = None,
        shared_embed_tokens: "DiffusionGemmaScaledWordEmbedding | None" = None,
    ) -> None:
        """
        HF ties every layer weight between encoder and decoder (see
        ``DiffusionGemmaModel._tied_weights_keys``) — attention Q/K/V/O, MoE router+experts,
        MLP, layer norms, layer_scalar. Also ties encoder.norm ↔ decoder.norm and
        encoder.embed_tokens ↔ decoder.embed_tokens.

        To honor those ties on device (crucial to avoid 2× DRAM for the full 30-layer
        model on WH T3K), the pipeline builds the encoder first and passes its
        ``language_model.layers``, ``.norm``, and ``.embed_tokens`` here as
        ``shared_layers`` / ``shared_norm`` / ``shared_embed_tokens``. When provided,
        this decoder aliases them (same Python object → same underlying ttnn tensors)
        instead of constructing new instances that would double-allocate the weights.

        Standalone tests (that don't share) pass ``None`` for these and get an
        independent copy — the pre-sharing behavior.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.layer_types = list(layer_types)

        if moe_state_dicts is None:
            moe_state_dicts = [None] * num_hidden_layers
        assert len(moe_state_dicts) == num_hidden_layers

        if shared_embed_tokens is not None:
            self.embed_tokens = shared_embed_tokens
        else:
            self.embed_tokens = DiffusionGemmaScaledWordEmbedding(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                padding_idx=pad_token_id,
                mesh_device=mesh_device,
            )
        # self_conditioning is decoder-only (no encoder counterpart to tie against).
        self.self_conditioning = DiffusionGemmaSelfConditioning(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            rms_norm_eps=rms_norm_eps,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
        )

        if shared_layers is not None:
            assert (
                len(shared_layers) == num_hidden_layers
            ), f"shared_layers has {len(shared_layers)} entries, expected {num_hidden_layers}"
            self.layers = shared_layers
        else:
            self.layers = ModuleList(
                DiffusionGemmaLayer(
                    is_sliding=(layer_types[i] == "sliding_attention"),
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_attention_heads=num_attention_heads,
                    num_kv_heads=(
                        num_key_value_heads if layer_types[i] == "sliding_attention" else num_global_key_value_heads
                    ),
                    head_dim=head_dim if layer_types[i] == "sliding_attention" else global_head_dim,
                    sliding_window=sliding_window if layer_types[i] == "sliding_attention" else None,
                    num_experts=num_experts,
                    top_k_experts=top_k_experts,
                    moe_intermediate_size=moe_intermediate_size,
                    rms_norm_eps=rms_norm_eps,
                    moe_state_dict=moe_state_dicts[i],
                    mesh_device=mesh_device,
                    ccl_manager=ccl_manager,
                    parallel_config=parallel_config,
                    num_links=num_links,
                    topology=topology,
                    expert_dtype=expert_dtype,
                    router_dtype=router_dtype,
                    tensor_cache_path=(f"{tensor_cache_path}/layer_{i}" if tensor_cache_path else None),
                )
                for i in range(num_hidden_layers)
            )

        if shared_norm is not None:
            self.norm = shared_norm
        else:
            self.norm = RMSNorm(
                embedding_dim=hidden_size,
                norm_eps=rms_norm_eps,
                norm_elementwise_affine=True,
                bias=False,
                mesh_device=mesh_device,
            )

        self.rope = Gemma4RotaryEmbedding(
            max_position_embeddings=max_position_embeddings,
            sliding_head_dim=head_dim,
            sliding_rope_theta=sliding_rope_theta,
            full_head_dim=global_head_dim,
            full_rope_theta=full_rope_theta,
            full_partial_rotary_factor=full_partial_rotary_factor,
            mesh_device=mesh_device,
        )

    def forward(
        self,
        decoder_input_ids: ttnn.Tensor,
        decoder_position_ids: torch.Tensor,
        encoder_kv_cache: list[tuple[ttnn.Tensor, ttnn.Tensor]],
        decoder_attention_masks: dict[str, ttnn.Tensor | None],
        self_conditioning_signal: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """
        Args:
            decoder_input_ids:        int [B, canvas_length] on device.
            decoder_position_ids:     torch [B, canvas_length] long tensor — absolute positions
                                       (continue after the encoder's cache length).
            encoder_kv_cache:         per-layer list of (K_enc, V_enc) tensors from the encoder.
            decoder_attention_masks:  ``{"sliding_attention": mask|None, "full_attention": mask|None}``
                                       — bf16 additive masks of shape ``[B, 1, canvas, src+canvas]``.
            self_conditioning_signal: optional [B, canvas, hidden_size] from the previous denoising
                                       step's soft-embeddings. ``None`` on the first step.

        Returns:
            decoder hidden states [B, canvas_length, hidden_size] (replicated).
        """
        assert (
            len(encoder_kv_cache) == self.num_hidden_layers
        ), f"encoder_kv_cache has {len(encoder_kv_cache)} entries, expected {self.num_hidden_layers}"

        inputs_embeds = self.embed_tokens(decoder_input_ids)
        # First denoising step has no prior logits → use a zero soft-embedding signal.
        # Build host-side and upload (rather than ttnn.zeros_like, which doesn't reliably
        # produce a tile-laid-out replicated tensor on all builds).
        owned_signal = False
        if self_conditioning_signal is None:
            B, S = decoder_input_ids.shape[0], decoder_input_ids.shape[1]
            zero_signal = torch.zeros(B, S, self.hidden_size, dtype=torch.bfloat16)
            self_conditioning_signal = ttnn.from_torch(
                zero_signal, device=inputs_embeds.device(), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
            )
            owned_signal = True
        h = self.self_conditioning(inputs_embeds, self_conditioning_signal)
        ttnn.deallocate(inputs_embeds)
        if owned_signal:
            ttnn.deallocate(self_conditioning_signal)

        # Lift to 4D ``1BND`` so the layer stack sees consistent rank. embed_tokens +
        # self_conditioning produce 3D [B, S, H]; the attention path emits 4D via unsqueeze
        # after concat_heads. Same fix pattern as text_encoder._forward_from_embeds.
        if len(h.shape) == 3:
            h = ttnn.unsqueeze(h, 0)

        # Decoder positions continue after the encoder cache, so use them directly.
        cos_sin = {
            layer_type: self.rope.get_cos_sin(layer_type, decoder_position_ids) for layer_type in set(self.layer_types)
        }

        for i in range(self.num_hidden_layers):
            layer_type = self.layer_types[i]
            cos, sin = cos_sin[layer_type]
            h, _, _ = self.layers[i](
                h,
                cos,
                sin,
                attention_mask=decoder_attention_masks.get(layer_type),
                encoder_kv=encoder_kv_cache[i],
            )

        return self.norm(h)
