# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full DiffusionGemma encoder text model.

Mirrors ``DiffusionGemmaEncoderTextModel`` from
``transformers.models.diffusion_gemma.modeling_diffusion_gemma``: scaled token
embedding, ``num_hidden_layers`` of ``DiffusionGemmaLayer`` (sliding/full
interleaved per ``config.layer_types``), and a final RMSNorm.

Per layer, this model returns the layer's K/V tensors (post-RoPE) so that the
decoder can later cross-attend to them. That's our "KV cache" — a plain Python
list of (K, V) tuples, indexed by layer.

Mask routing: each layer receives the appropriate mask for its type. The caller
must supply the masks (one for ``sliding_attention``, one for ``full_attention``).
Multimodal bidirectional-vision masking is the pipeline's responsibility.
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


class DiffusionGemmaEncoderTextModel(Module):
    """Embed → 30 layers (interleaved sliding/full) → final norm.

    Construction-time:
      - The encoder owns its layers' MoE substates (passed individually because
        ``DiffusionGemmaMoE`` requires its weights at __init__ time). Caller
        provides ``moe_state_dicts: list[dict]`` indexed by layer.
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
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.layer_types = list(layer_types)

        if moe_state_dicts is None:
            moe_state_dicts = [None] * num_hidden_layers
        assert len(moe_state_dicts) == num_hidden_layers

        self.embed_tokens = DiffusionGemmaScaledWordEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            padding_idx=pad_token_id,
            mesh_device=mesh_device,
        )

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

        self.norm = RMSNorm(
            embedding_dim=hidden_size,
            norm_eps=rms_norm_eps,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
        )

        # One RoPE module for all layers — dispatches per layer_type.
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
        input_ids: ttnn.Tensor,
        position_ids: torch.Tensor,
        attention_masks: dict[str, ttnn.Tensor | None],
    ) -> tuple[ttnn.Tensor, list[tuple[ttnn.Tensor, ttnn.Tensor]]]:
        """
        Args:
            input_ids:       int [B, S] on device.
            position_ids:    torch [B, S] long tensor (host-side, used to slice RoPE cache).
            attention_masks: ``{"sliding_attention": tt_mask|None, "full_attention": tt_mask|None}``
                              — bf16 additive masks already on device.

        Returns:
            (hidden_states, per_layer_kv): final hidden states and the list of (K, V)
            tensors per layer (in TP-sharded layout) for the decoder to read.
        """
        # Precompute (cos, sin) once per layer_type for this seq.
        h = self.embed_tokens(input_ids)
        return self._forward_from_embeds(h, position_ids, attention_masks)

    def _forward_from_embeds(
        self,
        inputs_embeds: ttnn.Tensor,
        position_ids: torch.Tensor,
        attention_masks: dict[str, ttnn.Tensor | None],
    ) -> tuple[ttnn.Tensor, list[tuple[ttnn.Tensor, ttnn.Tensor]]]:
        """Run the layer stack from pre-computed embeddings.

        Used by the multimodal encoder which builds merged text+vision embeddings before
        feeding them through the transformer layers.
        """
        cos_sin = {layer_type: self.rope.get_cos_sin(layer_type, position_ids) for layer_type in set(self.layer_types)}

        h = inputs_embeds
        # Layers + attention are written for the tt_dit 4D ``1BND`` convention. embed_tokens
        # and multimodal-merged embeddings arrive as 3D [B, S, H]. Lift to 4D once here so
        # the layer stack (attention returns 4D via unsqueeze after concat_heads) has
        # consistent-rank residual adds.
        if len(h.shape) == 3:
            h = ttnn.unsqueeze(h, 0)
        per_layer_kv: list[tuple[ttnn.Tensor, ttnn.Tensor]] = []
        for i in range(self.num_hidden_layers):
            layer_type = self.layer_types[i]
            cos, sin = cos_sin[layer_type]
            h, k, v = self.layers[i](
                h,
                cos,
                sin,
                attention_mask=attention_masks.get(layer_type),
                encoder_kv=None,
            )
            per_layer_kv.append((k, v))

        h = self.norm(h)
        return h, per_layer_kv
