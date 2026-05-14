# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PaliGemma backbone wrapper - TTNN Implementation

This module combines vision, language, and action expert components:
    - SigLIP Vision Tower: Processes images to embeddings
    - Gemma 2B Language Model: VLM backbone for prefix (images + language)
    - Gemma 300M Action Expert: Processes suffix (state + actions)

The dual-expert architecture shares attention layers:
    - VLM and Expert compute separate Q, K, V
    - K, V are concatenated for shared attention
    - Outputs are split and processed through separate MLPs

Optimizations:
    1. Fused QKV weights (single linear instead of 3)
    2. Native TTNN RoPE (ttnn.experimental.rotary_embedding)
    3. Pre-added RMSNorm weights (Gemma-style +1 offset)
"""

from typing import Dict, List, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0_5.common.configs import PaliGemmaConfig
from .ttnn_common import tensor_1d_to_2d_ttnn
from .ttnn_gemma import (
    GemmaBlockTTNN,
    rms_norm_ttnn,
    precompute_freqs_cis_meta_format,
)
from .ttnn_siglip import (
    SigLIPVisionTowerTTNN,
    MultiModalProjectorTTNN,
)


class PaliGemmaBackboneTTNN:
    """
    PaliGemma backbone using TTNN operations.
    """

    def __init__(
        self,
        config: PaliGemmaConfig,
        weights: Dict[str, Dict[str, torch.Tensor]],
        device: ttnn.Device,
    ):
        """
        Initialize PaliGemma backbone with TTNN.

        Args:
            config: PaliGemma configuration
            weights: Categorized PyTorch weights
            device: TTNN device
        """
        self.config = config
        self.device = device

        # Convert embedding to TTNN (use lm_head if embed_tokens not available - tied embeddings)
        embed_weight = weights["vlm_language"].get("model.embed_tokens.weight")
        if embed_weight is None:
            embed_weight = weights["vlm_language"].get("lm_head.weight")
        if embed_weight is not None:
            self.vlm_embed_tokens = ttnn.from_torch(
                embed_weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,  # Embeddings use row major
                device=device,
            )
        else:
            self.vlm_embed_tokens = None

        # Convert norms - OPTIMIZATION: Pre-add Gemma-style +1 offset
        # Note: +1.0 is done on host (torch), unsqueeze done on device via tensor_1d_to_2d_ttnn
        self.vlm_norm = tensor_1d_to_2d_ttnn(
            weights["vlm_language"]["model.norm.weight"] + 1.0, device, dtype=ttnn.bfloat16
        )
        self.expert_norm = tensor_1d_to_2d_ttnn(
            weights["action_expert"]["model.norm.weight"] + 1.0, device, dtype=ttnn.bfloat16
        )

        # Initialize vision tower
        self.vision_tower = SigLIPVisionTowerTTNN(
            config.siglip_config,
            weights["vlm_vision"],
            device,
        )

        # Initialize projector
        self.mm_projector = MultiModalProjectorTTNN(weights["vlm_projector"], device)

        # Store torch weights for blocks (converted on demand)
        self.torch_weights = weights

        # Precompute RoPE using pure TTNN for native ttnn.experimental.rotary_embedding
        # This format is required by ttnn.experimental.rotary_embedding (split-half pattern)
        self.cos_meta, self.sin_meta = precompute_freqs_cis_meta_format(
            config.vlm_config.head_dim,
            config.max_seq_len,
            device,
        )

        # Expert RoPE in Meta format (pure TTNN, no torch)
        self.expert_cos_meta, self.expert_sin_meta = precompute_freqs_cis_meta_format(
            config.expert_config.head_dim,
            config.max_seq_len,
            device,
        )

        # Initialize VLM transformer blocks (18 layers for Gemma 2B)
        # Pass meta cos/sin for native TTNN RoPE (split-half pattern)
        self.vlm_blocks = []
        for i in range(config.vlm_config.depth):
            block_weights = self._get_vlm_block_weights_ttnn(weights["vlm_language"], i)
            self.vlm_blocks.append(
                GemmaBlockTTNN(
                    config.vlm_config,
                    block_weights,
                    i,
                    device,
                    self.cos_meta,
                    self.sin_meta,
                )
            )

        # Initialize Expert transformer blocks (18 layers for Gemma 300M)
        self.expert_blocks = []
        for i in range(config.expert_config.depth):
            block_weights = self._get_expert_block_weights_ttnn(weights["action_expert"], i)
            self.expert_blocks.append(
                GemmaBlockTTNN(
                    config.expert_config,
                    block_weights,
                    i,
                    device,
                    self.expert_cos_meta,
                    self.expert_sin_meta,
                )
            )

    def _get_vlm_block_weights_ttnn(
        self,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
    ) -> Dict[str, ttnn.Tensor]:
        """Extract VLM block weights and convert to TTNN with fused QKV optimization."""
        prefix = f"model.layers.{layer_idx}."
        block_weights = {}

        # OPTIMIZATION: Create fused QKV weight for single linear call
        q_key = f"{prefix}self_attn.q_proj.weight"
        k_key = f"{prefix}self_attn.k_proj.weight"
        v_key = f"{prefix}self_attn.v_proj.weight"

        if q_key in weights and k_key in weights and v_key in weights:
            # Get Q, K, V weights, transpose for TTNN linear, and convert to TTNN.
            # OPTIMIZATION: bf16 -> bf8_b for attention weights. Halves DRAM bandwidth
            # on the per-step QKV linear (180 calls for expert, 18 for VLM prefill).
            wq_ttnn = ttnn.from_torch(
                weights[q_key].T.contiguous(),  # [hidden, num_heads * head_dim]
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            wk_ttnn = ttnn.from_torch(
                weights[k_key].T.contiguous(),  # [hidden, num_kv_heads * head_dim]
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            wv_ttnn = ttnn.from_torch(
                weights[v_key].T.contiguous(),  # [hidden, num_kv_heads * head_dim]
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )

            # Concatenate using TTNN: [hidden, Q_dim + K_dim + V_dim]
            block_weights["self_attn.wqkv"] = ttnn.concat(
                [wq_ttnn, wk_ttnn, wv_ttnn],
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(wq_ttnn)
            ttnn.deallocate(wk_ttnn)
            ttnn.deallocate(wv_ttnn)

        for key, value in weights.items():
            if key.startswith(prefix):
                new_key = key[len(prefix) :]

                # Skip individual Q, K, V weights (now fused)
                if new_key in ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"]:
                    continue

                # Transpose weight matrices for TTNN linear
                if "weight" in new_key and "layernorm" not in new_key and "norm" not in new_key:
                    value = value.T
                    layout = ttnn.TILE_LAYOUT
                elif "layernorm" in new_key or "norm" in new_key:
                    # OPTIMIZATION: Pre-add Gemma-style +1 offset to norm weights
                    value = value + 1.0
                    layout = ttnn.TILE_LAYOUT
                else:
                    layout = ttnn.TILE_LAYOUT

                # Handle 1D tensors (biases, norms) using tensor_1d_to_2d_ttnn (no torch.unsqueeze)
                if len(value.shape) == 1:
                    block_weights[new_key] = tensor_1d_to_2d_ttnn(value, self.device, dtype=ttnn.bfloat16)
                else:
                    # OPTIMIZATION: o_proj weight bf16 -> bf8_b (halves DRAM bandwidth).
                    # Norm weights stay bf16 (small and precision-sensitive).
                    is_norm = "layernorm" in new_key or "norm" in new_key
                    weight_dtype = ttnn.bfloat16 if is_norm else ttnn.bfloat8_b
                    block_weights[new_key] = ttnn.from_torch(
                        value,
                        dtype=weight_dtype,
                        layout=layout,
                        device=self.device,
                    )
        return block_weights

    def _get_expert_block_weights_ttnn(
        self,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
    ) -> Dict[str, ttnn.Tensor]:
        """Extract expert block weights and convert to TTNN with fused QKV optimization."""
        prefix = f"model.layers.{layer_idx}."
        block_weights = {}

        # OPTIMIZATION: Create fused QKV weight for single linear call
        q_key = f"{prefix}self_attn.q_proj.weight"
        k_key = f"{prefix}self_attn.k_proj.weight"
        v_key = f"{prefix}self_attn.v_proj.weight"

        if q_key in weights and k_key in weights and v_key in weights:
            # Get Q, K, V weights, transpose for TTNN linear, and convert to TTNN
            wq_ttnn = ttnn.from_torch(
                weights[q_key].T.contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            wk_ttnn = ttnn.from_torch(
                weights[k_key].T.contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            wv_ttnn = ttnn.from_torch(
                weights[v_key].T.contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )

            # Concatenate using TTNN: [hidden, Q_dim + K_dim + V_dim]
            block_weights["self_attn.wqkv"] = ttnn.concat(
                [wq_ttnn, wk_ttnn, wv_ttnn],
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(wq_ttnn)
            ttnn.deallocate(wk_ttnn)
            ttnn.deallocate(wv_ttnn)

        for key, value in weights.items():
            if key.startswith(prefix):
                new_key = key[len(prefix) :]

                # Skip individual Q, K, V weights (now fused)
                if new_key in ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"]:
                    continue

                # Transpose weight matrices for TTNN linear
                if "weight" in new_key and "layernorm" not in new_key and "norm" not in new_key:
                    value = value.T
                    layout = ttnn.TILE_LAYOUT
                elif "layernorm" in new_key or "norm" in new_key:
                    # OPTIMIZATION: Pre-add Gemma-style +1 offset to norm weights
                    value = value + 1.0
                    layout = ttnn.TILE_LAYOUT
                else:
                    layout = ttnn.TILE_LAYOUT

                # Handle 1D tensors (biases, norms) using tensor_1d_to_2d_ttnn (no torch.unsqueeze)
                if len(value.shape) == 1:
                    block_weights[new_key] = tensor_1d_to_2d_ttnn(value, self.device, dtype=ttnn.bfloat16)
                else:
                    # OPTIMIZATION: o_proj weight bf16 -> bf8_b (halves DRAM bandwidth).
                    # Norm weights stay bf16 (small and precision-sensitive).
                    is_norm = "layernorm" in new_key or "norm" in new_key
                    weight_dtype = ttnn.bfloat16 if is_norm else ttnn.bfloat8_b
                    block_weights[new_key] = ttnn.from_torch(
                        value,
                        dtype=weight_dtype,
                        layout=layout,
                        device=self.device,
                    )
        return block_weights

    def embed_image(self, pixel_values: torch.Tensor) -> ttnn.Tensor:
        """
        Embed images through vision tower and projector (TTNN).

        Args:
            pixel_values: PyTorch tensor (batch_size, channels, height, width)

        Returns:
            TTNN tensor (batch_size, num_patches, vlm_width)
        """
        vision_features = self.vision_tower.forward(pixel_values)
        return self.mm_projector.forward(vision_features)

    def embed_language_tokens(self, token_ids: ttnn.Tensor) -> ttnn.Tensor:
        """
        Embed language tokens using TTNN.

        Args:
            token_ids: TTNN tensor of token IDs

        Returns:
            TTNN tensor of embeddings
        """
        return ttnn.embedding(token_ids, self.vlm_embed_tokens)

    def forward_vlm(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
        past_key_values: Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[ttnn.Tensor, Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]]]:
        """
        Forward pass through VLM backbone using TTNN.

        Args:
            hidden_states: Prefix embeddings (TTNN tensor)
            attention_mask: Attention mask (TTNN tensor)
            position_ids: Position indices (TTNN tensor)
            past_key_values: Cached KV from previous forward
            use_cache: Whether to return updated cache

        Returns:
            Tuple of (output, optional_new_cache)
        """
        new_cache = [] if use_cache else None

        for i, block in enumerate(self.vlm_blocks):
            past_kv = past_key_values[i] if past_key_values else None
            hidden_states, new_kv = block.forward(
                hidden_states,
                None,  # cos - unused, native TTNN RoPE uses cos_meta stored in block
                None,  # sin - unused, native TTNN RoPE uses sin_meta stored in block
                attention_mask,
                position_ids,
                past_kv,
                use_cache,
            )
            if use_cache:
                new_cache.append(new_kv)

        # Final norm using TTNN
        hidden_states = rms_norm_ttnn(
            hidden_states,
            self.vlm_norm,
            self.config.vlm_config.rms_norm_eps,
        )

        return hidden_states, new_cache

    def forward_expert(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
        past_key_values: Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[ttnn.Tensor, Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]]]:
        """
        Forward pass through action expert using TTNN.

        Args:
            hidden_states: Suffix embeddings (TTNN tensor)
            attention_mask: Attention mask (TTNN tensor)
            position_ids: Position indices (TTNN tensor)
            past_key_values: Cached KV from VLM prefix (for cross-attention)
            use_cache: Whether to return updated cache

        Returns:
            Tuple of (output, optional_new_cache)
        """
        new_cache = [] if use_cache else None

        for i, block in enumerate(self.expert_blocks):
            past_kv = past_key_values[i] if past_key_values else None
            hidden_states, new_kv = block.forward(
                hidden_states,
                None,  # cos - unused, native TTNN RoPE uses cos_meta stored in block
                None,  # sin - unused, native TTNN RoPE uses sin_meta stored in block
                attention_mask,
                position_ids,
                past_kv,
                use_cache,
            )
            if use_cache:
                new_cache.append(new_kv)

        # Final norm using TTNN
        hidden_states = rms_norm_ttnn(
            hidden_states,
            self.expert_norm,
            self.config.expert_config.rms_norm_eps,
        )

        return hidden_states, new_cache

    def forward_shared_attention(
        self,
        prefix_embs: ttnn.Tensor,
        suffix_embs: ttnn.Tensor,
        prefix_mask: Optional[ttnn.Tensor] = None,
        suffix_mask: Optional[ttnn.Tensor] = None,
        prefix_position_ids: Optional[ttnn.Tensor] = None,
        suffix_position_ids: Optional[ttnn.Tensor] = None,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Forward pass with shared attention between VLM and Expert (TTNN).

        Args:
            prefix_embs: VLM prefix embeddings (TTNN tensor)
            suffix_embs: Expert suffix embeddings (TTNN tensor)
            prefix_mask: Prefix attention mask
            suffix_mask: Suffix attention mask
            prefix_position_ids: Prefix positions
            suffix_position_ids: Suffix positions

        Returns:
            Tuple of (vlm_output, expert_output)
        """
        # Process prefix through VLM
        vlm_output, vlm_cache = self.forward_vlm(
            prefix_embs,
            prefix_mask,
            prefix_position_ids,
            use_cache=True,
        )

        # Process suffix through expert
        expert_output, _ = self.forward_expert(
            suffix_embs,
            suffix_mask,
            suffix_position_ids,
            past_key_values=None,
            use_cache=False,
        )

        return vlm_output, expert_output


# Default export
PaliGemmaBackbone = PaliGemmaBackboneTTNN

# ============================================================================
# pi0.5-specific additions (subclasses, overrides)
# ============================================================================

from typing import Dict, List, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0_5.common.configs import PaliGemmaConfig
from models.experimental.pi0_5.tt.ttnn_common import tensor_1d_to_2d_ttnn

from models.experimental.pi0_5.tt.ttnn_gemma import (
    AdaRMSGemmaBlockTTNN,
    ada_rms_norm_no_gate_ttnn,
    ada_rms_norm_no_gate_precomputed_ttnn,
)


def _convert_linear_to_ttnn(w: torch.Tensor, device, dtype=ttnn.bfloat16) -> "ttnn.Tensor":
    return ttnn.from_torch(w.T.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


class Pi0_5PaliGemmaBackboneTTNN(PaliGemmaBackboneTTNN):
    """PaliGemma backbone with adaRMS action expert (TTNN)."""

    def __init__(
        self,
        config: PaliGemmaConfig,
        weights: Dict[str, Dict[str, torch.Tensor]],
        device: "ttnn.Device",
    ):
        # Parent reads plain `model.norm.weight` + plain `input_layernorm.weight`
        # tensors for the expert side. PI0.5 expert has neither — inject zero
        # placeholders so super().__init__ runs, then overwrite expert artifacts.
        ae = weights["action_expert"]
        expert_w = config.expert_config.width
        placeholders = {}
        for i in range(config.expert_config.depth):
            for name in ("input_layernorm.weight", "post_attention_layernorm.weight"):
                key = f"model.layers.{i}.{name}"
                if key not in ae:
                    placeholders[key] = torch.zeros(expert_w)
        if "model.norm.weight" not in ae:
            placeholders["model.norm.weight"] = torch.zeros(expert_w)
        ae.update(placeholders)

        super().__init__(config, weights, device)

        # Rebuild expert blocks with adaRMS, injecting modulation tensors.
        # adaRMS now fuses the (1+scale)/shift modulation into ttnn.rms_norm via
        # its weight/bias args — no separate "ones" identity weight needed.
        self.expert_blocks = []
        for i in range(config.expert_config.depth):
            block_weights = self._get_expert_block_weights_ttnn(weights["action_expert"], i)
            self._inject_adarms_weights(block_weights, weights["action_expert"], i)
            self.expert_blocks.append(
                AdaRMSGemmaBlockTTNN(
                    config.expert_config,
                    block_weights,
                    i,
                    device,
                    self.expert_cos_meta,
                    self.expert_sin_meta,
                )
            )

        # Final expert norm: adaRMS.
        self.expert_final_norm_mod_weight = _convert_linear_to_ttnn(
            weights["action_expert"]["model.norm.dense.weight"], device
        )
        if "model.norm.dense.bias" in weights["action_expert"]:
            self.expert_final_norm_mod_bias = tensor_1d_to_2d_ttnn(
                weights["action_expert"]["model.norm.dense.bias"], device, dtype=ttnn.bfloat16
            )
        else:
            self.expert_final_norm_mod_bias = None

        device_grid = device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=device_grid.y, x=device_grid.x)

    def _inject_adarms_weights(
        self,
        block_weights: Dict[str, "ttnn.Tensor"],
        all_weights: Dict[str, torch.Tensor],
        layer_idx: int,
    ) -> None:
        """
        Fuse the pre-attention and pre-FFW modulation Dense projections into
        a single (in=W, out=6*W) linear per block (mirrors tt-dit's `norm1_linear`
        pattern). At forward time one matmul produces all 6 modulation tensors
        (scale_a, shift_a, gate_a, scale_f, shift_f, gate_f) for the block.

        Concatenation order on the output dim:
          [pre_attn.weight (3*W rows), pre_ffw.weight (3*W rows)]
        so slicing `[:W]` gives scale_a, `[W:2W]` shift_a, `[2W:3W]` gate_a,
        `[3W:4W]` scale_f, `[4W:5W]` shift_f, `[5W:6W]` gate_f.
        """
        prefix = f"model.layers.{layer_idx}."

        names = ("input_layernorm.dense", "post_attention_layernorm.dense")
        w_keys = [f"{prefix}{n}.weight" for n in names]
        b_keys = [f"{prefix}{n}.bias" for n in names]

        for wk in w_keys:
            if wk not in all_weights:
                raise KeyError(f"PI0.5 expects adaRMS weight '{wk}' in the action_expert checkpoint.")

        # Concat along output dim. Each input is (3*W, W); result is (6*W, W).
        fused_w_torch = torch.cat([all_weights[wk] for wk in w_keys], dim=0).contiguous()
        block_weights["adarms_mod.weight"] = _convert_linear_to_ttnn(fused_w_torch, self.device)

        biases = [all_weights[bk] for bk in b_keys if bk in all_weights]
        if biases:
            assert len(biases) == 2, "expected biases for both modulation Denses or neither"
            fused_b_torch = torch.cat(biases, dim=0).contiguous()
            block_weights["adarms_mod.bias"] = tensor_1d_to_2d_ttnn(fused_b_torch, self.device, dtype=ttnn.bfloat16)

    def forward_expert(
        self,
        hidden_states: "ttnn.Tensor",
        adarms_cond: "ttnn.Tensor",
        attention_mask: Optional["ttnn.Tensor"] = None,
        position_ids: Optional["ttnn.Tensor"] = None,
        past_key_values: Optional[List[Tuple["ttnn.Tensor", "ttnn.Tensor"]]] = None,
        use_cache: bool = False,
        precomputed_block_mods: Optional[List[Tuple["ttnn.Tensor", ...]]] = None,
        precomputed_final_mod: Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]] = None,
    ) -> Tuple["ttnn.Tensor", Optional[List[Tuple["ttnn.Tensor", "ttnn.Tensor"]]]]:
        new_cache = [] if use_cache else None

        for i, block in enumerate(self.expert_blocks):
            past_kv = past_key_values[i] if past_key_values else None
            block_mod = precomputed_block_mods[i] if precomputed_block_mods is not None else None
            hidden_states, new_kv = block.forward(
                hidden_states,
                None,
                None,
                adarms_cond,
                attention_mask,
                position_ids,
                past_kv,
                use_cache,
                precomputed_mod=block_mod,
            )
            if use_cache:
                new_cache.append(new_kv)

        if precomputed_final_mod is not None:
            sf1, tf = precomputed_final_mod
            hidden_states = ada_rms_norm_no_gate_precomputed_ttnn(
                hidden_states, sf1, tf, self.config.expert_config.rms_norm_eps
            )
        else:
            hidden_states = ada_rms_norm_no_gate_ttnn(
                hidden_states,
                adarms_cond,
                self.expert_final_norm_mod_weight,
                self.expert_final_norm_mod_bias,
                self.config.expert_config.rms_norm_eps,
                self.core_grid,
            )
        return hidden_states, new_cache
