# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN model refactored with modular structure matching PyTorch."""

import json
from pathlib import Path

import torch
from consteval import run_const_evals

import ttnn
from models.common.lightweightmodule import LightweightModule


class CLIPVisionEncoderAndResamplerTTNN(LightweightModule):
    """
    CLIP Vision Encoder + IP-Adapter Plus Resampler in TTNN.

    Architecture:
        - Vision Embeddings (patch embedding + class token + position embedding)
        - Pre-LayerNorm
        - 31 Encoder Layers (layer_norm1 -> self_attn -> add -> layer_norm2 -> mlp -> add)
        - Resampler (proj_in -> 4 attention blocks -> proj_out -> norm_out)
    """

    def __init__(self, device, torch_weights):
        self.device = device
        self.weights = load_weights_from_pytorch(torch_weights, device)
        # Apply const-eval functions to weights
        self.weights = run_const_evals(self.weights, device)

    def forward(self, pixel_values):
        """
        Forward pass through CLIP Vision Encoder + IP-Adapter Resampler.

        Args:
            pixel_values: Input image tensor [1, 3, 224, 224] on host

        Returns:
            List containing output tensor [1, 16, 2048]
        """
        # Move input to device
        assert pixel_values.device() is None, "pixel_values must be on host"
        pixel_values = ttnn.to_device(
            pixel_values,
            self.device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        pixel_values = ttnn.to_layout(
            pixel_values,
            ttnn.Layout.TILE,
            None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Vision Embeddings: patch embedding + class token + position embedding
        x = self._vision_embeddings(pixel_values)

        # Pre-LayerNorm
        x = self._pre_layernorm(x)

        # Encoder Layers (31 layers, indices 0-30)
        for layer_idx in range(31):
            x = self._encoder_layer(x, layer_idx)

        # Use output from final encoder layer (layer 30)
        # Note: Original TTNN code uses final layer output, not penultimate
        patches = x

        # Resampler: proj_in -> 4 attention blocks -> proj_out -> norm_out
        encoder_hidden_states = self._resampler_proj_in(patches)

        # All resampler blocks operate on the same encoder_hidden_states
        # Block 0 uses precomputed latents
        latents = self._resampler_block(None, encoder_hidden_states, 0)

        # Blocks 1-3 use output from previous block
        for block_idx in range(1, 4):
            latents = self._resampler_block(latents, encoder_hidden_states, block_idx)

        # Output projection and final norm
        output = self._resampler_proj_out(latents)
        output = self._resampler_norm_out(output)

        return [output]

    # ============ Vision Embeddings ============

    def _vision_embeddings(self, pixel_values):
        """
        CLIPVisionEmbeddings: patch_embedding (Conv2d) + class_embedding + position_embedding.

        Input: [1, 3, 224, 224] -> Output: [1, 257, 1280]
        """

        # Permute NCHW -> NHWC for conv2d
        x = ttnn.permute(
            pixel_values,
            [0, 2, 3, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )

        # Reshape for conv2d: [1, 224, 224, 3] -> [1, 1, 50176, 3]
        x = ttnn.reshape(x, [1, 1, 50176, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Patch embedding Conv2d: kernel=14x14, stride=14
        x = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weights["image_encoder.vision_model.embeddings.patch_embedding.weight"],
            device=self.device,
            in_channels=3,
            out_channels=1280,
            batch_size=1,
            input_height=224,
            input_width=224,
            kernel_size=[14, 14],
            stride=[14, 14],
            padding=[0, 0, 0, 0],
            dilation=[1, 1],
            groups=1,
            bias_tensor=None,
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=0,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Reshape: [1, 1, 256, 1280] -> [1, 16, 16, 1280]
        x = ttnn.reshape(x, [1, 16, 16, 1280], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Permute to NCHW: [1, 16, 16, 1280] -> [1, 1280, 16, 16]
        x = ttnn.permute(x, [0, 3, 1, 2], memory_config=ttnn.DRAM_MEMORY_CONFIG, pad_value=0.0)

        # Reshape: [1, 1280, 16, 16] -> [1, 1280, 256]
        x = ttnn.reshape(x, [1, 1280, 256], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Concat with class embedding on dim 2: [1, 1280, 256] + [1, 1280, 1] -> [1, 1280, 257]
        class_embedding = self.weights["image_encoder.vision_model.embeddings.class_embedding"]
        x = ttnn.concat([class_embedding, x], 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Add position embedding (shape [1, 1280, 257])
        position_embedding = self.weights["__POSITION_EMBEDDING__"]
        x = ttnn.add(
            x,
            position_embedding,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Permute: [1, 1280, 257] -> [1, 257, 1280]
        x = ttnn.permute(x, [0, 2, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG, pad_value=0.0)

        return x

    def _pre_layernorm(self, x):
        """vision_model.pre_layrnorm: LayerNorm before encoder."""
        return ttnn.layer_norm(
            x,
            epsilon=9.9999997473787516e-06,
            weight=self.weights["image_encoder.vision_model.pre_layrnorm.weight"],
            bias=self.weights["image_encoder.vision_model.pre_layrnorm.bias"],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

    # ============ Encoder Layer ============

    def _encoder_layer(self, hidden_states, layer_idx: int):
        """
        CLIPEncoderLayer: Pre-norm transformer block.

        Structure:
            residual = hidden_states
            hidden_states = layer_norm1(hidden_states)
            hidden_states = self_attn(hidden_states)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = layer_norm2(hidden_states)
            hidden_states = mlp(hidden_states)
            hidden_states = residual + hidden_states

        Args:
            hidden_states: Input tensor [1, 257, 1280]
            layer_idx: Layer index (0-30)

        Returns:
            Output tensor [1, 257, 1280]
        """

        # First residual block: LayerNorm1 -> Self-Attention -> Add
        residual = hidden_states
        hidden_states = self._layer_norm(hidden_states, layer_idx, "layer_norm1")
        hidden_states = self._clip_attention(hidden_states, layer_idx)
        hidden_states = ttnn.add(
            residual,
            hidden_states,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Second residual block: LayerNorm2 -> MLP -> Add
        residual = hidden_states
        hidden_states = self._layer_norm(hidden_states, layer_idx, "layer_norm2")
        hidden_states = self._clip_mlp(hidden_states, layer_idx)
        hidden_states = ttnn.add(
            residual,
            hidden_states,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return hidden_states

    def _layer_norm(self, x, layer_idx: int, which: str):
        """LayerNorm for encoder layer with dynamic weight lookup."""
        weight_key = f"image_encoder.vision_model.encoder.layers.{layer_idx}.{which}.weight"
        bias_key = f"image_encoder.vision_model.encoder.layers.{layer_idx}.{which}.bias"

        return ttnn.layer_norm(
            x,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[weight_key],
            bias=self.weights[bias_key],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

    def _clip_attention(self, hidden_states, layer_idx: int):
        """
        CLIPAttention: Multi-head self-attention.

        Uses fused QKV weights from consteval.
        - 16 attention heads
        - Head dimension: 80
        - Scale: 1/sqrt(80) = 0.11180340498685837

        Args:
            hidden_states: [1, 257, 1280]
            layer_idx: Layer index (0-30)

        Returns:
            Output tensor [1, 257, 1280]
        """

        # Reshape to 2D for matmul: [1, 257, 1280] -> [257, 1280]
        x = ttnn.reshape(hidden_states, [257, 1280], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Fused QKV projection: [257, 1280] @ [3840, 1280]^T -> [257, 3840]
        layer_prefix = f"image_encoder.vision_model.encoder.layers.{layer_idx}.self_attn"
        qkv_weight = self.weights[f"{layer_prefix}.qkv_weight"]
        qkv_bias = self.weights[f"{layer_prefix}.qkv_bias"]

        x = ttnn.matmul(
            x,
            qkv_weight,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        x = ttnn.add(
            x,
            qkv_bias,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Split into Q, K, V (each 1280)
        # Order in fused weight: Q, K, V (based on consteval concat order)
        q = ttnn.slice(
            x,
            [0, 0, 0],
            [1, 257, 1280],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        k = ttnn.slice(
            x,
            [0, 0, 1280],
            [1, 257, 2560],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        v = ttnn.slice(
            x,
            [0, 0, 2560],
            [1, 257, 3840],
            [1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Reshape for multi-head attention: [257, 1280] -> [1, 257, 16, 80]
        q = ttnn.reshape(q, [1, 257, 16, 80], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.reshape(k, [1, 257, 16, 80], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.reshape(v, [1, 257, 16, 80], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Permute to [batch, heads, seq, head_dim]: [1, 257, 16, 80] -> [1, 16, 257, 80]
        q = ttnn.permute(q, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG, pad_value=0.0)
        k = ttnn.permute(k, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG, pad_value=0.0)
        v = ttnn.permute(v, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG, pad_value=0.0)

        # Pad head_dim from 80 to 96 for efficient SDPA
        q = ttnn.pad(
            q,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        k = ttnn.pad(
            k,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        v = ttnn.pad(
            v,
            [[0, 0], [0, 0], [0, 0], [0, 16]],
            0.0,
            use_multicore=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Scaled Dot-Product Attention
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=False,
            scale=0.11180340498685837,  # 1/sqrt(80)
            sliding_window_size=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Remove padding: [1, 16, 257, 96] -> [1, 16, 257, 80]
        attn_output = ttnn.slice(
            attn_output,
            [0, 0, 0, 0],
            [1, 16, 257, 80],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Permute back: [1, 16, 257, 80] -> [1, 257, 16, 80]
        attn_output = ttnn.permute(
            attn_output,
            [0, 2, 1, 3],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            pad_value=0.0,
        )

        # Reshape: [1, 257, 16, 80] -> [257, 1280]
        attn_output = ttnn.reshape(attn_output, [257, 1280], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Output projection
        out_proj_weight = self.weights[
            f"image_encoder.vision_model.encoder.layers.{layer_idx}.self_attn.out_proj.weight"
        ]
        out_proj_bias = self.weights[f"image_encoder.vision_model.encoder.layers.{layer_idx}.self_attn.out_proj.bias"]

        attn_output = ttnn.matmul(
            attn_output,
            out_proj_weight,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        attn_output = ttnn.add(
            attn_output,
            out_proj_bias,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return attn_output

    def _clip_mlp(self, hidden_states, layer_idx: int):
        """
        CLIPMLP: Two-layer MLP with GELU activation.

        Structure: fc1 (1280 -> 5120) -> GELU -> fc2 (5120 -> 1280)

        Args:
            hidden_states: Input tensor (either [1, 257, 1280] or [257, 1280])
            layer_idx: Layer index (0-30)

        Returns:
            Output tensor [257, 1280]
        """

        # Reshape to 2D if needed
        x = ttnn.reshape(hidden_states, [257, 1280], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # FC1: [257, 1280] -> [257, 5120]
        fc1_weight = self.weights[f"image_encoder.vision_model.encoder.layers.{layer_idx}.mlp.fc1.weight"]
        fc1_bias = self.weights[f"image_encoder.vision_model.encoder.layers.{layer_idx}.mlp.fc1.bias"]

        x = ttnn.matmul(
            x,
            fc1_weight,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        x = ttnn.add(
            x,
            fc1_bias,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # GELU activation
        x = ttnn.gelu(x, fast_and_approximate_mode=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Ensure correct shape for fc2
        x = ttnn.reshape(x, [257, 5120], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # FC2: [257, 5120] -> [257, 1280]
        fc2_weight = self.weights[f"image_encoder.vision_model.encoder.layers.{layer_idx}.mlp.fc2.weight"]
        fc2_bias = self.weights[f"image_encoder.vision_model.encoder.layers.{layer_idx}.mlp.fc2.bias"]

        x = ttnn.matmul(
            x,
            fc2_weight,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        x = ttnn.add(
            x,
            fc2_bias,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return x

    # ============ Resampler ============

    def _resampler_proj_in(self, x):
        """Resampler input projection: Linear 1280 -> 1280."""

        # Reshape to 2D: [1, 257, 1280] -> [257, 1280]
        x = ttnn.reshape(x, [257, 1280], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        x = ttnn.matmul(
            x,
            self.weights["resampler.proj_in.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        x = ttnn.add(
            x,
            self.weights["resampler.proj_in.bias"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return x

    def _resampler_block(self, latents, encoder_hidden_states, block_idx: int):
        """
        IPAdapterPlusImageProjectionBlock: Cross-attention block.

        Structure:
            encoder_hidden_states = ln0(encoder_hidden_states)
            latents = ln1(latents)
            encoder_hidden_states = concat([encoder_hidden_states, latents], dim=-2)
            latents = attn(latents, encoder_hidden_states) + residual
            latents = ff(latents) + latents

        For block 0, latents come from precomputed consteval (learned latents with ln1 and to_q applied).
        For blocks 1-3, latents come from the previous block output.

        Args:
            latents: For block 0, this is the projected encoder hidden states.
                     For blocks 1-3, this is the previous block's output.
            encoder_hidden_states: The projected encoder output [257, 1280]
            block_idx: Block index (0-3)

        Returns:
            Updated latents tensor
        """
        if block_idx == 0:
            # Block 0 uses precomputed latents (ln1 + to_q already applied)
            return self._resampler_block_0(encoder_hidden_states)
        else:
            # Blocks 1-3 use the output from the previous block
            return self._resampler_block_n(latents, encoder_hidden_states, block_idx)

    def _resampler_block_0(self, encoder_hidden_states):
        """
        First resampler block - uses precomputed latents.

        The learned latents have already had ln1 and to_q applied in consteval.
        Q = resampler.layers.0.attn.precomputed_q (already reshaped and permuted)
        """

        # ln0 on encoder hidden states
        x_ln0 = ttnn.layer_norm(
            encoder_hidden_states,
            epsilon=9.9999997473787516e-06,
            weight=self.weights["resampler.layers.0.ln0.weight"],
            bias=self.weights["resampler.layers.0.ln0.bias"],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        # Reshape for concatenation
        x_ln0 = ttnn.reshape(x_ln0, [257, 1280], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Precomputed: latents with ln1 applied, ready for concat
        # resampler.layers.0.ln1_latents_reshaped is the ln1(latents) output [16, 1280]
        # Concat encoder hidden states with latents
        concat_input = ttnn.concat(
            [x_ln0, self.weights["resampler.layers.0.ln1_latents_reshaped"]],
            0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Compute K and V from concatenated input
        k = ttnn.matmul(
            concat_input,
            self.weights["resampler.layers.0.attn.to_k.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        v = ttnn.matmul(
            concat_input,
            self.weights["resampler.layers.0.attn.to_v.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )

        # Typecast K and V to FLOAT32 before reshape (matching original)
        k = ttnn.typecast(k, ttnn.DataType.FLOAT32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.typecast(v, ttnn.DataType.FLOAT32, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Reshape for multi-head attention: [273, 1280] -> [1, 273, 20, 64]
        # Resampler has 20 heads with head_dim=64
        k = ttnn.reshape(k, [1, 273, 20, 64], memory_config=ttnn.DRAM_MEMORY_CONFIG)  # 257 + 16 = 273
        v = ttnn.reshape(v, [1, 273, 20, 64], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # K permutation: [0, 2, 3, 1] then [0, 1, 3, 2] = transpose K for attention
        k = ttnn.permute(k, [0, 2, 3, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG, pad_value=0.0)
        k = ttnn.permute(k, [0, 1, 3, 2], memory_config=ttnn.DRAM_MEMORY_CONFIG, pad_value=0.0)

        # V permutation: [0, 2, 1, 3]
        v = ttnn.permute(v, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG, pad_value=0.0)

        # Typecast K and V back to BFLOAT16
        k = ttnn.typecast(k, ttnn.DataType.BFLOAT16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.typecast(v, ttnn.DataType.BFLOAT16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Q is precomputed: resampler.layers.0.attn.precomputed_q is to_q(ln1(latents)) with reshape and permute
        q = self.weights["resampler.layers.0.attn.precomputed_q"]

        # Scaled Dot-Product Attention
        # Scale = 1/sqrt(8) = 0.35355338454246521 (not 1/sqrt(64))
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=False,
            scale=0.35355338454246521,
            sliding_window_size=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Use concatenate_heads to merge attention heads
        attn_output = ttnn.transformer.concatenate_heads(attn_output, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Reshape: [1, 16, 1280] -> [16, 1280]
        attn_output = ttnn.reshape(attn_output, [16, 1280], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Output projection (no bias in resampler attention)
        attn_output = ttnn.matmul(
            attn_output,
            self.weights["resampler.layers.0.attn.to_out.0.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )

        # Reshape back to [1, 16, 1280] for division
        attn_output = ttnn.reshape(attn_output, [1, 16, 1280], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Divide by ones tensor (matches original - may be for numerical precision)
        attn_output = ttnn.divide(
            attn_output,
            self.weights["__ONES_1_16_1280__"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Add residual: use resampler.latents (the learned queries)
        attn_output = ttnn.add(
            attn_output,
            self.weights["resampler.latents"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Feedforward
        latents = self._resampler_feedforward(attn_output, 0)

        return latents

    def _resampler_block_n(self, latents, encoder_hidden_states, block_idx: int):
        """Resampler blocks 1-3."""

        # ln0 on encoder hidden states (same input for all blocks)
        x_ln0 = ttnn.layer_norm(
            encoder_hidden_states,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[f"resampler.layers.{block_idx}.ln0.weight"],
            bias=self.weights[f"resampler.layers.{block_idx}.ln0.bias"],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        # ln1 on latents
        x_ln1 = ttnn.layer_norm(
            latents,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[f"resampler.layers.{block_idx}.ln1.weight"],
            bias=self.weights[f"resampler.layers.{block_idx}.ln1.bias"],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        # Reshape for concatenation
        x_ln0 = ttnn.reshape(x_ln0, [257, 1280], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x_ln1_reshaped = ttnn.reshape(x_ln1, [16, 1280], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Concat encoder hidden states with latents
        concat_input = ttnn.concat([x_ln0, x_ln1_reshaped], 0, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Compute Q from ln1(latents) - reshaped to [16, 1280]
        q = ttnn.matmul(
            x_ln1_reshaped,
            self.weights[f"resampler.layers.{block_idx}.attn.to_q.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )

        # Compute K and V from concatenated input
        k = ttnn.matmul(
            concat_input,
            self.weights[f"resampler.layers.{block_idx}.attn.to_k.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        v = ttnn.matmul(
            concat_input,
            self.weights[f"resampler.layers.{block_idx}.attn.to_v.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )

        # Typecast Q, K, V to FLOAT32 before reshape
        q = ttnn.typecast(q, ttnn.DataType.FLOAT32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.typecast(k, ttnn.DataType.FLOAT32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.typecast(v, ttnn.DataType.FLOAT32, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Reshape for multi-head attention
        q = ttnn.reshape(q, [1, 16, 20, 64], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.reshape(k, [1, 273, 20, 64], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.reshape(v, [1, 273, 20, 64], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Q permutation: [0, 2, 1, 3]
        q = ttnn.permute(q, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG, pad_value=0.0)

        # V permutation: [0, 2, 1, 3]
        v = ttnn.permute(v, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG, pad_value=0.0)

        # K permutation: [0, 2, 3, 1] then [0, 1, 3, 2]
        k = ttnn.permute(k, [0, 2, 3, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG, pad_value=0.0)
        k = ttnn.permute(k, [0, 1, 3, 2], memory_config=ttnn.DRAM_MEMORY_CONFIG, pad_value=0.0)

        # Typecast Q, K, V back to BFLOAT16
        q = ttnn.typecast(q, ttnn.DataType.BFLOAT16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.typecast(k, ttnn.DataType.BFLOAT16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.typecast(v, ttnn.DataType.BFLOAT16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # SDPA with scale = 0.125 (approximately 1/8)
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=False,
            scale=0.1249999925494194,
            sliding_window_size=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Use concatenate_heads to merge attention heads
        attn_output = ttnn.transformer.concatenate_heads(attn_output, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Reshape: [1, 16, 1280] -> [16, 1280]
        attn_output = ttnn.reshape(attn_output, [16, 1280], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Output projection
        attn_output = ttnn.matmul(
            attn_output,
            self.weights[f"resampler.layers.{block_idx}.attn.to_out.0.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )

        # Reshape back to [1, 16, 1280] for division
        attn_output = ttnn.reshape(attn_output, [1, 16, 1280], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Divide by ones tensor (matches original)
        attn_output = ttnn.divide(
            attn_output,
            self.weights["__ONES_1_16_1280__"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Add residual (latents is already [1, 16, 1280])
        attn_output = ttnn.add(
            attn_output,
            latents,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Feedforward
        latents = self._resampler_feedforward(attn_output, block_idx)

        return latents

    def _resampler_feedforward(self, x, block_idx: int):
        """
        Resampler feedforward: LayerNorm -> Linear -> GELU -> Linear.

        Input shape: [1, 16, 1280]
        Output shape: [1, 16, 1280]

        Structure:
            residual = x  # [1, 16, 1280]
            x = ff_ln(x)  # [1, 16, 1280]
            x = reshape(x, [16, 1280])
            x = ff_proj(x, activation="gelu")  # [16, 5120]
            x = ff_out(x)  # [16, 1280]
            x = reshape(x, [1, 16, 1280])
            x = x + residual
        """
        residual = x  # [1, 16, 1280]

        # LayerNorm on [1, 16, 1280]
        x = ttnn.layer_norm(
            x,
            epsilon=9.9999997473787516e-06,
            weight=self.weights[f"resampler.layers.{block_idx}.ff.0.weight"],
            bias=self.weights[f"resampler.layers.{block_idx}.ff.0.bias"],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        # Reshape to [16, 1280] for matmul
        x = ttnn.reshape(x, [16, 1280], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # First linear with GELU activation: [16, 1280] -> [16, 5120]
        x = ttnn.matmul(
            x,
            self.weights[f"resampler.layers.{block_idx}.ff.1.net.0.proj.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation="gelu",
        )

        # Second linear: [16, 5120] -> [16, 1280]
        x = ttnn.matmul(
            x,
            self.weights[f"resampler.layers.{block_idx}.ff.1.net.2.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )

        # Reshape back to [1, 16, 1280]
        x = ttnn.reshape(x, [1, 16, 1280], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Add residual
        x = ttnn.add(
            x,
            residual,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return x

    def _resampler_proj_out(self, x):
        """Resampler output projection: Linear 1280 -> 2048."""

        x = ttnn.matmul(
            x,
            self.weights["resampler.proj_out.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=None,
            program_config=None,
            activation=None,
        )
        x = ttnn.add(
            x,
            self.weights["resampler.proj_out.bias"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return x

    def _resampler_norm_out(self, x):
        """Resampler final LayerNorm."""

        return ttnn.layer_norm(
            x,
            epsilon=9.9999997473787516e-06,
            weight=self.weights["resampler.norm_out.weight"],
            bias=self.weights["resampler.norm_out.bias"],
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )


# ============ Weight Loading ============


def _create_position_ids():
    """
    Create position IDs tensor for CLIP vision model.

    The position IDs are [0, 1, 2, ..., 256] for 257 positions
    (256 patches + 1 CLS token).
    """
    # CLIP ViT-H has 257 positions (16x16 patches + CLS token)
    num_positions = 257
    pos_ids = torch.arange(num_positions, dtype=torch.int32).unsqueeze(0)

    # Convert to TTNN
    ttnn_tensor = ttnn.from_torch(pos_ids)
    ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.ROW_MAJOR)
    ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.INT32)

    return ttnn_tensor


def load_weights_from_pytorch(state_dict, device):
    """
    Load weights from PyTorch model and convert to TTNN format.

    Args:
        state_dict: PyTorch state_dict
        device: TTNN device

    Returns:
        Dictionary mapping weight names to TTNN tensors
    """
    # Load config
    with open(Path(__file__).parent / "tensor_load_config.json") as f:
        config = json.load(f)

    # Convert each weight to TTNN format
    weights = {}
    converted_count = 0

    for weight_name, cfg in config.items():
        layout_str = cfg.get("layout", "TILE")
        dtype_str = cfg.get("dtype", "BFLOAT16")
        on_device = cfg.get("on_device", False)

        # Handle special entries
        if weight_name == "__POSITION_IDS__":
            # Generate position IDs tensor
            pos_ids = _create_position_ids()
            weights[weight_name] = pos_ids
            continue

        # Get PyTorch tensor
        if weight_name not in state_dict:
            raise ValueError(f"Weight '{weight_name}' not found in PyTorch model")

        torch_tensor = state_dict[weight_name]

        # Convert to TTNN
        ttnn_tensor = ttnn.from_torch(torch_tensor)

        # Apply layout
        layout = ttnn.Layout.TILE if layout_str == "TILE" else ttnn.Layout.ROW_MAJOR
        ttnn_tensor = ttnn.to_layout(ttnn_tensor, layout)

        # Apply dtype
        if dtype_str == "BFLOAT16":
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.BFLOAT16)
        elif dtype_str == "INT32":
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.INT32)

        # Move to device if needed
        if on_device and device is not None:
            ttnn_tensor = ttnn.to_device(ttnn_tensor, device, ttnn.DRAM_MEMORY_CONFIG)

        weights[weight_name] = ttnn_tensor
        converted_count += 1

    print(f"Converted {converted_count} weights")
    return weights
