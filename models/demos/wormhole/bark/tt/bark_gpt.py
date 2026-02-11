# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
GPT-style Transformer block used across all three Bark stages.

HuggingFace Bark layer structure (from modeling_bark.py):
    BarkBlock:
        layernorm_1 -> BarkSelfAttention (att_proj + out_proj) -> layernorm_2 -> BarkMLP (in_proj + out_proj)

State dict key patterns:
    {prefix}.layers.{i}.layernorm_1.weight
    {prefix}.layers.{i}.attn.att_proj.weight  (combined QKV, 3*hidden x hidden)
    {prefix}.layers.{i}.attn.out_proj.weight
    {prefix}.layers.{i}.mlp.in_proj.weight    (hidden x 4*hidden)
    {prefix}.layers.{i}.mlp.out_proj.weight   (4*hidden x hidden)
    {prefix}.input_embeds_layer.weight
    {prefix}.position_embeds_layer.weight
    {prefix}.layernorm_final.weight
    {prefix}.lm_head.weight
"""

from typing import Optional

import torch
import ttnn

from models.demos.wormhole.bark.tt.common import BarkConfig, load_tt_tensor


class TtBarkLayerNorm:
    """LayerNorm for Bark GPT blocks, executed on TT device."""

    def __init__(self, state_dict: dict, prefix: str, device: ttnn.Device, config: BarkConfig, has_bias: bool = False):
        self.device = device
        self.eps = config.layer_norm_eps

        self.weight = load_tt_tensor(
            state_dict[f"{prefix}.weight"],
            device=device,
            dtype=ttnn.bfloat16,
        )

        if has_bias and f"{prefix}.bias" in state_dict:
            self.bias = load_tt_tensor(
                state_dict[f"{prefix}.bias"],
                device=device,
                dtype=ttnn.bfloat16,
            )
        else:
            self.bias = None

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.layer_norm(x, epsilon=self.eps, weight=self.weight, bias=self.bias)


class TtBarkSelfAttention:
    """
    Multi-head self-attention for Bark GPT blocks.

    HF key names:
        attn.att_proj.weight  -> [3*hidden, hidden] combined QKV
        attn.out_proj.weight  -> [hidden, hidden]
    """

    def __init__(
        self,
        state_dict: dict,
        prefix: str,
        device: ttnn.Device,
        config: BarkConfig,
        is_causal: bool = True,
    ):
        self.device = device
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.is_causal = is_causal
        self.has_bias = config.bias

        # QKV projection (combined): shape [3*hidden, hidden]
        self.att_proj_weight = load_tt_tensor(
            state_dict[f"{prefix}.att_proj.weight"].transpose(-1, -2),
            device=device,
            dtype=ttnn.bfloat8_b,
        )
        if self.has_bias and f"{prefix}.att_proj.bias" in state_dict:
            self.att_proj_bias = load_tt_tensor(
                state_dict[f"{prefix}.att_proj.bias"],
                device=device,
                dtype=ttnn.bfloat16,
            )
        else:
            self.att_proj_bias = None

        # Output projection: shape [hidden, hidden]
        self.out_proj_weight = load_tt_tensor(
            state_dict[f"{prefix}.out_proj.weight"].transpose(-1, -2),
            device=device,
            dtype=ttnn.bfloat8_b,
        )
        if self.has_bias and f"{prefix}.out_proj.bias" in state_dict:
            self.out_proj_bias = load_tt_tensor(
                state_dict[f"{prefix}.out_proj.bias"],
                device=device,
                dtype=ttnn.bfloat16,
            )
        else:
            self.out_proj_bias = None

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

    def __call__(self, x: ttnn.Tensor, attention_mask: Optional[ttnn.Tensor] = None) -> ttnn.Tensor:
        seq_len = x.shape[2]

        # QKV projection: [1, 1, seq_len, hidden] -> [1, 1, seq_len, 3*hidden]
        qkv = ttnn.linear(
            x,
            self.att_proj_weight,
            bias=self.att_proj_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        # Move to CPU for attention computation (Stage 1 bring-up: correctness first)
        qkv = ttnn.to_torch(qkv).to(torch.float32)
        # qkv shape: [1, 1, seq_len, 3*hidden]
        qkv = qkv.squeeze(0)  # [1, seq_len, 3*hidden]

        # Split into Q, K, V: each [batch, seq_len, hidden]
        q, k, v = qkv.split(self.hidden_size, dim=-1)

        # Reshape for multi-head attention: [batch, num_heads, seq_len, head_dim]
        batch_size = q.shape[0]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention (PyTorch for correctness in Stage 1)
        attn_mask_torch = None
        if attention_mask is not None:
            attn_mask_torch = ttnn.to_torch(attention_mask).to(torch.float32).squeeze(0).squeeze(0)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask_torch,
            is_causal=self.is_causal if attn_mask_torch is None else False,
        )

        # Reshape back: [batch, seq_len, hidden]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        # Move back to TT device
        attn_output = load_tt_tensor(
            attn_output, self.device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
        )

        # Output projection
        output = ttnn.linear(
            attn_output,
            self.out_proj_weight,
            bias=self.out_proj_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        return output


class TtBarkMLP:
    """
    MLP (feed-forward) block: Linear(hidden, 4*hidden) → GELU → Linear(4*hidden, hidden).

    HF key names:
        mlp.in_proj.weight   -> [4*hidden, hidden]
        mlp.out_proj.weight  -> [hidden, 4*hidden]
    """

    def __init__(self, state_dict: dict, prefix: str, device: ttnn.Device, config: BarkConfig):
        self.device = device
        self.has_bias = config.bias

        self.fc_weight = load_tt_tensor(
            state_dict[f"{prefix}.in_proj.weight"].transpose(-1, -2),
            device=device,
            dtype=ttnn.bfloat8_b,
        )
        if self.has_bias and f"{prefix}.in_proj.bias" in state_dict:
            self.fc_bias = load_tt_tensor(
                state_dict[f"{prefix}.in_proj.bias"],
                device=device,
                dtype=ttnn.bfloat16,
            )
        else:
            self.fc_bias = None

        self.proj_weight = load_tt_tensor(
            state_dict[f"{prefix}.out_proj.weight"].transpose(-1, -2),
            device=device,
            dtype=ttnn.bfloat8_b,
        )
        if self.has_bias and f"{prefix}.out_proj.bias" in state_dict:
            self.proj_bias = load_tt_tensor(
                state_dict[f"{prefix}.out_proj.bias"],
                device=device,
                dtype=ttnn.bfloat16,
            )
        else:
            self.proj_bias = None

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # Up projection: [hidden] -> [4*hidden]
        x = ttnn.linear(
            x,
            self.fc_weight,
            bias=self.fc_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        # GELU activation
        x = ttnn.gelu(x, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Down projection: [4*hidden] -> [hidden]
        x = ttnn.linear(
            x,
            self.proj_weight,
            bias=self.proj_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        return x


class TtBarkBlock:
    """
    Single Bark GPT Block.

    Forward:
        residual = x
        x = layernorm_1(x)
        x = attn(x) + residual
        x = x + mlp(layernorm_2(x))

    Note: Bark uses pre-norm (not post-norm) and the MLP residual
    is applied differently from standard GPT-2.
    """

    def __init__(
        self,
        state_dict: dict,
        prefix: str,
        device: ttnn.Device,
        config: BarkConfig,
        is_causal: bool = True,
    ):
        # For causal models (semantic, coarse), LayerNorm has bias=config.bias (False)
        # For non-causal model (fine), LayerNorm always has bias=True
        ln_has_bias = not is_causal or config.bias

        self.layernorm_1 = TtBarkLayerNorm(
            state_dict, f"{prefix}.layernorm_1", device, config, has_bias=ln_has_bias
        )
        self.layernorm_2 = TtBarkLayerNorm(
            state_dict, f"{prefix}.layernorm_2", device, config, has_bias=ln_has_bias
        )
        self.attn = TtBarkSelfAttention(
            state_dict, f"{prefix}.attn", device, config, is_causal=is_causal
        )
        self.mlp = TtBarkMLP(state_dict, f"{prefix}.mlp", device, config)

    def __call__(self, x: ttnn.Tensor, attention_mask: Optional[ttnn.Tensor] = None) -> ttnn.Tensor:
        # Self-attention with residual
        residual = x
        x = self.layernorm_1(x)
        attn_out = self.attn(x, attention_mask=attention_mask)
        intermediary = ttnn.add(attn_out, residual, memory_config=ttnn.L1_MEMORY_CONFIG)

        # MLP with residual (Bark style: intermediary + MLP(LN(intermediary)))
        mlp_input = self.layernorm_2(intermediary)
        mlp_out = self.mlp(mlp_input)
        output = ttnn.add(intermediary, mlp_out, memory_config=ttnn.L1_MEMORY_CONFIG)

        return output


class TtBarkGPT:
    """
    Full GPT model used by each Bark stage.

    Structure:
    - input_embeds_layer (Embedding)
    - position_embeds_layer (Embedding)
    - layers (N × BarkBlock)
    - layernorm_final (LayerNorm)
    - lm_head (Linear, no bias)
    """

    def __init__(
        self,
        state_dict: dict,
        prefix: str,
        device: ttnn.Device,
        config: BarkConfig,
        input_vocab_size: int,
        output_vocab_size: int,
        is_causal: bool = True,
    ):
        self.device = device
        self.config = config
        self.hidden_size = config.hidden_size
        self.is_causal = is_causal

        # Input embeddings
        self.input_embeds_weight = load_tt_tensor(
            state_dict[f"{prefix}.input_embeds_layer.weight"],
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
        )

        self.position_embeds_weight = load_tt_tensor(
            state_dict[f"{prefix}.position_embeds_layer.weight"],
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
        )

        # Transformer blocks
        self.layers = []
        for i in range(config.num_hidden_layers):
            block = TtBarkBlock(
                state_dict,
                f"{prefix}.layers.{i}",
                device,
                config,
                is_causal=is_causal,
            )
            self.layers.append(block)

        # Final layer norm
        ln_has_bias = not is_causal or config.bias
        self.layernorm_final = TtBarkLayerNorm(
            state_dict, f"{prefix}.layernorm_final", device, config, has_bias=ln_has_bias
        )

        # LM head (no bias in Bark)
        self.lm_head_weight = load_tt_tensor(
            state_dict[f"{prefix}.lm_head.weight"].transpose(-1, -2),
            device=device,
            dtype=ttnn.bfloat8_b,
        )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Forward pass through the GPT model.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Optional attention mask

        Returns:
            logits: [1, 1, seq_len, output_vocab_size]
        """
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
        seq_len = input_ids.shape[-1]

        # Token embeddings
        input_ids_tt = ttnn.from_torch(
            input_ids,
            device=self.device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.uint32,
        )
        token_embeds = ttnn.embedding(
            input_ids_tt,
            self.input_embeds_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Position embeddings
        position_ids = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0)
        position_ids_tt = ttnn.from_torch(
            position_ids,
            device=self.device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.uint32,
        )
        position_embeds = ttnn.embedding(
            position_ids_tt,
            self.position_embeds_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Combine embeddings: token + position
        x = ttnn.add(token_embeds, position_embeds, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.reshape(x, [1, 1, seq_len, self.hidden_size])
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # Transformer blocks
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)

        # Final layer norm
        x = self.layernorm_final(x)

        # LM head (no bias)
        logits = ttnn.linear(
            x,
            self.lm_head_weight,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        return logits
