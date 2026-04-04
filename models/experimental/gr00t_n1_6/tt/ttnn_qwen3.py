# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-1.7B Language Model - TTNN Implementation for GR00T N1.6.

Qwen3-1.7B architecture (as used in Eagle-Block2A backbone):
    - 28 transformer layers (GR00T uses only first 16)
    - 2048 hidden dimension
    - 16 attention heads, 8 KV heads (Grouped-Query Attention)
    - 128 head_dim (tile-aligned, no padding needed)
    - RMSNorm pre-norm (no bias)
    - QK normalization: RMSNorm applied per-head before RoPE
    - SiLU-gated MLP (gate_proj * up_proj -> down_proj)
    - RoPE with theta=1,000,000

GR00T usage:
    - Single forward pass (prefill only, no KV cache)
    - Token embedding on CPU (vocab_size=151680, too large for device table)
    - Vision features inserted at image placeholder positions before layer 0
    - Layer 16 hidden states at image token positions feed the DiT action head
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch

import ttnn
from models.experimental.gr00t_n1_6.common.configs import Qwen3Config
from models.experimental.gr00t_n1_6.tt.ttnn_common import (
    CORE_GRID_BH,
    preprocess_linear_weight,
    preprocess_rmsnorm_params,
)


# ---------------------------------------------------------------------------
# RoPE utilities (CPU)
# ---------------------------------------------------------------------------

def precompute_freqs_cis(
    head_dim: int,
    max_seq_len: int,
    theta: float = 1_000_000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute RoPE cos/sin tables on CPU.

    Args:
        head_dim: Attention head dimension (128 for Qwen3-1.7B).
        max_seq_len: Maximum sequence length to precompute.
        theta: RoPE base frequency (1_000_000 for Qwen3).

    Returns:
        cos, sin: each [max_seq_len, head_dim//2] in float32.
    """
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))  # [half]
    t = torch.arange(max_seq_len, dtype=torch.float32)                           # [seq]
    freqs = torch.outer(t, freqs)                                                 # [seq, half]
    return freqs.cos(), freqs.sin()


def apply_rotary_emb_cpu(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary position embeddings on CPU.

    Args:
        x:   [batch, heads, seq, head_dim]
        cos: [seq, head_dim//2]
        sin: [seq, head_dim//2]

    Returns:
        [batch, heads, seq, head_dim] with RoPE applied.
    """
    head_dim = x.shape[-1]
    x1 = x[..., : head_dim // 2]   # [B, H, seq, half]
    x2 = x[..., head_dim // 2 :]   # [B, H, seq, half]
    # Broadcast cos/sin over batch and heads dimensions
    c = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, half]
    s = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, half]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


# ---------------------------------------------------------------------------
# RMSNorm helper (on-device)
# ---------------------------------------------------------------------------

def _rms_norm_tt(
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    eps: float,
) -> ttnn.Tensor:
    """Apply RMSNorm using TTNN's rms_norm op."""
    return ttnn.rms_norm(x, weight=weight, epsilon=eps, memory_config=ttnn.L1_MEMORY_CONFIG)


# ---------------------------------------------------------------------------
# Qwen3 MLP
# ---------------------------------------------------------------------------

class Qwen3MLPTTNN:
    """
    SiLU-gated MLP block for Qwen3.

    Computes: down_proj(silu(gate_proj(x)) * up_proj(x))

    All inputs arrive post-norm (RMSNorm is applied in the decoder layer
    before calling this module).  No bias on any projection.
    """

    def __init__(
        self,
        gate_proj_weight: torch.Tensor,   # [intermediate, hidden]
        up_proj_weight: torch.Tensor,     # [intermediate, hidden]
        down_proj_weight: torch.Tensor,   # [hidden, intermediate]
        device: Any,
    ):
        # Use bfloat16 (not bfloat8_b) to preserve numerical fidelity over 16 layers.
        self.gate_weight = preprocess_linear_weight(gate_proj_weight, device, dtype=ttnn.bfloat16)
        self.up_weight   = preprocess_linear_weight(up_proj_weight,   device, dtype=ttnn.bfloat16)
        self.down_weight = preprocess_linear_weight(down_proj_weight,  device, dtype=ttnn.bfloat16)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Args:
            x: [batch, seq, hidden_size] – pre-normed activations.

        Returns:
            [batch, seq, hidden_size]
        """
        # gate path: silu(gate_proj(x))
        gate = ttnn.linear(
            x, self.gate_weight,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            core_grid=CORE_GRID_BH,
            activation="silu",
        )

        # up path: up_proj(x)
        up = ttnn.linear(
            x, self.up_weight,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            core_grid=CORE_GRID_BH,
        )

        # element-wise product
        hidden = ttnn.mul(gate, up, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        # down projection
        out = ttnn.linear(
            hidden, self.down_weight,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            core_grid=CORE_GRID_BH,
        )
        ttnn.deallocate(hidden)
        return out


# ---------------------------------------------------------------------------
# Qwen3 Attention
# ---------------------------------------------------------------------------

class Qwen3AttentionTTNN:
    """
    Grouped-Query Attention (GQA) for Qwen3-1.7B.

    Key differences from standard MHA:
      - num_kv_heads=8, num_q_heads=16  -> KV heads are repeated 2x before attention.
      - QK normalization: RMSNorm applied to Q and K per-head before RoPE.
      - RoPE with theta=1,000,000.
      - Causal mask for prefill.
      - No bias on any projection.

    RoPE and QK-norm are applied on CPU (the tensors are brought back from device,
    manipulated, then re-uploaded).  This avoids complex on-device gather/scatter
    for the rotate-half operation which is not natively supported in TTNN.
    """

    def __init__(
        self,
        q_proj_weight: torch.Tensor,   # [num_heads*head_dim, hidden]
        k_proj_weight: torch.Tensor,   # [num_kv_heads*head_dim, hidden]
        v_proj_weight: torch.Tensor,   # [num_kv_heads*head_dim, hidden]
        o_proj_weight: torch.Tensor,   # [hidden, num_heads*head_dim]
        q_norm_weight: torch.Tensor,   # [head_dim]
        k_norm_weight: torch.Tensor,   # [head_dim]
        config: Qwen3Config,
        device: Any,
    ):
        self.config = config
        self.device = device
        self.num_heads    = config.num_attention_heads   # 16
        self.num_kv_heads = config.num_key_value_heads   # 8
        self.head_dim     = config.head_dim              # 128
        self.scale        = 1.0 / math.sqrt(self.head_dim)

        # bfloat16 weights for numerical stability
        self.q_weight = preprocess_linear_weight(q_proj_weight, device, dtype=ttnn.bfloat16)
        self.k_weight = preprocess_linear_weight(k_proj_weight, device, dtype=ttnn.bfloat16)
        self.v_weight = preprocess_linear_weight(v_proj_weight, device, dtype=ttnn.bfloat16)
        self.o_weight = preprocess_linear_weight(o_proj_weight, device, dtype=ttnn.bfloat16)

        # QK norm weights: [head_dim] -> stored as [1, head_dim] for TTNN rms_norm
        # We store as CPU tensors and apply on CPU since QK-norm is per-head
        self.q_norm_weight_cpu = q_norm_weight.to(torch.float32)  # [head_dim]
        self.k_norm_weight_cpu = k_norm_weight.to(torch.float32)  # [head_dim]

    def _qk_norm_cpu(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """
        RMSNorm along the last dimension (head_dim).

        Args:
            x:      [..., head_dim]
            weight: [head_dim]
            eps:    epsilon

        Returns:
            [..., head_dim]
        """
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
        return (x.float() / rms * weight).to(torch.bfloat16)

    def __call__(
        self,
        x: ttnn.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> ttnn.Tensor:
        """
        Forward pass for one attention layer.

        Args:
            x:           [batch, seq, hidden_size] on device.
            cos:         [seq, head_dim//2] RoPE cosines on CPU.
            sin:         [seq, head_dim//2] RoPE sines on CPU.
            causal_mask: [seq, seq] additive causal mask on CPU (0 or -1e9).

        Returns:
            [batch, seq, hidden_size] on device.
        """
        batch_size, seq_len, hidden_size = x.shape

        # --- Q, K, V projections (on device) ---
        q = ttnn.linear(x, self.q_weight, memory_config=ttnn.L1_MEMORY_CONFIG,
                        dtype=ttnn.bfloat16, core_grid=CORE_GRID_BH)
        k = ttnn.linear(x, self.k_weight, memory_config=ttnn.L1_MEMORY_CONFIG,
                        dtype=ttnn.bfloat16, core_grid=CORE_GRID_BH)
        v = ttnn.linear(x, self.v_weight, memory_config=ttnn.L1_MEMORY_CONFIG,
                        dtype=ttnn.bfloat16, core_grid=CORE_GRID_BH)

        # --- Transfer Q, K, V to CPU for QK-norm and RoPE ---
        q_cpu = ttnn.to_torch(q).to(torch.bfloat16)
        k_cpu = ttnn.to_torch(k).to(torch.bfloat16)
        v_cpu = ttnn.to_torch(v).to(torch.bfloat16)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # Reshape to [batch, heads, seq, head_dim]
        q_cpu = q_cpu.reshape(batch_size, seq_len, self.num_heads,    self.head_dim).permute(0, 2, 1, 3)
        k_cpu = k_cpu.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v_cpu = v_cpu.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        # --- QK normalization (RMSNorm per-head, before RoPE) ---
        q_cpu = self._qk_norm_cpu(q_cpu, self.q_norm_weight_cpu, self.config.rms_norm_eps)
        k_cpu = self._qk_norm_cpu(k_cpu, self.k_norm_weight_cpu, self.config.rms_norm_eps)

        # --- RoPE ---
        # cos/sin are [seq, head_dim//2]; slice to current seq_len
        cos_s = cos[:seq_len]
        sin_s = sin[:seq_len]
        q_cpu = apply_rotary_emb_cpu(q_cpu, cos_s, sin_s)
        k_cpu = apply_rotary_emb_cpu(k_cpu, cos_s, sin_s)

        # --- Expand KV heads (GQA: 8 -> 16) ---
        repeat_factor = self.num_heads // self.num_kv_heads  # 2
        k_cpu = k_cpu.repeat_interleave(repeat_factor, dim=1)  # [B, 16, seq, head_dim]
        v_cpu = v_cpu.repeat_interleave(repeat_factor, dim=1)  # [B, 16, seq, head_dim]

        # --- Scale Q ---
        q_cpu = q_cpu * self.scale  # [B, 16, seq, head_dim]

        # --- Attention scores: Q @ K^T ---
        # [B, 16, seq, seq]
        attn_scores = torch.matmul(q_cpu.float(), k_cpu.float().transpose(-2, -1))

        # --- Causal mask ---
        # causal_mask: [seq, seq] additive (0 or -1e9)
        attn_scores = attn_scores + causal_mask.unsqueeze(0).unsqueeze(0).float()

        # --- Softmax ---
        attn_weights = torch.softmax(attn_scores, dim=-1).to(torch.bfloat16)

        # --- Context: weights @ V ---
        # [B, 16, seq, head_dim]
        context = torch.matmul(attn_weights.float(), v_cpu.float()).to(torch.bfloat16)

        # --- Reshape back: [B, seq, hidden_size] ---
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.reshape(batch_size, seq_len, self.num_heads * self.head_dim)

        # --- Transfer back to device for output projection ---
        context_tt = ttnn.from_torch(
            context,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        out = ttnn.linear(
            context_tt, self.o_weight,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            core_grid=CORE_GRID_BH,
        )
        ttnn.deallocate(context_tt)
        return out


# ---------------------------------------------------------------------------
# Qwen3 Decoder Layer
# ---------------------------------------------------------------------------

class Qwen3DecoderLayerTTNN:
    """
    Single Qwen3 transformer decoder layer.

    Pre-norm design:
        x -> input_layernorm -> self_attn -> residual
          -> post_attention_layernorm -> mlp -> residual
    """

    def __init__(
        self,
        layer_weights: Dict[str, torch.Tensor],
        config: Qwen3Config,
        device: Any,
    ):
        """
        Args:
            layer_weights: Dict of weight tensors with keys matching Qwen3 naming.
            config:        Qwen3Config instance.
            device:        TTNN device.
        """
        self.config = config
        self.device = device
        eps = config.rms_norm_eps

        # Pre-attention RMSNorm
        self.input_ln_weight = preprocess_rmsnorm_params(
            layer_weights["input_layernorm.weight"], device,
        )

        # Self-attention
        self.attn = Qwen3AttentionTTNN(
            q_proj_weight=layer_weights["self_attn.q_proj.weight"],
            k_proj_weight=layer_weights["self_attn.k_proj.weight"],
            v_proj_weight=layer_weights["self_attn.v_proj.weight"],
            o_proj_weight=layer_weights["self_attn.o_proj.weight"],
            q_norm_weight=layer_weights["self_attn.q_norm.weight"],
            k_norm_weight=layer_weights["self_attn.k_norm.weight"],
            config=config,
            device=device,
        )

        # Post-attention RMSNorm
        self.post_attn_ln_weight = preprocess_rmsnorm_params(
            layer_weights["post_attention_layernorm.weight"], device,
        )

        # MLP
        self.mlp = Qwen3MLPTTNN(
            gate_proj_weight=layer_weights["mlp.gate_proj.weight"],
            up_proj_weight=layer_weights["mlp.up_proj.weight"],
            down_proj_weight=layer_weights["mlp.down_proj.weight"],
            device=device,
        )

        self._eps = eps

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> ttnn.Tensor:
        """
        Args:
            hidden_states: [batch, seq, hidden_size] on device.
            cos:           [max_seq, head_dim//2] RoPE cosines on CPU.
            sin:           [max_seq, head_dim//2] RoPE sines on CPU.
            causal_mask:   [seq, seq] additive mask on CPU.

        Returns:
            [batch, seq, hidden_size] on device.
        """
        residual = hidden_states

        # Pre-attention norm
        normed = _rms_norm_tt(hidden_states, self.input_ln_weight, self._eps)

        # Self-attention
        attn_out = self.attn(normed, cos, sin, causal_mask)
        ttnn.deallocate(normed)

        # Residual connection
        hidden_states = ttnn.add(
            residual, attn_out,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(attn_out)

        residual = hidden_states

        # Post-attention norm
        normed = _rms_norm_tt(hidden_states, self.post_attn_ln_weight, self._eps)

        # MLP
        mlp_out = self.mlp(normed)
        ttnn.deallocate(normed)

        # Residual connection
        hidden_states = ttnn.add(
            residual, mlp_out,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(mlp_out)

        return hidden_states


# ---------------------------------------------------------------------------
# Full Qwen3 Model
# ---------------------------------------------------------------------------

class Qwen3ModelTTNN:
    """
    Qwen3-1.7B language model for GR00T N1.6 on TTNN.

    Runs the first ``select_layer`` layers (default 16 out of 28) and
    returns the hidden states at that layer.  Those hidden states at the
    image token positions are used as backbone features for the DiT action head.

    Key design decisions:
      - Token embedding is performed on CPU (vocab_size 151680 is too large
        for a device embedding table lookup in TTNN).
      - Vision features are spliced in on CPU before transfer to device.
      - QK-norm and RoPE are performed on CPU inside each attention layer to
        avoid the complexity of implementing rotate-half natively in TTNN.
      - All weights are stored in bfloat16 to prevent numerical drift over 16 layers.
      - Causal mask and RoPE tables are precomputed once and reused across calls.

    Args:
        config:  Qwen3Config instance.
        weights: Dict mapping weight keys (after stripping the backbone prefix) to tensors.
        device:  TTNN device handle.
    """

    def __init__(
        self,
        config: Qwen3Config,
        weights: Dict[str, torch.Tensor],
        device: Any,
    ):
        self.config = config
        self.device = device
        self._eps = config.rms_norm_eps
        self._select_layer = config.select_layer  # 16

        # --- Token embedding (kept on CPU) ---
        embed_w = weights.get("model.embed_tokens.weight")
        if embed_w is None:
            raise KeyError("Missing 'model.embed_tokens.weight' in Qwen3 weights.")
        # [vocab_size, hidden_size] in float32 for embedding lookup
        self._embed_weight_cpu = embed_w.to(torch.float32)

        # --- Precompute RoPE tables (CPU) ---
        self._rope_cos, self._rope_sin = precompute_freqs_cis(
            config.head_dim,
            config.max_position_embeddings,
            theta=config.rope_theta,
        )  # [max_pos, head_dim//2]

        # --- Decoder layers ---
        num_layers = min(config.select_layer, config.num_hidden_layers)
        self.layers: List[Qwen3DecoderLayerTTNN] = []
        for i in range(num_layers):
            prefix = f"model.layers.{i}."
            layer_w = {
                "input_layernorm.weight":          weights[f"{prefix}input_layernorm.weight"],
                "self_attn.q_proj.weight":         weights[f"{prefix}self_attn.q_proj.weight"],
                "self_attn.k_proj.weight":         weights[f"{prefix}self_attn.k_proj.weight"],
                "self_attn.v_proj.weight":         weights[f"{prefix}self_attn.v_proj.weight"],
                "self_attn.o_proj.weight":         weights[f"{prefix}self_attn.o_proj.weight"],
                "self_attn.q_norm.weight":         weights[f"{prefix}self_attn.q_norm.weight"],
                "self_attn.k_norm.weight":         weights[f"{prefix}self_attn.k_norm.weight"],
                "post_attention_layernorm.weight": weights[f"{prefix}post_attention_layernorm.weight"],
                "mlp.gate_proj.weight":            weights[f"{prefix}mlp.gate_proj.weight"],
                "mlp.up_proj.weight":              weights[f"{prefix}mlp.up_proj.weight"],
                "mlp.down_proj.weight":            weights[f"{prefix}mlp.down_proj.weight"],
            }
            self.layers.append(Qwen3DecoderLayerTTNN(layer_w, config, device))

        # --- Final RMSNorm (applied after select_layer) ---
        norm_w = weights.get("model.norm.weight")
        if norm_w is None:
            raise KeyError("Missing 'model.norm.weight' in Qwen3 weights.")
        self.final_norm_weight = preprocess_rmsnorm_params(norm_w, device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_tokens_cpu(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Token embedding on CPU.

        Args:
            input_ids: [batch, seq_len] integer token IDs.

        Returns:
            [batch, seq_len, hidden_size] in bfloat16.
        """
        # torch.nn.functional.embedding handles arbitrary vocab sizes on CPU.
        return torch.nn.functional.embedding(
            input_ids.long(), self._embed_weight_cpu,
        ).to(torch.bfloat16)

    def _build_causal_mask(self, seq_len: int) -> torch.Tensor:
        """
        Build an additive causal mask for the given sequence length.

        Returns:
            [seq_len, seq_len] float32 tensor; 0 for allowed positions,
            -1e9 for masked (future) positions.
        """
        mask = torch.triu(
            torch.full((seq_len, seq_len), -1e9, dtype=torch.float32),
            diagonal=1,
        )
        return mask

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        input_ids: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        image_token_positions: Optional[List[int]] = None,
    ) -> ttnn.Tensor:
        """
        Forward pass through the first ``select_layer`` Qwen3 layers.

        Args:
            input_ids:
                [batch, seq_len] token IDs on CPU.  Image placeholder tokens
                should already be present in the sequence; their embeddings
                will be overwritten by ``image_features``.
            image_features:
                Optional [batch, num_image_tokens, hidden_size] vision features
                (output of pixel-shuffle + MLP connector) as a CPU tensor or
                TTNN tensor.  If provided, these replace the token embeddings
                at ``image_token_positions``.
            image_token_positions:
                List of sequence positions that correspond to image tokens.
                Must be provided when ``image_features`` is not None.

        Returns:
            [batch, seq_len, hidden_size] hidden states from layer ``select_layer``
            after final RMSNorm, on device as a TTNN tensor in bfloat16.
        """
        batch_size, seq_len = input_ids.shape

        # Step 1: Embed tokens on CPU
        hidden_states_cpu = self._embed_tokens_cpu(input_ids)
        # [batch, seq_len, hidden_size] in bfloat16

        # Step 2: Splice vision features into image placeholder positions
        if image_features is not None:
            if image_token_positions is None:
                raise ValueError(
                    "image_token_positions must be provided when image_features is not None."
                )
            # If image_features is a TTNN tensor, pull it back to CPU
            if isinstance(image_features, ttnn.Tensor):
                image_features = ttnn.to_torch(image_features).to(torch.bfloat16)
            else:
                image_features = image_features.to(torch.bfloat16)

            num_image_tokens = len(image_token_positions)
            if image_features.shape[1] != num_image_tokens:
                raise ValueError(
                    f"image_features has {image_features.shape[1]} tokens but "
                    f"image_token_positions has {num_image_tokens} positions."
                )
            # Overwrite embeddings at image positions
            for local_idx, seq_idx in enumerate(image_token_positions):
                hidden_states_cpu[:, seq_idx, :] = image_features[:, local_idx, :]

        # Step 3: Transfer hidden states to device
        hidden_states_tt = ttnn.from_torch(
            hidden_states_cpu,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Step 4: Precompute causal mask and RoPE slices for this sequence length
        causal_mask = self._build_causal_mask(seq_len)  # [seq, seq] on CPU
        cos = self._rope_cos[:seq_len]  # [seq, head_dim//2]
        sin = self._rope_sin[:seq_len]  # [seq, head_dim//2]

        # Step 5: Run through select_layer decoder layers
        for layer in self.layers:
            hidden_states_tt = layer(hidden_states_tt, cos, sin, causal_mask)

        # Step 6: Final RMSNorm
        output = _rms_norm_tt(hidden_states_tt, self.final_norm_weight, self._eps)
        if output is not hidden_states_tt:
            ttnn.deallocate(hidden_states_tt)

        return output
