# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN Gemma-3 text encoder for LTX-2.

Forward-only encoder (no KV cache, no autoregressive generation).
Runs all tokens through the decoder layers and returns hidden states.
Follows the T5Encoder pattern from tt_dit/encoders/t5/model_t5.py.

Architecture: Gemma-3 12B text model
- 48 layers, dim=3840, 16 Q heads, 8 KV heads (GQA), head_dim=256
- SiLU-gated MLP, RMSNorm, RoPE (theta=1e6)
"""

from __future__ import annotations

import math

import torch

import ttnn

from ...layers.embeddings import Embedding
from ...layers.linear import ColParallelLinear, RowParallelLinear
from ...layers.module import Module, ModuleList
from ...layers.normalization import RMSNorm
from ...parallel.config import EncoderParallelConfig
from ...parallel.manager import CCLManager
from ...utils.substate import pop_substate, rename_substate


class GemmaConfig:
    """Configuration for Gemma-3 text encoder."""

    def __init__(
        self,
        vocab_size: int = 262208,
        hidden_size: int = 3840,
        intermediate_size: int = 15360,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 256,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        max_position_embeddings: int = 8192,
        hidden_layer_index: int = -1,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.hidden_layer_index = hidden_layer_index


class GemmaRMSNorm(RMSNorm):
    """Gemma RMSNorm — wrapper matching T5RMSNorm pattern."""

    def __init__(self, config: GemmaConfig, mesh_device):
        super().__init__(
            embedding_dim=config.hidden_size,
            norm_eps=config.rms_norm_eps,
            bias=False,
            mesh_device=mesh_device,
        )


class GemmaRotaryEmbedding(Module):
    """Precompute RoPE cos/sin tables on host, store on device."""

    def __init__(self, config: GemmaConfig, mesh_device):
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        self.max_seq_len = config.max_position_embeddings
        self.mesh_device = mesh_device

        # Precompute frequencies
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float64) / self.head_dim))
        t = torch.arange(min(self.max_seq_len, 8192), dtype=torch.float64)
        freqs = torch.outer(t, inv_freq).float()
        self._cos_cached = freqs.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, seq, D/2)
        self._sin_cached = freqs.sin().unsqueeze(0).unsqueeze(0)

    def get_cos_sin(self, seq_len: int, device) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Get cos/sin for given sequence length, push to device."""
        cos = self._cos_cached[:, :, :seq_len, :].bfloat16()
        sin = self._sin_cached[:, :, :seq_len, :].bfloat16()
        tt_cos = ttnn.from_torch(cos, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        tt_sin = ttnn.from_torch(sin, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        return tt_cos, tt_sin


class GemmaAttention(Module):
    """Gemma GQA self-attention with RoPE. No KV cache."""

    def __init__(
        self,
        config: GemmaConfig,
        mesh_device,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
    ):
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        tp = parallel_config.tensor_parallel.factor
        self.num_local_heads = self.num_heads // tp
        self.num_local_kv_heads = self.num_kv_heads // tp

        col_kwargs = {
            "bias": False,
            "mesh_device": mesh_device,
            "mesh_axis": parallel_config.tensor_parallel.mesh_axis,
        }

        self.q_proj = ColParallelLinear(self.hidden_size, self.num_heads * self.head_dim, **col_kwargs)
        self.k_proj = ColParallelLinear(self.hidden_size, self.num_kv_heads * self.head_dim, **col_kwargs)
        self.v_proj = ColParallelLinear(self.hidden_size, self.num_kv_heads * self.head_dim, **col_kwargs)
        self.o_proj = ColParallelLinear(self.num_heads * self.head_dim, self.hidden_size, **col_kwargs)

        self.input_layernorm = GemmaRMSNorm(config, mesh_device)

        self.sdpa_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=256,
            k_chunk_size=256,
            exp_approx_mode=False,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "self_attn.q_proj", "q_proj")
        rename_substate(state, "self_attn.k_proj", "k_proj")
        rename_substate(state, "self_attn.v_proj", "v_proj")
        rename_substate(state, "self_attn.o_proj", "o_proj")

    def _apply_rope(self, x, cos, sin):
        """Apply rotary embedding: split x into halves, rotate."""
        d = self.head_dim // 2
        x1 = x[:, :, :, :d]
        x2 = x[:, :, :, d:]
        out1 = ttnn.subtract(ttnn.multiply(x1, cos), ttnn.multiply(x2, sin))
        out2 = ttnn.add(ttnn.multiply(x2, cos), ttnn.multiply(x1, sin))
        return ttnn.concat([out1, out2], dim=-1)

    def forward(self, hidden_states, cos, sin):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # QKV projections
        q = self.q_proj(hidden_states, compute_kernel_config=self.compute_config)
        k = self.k_proj(hidden_states, compute_kernel_config=self.compute_config)
        v = self.v_proj(hidden_states, compute_kernel_config=self.compute_config)

        # Split into heads: (B, seq, H*D) → (B, H, seq, D)
        B, seq_len = q.shape[0], q.shape[1]
        q = ttnn.reshape(q, (B, seq_len, self.num_local_heads, self.head_dim))
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.reshape(k, (B, seq_len, self.num_local_kv_heads, self.head_dim))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.reshape(v, (B, seq_len, self.num_local_kv_heads, self.head_dim))
        v = ttnn.permute(v, (0, 2, 1, 3))

        # Apply RoPE
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        # GQA: repeat KV heads to match Q heads
        if self.num_local_kv_heads < self.num_local_heads:
            repeats = self.num_local_heads // self.num_local_kv_heads
            k = ttnn.repeat(k, ttnn.Shape([1, repeats, 1, 1]))
            v = ttnn.repeat(v, ttnn.Shape([1, repeats, 1, 1]))

        # SDPA
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            program_config=self.sdpa_config,
            compute_kernel_config=self.compute_config,
        )

        # Concat heads: (B, H, seq, D) → (B, seq, H*D)
        attn_output = ttnn.transformer.concatenate_heads(attn_output)

        # Output projection
        attn_output = ttnn.unsqueeze(attn_output, 0)
        if self.parallel_config.tensor_parallel.factor > 1:
            attn_output = self.ccl_manager.all_gather(
                attn_output,
                dim=3,
                mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
                use_hyperparams=True,
            )
        output = self.o_proj(attn_output, compute_kernel_config=self.compute_config)
        if self.parallel_config.tensor_parallel.factor > 1:
            output = self.ccl_manager.all_gather(
                output,
                dim=3,
                mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
                use_hyperparams=True,
            )
        output = ttnn.squeeze(output, 0)

        return output + residual


class GemmaFF(Module):
    """Gemma SiLU-gated MLP: gate_proj * silu(up_proj) → down_proj."""

    def __init__(
        self,
        config: GemmaConfig,
        mesh_device,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
    ):
        super().__init__()
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager

        col_kwargs = {
            "bias": False,
            "mesh_device": mesh_device,
            "mesh_axis": parallel_config.tensor_parallel.mesh_axis,
        }

        self.gate_proj = ColParallelLinear(
            config.hidden_size, config.intermediate_size, activation_fn="silu", **col_kwargs
        )
        self.up_proj = ColParallelLinear(config.hidden_size, config.intermediate_size, **col_kwargs)
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )

        self.post_attention_layernorm = GemmaRMSNorm(config, mesh_device)

        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "mlp.gate_proj", "gate_proj")
        rename_substate(state, "mlp.up_proj", "up_proj")
        rename_substate(state, "mlp.down_proj", "down_proj")

    def forward(self, x):
        residual = x
        x = self.post_attention_layernorm(x)

        gate = self.gate_proj(x, compute_kernel_config=self.compute_config)
        up = self.up_proj(x, compute_kernel_config=self.compute_config)
        x = gate * up

        x = self.down_proj(x, compute_kernel_config=self.compute_config)
        x = ttnn.unsqueeze(x, 0)
        if self.parallel_config.tensor_parallel.factor > 1:
            x = self.ccl_manager.all_gather(
                x,
                dim=3,
                mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
                use_hyperparams=True,
            )
        x = ttnn.squeeze(x, 0)

        return x + residual


class GemmaEncoderLayer(Module):
    """Single Gemma decoder layer used as encoder (no KV cache)."""

    def __init__(self, config, mesh_device, ccl_manager, parallel_config):
        super().__init__()
        self.self_attn = GemmaAttention(config, mesh_device, ccl_manager, parallel_config)
        self.ff = GemmaFF(config, mesh_device, ccl_manager, parallel_config)

    def _prepare_torch_state(self, state):
        # HF keys: self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj, input_layernorm, post_attention_layernorm
        # Route self_attn and mlp substates to their respective modules
        rename_substate(state, "input_layernorm", "self_attn.input_layernorm")
        rename_substate(state, "post_attention_layernorm", "ff.post_attention_layernorm")
        rename_substate(state, "mlp.gate_proj", "ff.gate_proj")
        rename_substate(state, "mlp.up_proj", "ff.up_proj")
        rename_substate(state, "mlp.down_proj", "ff.down_proj")

    def forward(self, hidden_states, cos, sin):
        hidden_states = self.self_attn(hidden_states, cos, sin)
        hidden_states = self.ff(hidden_states)
        return hidden_states


class GemmaEncoder(Module):
    """
    TTNN Gemma-3 text encoder.

    Runs the full decoder stack on input tokens and returns hidden states
    from the specified layer (default: last layer before final norm).

    Usage:
        encoder = GemmaEncoder(config, mesh_device, ccl_manager, parallel_config)
        encoder.load_torch_state_dict(state_dict)
        hidden_states = encoder(token_ids, attention_mask)
    """

    def __init__(
        self,
        config: GemmaConfig,
        mesh_device,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
    ):
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device

        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, device=mesh_device)
        self.rotary_emb = GemmaRotaryEmbedding(config, mesh_device)

        self.layers = ModuleList(
            GemmaEncoderLayer(config, mesh_device, ccl_manager, parallel_config)
            for _ in range(config.num_hidden_layers)
        )

        self.norm = GemmaRMSNorm(config, mesh_device)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # HF key prefix: model.embed_tokens, model.layers.N.*, model.norm
        rename_substate(state, "model.embed_tokens", "embed_tokens")
        rename_substate(state, "model.norm", "norm")
        rename_substate(state, "model.layers", "layers")
        # Remove lm_head (not needed for encoding)
        pop_substate(state, "lm_head")

    def forward(self, token_ids: ttnn.Tensor, attention_mask: ttnn.Tensor | None = None) -> list[ttnn.Tensor]:
        """Run forward and return hidden states from all layers.

        Args:
            token_ids: (B, seq_len) int32 token IDs on device
            attention_mask: (B, seq_len) float mask on device (1=attend, 0=pad)

        Returns:
            List of hidden states tensors, one per layer + final norm.
        """
        # Embed tokens
        hidden_states = self.embed_tokens(token_ids)

        # Scale embeddings (Gemma-specific)
        hidden_states = ttnn.multiply(hidden_states, math.sqrt(self.config.hidden_size))

        # Get RoPE cos/sin for this sequence length
        seq_len = token_ids.shape[-1]
        cos, sin = self.rotary_emb.get_cos_sin(seq_len, self.mesh_device)

        # Run through all layers
        all_hidden_states = [hidden_states]
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin)
            all_hidden_states.append(hidden_states)

        # Final norm
        output = self.norm(hidden_states)
        all_hidden_states.append(output)

        return all_hidden_states
