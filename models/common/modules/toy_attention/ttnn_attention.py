"""
TTNN-first attention implementations for TTTv2.

This file keeps the core semantics TTNN-only (no torch/HF/vLLM) and
provides a minimal multi-head attention core that honors the
AttentionSpec/AttentionConfig interfaces described in the design doc.
"""

import math
from dataclasses import dataclass
from typing import Any, Optional

import ttnn

from .core import AttentionConfig, AttentionSpec, BaseAttentionCore, OpConfig


def default_memory_config(shard_layout: Optional[str] = None) -> ttnn.MemoryConfig:
    """
    Provide a simple per-module default memory config.

    The design doc calls for module-owned defaults that can be overridden
    by callers; this keeps the default small and explicit.
    """
    layout_map = {
        "block": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        "height": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "width": ttnn.TensorMemoryLayout.WIDTH_SHARDED if hasattr(ttnn.TensorMemoryLayout, "WIDTH_SHARDED") else None,
    }
    memory_layout = layout_map.get(shard_layout, ttnn.TensorMemoryLayout.INTERLEAVED)
    buffer_type = ttnn.BufferType.L1 if memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED else ttnn.BufferType.DRAM
    return ttnn.MemoryConfig(memory_layout=memory_layout, buffer_type=buffer_type)


def ensure_config_defaults(config: AttentionConfig) -> AttentionConfig:
    """
    Ensure each op config carries module defaults when the caller does not override them.
    """

    def _fill(op_cfg: OpConfig) -> None:
        if op_cfg.memory_config is None:
            op_cfg.memory_config = default_memory_config(
                shard_layout=getattr(config, "shard_layout", None),
            )

    _fill(config.qkv)
    _fill(config.q)
    _fill(config.k)
    _fill(config.v)
    _fill(config.attn_scores)
    _fill(config.attn_output)
    _fill(config.out_proj)
    return config


@dataclass
class AttentionWeights:
    """
    Container for attention weights.

    Callers can supply fused qkv weights or separate q/k/v weights. Output
    projection weights are always required.
    """

    # Fused path
    qkv_weight: Optional[ttnn.Tensor] = None
    qkv_bias: Optional[ttnn.Tensor] = None

    # Unfused path
    q_weight: Optional[ttnn.Tensor] = None
    k_weight: Optional[ttnn.Tensor] = None
    v_weight: Optional[ttnn.Tensor] = None
    q_bias: Optional[ttnn.Tensor] = None
    k_bias: Optional[ttnn.Tensor] = None
    v_bias: Optional[ttnn.Tensor] = None

    # Output projection
    o_weight: Optional[ttnn.Tensor] = None
    o_bias: Optional[ttnn.Tensor] = None

    def validate(self, fused_qkv: bool) -> None:
        if fused_qkv:
            if self.qkv_weight is None:
                raise ValueError("qkv_weight is required for fused attention.")
        else:
            if self.q_weight is None or self.k_weight is None or self.v_weight is None:
                raise ValueError("q_weight, k_weight, and v_weight are required for unfused attention.")
        if self.o_weight is None:
            raise ValueError("o_weight is required for output projection.")


class TTNNMultiheadAttentionCore(BaseAttentionCore):
    """
    Standard multi-head attention implemented with TTNN ops.

    - TTNN-only: no torch/HF/vLLM dependencies.
    - Uses attention spec for shape, config for device-facing knobs.
    - Supports fused or unfused QKV projections and optional flash attention.
    """

    def __init__(
        self,
        spec: AttentionSpec,
        config: Optional[AttentionConfig] = None,
        *,
        fused_qkv: bool = True,
        use_flash: bool = False,
    ):
        super().__init__(spec=spec, config=config or AttentionConfig())
        self.config = ensure_config_defaults(self.config)
        self.fused_qkv = fused_qkv
        self.use_flash = use_flash

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        weights: AttentionWeights,
        attention_mask: Optional[ttnn.Tensor] = None,
        mode: str = "decode",
        training: bool = False,
        **kwargs: Any,
    ) -> ttnn.Tensor:
        """
        Compute attention for `hidden_states`.

        Args:
            hidden_states: TTNN tensor shaped [batch, seq, hidden].
            weights: attention weights (fused or unfused).
            attention_mask: optional additive mask broadcastable to [batch, heads, seq, seq].
            mode: "prefill" or "decode" (placeholder, future hooks for kv-cache policy).
            training: toggles dropout.
        """
        weights.validate(fused_qkv=self.fused_qkv)

        if self.fused_qkv:
            qkv = ttnn.linear(
                hidden_states,
                weights.qkv_weight,
                bias=weights.qkv_bias,
                memory_config=self.config.qkv.memory_config,
                program_config=self.config.qkv.program_config,
                compute_kernel_config=self.config.qkv.compute_kernel_config,
            )
            batch_size, seq_len, _ = hidden_states.shape
            qkv = ttnn.reshape(
                qkv,
                [
                    batch_size,
                    seq_len,
                    3,
                    self.spec.num_heads,
                    self.spec.head_dim,
                ],
            )
            query_states = qkv[:, :, 0, :, :]
            key_states = qkv[:, :, 1, :, :]
            value_states = qkv[:, :, 2, :, :]
        else:
            query_states = ttnn.linear(
                hidden_states,
                weights.q_weight,
                bias=weights.q_bias,
                memory_config=self.config.q.memory_config,
                program_config=self.config.q.program_config,
                compute_kernel_config=self.config.q.compute_kernel_config,
            )
            key_states = ttnn.linear(
                hidden_states,
                weights.k_weight,
                bias=weights.k_bias,
                memory_config=self.config.k.memory_config,
                program_config=self.config.k.program_config,
                compute_kernel_config=self.config.k.compute_kernel_config,
            )
            value_states = ttnn.linear(
                hidden_states,
                weights.v_weight,
                bias=weights.v_bias,
                memory_config=self.config.v.memory_config,
                program_config=self.config.v.program_config,
                compute_kernel_config=self.config.v.compute_kernel_config,
            )
            batch_size, seq_len, _ = hidden_states.shape
            target_shape = [batch_size, seq_len, self.spec.num_heads, self.spec.head_dim]
            query_states = ttnn.reshape(query_states, target_shape)
            key_states = ttnn.reshape(key_states, target_shape)
            value_states = ttnn.reshape(value_states, target_shape)

        # Flash attention path
        if self.use_flash:
            attention_output = ttnn.flash_attention(
                query_states,
                key_states,
                value_states,
                is_causal=True,
                attention_mask=attention_mask,
                compute_kernel_config=self.config.attn_scores.compute_kernel_config,
                memory_config=self.config.attn_scores.memory_config,
                program_config=self.config.attn_scores.program_config,
            )
        else:
            # Standard attention: [batch, heads, seq, head_dim]
            query_states = ttnn.transpose(query_states, 1, 2)
            key_states = ttnn.transpose(key_states, 1, 2)
            value_states = ttnn.transpose(value_states, 1, 2)

            attention_scores = ttnn.matmul(
                query_states,
                ttnn.transpose(key_states, -2, -1),
                memory_config=self.config.attn_scores.memory_config,
                program_config=self.config.attn_scores.program_config,
                compute_kernel_config=self.config.attn_scores.compute_kernel_config,
            )

            scale = 1.0 / math.sqrt(self.spec.head_dim)
            attention_scores = attention_scores * scale

            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

            attention_probs = ttnn.softmax(attention_scores, dim=-1)

            if training and self.spec.dropout > 0:
                attention_probs = ttnn.dropout(attention_probs, p=self.spec.dropout)

            attention_output = ttnn.matmul(
                attention_probs,
                value_states,
                memory_config=self.config.attn_output.memory_config,
                program_config=self.config.attn_output.program_config,
                compute_kernel_config=self.config.attn_output.compute_kernel_config,
            )

            attention_output = ttnn.transpose(attention_output, 1, 2)

        # Output projection back to [batch, seq, hidden]
        attention_output = ttnn.reshape(
            attention_output,
            [
                hidden_states.shape[0],
                hidden_states.shape[1],
                self.spec.total_dim,
            ],
        )
        output = ttnn.linear(
            attention_output,
            weights.o_weight,
            bias=weights.o_bias,
            memory_config=self.config.out_proj.memory_config,
            program_config=self.config.out_proj.program_config,
            compute_kernel_config=self.config.out_proj.compute_kernel_config,
        )
        return output
