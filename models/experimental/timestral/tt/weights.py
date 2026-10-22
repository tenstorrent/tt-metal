# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Host-side Timestral weights, rearranged for the TTNN dataflow (no ttnn import).

reference : models/experimental/timestral/reference/src/timestral/layers.py
    - ``wQKV`` rows are laid out ``(qkv, head_dim, n_heads)``; they are reordered to
      ``(qkv, n_heads, head_dim)``, the layout ``split_query_key_value_and_split_heads`` reads.
    - Time attention Q and K rows are further reordered within each head from rotate-half order
      to RoPE pair order (``rope_pair_order``); q . k is unchanged because both move together.
    - Every RMSNorm gain is folded into the input columns of the linears that read the norm
      (``(x_hat * g) @ W.T == x_hat @ (W * g).T``), so the device norms carry no weight.
    - The fused SwiGLU ``w12`` is split into its ``gate`` and ``up`` halves.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from models.experimental.timestral.common.timestral_src import require_timestral

require_timestral()

from timestral.config import LAYER_KIND_AXES, TimestralConfig  # noqa: E402


def _f32(t: Tensor) -> Tensor:
    return t.detach().to(torch.float32).clone()


def permute_qkv_rows(wqkv: Tensor, num_heads: int, head_dim: int) -> Tensor:
    """Reorder fused QKV output rows from ``(qkv, head_dim, heads)`` to ``(qkv, heads, head_dim)``."""
    d_in = wqkv.shape[-1]
    return wqkv.reshape(3, head_dim, num_heads, d_in).permute(0, 2, 1, 3).reshape(3 * num_heads * head_dim, d_in)


def rope_pair_order(t: Tensor, dim: int = -1) -> Tensor:
    """Reorder a rotate-half ``head_dim`` axis ``[a_0 .. a_{n/2-1}, b_0 .. b_{n/2-1}]`` into adjacent
    pairs ``[a_0, b_0, a_1, b_1, ...]``, the layout ``rotary_embedding_llama`` rotates."""
    n = t.shape[dim]
    return t.index_select(dim, torch.arange(n).reshape(2, n // 2).t().flatten())


def pair_qk_rows(wqkv: Tensor, num_heads: int, head_dim: int) -> Tensor:
    """``(qkv, heads, head_dim)`` rows -> Q and K heads in ``rope_pair_order``; V unchanged."""
    w = wqkv.reshape(3, num_heads, head_dim, -1)
    return torch.cat([rope_pair_order(w[:2], dim=2), w[2:]]).reshape(wqkv.shape)


def fold_norm_gain(weight: Tensor, gain: Tensor) -> Tensor:
    """Fold an RMSNorm gain into the input columns of an ``nn.Linear`` weight ``(out, in)``."""
    return weight * gain.reshape(1, -1)


@dataclass(frozen=True)
class ResidualBlockWeights:
    """``output(silu(hidden(x))) + residual(x)``; ``nn.Linear`` layout ``(out, in)``, biases may be None."""

    hidden_w: Tensor
    hidden_b: Tensor | None
    output_w: Tensor
    output_b: Tensor | None
    residual_w: Tensor
    residual_b: Tensor | None

    @classmethod
    def from_reference(cls, block, input_gain: Tensor | None = None) -> "ResidualBlockWeights":
        """``input_gain`` folds a preceding RMSNorm gain into the hidden and residual layers."""

        def _bias(layer):
            return None if layer.bias is None else _f32(layer.bias)

        hidden_w, residual_w = _f32(block.hidden_layer.weight), _f32(block.residual_layer.weight)
        if input_gain is not None:
            hidden_w = fold_norm_gain(hidden_w, _f32(input_gain))
            residual_w = fold_norm_gain(residual_w, _f32(input_gain))
        return cls(
            hidden_w=hidden_w,
            hidden_b=_bias(block.hidden_layer),
            output_w=_f32(block.output_layer.weight),
            output_b=_bias(block.output_layer),
            residual_w=residual_w,
            residual_b=_bias(block.residual_layer),
        )


@dataclass(frozen=True)
class AttentionWeights:
    """One gated attention sublayer; the pre-norm gain is folded into ``wqkv`` and ``wg``."""

    wqkv: Tensor  # (3 * D, D), rows (qkv, heads, head_dim); time Q / K head_dim in rope_pair_order
    wg: Tensor  # (D, D) sigmoid gate
    wo: Tensor  # (D, D)
    num_heads: int
    head_dim: int
    is_time: bool
    eps: float

    @property
    def wv(self) -> Tensor:
        d = self.num_heads * self.head_dim
        return self.wqkv[2 * d : 3 * d]

    @classmethod
    def from_reference(cls, attn, norm) -> "AttentionWeights":
        if attn.wQKV.bias is not None or attn.wO.bias is not None:
            raise NotImplementedError("attention biases are not supported (released checkpoints have none)")
        if attn._gate is None:
            raise NotImplementedError("only the sdpa_sigmoid attention gate is supported")
        gain = _f32(norm.scale)
        wqkv = permute_qkv_rows(_f32(attn.wQKV.weight), attn.num_heads, attn.head_dim)
        if attn.is_time:
            wqkv = pair_qk_rows(wqkv, attn.num_heads, attn.head_dim)
        return cls(
            wqkv=fold_norm_gain(wqkv, gain),
            wg=fold_norm_gain(_f32(attn._gate.weight), gain),
            wo=_f32(attn.wO.weight),
            num_heads=attn.num_heads,
            head_dim=attn.head_dim,
            is_time=attn.is_time,
            eps=norm.eps,
        )


@dataclass(frozen=True)
class MlpWeights:
    """``w_down(w_gate(x_hat) * silu(w_up(x_hat)))``; the pre-norm gain is folded into gate and up."""

    w_gate: Tensor  # (hidden, D)
    w_up: Tensor  # (hidden, D)
    w_down: Tensor  # (D, hidden)
    eps: float

    @classmethod
    def from_reference(cls, mlp, norm) -> "MlpWeights":
        if mlp.w12.bias is not None or mlp.w3.bias is not None:
            raise NotImplementedError("MLP biases are not supported (released checkpoints have none)")
        gain = _f32(norm.scale)
        gate, up = _f32(mlp.w12.weight).chunk(2, dim=0)
        return cls(
            w_gate=fold_norm_gain(gate, gain),
            w_up=fold_norm_gain(up, gain),
            w_down=_f32(mlp.w3.weight),
            eps=norm.eps,
        )


@dataclass(frozen=True)
class BlockWeights:
    kind: str
    attns: tuple[AttentionWeights, ...]
    mlp: MlpWeights

    @classmethod
    def from_reference(cls, kind: str, layer) -> "BlockWeights":
        attns = tuple(AttentionWeights.from_reference(a, n) for a, n in zip(layer.attns, layer.attn_norms))
        assert tuple(a.is_time for a in attns) == LAYER_KIND_AXES[kind], f"layer kind {kind!r} does not match its attns"
        return cls(kind=kind, attns=attns, mlp=MlpWeights.from_reference(layer.mlp, layer.mlp_norm))


@dataclass(frozen=True)
class TimestralWeights:
    config: TimestralConfig
    patch_proj: ResidualBlockWeights
    forecast_query: Tensor  # (D,)
    layers: tuple[BlockWeights, ...]
    out_norm_eps: float
    head: ResidualBlockWeights  # out_norm's gain folded into its hidden and residual layers
    inv_freq: Tensor  # (head_dim / 2,) RoPE inverse frequencies, zero past rope_dim / 2

    @classmethod
    def from_reference(cls, model) -> "TimestralWeights":
        """Extract weights from an upstream ``TimestralModel``."""
        config = model.config
        transformer = model.transformer
        return cls(
            config=config,
            patch_proj=ResidualBlockWeights.from_reference(model.representation.patch_proj),
            forecast_query=_f32(model.representation.forecast_query),
            layers=tuple(
                BlockWeights.from_reference(kind, layer)
                for kind, layer in zip(config.block_schedule, transformer.layers)
            ),
            out_norm_eps=transformer.out_norm.eps,
            head=ResidualBlockWeights.from_reference(model.head.projection, input_gain=transformer.out_norm.scale),
            inv_freq=_f32(transformer.rope.inv_freq),
        )
