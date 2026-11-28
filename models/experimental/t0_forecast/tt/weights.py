# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Host-side t0 weights (fp32 torch, no ttnn import), read straight from ``model.safetensors``.

reference : t0/torch/model/model.py (``T0Forecaster``) and t0/torch/model/layers/*
    patch_encoder.projection   ResidualBlock(96 -> d -> d), ReLU, plus type_embeddings (3 x d)
    transformer.layers[i]      [time, time, group] x 8; pre-norm attention + SwiGLU MLP, biases everywhere
    transformer.out_norm       RMSNorm(d), eps 1e-8
    decoder                    ResidualBlock(d -> d -> 32 * 21), ReLU
    head                       cumulative softplus over the 21 levels (no parameters)

``nn.Linear`` weights stay ``(out, in)``. wQKV rows are ``(three, heads, head_dim)``; q and k keep t0's
native interleaved RoPE pairs, which ``rotary_embedding_llama`` rotates, so no row permutation is needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import Tensor

TIME = "time"
GROUP = "group"


@dataclass(frozen=True)
class T0Config:
    embed_dim: int
    num_layers: int
    num_heads: int
    mlp_hidden_dim: int
    patch_size: int
    group_every_n: int
    quantile_levels: tuple[float, ...]
    scaler_use_arcsinh: bool
    scaler_eps: float
    scaler_eps_mode: str
    xpos_scale_layout: str
    scaler_zero_no_value: bool
    norm_eps: float = 1e-8
    rope_theta: float = 10000.0
    xpos_scale_base: float = 512.0

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    @property
    def n_quantiles(self) -> int:
        return len(self.quantile_levels)

    @property
    def layer_kinds(self) -> tuple[str, ...]:
        if self.group_every_n <= 0:
            return (TIME,) * self.num_layers
        block = (TIME,) * (self.group_every_n - 1) + (GROUP,)
        return block * (self.num_layers // self.group_every_n)

    @classmethod
    def from_json(cls, path: Path) -> "T0Config":
        raw = json.loads(Path(path).read_text())
        defaults = dict(scaler_eps=0.1, scaler_eps_mode="variance_offset", xpos_scale_layout="interleaved")
        defaults.update(scaler_zero_no_value=True, scaler_use_arcsinh=True)
        get = lambda k: raw.get(k, defaults.get(k))
        return cls(
            embed_dim=raw["embed_dim"],
            num_layers=raw["num_layers"],
            num_heads=raw["num_heads"],
            mlp_hidden_dim=raw["mlp_hidden_dim"],
            patch_size=raw["patch_size"],
            group_every_n=raw["group_every_n"],
            quantile_levels=tuple(float(q) for q in raw["quantile_levels"]),
            scaler_use_arcsinh=bool(get("scaler_use_arcsinh")),
            scaler_eps=float(get("scaler_eps")),
            scaler_eps_mode=str(get("scaler_eps_mode")),
            xpos_scale_layout=str(get("xpos_scale_layout")),
            scaler_zero_no_value=bool(get("scaler_zero_no_value")),
        )


@dataclass(frozen=True)
class ResidualBlockWeights:
    """``output(act(hidden(x))) + residual(x)``, all with bias."""

    hidden_w: Tensor
    hidden_b: Tensor
    output_w: Tensor
    output_b: Tensor
    residual_w: Tensor
    residual_b: Tensor


@dataclass(frozen=True)
class AttentionWeights:
    """Pre-norm attention sublayer: ``x + wO(attn(RMSNorm(x) * norm))``."""

    kind: str  # TIME or GROUP
    norm: Tensor  # (d,)
    wqkv: Tensor  # (3d, d), rows (three, heads, head_dim)
    bqkv: Tensor  # (3d,)
    q_norm: Tensor  # (head_dim,)
    k_norm: Tensor
    wo: Tensor  # (d, d)
    bo: Tensor
    num_heads: int
    head_dim: int

    def part(self, which: int) -> tuple[Tensor, Tensor]:
        """``(weight, bias)`` of q (0), k (1) or v (2)."""
        d = self.num_heads * self.head_dim
        return self.wqkv[which * d : (which + 1) * d], self.bqkv[which * d : (which + 1) * d]


@dataclass(frozen=True)
class MlpWeights:
    """Pre-norm SwiGLU: ``x + w2(silu(gate) * up)``, ``gate, up = (RMSNorm(x) * norm) @ w1.T + b1`` (gate first)."""

    norm: Tensor  # (d,)
    w1: Tensor  # (2h, d): rows [gate | up]
    b1: Tensor
    w2: Tensor  # (d, h)
    b2: Tensor

    @property
    def hidden(self) -> int:
        return self.w2.shape[1]


@dataclass(frozen=True)
class LayerWeights:
    attention: AttentionWeights
    mlp: MlpWeights

    @property
    def kind(self) -> str:
        return self.attention.kind


@dataclass(frozen=True)
class T0Weights:
    config: T0Config
    patch_encoder: ResidualBlockWeights
    type_embeddings: Tensor  # (3, d)
    layers: tuple[LayerWeights, ...]
    out_norm: Tensor
    decoder: ResidualBlockWeights

    @classmethod
    def from_state_dict(cls, config: T0Config, sd: dict[str, Tensor]) -> "T0Weights":
        f = lambda k: sd[k].detach().to(torch.float32).contiguous()

        def block(prefix: str) -> ResidualBlockWeights:
            return ResidualBlockWeights(
                hidden_w=f(f"{prefix}.mlp.hidden_layer.weight"),
                hidden_b=f(f"{prefix}.mlp.hidden_layer.bias"),
                output_w=f(f"{prefix}.mlp.output_layer.weight"),
                output_b=f(f"{prefix}.mlp.output_layer.bias"),
                residual_w=f(f"{prefix}.residual_layer.weight"),
                residual_b=f(f"{prefix}.residual_layer.bias"),
            )

        layers = []
        for i, kind in enumerate(config.layer_kinds):
            p = f"transformer.layers.{i}"
            a = f"{p}.attention_block.attention"
            attention = AttentionWeights(
                kind=kind,
                norm=f(f"{p}.attention_block.norm.scale"),
                wqkv=f(f"{a}.wQKV.weight"),
                bqkv=f(f"{a}.wQKV.bias"),
                q_norm=f(f"{a}.q_norm.scale"),
                k_norm=f(f"{a}.k_norm.scale"),
                wo=f(f"{a}.wO.weight"),
                bo=f(f"{a}.wO.bias"),
                num_heads=config.num_heads,
                head_dim=config.head_dim,
            )
            mlp = MlpWeights(
                norm=f(f"{p}.norm.scale"),
                w1=f(f"{p}.mlp.0.weight"),
                b1=f(f"{p}.mlp.0.bias"),
                w2=f(f"{p}.mlp.2.weight"),
                b2=f(f"{p}.mlp.2.bias"),
            )
            layers.append(LayerWeights(attention, mlp))
        return cls(
            config=config,
            patch_encoder=block("patch_encoder.projection"),
            type_embeddings=f("patch_encoder.type_embeddings.weight"),
            layers=tuple(layers),
            out_norm=f("transformer.out_norm.scale"),
            decoder=block("decoder"),
        )

    @classmethod
    def load(cls, path: Path | None = None) -> "T0Weights":
        """Load the pinned checkpoint directory (default: the HF cache snapshot)."""
        from safetensors.torch import load_file

        from models.experimental.t0_forecast.common import checkpoint_dir

        path = Path(path) if path is not None else checkpoint_dir()
        return cls.from_state_dict(T0Config.from_json(path / "config.json"), load_file(str(path / "model.safetensors")))

    def reduced(self, layer_indices) -> "T0Weights":
        """A copy keeping only ``layer_indices`` (for the reduced-layer profile)."""
        return replace(self, layers=tuple(self.layers[i] for i in layer_indices))
