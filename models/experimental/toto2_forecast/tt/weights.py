# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Toto 2.0 checkpoint -> folded fp32 weights (pure torch; reads safetensors, never imports the reference).

The u-muP plumbing of ``toto2/toto2/model.py`` @ 9213d2c is all constants, folded here once:

    uu.Linear          y = s * (x W^T + b),  s = fan_in^-1/2 (readout: fan_in^-1)   ->  W' = s W, b' = s b
    scale_bwd          forward identity (the ``1.0 * W`` copies)                     ->  dropped
    U.silu             c * silu(z), c = 1.76678 (``SILU_GAIN``)                       ->  into the next linear's W
    ResidualMLP        (h + skip) / sqrt(2)  (tau = 1)                                ->  into W2, b2, Ws, bs
    PerDimScale        q *= softplus(p) / 0.52103 / ln 2 * 0.52103, before RoPE        ->  into the q rows of in_proj
    tau residual       x <- (tau f(norm(x)) + x) / sqrt(1 + tau^2) = alpha f + beta x ->  alpha into out_proj / fc2
    Dropout(p=0)       identity                                                       ->  dropped

so a time layer is the plain pre-norm decoder

    xn = x / sqrt(mean(x^2) + eps);  q, k, v = heads(xn W_qkv'^T + b_qkv')
    q, k = RoPE + xPos (dims 0-31);  a = SDPA(q, k, v, scale 1/64, causal);  x = beta_a x + a W_o'^T + b_o'
    xn = RMSNorm(x);  [g, u] = xn W_fc1'^T;  x = beta_m x + (g * silu(u)) W_fc2'^T

and the patch embedding and head are ``silu(x W1'^T + b1') W2'^T + x Ws'^T + b'`` with ``b' = b2' + bs'``.
Every fold is computed in float64 and stored in float32.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import Tensor

# unit_scaling.functional.silu at mult = 1: logarithmic_interpolation(0.8, 2, sqrt(2 / (1 - 1/pi))).
SILU_GAIN = math.exp(0.8 * math.log((2 / (1 - 1 / math.pi)) ** 0.5) + 0.2 * math.log(2.0))
# dd_unit_scaling softplus output scale (1 / 0.52103) and per_dim_scale output scale (0.52103).
_SOFTPLUS_SCALE = 1.0 / 0.52103
_PDS_SCALE = 0.52103


@dataclass(frozen=True)
class Toto2Config:
    patch_size: int
    d_model: int
    num_heads: int
    num_layers: int
    d_ff: int
    head_dim: int
    norm_eps: float
    variate_layers: tuple[int, ...]
    num_quantiles: int = 9
    rope_dims: int = 32  # partial_factor (0.0, 0.5) of the 64-dim head
    rope_base: float = 10000.0
    xpos_scale_base: float = 256.0

    @classmethod
    def from_json(cls, raw: dict) -> "Toto2Config":
        assert raw["attn_bias"] and not raw["mlp_bias"] and not raw["qk_norm"] and not raw["norm_include_weight"]
        assert raw["per_dim_scale"] and raw["use_xpos"] and raw["num_output_patches"] == 1
        assert raw.get("heads_per_group", 1) == 1 and raw["qk_dim"] == raw["v_dim"] == 64
        n, group, per_group = raw["num_layers"], raw["layer_group_size"], raw["num_variate_layers_per_group"]
        if raw["variate_layer_first"]:
            variate = tuple(i for i in range(n) if i % group < per_group)
        else:
            variate = tuple(i for i in range(n) if i % group >= group - per_group)
        return cls(
            patch_size=raw["patch_size"],
            d_model=raw["d_model"],
            num_heads=raw["num_heads"],
            num_layers=n,
            d_ff=raw["d_ff"],
            head_dim=raw["qk_dim"],
            norm_eps=raw["norm_eps"],
            variate_layers=variate,
        )


@dataclass(frozen=True)
class ResidualMLPWeights:
    """``silu(x W1^T + b1) W2^T + x Ws^T + b`` (patch embedding 64 -> 4d -> d; head d -> 4d -> 288)."""

    w1: Tensor  # (hidden, in)
    b1: Tensor  # (hidden,)
    w2: Tensor  # (out, hidden)
    ws: Tensor  # (out, in)
    b: Tensor  # (out,)


@dataclass(frozen=True)
class LayerWeights:
    index: int
    is_variate: bool
    w_qkv: Tensor  # (3d, d): rows (q | k | v, head, head_dim); the q rows carry the PerDimScale
    b_qkv: Tensor  # (3d,)
    w_o: Tensor  # (d, d), alpha_attn folded
    b_o: Tensor  # (d,)
    beta_attn: float
    w_fc1: Tensor  # (2 d_ff, d): rows [g; u], out = g * silu(u)
    w_fc2: Tensor  # (d, d_ff), alpha_mlp folded
    beta_mlp: float
    attn_tau: float
    mlp_tau: float
    eps_attn: float | None = None  # set by fold_residual_betas (eps / c^2); None = config.norm_eps
    eps_mlp: float | None = None


@dataclass(frozen=True)
class Toto2Weights:
    config: Toto2Config
    patch_proj: ResidualMLPWeights
    layers: tuple[LayerWeights, ...]
    head: ResidualMLPWeights
    name: str = ""
    head_eps: float | None = None  # set by fold_residual_betas

    @classmethod
    def from_checkpoint(cls, name: str) -> "Toto2Weights":
        from models.experimental.toto2_forecast.tt.checkpoints import snapshot_dir

        return cls.from_dir(snapshot_dir(name), name=name)

    @classmethod
    def from_dir(cls, path: str | Path, name: str = "") -> "Toto2Weights":
        from safetensors import safe_open

        path = Path(path)
        config = Toto2Config.from_json(json.loads((path / "config.json").read_text()))
        with safe_open(str(path / "model.safetensors"), "pt") as f:

            def get(key):
                return f.get_tensor(key).to(torch.float64)

            patch_proj = _residual_mlp(get, "patch_proj", readout=False)
            head = _residual_mlp(get, "output_head.param_projection.proj", readout=True)
            layers = tuple(_layer(get, config, i) for i in range(config.num_layers))
        return cls(config=config, patch_proj=patch_proj, layers=layers, head=head, name=name)

    def reduced(self, indices) -> "Toto2Weights":
        """Same model with only ``layers[indices]`` (profile runs); the config keeps its layer kinds."""
        return replace(self, layers=tuple(self.layers[i] for i in indices))

    def fold_residual_betas(self) -> "Toto2Weights":
        """R2-1: carry ``y = x / c`` with ``c`` the running product of the residual betas, so every residual is a
        plain add. The norms are weightless, so ``RMSNorm(c y; eps) = RMSNorm(y; eps / c^2)``; ``1 / c`` after each
        sublayer folds into ``W_o``, ``b_o`` and ``W_fc2`` (float64). Exact in real arithmetic. Apply after
        ``reduced``."""
        assert all(lw.eps_attn is None for lw in self.layers), "already folded"
        eps, c, layers = self.config.norm_eps, 1.0, []
        for lw in self.layers:
            eps_attn = eps / c**2
            c *= lw.beta_attn
            w_o, b_o = _f32(lw.w_o.double() / c), _f32(lw.b_o.double() / c)
            eps_mlp = eps / c**2
            c *= lw.beta_mlp
            layers.append(
                replace(
                    lw,
                    w_o=w_o,
                    b_o=b_o,
                    w_fc2=_f32(lw.w_fc2.double() / c),
                    beta_attn=1.0,
                    beta_mlp=1.0,
                    eps_attn=eps_attn,
                    eps_mlp=eps_mlp,
                )
            )
        return replace(self, layers=tuple(layers), head_eps=eps / c**2)


def _f32(t: Tensor) -> Tensor:
    return t.to(torch.float32).contiguous()


def _scale(weight: Tensor, readout: bool) -> float:
    fan_in = weight.shape[1]
    return 1.0 / fan_in if readout else fan_in**-0.5


def _residual_mlp(get, prefix: str, *, readout: bool) -> ResidualMLPWeights:
    """``InputResidualMLP`` (linear1 and skip hidden-scaled, linear2 hidden-scaled) or ``OutputResidualMLP``
    (linear1 hidden-scaled, linear2 and skip readouts); both with tau = 1."""
    combine = 2.0**-0.5
    w1, b1 = get(f"{prefix}.linear1.weight"), get(f"{prefix}.linear1.bias")
    w2, b2 = get(f"{prefix}.linear2.weight"), get(f"{prefix}.linear2.bias")
    ws, bs = get(f"{prefix}.skip_proj.weight"), get(f"{prefix}.skip_proj.bias")
    s1, s2, ss = _scale(w1, False), _scale(w2, readout), _scale(ws, readout)
    return ResidualMLPWeights(
        w1=_f32(w1 * s1),
        b1=_f32(b1 * s1),
        w2=_f32(w2 * (s2 * SILU_GAIN * combine)),
        ws=_f32(ws * (ss * combine)),
        b=_f32(b2 * (s2 * combine) + bs * (ss * combine)),
    )


def per_dim_scale(p: Tensor) -> Tensor:
    """``PerDimScale`` gain per head dim, float64."""
    return torch.nn.functional.softplus(p.to(torch.float64)) * _SOFTPLUS_SCALE / math.log(2.0) * _PDS_SCALE


def _layer(get, config: Toto2Config, i: int) -> LayerWeights:
    pre = f"transformer.layers.{i}"
    d, h, dh = config.d_model, config.num_heads, config.head_dim
    w_in, b_in = get(f"{pre}.attn.in_proj.weight"), get(f"{pre}.attn.in_proj.bias")
    row_scale = torch.full((w_in.shape[0],), _scale(w_in, False), dtype=torch.float64)
    row_scale[: h * dh] *= per_dim_scale(get(f"{pre}.attn._pds.per_dim_scale")).repeat(h)
    attn_tau, mlp_tau = float(get(f"{pre}.attn_tau")), float(get(f"{pre}.mlp_tau"))
    alpha_a, beta_a = attn_tau / math.sqrt(1 + attn_tau**2), 1 / math.sqrt(1 + attn_tau**2)
    alpha_m, beta_m = mlp_tau / math.sqrt(1 + mlp_tau**2), 1 / math.sqrt(1 + mlp_tau**2)
    w_o, b_o = get(f"{pre}.attn.out_proj.weight"), get(f"{pre}.attn.out_proj.bias")
    w_fc1, w_fc2 = get(f"{pre}.ffn.fc1.weight"), get(f"{pre}.ffn.fc2.weight")
    assert w_in.shape == (3 * d, d) and w_fc1.shape == (2 * config.d_ff, d)
    return LayerWeights(
        index=i,
        is_variate=i in config.variate_layers,
        w_qkv=_f32(w_in * row_scale[:, None]),
        b_qkv=_f32(b_in * row_scale),
        w_o=_f32(w_o * (_scale(w_o, False) * alpha_a)),
        b_o=_f32(b_o * (_scale(w_o, False) * alpha_a)),
        beta_attn=beta_a,
        w_fc1=_f32(w_fc1 * _scale(w_fc1, False)),
        w_fc2=_f32(w_fc2 * (_scale(w_fc2, False) * alpha_m)),
        beta_mlp=beta_m,
        attn_tau=attn_tau,
        mlp_tau=mlp_tau,
    )
