# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0
#
# FP32 PyTorch implementation of amazon/chronos-2's direct prediction path
# (encoder-only T5-style backbone with a multi-patch quantile head), written
# from the checkpoint and the chronos-forecasting sources. It is a small,
# dependency-light CPU oracle for the tests; tests/test_components.py checks it
# against chronos-forecasting when that package is installed.
#
# Conventions it relies on:
#   * quantile head: per output patch the channels are quantile-major
#     [num_quantiles, output_patch_size]; rearrange 'b p (q t) -> b q (p t)';
#   * RoPE: NeoX rotate_half, position ids arange(S), theta from config, applied
#     to q and k only; no 1/sqrt(d_kv) attention score scaling;
#   * additive attention mask = finfo(fp32).min on masked keys;
#   * group self-attention over independent series fused to W_o @ W_v (exact in
#     real arithmetic).

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from models.experimental.chronos2.tt.model_config import Chronos2Config
from models.experimental.chronos2.tt.preprocess import Preprocessor
from models.experimental.chronos2.tt.weights import fuse_group_attention, load_weights_fp32


# --------------------------------------------------------------------------- #
# Ops (shared with the TTNN port; per-op checked in tests/test_reference_port_ops.py)
# --------------------------------------------------------------------------- #


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """T5-style RMSNorm: FP32 mean-of-squares statistics, affine weight."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return weight * x * torch.rsqrt(variance + eps)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """NeoX-style rotate_half: [-x2, x1] over the last dim (even size)."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rope_tables(seq_len: int, d_kv: int, theta: float):
    """cos/sin tables [S, d_kv] for position ids arange(S) (FP32)."""
    inv = 1.0 / (theta ** (torch.arange(0, d_kv, 2, dtype=torch.float32) / d_kv))
    pos = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv)  # [S, d_kv/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [S, d_kv]
    return emb.cos(), emb.sin()


class Chronos2ReferenceFP32:
    """FP32 torch implementation of Chronos2Model's direct prediction path.

    Shares Chronos2Config and the FP32 host Preprocessor (instance-norm/arcsinh
    scaling, patching, time encoding) with the TTNN backend.
    """

    def __init__(self, weights_path, cfg: Chronos2Config | None = None):
        self.cfg = cfg or Chronos2Config.from_json(Path(weights_path) / "config.json")
        raw = load_weights_fp32(weights_path)
        self._init_from_raw(raw)

    @classmethod
    def from_state(cls, cfg: Chronos2Config, raw_weights: dict[str, np.ndarray]):
        """Build from already-loaded FP32 weights (no second disk read)."""
        obj = cls.__new__(cls)
        obj.cfg = cfg
        obj._init_from_raw(raw_weights)
        return obj

    def _init_from_raw(self, raw: dict[str, np.ndarray]) -> None:
        self.preprocessor = Preprocessor(self.cfg)
        self.w = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in raw.items()}
        self.fused_group = [
            torch.from_numpy(fuse_group_attention(raw, f"encoder.block.{i}.layer.1"))
            for i in range(self.cfg.num_layers)
        ]
        self._reg = self.w["shared.weight"][1]  # reg_token_id = 1 (config-verified)

    # -- submodules --------------------------------------------------------- #

    def _residual_block(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        """ResidualBlock: act(x@Wh+bh)@Wo+bo + x@Wr+br (relu, with biases)."""
        w = self.w
        h = torch.relu(x @ w[f"{prefix}.hidden_layer.weight"].T + w[f"{prefix}.hidden_layer.bias"])
        out = h @ w[f"{prefix}.output_layer.weight"].T + w[f"{prefix}.output_layer.bias"]
        return out + x @ w[f"{prefix}.residual_layer.weight"].T + w[f"{prefix}.residual_layer.bias"]

    def _block(self, i: int, x: torch.Tensor, attn_mask: torch.Tensor, cos, sin) -> torch.Tensor:
        """One encoder block: time self-attn (RoPE, no score scaling), fused
        group self-attn (independent rows), ReLU FFN; all pre-norm."""
        w, cfg = self.w, self.cfg
        B, S, D = x.shape
        H, dk = cfg.num_heads, cfg.d_kv
        p = f"encoder.block.{i}"

        # 1) time self-attention
        h = rms_norm(x, w[f"{p}.layer.0.layer_norm.weight"], cfg.layer_norm_epsilon)
        q = (h @ w[f"{p}.layer.0.self_attention.q.weight"].T).reshape(B, S, H, dk).transpose(1, 2)
        k = (h @ w[f"{p}.layer.0.self_attention.k.weight"].T).reshape(B, S, H, dk).transpose(1, 2)
        v = (h @ w[f"{p}.layer.0.self_attention.v.weight"].T).reshape(B, S, H, dk).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        scores = q @ k.transpose(-1, -2)  # reference uses scale=1.0 (no 1/sqrt(d_kv))
        add = (1.0 - attn_mask)[:, None, None, :] * torch.finfo(scores.dtype).min
        probs = torch.softmax(scores + add, dim=-1)  # FP32 softmax
        att = (probs @ v).transpose(1, 2).reshape(B, S, D)
        x = x + att @ w[f"{p}.layer.0.self_attention.o.weight"].T

        # 2) group self-attention: independent rows => exact fused linear
        h = rms_norm(x, w[f"{p}.layer.1.layer_norm.weight"], cfg.layer_norm_epsilon)
        x = x + h @ self.fused_group[i].T

        # 3) feed-forward (no biases)
        h = rms_norm(x, w[f"{p}.layer.2.layer_norm.weight"], cfg.layer_norm_epsilon)
        h = torch.relu(h @ w[f"{p}.layer.2.mlp.wi.weight"].T)
        x = x + h @ w[f"{p}.layer.2.mlp.wo.weight"].T
        return x

    # -- forward ------------------------------------------------------------- #

    def forward(self, prepared: dict) -> np.ndarray:
        """Encoder + head on prepared features -> [B, Q, n_out*patch] FP32 (scaled space)."""
        cfg = self.cfg
        ctx = torch.from_numpy(prepared["ctx_features"])
        fut = torch.from_numpy(prepared["fut_features"])
        mask = torch.from_numpy(prepared["attn_mask"])
        B, _, _ = ctx.shape
        n_out = prepared["num_output_patches"]

        reg = self._reg.reshape(1, 1, -1).expand(B, 1, cfg.d_model)
        tokens = torch.cat(
            (
                self._residual_block(ctx, "input_patch_embedding"),
                reg,
                self._residual_block(fut, "input_patch_embedding"),
            ),
            dim=1,
        )
        cos, sin = rope_tables(tokens.shape[1], cfg.d_kv, cfg.rope_theta)
        for i in range(cfg.num_layers):
            tokens = self._block(i, tokens, mask, cos, sin)
        tokens = rms_norm(tokens, self.w["encoder.final_layer_norm.weight"], cfg.layer_norm_epsilon)

        head = self._residual_block(tokens[:, -n_out:, :], "output_patch_embedding")
        # quantile-major head channels: 'b p (q t) -> b q (p t)'
        pred = head.reshape(B, n_out, cfg.num_quantiles, cfg.output_patch_size)
        return pred.permute(0, 2, 1, 3).reshape(B, cfg.num_quantiles, -1).contiguous().numpy()

    def forward_np(self, prepared: dict) -> np.ndarray:
        """forward() under torch.inference_mode() (no-grad, deterministic)."""
        with torch.inference_mode():
            return self.forward(prepared)

    # -- Backend.forecast-shaped prediction ------------------------------------ #

    def num_output_patches(self, prediction_length: int) -> int:
        n = math.ceil(prediction_length / self.cfg.output_patch_size)
        return min(n, self.cfg.max_output_patches)

    def predict(self, past_values, past_observed_mask, prediction_length) -> dict:
        """Mirror of Backend.forecast semantics: [B,T] fp32 + mask + h -> {'quantiles': [B,h,Q]}."""
        values = np.array(past_values, dtype=np.float32, copy=True)
        mask = np.array(past_observed_mask, dtype=np.float32, copy=True)
        h = int(prediction_length)
        if not (1 <= h <= 64):
            raise ValueError(f"prediction_length must be in 1..64, got {h}")
        prepared = self.preprocessor.prepare(values, mask, self.num_output_patches(h))
        with torch.inference_mode():
            preds_scaled = self.forward(prepared)  # [B, Q, n_out*patch]
        preds = preds_scaled[:, :, :h]
        # per-row loc/scale [B,1] -> [B,1,1] to broadcast over [B, Q, h]
        preds = self.preprocessor.unscale(preds, prepared["loc"][:, :, None], prepared["scale"][:, :, None])
        return {"quantiles": np.ascontiguousarray(preds.transpose(0, 2, 1)).astype(np.float32)}
