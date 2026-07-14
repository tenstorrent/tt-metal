# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Authoritative torch reference for the DFlash drafter PREFILL delta, from the tt-blaze #1674
# spec (dflash_pseudocode.py, Kimi-K2.5-DFlash). This is the sign-off reference the device
# module (tt/dflash/tt_dflash_drafter_kv.py) is PCC'd against — kept faithful to the spec; do
# not "improve" it. RoPE parity comes from injecting the real model's rotary_emb via
# build_from_hf (YARN-correct); DefaultRotary is only a no-YARN fallback.

"""
DFlash drafter — PREFILL forward pass (draft KV-cache build).

Faithful reference port of the *context* path in z-lab/dflash `dflash/model.py`
(`Qwen3DFlashAttention.forward` / `DFlashDraftModel.forward`), isolated to exactly
what prefill must do: populate the drafter's per-layer KV cache from the target's
fused hidden states — WITHOUT running the proposal (mask-token) pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class DFlashPrefillConfig:
    hidden_size: int = 7168
    num_hidden_layers: int = 6
    num_attention_heads: int = 64  # only used to sanity-check; q is not built here
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0  # standard fallback; YARN handled via injected rotary_emb
    target_num_layers: int = 61
    target_layer_ids: list[int] = field(default_factory=lambda: [1, 12, 24, 35, 47, 58])

    @property
    def n_ctx_inputs(self) -> int:
        return len(self.target_layer_ids)


# ---------------------------------------------------------------------------
# Primitives (match transformers.models.qwen3 semantics)
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Equivalent to Qwen3RMSNorm."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dt)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class DefaultRotary(nn.Module):
    """Standard Qwen3/Llama RoPE (no YARN). Inject the real rotary for exact parity."""

    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor):
        # returns (cos, sin) each [B, S, head_dim]
        inv = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        pos = position_ids[:, None, :].float()
        freqs = (inv @ pos).transpose(1, 2)  # [B, S, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [B, S, head_dim]
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def apply_rope_k(k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # k: [B, n_kv, S, head_dim]; cos/sin: [B, S, head_dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (k * cos) + (rotate_half(k) * sin)


# ---------------------------------------------------------------------------
# Per-layer weights actually used by prefill
# ---------------------------------------------------------------------------
class DraftLayerKV(nn.Module):
    """Just the K/V path of one Qwen3DFlashAttention (the part prefill needs)."""

    def __init__(self, cfg: DFlashPrefillConfig, attention_bias: bool = False):
        super().__init__()
        kv_out = cfg.num_key_value_heads * cfg.head_dim
        self.k_proj = nn.Linear(cfg.hidden_size, kv_out, bias=attention_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, kv_out, bias=attention_bias)
        self.k_norm = RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)
        self.n_kv = cfg.num_key_value_heads
        self.hd = cfg.head_dim

    def forward(self, target_hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        # target_hidden: [B, S, H] (already fc + hidden_norm)
        b, s, _ = target_hidden.shape
        k = self.k_proj(target_hidden).view(b, s, self.n_kv, self.hd)
        v = self.v_proj(target_hidden).view(b, s, self.n_kv, self.hd)
        k = self.k_norm(k).transpose(1, 2)  # [B, n_kv, S, hd]
        v = v.transpose(1, 2)  # [B, n_kv, S, hd]
        k = apply_rope_k(k, cos, sin)  # RoPE on context keys
        return k, v


# ---------------------------------------------------------------------------
# Drafter prefill module
# ---------------------------------------------------------------------------
class DFlashDrafterPrefill(nn.Module):
    """
    Owns only what prefill uses: fc + hidden_norm (model level) and, per draft layer,
    {k_proj, v_proj, k_norm}. Produces the persistent draft KV cache for the prompt.
    """

    def __init__(self, cfg: DFlashPrefillConfig, rotary_emb: Optional[Callable] = None, attention_bias: bool = False):
        super().__init__()
        self.cfg = cfg
        self.fc = nn.Linear(cfg.n_ctx_inputs * cfg.hidden_size, cfg.hidden_size, bias=False)
        self.hidden_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.layers = nn.ModuleList([DraftLayerKV(cfg, attention_bias) for _ in range(cfg.num_hidden_layers)])
        self.rotary_emb = rotary_emb or DefaultRotary(cfg.head_dim, cfg.rope_theta)

    # --- fuse the tapped target layers (mirrors extract_context_feature, offset=1) ---
    def fuse_target_hidden(self, target_hidden_states: list[torch.Tensor]) -> torch.Tensor:
        """
        target_hidden_states: tuple/list of per-layer hidden states from the TARGET prefill
            (length target_num_layers + 1; index 0 = embeddings, index i+1 = output of layer i),
            exactly as HF returns with output_hidden_states=True.
        Returns concatenated context feature [B, S, n_ctx_inputs * H].
        """
        offset = 1  # same as dflash/model.py extract_context_feature
        selected = [target_hidden_states[lid + offset] for lid in self.cfg.target_layer_ids]
        return torch.cat(selected, dim=-1)

    @torch.inference_mode()
    def prefill(
        self,
        target_hidden_states: list[torch.Tensor] | torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Build the drafter's per-layer KV cache over the whole prompt.

        Args:
            target_hidden_states: either
                (a) list/tuple of target per-layer hidden states (HF output_hidden_states), or
                (b) a pre-fused tensor [B, S, n_ctx_inputs * H].
            position_ids: [B, S] absolute positions of the prompt tokens (default arange).

        Returns:
            draft_kv: list of length num_hidden_layers, each (k, v) with
                      k,v shape [B, num_key_value_heads, S, head_dim].
        """
        if isinstance(target_hidden_states, (list, tuple)):
            ctx = self.fuse_target_hidden(target_hidden_states)  # [B, S, n_ctx*H]
        else:
            ctx = target_hidden_states
        b, s, _ = ctx.shape

        if position_ids is None:
            position_ids = torch.arange(s, device=ctx.device).unsqueeze(0).expand(b, -1)

        # model-level fusion: computed ONCE, shared by all layers
        target_hidden = self.hidden_norm(self.fc(ctx))  # [B, S, H]

        cos, sin = self.rotary_emb(target_hidden, position_ids)  # [B, S, head_dim]

        draft_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.layers:
            k, v = layer(target_hidden, cos, sin)  # [B, n_kv, S, hd]
            draft_kv.append((k, v))
        return draft_kv

    # --- optional: load real weights from a HF DFlashDraftModel ---
    @classmethod
    def build_from_hf(cls, hf_model) -> "DFlashDrafterPrefill":
        """
        hf_model: a loaded z-lab DFlashDraftModel (transformers). Copies fc, hidden_norm,
        per-layer k_proj/v_proj/k_norm, and reuses the model's own rotary_emb (YARN-correct).
        """
        c = hf_model.config
        cfg = DFlashPrefillConfig(
            hidden_size=c.hidden_size,
            num_hidden_layers=c.num_hidden_layers,
            num_attention_heads=c.num_attention_heads,
            num_key_value_heads=c.num_key_value_heads,
            head_dim=getattr(c, "head_dim", c.hidden_size // c.num_attention_heads),
            rms_norm_eps=c.rms_norm_eps,
            target_layer_ids=list(hf_model.target_layer_ids),
        )
        attn_bias = bool(getattr(c, "attention_bias", False))
        self = cls(cfg, rotary_emb=hf_model.rotary_emb, attention_bias=attn_bias)
        self.fc.load_state_dict(hf_model.fc.state_dict())
        self.hidden_norm.load_state_dict(hf_model.hidden_norm.state_dict())
        for dst, src in zip(self.layers, hf_model.layers):
            dst.k_proj.load_state_dict(src.self_attn.k_proj.state_dict())
            dst.v_proj.load_state_dict(src.self_attn.v_proj.state_dict())
            dst.k_norm.load_state_dict(src.self_attn.k_norm.state_dict())
        return self.eval()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = DFlashPrefillConfig()  # Kimi-K2.5-DFlash dims
    drafter = DFlashDrafterPrefill(cfg).eval()

    B, S = 1, 37  # one user, 37-token prompt
    # Fake target prefill output: tuple of (target_num_layers + 1) hidden states.
    target_hidden_states = [torch.randn(B, S, cfg.hidden_size) for _ in range(cfg.target_num_layers + 1)]

    draft_kv = drafter.prefill(target_hidden_states)

    print(f"draft layers: {len(draft_kv)} (expected {cfg.num_hidden_layers})")
    k0, v0 = draft_kv[0]
    print(f"per-layer K shape: {tuple(k0.shape)}  V shape: {tuple(v0.shape)}")
    exp = (B, cfg.num_key_value_heads, S, cfg.head_dim)
    assert k0.shape == v0.shape == torch.Size(exp), f"got {tuple(k0.shape)}, expected {exp}"

    # total cached elements (per user): num_layers * 2 * n_kv * S * head_dim
    elems = cfg.num_hidden_layers * 2 * cfg.num_key_value_heads * S * cfg.head_dim
    print(
        f"draft KV cache holds {elems:,} values for S={S} "
        f"(grows linearly with prompt length, windowed to ~4096 at decode)"
    )
    print("OK — drafter prefill produced a per-position, per-layer KV cache with NO proposal/mask pass.")
