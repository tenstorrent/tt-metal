# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Torch golden for the DFlash drafter *context-KV* path (issue #49586, Phase 1).

Thin wrapper over the authoritative reference port `dflash_prefill_reference.DFlashDrafterPrefill`
(from the tt-blaze #1674 spec). Feeds a seeded-random target context feature and emits the reference
K/V (+ fc/hidden_norm intermediates) the device test PCCs against.

**RoPE source matters** — this is why the golden defers to the reference module:
  * If a loadable HF DFlashDraftModel dir is provided (``hf_model_path`` / env ``DFLASH_DRAFTER_HF``),
    the golden reuses the model's OWN rotary_emb + weights via ``build_from_hf`` — the true reference.
    This is the sign-off path: it will EXPOSE any mismatch between the device's deepseek-yarn rope and
    the trained model's actual rope (K2.5 uses rope_type "yarn"; K2.6 "deepseek_yarn").
  * Otherwise it loads weights from the safetensors and injects a deepseek-yarn rotary that mirrors the
    device's ``rope.get_cos_sin_matrix(interleave=False)``. Self-consistent with the device, but does
    NOT independently verify the trained model's rope — prefer the HF path before declaring victory.

Emit a reference .pt:
    python -m models.demos.deepseek_v3_d_p.tests.dflash.torch_dflash_golden \
        --weights /path/Kimi-K2.6-DFlash/model.safetensors --seq-len 512 --out /tmp/dflash_golden.pt
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn

from models.demos.deepseek_v3_d_p.tests.dflash.dflash_prefill_reference import DFlashDrafterPrefill, DFlashPrefillConfig

# Canonical drafter config (pure-Python) is shared with the device module so the two never drift.
from models.demos.deepseek_v3_d_p.tt.dflash.dflash_drafter_config import DFlashDrafterConfig

# --------------------------------------------------------------------------------------
# deepseek_yarn RoPE fallback — mirrors the device's rope.get_cos_sin_matrix (which builds
# DeepseekV3YarnRotaryEmbedding tables). Copied from reference/kimi_k2_6/modeling_deepseek.py;
# kept numerically identical. Used ONLY when no real HF model is available.
# --------------------------------------------------------------------------------------


def _yarn_find_correction_dim(num_rotations, dim, base, max_position_embeddings):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings):
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def _yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _yarn_linear_ramp_mask(mn, mx, dim):
    if mn == mx:
        mx += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - mn) / (mx - mn)
    return torch.clamp(linear_func, 0, 1)


def build_yarn_cos_sin(cfg: DFlashDrafterConfig, seq_len: int, dtype=torch.float32):
    """(cos, sin) each [seq_len, head_dim] for the drafter's deepseek_yarn rope (full-head)."""
    dim, base, scaling_factor = cfg.head_dim, cfg.rope_theta, cfg.rope_factor
    freq_extra = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freq_inter = 1.0 / (scaling_factor * base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    low, high = _yarn_find_correction_range(cfg.rope_beta_fast, cfg.rope_beta_slow, dim, base, cfg.rope_orig_max_pos)
    inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, dim // 2).to(torch.float32)
    inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    mscale = float(
        _yarn_get_mscale(scaling_factor, cfg.rope_mscale) / _yarn_get_mscale(scaling_factor, cfg.rope_mscale_all_dim)
    )
    emb = torch.cat((freqs, freqs), dim=-1)
    return (emb.cos() * mscale).to(dtype), (emb.sin() * mscale).to(dtype)


class _DeepseekYarnRotary(nn.Module):
    """rotary_emb(x, position_ids) -> (cos, sin) each [B, S, head_dim], deepseek-yarn (device-matching)."""

    def __init__(self, cfg: DFlashDrafterConfig):
        super().__init__()
        self.cfg = cfg

    @torch.no_grad()
    def forward(self, x, position_ids):
        seq = int(position_ids.max().item()) + 1
        cos, sin = build_yarn_cos_sin(self.cfg, seq, x.dtype)  # [seq, head_dim]
        return cos[position_ids].to(x.dtype), sin[position_ids].to(x.dtype)


# --------------------------------------------------------------------------------------
# Build / load the reference drafter
# --------------------------------------------------------------------------------------


def _prefill_config(dcfg: DFlashDrafterConfig) -> DFlashPrefillConfig:
    return DFlashPrefillConfig(
        hidden_size=dcfg.hidden_size,
        num_hidden_layers=dcfg.num_hidden_layers,
        num_attention_heads=dcfg.num_attention_heads,
        num_key_value_heads=dcfg.num_key_value_heads,
        head_dim=dcfg.head_dim,
        rms_norm_eps=dcfg.rms_norm_eps,
        target_num_layers=dcfg.num_target_layers,
        target_layer_ids=list(dcfg.target_layer_ids),
    )


def _load_weights_into(drafter: DFlashDrafterPrefill, safetensors_path: str) -> None:
    from safetensors.torch import load_file

    sd = load_file(safetensors_path)
    drafter.fc.weight.data.copy_(sd["fc.weight"])
    drafter.hidden_norm.weight.data.copy_(sd["hidden_norm.weight"])
    for i, layer in enumerate(drafter.layers):
        layer.k_proj.weight.data.copy_(sd[f"layers.{i}.self_attn.k_proj.weight"])
        layer.v_proj.weight.data.copy_(sd[f"layers.{i}.self_attn.v_proj.weight"])
        layer.k_norm.weight.data.copy_(sd[f"layers.{i}.self_attn.k_norm.weight"])


def build_drafter(dcfg: DFlashDrafterConfig, weights: str | None = None, hf_model=None) -> DFlashDrafterPrefill:
    """hf_model (a loaded z-lab DFlashDraftModel) → real rotary + weights (sign-off). Else safetensors
    weights + deepseek-yarn rotary matching the device."""
    if hf_model is not None:
        return DFlashDrafterPrefill.build_from_hf(hf_model)
    drafter = DFlashDrafterPrefill(_prefill_config(dcfg), rotary_emb=_DeepseekYarnRotary(dcfg)).eval()
    if weights is not None:
        _load_weights_into(drafter, weights)
    return drafter


def generate_reference(
    weights: str | None,
    seq_len: int,
    seed: int = 0,
    dcfg: DFlashDrafterConfig | None = None,
    hf_model=None,
):
    """Seeded-random context feature → reference dict. The device test feeds the SAME context_feature
    and PCCs each stage (reduced, target_hidden, per-layer k/v)."""
    dcfg = dcfg or DFlashDrafterConfig()
    gen = torch.Generator().manual_seed(seed)
    drafter = build_drafter(dcfg, weights=weights, hf_model=hf_model)
    dt = drafter.fc.weight.dtype

    ctx = torch.randn(1, seq_len, dcfg.target_feature_size, generator=gen, dtype=torch.float32).to(dt)
    with torch.inference_mode():
        reduced = drafter.fc(ctx)
        target_hidden = drafter.hidden_norm(reduced)
        kv = drafter.prefill(ctx)  # list[(k, v)] each [1, n_kv, S, head_dim]
    return {
        "config": dcfg.__dict__,
        "seq_len": seq_len,
        "seed": seed,
        "positions": torch.arange(seq_len),
        "context_feature": ctx,  # [1, S, 43008] — input to fc
        "reduced": reduced,  # [1, S, 7168] — fc output
        "target_hidden": target_hidden,  # [1, S, 7168] — hidden_norm output (drafter KV input)
        "k": torch.stack([k for k, _ in kv]),  # [num_layers, 1, n_kv, S, head_dim]
        "v": torch.stack([v for _, v in kv]),
    }


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to Kimi-K2.6-DFlash/model.safetensors")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="Output .pt path")
    args = ap.parse_args()
    ref = generate_reference(args.weights, args.seq_len, args.seed)
    torch.save(ref, args.out)
    print(
        f"wrote {args.out}: k={tuple(ref['k'].shape)} v={tuple(ref['v'].shape)} target_hidden={tuple(ref['target_hidden'].shape)}"
    )


if __name__ == "__main__":
    _main()
