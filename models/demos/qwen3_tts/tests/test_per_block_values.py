# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Per-block value-level comparison: TTNN vs CPU fp32 reference for one Talker
decoder layer.

Goes beyond PCC — for each sub-block (input layernorm → attention → residual →
post-attention layernorm → MLP → residual) we print:
    * PCC, cosine similarity
    * max |Δ|, mean |Δ|, RMS_diff/RMS_ref ratio
    * top-K largest absolute disagreements with their (idx, ref, tt, |Δ|) tuples
    * histogram of |Δ| in log-spaced bins

Useful when codec stats look healthy and PCC is "fine" but the audio still has
artifacts — the value distribution tells you whether a few outlier values are
drifting (a quantization-cliff issue) vs broad small drift (a precision issue).

Run:
    pytest -s models/demos/qwen3_tts/tests/test_per_block_values.py
or:
    python models/demos/qwen3_tts/tests/test_per_block_values.py --layer 0 --seq 128
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.qwen3_tts.reference import functional as F_ref
from models.demos.qwen3_tts.reference.functional import get_default_talker_config
from models.demos.qwen3_tts.tt.attention import Attention
from models.demos.qwen3_tts.tt.mlp import MLP
from models.demos.qwen3_tts.tt.rope import compute_rope_frequencies, get_transformation_mat


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def _flatten(*ts):
    return [t.detach().cpu().float().flatten() for t in ts]


def _pcc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    n = a.norm() * b.norm()
    return float((a * b).sum() / n) if n > 1e-12 else 0.0


def _cos(a, b):
    n = a.norm() * b.norm()
    return float((a * b).sum() / n) if n > 1e-12 else 0.0


def _hist_log(diff_abs: torch.Tensor, edges=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0)) -> str:
    """Histogram of |Δ| values in log-spaced bins, returned as a one-line string."""
    counts = []
    edges_full = [-float("inf"), *edges, float("inf")]
    labels = [f"<{edges[0]}"] + [f"<{e}" for e in edges[1:]] + [f">={edges[-1]}"]
    total = diff_abs.numel()
    for lo, hi in zip(edges_full[:-1], edges_full[1:]):
        c = ((diff_abs >= lo) & (diff_abs < hi)).sum().item()
        counts.append(c)
    parts = [f"{lab}:{c/total*100:5.1f}%" for lab, c in zip(labels, counts)]
    return "  ".join(parts)


def _topk_disagree(ref: torch.Tensor, tt: torch.Tensor, k: int = 5) -> str:
    """Return a string describing the top-k indices with largest |ref - tt|."""
    ref_f = ref.detach().cpu().float().flatten()
    tt_f = tt.detach().cpu().float().flatten()
    diff = (ref_f - tt_f).abs()
    if diff.numel() == 0:
        return "(empty)"
    k = min(k, diff.numel())
    vals, idxs = torch.topk(diff, k)
    lines = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        lines.append(f"    idx={i:>8d}  ref={ref_f[i]:+9.4f}  tt={tt_f[i]:+9.4f}  |Δ|={v:.5f}")
    return "\n".join(lines)


def report(label: str, ref: torch.Tensor, tt: torch.Tensor) -> None:
    """Print rich value-level diagnostics for one tensor pair."""
    if ref.shape != tt.shape:
        print(f"\n[{label}] SHAPE MISMATCH: ref={tuple(ref.shape)} tt={tuple(tt.shape)}")
        return
    rf, tf = _flatten(ref, tt)
    diff = (rf - tf).abs()
    rms_ref = float((rf**2).mean().sqrt())
    rms_diff = float((diff**2).mean().sqrt())
    print(
        f"\n[{label}]  shape={tuple(ref.shape)}  N={rf.numel()}\n"
        f"  PCC={_pcc(rf,tf):.6f}  cos={_cos(rf,tf):.6f}  "
        f"max|Δ|={float(diff.max()):.5f}  mean|Δ|={float(diff.mean()):.5f}  "
        f"RMS_diff/RMS_ref={rms_diff/(rms_ref+1e-12):.4f}\n"
        f"  ref stats: mean={float(rf.mean()):+.4f}  std={float(rf.std()):.4f}  "
        f"min={float(rf.min()):+.4f}  max={float(rf.max()):+.4f}\n"
        f"  |Δ| hist:  {_hist_log(diff)}\n"
        f"  top-5 worst absolute disagreements:\n{_topk_disagree(rf, tf, 5)}"
    )


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------
def load_state_dict():
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    model_path = Path(snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", allow_patterns=["*.safetensors"]))
    sd = {}
    for f in model_path.glob("*.safetensors"):
        if "speech_tokenizer" in str(f):
            continue
        sd.update(load_file(f))
    return sd


# ---------------------------------------------------------------------------
# CPU reference per sub-block (so we can compare each step)
# ---------------------------------------------------------------------------
def reference_layer_with_intermediates(x, layer_w, cos, sin, cfg) -> dict:
    """Run the reference layer step-by-step, returning every intermediate."""
    out = {"input": x}
    residual = x
    h = F_ref.rms_norm(x, layer_w["input_layernorm.weight"], cfg.rms_norm_eps)
    out["after_input_norm"] = h
    h = F_ref.attention(
        h,
        q_proj_weight=layer_w["self_attn.q_proj.weight"],
        k_proj_weight=layer_w["self_attn.k_proj.weight"],
        v_proj_weight=layer_w["self_attn.v_proj.weight"],
        o_proj_weight=layer_w["self_attn.o_proj.weight"],
        q_norm_weight=layer_w["self_attn.q_norm.weight"],
        k_norm_weight=layer_w["self_attn.k_norm.weight"],
        cos=cos,
        sin=sin,
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        rms_norm_eps=cfg.rms_norm_eps,
        use_mrope=False,
    )
    out["after_attention"] = h
    out["after_attn_residual"] = residual + h

    residual = out["after_attn_residual"]
    h = F_ref.rms_norm(residual, layer_w["post_attention_layernorm.weight"], cfg.rms_norm_eps)
    out["after_post_attn_norm"] = h
    h = F_ref.swiglu_mlp(
        h,
        gate_proj_weight=layer_w["mlp.gate_proj.weight"],
        up_proj_weight=layer_w["mlp.up_proj.weight"],
        down_proj_weight=layer_w["mlp.down_proj.weight"],
    )
    out["after_mlp"] = h
    out["layer_output"] = residual + h
    return out


# ---------------------------------------------------------------------------
# TTNN sub-block runner
# ---------------------------------------------------------------------------
def ttnn_layer_with_intermediates(device, state_dict, layer_idx, x_torch, cos, sin, cfg) -> dict:
    """Build attention + MLP individually, drive them with the same input as the
    reference, and capture each sub-block output. We don't use DecoderLayer.forward
    because that returns only the final tensor; sub-block calls expose the
    intermediates we want to inspect."""
    layer_prefix = f"talker.model.layers.{layer_idx}"

    # Reference RMSNorm weight tensors (we reuse F_ref.rms_norm on torch tensors
    # for a clean point-of-comparison of the *entry* into each TTNN sub-block).
    ln_in_w = state_dict[f"{layer_prefix}.input_layernorm.weight"]
    ln_post_w = state_dict[f"{layer_prefix}.post_attention_layernorm.weight"]

    # Run input layernorm in fp32 on host so the attention input is identical to
    # the reference's "after_input_norm". This keeps any TTNN drift confined to
    # attention itself.
    h_after_in_norm = F_ref.rms_norm(x_torch, ln_in_w, cfg.rms_norm_eps)

    attn = Attention(
        device=device,
        hidden_size=cfg.hidden_size,
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        state_dict=state_dict,
        layer_prefix=layer_prefix,
        rms_norm_eps=cfg.rms_norm_eps,
        weight_dtype=ttnn.bfloat16,
    )
    mlp = MLP(
        device=device,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        state_dict=state_dict,
        layer_prefix=layer_prefix,
        weight_dtype=ttnn.bfloat16,
    )

    # Push the post-input-norm tensor onto the device.
    seq_len = h_after_in_norm.shape[1]
    h_dev = ttnn.from_torch(
        h_after_in_norm.unsqueeze(1),  # [B, 1, S, H]
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cos_tt = ttnn.from_torch(
        cos.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sin_tt = ttnn.from_torch(
        sin.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    trans_mat = get_transformation_mat(cfg.head_dim, device)

    attn_out_dev, _ = attn.forward(h_dev, cos_tt, sin_tt, trans_mat, mode="prefill")
    attn_out = ttnn.to_torch(attn_out_dev).squeeze(1)  # [B, S, H]
    ttnn.deallocate(attn_out_dev)
    ttnn.deallocate(h_dev)

    after_attn_residual = x_torch + attn_out
    h_after_post_norm = F_ref.rms_norm(after_attn_residual, ln_post_w, cfg.rms_norm_eps)

    mlp_in_dev = ttnn.from_torch(
        h_after_post_norm.unsqueeze(1),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    mlp_out_dev = mlp(mlp_in_dev, mode="prefill")
    mlp_out = ttnn.to_torch(mlp_out_dev).squeeze(1)
    ttnn.deallocate(mlp_in_dev)
    ttnn.deallocate(mlp_out_dev)
    ttnn.deallocate(cos_tt)
    ttnn.deallocate(sin_tt)
    ttnn.deallocate(trans_mat)

    layer_out = after_attn_residual + mlp_out

    return {
        "after_input_norm": h_after_in_norm,
        "after_attention": attn_out,
        "after_attn_residual": after_attn_residual,
        "after_post_attn_norm": h_after_post_norm,
        "after_mlp": mlp_out,
        "layer_output": layer_out,
    }


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------
def _do_layer_compare(layer_idx: int = 0, seq_len: int = 128, seed: int = 42):
    cfg = get_default_talker_config()
    torch.manual_seed(seed)
    state_dict = load_state_dict()
    layer_w = F_ref.extract_layer_weights(state_dict, layer_idx, prefix="talker.model.")
    if not layer_w:
        raise RuntimeError(f"No weights found for layer {layer_idx} (talker.model.layers.{layer_idx}.*)")

    x_torch = torch.randn(1, seq_len, cfg.hidden_size, dtype=torch.bfloat16) * 0.5
    cos, sin = compute_rope_frequencies(cfg.head_dim, seq_len, 1000000.0)

    print(f"\n==================  Layer {layer_idx}, seq_len={seq_len}  ==================")
    ref = reference_layer_with_intermediates(x_torch, layer_w, cos, sin, cfg)

    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    device.enable_program_cache()
    try:
        tt = ttnn_layer_with_intermediates(device, state_dict, layer_idx, x_torch, cos, sin, cfg)
    finally:
        ttnn.close_device(device)

    # Compare every sub-block with rich diagnostics.
    for key in [
        "after_input_norm",
        "after_attention",
        "after_attn_residual",
        "after_post_attn_norm",
        "after_mlp",
        "layer_output",
    ]:
        if key in ref and key in tt:
            report(key, ref[key], tt[key])


@pytest.mark.parametrize("layer_idx", [0])
@pytest.mark.parametrize("seq_len", [128])
def test_per_block_values_layer(layer_idx, seq_len):
    """Print per-sub-block value-level comparison. Always passes; this is a
    diagnostic test, not a pass/fail gate."""
    _do_layer_compare(layer_idx=layer_idx, seq_len=seq_len)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    _do_layer_compare(layer_idx=args.layer, seq_len=args.seq, seed=args.seed)


if __name__ == "__main__":
    main()
    sys.exit(0)
