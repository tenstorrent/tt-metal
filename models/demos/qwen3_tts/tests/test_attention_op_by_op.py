# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Op-by-op diagnostic for the Talker attention sub-block.

Walks the reference (`F_ref.attention`) and a hand-written op-by-op TTNN
implementation (using the same primitives the production `Attention` class
uses) through identical inputs, capturing intermediates at every meaningful
boundary. For each boundary we print PCC, cosine, max|Δ|, mean|Δ|, RMS ratio,
top-5 worst-disagreement indices, and a log-binned |Δ| histogram.

The point: when `test_per_block_values.py::after_attention` reports PCC=0.69,
this test tells you whether it's the projection, qk-norm, rope, SDPA scores,
softmax, attn@v, or the o_proj that drifts. Only one of those should be the
culprit.

Run:
    python models/demos/qwen3_tts/tests/test_attention_op_by_op.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

import ttnn
from models.demos.qwen3_tts.reference import functional as F_ref
from models.demos.qwen3_tts.reference.functional import get_default_talker_config
from models.demos.qwen3_tts.tt.rope import compute_rope_frequencies


# ---------------------------------------------------------------------------
# Diagnostics (same as test_per_block_values.py — duplicated to keep this
# file standalone)
# ---------------------------------------------------------------------------
def _hist_log(diff_abs: torch.Tensor, edges=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0)) -> str:
    counts = []
    edges_full = [-float("inf"), *edges, float("inf")]
    labels = [f"<{edges[0]}"] + [f"<{e}" for e in edges[1:]] + [f">={edges[-1]}"]
    total = diff_abs.numel()
    for lo, hi in zip(edges_full[:-1], edges_full[1:]):
        c = ((diff_abs >= lo) & (diff_abs < hi)).sum().item()
        counts.append(c)
    return "  ".join(f"{lab}:{c/total*100:5.1f}%" for lab, c in zip(labels, counts))


def _topk_disagree(ref: torch.Tensor, tt: torch.Tensor, k: int = 5) -> str:
    rf = ref.detach().cpu().float().flatten()
    tf = tt.detach().cpu().float().flatten()
    diff = (rf - tf).abs()
    if diff.numel() == 0:
        return "    (empty)"
    k = min(k, diff.numel())
    vals, idxs = torch.topk(diff, k)
    out = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        out.append(f"    flat_idx={i:>10d}  ref={rf[i]:+9.4f}  tt={tf[i]:+9.4f}  |Δ|={v:.5f}")
    return "\n".join(out)


def report(label: str, ref: torch.Tensor, tt: torch.Tensor) -> None:
    if ref.shape != tt.shape:
        print(f"\n[{label}] SHAPE MISMATCH: ref={tuple(ref.shape)} tt={tuple(tt.shape)}")
        return
    rf = ref.detach().cpu().float().flatten()
    tf = tt.detach().cpu().float().flatten()
    diff = (rf - tf).abs()
    a = rf - rf.mean()
    b = tf - tf.mean()
    pcc = float((a * b).sum() / (a.norm() * b.norm() + 1e-12))
    cos = float((rf * tf).sum() / (rf.norm() * tf.norm() + 1e-12))
    rms_ref = float((rf**2).mean().sqrt())
    rms_diff = float((diff**2).mean().sqrt())
    print(
        f"\n[{label}]  shape={tuple(ref.shape)}\n"
        f"  PCC={pcc:.6f}  cos={cos:.6f}  "
        f"max|Δ|={float(diff.max()):.5f}  mean|Δ|={float(diff.mean()):.5f}  "
        f"RMS_diff/RMS_ref={rms_diff/(rms_ref+1e-12):.4f}\n"
        f"  ref stats: mean={float(rf.mean()):+.4f}  std={float(rf.std()):.4f}\n"
        f"  |Δ| hist:  {_hist_log(diff)}\n"
        f"  top-5 worst:\n{_topk_disagree(rf, tf, 5)}"
    )


# ---------------------------------------------------------------------------
# Weight loading + reference: step-by-step attention returning every op output
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


def reference_attention_step_by_step(x, layer_w, cos, sin, cfg) -> dict:
    """Mirrors F_ref.attention exactly, but returns a dict of every intermediate.
    Uses use_mrope=False (1D RoPE) to match what the test_per_block_values setup
    does — and what TTNN is mathematically equivalent to for text-only inputs."""
    B, S, H = x.shape
    nh = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    d = cfg.head_dim
    n_groups = nh // nkv

    out = {}

    q = F.linear(x, layer_w["self_attn.q_proj.weight"])
    k = F.linear(x, layer_w["self_attn.k_proj.weight"])
    v = F.linear(x, layer_w["self_attn.v_proj.weight"])
    out["01_q_proj"] = q  # [B, S, nh*d]
    out["02_k_proj"] = k  # [B, S, nkv*d]
    out["03_v_proj"] = v  # [B, S, nkv*d]

    # reshape (head split) — pre-transpose, before qk-norm
    q = q.view(B, S, nh, d)
    k = k.view(B, S, nkv, d)
    v = v.view(B, S, nkv, d)
    out["04_q_split"] = q  # [B, S, nh, d]
    out["05_k_split"] = k

    # qk-norm on head_dim
    q = F_ref.rms_norm(q, layer_w["self_attn.q_norm.weight"], cfg.rms_norm_eps)
    k = F_ref.rms_norm(k, layer_w["self_attn.k_norm.weight"], cfg.rms_norm_eps)
    out["06_q_norm"] = q
    out["07_k_norm"] = k

    # transpose to [B, H, S, D]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # 1D RoPE (non-interleaved rotate_half)
    q, k = F_ref.apply_rotary_pos_emb(q, k, cos, sin)
    out["08_q_rope"] = q  # [B, nh, S, d]
    out["09_k_rope"] = k  # [B, nkv, S, d]

    # GQA expand
    k = F_ref.repeat_kv(k, n_groups)
    v = F_ref.repeat_kv(v, n_groups)
    out["10_k_expanded"] = k  # [B, nh, S, d]
    out["11_v_expanded"] = v

    # Scaled scores
    scaling = d**-0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scaling
    out["12_scores_pre_mask"] = scores  # [B, nh, S, S]

    # Causal mask
    causal_mask = torch.full((S, S), float("-inf"), dtype=scores.dtype)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    scores = scores + causal_mask
    out["13_scores_post_mask"] = scores

    # Softmax (fp32 internal, cast back)
    attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out["14_attn_weights"] = attn_weights

    # attn @ V
    attn_out = torch.matmul(attn_weights, v)
    out["15_attn_at_v"] = attn_out  # [B, nh, S, d]

    # Concat heads
    attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, S, -1)
    out["16_concat_heads"] = attn_out  # [B, S, nh*d]

    # o_proj
    attn_out = F.linear(attn_out, layer_w["self_attn.o_proj.weight"])
    out["17_o_proj"] = attn_out  # [B, S, hidden]

    return out


# ---------------------------------------------------------------------------
# Op-by-op TTNN — uses the same primitives & weight permutation as the
# production Attention class but runs each op in-line so we can read every
# intermediate back to torch and compare.
# ---------------------------------------------------------------------------
def _permute_rope_rows(weight_2d: torch.Tensor, local_heads: int, head_dim: int) -> torch.Tensor:
    """Same as Attention._permute_rope_head_dim_rows: convert each head block
    from non-interleaved [d0..d63, d64..d127] to interleaved [d0, d64, d1, d65, ...]
    so that the interleaved-RoPE TTNN kernel produces the same result as
    non-interleaved RoPE on the original weights."""
    h_in = int(weight_2d.shape[1])
    half = head_dim // 2
    w = weight_2d.reshape(local_heads, head_dim, h_in)
    out = w.clone()
    out[:, 0::2, :] = w[:, :half, :]
    out[:, 1::2, :] = w[:, half:, :]
    return out.reshape(local_heads * head_dim, h_in)


def _permute_rope_vector(weight_1d: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Same permutation, applied to the per-head qk-norm gain vector."""
    half = head_dim // 2
    out = weight_1d.clone()
    out[0::2] = weight_1d[:half]
    out[1::2] = weight_1d[half:]
    return out


def _to_dev(t: torch.Tensor, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    return ttnn.from_torch(t, device=device, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _to_torch(t: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(t)


def ttnn_attention_step_by_step(device, layer_w, x_torch, cos_torch, sin_torch, cfg) -> dict:
    """Op-by-op TTNN attention. Captures the same boundaries as the reference.

    The interleaved-RoPE permutation on Q/K projection rows + qk-norm vectors
    is applied here exactly as the production Attention.__init__ does, so this
    function is a faithful op-by-op standin for the production path."""
    nh = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    d = cfg.head_dim

    # Permuted weights (so interleaved-kernel RoPE on permuted Q/K equals
    # non-interleaved-RoPE on original Q/K — same as production).
    qw = _permute_rope_rows(layer_w["self_attn.q_proj.weight"], nh, d)
    kw = _permute_rope_rows(layer_w["self_attn.k_proj.weight"], nkv, d)
    vw = layer_w["self_attn.v_proj.weight"]
    ow = layer_w["self_attn.o_proj.weight"]
    qn = _permute_rope_vector(layer_w["self_attn.q_norm.weight"], d)
    kn = _permute_rope_vector(layer_w["self_attn.k_norm.weight"], d)

    out = {}

    # 01-03: linear projections
    x_dev = _to_dev(x_torch, device)
    qw_dev = _to_dev(qw.t().contiguous().unsqueeze(0).unsqueeze(0), device)  # [1,1,H,nh*d]
    kw_dev = _to_dev(kw.t().contiguous().unsqueeze(0).unsqueeze(0), device)
    vw_dev = _to_dev(vw.t().contiguous().unsqueeze(0).unsqueeze(0), device)
    q_dev = ttnn.linear(x_dev, qw_dev, memory_config=ttnn.L1_MEMORY_CONFIG)
    k_dev = ttnn.linear(x_dev, kw_dev, memory_config=ttnn.L1_MEMORY_CONFIG)
    v_dev = ttnn.linear(x_dev, vw_dev, memory_config=ttnn.L1_MEMORY_CONFIG)
    # NOTE: production permutes Q/K projection rows, so q_proj output of TTNN
    # has head_dim in INTERLEAVED order (d0,d64,d1,d65,...). To compare to the
    # reference's q_proj (in original head_dim order), we have to apply the
    # inverse permutation on the TTNN output.
    q_torch_perm = _to_torch(q_dev).reshape(*x_torch.shape[:2], nh, d)
    k_torch_perm = _to_torch(k_dev).reshape(*x_torch.shape[:2], nkv, d)
    v_torch = _to_torch(v_dev).reshape(*x_torch.shape[:2], -1)

    def _inverse_perm(t):  # interleaved -> non-interleaved on last dim
        half = d // 2
        out_t = t.clone()
        out_t[..., :half] = t[..., 0::2]
        out_t[..., half:] = t[..., 1::2]
        return out_t

    out["01_q_proj"] = _inverse_perm(q_torch_perm).reshape(*x_torch.shape[:2], -1)
    out["02_k_proj"] = _inverse_perm(k_torch_perm).reshape(*x_torch.shape[:2], -1)
    out["03_v_proj"] = v_torch
    out["04_q_split"] = _inverse_perm(q_torch_perm)
    out["05_k_split"] = _inverse_perm(k_torch_perm)

    # 06-07: qk-norm (with permuted gain vectors so the result is the same as
    # qk-norm with original gain on un-permuted Q/K). To compare to the reference,
    # we run qk-norm on host using the permuted weight, then inverse-permute.
    q_normed_perm = F_ref.rms_norm(q_torch_perm, qn, cfg.rms_norm_eps)
    k_normed_perm = F_ref.rms_norm(k_torch_perm, kn, cfg.rms_norm_eps)
    out["06_q_norm"] = _inverse_perm(q_normed_perm)
    out["07_k_norm"] = _inverse_perm(k_normed_perm)

    # Beyond this point the TTNN production path uses
    # rotary_embedding_llama+SDPA with very specific layouts (sharded, fp32
    # cast, etc). Doing them op-by-op via direct ttnn calls re-implements
    # production attention, which we don't want here. Stop the TTNN side at
    # post-qk-norm — that's enough to localize whether divergence is upstream
    # (linear / qk-norm) or downstream (rope / SDPA / o_proj).
    ttnn.deallocate(x_dev)
    ttnn.deallocate(q_dev)
    ttnn.deallocate(k_dev)
    ttnn.deallocate(v_dev)
    ttnn.deallocate(qw_dev)
    ttnn.deallocate(kw_dev)
    ttnn.deallocate(vw_dev)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = get_default_talker_config()
    torch.manual_seed(args.seed)
    state_dict = load_state_dict()
    layer_w = F_ref.extract_layer_weights(state_dict, args.layer, prefix="talker.model.")
    if not layer_w:
        raise RuntimeError(f"No weights for layer {args.layer}")

    x_torch = torch.randn(1, args.seq, cfg.hidden_size, dtype=torch.bfloat16) * 0.5
    cos, sin = compute_rope_frequencies(cfg.head_dim, args.seq, 1000000.0)

    print(f"\n=========== Layer {args.layer}, seq_len={args.seq}, op-by-op attention ===========")
    print(
        f"  hidden={cfg.hidden_size}  heads={cfg.num_attention_heads}/{cfg.num_key_value_heads}  head_dim={cfg.head_dim}"
    )

    print("\n--- Reference: 17 boundaries ---")
    ref = reference_attention_step_by_step(x_torch, layer_w, cos, sin, cfg)

    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    device.enable_program_cache()
    try:
        print("\n--- TTNN: linear+qk-norm boundaries (early ops only) ---")
        tt = ttnn_attention_step_by_step(device, layer_w, x_torch, cos, sin, cfg)
    finally:
        ttnn.close_device(device)

    print(f"\n=========== Op-by-op comparison (reference vs TTNN, post-inverse-permute) ===========")
    for key in ["01_q_proj", "02_k_proj", "03_v_proj", "06_q_norm", "07_k_norm"]:
        if key in ref and key in tt:
            report(key, ref[key], tt[key])

    print(
        "\nNote: only linear+qk-norm boundaries are compared here. If those all\n"
        "show PCC>0.999, the divergence is downstream (RoPE / SDPA / o_proj).\n"
        "If any of the linear/qk-norm boundaries already drift, the bug is in\n"
        "the projection or the qk-norm — not in RoPE."
    )


if __name__ == "__main__":
    main()
