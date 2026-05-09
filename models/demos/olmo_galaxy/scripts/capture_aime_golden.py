#!/usr/bin/env python3
"""
CPU-only HF reference golden capture for OLMo-3.1-32B-Think on AIME problems.

For one AIME problem, runs the HF reference (pure CPU) for prefill + N decode
tokens (free-run argmax), capturing every per-(layer, op) intermediate tensor
at every step. Persists to per-layer .pt files in <out_dir>/aime24_p{N}/.

The per-op names match exactly what the TT-side _capture_attn / _capture_mlp
hooks emit in models/demos/olmo_galaxy/tt/llama_{attention,mlp}.py, so a
follow-up TTNN test can index by the same keys.

Run as a CLI script (no pytest, no Galaxy mesh):

    python capture_aime_golden.py \\
        --problem 1 --out-dir /data/aime_goldens \\
        --n-decode-tokens 2000 --n-layers 64 \\
        --hf-model /home/cust-team/models/OLMo-3.1-32B-Think

Designed for 4 problems running concurrently in separate processes.
Memory: ~120 GB RAM per process (fp32 reference model + activations).
Disk: ~13 GB output per problem at fp16.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]  # tt-metal repo root
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

# GPT2Tokenizer used by the demo / TT-side tokenizer fallback.
from transformers import GPT2Tokenizer  # noqa: E402

from models.demos.olmo_galaxy.reference.olmo import Attention as RefAttention  # noqa: E402
from models.demos.olmo_galaxy.reference.olmo import TransformerBlock, apply_rotary_emb, repeat_kv

# --- Reuse the existing reference + state-dict loading helpers ---------------
# build_ref_model and load_hf_raw_state_dict live in the e2e test module.
# Importing the test module is the path of least duplication; pytest is not
# triggered because we don't call any test_* function.
from models.demos.olmo_galaxy.tests.test_olmo_e2e_pcc import build_ref_model, load_hf_raw_state_dict  # noqa: E402


# ----- Layout helper: HF split-half  ->  Meta interleaved per head -----------
# TT applies Meta-style RoPE on Meta-permuted Q/K weights. To make the saved
# golden directly compatible with TT captures (which arrive in Meta layout
# after our _tt_to_hf_layout conversion in the comparison test), we save the
# ref Q/K in HF layout and let the comparison side handle the conversion.
# This matches the existing test_decode_per_op_pcc_4layers convention.
def hf_to_meta_qk(t: torch.Tensor, head_dim: int = 128) -> torch.Tensor:
    """[..., r0..r63, i0..i63] per head  →  [..., r0,i0,r1,i1,...]."""
    orig_shape = t.shape
    total_dim = orig_shape[-1]
    n_heads = total_dim // head_dim
    t = t.view(*orig_shape[:-1], n_heads, head_dim)
    reals = t[..., : head_dim // 2]
    imags = t[..., head_dim // 2 :]
    return torch.stack((reals, imags), dim=-1).flatten(start_dim=-2).view(orig_shape)


# ----- Capture buffers --------------------------------------------------------
# per_layer_capture[L]: dict of {step_idx: {op_name: tensor[fp16]}}
# Keep in RAM during the run (~6 GB total at fp16 for full 64L × 2000 steps),
# write to disk per-layer at the end. If memory becomes an issue we can flush
# every K steps; for now the simple path is fine.
PER_LAYER_CAPTURE: dict[int, dict[int, dict[str, torch.Tensor]]] = {}
CURRENT_STEP_IDX: int = 0  # written each step before forward()


def _store(layer_idx: int, name: str, tensor: torch.Tensor) -> None:
    """Stash user-0 last-position only, fp16 cast, into the layer/step bucket.

    Tensor shape: [bsz, seqlen, ...]. We always take the last position
    (`tensor[0, -1, ...]`) so prefill (seqlen=padded_prefill) collapses to
    the same shape as a single decode step. The first decoded token's input
    is exactly this last-prefill-position state, so it's the most informative
    slice; full per-position prefill state at all 64 layers would be
    ~150 MB/layer/step, blowing the disk budget."""
    last = tensor[0, -1].detach().to(torch.float16).cpu().contiguous().clone()
    PER_LAYER_CAPTURE.setdefault(layer_idx, {}).setdefault(CURRENT_STEP_IDX, {})[name] = last


# ----- Monkey-patched forwards (lifted from test_olmo_e2e_pcc:654-742) -------
def _patched_attn_forward(self_attn, x, start_pos, freqs_cis, mask):
    li = self_attn.layer_id
    bsz, seqlen, _ = x.shape

    xq = self_attn.wq(x)
    xk = self_attn.wk(x)
    xv = self_attn.wv(x)

    def _global_rms_norm(t, weight):
        t_f = t.float()
        rms = torch.rsqrt(t_f.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (t_f * rms * weight.to(t_f.device)).type_as(t)

    # Pre-norm Q/K/V (in Meta layout to match TT)
    _store(li, "q_pre_norm", hf_to_meta_qk(xq))
    _store(li, "k_pre_norm", hf_to_meta_qk(xk))
    _store(li, "v_heads", xv)

    # OLMo3 QK-norm: global RMSNorm with per-head weight, applied in Meta layout.
    xq_meta = hf_to_meta_qk(xq)
    xk_meta = hf_to_meta_qk(xk)
    q_norm_meta = hf_to_meta_qk(self_attn.q_norm_weight.unsqueeze(0), head_dim=128).squeeze(0)
    k_norm_meta = hf_to_meta_qk(self_attn.k_norm_weight.unsqueeze(0), head_dim=128).squeeze(0)
    xq_n = _global_rms_norm(xq_meta, q_norm_meta)
    xk_n = _global_rms_norm(xk_meta, k_norm_meta)
    _store(li, "q_post_norm", xq_n)
    _store(li, "k_post_norm", xk_n)

    # RoPE (HF format internally — we'll permute to Meta after for storage)
    xq_h = xq_n.view(bsz, seqlen, self_attn.n_heads, self_attn.head_dim)
    xk_h = xk_n.view(bsz, seqlen, self_attn.n_kv_heads, self_attn.head_dim)
    xv_h = xv.view(bsz, seqlen, self_attn.n_kv_heads, self_attn.head_dim)

    # apply_rotary_emb expects HF-layout [r..r, i..i]; xq_n/xk_n are Meta-layout.
    # The test stores Meta-format ref captures, so we re-permute back.
    # For SDPA we need HF-style heads; the original (non-Meta) values are still
    # available via the plain wq/wk projections. Re-derive them for SDPA only:
    xq_hf = xq.view(bsz, seqlen, self_attn.n_heads, self_attn.head_dim)
    xk_hf = xk.view(bsz, seqlen, self_attn.n_kv_heads, self_attn.head_dim)
    # Apply HF QK-norm (matches reference implementation exactly)
    xq_hf_n = _global_rms_norm(xq_hf.reshape(bsz, seqlen, -1), self_attn.q_norm_weight).view(
        bsz, seqlen, self_attn.n_heads, self_attn.head_dim
    )
    xk_hf_n = _global_rms_norm(xk_hf.reshape(bsz, seqlen, -1), self_attn.k_norm_weight).view(
        bsz, seqlen, self_attn.n_kv_heads, self_attn.head_dim
    )
    xq_r, xk_r = apply_rotary_emb(xq_hf_n, xk_hf_n, freqs_cis)

    _store(li, "q_post_rope", xq_r.reshape(bsz, seqlen, -1))
    _store(li, "k_post_rope", xk_r.reshape(bsz, seqlen, -1))

    # Update KV cache (HF-layout K post-rope) and run attention.
    self_attn.cache_k[:bsz, start_pos : start_pos + seqlen] = xk_r
    self_attn.cache_v[:bsz, start_pos : start_pos + seqlen] = xv_h
    keys = self_attn.cache_k[:bsz, : start_pos + seqlen]
    values = self_attn.cache_v[:bsz, : start_pos + seqlen]
    keys = repeat_kv(keys, self_attn.n_rep)
    values = repeat_kv(values, self_attn.n_rep)
    xq_t = xq_r.transpose(1, 2)
    keys_t = keys.transpose(1, 2)
    values_t = values.transpose(1, 2)
    scale = (self_attn.head_dim**-0.5) * self_attn.mscale
    scores = torch.matmul(xq_t, keys_t.transpose(2, 3)) * scale
    if mask is not None:
        scores = scores + mask
    attn_weights = torch.nn.functional.softmax(scores.float(), dim=-1).type_as(xq_t)
    sdpa_raw = torch.matmul(attn_weights, values_t)
    sdpa_out = sdpa_raw.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    _store(li, "sdpa_out", sdpa_out)

    wo_out = self_attn.wo(sdpa_out)
    _store(li, "wo_out", wo_out)
    return wo_out


def _patched_block_forward(self_block, x, start_pos, freqs_cis, mask):
    li = self_block.layer_id
    _store(li, "layer_in", x)

    attn_raw = self_block.attention(x, start_pos, freqs_cis, mask)
    _store(li, "attn_out", attn_raw)

    attn_normed = self_block.attention_norm(attn_raw)
    _store(li, "attn_normed", attn_normed)

    h = x + attn_normed
    _store(li, "h_attn", h)

    ff = self_block.feed_forward
    w1_out = ff.w1(h)
    w3_out = ff.w3(h)
    _store(li, "w1_out", w1_out)
    _store(li, "w3_out", w3_out)

    ff1ff3 = torch.nn.functional.silu(w1_out) * w3_out
    _store(li, "ff1ff3", ff1ff3)

    ff_raw = ff.w2(ff1ff3)
    _store(li, "w2_out_pre_ar", ff_raw)
    _store(li, "ff_out", ff_raw)

    ff_normed = self_block.ffn_norm(ff_raw)
    _store(li, "ff_normed", ff_normed)

    out = h + ff_normed
    _store(li, "layer_out", out)
    return out


# ----- Main capture loop -----------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", type=int, choices=[1, 2, 3, 4], required=True)
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output root, e.g. /data/aime_goldens",
    )
    ap.add_argument("--n-decode-tokens", type=int, default=2000)
    ap.add_argument("--n-layers", type=int, default=64)
    ap.add_argument("--max-seq-len", type=int, default=2560)
    ap.add_argument(
        "--hf-model",
        type=str,
        default=os.environ.get("HF_MODEL", "/home/cust-team/models/OLMo-3.1-32B-Think"),
    )
    ap.add_argument(
        "--padded-prefill",
        type=int,
        default=512,
        help="Pad prompt to this length; AIME prompts are < 400 tokens.",
    )
    args = ap.parse_args()

    out_dir = args.out_dir / f"aime24_p{args.problem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[capture] writing to {out_dir}", flush=True)

    # ---- Load AIME prompt --------------------------------------------------
    prompt_path = REPO_ROOT / "models/demos/olmo_galaxy/demo/sample_prompts" / f"input_data_aime24_p{args.problem}.json"
    with open(prompt_path) as f:
        prompt_obj = json.load(f)[0]
    prompt = prompt_obj["prompt"]
    print(f"[capture] prompt: {prompt[:120]}...", flush=True)

    # ---- Build CPU reference model ----------------------------------------
    print(f"[capture] loading HF state dict from {args.hf_model}", flush=True)
    t0 = time.time()
    hf_sd = load_hf_raw_state_dict(args.hf_model)
    print(f"[capture] state dict loaded in {time.time() - t0:.1f}s", flush=True)

    print(f"[capture] building reference model ({args.n_layers} layers)...", flush=True)
    t0 = time.time()
    ref_model = build_ref_model(hf_sd, n_layers=args.n_layers, max_seq_len=args.max_seq_len, max_batch_size=1)
    ref_model.eval()
    print(f"[capture] ref_model built in {time.time() - t0:.1f}s", flush=True)
    del hf_sd  # free memory

    # ---- Tokenize ----------------------------------------------------------
    # Resolve tokenizer from HF snapshot (matches demo behavior).
    import glob

    base_path = os.path.expanduser(args.hf_model)
    if os.path.exists(os.path.join(base_path, "snapshots")):
        snap_dirs = glob.glob(os.path.join(base_path, "snapshots", "*"))
        if snap_dirs:
            base_path = snap_dirs[0]
    tokenizer = GPT2Tokenizer.from_pretrained(base_path)

    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    seq_len = len(input_ids)
    padded_len = args.padded_prefill
    if seq_len > padded_len:
        raise ValueError(f"prompt has {seq_len} tokens, exceeds padded_prefill={padded_len}")
    eos = tokenizer.eos_token_id or 50256
    input_ids_padded = input_ids + [eos] * (padded_len - seq_len)
    tokens_pt = torch.tensor(input_ids_padded, dtype=torch.long).unsqueeze(0)

    print(
        f"[capture] seq_len={seq_len}, padded_len={padded_len}, " f"n_decode={args.n_decode_tokens}",
        flush=True,
    )

    # ---- Install monkey-patches -------------------------------------------
    original_attn_forward = RefAttention.forward
    original_block_forward = TransformerBlock.forward
    RefAttention.forward = _patched_attn_forward
    TransformerBlock.forward = _patched_block_forward

    # ---- Prefill (step_idx = -1 = "prefill") ------------------------------
    global CURRENT_STEP_IDX
    CURRENT_STEP_IDX = -1
    print(f"[capture] prefill (seq_len={seq_len})...", flush=True)
    t0 = time.time()
    embeddings = ref_model.tok_embeddings(tokens_pt[:, :padded_len]).float()
    prefill_out = ref_model.forward(embeddings, start_pos=0, mode="decode")
    # prefill_out: [bsz=1, padded_len, vocab]; logits at the last real token
    prefill_logits = prefill_out[:, seq_len - 1, :].squeeze(0)
    first_token = int(prefill_logits.argmax(dim=-1).item())
    ref_tokens = [first_token]
    print(
        f"[capture] prefill done in {time.time() - t0:.1f}s; "
        f"first token = {first_token} ({tokenizer.decode([first_token])})",
        flush=True,
    )

    # ---- Decode loop ------------------------------------------------------
    print(f"[capture] decode {args.n_decode_tokens} tokens...", flush=True)
    decode_t0 = time.time()
    for step in range(args.n_decode_tokens):
        CURRENT_STEP_IDX = step
        pos = seq_len + step
        tok_emb = ref_model.tok_embeddings(torch.tensor([[ref_tokens[-1]]])).float()
        out = ref_model.forward(tok_emb, start_pos=pos, mode="decode")
        next_tok = int(out[:, -1, :].argmax(dim=-1).item())
        ref_tokens.append(next_tok)

        if step % 50 == 0 or step == args.n_decode_tokens - 1:
            elapsed = time.time() - decode_t0
            rate = (step + 1) / elapsed if elapsed > 0 else 0
            print(
                f"[capture] step {step+1}/{args.n_decode_tokens} "
                f"({rate:.2f} steps/s, eta {(args.n_decode_tokens - step - 1) / max(rate, 1e-6):.0f}s); "
                f"tok={next_tok}",
                flush=True,
            )

    # ---- Restore originals -----------------------------------------------
    RefAttention.forward = original_attn_forward
    TransformerBlock.forward = original_block_forward

    # ---- Persist captures -------------------------------------------------
    print(f"[capture] saving per-layer files to {out_dir}...", flush=True)
    save_t0 = time.time()
    for layer_idx in sorted(PER_LAYER_CAPTURE.keys()):
        layer_path = out_dir / f"layer_{layer_idx:02d}.pt"
        torch.save(PER_LAYER_CAPTURE[layer_idx], layer_path)
    print(f"[capture] saved {len(PER_LAYER_CAPTURE)} layer files in {time.time() - save_t0:.1f}s", flush=True)

    # ---- Persist meta ----------------------------------------------------
    meta = {
        "problem": args.problem,
        "prompt": prompt,
        "n_layers": args.n_layers,
        "n_decode_tokens": args.n_decode_tokens,
        "max_seq_len": args.max_seq_len,
        "padded_prefill": padded_len,
        "seq_len": seq_len,
        "ref_tokens": ref_tokens,  # length = n_decode_tokens + 1 (includes prefill first token)
        "hf_model_path": args.hf_model,
        "dtype": "float16",
        "op_names": [
            # block-level (in patched_block_forward order)
            "layer_in",
            "attn_out",
            "attn_normed",
            "h_attn",
            "w1_out",
            "w3_out",
            "ff1ff3",
            "w2_out_pre_ar",
            "ff_out",
            "ff_normed",
            "layer_out",
            # attention-level (in patched_attn_forward order)
            "q_pre_norm",
            "k_pre_norm",
            "v_heads",
            "q_post_norm",
            "k_post_norm",
            "q_post_rope",
            "k_post_rope",
            "sdpa_out",
            "wo_out",
        ],
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[capture] DONE — wrote {out_dir}/meta.json", flush=True)


if __name__ == "__main__":
    main()
