#!/usr/bin/env python3
"""A standalone ACE-Step 1.5 DiT-block op sequence in raw ttnn. **Historical / scaffolding only.**

    export TT_METAL_HOME=<tt-metal>  PYTHONPATH=<tt-metal>
    python models/experimental/ace_step_v15/repro_dit_block_stall.py

No ACE-Step model code, no checkpoints, no goldens -- only raw ttnn ops on random tensors.

======================================================================================
THIS SCRIPT DOES **NOT** REPRODUCE THE BUG IT WAS WRITTEN FOR. IT PASSES.
Use `repro_sdpa_fp32_window_hang.py` instead (~30 lines, reproduces reliably).
======================================================================================

It was built to chase the hang later recorded as TRAP-13, and an earlier version of this
docstring stated a "SYMPTOM" (all ops enqueue, then the first `ttnn.to_torch` never returns)
plus a six-item "what is not the cause" list. **Both were wrong**, and are deleted rather than
corrected in place because nothing in them survived:

* The whole sequence completes in **1.90 s** on a genuinely reset board, sync included.
* Every "eliminated cause" was measured on a card degraded by a previous mid-operation SIGTERM.
  Such a card still opens in a normal ~0.8 s but enqueues ~4x slower and then hangs at the first
  sync -- i.e. it mimics exactly the bug being hunted. `open_device` succeeding does **not** mean
  the board is healthy; reset with `tt-smi -r` before every measurement.

The real cause: `ttnn.transformer.scaled_dot_product_attention` deadlocks when its
`compute_kernel_config` sets `fp32_dest_acc_en=True` **and** `sliding_window_size` is passed.
Either alone is fine.

Why this script misses it -- a genuine near-miss worth understanding: it *does* build a
`fp32_dest_acc_en=True` config (`ckc` below), but only hands it to the **matmuls** via `mm()`.
The SDPA call passes no `compute_kernel_config` at all, so it silently gets the op default and
never hits the bad combination. Adding `compute_kernel_config=ckc` to the SDPA call below makes
this script hang too.

WHAT IS STILL USEFUL HERE
    The `--stop-after` scaffolding: it syncs at a named point instead of at the end, which
    bisects an op *sequence* rather than a single op. Handy for a future
    enqueue-fine-then-stall-at-sync problem. Stages: norm|qkv|heads|qknorm|rope|sdpa|concat|
    out|cross|end. Escapes: `--no-window`, `--plain-matmul`.
    ⚠ One data point per board reset -- a stall's SIGTERM contaminates the next process, so a
    `for stage in ...` loop yields exactly one trustworthy result and then garbage.

ENVIRONMENT
    Wormhole b0 (N150), single chip, 12 GB. tt-metal HEAD as of 2026-08-03.
    bf16 throughout, batch 1, no mesh parallelism (1x1 mesh).

See TRAP-13 in model-bringup/ace_step_1_5/ACE_STEP_1_5_BUGS.md for the full hunt.
"""
import argparse
import atexit
import time

import torch
import ttnn

# ACE-Step 1.5 2B DiT block geometry
HIDDEN = 2048
INTER = 6144  # SwiGLU inner; packed weight is 2 * INTER
NQ, NKV, HEAD_DIM = 16, 8, 128
S = 128  # sequence after patchify (10.24 s of audio at 25 Hz)
ENC_L = 96  # cross-attention context length
WINDOW = 256  # TTNN's parameter is the TOTAL width => |i-j| <= 128
EPS = 1e-6

t0 = time.time()


def mark(m):
    print(f"[{time.time()-t0:7.2f}s] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stop-after",
        default="end",
        help="sync early to bisect: norm|qkv|heads|qknorm|rope|sdpa|concat|out|" "cross|mlp|end",
    )
    ap.add_argument("--no-window", action="store_true", help="drop sliding_window_size")
    ap.add_argument(
        "--plain-matmul", action="store_true", help="use ttnn.linear instead of ttnn.experimental.minimal_matmul"
    )
    args = ap.parse_args()

    dev = ttnn.open_device(device_id=0, l1_small_size=65536)
    mark(f"device open ({type(dev).__name__} {tuple(dev.shape)})")

    # Every `--stop-after` path returns early, so close the device via atexit rather than at
    # the end of main() -- otherwise a bisection run leaves the card half-open and the *next*
    # invocation in the sweep inherits a dirty device.
    _closed = [False]

    def _close():
        if not _closed[0]:
            _closed[0] = True
            mark("closing device")
            ttnn.close_device(dev)

    atexit.register(_close)

    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def up(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

    def mm(x, w, *, fuse_swiglu=False):
        if args.plain_matmul:
            return ttnn.linear(x, w, compute_kernel_config=ckc)
        return ttnn.experimental.minimal_matmul(
            input_tensor=x,
            weight_tensor=w,
            bias_tensor=None,
            config=None,
            fused_activation=None,
            compute_kernel_config=ckc,
            dtype=ttnn.bfloat16,
            fuse_swiglu=fuse_swiglu,
        )

    def norm(x, w):
        return ttnn.experimental.dit_rms_norm_unary_fused(x, weight=w, bias=None, epsilon=EPS)

    def sync(tag, t):
        mark(f"--- SYNC at {tag}: calling ttnn.to_torch{tuple(int(t.shape[i]) for i in range(len(t.shape)))} ---")
        s = time.time()
        h = ttnn.to_torch(t)
        mark(f"*** SYNC OK in {time.time()-s:.2f}s  sum={h.float().sum().item():.4f} ***")
        return h

    torch.manual_seed(0)
    # ---- weights (random; values are irrelevant to the stall) ----
    w_n1 = up(torch.randn(1, HIDDEN) * 0.02)
    w_n2 = up(torch.randn(1, HIDDEN) * 0.02)
    w_n3 = up(torch.randn(1, HIDDEN) * 0.02)
    w_qkv = up(torch.randn(HIDDEN, (NQ + 2 * NKV) * HEAD_DIM) * 0.02)  # 2048 x 4096
    w_o = up(torch.randn(NQ * HEAD_DIM, HIDDEN) * 0.02)  # 2048 x 2048
    w_qn = up(torch.randn(1, HEAD_DIM) * 0.02)
    w_kn = up(torch.randn(1, HEAD_DIM) * 0.02)
    w_cq = up(torch.randn(HIDDEN, NQ * HEAD_DIM) * 0.02)
    w_ck = up(torch.randn(HIDDEN, NKV * HEAD_DIM) * 0.02)  # 2048 x 1024
    w_cv = up(torch.randn(HIDDEN, NKV * HEAD_DIM) * 0.02)  # 2048 x 1024
    w_co = up(torch.randn(NQ * HEAD_DIM, HIDDEN) * 0.02)
    w_ff1 = up(torch.randn(HIDDEN, 2 * INTER) * 0.02)  # 2048 x 12288
    w_ff2 = up(torch.randn(INTER, HIDDEN) * 0.02)  # 6144 x 2048
    # ---- activations ----
    x = up(torch.randn(1, 1, S, HIDDEN))
    enc = up(torch.randn(1, 1, ENC_L, HIDDEN))
    shift = up(torch.randn(1, 1, 1, HIDDEN) * 0.1)
    scale = up(torch.randn(1, 1, 1, HIDDEN) * 0.1)
    cos = up(torch.randn(1, 1, S, HEAD_DIM))
    sin = up(torch.randn(1, 1, S, HEAD_DIM))
    mark("weights + activations uploaded")

    # ================================================= self-attention =================
    h = norm(x, w_n1)
    h = ttnn.add(ttnn.multiply(h, scale), shift)
    mark("op: norm + adaLN modulate")
    if args.stop_after == "norm":
        return sync("norm", h)

    qkv = mm(h, w_qkv)
    mark(f"op: qkv matmul -> {tuple(int(qkv.shape[i]) for i in range(len(qkv.shape)))}")
    if args.stop_after == "qkv":
        return sync("qkv", qkv)

    q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=NQ, num_kv_heads=NKV, transpose_k_heads=False)
    mark("op: nlp_create_qkv_heads")
    if args.stop_after == "heads":
        return sync("heads", q)

    q = norm(q, w_qn)
    k = norm(k, w_kn)
    mark("op: per-head QK-RMSNorm")
    if args.stop_after == "qknorm":
        return sync("qknorm", q)

    q = ttnn.experimental.rotary_embedding_hf(q, cos, sin, is_decode_mode=False)
    k = ttnn.experimental.rotary_embedding_hf(k, cos, sin, is_decode_mode=False)
    mark("op: rotary_embedding_hf")
    if args.stop_after == "rope":
        return sync("rope", q)

    kw = dict(is_causal=False, scale=HEAD_DIM**-0.5)
    if not args.no_window:
        kw["sliding_window_size"] = WINDOW
    attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, **kw)
    mark(f"op: SDPA window={kw.get('sliding_window_size')}")
    if args.stop_after == "sdpa":
        return sync("sdpa", attn)

    merged = ttnn.experimental.nlp_concat_heads(attn)
    mark("op: nlp_concat_heads")
    if args.stop_after == "concat":
        return sync("concat", merged)

    x = ttnn.add(x, mm(merged, w_o))
    mark("op: to_out + residual")
    if args.stop_after == "out":
        return sync("out", x)

    # ================================================= cross-attention ================
    # NOTE: nlp_create_qkv_heads rejects `input_kv` whose seq_len differs from Q
    # ("KV tensor seq_len dim must be same as Q tensor seq_len"), which is always the case for
    # cross-attention (S=128 vs enc_L=96). So Q, K and V are each split separately with
    # num_kv_heads=0 -- this mirrors what the real block does.
    hc = norm(x, w_n2)
    cq = mm(hc, w_cq)  # (1,1,S,2048)
    ck = mm(enc, w_ck)  # (1,1,enc_L,1024)
    cv = mm(enc, w_cv)  # (1,1,enc_L,1024)
    cq = ttnn.experimental.nlp_create_qkv_heads(cq, num_heads=NQ, num_kv_heads=0, transpose_k_heads=False)[0]
    ck = ttnn.experimental.nlp_create_qkv_heads(ck, num_heads=NKV, num_kv_heads=0, transpose_k_heads=False)[0]
    cv = ttnn.experimental.nlp_create_qkv_heads(cv, num_heads=NKV, num_kv_heads=0, transpose_k_heads=False)[0]
    cq = norm(cq, w_qn)
    ck = norm(ck, w_kn)
    cattn = ttnn.transformer.scaled_dot_product_attention(
        cq, ck, cv, is_causal=False, scale=HEAD_DIM**-0.5
    )  # NO window on cross-attn
    x = ttnn.add(x, mm(ttnn.experimental.nlp_concat_heads(cattn), w_co))
    mark("op: cross-attention block")
    if args.stop_after == "cross":
        return sync("cross", x)

    # ================================================= SwiGLU MLP =====================
    hm = norm(x, w_n3)
    hm = ttnn.add(ttnn.multiply(hm, scale), shift)
    hm = mm(hm, w_ff1, fuse_swiglu=True)  # 2048 -> 12288, fused silu(gate)*up
    x = ttnn.add(x, mm(hm, w_ff2))  # 6144 -> 2048
    mark("op: SwiGLU MLP + residual")

    mark("=== ALL OPS ENQUEUED WITHOUT ERROR ===")
    sync("end", x)
    mark("=== COMPLETED — did not reproduce ===")
    _close()


if __name__ == "__main__":
    main()
