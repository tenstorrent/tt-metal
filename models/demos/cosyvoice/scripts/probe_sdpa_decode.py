# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Can the AR decode step use flash attention after all?

`03_plan.md` P5 says relative-position attention needs a new SDPA variant --
"~1500+ LOC, high risk, propose last" -- and PERF.md says the flow's estimator takes
`ttnn.transformer.scaled_dot_product_attention` as a drop-in *"unlike the AR decoder"*.
Both readings assume the rel-pos term puts the decoder outside what a fused kernel can
express.

Reading the device op says otherwise. `sdpa_decode_device_operation.cpp:111` allows an
`attn_mask` **whenever `is_causal=False`**, shaped `[B, 1, n_heads, k_len]`, added to the
scores before the softmax. At `T = 1` ESPnet's `(q+u)K^T + (q+v)P^T` has exactly that
shape: the positional half is a per-head row vector over the key axis. So the term this
model was said to need a new kernel for is **already an additive bias**, and the padding
mask folds into the same tensor.

It lands on 61 % of runtime. Every other item in `03_plan.md` targets the vocoder, which
is 3 %.

Whether it *pays* is the open question, and `F35` is the reason to ask rather than
assume: fusing QKV took the flow 1.075 -> 0.719 s and the decode step 8.29 -> 8.31 ms.
Same change, opposite outcomes, because op count is a proxy for cost and not the cost.
Here the arithmetic is unpromising on its face -- four ops out, three in -- and the case
rests entirely on the `[1, 16, 1, W]` score matrix never being written.

    python3 models/demos/cosyvoice/scripts/probe_sdpa_decode.py

Arm A is what ships today, transcribed from `flow/encoder.py:441-505`. Arm B is the same
arithmetic through `scaled_dot_product_attention_decode`. Both are timed traced, at 14
invocations per replay -- one AR layer each -- so a millisecond here is a millisecond per
token.

**Both arms are scored against a torch golden, not against each other.** Arm-vs-arm PCC
says the two disagree without saying which is wrong, and the first run of this probe
burned a cycle on that: it read a mismatch as a broadcast bug when the open question was
the *convention* -- whether the kernel computes `QK^T*scale + M` or `(QK^T + M)*scale`.
The two need different bias tensors and only one of them is right. A third number
settles it, the same way F45's width control did.
"""
from __future__ import annotations

import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

H = 16  # heads
DK = 64  # head dim
LAYERS = 14  # AR decoder depth -- one replay is a whole token's attention
REPS = 5
# 384 is the shipped key width, 448 the in-place path's (max_len + 2 tiles of scratch).
WIDTHS = (384, 448)


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def main() -> int:
    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=134217728)
    cc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    scale = DK**-0.5

    def dev(v):
        return ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    for w in WIDTHS:
        print(f"\n  === key width {w} ({w // 32} tiles) " + "=" * 28)
        torch.manual_seed(0)
        # `qu`/`qv` are q + bias_u / q + bias_v; the two adds that build them are common
        # to both arms and sit outside the comparison. `bd` is (q+v)(W_p p)^T, already
        # sliced to the window.
        qu_t = torch.randn(1, H, 1, DK) * 0.1
        k_t = torch.randn(1, H, w, DK) * 0.1
        v_t = torch.randn(1, H, w, DK) * 0.1
        bd_t = torch.randn(1, H, 1, w) * 0.1
        # Padding mask: the last 32 slots are unfilled, as they are mid-utterance.
        mask_t = torch.zeros(1, 1, 1, w)
        mask_t[..., w - 32 :] = -1e4  # bfloat16-safe; -1e9 survives but buys nothing

        gold = (torch.softmax((qu_t @ k_t.transpose(-1, -2) + bd_t) * scale + mask_t, dim=-1) @ v_t).reshape(H, DK)

        qu, k, v, bd, mask = (dev(x) for x in (qu_t, k_t, v_t, bd_t, mask_t))
        # Arm B's bias puts heads on dim 2 -- a **tiled** axis -- so the mask is expanded
        # per head on the host. Free in the real model: the mask is a per-token constant
        # shared by all 14 layers.
        mask_h = dev(mask_t.expand(1, 1, H, w).contiguous())
        # Decode-mode q is `[1, B, n_heads, d]` and the bias `[B, 1, n_heads, W]`, so
        # both need the head axis moved off dim 1. Built on the **host** here, so that a
        # wrong answer cannot be blamed on `ttnn.permute` moving a length-1 tiled axis to
        # a length-16 one -- an exotic enough re-tile to be worth ruling out before
        # concluding anything about the kernel. The permute's cost is added back in the
        # timing section, where it belongs.
        bd_p = dev(bd_t.permute(0, 2, 1, 3).contiguous())  # [1, 1, H, W]
        q4 = dev(qu_t.permute(0, 2, 1, 3).contiguous())  # [1, 1, H, DK]
        # `k_chunk_size` follows the op's own tests: the largest power of two dividing
        # the key length, capped at 128. 384 -> 128, 448 -> 64. `q_chunk_size` is the
        # padded head count. `exp_approx_mode=False` because accuracy is the gate here.
        kc = min(128, max(2**i for i in range(1, 10) if w % (2**i) == 0))
        prog = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            q_chunk_size=32,
            k_chunk_size=kc,
            exp_approx_mode=False,
        )
        print(f"    k_chunk_size {kc}")

        # ---- Arm A: the explicit chain that ships --------------------------------
        def body_a():
            out = None
            for _ in range(LAYERS):
                ac = ttnn.matmul(qu, k, transpose_b=True, compute_kernel_config=cc)
                raw = ttnn.add(ac, bd)
                ttnn.deallocate(ac)
                attn = ttnn.scale_mask_softmax(raw, scale, mask)
                ttnn.deallocate(raw)
                ctx = ttnn.matmul(attn, v, compute_kernel_config=cc)
                ttnn.deallocate(attn)
                if out is not None:
                    ttnn.deallocate(out)
                out = ctx
            return out

        a_out = body_a()
        print(
            f"    arm A (explicit chain)          vs golden  {pcc(ttnn.to_torch(a_out).float().reshape(H, DK), gold):.10f}"
        )

        # ---- which bias convention does the kernel use? --------------------------
        # (name, scale passed to sdpa, factor on bd, factor on mask)
        candidates = (
            ("QK^T*s + M", scale, scale, 1.0),
            ("(QK^T + M)*s", scale, 1.0, 1.0 / scale),
            ("QK^T + M (s=1)", 1.0, scale, 1.0),
        )
        winner = None
        for name, sdpa_scale, bd_f, mask_f in candidates:
            bd_c = ttnn.multiply(bd_p, bd_f)
            mask_c = ttnn.multiply(mask_h, mask_f)
            bias = ttnn.add(bd_c, mask_c)
            ttnn.deallocate(bd_c)
            ttnn.deallocate(mask_c)
            if name == "QK^T + M (s=1)":  # q must carry the scale instead
                q_use = ttnn.multiply(q4, scale)
            else:
                q_use = q4
            try:
                out = ttnn.transformer.scaled_dot_product_attention_decode(
                    q_use,
                    k,
                    v,
                    is_causal=False,
                    attn_mask=bias,
                    scale=sdpa_scale,
                    program_config=prog,
                    compute_kernel_config=cc,
                )
            except Exception as exc:  # noqa: BLE001 -- report, do not raise
                print(f"    sdpa_decode, {name:<16} FAILED: {str(exc)[:90]}")
                ttnn.deallocate(bias)
                if q_use is not q4:
                    ttnn.deallocate(q_use)
                continue
            p = pcc(ttnn.to_torch(out).float().reshape(H, DK), gold)
            print(f"    sdpa_decode, {name:<16}   vs golden  {p:.10f}")
            ttnn.deallocate(out)
            ttnn.deallocate(bias)
            if q_use is not q4:
                ttnn.deallocate(q_use)
            if winner is None or p > winner[1]:
                winner = (name, p, sdpa_scale, bd_f, mask_f)

        if winner is None or winner[1] < 0.99:
            print(f"    -> no convention reaches PCC 0.99 (best {winner[1]:.6f} via {winner[0]}); not timing.")
            for t in (qu, k, v, bd, mask, mask_h, bd_p, q4, a_out):
                ttnn.deallocate(t)
            continue
        name, best_p, sdpa_scale, bd_f, mask_f = winner
        print(f"    -> convention: {name}, PCC {best_p:.10f}")

        # ---- two controls, one variable each ------------------------------------
        # The first run of this probe got PCC 0.01-0.77 and the fix changed *two*
        # things: the chunk size, and where the decode-layout tensors were built. That
        # is the mistake this repo's notes keep warning about, so each is now backed out
        # on its own.
        def _try(label, q_in, bd_in, chunk):
            pcfg = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                q_chunk_size=32,
                k_chunk_size=chunk,
                exp_approx_mode=False,
            )
            bias = ttnn.add(ttnn.multiply(bd_in, bd_f), ttnn.multiply(mask_h, mask_f))
            try:
                out = ttnn.transformer.scaled_dot_product_attention_decode(
                    q_in,
                    k,
                    v,
                    is_causal=False,
                    attn_mask=bias,
                    scale=sdpa_scale,
                    program_config=pcfg,
                    compute_kernel_config=cc,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"    control: {label:<28} FAILED: {str(exc)[:70]}")
                ttnn.deallocate(bias)
                return
            print(f"    control: {label:<28}   vs golden  {pcc(ttnn.to_torch(out).float().reshape(H, DK), gold):.10f}")
            ttnn.deallocate(out)
            ttnn.deallocate(bias)

        _try("k_chunk_size 32, host layout", q4, bd_p, 32)
        q4_dev = ttnn.permute(qu, (0, 2, 1, 3))
        bd_dev = ttnn.permute(bd, (0, 2, 1, 3))
        _try(f"k_chunk_size {kc}, ttnn.permute", q4_dev, bd_dev, kc)

        # Pre-scale outside the loop: in the real model both factors fold into weights
        # the host already owns, so the per-layer cost is the one `add` below.
        mask_fixed = ttnn.multiply(mask_h, mask_f)

        # Arm B pays for the two permutes into decode layout on every layer, because the
        # real model gets `q` from the QKV split and `bd` from a matmul, both with heads
        # on dim 1. Building them on the host was an isolation device for the PCC check,
        # not something the model could do. Charging them here is what makes the timing
        # a like-for-like against arm A.
        def body_b():
            out = None
            for _ in range(LAYERS):
                q_l = ttnn.permute(qu, (0, 2, 1, 3))
                bd_l = ttnn.permute(bd, (0, 2, 1, 3))
                bias = ttnn.add(ttnn.multiply(bd_l, bd_f), mask_fixed)
                ttnn.deallocate(bd_l)
                ctx = ttnn.transformer.scaled_dot_product_attention_decode(
                    q_l,
                    k,
                    v,
                    is_causal=False,
                    attn_mask=bias,
                    scale=sdpa_scale,
                    program_config=prog,
                    compute_kernel_config=cc,
                )
                ttnn.deallocate(bias)
                ttnn.deallocate(q_l)
                if out is not None:
                    ttnn.deallocate(out)
                out = ctx
            return out

        # ---- timing ---------------------------------------------------------------
        results = {}
        for label, body, held0 in (("explicit chain", body_a, a_out), ("sdpa_decode", body_b, None)):
            if held0 is not None:
                ttnn.deallocate(held0)
            ttnn.deallocate(body())  # warm: every program must compile before capture
            ttnn.synchronize_device(device)
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            held = body()
            ttnn.end_trace_capture(device, tid, cq_id=0)
            best = None
            for _ in range(REPS):
                t0 = time.perf_counter()
                ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
                # Best of N: a slow replay is host noise, not a property of the arm.
                best = min(best or 1e9, time.perf_counter() - t0)
            results[label] = best * 1e3
            ttnn.release_trace(device, tid)
            ttnn.deallocate(held)

        base = results["explicit chain"]
        print(f"\n    {'arm':<18}{'ms / 14 layers':>16}{'vs A':>9}")
        for label, ms in results.items():
            ratio = "--" if label == "explicit chain" else f"{base / ms:.2f}x"
            print(f"    {label:<18}{ms:>16.3f}{ratio:>9}")
        print(f"    saved per token: {base - results['sdpa_decode']:.3f} ms")

        for t in (qu, k, v, bd, mask, mask_h, bd_p, q4, q4_dev, bd_dev, mask_fixed):
            ttnn.deallocate(t)

    ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
