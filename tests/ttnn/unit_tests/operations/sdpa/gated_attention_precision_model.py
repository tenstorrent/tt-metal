# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU precision model for Qwen3.6 Gated Attention prefill.

Why this exists: the accuracy risk of dropping precision on the O(S^2) work in
issue #50475 is *ISL-dependent*. Flash attention accumulates the output over
S/k_chunk chunks, each with a rescale, so error compounds with sequence length.
A PCC measured at S=128 -- the only size ttsim can reach in reasonable time --
says nothing about S=128k, and ttsim cannot answer this because it is a
functional simulator running at a kHz clock, not because of any numerical
limitation.

So the ISL sweep is done here, on CPU, in torch. This models the two effects
that actually dominate:

  1. bfloat8_b block-float quantization of Q/K/V (what QWEN_SDPA_BF8 turns on).
  2. The precision of the flash-attention output accumulator, i.e.
     fp32_dest_acc_en on/off.

Deliberately NOT modeled: the per-fidelity mantissa truncation inside the
Tensix matmul (LoFi/HiFi2/HiFi4 pass counts). That is a microarchitectural
detail this file would have to guess at, and a wrong guess is worse than no
number. Fidelity is swept on real hardware via the parametrization in
test_gated_attention_prefill.py::test_gated_attention_prefill_precision_sweep.

Run standalone:
    python tests/ttnn/unit_tests/operations/sdpa/gated_attention_precision_model.py
"""

import argparse
import math

import torch

QWEN36_NH = 6
QWEN36_NKV = 1
QWEN36_HD = 256


def quantize_bfp8_b(x, block=16):
    """Model Tenstorrent bfloat8_b: 16 datums share an 8-bit exponent, each datum
    keeps a sign and 7 mantissa bits.

    The shared exponent is taken from the block maximum, then every value in the
    block is quantized to a multiple of 2^(e-6). Values far below the block max
    lose precision -- which is exactly the failure mode worth measuring, since
    attention scores within a k-chunk span a wide dynamic range.
    """
    shape = x.shape
    assert shape[-1] % block == 0, f"last dim {shape[-1]} must be a multiple of {block}"
    xb = x.reshape(*shape[:-1], shape[-1] // block, block).to(torch.float32)

    absmax = xb.abs().amax(dim=-1, keepdim=True)
    nonzero = absmax > 0
    e = torch.where(nonzero, torch.floor(torch.log2(absmax.clamp(min=1e-38))), torch.zeros_like(absmax))
    step = torch.pow(2.0, e - 6)
    q = torch.where(nonzero, torch.round(xb / step).clamp(-127, 127) * step, torch.zeros_like(xb))
    return q.reshape(shape)


def to_bf16(x):
    return x.to(torch.bfloat16).to(torch.float32)


def cast_inputs(t, dtype):
    if dtype == "bf16":
        return to_bf16(t)
    if dtype == "bfp8":
        return quantize_bfp8_b(t)
    if dtype == "fp32":
        return t.to(torch.float32)
    raise ValueError(f"unknown dtype {dtype}")


def flash_attention(Q, K, V, scale, k_chunk, acc_dtype):
    """Causal flash attention with an explicit online-softmax accumulator, so the
    accumulator precision (fp32_dest_acc_en) can be varied.

    Q [b, nh, s, d], K/V [b, nh, s, d] (already GQA-expanded). Returns [b, nh, s, d].
    """
    b, nh, s, d = Q.shape
    cast = to_bf16 if acc_dtype == "bf16" else (lambda t: t)

    out = torch.zeros(b, nh, s, d, dtype=torch.float32)
    running_max = torch.full((b, nh, s, 1), -float("inf"), dtype=torch.float32)
    running_sum = torch.zeros(b, nh, s, 1, dtype=torch.float32)
    pos = torch.arange(s)

    for start in range(0, s, k_chunk):
        end = min(start + k_chunk, s)
        k_pos = pos[start:end]

        scores = torch.matmul(Q, K[:, :, start:end, :].transpose(-2, -1)) * scale
        # Causal: a query at position i may not see key positions > i.
        causal = k_pos.view(1, -1) > pos.view(-1, 1)
        scores = scores.masked_fill(causal.view(1, 1, s, end - start), -float("inf"))

        chunk_max = scores.amax(dim=-1, keepdim=True)
        new_max = torch.maximum(running_max, chunk_max)
        # A query row with no visible keys yet stays at -inf; keep it finite so
        # exp() does not produce NaN, and let the zero weight carry the result.
        safe_max = torch.where(torch.isinf(new_max), torch.zeros_like(new_max), new_max)

        rescale = torch.exp(
            torch.where(torch.isinf(running_max), torch.full_like(running_max, -1e30), running_max) - safe_max
        )
        p = torch.exp(scores - safe_max)
        p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)

        # This is the accumulation whose precision we are studying.
        out = cast(out * rescale + torch.matmul(p, V[:, :, start:end, :]))
        running_sum = cast(running_sum * rescale + p.sum(dim=-1, keepdim=True))
        running_max = new_max

    return out / running_sum.clamp(min=1e-30)


def pcc(a, b):
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return float((a @ b) / denom) if denom > 0 else float("nan")


def run_case(s, dtype, acc_dtype, k_chunk, nh=QWEN36_NH, nkv=QWEN36_NKV, d=QWEN36_HD, seed=0):
    torch.manual_seed(seed)
    Q = torch.randn(1, nh, s, d)
    K = torch.randn(1, nkv, s, d)
    V = torch.randn(1, nkv, s, d)
    gate = torch.randn(1, 1, s, nh * d)

    K_rep = K.repeat_interleave(nh // nkv, dim=1)
    V_rep = V.repeat_interleave(nh // nkv, dim=1)
    scale = 1.0 / math.sqrt(d)

    golden = torch.nn.functional.scaled_dot_product_attention(
        Q.double(), K_rep.double(), V_rep.double(), is_causal=True, scale=scale
    ).float()

    got = flash_attention(
        cast_inputs(Q, dtype), cast_inputs(K_rep, dtype), cast_inputs(V_rep, dtype), scale, k_chunk, acc_dtype
    )

    b, _, _, _ = Q.shape
    g_concat = golden.permute(0, 2, 1, 3).reshape(b, 1, s, nh * d) * torch.sigmoid(gate)
    o_concat = got.permute(0, 2, 1, 3).reshape(b, 1, s, nh * d) * torch.sigmoid(gate)

    return {
        "attn_pcc": pcc(golden, got),
        "gated_pcc": pcc(g_concat, o_concat),
        "rmse": float(torch.sqrt(((g_concat - o_concat) ** 2).mean())),
        "n_chunks": (s + k_chunk - 1) // k_chunk,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # The fp64 golden is O(S^2) in memory, so the default stops at 4k. The trend
    # across chunk counts is what matters; extend with --seq-lens when you want
    # the 8k/16k points and have the RAM for them.
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--k-chunk", type=int, default=128)
    ap.add_argument("--heads", type=int, default=2, help="Q heads; fewer keeps the CPU sweep quick")
    ap.add_argument("--head-dim", type=int, default=QWEN36_HD)
    args = ap.parse_args()

    combos = [
        ("bf16", "fp32"),  # today's default: HiFi2 + fp32_dest_acc_en=True
        ("bf16", "bf16"),  # fp32_dest_acc_en=False
        ("bfp8", "fp32"),  # QWEN_SDPA_BF8=1
        ("bfp8", "bf16"),  # both
    ]

    print(f"{'S':>7} {'in':>6} {'acc':>5} {'chunks':>7} {'attn_pcc':>12} {'gated_pcc':>12} {'rmse':>10}")
    for s in args.seq_lens:
        for dtype, acc in combos:
            r = run_case(s, dtype, acc, args.k_chunk, nh=args.heads, nkv=1, d=args.head_dim)
            print(
                f"{s:>7} {dtype:>6} {acc:>5} {r['n_chunks']:>7} "
                f"{r['attn_pcc']:>12.8f} {r['gated_pcc']:>12.8f} {r['rmse']:>10.6f}"
            )
        print()


if __name__ == "__main__":
    main()
