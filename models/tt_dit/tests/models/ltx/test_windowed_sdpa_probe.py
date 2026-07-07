# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
L4 feasibility PROBE (CPU / torch-only, no device): does a training-free 3D-local
window mask preserve the output of LTX Stage-2 video self-attention?

This does NOT touch ttnn or the device. It answers the *quality* question that
gates whether building a compute-skipping windowed/sliding-tile SDPA kernel is
worth it. The existing ttnn.transformer.windowed_scaled_dot_product_attention is
dense-compute + block-diagonal mask (verified in source), so it yields no speedup;
a real lever needs a NEW kernel. Before spending that effort we measure: at the
real Stage-2 attention shape, how much attention output PCC survives a local
window, and what sparsity (=> ideal speedup) each window buys.

Run:
  ./python_env/bin/python models/tt_dit/tests/models/ltx/test_windowed_sdpa_probe.py

Honesty note baked in: with random Q/K the attention has no spatial structure, so
windowed PCC is a pessimistic lower bound. We therefore ALSO run a
spatially-smooth synthetic (nearby tokens correlated, mimicking real video latents)
as an optimistic bound. The true model number needs CAPTURED Q/K/V from a real
denoise step (see the capture hook printed at the end).
"""

import math

import torch

# ---- Real LTX Stage-2 video self-attention geometry --------------------------
F, H, W = 19, 34, 60  # latent grid: (145-1)//8+1 frames, 1088/32, 1920/32
N_LOGICAL = F * H * W  # 38760 real video tokens (padded to 38912 across SP=8)
TOKENS_PER_FRAME = H * W  # 2040
HEAD_DIM = 128
SP = 8
Q_PER_CHIP = 38912 // SP  # 4864 (ring gathers full K; Q is the local shard)
N_HEADS_PROBE = 2  # subset of the 8 local heads; PCC is per-head, avg reported
SCALE = HEAD_DIM**-0.5

torch.manual_seed(10)


def frame_major_coords(n):
    """token idx -> (f,h,w) under frame-major f*H*W + h*W + w layout."""
    f = n // TOKENS_PER_FRAME
    r = n % TOKENS_PER_FRAME
    return f, r // W, r % W


def make_qkv(locality):
    """Synthesize Q/K/V [heads, N, d].
    locality=0.0 -> pure random (no spatial structure, pessimistic).
    locality>0   -> add a smooth low-freq field over (f,h,w) so neighbours
                    correlate (optimistic, mimics real latents).
    """
    idx = torch.arange(N_LOGICAL)
    f = (idx // TOKENS_PER_FRAME).float()
    h = ((idx % TOKENS_PER_FRAME) // W).float()
    w = (idx % W).float()
    q = torch.randn(N_HEADS_PROBE, N_LOGICAL, HEAD_DIM)
    k = torch.randn(N_HEADS_PROBE, N_LOGICAL, HEAD_DIM)
    v = torch.randn(N_HEADS_PROBE, N_LOGICAL, HEAD_DIM)
    if locality > 0:
        # a handful of smooth spatial modes shared by Q and K -> local similarity
        modes = []
        for af, ah, aw in [(1, 2, 3), (2, 1, 2), (0, 3, 1), (1, 1, 4)]:
            phase = 2 * math.pi * (af * f / F + ah * h / H + aw * w / W)
            modes.append(torch.cos(phase))
            modes.append(torch.sin(phase))
        smooth = torch.stack(modes, dim=1)  # [N, 8]
        proj = torch.randn(smooth.shape[1], HEAD_DIM)
        field = smooth @ proj  # [N, d] shared low-freq structure
        q = q + locality * field.unsqueeze(0)
        k = k + locality * field.unsqueeze(0)
    return q, k, v


def build_window_mask(q_start, q_count, tf, sh, sw):
    """Boolean keep-mask [q_count, N_LOGICAL]: local Q token attends to K within a
    3D box of half-widths (tf frames, sh rows, sw cols). tf/sh/sw = None -> full."""
    qg = torch.arange(q_start, q_start + q_count)
    qf, qh, qw = frame_major_coords(qg)  # [q_count]
    kg = torch.arange(N_LOGICAL)
    kf, kh, kw = frame_major_coords(kg)  # [N]
    keep = torch.ones(q_count, N_LOGICAL, dtype=torch.bool)
    if tf is not None:
        keep &= (qf[:, None] - kf[None, :]).abs() <= tf
    if sh is not None:
        keep &= (qh[:, None] - kh[None, :]).abs() <= sh
    if sw is not None:
        keep &= (qw[:, None] - kw[None, :]).abs() <= sw
    return keep


def dense_out(q, k, v, q_start, q_count):
    """Full attention for the local Q shard over all K. Returns [heads, q_count, d]."""
    ql = q[:, q_start : q_start + q_count, :]
    scores = torch.matmul(ql, k.transpose(1, 2)) * SCALE  # [heads, q_count, N]
    return torch.matmul(torch.softmax(scores, dim=-1), v)


def windowed_out(q, k, v, q_start, q_count, keep):
    ql = q[:, q_start : q_start + q_count, :]
    scores = torch.matmul(ql, k.transpose(1, 2)) * SCALE
    scores = scores.masked_fill(~keep.unsqueeze(0), float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v)


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return float((a @ b) / denom) if denom > 0 else 0.0


WINDOWS = [
    # (name, tf, sh, sw)
    ("T=all, S=full (DENSE sanity)", None, None, None),
    ("T=+-1f, S=full", 1, None, None),
    ("T=+-2f, S=full", 2, None, None),
    ("T=all, S=+-8x14", None, 8, 14),
    ("T=+-2f, S=+-8x14", 2, 8, 14),
    ("T=+-1f, S=+-4x7 (tight box)", 1, 4, 7),
    ("T=+-3f, S=+-12x20", 3, 12, 20),
]


def run(locality, label):
    q, k, v = make_qkv(locality)
    q_start = 3 * Q_PER_CHIP  # middle chip (chip 3), avoids sequence-edge bias
    q_count = Q_PER_CHIP
    ref = dense_out(q, k, v, q_start, q_count)
    print(
        f"\n=== {label} (locality={locality}) | Q shard [{q_start}:{q_start+q_count}] "
        f"({q_count} tok) over {N_LOGICAL} K, {N_HEADS_PROBE} heads, d={HEAD_DIM} ==="
    )
    print(f"{'window':<34}{'kept%':>8}{'ideal_x':>9}{'PCC_out':>10}")
    for name, tf, sh, sw in WINDOWS:
        keep = build_window_mask(q_start, q_count, tf, sh, sw)
        kept_frac = keep.float().mean().item()
        if tf is None and sh is None and sw is None:
            out = ref
        else:
            out = windowed_out(q, k, v, q_start, q_count, keep)
        p = pcc(ref, out)
        ideal_x = 1.0 / kept_frac if kept_frac > 0 else float("inf")
        print(f"{name:<34}{kept_frac*100:>7.2f}%{ideal_x:>8.2f}x{p:>10.4f}")


if __name__ == "__main__":
    print("LTX Stage-2 video self-attn windowed-mask feasibility probe (torch/CPU)")
    print(f"grid F={F} H={H} W={W}  N_logical={N_LOGICAL}  tokens/frame={TOKENS_PER_FRAME}")
    run(0.0, "PESSIMISTIC: random Q/K (no spatial structure)")
    run(1.5, "OPTIMISTIC: spatially-smooth synthetic Q/K")
    print(
        "\nNEXT: replace synthetic Q/K/V with CAPTURED tensors from a real denoise "
        "step to get the true number. Capture hook: in attention_ltx.py forward, "
        "dump q_BHNE/k_BHNE/v_BHNE (post-RoPE, post-gather) for attn1 at one block/"
        "timestep to .pt, load here in place of make_qkv()."
    )
