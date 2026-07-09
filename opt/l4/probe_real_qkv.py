# SPDX-License-Identifier: Apache-2.0
"""
L4 DECISIVE probe on REAL captured activations.

Loads the post-RoPE attn1 (video self-attn) Q/K/V captured from a live LTX Stage-2
denoise step (tmp/real_attn_qkv.pt, produced by capture_qkv_plugin.py) and measures,
per candidate 3D-local window, the output PCC of windowed-masked attention vs dense
attention -- the SAME question the synthetic probe answered, now on real learned
activations that carry real spatial locality.

Also reports the fraction of dense softmax mass that lands INSIDE each window (the
mechanistic driver of PCC), and per-head PCC spread.

Run:  ./python_env/bin/python tmp/probe_real_qkv.py [path.pt]
"""
import sys

import torch

# ---- Real LTX Stage-2 video self-attention geometry --------------------------
F, H, W = 19, 34, 60
N_LOGICAL = F * H * W  # 38760 real tokens
TOKENS_PER_FRAME = H * W  # 2040
HEAD_DIM = 128
SP = 8
Q_PER_CHIP = 38912 // SP  # 4864
SCALE = HEAD_DIM**-0.5

PT = sys.argv[1] if len(sys.argv) > 1 else "tmp/real_attn_qkv.pt"


def frame_major_coords(n):
    f = n // TOKENS_PER_FRAME
    r = n % TOKENS_PER_FRAME
    return f, r // W, r % W


def build_window_mask(q_start, q_count, tf, sh, sw):
    qg = torch.arange(q_start, q_start + q_count)
    qf, qh, qw = frame_major_coords(qg)
    kg = torch.arange(N_LOGICAL)
    kf, kh, kw = frame_major_coords(kg)
    keep = torch.ones(q_count, N_LOGICAL, dtype=torch.bool)
    if tf is not None:
        keep &= (qf[:, None] - kf[None, :]).abs() <= tf
    if sh is not None:
        keep &= (qh[:, None] - kh[None, :]).abs() <= sh
    if sw is not None:
        keep &= (qw[:, None] - kw[None, :]).abs() <= sw
    return keep


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return float((a @ b) / denom) if denom > 0 else 0.0


WINDOWS = [
    # (name, tf, sh, sw)
    ("T=all,  S=full  (DENSE sanity)", None, None, None),
    ("T=+-4f, S=full", 4, None, None),
    ("T=+-3f, S=full", 3, None, None),
    ("T=+-2f, S=full", 2, None, None),
    ("T=+-1f, S=full", 1, None, None),
    ("T=all,  S=+-12x20", None, 12, 20),
    ("T=all,  S=+-8x14", None, 8, 14),
    ("T=+-3f, S=+-12x20", 3, 12, 20),
    ("T=+-2f, S=+-8x14", 2, 8, 14),
    ("T=+-1f, S=+-4x7 (tight box)", 1, 4, 7),
]


def main():
    print(f"LTX Stage-2 attn1 REAL-ACTIVATION windowed-mask probe  (load {PT})")
    blob = torch.load(PT, map_location="cpu")
    meta = blob.get("meta", {})
    print(f"meta: {meta}")
    q = blob["q"].float()
    k = blob["k"].float()
    v = blob["v"].float()
    print(f"loaded q{tuple(q.shape)} k{tuple(k.shape)} v{tuple(v.shape)} dtype(saved)")

    # [1, Hh, Ntot, d] -> [Hh, Ntot, d]; slice to the real (frame-major) tokens.
    q = q[0][:, :N_LOGICAL, :]
    k = k[0][:, :N_LOGICAL, :]
    v = v[0][:, :N_LOGICAL, :]
    Hh = q.shape[0]
    print(f"heads={Hh}  N_logical={N_LOGICAL}  d={HEAD_DIM}  grid F={F} H={H} W={W}\n")

    # NaN / scale sanity on real data
    print(
        f"sanity: q.absmean={q.abs().mean():.4f} k.absmean={k.abs().mean():.4f} "
        f"v.absmean={v.abs().mean():.4f} nan={bool(torch.isnan(q).any() or torch.isnan(k).any())}\n"
    )

    q_start = 3 * Q_PER_CHIP  # middle chip (chip 3): tokens [14592:19456], all real
    q_count = Q_PER_CHIP
    ql = q[:, q_start : q_start + q_count, :]  # [Hh, q_count, d]

    # dense reference: softmax(ql @ k^T * scale) @ v  over ALL real K
    scores = torch.matmul(ql, k.transpose(1, 2)) * SCALE  # [Hh, q_count, N]
    probs = torch.softmax(scores, dim=-1)
    ref = torch.matmul(probs, v)  # [Hh, q_count, d]

    print(f"Q shard [{q_start}:{q_start+q_count}] ({q_count} queries) over {N_LOGICAL} K, " f"{Hh} heads, d={HEAD_DIM}")
    print(f"{'window':<32}{'kept%':>8}{'ideal_x':>9}{'mass_in':>9}{'PCC_avg':>9}{'PCC_min':>9}{'PCC_max':>9}")
    for name, tf, sh, sw in WINDOWS:
        keep = build_window_mask(q_start, q_count, tf, sh, sw)  # [q_count, N]
        kept_frac = keep.float().mean().item()
        if tf is None and sh is None and sw is None:
            out = ref
            mass_in = 1.0
        else:
            masked = scores.masked_fill(~keep.unsqueeze(0), float("-inf"))
            wprobs = torch.softmax(masked, dim=-1)
            out = torch.matmul(wprobs, v)
            # fraction of DENSE softmax mass captured inside the window (mechanistic driver)
            mass_in = (probs * keep.unsqueeze(0)).sum(-1).mean().item()
        pccs = [pcc(ref[h], out[h]) for h in range(Hh)]
        p_avg = sum(pccs) / len(pccs)
        ideal_x = 1.0 / kept_frac if kept_frac > 0 else float("inf")
        print(
            f"{name:<32}{kept_frac*100:>7.2f}%{ideal_x:>8.2f}x{mass_in*100:>8.1f}%"
            f"{p_avg:>9.4f}{min(pccs):>9.4f}{max(pccs):>9.4f}"
        )

    print("\nVerdict band = p in [0.30, 0.50] (ideal 2-3.3x). Gate = PCC >= 0.85.")


if __name__ == "__main__":
    main()
