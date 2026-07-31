# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Voxtral-TTS Block 1 (3.4B AR backbone) on TTNN, ours rather than `tt_transformers`.

WORK IN PROGRESS -- replaces `ttnn_voxtral_backbone.py` once it validates. Increment order and the
rationale for owning this are in STATUS.md's Block 1 section; each increment gates on PCC against
`reference/voxtral_backbone_ref.py` before the next. Done so far: weight load, one prefill layer.

Mirrors `voxtral_backbone_ref._layer` op-for-op. Structurally this is Block 2's `_block` plus
three things -- RoPE, a causal mask, and (later) a KV cache -- so what Block 2 already proved
carries over unchanged: the row fold, k+v fused into one weight, `nlp_create_qkv_heads`, HiFi4
with fp32_dest_acc_en, bf16 activations.

TWO THINGS THAT ARE NOT LIKE BLOCK 2:

1. ROPE, AND WE PERMUTE THE WEIGHTS TO AVOID THE HARD VERSION. The checkpoint is Mistral-native, so
   q/k are stored for INTERLEAVED-pair rotation (r1,i1,r2,i2,...). Applying that on device means
   shuffling even/odd lanes inside a tile, which is awkward. Instead we permute wq/wk ONCE at load
   time into the half-split layout (r1,r2,...,i1,i2,...) and then apply the easy `rotate_half` form.
   The two are equivalent, and the permute is the same one `scripts/export_backbone_hf` uses and
   asserts bit-exact. Getting this wrong does NOT raise -- it produces fluent nonsense.

2. `n_heads * head_dim` (4096) != `dim` (3072), so wq and wo are NOT square. Anything assuming
   dim throughout will be silently wrong on shapes that happen to broadcast.
"""

import torch
import ttnn

from models.experimental.voxtral_tts.reference.voxtral_backbone_ref import load_backbone_state
from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
    DEFAULT_CKPT,
    DIM,
    HEAD_DIM,
    HIDDEN_DIM,
    N_HEADS,
    N_KV_HEADS,
    N_LAYERS,
    NORM_EPS,
    ROPE_THETA,
)

SCALE = HEAD_DIM**-0.5
Q_WIDTH = N_HEADS * HEAD_DIM          # 4096, deliberately != DIM
KV_WIDTH = N_KV_HEADS * HEAD_DIM      # 1024

COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
DTYPE = ttnn.bfloat16
WEIGHT_DTYPE = None                   # None = same as DTYPE; see STATUS.md before changing


def interleaved_to_halfsplit(t, n_heads):
    """Mistral-native (interleaved-pair) q/k weight -> half-split layout, so `rotate_half` applies.

    Identical to `scripts/export_backbone_hf.meta_to_hf_permute`, which asserts the round trip
    bit-exact. `t` is torch [out, in]; rows are grouped per head.
    """
    d1, d2 = t.shape
    return t.view(n_heads, d1 // n_heads // 2, 2, d2).transpose(1, 2).reshape(d1, d2)


def rope_tables(seq_len, offset=0, head_dim=HEAD_DIM, theta=ROPE_THETA):
    """-> (cos, sin) torch [seq_len, head_dim], each half duplicated for the half-split form.

    `rope_cis` in the reference returns complex [S, d/2] for interleaved pairs; the half-split form
    wants the same angles laid out as [cos(theta), cos(theta)] so one elementwise multiply covers
    both halves.
    """
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    ang = torch.outer(torch.arange(offset, offset + seq_len, dtype=torch.float64), inv)
    return (torch.cat([ang.cos(), ang.cos()], dim=-1).float(),
            torch.cat([ang.sin(), ang.sin()], dim=-1).float())


class TtVoxtralGPT:
    """Block 1 on device. WIP: prefill of `n_layers` layers so far."""

    def __init__(self, device, ckpt_path=DEFAULT_CKPT, dtype=DTYPE, weight_dtype=None,
                 n_layers=N_LAYERS):
        self.device = device
        self.dtype = dtype
        self.n_layers = n_layers
        wd = weight_dtype or WEIGHT_DTYPE or dtype
        w = load_backbone_state(ckpt_path)

        up = lambda t, d: ttnn.from_torch(t.contiguous(), dtype=d, layout=ttnn.TILE_LAYOUT,
                                         device=device)
        vec = lambda t: up(t.reshape(1, 1, -1), dtype)      # norm gammas: no bandwidth, keep bf16
        lin = lambda t: up(t.t(), wd)                       # torch [out,in] -> ttnn wants [in,out]

        self.norm = vec(w["norm"])
        self.layers = []
        for i in range(n_layers):
            p = f"layers.{i}."
            wq = interleaved_to_halfsplit(w[p + "attention.wq"], N_HEADS)
            wk = interleaved_to_halfsplit(w[p + "attention.wk"], N_KV_HEADS)  # n_kv, not n_heads
            self.layers.append({
                "an": vec(w[p + "attention_norm"]),
                "fn": vec(w[p + "ffn_norm"]),
                "wq": lin(wq),
                # k and v fused into one weight -> one matmul, one weight stream (as in Block 2)
                "wkv": lin(torch.cat([wk, w[p + "attention.wv"]], dim=0)),
                "wo": lin(w[p + "attention.wo"]),
                "w1": lin(w[p + "feed_forward.w1"]),
                "w2": lin(w[p + "feed_forward.w2"]),
                "w3": lin(w[p + "feed_forward.w3"]),
            })
        self._assert_shapes()

    def _assert_shapes(self):
        """Cheap guard against a silently wrong load: non-square wq/wo are what bite here."""
        exp = {"wq": (DIM, Q_WIDTH), "wkv": (DIM, 2 * KV_WIDTH), "wo": (Q_WIDTH, DIM),
               "w1": (DIM, HIDDEN_DIM), "w3": (DIM, HIDDEN_DIM), "w2": (HIDDEN_DIM, DIM)}
        for i, L in enumerate(self.layers):
            for k, e in exp.items():
                got = tuple(L[k].shape)[-2:]
                assert got == e, f"layer {i} {k}: expected {e}, got {got}"

    def _rope(self, x, cos, sin):
        """Half-split RoPE on [B, heads, S, head_dim]: x*cos + rotate_half(x)*sin."""
        h = HEAD_DIM // 2
        b, nh, s = x.shape[0], x.shape[1], x.shape[2]   # ttnn.Shape has no Python slicing
        x1 = ttnn.slice(x, [0, 0, 0, 0], [b, nh, s, h])
        x2 = ttnn.slice(x, [0, 0, 0, h], [b, nh, s, HEAD_DIM])
        rot = ttnn.concat([ttnn.neg(x2), x1], dim=-1)
        return ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(rot, sin))

    def _layer(self, x, w, S, cos, sin, mask):
        """x [1,S,3072] -> same. Pre-norm GQA with RoPE + causal mask, then SwiGLU.

        Rows are already folded (batch 1 here), so every linear reads its weights once. Attention is
        the ONLY row-mixing op: it runs on the unfolded [1, heads, S, d] view. Any future row-mixing
        op must go inside that same window -- see ttnn_voxtral_flow._block.
        """
        h = ttnn.rms_norm(x, weight=w["an"], epsilon=NORM_EPS, compute_kernel_config=COMPUTE_CONFIG)
        q = ttnn.linear(h, w["wq"], compute_kernel_config=COMPUTE_CONFIG)
        kv = ttnn.linear(h, w["wkv"], compute_kernel_config=COMPUTE_CONFIG)
        qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads(
            ttnn.reshape(q, [1, 1, S, Q_WIDTH]), ttnn.reshape(kv, [1, 1, S, 2 * KV_WIDTH]),
            num_heads=N_HEADS, num_kv_heads=N_KV_HEADS,
            transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        qh, kh = self._rope(qh, cos, sin), self._rope(kh, cos, sin)   # v carries no RoPE
        rep = N_HEADS // N_KV_HEADS
        kr, vr = ttnn.repeat_interleave(kh, rep, dim=1), ttnn.repeat_interleave(vh, rep, dim=1)
        s = ttnn.matmul(qh, ttnn.transpose(kr, -2, -1), compute_kernel_config=COMPUTE_CONFIG)
        s = ttnn.add(ttnn.multiply(s, SCALE), mask)          # causal: 0 / -inf additive bias
        a = ttnn.softmax(s, dim=-1, numeric_stable=True, compute_kernel_config=COMPUTE_CONFIG)
        a = ttnn.matmul(a, vr, compute_kernel_config=COMPUTE_CONFIG)
        a = ttnn.reshape(ttnn.permute(a, (0, 2, 1, 3)), [1, S, Q_WIDTH])
        x = ttnn.add(x, ttnn.linear(a, w["wo"], compute_kernel_config=COMPUTE_CONFIG))
        h = ttnn.rms_norm(x, weight=w["fn"], epsilon=NORM_EPS, compute_kernel_config=COMPUTE_CONFIG)
        g = ttnn.silu(ttnn.linear(h, w["w1"], compute_kernel_config=COMPUTE_CONFIG))
        u = ttnn.multiply(g, ttnn.linear(h, w["w3"], compute_kernel_config=COMPUTE_CONFIG))
        return ttnn.add(x, ttnn.linear(u, w["w2"], compute_kernel_config=COMPUTE_CONFIG))

    @torch.no_grad()
    def prefill(self, embeds, apply_final_norm=True):
        """embeds torch [1,S,3072] -> hidden torch [1,S,3072]. No cache yet (increment 4)."""
        S = embeds.shape[1]
        cosb, sinb = rope_tables(S)
        up = lambda t, d=None: ttnn.from_torch(t.contiguous(), dtype=d or self.dtype,
                                              layout=ttnn.TILE_LAYOUT, device=self.device)
        cos = up(cosb.reshape(1, 1, S, HEAD_DIM))
        sin = up(sinb.reshape(1, 1, S, HEAD_DIM))
        m = torch.full((S, S), float("-inf")).triu(1).reshape(1, 1, S, S)
        mask = up(m, ttnn.bfloat16)
        x = up(embeds.reshape(1, S, DIM))
        for w in self.layers:
            x = self._layer(x, w, S, cos, sin, mask)
        if apply_final_norm:
            x = ttnn.rms_norm(x, weight=self.norm, epsilon=NORM_EPS,
                              compute_kernel_config=COMPUTE_CONFIG)
        return ttnn.to_torch(x).float().reshape(1, S, DIM)


def main():
    """Increment 2 gate: ONE layer against the reference. A RoPE convention error shows up here."""
    from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as ref
    from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
        causal_bias, pcc, rope_cis)

    dev = ttnn.open_device(device_id=0, l1_small_size=65536)
    try:
        S = 128
        gen = TtVoxtralGPT(dev, n_layers=1)
        w = ref.load_backbone_state()
        torch.manual_seed(0)
        x = torch.randn(1, S, DIM) * 0.02
        exp = ref._layer(x, w, "layers.0.", rope_cis(S, HEAD_DIM, ROPE_THETA),
                         causal_bias(S, torch.float32))
        got = gen.prefill(x, apply_final_norm=False)
        print(f"  [1 layer prefill] PCC {pcc(got, exp):.8f}  "
              f"maxabs {(got - exp).abs().max():.3e}")
        print("  NOTE: random inputs are a pessimistic proxy (trap #12) -- this gate is for WIRING")
        print("  and the RoPE convention. Judge accuracy on real prompts at 26 layers.")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
