# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of the XTTS-v2 conditioning branch (Block 1): conditioning encoder + Perceiver resampler.

Mirrors reference/xtts_cond_ref.py op-for-op, running ENTIRELY on device:

    mel [1,80,T] --init(Conv1d k1)--> [1,T,1024]
      --6x AttentionBlock(GroupNorm32 -> 16-head attn -> proj, residual on the NORMED input)--> enc [1,T,1024]
      --PerceiverResampler(32 latents, depth 2: cross-attn + GEGLU FFN, final RMSNorm)--> perc [1,32,1024]

`perc` is the gpt_cond_latent that conditions the GPT (Block 3). One-shot per speaker (not per token),
so this is fidelity-first: fp32 tensors + fp32 accumulation (HiFi3 + fp32_dest_acc), like the vocoder.

KEY LAYOUT CHOICE: every Conv1d in Block 1 has kernel_size=1, i.e. a matmul over the channel dim, so
the signal stays channels-last [1, S, C] and there is NO conv machinery — just ttnn.linear + attention.
The tortoise encoder's qkv is per-head-interleaved ([h0:q,k,v | h1:q,k,v | ...]) and GEGLU splits at
2730 (not tile-aligned); both are handled by SPLITTING THE WEIGHT MATRICES ON HOST at load, leaving the
device side clean matmuls + standard [1,S,H*d] <-> [1,H,S,d] head reshapes. GroupNorm(32,1024) over
(32 ch/group x T) uses the DRAM-interleaved ttnn.group_norm (grid/mask depend on T -> built + cached per T).

Validate + time vs the CPU reference:
    TT_METAL_HOME=<repo> PYTHONPATH=<repo> python models/experimental/xtts_v2/tt/ttnn_xtts_cond.py
"""

import torch
import ttnn

from models.experimental.xtts_v2.reference.xtts_cond_ref import (
    DIM,
    ENC_BLOCKS,
    ENC_HEADS,
    GN_EPS,
    GN_GROUPS,
    PERC_DEPTH,
    PERC_DIM_HEAD,
    PERC_HEADS,
    load_cond_state,
)
from models.experimental.xtts_v2.reference.xtts_gpt_ref import DEFAULT_CKPT

HEAD_DIM = DIM // ENC_HEADS  # 64
ENC_SCALE = 1.0 / (HEAD_DIM**0.5)  # 1/8; reference applies 1/sqrt(sqrt(64)) to q AND k -> product 1/8
PERC_SCALE = PERC_DIM_HEAD**-0.5  # 1/8
FF_INNER = int(DIM * 4 * 2 / 3)  # 2730 (GEGLU half); ff0 -> 2*FF_INNER = 5460

DTYPE = ttnn.float32  # fidelity-first (one-shot block); fp32 accumulation via COMPUTE_CONFIG
COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi3, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)


class TtConditioningEncoder:
    """On-device conditioning encoder + Perceiver resampler. __call__(mel) -> (enc, perc) matching
    reference.get_style_emb: enc [1,1024,T], perc [1,32,1024] (= gpt_cond_latent)."""

    def __init__(self, device, ckpt_path=DEFAULT_CKPT):
        self.device = device
        enc_w, perc_w = load_cond_state(ckpt_path)
        dev = lambda t: ttnn.from_torch(t.contiguous(), dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
        dev3 = lambda t: ttnn.from_torch(t.reshape(1, 1, -1).contiguous(), dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
        lin = lambda W: dev(W.t())  # reference computes x @ W.t() (W is [out,in]) -> store [in,out]
        # Manual fp32 GroupNorm(32,1024): reduce per group over (32 ch x T). A fixed 0/1 group matrix
        # G[c,g]=1 iff c//32==g does the channel<->group reductions as matmuls (no tiled reshapes):
        # per-group stat = per-channel_stat @ G; broadcast back to channels = per-group_stat @ G.t().
        Gm = torch.zeros(DIM, GN_GROUPS)
        for c in range(DIM):
            Gm[c, c // (DIM // GN_GROUPS)] = 1.0
        self.G = dev(Gm)  # [1024,32]
        self.Gt = dev(Gm.t())  # [32,1024]

        # --- conditioning encoder ---
        self.init_w = lin(enc_w["init.weight"].squeeze(-1))  # Conv1d k1 [1024,80,1] -> matmul [80,1024]
        self.init_b = dev(enc_w["init.bias"])
        # qkv is per-head interleaved: out channel c -> head c//192, role (c%192)//64, dim (c%192)%64.
        # Gather the q/k/v rows on host so each becomes a standard head-contiguous [1024,1024] matmul.
        q_rows = [h * 3 * HEAD_DIM + d for h in range(ENC_HEADS) for d in range(HEAD_DIM)]
        k_rows = [r + HEAD_DIM for r in q_rows]
        v_rows = [r + 2 * HEAD_DIM for r in q_rows]
        self.enc = []
        for i in range(ENC_BLOCKS):
            qkv_w = enc_w[f"attn.{i}.qkv.weight"].squeeze(-1)  # [3072,1024]
            qkv_b = enc_w[f"attn.{i}.qkv.bias"]  # [3072]
            self.enc.append(
                {
                    "gnw": dev3(enc_w[f"attn.{i}.norm.weight"]), "gnb": dev3(enc_w[f"attn.{i}.norm.bias"]),  # [1,1,1024]
                    "qw": lin(qkv_w[q_rows]), "qb": dev(qkv_b[q_rows]),
                    "kw": lin(qkv_w[k_rows]), "kb": dev(qkv_b[k_rows]),
                    "vw": lin(qkv_w[v_rows]), "vb": dev(qkv_b[v_rows]),
                    "pw": lin(enc_w[f"attn.{i}.proj_out.weight"].squeeze(-1)), "pb": dev(enc_w[f"attn.{i}.proj_out.bias"]),
                }
            )

        # --- Perceiver resampler ---
        self.latents = dev(perc_w["latents"].unsqueeze(0))  # [1,32,1024]
        self.perc = []
        for i in range(PERC_DEPTH):
            kv_w = perc_w[f"layers.{i}.0.to_kv.weight"]  # [1024,1024]; chunk(2,-1) on output -> k rows[:512], v rows[512:]
            ff0_w = perc_w[f"layers.{i}.1.0.weight"]  # [5460,1024]; GEGLU: a=rows[:2730], gates=rows[2730:]
            ff0_b = perc_w[f"layers.{i}.1.0.bias"]
            self.perc.append(
                {
                    "toq": lin(perc_w[f"layers.{i}.0.to_q.weight"]),  # [1024,512]
                    "tok": lin(kv_w[:DIM // 2]), "tov": lin(kv_w[DIM // 2 :]),  # [1024,512] each
                    "toout": lin(perc_w[f"layers.{i}.0.to_out.weight"]),  # [512,1024]
                    "ff0a": lin(ff0_w[:FF_INNER]), "ff0ab": dev(ff0_b[:FF_INNER]),  # -> 2730
                    "ff0g": lin(ff0_w[FF_INNER:]), "ff0gb": dev(ff0_b[FF_INNER:]),  # gate -> 2730
                    "ff2": lin(perc_w[f"layers.{i}.1.2.weight"]), "ff2b": dev(perc_w[f"layers.{i}.1.2.bias"]),  # [2730,1024]
                }
            )
        self.norm_gamma = dev(perc_w["norm.gamma"])  # final RMSNorm weight

    # -- manual fp32 GroupNorm(32,1024) over channels-last x [1,T,1024]: two-pass (mean, then var),
    #    per-group reductions done as matmuls with the fixed 0/1 group matrix G. Native ttnn.group_norm
    #    is bf16-only and too lossy for the resampler (perc PCC ~0.98); fp32 here keeps it ~1.0. --
    def _mm(self, a, b):
        return ttnn.matmul(a, b, compute_kernel_config=COMPUTE_CONFIG)

    def _group_norm(self, x, gnw, gnb):  # x [1,T,1024] fp32 TILE -> [1,T,1024]
        n = 1.0 / (x.shape[1] * (DIM // GN_GROUPS))  # 1/(T*32) elements per group
        mean_c = self._mm(ttnn.multiply(self._mm(ttnn.sum(x, dim=1, keepdim=True), self.G), n), self.Gt)  # [1,1,1024]
        xc = ttnn.subtract(x, mean_c)  # centered (H-broadcast over T)
        var_g = ttnn.multiply(self._mm(ttnn.sum(ttnn.multiply(xc, xc), dim=1, keepdim=True), self.G), n)  # [1,1,32]
        inv_c = self._mm(ttnn.rsqrt(ttnn.add(var_g, GN_EPS)), self.Gt)  # [1,1,1024]
        return ttnn.add(ttnn.multiply(ttnn.multiply(xc, inv_c), gnw), gnb)  # normalize + per-channel affine

    # -- attention primitives (fp32, non-causal) --
    def _split_heads(self, x, n_heads):  # [1,S,H*d] -> [1,H,S,d]
        S, d = x.shape[1], x.shape[2] // n_heads
        return ttnn.permute(ttnn.reshape(x, (1, S, n_heads, d)), (0, 2, 1, 3))

    def _merge_heads(self, x):  # [1,H,S,d] -> [1,S,H*d]
        _, H, S, d = x.shape
        return ttnn.reshape(ttnn.permute(x, (0, 2, 1, 3)), (1, S, H * d))

    def _attention(self, q, k, v, scale):  # q[1,H,Sq,d], k/v[1,H,Sk,d] -> [1,H,Sq,d]
        scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1), compute_kernel_config=COMPUTE_CONFIG)  # [1,H,Sq,Sk]
        # numeric_stable=True (max-subtraction) is REQUIRED: the default softmax leaves a structured,
        # dominant-aligned error that the resampler amplifies (perc PCC 0.996 -> 0.9999).
        attn = ttnn.softmax(ttnn.multiply(scores, scale), dim=-1, numeric_stable=True, compute_kernel_config=COMPUTE_CONFIG)
        return ttnn.matmul(attn, v, compute_kernel_config=COMPUTE_CONFIG)

    def _lin(self, x, w, b=None):
        return ttnn.linear(x, w, bias=b, compute_kernel_config=COMPUTE_CONFIG)

    # -- conditioning encoder --
    def encoder(self, mel):  # mel torch [1,80,T] -> device [1,T,1024]
        T = mel.shape[2]
        x = ttnn.from_torch(mel.permute(0, 2, 1).contiguous(), dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=self.device)
        x = self._lin(x, self.init_w, self.init_b)  # [1,T,1024]
        for i in range(ENC_BLOCKS):
            w = self.enc[i]
            xn = self._group_norm(x, w["gnw"], w["gnb"])
            q = self._split_heads(self._lin(xn, w["qw"], w["qb"]), ENC_HEADS)
            k = self._split_heads(self._lin(xn, w["kw"], w["kb"]), ENC_HEADS)
            v = self._split_heads(self._lin(xn, w["vw"], w["vb"]), ENC_HEADS)
            h = self._merge_heads(self._attention(q, k, v, ENC_SCALE))  # [1,T,1024]
            x = ttnn.add(xn, self._lin(h, w["pw"], w["pb"]))  # residual on the NORMED input
        return x

    # -- Perceiver resampler --
    def _perc_attn(self, latents, frames, i):  # latents [1,32,1024], frames [1,T,1024] -> [1,32,1024]
        w = self.perc[i]
        context = ttnn.concat([latents, frames], dim=1)  # cross_attn_include_queries=True
        q = self._split_heads(self._lin(latents, w["toq"]), PERC_HEADS)
        k = self._split_heads(self._lin(context, w["tok"]), PERC_HEADS)
        v = self._split_heads(self._lin(context, w["tov"]), PERC_HEADS)
        out = self._merge_heads(self._attention(q, k, v, PERC_SCALE))  # [1,32,512]
        return self._lin(out, w["toout"])  # [1,32,1024]

    def _perc_ff(self, x, i):  # GEGLU FFN, x [1,32,1024]
        w = self.perc[i]
        gate = ttnn.gelu(self._lin(x, w["ff0g"], w["ff0gb"]))
        h = ttnn.multiply(self._lin(x, w["ff0a"], w["ff0ab"]), gate)  # [1,32,2730]
        return self._lin(h, w["ff2"], w["ff2b"])  # [1,32,1024]

    def perceiver(self, frames):  # frames device [1,T,1024] -> [1,32,1024]
        latents = self.latents
        for i in range(PERC_DEPTH):
            latents = ttnn.add(self._perc_attn(latents, frames, i), latents)
            latents = ttnn.add(self._perc_ff(latents, i), latents)
        # final RMSNorm: F.normalize(x,-1)*sqrt(dim)*gamma == x/sqrt(mean(x^2))*gamma
        return ttnn.rms_norm(latents, epsilon=1e-12, weight=self.norm_gamma, compute_kernel_config=COMPUTE_CONFIG)

    @torch.no_grad()
    def __call__(self, mel):  # mel torch [1,80,T] -> (enc [1,1024,T], perc [1,32,1024])
        enc_dev = self.encoder(mel)
        perc_dev = self.perceiver(enc_dev)
        enc = ttnn.to_torch(enc_dev).permute(0, 2, 1).float()  # [1,1024,T] to match reference get_style_emb
        perc = ttnn.to_torch(perc_dev).float()  # [1,32,1024]
        return enc, perc


def main():
    import time

    from models.experimental.xtts_v2.reference import xtts_cond_ref as ref
    from models.experimental.xtts_v2.reference.xtts_gpt_ref import pcc

    device = ttnn.open_device(device_id=0, l1_small_size=131072)
    try:
        enc_w, perc_w = ref.load_cond_state()
        gen = TtConditioningEncoder(device)
        mel = ref.make_synthetic_mel(n_frames=128)
        ref_enc, ref_perc = ref.get_style_emb(mel, enc_w, perc_w)
        got_enc, got_perc = gen(mel)
        print(f"[cond] enc  PCC vs reference: {pcc(got_enc, ref_enc):.5f}  {tuple(got_enc.shape)}")
        print(f"[cond] perc PCC vs reference: {pcc(got_perc, ref_perc):.5f}  {tuple(got_perc.shape)}")
        for T in (128, 500):
            m = ref.make_synthetic_mel(n_frames=T)
            gen(m)  # warm
            t0 = time.perf_counter()
            gen(m)
            print(f"[cond] mel T={T:4d} -> perc [1,32,1024]: {time.perf_counter() - t0:.3f}s")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
