# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host reference of the resident DFlash2 drafter in the D-program's structure (M6b), fp32 math on the cached BF8
weights (drafter_weights.dequantized): the ingest (taps -> context K / V) and one layer of the block, with the
attention output injectable (the D-program's teacher-forced mode). Math as bench/dflash2/dflash2_ref.py and
tt/dflash2.py: Qwen3 pre-norm layers (input / post norms, q / k norms, RoPE theta 1e4), two dynamic 2-tap block convs
per layer (the coefficient projection of the sublayer's normed input gives both its in and out convs), SiLU gating."""
import torch

from models.demos.gemma4_31b_qb2.resident_m7.common import HIDDEN, IC, N_CHIPS
from models.demos.gemma4_31b_qb2.resident_m7.drafter_common import (
    D_DOWN,
    D_FC,
    D_GU,
    D_HEAD_DIM,
    D_HP,
    D_KP_ATTN,
    D_KP_MLP,
    D_KV,
    D_LAYERS,
    D_O,
    D_Q,
    D_QKV,
    GROUP,
    GROUPS,
    SECTION,
    TAP_COLS,
)


def rms(x, w, eps):
    x = x.float()
    return w.float() * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def rope(x, pos, theta, hd=D_HEAD_DIM):
    """x [rows, heads, hd] at positions pos [rows] (rotate-half, as HF / the target's sliding tables)"""
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    f = torch.outer(pos.float(), inv)
    emb = torch.cat([f, f], -1)
    cos, sin = emb.cos()[:, None, :], emb.sin()[:, None, :]
    x1, x2 = x[..., : hd // 2], x[..., hd // 2 :]
    return x * cos + torch.cat([-x2, x1], -1) * sin


class DrafterWeights:
    """the cached BF8 weights reassembled to full matrices ([K, N], fp32) and the vectors"""

    def __init__(self, cache):
        from models.demos.gemma4_31b_qb2.resident_m7.drafter_weights import dequantized

        self.vec = torch.load(f"{cache}/vectors.pt")
        self.eps, self.theta = self.vec["eps"], self.vec["theta"]
        self.layers = []
        for i in range(D_LAYERS):
            qkv, o, gu, down, kpa, kpm = (dequantized(cache, e, i) for e in (D_QKV, D_O, D_GU, D_DOWN, D_KP_ATTN, D_KP_MLP))
            self.layers.append(
                {
                    # [H, 32 heads x 256] etc.: chip c's q heads are [8c, 8c+8), its kv heads [4c, 4c+4)
                    "q": torch.cat([m[:, :D_Q] for m in qkv], 1).float(),
                    "k": torch.cat([m[:, D_Q : D_Q + D_KV] for m in qkv], 1).float(),
                    "v": torch.cat([m[:, D_Q + D_KV :] for m in qkv], 1).float(),
                    "o": torch.cat(o, 0).float(),  # [8192, H]
                    "gate": torch.cat([m[:, :IC] for m in gu], 1).float(),
                    "up": torch.cat([m[:, IC:] for m in gu], 1).float(),
                    "down": torch.cat(down, 0).float(),
                    "kp_attn": kpa[0].float(),  # replicated
                    "kp_mlp": kpm[0].float(),
                    **self.vec["layers"][i],
                }
            )
        self.fc_chips = [m.float() for m in dequantized(cache, D_FC)]  # chip c: [5 x TAP_COLS, H]
        self.hp = dequantized(cache, D_HP)[0].float()

    # ---- ingest
    def ctx(self, taps):
        """taps [5, n, H] -> hidden_norm(fc(taps)) [n, H]; the input-sharded fc: chip c's partial over its columns,
        summed over the chips in chip order"""
        n = taps.shape[1]
        acc = torch.zeros(n, HIDDEN)
        for c in range(N_CHIPS):
            xc = torch.cat([taps[t, :, c * TAP_COLS : (c + 1) * TAP_COLS] for t in range(taps.shape[0])], 1).float()
            acc = acc + xc @ self.fc_chips[c]
        return rms(acc, self.vec["hidden_norm"], self.eps)

    def ctx_kv(self, ctx, pos, layer):
        """the context rows' (K, V) of `layer`: K = RoPE(k_norm(ctx Wk)), V = ctx Wv, [n, 16 kv heads, 256]"""
        L = self.layers[layer]
        n = ctx.shape[0]
        k = (ctx.float() @ L["k"]).view(n, -1, D_HEAD_DIM)
        v = (ctx.float() @ L["v"]).view(n, -1, D_HEAD_DIM)
        k = rope(rms(k, L["k_norm"], self.eps), pos, self.theta)
        return k, v

    # ---- the block
    @staticmethod
    def conv(h, delta, base, block):
        """h [B, H] of contiguous blocks of `block` rows; delta [B, 2 taps, GROUPS]; base [2 taps, H]"""
        B = h.shape[0]
        hg = h.float().view(B, GROUPS, GROUP)
        pos = torch.arange(B) % block
        out = torch.zeros_like(hg)
        for tap in range(2):
            shifted = hg if tap == 0 else torch.cat([hg.new_zeros(1, GROUPS, GROUP), hg[:-1]])
            coef = base[tap].float().view(1, GROUPS, GROUP) + delta[:, tap, :, None]
            out = out + shifted * coef * pos.ge(tap).view(-1, 1, 1)
        return out.reshape(B, HIDDEN)

    @staticmethod
    def coefs(h, kp):
        """kernel projection [B, KP_COLS] -> (in delta [B, 2, GROUPS], out delta [B, 2, GROUPS])"""
        k = h.float() @ kp
        sec = torch.stack([k[:, s * SECTION : s * SECTION + GROUPS] for s in range(4)], 1)  # [B, 4, GROUPS]
        return sec[:, 0:2], sec[:, 2:4]

    def layer(self, x, layer, block, attn=None, attn_fn=None):
        """one block layer on rows x [B, H] (contiguous blocks of `block` rows). The attention output (after o_proj,
        before the out conv) is `attn` [B, H] (injected) or attn_fn(xin) -> [B, H]. Returns the intermediates."""
        L = self.layers[layer]
        h = rms(x, L["input_norm"], self.eps)
        din, dout = self.coefs(h, L["kp_attn"])
        xin = self.conv(h, din, L["attn_base"][0], block)
        a = attn if attn is not None else attn_fn(xin)
        x_attn = x.float() + self.conv(a.float(), dout, L["attn_base"][1], block)
        h2 = rms(x_attn, L["post_norm"], self.eps)
        din2, dout2 = self.coefs(h2, L["kp_mlp"])
        m_in = self.conv(h2, din2, L["mlp_base"][0], block)
        act = torch.nn.functional.silu(m_in @ L["gate"]) * (m_in @ L["up"])
        x_out = x_attn + self.conv(act @ L["down"], dout2, L["mlp_base"][1], block)
        return {"h": h, "xin": xin, "x_attn": x_attn, "h2": h2, "x_out": x_out, "din": din, "dout": dout}

    def final(self, x):
        return rms(x, self.vec["final_norm"], self.eps)
