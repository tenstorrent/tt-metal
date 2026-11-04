# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 17 DFlash2 block drafter: the host (torch, float32 math with the drafter's bf16 rounding points) reference of
the Apache-2.0 drafter's forward, sized by a config dict (the HF config.json of the drafter, or `small_config()` for
host tests on random weights). This is the semantics tt/dflash2.py proved 16 / 16 against the GPU reference
(doc/ttnn/17-mtp-spec-decode/dflash2_design.md): the in-trace kind (tt/dflash2_kind.py) is tested against it.

Drafter (config.json): `L` dense Qwen3-style layers (hidden H, SwiGLU I, nh q heads / nkv kv heads x hd, RMS eps,
RoPE theta, NON-causal block attention), fc [n_taps x H -> H] + hidden_norm (the target taps -> the drafter's context),
own embedding / lm_head / mask embedding, per layer two 2-tap grouped dynamic convs (groups of `group` channels, the
per-row coefficients from kernel_projection [H -> 2 sides x 2 taps x H / group]), a candidate selector (top_k
candidates per draft row, predecessor / successor codebooks [V, rank], hidden_projection [H -> rank], a greedy walk).

Forward of one step (positions are absolute):
  ctx(taps[n, n_taps, H] at positions p0 .. p0+n-1)  -> each layer's context K / V rows (k_norm + RoPE applied to K)
  block(bonus, positions q0 .. q0+K)                  -> hid [K+1, H] (final norm), logits [K+1, V]; rows 1..K are the
                                                        mask rows; the block attends to context rows < n_ctx and to
                                                        all K+1 block rows (non-causal)
  select(hid[1..K], top_k candidates per row)          -> the K drafts (greedy walk from the bonus)
"""

from __future__ import annotations

import math

import torch

BF16 = torch.bfloat16


def small_config(tp: int = 8, n_taps: int = 3, layers: int = 2) -> dict:
    """a host-test drafter: every structural property of the real one (GQA, groups, selector), small sizes"""
    return dict(hidden_size=256, intermediate_size=512, num_hidden_layers=layers, num_attention_heads=8 * (tp // 8 or 1),
                num_key_value_heads=tp, head_dim=32, rms_norm_eps=1e-5, vocab_size=64 * tp * 2,
                rope_parameters=dict(rope_theta=10000.0), target_layer_ids=list(range(n_taps)), mask_token_id=7,
                dflash_config=dict(block_size=8, mask_token_id=7, conv_kernel_size=2, conv_group_size=16,
                                   selector_rank=32, selector_top_k=16))  # fmt: skip


def hf_config(path: str) -> dict:
    import json

    return json.load(open(path))


class Dims:
    def __init__(self, cfg: dict):
        c, dc = cfg, cfg["dflash_config"]
        self.H, self.I, self.L = c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"]
        self.nh, self.nkv, self.hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.eps, self.theta, self.V = c["rms_norm_eps"], c["rope_parameters"]["rope_theta"], c["vocab_size"]
        self.n_taps, self.mask_id = len(c["target_layer_ids"]), dc["mask_token_id"]
        self.group, self.rank, self.top_k, self.K = (
            dc["conv_group_size"],
            dc["selector_rank"],
            dc["selector_top_k"],
            dc["block_size"] - 1,
        )
        self.ng = self.H // self.group
        assert self.nh % self.nkv == 0 and self.H % self.group == 0, (self.nh, self.nkv, self.H, self.group)


def random_weights(cfg: dict, seed: int = 0, scale: float = 0.02) -> dict:
    """HF-named bf16 weights (the safetensors layout: linear weights [out, in]) + the mask embedding"""
    d = Dims(cfg)
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: (torch.randn(*s, generator=g) * scale).to(BF16)  # noqa: E731
    w = {"fc.weight": r(d.H, d.n_taps * d.H), "hidden_norm.weight": (1 + 0.1 * torch.randn(d.H, generator=g)).to(BF16),
         "norm.weight": (1 + 0.1 * torch.randn(d.H, generator=g)).to(BF16), "embed_tokens.weight": r(d.V, d.H),
         "lm_head.weight": r(d.V, d.H), "candidate_selector.hidden_projection.weight": r(d.rank, d.H),
         "candidate_selector.predecessor_codebook": r(d.V, d.rank), "candidate_selector.successor_codebook": r(d.V, d.rank),
         "mask_embedding": r(d.H)}  # fmt: skip
    for i in range(d.L):
        p, a = f"layers.{i}.", f"layers.{i}.self_attn."
        w.update({p + "input_layernorm.weight": (1 + 0.1 * torch.randn(d.H, generator=g)).to(BF16),
                  p + "post_attention_layernorm.weight": (1 + 0.1 * torch.randn(d.H, generator=g)).to(BF16),
                  a + "q_proj.weight": r(d.nh * d.hd, d.H), a + "k_proj.weight": r(d.nkv * d.hd, d.H),
                  a + "v_proj.weight": r(d.nkv * d.hd, d.H), a + "o_proj.weight": r(d.H, d.nh * d.hd),
                  a + "q_norm.weight": (1 + 0.1 * torch.randn(d.hd, generator=g)).to(BF16),
                  a + "k_norm.weight": (1 + 0.1 * torch.randn(d.hd, generator=g)).to(BF16),
                  p + "mlp.gate_proj.weight": r(d.I, d.H), p + "mlp.up_proj.weight": r(d.I, d.H),
                  p + "mlp.down_proj.weight": r(d.H, d.I),
                  p + "attention_conv.base_kernel": (torch.tensor([[1.0, 0.0]]).reshape(1, 2, 1).repeat(2, 1, d.H)
                                                     + 0.05 * torch.randn(2, 2, d.H, generator=g)).to(BF16),
                  p + "attention_conv.kernel_projection.weight": r(4 * d.ng, d.H),
                  p + "mlp_conv.base_kernel": (torch.tensor([[1.0, 0.0]]).reshape(1, 2, 1).repeat(2, 1, d.H)
                                               + 0.05 * torch.randn(2, 2, d.H, generator=g)).to(BF16),
                  p + "mlp_conv.kernel_projection.weight": r(4 * d.ng, d.H)})  # fmt: skip
    return w


def load_weights(drafter_dir: str) -> dict:
    """the drafter's safetensors + mask_embedding.pt (bf16 host tensors, HF names)"""
    import os

    from safetensors import safe_open

    f = safe_open(os.path.join(drafter_dir, "model.safetensors"), framework="pt", device="cpu")
    w = {k: f.get_tensor(k) for k in f.keys()}
    m = torch.load(os.path.join(drafter_dir, "mask_embedding.pt"), map_location="cpu", weights_only=False)
    w["mask_embedding"] = (m["embedding"] if isinstance(m, dict) else m).reshape(-1).to(BF16)
    return w


def meta_weights(drafter_dir: str) -> dict:
    """the drafter's weights as META tensors of the safetensors header's shapes (nothing read but the header; the
    mask embedding's row from config.json's hidden size): the image RESTORE path (tt/tp_image.py replays a meta
    from_torch as the recorded DRAM allocation and pulls the bytes from the image) -- an image built with the drafter
    (GLM53_DF2_WEIGHTS=safetensors at the image build) restores it without the 6.2 GB host read"""
    import json
    import os
    import struct

    p = os.path.join(drafter_dir, "model.safetensors")
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    w = {k: torch.empty(tuple(v["shape"]), dtype=BF16, device="meta") for k, v in hdr.items() if k != "__metadata__"}
    cfg = hf_config(os.path.join(drafter_dir, "config.json"))
    w["mask_embedding"] = torch.empty(int(cfg["hidden_size"]), dtype=BF16, device="meta")
    return w


def rope_cos_sin(pos: torch.Tensor, theta: float, hd: int):
    """neox (rotate-half) cos / sin rows [n, hd] fp32 for absolute positions pos [n]"""
    half = hd // 2
    inv = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    ang = pos.float()[:, None] * inv[None, :]
    return torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], -1)


def lin(x: torch.Tensor, w: torch.Tensor, out=BF16) -> torch.Tensor:
    """bf16 inputs, fp32 accumulate, `out` rounding (the drafter's F.linear)"""
    return (x.to(BF16).float() @ w.to(BF16).float().t()).to(out)


def rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """HF Qwen3RMSNorm: fp32 normalise, bf16, then the weight"""
    xf = x.float()
    y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(BF16)
    return (w.to(BF16).float() * y.float()).to(BF16)


class DFlash2Ref:
    def __init__(self, cfg: dict, w: dict):
        self.d, self.w = Dims(cfg), w
        d = self.d
        self.Kc = [torch.zeros(0, d.nkv, d.hd) for _ in range(d.L)]
        self.Vc = [torch.zeros(0, d.nkv, d.hd) for _ in range(d.L)]
        self.cpos = torch.zeros(0, dtype=torch.long)  # the positions held (the cache is dense, positions 0..n-1)
        self.probe = None  # stage probes (name -> tensor) when a dict: the kind's names (tt/dflash2_kind.py _p)

    def _p(self, name: str, t):
        if self.probe is not None:
            self.probe[name] = t
        return t

    # ------------------------------------------------------------------ pieces
    def conv(self, x: torch.Tensor, d: torch.Tensor, side: int, base: torch.Tensor) -> torch.Tensor:
        """x [r, H] bf16, d = kernel_projection(..) [r, 4 ng] bf16, base [2 sides, 2 taps, H] -> bf16 (fp32 math:
        coef = base + delta, out = coef0 x + coef1 x_prev; row 0 has no predecessor)"""
        dm = self.d
        xf = x.float()
        xp = torch.cat([torch.zeros_like(xf[:1]), xf[:-1]], 0)
        o = side * 2 * dm.ng
        out = torch.zeros_like(xf)
        for t, src in ((0, xf), (1, xp)):
            delta = d[:, o + t * dm.ng : o + (t + 1) * dm.ng].float().repeat_interleave(dm.group, dim=-1)
            out = out + (base[side, t].float()[None, :] + delta) * src
        return out.to(BF16)

    def heads(self, x: torch.Tensor, n: int) -> torch.Tensor:
        return x.reshape(x.shape[0], n, self.d.hd)

    def rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """x [r, heads, hd] bf16 at positions pos [r] -> bf16 (fp32 math)"""
        cos, sin = rope_cos_sin(pos, self.d.theta, self.d.hd)
        xf = x.float()
        return (xf * cos[:, None, :] + rotate_half(xf) * sin[:, None, :]).to(BF16)

    # ------------------------------------------------------------------ the context
    def ctx(self, taps: torch.Tensor, p0: int):
        """taps [n, n_taps, H] (bf16 or float; the mean over the target's residual streams already taken) at positions
        p0 .. p0+n-1 -> this reference's context K / V rows (appended / overwritten by position)"""
        d, w = self.d, self.w
        n = int(taps.shape[0])
        x = taps.to(BF16).reshape(n, d.n_taps * d.H)
        nrm = self._p("ctx.nrm", rms(lin(x, w["fc.weight"]), w["hidden_norm.weight"], d.eps))
        pos = torch.arange(p0, p0 + n)
        need = p0 + n
        for i in range(d.L):
            a = f"layers.{i}.self_attn."
            k = self.rope(rms(self.heads(lin(nrm, w[a + "k_proj.weight"]), d.nkv), w[a + "k_norm.weight"], d.eps), pos)
            v = self.heads(lin(nrm, w[a + "v_proj.weight"]), d.nkv)
            if self.Kc[i].shape[0] < need:
                pad = need - self.Kc[i].shape[0]
                self.Kc[i] = torch.cat([self.Kc[i], torch.zeros(pad, d.nkv, d.hd)], 0)
                self.Vc[i] = torch.cat([self.Vc[i], torch.zeros(pad, d.nkv, d.hd)], 0)
            self.Kc[i][p0 : p0 + n] = k.float()
            self.Vc[i][p0 : p0 + n] = v.float()

    # ------------------------------------------------------------------ the block
    def block(self, bonus: int, q0: int, n_ctx: int, K: int | None = None) -> dict:
        """the block rows [bonus, mask x K] at positions q0 .. q0+K, attending to context positions < n_ctx and to the
        K+1 block rows (non-causal) -> dict(hid [K+1, H] bf16 of the final norm, logits [K+1, V] fp32)"""
        d, w = self.d, self.w
        K = d.K if K is None else K
        g = d.nh // d.nkv
        scale = 1.0 / math.sqrt(d.hd)
        x_in = torch.cat(
            [w["embed_tokens.weight"][bonus].reshape(1, -1), w["mask_embedding"].reshape(1, -1).repeat(K, 1)], 0
        )
        pos = torch.arange(q0, q0 + K + 1)
        h, res = self._p("emb", x_in.to(BF16)), None
        for i in range(d.L):
            p, a = f"layers.{i}.", f"layers.{i}.self_attn."
            P = lambda n, t: self._p(f"L{i}.{n}", t)  # noqa: E731
            res = h if res is None else (res.float() + h.float()).to(BF16)
            x = P("x_ln", rms(res, w[p + "input_layernorm.weight"], d.eps))
            cf = P("cf", lin(x, w[p + "attention_conv.kernel_projection.weight"]))
            x = P("conv_a", self.conv(x, cf, 0, w[p + "attention_conv.base_kernel"]))
            q = P(
                "q",
                self.rope(rms(self.heads(lin(x, w[a + "q_proj.weight"]), d.nh), w[a + "q_norm.weight"], d.eps), pos),
            )
            k = P(
                "k",
                self.rope(rms(self.heads(lin(x, w[a + "k_proj.weight"]), d.nkv), w[a + "k_norm.weight"], d.eps), pos),
            )
            v = P("v", self.heads(lin(x, w[a + "v_proj.weight"]), d.nkv))
            Kall = torch.cat([self.Kc[i][:n_ctx], k.float()], 0)  # [n_ctx + K+1, nkv, hd]
            Vall = torch.cat([self.Vc[i][:n_ctx], v.float()], 0)
            qf = q.float().reshape(K + 1, d.nkv, g, d.hd)  # head h = kv * g + j
            s = torch.einsum("rkgd,skd->rkgs", qf, Kall) * scale
            pr = torch.softmax(s, -1)
            att = P("att", torch.einsum("rkgs,skd->rkgd", pr, Vall).reshape(K + 1, d.nh * d.hd).to(BF16))
            o = P(
                "conv_o",
                self.conv(P("o", lin(att, w[a + "o_proj.weight"])), cf, 1, w[p + "attention_conv.base_kernel"]),
            )
            res = P("res", (res.float() + o.float()).to(BF16))
            x = P("x_post", rms(res, w[p + "post_attention_layernorm.weight"], d.eps))
            cf2 = P("cf2", lin(x, w[p + "mlp_conv.kernel_projection.weight"]))
            x = P("conv_m", self.conv(x, cf2, 0, w[p + "mlp_conv.base_kernel"]))
            gate, up = lin(x, w[p + "mlp.gate_proj.weight"]), lin(x, w[p + "mlp.up_proj.weight"])
            y = P(
                "y",
                lin(
                    (torch.nn.functional.silu(gate.float()).to(BF16).float() * up.float()).to(BF16),
                    w[p + "mlp.down_proj.weight"],
                ),
            )
            h = P("h", self.conv(y, cf2, 1, w[p + "mlp_conv.base_kernel"]))
        hid = self._p("hid", rms((res.float() + h.float()).to(BF16), w["norm.weight"], d.eps))
        logits = lin(hid, w["lm_head.weight"], torch.float32)
        return dict(hid=hid, logits=logits)

    # ------------------------------------------------------------------ the selector
    def select(self, hid: torch.Tensor, logits: torch.Tensor, bonus: int, K: int | None = None) -> dict:
        """hid / logits of the block (rows 0..K; row 0 = the bonus row, unused) -> the K drafts: top-k candidates per
        mask row, bilinear scores with the predecessor / successor codebooks, the greedy walk from the bonus"""
        d, w = self.d, self.w
        K = d.K if K is None else K
        unary, cand = torch.topk(logits[1 : K + 1].float(), d.top_k, dim=-1, largest=True, sorted=True)
        r = self.walk(hid[1 : K + 1], cand, unary, bonus)
        return dict(r, cand=cand, unary=unary, plain=[int(v) for v in cand[:, 0]])

    def walk(self, hid: torch.Tensor, cand: torch.Tensor, unary: torch.Tensor, bonus: int) -> dict:
        """the selector on given candidates: hid [K, H] (mask rows), cand / unary [K, top_k] -> drafts, scores"""
        w = self.w
        K = int(cand.shape[0])
        hp = hid.to(BF16).float() @ w["candidate_selector.hidden_projection.weight"].float().t()
        succ = w["candidate_selector.successor_codebook"][cand.long()].float()  # [K, top_k, rank]
        prev_ids = torch.cat([torch.full((1, cand.shape[1]), bonus, dtype=cand.dtype), cand[:-1]], 0)
        pred = w["candidate_selector.predecessor_codebook"][prev_ids.long()].float()
        scores = unary.float()[:, None, :] + torch.einsum("lpr,lcr->lpc", pred * hp[:, None, :], succ)
        prev, toks = 0, []
        for s in range(K):
            j = int(torch.argmax(scores[s, prev]))
            toks.append(int(cand[s, j]))
            prev = j
        return dict(drafts=toks, scores=scores)
