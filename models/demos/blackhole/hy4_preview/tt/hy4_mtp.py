# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4-Preview MTP block on device (task mtp-device): ONE full layer (index 78, no iHC, plain residual) reusing the chain layer class (attention pool + indexer + shared expert + MoE).
Row r of a call = (token t_r, hidden h_r [post model.norm], position p_r): x = eh_proj(cat[enorm(embed t), hnorm(h)]); x += attn(input_layernorm x); x += mlp(post_attention_layernorm x);
out = final_layernorm(x) (the MTP hidden: fed to the shared lm_head for the draft, and back as ``h`` of the next recursive step). See $PROGRAM/spec-mtp/DESIGN.md s1.
Contract: ``forward`` takes S rows of ONE user per mesh row in the pool decode layout of the main model (PoolCtx.S = S; ctx.set_step(pos) for the first position of the rows);
``moe_fn`` = None -> the layer's Hy4MoEBlock (4x8 mesh), else a host callable (1-slot tests: routed experts on the host oracle, shared expert on device).
"""


import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tt import hy4_dense as D
from models.demos.blackhole.hy4_preview.tt.hy4_chain import HID, rms_exact
from models.demos.blackhole.hy4_preview.tt.hy4_common import compute_cfg, upload
from models.demos.blackhole.hy4_preview.tt.hy4_decode import with_decode_cfg


class Hy4MTP:
    def __init__(self, ch, ly, sd, opt, moe_fn=None, zero_pos0=False):
        """ch: Hy4Chain (embed, lm_head, ccl); ly: its Hy4ChainLayer 78; sd: the host state dict of model.mtp_layers.0 (norm weights)."""
        self.ch, self.ly, self.opt, self.md = ch, ly, opt, ch.md
        self.moe_fn = moe_fn
        self.zero_pos0 = zero_pos0
        self.eps = ch.cfg.rms_norm_eps
        f32 = lambda k: upload(self.md, sd[k + ".weight"].float().reshape(1, 1, 1, HID), ttnn.float32)
        self.w_e, self.w_h, self.w_fin = f32("enorm"), f32("hnorm"), f32("final_layernorm")
        self.ckc = compute_cfg(self.md, "hifi4", True)
        # HY4_MTP_NORM=fast (verify-round track, DRAFT-ONLY numerics: the MTP output only proposes drafts, the verify rows decide): the 7-op fp32 rms_exact composite (~100 us per call, 5 calls per MTP pass)
        # replaced by typecast + ttnn.rms_norm on bf16 (bf16 sum of squares: ~3-4 % scale error, acceptance is measured by the spec_eq histogram)
        self.fast_norm = os.environ.get("HY4_MTP_NORM", "exact") == "fast"
        if self.fast_norm:
            wb = lambda t: upload(
                self.md,
                ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float().reshape(1, 1, HID // 32, 32),
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
            )
            self._wb = {id(t): wb(t) for t in (self.w_e, self.w_h, self.w_fin, ly.w_in32, ly.w_post32)}
            self.ckc_norm = compute_cfg(self.md, "hifi4", False)
        # HY4_MTP_NORM=fused (verify-opt track, draft-only numerics, fp32 throughout): 6-op rms (eps*D folded into one add, sqrt(D) folded into the weight) and ONE batched call for the
        # enorm/hnorm pair of ``pre`` (rows [e; h] share the sum-of-squares ops; the two weights ride in one [2S, D] tensor) instead of 2 x 7 ops
        self.fused_norm = os.environ.get("HY4_MTP_NORM", "exact") == "fused"
        if self.fused_norm:
            self._sq = float(HID) ** 0.5
            self._wsc = {}
            self._wh = {
                k: sd[k + ".weight"].float().reshape(1, 1, 1, HID) for k in ("enorm", "hnorm", "final_layernorm")
            }
        self.shared = ly.shared_dec if getattr(ly, "shared_dec", None) is not None else with_decode_cfg(ly.shared)

    def _rms6(self, x32, wsc):
        """x32 [1,1,n,D] fp32, wsc = weight * sqrt(D) (fp32, [1,1,1,D] or [1,1,n,D]) -> bf16 [1,1,n,D]."""
        sq = ttnn.multiply(x32, x32)
        ss = ttnn.sum(sq, dim=3, keepdim=True)
        ttnn.deallocate(sq)
        r = ttnn.rsqrt(ttnn.add(ss, self.eps * HID))
        ttnn.deallocate(ss)
        y = ttnn.multiply(x32, r)
        ttnn.deallocate(r)
        z = ttnn.multiply(y, wsc, dtype=ttnn.bfloat16)
        ttnn.deallocate(y)
        return z

    def _wscaled(self, key, w32, n=1):
        k = (key, n)
        if k not in self._wsc:
            w = (
                self._wh[key]
                if key in self._wh
                else ttnn.to_torch(ttnn.get_device_tensors(w32)[0]).float().reshape(1, 1, 1, HID)
            )
            self._wsc[k] = upload(self.md, (w * self._sq).expand(1, 1, n, HID).contiguous(), ttnn.float32)
        return self._wsc[k]

    def _norm(self, x32, w32):
        if self.fused_norm:
            key = {id(self.w_e): "enorm", id(self.w_h): "hnorm", id(self.w_fin): "final_layernorm"}.get(
                id(w32), id(w32)
            )
            return self._rms6(x32, self._wscaled(key, w32))
        if not self.fast_norm:
            return rms_exact(x32, w32, self.eps)
        xb = ttnn.typecast(x32, ttnn.bfloat16)
        return ttnn.rms_norm(xb, weight=self._wb[id(w32)], epsilon=self.eps, compute_kernel_config=self.ckc_norm)

    def _wscaled2(self, S):
        k = ("eh", S)
        if k not in self._wsc:
            import torch as _t

            w = (
                _t.cat([self._wh["enorm"].expand(1, 1, S, HID), self._wh["hnorm"].expand(1, 1, S, HID)], dim=2)
                * self._sq
            )
            self._wsc[k] = upload(self.md, w.contiguous(), ttnn.float32)
        return self._wsc[k]

    def embed(self, tok):
        """tok uint32 [1, S] ROW_MAJOR replicated -> bf16 [1,1,S,D] replicated (shared embedding table, vocab column-sharded on the hidden dim)."""
        ch = self.ch
        S = int(tok.shape[1])
        e = ttnn.embedding(tok, ch.embed, layout=ttnn.TILE_LAYOUT)
        e = ttnn.reshape(e, [1, 1, S, HID // 8])
        return ch.mc.allgather(e, ch.ccl, axis=1, dim=3)

    def pre(self, e, h):
        """e bf16 [1,1,S,D] embeddings, h bf16 [1,1,S,D] main hidden -> x fp32 [1,1,S,D] = eh_proj(cat[enorm e, hnorm h])."""
        ch = self.ch
        e32 = ttnn.typecast(e, ttnn.float32)
        h32 = ttnn.typecast(h, ttnn.float32)
        if self.fused_norm:
            S = int(e32.shape[2])
            both = ttnn.concat([e32, h32], dim=2)  # [1,1,2S,D]
            for t in (e32, h32):
                ttnn.deallocate(t)
            w2 = self._wscaled2(S)
            z = self._rms6(both, w2)
            ttnn.deallocate(both)
            cat = ttnn.concat(
                [ttnn.slice(z, [0, 0, 0, 0], [1, 1, S, HID]), ttnn.slice(z, [0, 0, S, 0], [1, 1, 2 * S, HID])], dim=3
            )
            ttnn.deallocate(z)
        else:
            en = self._norm(e32, self.w_e)
            hn = self._norm(h32, self.w_h)
            for t in (e32, h32):
                ttnn.deallocate(t)
            cat = ttnn.concat([en, hn], dim=3)
            for t in (en, hn):
                ttnn.deallocate(t)
        x = ttnn.linear(cat, self.ly.eh_proj, compute_kernel_config=self.ckc)  # [1,1,S,D/8] per column
        ttnn.deallocate(cat)
        xg = ch.mc.allgather(x, ch.ccl, axis=1, dim=3)
        ttnn.deallocate(x)
        x32 = ttnn.typecast(xg, ttnn.float32)
        ttnn.deallocate(xg)
        return x32

    def mlp(self, h2, taps=None):
        """h2 bf16 [1,1,S,D] -> y bf16 [1,1,S,D] (routed + shared expert)."""
        ly, ch = self.ly, self.ch
        part = self.shared.forward(h2)
        if self.moe_fn is None:
            S = int(h2.shape[2])
            tok = ttnn.reshape(ttnn.to_layout(h2, ttnn.ROW_MAJOR_LAYOUT), [S, 1, 1, HID])
            _blk = ly.blk_tp if getattr(ly, "blk_tp", None) is not None else ly.blk  # B42: tp24 MTP MoE when built
            out = _blk.forward(h2, tok, fold=part)
            ttnn.deallocate(part)
            out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
            y = ch.mc.allgather(out, ch.ccl, axis=1, dim=3)
            ttnn.deallocate(out)
            return y
        sh = D.allreduce_cols(part)
        ttnn.deallocate(part)
        if self.moe_fn == "skip":  # timing probes: no routed experts (cost added from the MoE measurements)
            return sh
        x = ttnn.to_torch(ttnn.get_device_tensors(h2)[0]).reshape(-1, HID).float()
        routed = self.moe_fn(x)
        if taps is not None:
            taps["h2"], taps["routed"] = x, routed
            taps["sh"] = ttnn.to_torch(ttnn.get_device_tensors(sh)[0]).reshape(-1, HID).float()
        r = upload(self.md, routed.reshape(1, 1, -1, HID).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        y = ttnn.add(sh, r)
        for t in (sh, r):
            ttnn.deallocate(t)
        return y

    def forward(self, tok, h, tabs, dense8, taps=None):
        """one MTP step on S rows -> out bf16 [1,1,S,D] (final_layernorm); taps (dict) receives host copies of x0 / a / x1 / y / x2."""
        ly, ch = self.ly, self.ch
        tp = (
            (lambda n, t: taps.__setitem__(n, ttnn.to_torch(ttnn.get_device_tensors(t)[0]).reshape(-1, HID).float()))
            if taps is not None
            else (lambda n, t: None)
        )
        e = self.embed(tok)
        x = self.pre(e, h)
        ttnn.deallocate(e)
        tp("x0", x)
        h1 = self._norm(x, ly.w_in32)
        a = ly.attn.decode_step_pool(h1, self.opt, tabs, dense8)
        ttnn.deallocate(h1)
        tp("a", a)
        x = ttnn.add(x, ttnn.typecast(a, ttnn.float32))
        ttnn.deallocate(a)
        tp("x1", x)
        h2 = self._norm(x, ly.w_post32)
        y = self.mlp(h2, taps)
        ttnn.deallocate(h2)
        tp("y", y)
        x = ttnn.add(x, ttnn.typecast(y, ttnn.float32))
        ttnn.deallocate(y)
        tp("x2", x)
        out = self._norm(x, self.w_fin)
        ttnn.deallocate(x)
        return out


def greedy_tokens(ch, logits):
    """greedy global argmax of the column-sharded logits [1,1,S,V/8] (fp32 or bf16) -> token ids uint32 [S, 1] identical on every device (per column max + idx, all-gather, best column; the Hy4DeviceHead.sample_global scheme)."""
    cols = int(ch.md.shape[1])
    shard = int(logits.shape[3])
    lg = logits if logits.dtype == ttnn.float32 else ttnn.typecast(logits, ttnn.float32)
    rm = ttnn.to_layout(lg, ttnn.ROW_MAJOR_LAYOUT)
    idx = ttnn.argmax(rm, dim=-1, keepdim=True)
    mx = ttnn.to_layout(ttnn.gather(rm, -1, index=idx), ttnn.TILE_LAYOUT)
    idxf = ttnn.typecast(ttnn.to_layout(idx, ttnn.TILE_LAYOUT), ttnn.float32)
    g = ch.mc.allgather(ttnn.concat([mx, idxf], dim=-1), ch.ccl, axis=1, dim=3)
    T = int(g.shape[2])
    g = ttnn.reshape(g, [1, T, cols, 2])
    mx_all = ttnn.reshape(ttnn.slice(g, [0, 0, 0, 0], [1, T, cols, 1]), [1, 1, T, cols])
    ix_all = ttnn.reshape(ttnn.slice(g, [0, 0, 0, 1], [1, T, cols, 2]), [1, 1, T, cols])
    best = ttnn.argmax(ttnn.to_layout(mx_all, ttnn.ROW_MAJOR_LAYOUT), dim=-1, keepdim=True)
    bestf = ttnn.typecast(ttnn.to_layout(best, ttnn.TILE_LAYOUT), ttnn.float32)
    if not hasattr(ch, "_col_ids"):
        ch._col_ids = upload(ch.md, torch.arange(cols, dtype=torch.float32).reshape(1, 1, 1, cols), ttnn.float32)
    sel = ttnn.sum(ttnn.multiply(ttnn.eq(bestf, ch._col_ids), ix_all), dim=-1, keepdim=True)
    tok = ttnn.add(ttnn.multiply(bestf, float(shard)), sel)
    return ttnn.reshape(ttnn.to_layout(ttnn.typecast(tok, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [T, 1])
