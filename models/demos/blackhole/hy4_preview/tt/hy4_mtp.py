# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4-Preview MTP block on device (task mtp-device): ONE full layer (index 78, no iHC, plain residual) reusing the chain layer class (attention pool + indexer + shared expert + MoE).
Row r of a call = (token t_r, hidden h_r [post model.norm], position p_r): x = eh_proj(cat[enorm(embed t), hnorm(h)]); x += attn(input_layernorm x); x += mlp(post_attention_layernorm x);
out = final_layernorm(x) (the MTP hidden: fed to the shared lm_head for the draft, and back as ``h`` of the next recursive step). See $PROGRAM/spec-mtp/DESIGN.md s1.
Contract: ``forward`` takes S rows of ONE user per mesh row in the pool decode layout of the main model (PoolCtx.S = S; ctx.set_step(pos) for the first position of the rows);
``moe_fn`` = None -> the layer's Hy4MoEBlock (4x8 mesh), else a host callable (1-slot tests: routed experts on the host oracle, shared expert on device).
"""


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
        self.shared = ly.shared_dec if getattr(ly, "shared_dec", None) is not None else with_decode_cfg(ly.shared)

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
        en = rms_exact(e32, self.w_e, self.eps)
        hn = rms_exact(h32, self.w_h, self.eps)
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
            out = ly.blk.forward(h2, tok, fold=part)
            ttnn.deallocate(part)
            out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
            y = ch.mc.allgather(out, ch.ccl, axis=1, dim=3)
            ttnn.deallocate(out)
            return y
        sh = D.allreduce_cols(part)
        ttnn.deallocate(part)
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
        h1 = rms_exact(x, ly.w_in32, self.eps)
        a = ly.attn.decode_step_pool(h1, self.opt, tabs, dense8)
        ttnn.deallocate(h1)
        tp("a", a)
        x = ttnn.add(x, ttnn.typecast(a, ttnn.float32))
        ttnn.deallocate(a)
        tp("x1", x)
        h2 = rms_exact(x, ly.w_post32, self.eps)
        y = self.mlp(h2, taps)
        ttnn.deallocate(h2)
        tp("y", y)
        x = ttnn.add(x, ttnn.typecast(y, ttnn.float32))
        ttnn.deallocate(y)
        tp("x2", x)
        out = rms_exact(x, self.w_fin, self.eps)
        ttnn.deallocate(x)
        return out
