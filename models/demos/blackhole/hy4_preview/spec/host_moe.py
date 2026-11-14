# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host routed-expert MoE of one layer (oracle router + lazily cached bf16 experts) for 1-slot device tests of the MTP block, where the 4-row MoE decode block cannot be built."""
import torch

from models.demos.blackhole.hy4_preview.reference.loader import Hy4Checkpoint, split_gate_up
from models.demos.blackhole.hy4_preview.reference.oracle import router, swiglu


class HostRoutedMoE:
    def __init__(self, ckpt, layer=-1, clamp_mode="routed", exact=False, quant=None):
        self.quant = quant  # None | "bfp8" | "bfp4": experts round-tripped through the device block formats (acceptance A/B of the MTP expert precision)
        self.exact = exact  # fp32 matmuls (oracle numerics) instead of bf16
        self.ck = Hy4Checkpoint(ckpt)
        self.cfg, self.layer, self.clamp = self.ck.cfg, layer, clamp_mode
        self.sd = {
            k: v
            for k, v in self.ck.layer_state(layer).items()
            if k in ("mlp.gate.weight", "mlp.gate.e_score_correction_bias")
        }
        self.cache = {}
        self.last_sel = None

    def _q(self, m):
        """[out, in] bf16 weight -> block-format round trip (the device stores [in, out] tiles)."""
        import ttnn

        dt = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[self.quant]
        t = m.t().contiguous().reshape(1, 1, m.shape[1], m.shape[0])
        r = ttnn.to_torch(ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT))
        return r.reshape(m.shape[1], m.shape[0]).t().contiguous().to(torch.bfloat16)

    def expert(self, e):
        if e not in self.cache:
            p = self.ck.layer_prefix(self.layer) + "mlp.experts."
            gu = self.ck.get(p + "gate_up_proj", rows=e)
            dn = self.ck.get(p + "down_proj", rows=e)
            g, u = split_gate_up(gu)
            g, u, dn = g.contiguous(), u.contiguous(), dn.contiguous()
            if self.quant:
                g, u, dn = (self._q(m) for m in (g, u, dn))
            self.cache[e] = (g, u, dn)
        return self.cache[e]

    def __call__(self, x):
        """x [N,D] float (the device's normed MoE input) -> routed output [N,D] fp32 (shared expert NOT included)."""
        x = x.float()
        w, sel = router(x, self.sd, self.cfg)
        self.last_sel = sel
        y = torch.zeros_like(x)
        for e in sorted(set(sel.flatten().tolist())):
            tok, slot = torch.where(sel == e)
            g, u, d = self.expert(e)
            if self.exact:
                xt = x[tok]
                a = swiglu(xt @ g.float().T, xt @ u.float().T, self.cfg.swiglu_limit, self.clamp, "routed")
                y.index_add_(0, tok, (a @ d.float().T) * w[tok, slot].unsqueeze(-1))
                continue
            xt = x[tok].bfloat16()
            a = swiglu((xt @ g.T).float(), (xt @ u.T).float(), self.cfg.swiglu_limit, self.clamp, "routed")
            y.index_add_(0, tok, (a.bfloat16() @ d.T).float() * w[tok, slot].unsqueeze(-1))
        return y
