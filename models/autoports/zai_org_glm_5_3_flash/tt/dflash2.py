# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 17 DFlash2 block drafter, device prototype (doc/ttnn/17-mtp-spec-decode/dflash2_design.md; coordinator 09:38Z:
"the DFlash2 drafter as a traced standalone program ... the 8-layer drafter forward on golden taps"). The Apache-2.0
drafter for GLM-5.3-Flash ONLY (S5 report D4). Every chip of the mesh holds the whole drafter (replicated, bf16, ~6.2
GB per chip) and runs the same forward -- the TP8-sharded production layout comes after this numerics prototype.

  ctx(taps)   the committed positions' target taps [n, 9 x 4096] -> fc -> hidden_norm -> per layer k / v proj, k_norm,
              RoPE at the absolute positions -> the drafter's context K / V [1, 8, S, 128] per layer (capacity S)
  block()     the block [bonus, mask x K] (rows padded to 32) at positions L+1 .. L+1+K: 8 layers of RMS -> grouped
              2-tap conv (prepare) -> attention (q / k norm, RoPE, NON-causal over the context rows < n and the block
              rows; additive mask) -> o_proj -> conv (finish) -> residual; RMS -> conv -> SwiGLU -> conv -> residual;
              final RMS -> lm_head -> top-16 candidates (values / ids) per row
  selector    (host, from the block's hidden / candidates): hidden_projection, the predecessor / successor codebook
              bilinear scores and the greedy walk from the bonus (tests/s17/jobs/dflash2_proto.py)
Grouped conv (2 taps, groups of 16 channels): out[r] = (base[0] + d[r,0,g]) x[r] + (base[1] + d[r,1,g]) x[r-1] (row 0
has no predecessor), d = kernel_projection(the conv-pair's input) -> [r, 2 (prepare / finish), 2 taps, 256 groups]; the
group -> channel expansion is a [256, 4096] 0/1 matmul, the row shift a [32, 32] 0/1 matmul (exact in bf16 / fp32).
Inputs the host writes before a block replay (no other host work): the block embedding rows, the block's RoPE rows, the
additive mask row."""

from __future__ import annotations

import json
import math
import os

import torch

import ttnn

TILE = 32
BF16, FP32 = ttnn.bfloat16, ttnn.float32
DRAM = ttnn.DRAM_MEMORY_CONFIG


def rope_rows(pos: torch.Tensor, theta: float, hd: int) -> tuple:
    """neox (rotate-half) cos / sin rows [n, hd] fp32 for absolute positions pos [n]"""
    half = hd // 2
    inv = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    ang = pos.float()[:, None] * inv[None, :]
    return torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)


class DFlash2TT:
    def __init__(self, mesh, d: str, cap: int, log=print):
        from safetensors import safe_open

        self.mesh = mesh
        self.cfg = json.load(open(os.path.join(d, "config.json")))
        c, dc = self.cfg, self.cfg["dflash_config"]
        self.H, self.nh, self.nkv, self.hd = (
            c["hidden_size"],
            c["num_attention_heads"],
            c["num_key_value_heads"],
            c["head_dim"],
        )
        self.eps, self.theta, self.nl = c["rms_norm_eps"], c["rope_parameters"]["rope_theta"], c["num_hidden_layers"]
        self.mask_id, self.top_k, self.group = dc["mask_token_id"], dc["selector_top_k"], dc["conv_group_size"]
        self.ng = self.H // self.group
        self.cap = -(-cap // TILE) * TILE  # context capacity (rows), tile-padded
        self.rep = ttnn.ReplicateTensorToMesh(mesh) if int(mesh.get_num_devices()) > 1 else None
        self.ckc = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                          math_approx_mode=False, fp32_dest_acc_en=True,
                                                          packer_l1_acc=False)  # fmt: skip
        f = safe_open(os.path.join(d, "model.safetensors"), framework="pt", device="cpu")
        self.host = {k: f.get_tensor(k) for k in ("embed_tokens.weight", "candidate_selector.predecessor_codebook",
                                                     "candidate_selector.successor_codebook",
                                                     "candidate_selector.hidden_projection.weight")}  # fmt: skip
        m = torch.load(os.path.join(d, "mask_embedding.pt"), map_location="cpu", weights_only=False)
        self.mask_emb = (m["embedding"] if isinstance(m, dict) else m).reshape(-1).to(torch.bfloat16)
        lin = lambda k: self._dev(f.get_tensor(k).t().contiguous())  # noqa: E731  [in, out] for ttnn.linear
        row = lambda k: self._dev(f.get_tensor(k).reshape(1, 1, 1, -1))  # noqa: E731  [1, 1, 1, W] norm weights
        self.fc, self.hidden_norm, self.norm = lin("fc.weight"), row("hidden_norm.weight"), row("norm.weight")
        self.lm_head = lin("lm_head.weight")
        self.L = []
        for i in range(self.nl):
            p = f"layers.{i}."
            a = p + "self_attn."
            self.L.append(dict(
                in_ln=row(p + "input_layernorm.weight"), post_ln=row(p + "post_attention_layernorm.weight"),
                q=lin(a + "q_proj.weight"), k=lin(a + "k_proj.weight"), v=lin(a + "v_proj.weight"),
                o=lin(a + "o_proj.weight"), q_norm=row(a + "q_norm.weight"), k_norm=row(a + "k_norm.weight"),
                gate=lin(p + "mlp.gate_proj.weight"), up=lin(p + "mlp.up_proj.weight"), down=lin(p + "mlp.down_proj.weight"),
                kp_attn=lin(p + "attention_conv.kernel_projection.weight"),
                kp_mlp=lin(p + "mlp_conv.kernel_projection.weight"),
                base_attn=self._base(f.get_tensor(p + "attention_conv.base_kernel")),
                base_mlp=self._base(f.get_tensor(p + "mlp_conv.base_kernel"))))  # fmt: skip
            log(f"dflash2: layer {i} weights on device")
        # group -> channel expansion [256, 4096] and the row shift [32, 32] (x_prev = S x), exact 0/1 matrices
        E = torch.zeros(self.ng, self.H)
        E[torch.arange(self.H) // self.group, torch.arange(self.H)] = 1
        self.E = self._dev(E, FP32)
        Sh = torch.zeros(TILE, TILE)
        Sh[torch.arange(1, TILE), torch.arange(TILE - 1)] = 1
        self.Sh = self._dev(Sh.reshape(1, 1, TILE, TILE), FP32)
        R = torch.zeros(self.hd, self.hd)  # rotate-half as a matmul: x R = cat(-x2, x1)
        h2 = self.hd // 2
        R[torch.arange(h2) + h2, torch.arange(h2)] = -1
        R[torch.arange(h2), torch.arange(h2) + h2] = 1
        self.R = self._dev(R, FP32)
        # the context K / V per layer, [1, nkv, cap, hd] bf16, and the persistent block inputs
        self.Kc = [self._dev(torch.zeros(1, self.nkv, self.cap, self.hd)) for _ in range(self.nl)]
        self.Vc = [self._dev(torch.zeros(1, self.nkv, self.cap, self.hd)) for _ in range(self.nl)]
        self.x_in = self._dev(torch.zeros(1, 1, TILE, self.H))
        self.cos_b = self._dev(torch.zeros(1, 1, TILE, self.hd), FP32)
        self.sin_b = self._dev(torch.zeros(1, 1, TILE, self.hd), FP32)
        g = self.nh // self.nkv  # q heads per kv head: q rows grouped [1, nkv, g x 32, hd]
        self.mask = self._dev(torch.zeros(1, 1, g * TILE, self.cap + TILE), FP32)
        self.taps = None
        self.out = None

    # ------------------------------------------------------------------ helpers
    def _dev(self, t: torch.Tensor, dt=BF16):
        return ttnn.from_torch(t.to(torch.float32 if dt == FP32 else torch.bfloat16), dtype=dt, layout=ttnn.TILE_LAYOUT,
                               device=self.mesh, memory_config=DRAM, mesh_mapper=self.rep)  # fmt: skip

    def _write(self, dst, t: torch.Tensor, dt=BF16):
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(t.to(torch.float32 if dt == FP32 else torch.bfloat16), dtype=dt,
                                                        layout=ttnn.TILE_LAYOUT, mesh_mapper=self.rep), dst)  # fmt: skip

    def _base(self, b: torch.Tensor):
        """base_kernel [2 (prepare / finish), 2 taps, C] -> 4 fp32 rows [1, 1, 1, C]"""
        return [[self._dev(b[s, t].reshape(1, 1, 1, -1), FP32) for t in range(2)] for s in range(2)]

    def _lin(self, x, w, dt=BF16):
        return ttnn.linear(x, w, compute_kernel_config=self.ckc, dtype=dt, memory_config=DRAM)

    def _rms(self, x, w):
        return ttnn.rms_norm(x, epsilon=self.eps, weight=w, compute_kernel_config=self.ckc, memory_config=DRAM)

    def _rope(self, x, cos, sin):
        """x [1, heads, rows, hd] -> bf16 (fp32 math as the reference)"""
        xf = ttnn.typecast(x, FP32)
        rot = ttnn.matmul(xf, self.R, compute_kernel_config=self.ckc, memory_config=DRAM)
        y = ttnn.add(ttnn.multiply(xf, cos), ttnn.multiply(rot, sin))
        return ttnn.typecast(y, BF16)

    def _heads(self, t, rows, n):
        return ttnn.permute(ttnn.reshape(t, (1, rows, n, self.hd)), (0, 2, 1, 3))

    def _conv(self, x, cf, side: int, base):
        """x [1, 1, 32, C] bf16, cf = kernel_projection(..) [1, 1, 32, 1024] bf16 -> bf16 (fp32 math as the reference:
        coef = base + delta in fp32, out = coef0 x + coef1 x_prev, one bf16 rounding)"""
        o = side * 2 * self.ng
        xf = ttnn.typecast(x, FP32)
        out = None
        for t in range(2):
            d = ttnn.typecast(ttnn.slice(cf, [0, 0, 0, o + t * self.ng], [1, 1, TILE, o + (t + 1) * self.ng]), FP32)
            coef = ttnn.add(ttnn.matmul(d, self.E, compute_kernel_config=self.ckc), base[side][t])
            src = xf if t == 0 else ttnn.matmul(self.Sh, xf, compute_kernel_config=self.ckc)
            term = ttnn.multiply(coef, src)
            out = term if out is None else ttnn.add(out, term)
        return ttnn.typecast(out, BF16)

    # ------------------------------------------------------------------ context K / V
    def ctx(self, taps: torch.Tensor, pos0: int = 0):
        """taps [9, n, 4096] (positions pos0 .. pos0 + n - 1, n <= cap) -> the context K / V (eager-compiled; callers
        capture it). Rows past n are zero."""
        n = int(taps.shape[1])
        assert pos0 == 0 and n <= self.cap, (pos0, n, self.cap)
        x = torch.zeros(1, 1, self.cap, taps.shape[0] * self.H)
        x[0, 0, :n] = taps.permute(1, 0, 2).reshape(n, -1).float()
        if self.taps is None:
            self.taps = self._dev(x)
        else:
            self._write(self.taps, x)
        cos, sin = rope_rows(torch.arange(self.cap), self.theta, self.hd)
        self.cos_c = self._dev(cos.reshape(1, 1, self.cap, self.hd), FP32)
        self.sin_c = self._dev(sin.reshape(1, 1, self.cap, self.hd), FP32)

    def ctx_program(self):
        nrm = self._rms(self._lin(self.taps, self.fc), self.hidden_norm)
        for i, Ly in enumerate(self.L):
            k = self._rms(self._heads(self._lin(nrm, Ly["k"]), self.cap, self.nkv), Ly["k_norm"])
            ttnn.copy(self._rope(k, self.cos_c, self.sin_c), self.Kc[i])
            ttnn.copy(self._heads(self._lin(nrm, Ly["v"]), self.cap, self.nkv), self.Vc[i])

    # ------------------------------------------------------------------ the block
    def set_block(self, bonus: int, L: int, n_ctx: int, K: int):
        """host inputs of one block: embedding rows [bonus, mask x K] (rows 0..K), RoPE rows at L+1 .., the mask"""
        e = torch.zeros(1, 1, TILE, self.H)
        e[0, 0, 0] = self.host["embed_tokens.weight"][bonus].float()
        e[0, 0, 1 : K + 1] = self.mask_emb.float()
        self._write(self.x_in, e)
        cos, sin = rope_rows(torch.arange(L + 1, L + 1 + TILE), self.theta, self.hd)
        self._write(self.cos_b, cos.reshape(1, 1, TILE, self.hd), FP32)
        self._write(self.sin_b, sin.reshape(1, 1, TILE, self.hd), FP32)
        m = torch.full((self.cap + TILE,), float("-inf"))
        m[:n_ctx] = 0
        m[self.cap : self.cap + K + 1] = 0
        g = self.nh // self.nkv
        self._write(self.mask, m.reshape(1, 1, 1, -1).expand(1, 1, g * TILE, -1).contiguous(), FP32)

    def block_program(self):
        """the 8 layers + final norm + lm_head top-16 over the block rows -> self.out = (hid, vals, ids)"""
        g = self.nh // self.nkv
        scale = 1.0 / math.sqrt(self.hd)
        h, res = self.x_in, None
        for i, Ly in enumerate(self.L):
            res = h if res is None else ttnn.add(res, h)
            x = self._rms(res, Ly["in_ln"])
            cf = self._lin(x, Ly["kp_attn"])  # bf16, as the reference's F.linear
            x = self._conv(x, cf, 0, Ly["base_attn"])
            q = self._rope(
                self._rms(self._heads(self._lin(x, Ly["q"]), TILE, self.nh), Ly["q_norm"]), self.cos_b, self.sin_b
            )
            k = self._rope(
                self._rms(self._heads(self._lin(x, Ly["k"]), TILE, self.nkv), Ly["k_norm"]), self.cos_b, self.sin_b
            )
            v = self._heads(self._lin(x, Ly["v"]), TILE, self.nkv)
            K_ = ttnn.typecast(ttnn.concat([self.Kc[i], k], dim=2), FP32)  # [1, nkv, cap + 32, hd]
            V_ = ttnn.typecast(ttnn.concat([self.Vc[i], v], dim=2), FP32)
            qg = ttnn.reshape(ttnn.typecast(q, FP32), (1, self.nkv, g * TILE, self.hd))  # head h = kv * g + j
            s = ttnn.matmul(qg, ttnn.transpose(K_, 2, 3), compute_kernel_config=self.ckc, memory_config=DRAM)
            s = ttnn.add(ttnn.multiply(s, scale), self.mask)
            pr = ttnn.softmax(s, dim=-1, compute_kernel_config=self.ckc, numeric_stable=True)
            a = ttnn.matmul(pr, V_, compute_kernel_config=self.ckc, memory_config=DRAM)  # [1, nkv, g 32, hd] fp32
            a = ttnn.reshape(
                ttnn.permute(ttnn.reshape(a, (1, self.nh, TILE, self.hd)), (0, 2, 1, 3)), (1, 1, TILE, self.H)
            )
            o = self._lin(ttnn.typecast(a, BF16), Ly["o"])
            o = self._conv(o, cf, 1, Ly["base_attn"])
            res = ttnn.add(res, o)
            x = self._rms(res, Ly["post_ln"])
            cf2 = self._lin(x, Ly["kp_mlp"])
            x = self._conv(x, cf2, 0, Ly["base_mlp"])
            y = self._lin(ttnn.multiply(ttnn.silu(self._lin(x, Ly["gate"])), self._lin(x, Ly["up"])), Ly["down"])
            h = self._conv(y, cf2, 1, Ly["base_mlp"])
        hid = self._rms(ttnn.add(res, h), self.norm)
        lg = self._lin(hid, self.lm_head, FP32)
        vals, ids = ttnn.topk(lg, k=self.top_k, dim=-1, largest=True, sorted=True)
        if self.out is None:
            self.out = (hid, vals, ids)
        else:
            for dst, src in zip(self.out, (hid, vals, ids)):
                ttnn.copy(src, dst)

    def read_block(self, K: int) -> dict:
        """chip 0's block outputs: hid [K, 4096] fp32 (mask rows), candidates / unary [K, 16]"""
        one = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0] if self.rep is not None else t)  # noqa: E731
        hid, vals, ids = (one(t) for t in self.out)
        return dict(hid=hid.reshape(TILE, -1)[1: K + 1].float(), unary=vals.reshape(TILE, -1)[1: K + 1].float(),
                    cand=ids.reshape(TILE, -1)[1: K + 1].to(torch.int64))  # fmt: skip

    def select(self, r: dict, bonus: int) -> list:
        """the candidate selector on the host (fp32): bilinear scores + the greedy walk from the bonus"""
        hp = r["hid"].to(torch.bfloat16).float() @ self.host["candidate_selector.hidden_projection.weight"].float().t()
        cand = r["cand"]
        succ = self.host["candidate_selector.successor_codebook"][cand].float()
        prev_ids = torch.cat([torch.full((1, cand.shape[1]), bonus, dtype=cand.dtype), cand[:-1]], 0)
        pred = self.host["candidate_selector.predecessor_codebook"][prev_ids].float()
        scores = r["unary"][:, None, :] + torch.einsum("lpr,lcr->lpc", pred * hp[:, None, :], succ)
        prev, toks = 0, []
        for s in range(cand.shape[0]):
            j = int(torch.argmax(scores[s, prev]))
            toks.append(int(cand[s, j]))
            prev = j
        return toks
