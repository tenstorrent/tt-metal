# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 17 DFlash2 block drafter, TENSOR-PARALLEL prototype on the image layout (tests/s17/dflash2_convert.py; the
Apache-2.0 drafter only). Same forward as tt/dflash2.py (the replicated numerics reference), with the matmul weights
sharded over the mesh's tp chips from the converted tile cache (bfloat8_b):
  column split (chip r: output columns r * out / tp ..): fc (then an all-gather: hidden_norm needs the full 4096),
      q / k / v (chip r: q heads r * nh / tp .., kv heads r * nkv / tp ..: whole GQA groups), gate / up, lm_head
      (chip r: vocab rows r * V / tp ..; top-16 per chip, merged on the host)
  row split (chip r: input rows r * in / tp ..): o_proj, down -> fp32 partials, all-reduced, one bf16 rounding
  replicated: the conv kernel projections (bfp8); norms / conv base kernels from the drafter's safetensors (tiny)
Collectives: stock ttnn.all_gather / all_reduce per mesh axis (axis 1 then 0 on a 2-D mesh: row-major chip order =
the shard order of ttnn.from_host_shards), as tt/prefill_tp.py. Context K / V per chip: its kv heads only."""

from __future__ import annotations

import json
import os

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.dflash2 import BF16, DRAM, FP32, TILE, DFlash2TT


class DFlash2TP(DFlash2TT):
    def __init__(self, mesh, d: str, cache: str, cap: int, log=print):
        from safetensors import safe_open

        self.mesh = mesh
        self.tp = int(mesh.get_num_devices())
        self.man = json.load(open(os.path.join(cache, "manifest.json")))
        assert int(self.man["tp"]) == self.tp, ("converted for tp", self.man["tp"], "mesh", self.tp)
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
        self.V = c["vocab_size"]
        assert self.nkv % self.tp == 0 and self.V % (self.tp * TILE) == 0, (self.nkv, self.V, self.tp)
        self.nh_l, self.nkv_l, self.V_l = self.nh // self.tp, self.nkv // self.tp, self.V // self.tp
        self.ng = self.H // self.group
        self.cap = -(-cap // TILE) * TILE
        self.rep = ttnn.ReplicateTensorToMesh(mesh)
        r, cl = int(mesh.shape[0]), int(mesh.shape[1])
        self.axes = [1, 0] if (r > 1 and cl > 1) else [None]
        self.ckc = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                          math_approx_mode=False, fp32_dest_acc_en=True,
                                                          packer_l1_acc=False)  # fmt: skip
        self.cache = cache
        f = safe_open(os.path.join(d, "model.safetensors"), framework="pt", device="cpu")
        self.host = {k: f.get_tensor(k) for k in ("embed_tokens.weight", "candidate_selector.predecessor_codebook",
                                                     "candidate_selector.successor_codebook",
                                                     "candidate_selector.hidden_projection.weight")}  # fmt: skip
        m = torch.load(os.path.join(d, "mask_embedding.pt"), map_location="cpu", weights_only=False)
        self.mask_emb = (m["embedding"] if isinstance(m, dict) else m).reshape(-1).to(torch.bfloat16)
        row = lambda k: self._dev(f.get_tensor(k).reshape(1, 1, 1, -1))  # noqa: E731
        self.fc, self.hidden_norm, self.norm = self._mm("fc.weight"), row("hidden_norm.weight"), row("norm.weight")
        self.lm_head = self._mm("lm_head.weight")
        self.L = []
        for i in range(self.nl):
            p = f"layers.{i}."
            a = p + "self_attn."
            self.L.append(dict(
                in_ln=row(p + "input_layernorm.weight"), post_ln=row(p + "post_attention_layernorm.weight"),
                q=self._mm(a + "q_proj.weight"), k=self._mm(a + "k_proj.weight"), v=self._mm(a + "v_proj.weight"),
                o=self._mm(a + "o_proj.weight"), q_norm=row(a + "q_norm.weight"), k_norm=row(a + "k_norm.weight"),
                gate=self._mm(p + "mlp.gate_proj.weight"), up=self._mm(p + "mlp.up_proj.weight"),
                down=self._mm(p + "mlp.down_proj.weight"),
                kp_attn=self._mm(p + "attention_conv.kernel_projection.weight"),
                kp_mlp=self._mm(p + "mlp_conv.kernel_projection.weight"),
                base_attn=self._base(f.get_tensor(p + "attention_conv.base_kernel")),
                base_mlp=self._base(f.get_tensor(p + "mlp_conv.base_kernel"))))  # fmt: skip
            log(f"dflash2 tp{self.tp}: layer {i} weights on device")
        E = torch.zeros(self.ng, self.H)
        E[torch.arange(self.H) // self.group, torch.arange(self.H)] = 1
        self.E = self._dev(E, FP32)
        Sh = torch.zeros(TILE, TILE)
        Sh[torch.arange(1, TILE), torch.arange(TILE - 1)] = 1
        self.Sh = self._dev(Sh.reshape(1, 1, TILE, TILE), FP32)
        R = torch.zeros(self.hd, self.hd)
        h2 = self.hd // 2
        R[torch.arange(h2) + h2, torch.arange(h2)] = -1
        R[torch.arange(h2), torch.arange(h2) + h2] = 1
        self.R = self._dev(R, FP32)
        # per chip: its kv heads' context K / V [1, nkv / tp, cap, hd] (one mesh tensor, every chip its own heads)
        self.Kc = [self._dev(torch.zeros(1, self.nkv_l, self.cap, self.hd)) for _ in range(self.nl)]
        self.Vc = [self._dev(torch.zeros(1, self.nkv_l, self.cap, self.hd)) for _ in range(self.nl)]
        self.x_in = self._dev(torch.zeros(1, 1, TILE, self.H))
        self.cos_b = self._dev(torch.zeros(1, 1, TILE, self.hd), FP32)
        self.sin_b = self._dev(torch.zeros(1, 1, TILE, self.hd), FP32)
        g = self.nh // self.nkv
        self.mask = self._dev(torch.zeros(1, 1, g * TILE, self.cap + TILE), FP32)
        self.taps = None
        self.out = None

    # ------------------------------------------------------------------ the converted cache
    def _mm(self, name: str):
        """a converted matmul weight [in, out] (bfloat8_b): the tp shards (split) or one tensor on every chip"""
        e = self.man["tensors"][name]
        shards = [ttnn.load_tensor(os.path.join(self.cache, p)) for p in e["paths"]]
        if len(shards) == 1:
            shards = shards * self.tp
        assert len(shards) == self.tp, (name, len(shards))
        return ttnn.to_device(ttnn.from_host_shards(shards, self.mesh.shape), self.mesh, memory_config=DRAM)

    # ------------------------------------------------------------------ collectives
    def _gather(self, x, dim: int):
        for ax in self.axes:
            x = ttnn.all_gather(x, dim=dim, cluster_axis=ax, memory_config=DRAM)
        return x

    def _reduce(self, x):
        """fp32 partials -> the sum on every chip (axis 0 then 1 on a 2-D mesh) -> bf16"""
        for ax in self.axes[::-1]:
            x = ttnn.all_reduce(x, cluster_axis=ax, memory_config=DRAM)
        return ttnn.typecast(x, BF16)

    # ------------------------------------------------------------------ context K / V
    def ctx_program(self):
        nrm = self._rms(self._gather(self._lin(self.taps, self.fc), 3), self.hidden_norm)
        for i, Ly in enumerate(self.L):
            k = self._rms(self._heads(self._lin(nrm, Ly["k"]), self.cap, self.nkv_l), Ly["k_norm"])
            ttnn.copy(self._rope(k, self.cos_c, self.sin_c), self.Kc[i])
            ttnn.copy(self._heads(self._lin(nrm, Ly["v"]), self.cap, self.nkv_l), self.Vc[i])

    # ------------------------------------------------------------------ the block
    def block_program(self):
        """the 8 layers + final norm + lm_head (this chip's vocab slice) top-16 -> self.out = (hid, vals, ids)"""
        import math

        g = self.nh // self.nkv
        scale = 1.0 / math.sqrt(self.hd)
        h, res = self.x_in, None
        for i, Ly in enumerate(self.L):
            res = h if res is None else ttnn.add(res, h)
            x = self._rms(res, Ly["in_ln"])
            cf = self._lin(x, Ly["kp_attn"])
            x = self._conv(x, cf, 0, Ly["base_attn"])
            q = self._rope(
                self._rms(self._heads(self._lin(x, Ly["q"]), TILE, self.nh_l), Ly["q_norm"]), self.cos_b, self.sin_b
            )
            k = self._rope(
                self._rms(self._heads(self._lin(x, Ly["k"]), TILE, self.nkv_l), Ly["k_norm"]), self.cos_b, self.sin_b
            )
            v = self._heads(self._lin(x, Ly["v"]), TILE, self.nkv_l)
            K_ = ttnn.typecast(ttnn.concat([self.Kc[i], k], dim=2), FP32)
            V_ = ttnn.typecast(ttnn.concat([self.Vc[i], v], dim=2), FP32)
            qg = ttnn.reshape(ttnn.typecast(q, FP32), (1, self.nkv_l, g * TILE, self.hd))
            s = ttnn.matmul(qg, ttnn.transpose(K_, 2, 3), compute_kernel_config=self.ckc, memory_config=DRAM)
            s = ttnn.add(ttnn.multiply(s, scale), self.mask)
            pr = ttnn.softmax(s, dim=-1, compute_kernel_config=self.ckc, numeric_stable=True)
            a = ttnn.matmul(pr, V_, compute_kernel_config=self.ckc, memory_config=DRAM)
            a = ttnn.reshape(ttnn.permute(ttnn.reshape(a, (1, self.nh_l, TILE, self.hd)), (0, 2, 1, 3)),
                             (1, 1, TILE, self.nh_l * self.hd))  # fmt: skip
            o = self._reduce(self._lin(ttnn.typecast(a, BF16), Ly["o"], FP32))
            o = self._conv(o, cf, 1, Ly["base_attn"])
            res = ttnn.add(res, o)
            x = self._rms(res, Ly["post_ln"])
            cf2 = self._lin(x, Ly["kp_mlp"])
            x = self._conv(x, cf2, 0, Ly["base_mlp"])
            y = self._reduce(self._lin(ttnn.multiply(ttnn.silu(self._lin(x, Ly["gate"])), self._lin(x, Ly["up"])),
                                       Ly["down"], FP32))  # fmt: skip
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
        """hid from chip 0 (replicated), candidates = the top-16 of the tp x 16 per-chip candidates (exact: each global
        top-16 entry is in its own slice's top-16), vocab ids offset by the chip's slice"""
        hid = ttnn.to_torch(ttnn.get_device_tensors(self.out[0])[0]).reshape(TILE, -1)[1 : K + 1].float()
        vs = [ttnn.to_torch(t).reshape(TILE, -1)[1 : K + 1].float() for t in ttnn.get_device_tensors(self.out[1])]
        ids = [ttnn.to_torch(t).reshape(TILE, -1)[1: K + 1].to(torch.int64) + r * self.V_l
               for r, t in enumerate(ttnn.get_device_tensors(self.out[2]))]  # fmt: skip
        v, i = torch.cat(vs, -1), torch.cat(ids, -1)
        top, j = torch.topk(v, self.top_k, dim=-1, largest=True, sorted=True)
        return dict(hid=hid, unary=top, cand=torch.gather(i, -1, j))
