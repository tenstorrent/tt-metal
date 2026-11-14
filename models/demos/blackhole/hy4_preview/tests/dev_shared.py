# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""moe-perf X21: shared expert partial (TP8 over mesh columns) variants, traced us and PCC vs the torch reference (real layer-1 shared weights + real activations).
Variants: mid-tensor memory (DRAM/L1), fidelity, weight dtype, program-config knobs (pn1/bw1/pn2/bw2), T. Env: HY4_T=1, SH_VARIANTS=comma list of names."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import ACTS, close_mesh, open_mesh, p, pcc, traced_us
from models.demos.blackhole.hy4_preview.tt.hy4_dense import make_shared
from models.demos.blackhole.hy4_preview.tt.hy4_moe_weights import load_moe_layer

T = int(os.environ.get("HY4_T", "1"))
OUT = os.environ.get("OUT", "/tmp/shared.json")
HID = 6144
L1 = ttnn.L1_MEMORY_CONFIG
DR = ttnn.DRAM_MEMORY_CONFIG

VARS = {
    "base": {},
    "l1mid": {"mid": L1},
    "l1all": {"mid": L1, "outmc": L1},
    "lofi": {"fid": "LoFi"},
    "hifi2": {"fid": "HiFi2"},
    "bfp4": {"dtype": ttnn.bfloat4_b},
    "pn1_2": {"pn1": 2},
    "bw1_16": {"bw1": 16},
    "bw1_64": {"bw1": 64},
    "bw2_8": {"bw2": 8},
    "bw2_2": {"bw2": 2},
    "pn2_2": {"pn2": 2},
    "pn2_8": {"pn2": 8},
    "l1_hifi2": {"mid": L1, "fid": "HiFi2", "outmc": L1},
    "split": {"split": True},
    "split_l1": {"split": True, "mid": L1, "outmc": L1},
    "split_bw16": {"split": True, "bw1s": 16},
    "dram": {"dram": True},
    "dram_nores": {"dram": True, "nores": True},
    "pre_gu": {"prefix": 1},
    "pre_slice": {"prefix": 2},
    "pre_mul": {"prefix": 3},
}


class Fast:
    def __init__(
        self,
        md,
        W,
        mid=DR,
        outmc=DR,
        fid=None,
        dtype=None,
        split=False,
        bw1s=32,
        prefix=9,
        dram=False,
        nores=False,
        **kw,
    ):
        k = dict(kw)
        self.split, self.prefix, self.dram, self.nores = split, prefix, dram, nores
        if fid:
            k["fid"] = fid
        if dtype is not None:
            k["dtype"] = dtype
        self.s = make_shared(md, *W, decode=True, **k)
        self.mid, self.outmc = mid, outmc
        if dram:
            import math

            rows, cols = tuple(md.shape)
            cm = ttnn.ShardTensor2dMesh(md, dims=(None, 3), mesh_shape=(rows, cols))
            cm2 = ttnn.ShardTensor2dMesh(md, dims=(None, 2), mesh_shape=(rows, cols))
            nb = md.dram_grid_size().x
            self.nb = nb
            K, ic = self.s.dim, self.s.ic
            dcrs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nb - 1, 0))})
            dmem = lambda k, n: ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(dcrs, (k, math.ceil(n / (32 * nb)) * 32), ttnn.ShardOrientation.ROW_MAJOR),
            )
            up = lambda t, m, mm: ttnn.from_torch(
                t.contiguous(),
                device=md,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=m,
                mesh_mapper=mm,
            )
            self.dw0, self.dw1 = up(W[0], dmem(K, ic), cm), up(W[1], dmem(K, ic), cm)
            self.dw2 = up(W[2], dmem(ic, K), cm2)
            nc = nb
            grid = md.compute_with_storage_grid_size()
            crs = ttnn.num_cores_to_corerangeset(nc, grid, row_wise=True)
            sh = lambda width: ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(crs, (32, width // nc), ttnn.ShardOrientation.ROW_MAJOR),
            )
            self.m_in, self.m_mid, self.m_out = sh(K), sh(ic), sh(K)
            pcf = lambda n, bw: ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=bw, per_core_M=1, per_core_N=math.ceil(n / 32 / nc), fused_activation=None
            )
            self.dpc1, self.dpc2 = pcf(ic, K // 32 // nc), pcf(K, ic // 32 // nc)
        if split:
            from models.demos.blackhole.deepseek_v41_flash.tt.shared_expert_v2 import _cfg1d

            rows, cols = tuple(md.shape)
            cm = ttnn.ShardTensor2dMesh(md, dims=(None, 3), mesh_shape=(rows, cols))
            up = lambda t: ttnn.from_torch(
                t.contiguous(),
                device=md,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=DR,
                mesh_mapper=cm,
            )
            self.w0, self.w1 = up(W[0]), up(W[1])
            self.pcs = _cfg1d((12, 8), self.s.ic // 32, 1, self.s.dim // 32, bw1s)

    def forward(self, h):
        s = self.s
        kw = lambda pc, dt, mc: dict(
            dtype=dt, compute_kernel_config=s.ckc, memory_config=mc, **({"program_config": pc} if pc else {})
        )
        if self.dram:
            hs = ttnn.to_memory_config(h, self.m_in)
            g = ttnn.linear(hs, self.dw0, **kw(self.dpc1, s.mid_dtype, self.m_mid))
            u = ttnn.linear(hs, self.dw1, **kw(self.dpc1, s.mid_dtype, self.m_mid))
            ttnn.deallocate(hs)
            act = ttnn.multiply(
                g,
                u,
                input_tensor_a_activations=s.act_a,
                input_tensor_b_activations=s.act_b,
                dtype=s.mid_dtype,
                memory_config=self.m_mid,
            )
            ttnn.deallocate(g)
            ttnn.deallocate(u)
            o = ttnn.linear(act, self.dw2, **kw(self.dpc2, s.out_dtype, self.m_out))
            ttnn.deallocate(act)
            r = ttnn.to_memory_config(o, DR)
            ttnn.deallocate(o)
            return r
        if self.split:
            g = ttnn.linear(h, self.w0, **kw(self.pcs, s.mid_dtype, self.mid))
            u = ttnn.linear(h, self.w1, **kw(self.pcs, s.mid_dtype, self.mid))
            act = ttnn.multiply(
                g,
                u,
                input_tensor_a_activations=s.act_a,
                input_tensor_b_activations=s.act_b,
                dtype=s.mid_dtype,
                memory_config=self.mid,
            )
            return ttnn.linear(act, s.w2, **kw(s.pc2, s.out_dtype, self.outmc))
        gu = ttnn.linear(h, s.w01, **kw(s.pc1, s.mid_dtype, self.mid))
        if self.prefix == 1:
            return gu
        g, u = gu[:, :, :, : s.ic], gu[:, :, :, s.ic :]
        if self.prefix == 2:
            return [g, u]
        act = ttnn.multiply(
            g,
            u,
            input_tensor_a_activations=s.act_a,
            input_tensor_b_activations=s.act_b,
            dtype=s.mid_dtype,
            memory_config=self.mid,
        )
        if self.prefix == 3:
            return act
        return ttnn.linear(act, s.w2, **kw(s.pc2, s.out_dtype, self.outmc))


acts = torch.load(ACTS + "acts_l0_l1.pt")
X = torch.cat([h.reshape(-1, HID) for h, y, r in acts["moe"]])
md = open_mesh()
rows, cols = tuple(md.shape)
res = {"T": T, "v": {}}
try:
    W = load_moe_layer(1, load_experts=False)
    Wsh = [W[k][256] for k in ("shared_w0", "shared_w1", "shared_w2")]
    B = T * rows
    x = X[torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(7))[:B]].to(torch.bfloat16)
    tt_x = ttnn.from_torch(
        x.reshape(1, 1, B, HID),
        device=md,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=DR,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(2, None), mesh_shape=(rows, cols)),
    )
    comp = ttnn.ConcatMesh2dToTensor(md, dims=(2, 3), mesh_shape=(rows, cols))
    # reference: full shared expert (sum over columns of the partials == replicated expert)
    w0, w1, w2 = [w.reshape(w.shape[-2], w.shape[-1]).float() for w in Wsh]
    ref = (torch.nn.functional.silu((x.float() @ w0).clamp(max=10.0)) * (x.float() @ w1).clamp(-10, 10)) @ w2
    names = os.environ.get("SH_VARIANTS", ",".join(VARS)).split(",")
    for n in names:
        try:
            f = Fast(md, Wsh, **VARS[n])
            if f.prefix != 9:
                us = traced_us(md, lambda: f.forward(tt_x))
                res["v"][n] = {"us": us}
                p(f"T={T} shared {n:10s} {us:6.1f} us (prefix)")
                continue
            o = f.forward(tt_x)
            ttnn.synchronize_device(md)
            parts = ttnn.to_torch(
                o, mesh_composer=ttnn.ConcatMesh2dToTensor(md, dims=(0, 3), mesh_shape=(rows, cols))
            ).float()  # [rows*?]
            parts = parts.reshape(-1, B // rows * 0 + 1, 1, 1) if False else parts
            # partial per (row, col): concat over cols at dim 3 -> [rows,1,T,cols*HID]
            full = parts.reshape(rows, T, cols, HID).sum(2).reshape(B, HID)
            ttnn.deallocate(o)
            us = traced_us(md, lambda: f.forward(tt_x))
            c = pcc(full, ref)
            res["v"][n] = {"us": us, "pcc": c}
            p(f"T={T} shared {n:10s} {us:6.1f} us  PCC vs torch {c:.5f}")
            del f
        except Exception as e:
            res["v"][n] = {"err": str(e)[:200]}
            p(f"T={T} shared {n} ERROR {str(e)[:160]}")
finally:
    close_mesh(md)
    json.dump(res, open(OUT, "w"), indent=1)
