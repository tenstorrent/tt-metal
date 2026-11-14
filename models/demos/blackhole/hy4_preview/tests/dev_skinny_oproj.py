# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""skinny-kernels X26: Hy4 attention o_proj (K = 2048 per mesh column, N = 6144, bfp8) custom bank-local kernel (tt/hy4_skinny.SkinnyLinear) vs the stock tuned ttnn.linear (HY4_OPT tune config), 1x8 slot.
Activations: the real layer-1 o_lat of the attention goldens (column 0, rows of chunk0) x the REAL w_uv_t / gate-gated like Hy4AttentionOpt.back; real o_proj weights.
PCC of the per-column partial outputs vs the stock op (all 8 columns), traced us of both. Env: SK_T=1,2,4,8, SK_VARIANTS, OUT."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.config import Hy4Config
from models.demos.blackhole.hy4_preview.tests.dev_common import p, pcc, traced_us
from models.demos.blackhole.hy4_preview.tt.hy4_attn_opt import CAPS, cfg_1d
from models.demos.blackhole.hy4_preview.tt.hy4_attn_weights import build_attn_host_weights
from models.demos.blackhole.hy4_preview.tt.hy4_common import compute_cfg, upload
from models.demos.blackhole.hy4_preview.tt.hy4_skinny import SkinnyLinear

CKPT = os.environ.get("HY4_CKPT", "/mnt/tt-data/Hy4-Preview")
OUT = os.environ.get("OUT", "/tmp/skinny_oproj.json")
TS = [int(t) for t in os.environ.get("SK_T", "1,2,4,8").split(",")]
VARS = os.environ.get("SK_VARIANTS", "base").split("|")
LAYER = int(os.environ.get("SK_LAYER", "1"))


def parse(v):
    name, _, kv = v.partition(":")
    d = {}
    for item in filter(None, kv.split(";")):
        k, val = item.split("=")
        d[k] = val if k == "remap" else int(val)
    return name, d


def load_sd(layer):
    import json as _j

    from safetensors import safe_open

    wm = _j.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
    pre = f"model.layers.{layer}."
    sd = {}
    for k, f in wm.items():
        if k.startswith(pre + "self_attn."):
            with safe_open(f"{CKPT}/{f}", "pt") as fh:
                sd[k[len(pre) :]] = fh.get_tensor(k)
    return sd


cfg = Hy4Config.from_checkpoint(CKPT)
hw = build_attn_host_weights(cfg, load_sd(LAYER), 8)
olat = torch.load("/mnt/tt-data/ssinghal/hy4-logs/program/layer-attn/dump_L1_chunk0.pt")["o_lat_c0"].float()  # [1,8,512,512] column 0
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=100_000_000)
cols = 8
res = {}
try:
    w_o = upload(md, hw["o_proj"].tensor, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, shard_dim=2)
    ckc = compute_cfg(md, "hifi4", True)
    sks = {}
    for v in VARS:
        name, kw = parse(v)
        sks[name] = SkinnyLinear(md, hw["o_proj"].tensor.reshape(-1, 6144), **kw)
        p(name, "cores", [(c.x, c.y) for c in sks[name].cores])
    K, N, cap = CAPS["o_proj"]
    pc = cfg_1d(K, N, cap, None)
    uvt = hw["w_uv_t"].tensor  # [1,64,512,256]
    gen = torch.Generator().manual_seed(3)
    for T in TS:
        # og_c = concat_heads(o_lat[rows] @ w_uv_t[heads of column c]) * sigmoid(g)  -> [1,1,T,16384]
        rows = olat[:, :, 17 : 17 + T, :]  # [1,8,T,512]
        og = []
        for c in range(cols):
            x = rows @ uvt[:, 8 * c : 8 * c + 8]  # [1,8,T,256]
            x = x.permute(0, 2, 1, 3).reshape(1, 1, T, 2048)
            g = torch.sigmoid(0.5 * torch.randn(1, 1, T, 2048, generator=gen))
            og.append(x * g)
        og = torch.cat(og, dim=3).to(torch.bfloat16)
        tt_og = ttnn.from_torch(
            og, device=md, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(None, 3), mesh_shape=(1, 8)),
        )
        stock = lambda: ttnn.linear(tt_og, w_o, compute_kernel_config=ckc, program_config=pc)
        o = stock()
        ttnn.synchronize_device(md)
        s_out = ttnn.to_torch(o, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).float()[:, 0, :T]  # [8, T, 6144]
        ttnn.deallocate(o)
        ref = torch.stack([og[0, 0, :, 2048 * c : 2048 * (c + 1)].float() @ hw["o_proj"].tensor.reshape(16384, 6144)[2048 * c : 2048 * (c + 1)] for c in range(cols)])
        us_s = traced_us(md, stock)
        p(f"T={T} stock  {us_s:6.1f} us  PCC vs torch (all cols) {pcc(s_out, ref):.6f}")
        res[f"T{T}_stock"] = {"us": us_s, "pcc_ref": pcc(s_out, ref)}
        for name, sk in sks.items():
            try:
                o = sk(tt_og, T)
                ttnn.synchronize_device(md)
                k_out = ttnn.to_torch(o, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).float()[:, 0, :T]
                ttnn.deallocate(o)
                us = traced_us(md, lambda: sk(tt_og, T))
                r = {"us": us, "pcc_stock": pcc(k_out, s_out), "pcc_ref": pcc(k_out, ref), "min_col_pcc_stock": min(pcc(k_out[c], s_out[c]) for c in range(cols))}
                res[f"T{T}_{name}"] = r
                p(f"T={T} skinny {name:6s} {us:6.1f} us  PCC vs stock {r['pcc_stock']:.6f} (min col {r['min_col_pcc_stock']:.6f})  vs torch {r['pcc_ref']:.6f}")
            except Exception as e:
                res[f"T{T}_{name}"] = {"err": str(e)[:300]}
                p(f"T={T} skinny {name} ERROR {str(e)[:300]}")
finally:
    ttnn.close_mesh_device(md)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    json.dump(res, open(OUT, "w"), indent=1)
