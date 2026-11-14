# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""skinny-kernels X27: Hy4 fused h-proj (q_a | kv_a | gate | idx_wk | idx_wp, 3040 cols per mesh column on a full-indexer layer, 2880 on a shared layer; bfp8) on the bank-local kernel
(tt/hy4_skinny.SkinnyProj, output split in the writer) vs the stock fuseproj path of Hy4AttentionOpt (tuned 1D ttnn.linear + the 4-5 ttnn.slice ops). 1x8 slot, real weights, real layer-1 MoE input
activations as h (post-attention hidden states: same magnitude class).  Env: SK_T=1,4,8  SK_LAYER=1 (full) / 2 (shared)  SK_VARIANTS  OUT."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.config import Hy4Config
from models.demos.blackhole.hy4_preview.tests.dev_common import ACTS, p, pcc, traced_us
from models.demos.blackhole.hy4_preview.tt.hy4_attn_opt import cfg_1d
from models.demos.blackhole.hy4_preview.tt.hy4_attn_weights import build_attn_host_weights
from models.demos.blackhole.hy4_preview.tt.hy4_common import compute_cfg, upload
from models.demos.blackhole.hy4_preview.tt.hy4_skinny import SkinnyProj

CKPT = os.environ.get("HY4_CKPT", "/mnt/tt-data/Hy4-Preview")
OUT = os.environ.get("OUT", "/tmp/skinny_proj.json")
TS = [int(t) for t in os.environ.get("SK_T", "1,4,8").split(",")]
VARS = os.environ.get("SK_VARIANTS", "base").split("|")
LAYER = int(os.environ.get("SK_LAYER", "1"))
HID = 6144


def parse(v):
    name, _, kv = v.partition(":")
    d = {}
    for item in filter(None, kv.split(";")):
        k, val = item.split("=")
        d[k] = val if k == "remap" else int(val)
    return name, d


def load_sd(layer):
    from safetensors import safe_open

    wm = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
    pre = f"model.layers.{layer}."
    sd = {}
    for k, f in wm.items():
        if k.startswith(pre + "self_attn."):
            with safe_open(f"{CKPT}/{f}", "pt") as fh:
                sd[k[len(pre) :]] = fh.get_tensor(k)
    return sd


cfg = Hy4Config.from_checkpoint(CKPT)
hw = build_attn_host_weights(cfg, load_sd(LAYER), 8)
has_idx = "idx_wk" in hw
parts = []
for c in range(8):
    p_ = [hw["q_a"].tensor[..., c * 256 : (c + 1) * 256], hw["kv_a"].tensor, hw["gate"].tensor[..., c * 2048 : (c + 1) * 2048]]
    if has_idx:
        p_ += [hw["idx_wk"].tensor, hw["idx_wp"].tensor]
    parts.append(torch.cat(p_, dim=-1))
wcat = torch.cat(parts, dim=-1)
nloc = wcat.shape[-1] // 8
offs = [0, 256, 256 + 576, 256 + 576 + 2048] + ([256 + 576 + 2048 + 128, nloc] if has_idx else [nloc])
widths = [offs[i + 1] - offs[i] for i in range(len(offs) - 1) if offs[i + 1] > offs[i]]
acts = torch.load(ACTS + "acts_l0_l1.pt")
X = torch.cat([h.reshape(-1, HID) for h, y, r in acts["moe"]]).to(torch.bfloat16)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=100_000_000)
res = {"layer": LAYER, "widths": widths}
try:
    wf = upload(md, wcat, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, shard_dim=3)
    ckc = compute_cfg(md, "hifi4", True)
    pc = cfg_1d(6144, nloc, 32)
    sks = {}
    for v in VARS:
        name, kw = parse(v)
        sks[name] = SkinnyProj(md, wcat, widths, **kw)
        p(name, "cores", [(c.x, c.y) for c in sks[name].cores], "valid", sks[name].valid, "PNC", sks[name].PNC)
    for T in TS:
        h = X[torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(5))[:T]]
        tt_h = ttnn.from_torch(h.reshape(1, 1, T, HID), device=md, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(md))

        def stock():
            fp = ttnn.linear(tt_h, wf, compute_kernel_config=ckc, program_config=pc if T <= 4 else None)
            sl = [ttnn.slice(fp, [0, 0, 0, offs[i]], [1, 1, T, offs[i + 1]]) for i in range(len(offs) - 1) if offs[i + 1] > offs[i]]
            ttnn.deallocate(fp)
            return sl

        def host(ts):
            return [ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).float()[:, 0, :T] for t in ts]

        s_parts = stock()
        ttnn.synchronize_device(md)
        s_h = host(s_parts)
        for t in s_parts:
            ttnn.deallocate(t)
        us_s = traced_us(md, stock)
        p(f"T={T} stock (linear + {len(widths) - 1}+ slices) {us_s:6.1f} us")
        res[f"T{T}_stock"] = {"us": us_s}
        for name, sk in sks.items():
            try:
                o = sk(tt_h, T)
                ttnn.synchronize_device(md)
                k_h = host(o)
                for t in o:
                    ttnn.deallocate(t)
                us = traced_us(md, lambda: sk(tt_h, T))
                pc_ = [pcc(k_h[i], s_h[i]) for i in range(len(s_h))]
                res[f"T{T}_{name}"] = {"us": us, "pcc_parts": pc_}
                p(f"T={T} skinny {name:6s} {us:6.1f} us  PCC vs stock per part {['%.6f' % v for v in pc_]}")
            except Exception as e:
                res[f"T{T}_{name}"] = {"err": str(e)[:300]}
                p(f"T={T} skinny {name} ERROR {str(e)[:300]}")
finally:
    ttnn.close_mesh_device(md)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    json.dump(res, open(OUT, "w"), indent=1)
