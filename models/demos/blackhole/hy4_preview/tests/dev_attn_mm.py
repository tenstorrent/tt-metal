"""attn-fusion: the two small batched head matmuls of the decode attention (q_abs: [8,1,192]x[8,192,512]; w_uv: [8,1,512]x[8,512,256]) on ONE chip: stock vs program-config / dtype variants. us per call (trace) + PCC vs stock."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import p, pcc, traced_us
from models.demos.blackhole.hy4_preview.tt.hy4_common import compute_cfg

OUT = os.environ.get("OUT", "/tmp/attn_mm.json")
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=16384, trace_region_size=100_000_000)
res = {}
torch.manual_seed(0)


def up(t, dt=ttnn.bfloat16, mc=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        t, device=md, dtype=dt, layout=ttnn.TILE_LAYOUT, memory_config=mc, mesh_mapper=ttnn.ReplicateTensorToMesh(md)
    )


def d0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


try:
    for name, K, N in (("q_abs", 192, 512), ("w_uv", 512, 256)):
        x = torch.randn(1, 8, 1, K).bfloat16()
        w = (torch.randn(1, 8, K, N) / K**0.5).bfloat16()
        ref = torch.matmul(x.float(), w.float())
        xt = up(x)
        xl1 = up(x, mc=ttnn.L1_MEMORY_CONFIG)
        variants = {}
        for wn, wdt in (("bf16", ttnn.bfloat16), ("bf8", ttnn.bfloat8_b)):
            wt = up(w, wdt)
            for cn, ckc in (
                ("hifi4", compute_cfg(md, "hifi4", True)),
                ("hifi2", compute_cfg(md, "hifi2", True)),
                ("hifi2nf", compute_cfg(md, "hifi2", False)),
            ):
                variants[f"stock_{wn}_{cn}"] = lambda wt=wt, ckc=ckc, xt=xt: ttnn.matmul(
                    xt, wt, compute_kernel_config=ckc
                )
                variants[f"stockL1out_{wn}_{cn}"] = lambda wt=wt, ckc=ckc, xt=xt: ttnn.matmul(
                    xt, wt, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG
                )
                variants[f"stockL1in_{wn}_{cn}"] = lambda wt=wt, ckc=ckc, xl1=xl1: ttnn.matmul(
                    xl1, wt, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG
                )
                for pcn in (1, 2, 4):
                    if (N // 32) % pcn:
                        continue
                    for ibw in (2, 4, 8):
                        if (K // 32) % ibw:
                            continue
                        for sbw in (1, 2):
                            if pcn % sbw:
                                continue
                            try:
                                pc = ttnn.MatmulMultiCoreReuseProgramConfig(
                                    compute_with_storage_grid_size=(12, 10),
                                    in0_block_w=ibw,
                                    out_subblock_h=1,
                                    out_subblock_w=sbw,
                                    per_core_M=1,
                                    per_core_N=pcn,
                                )
                            except Exception as e:
                                continue
                            variants[
                                f"reuse_pcn{pcn}_ibw{ibw}_sbw{sbw}_{wn}_{cn}"
                            ] = lambda wt=wt, ckc=ckc, pc=pc, xt=xt: ttnn.matmul(
                                xt, wt, compute_kernel_config=ckc, program_config=pc
                            )
                            variants[
                                f"reuseL1_pcn{pcn}_ibw{ibw}_sbw{sbw}_{wn}_{cn}"
                            ] = lambda wt=wt, ckc=ckc, pc=pc, xl1=xl1: ttnn.matmul(
                                xl1,
                                wt,
                                compute_kernel_config=ckc,
                                program_config=pc,
                                memory_config=ttnn.L1_MEMORY_CONFIG,
                            )
        r = {}
        for vn, fn in variants.items():
            try:
                o = fn()
                pc_ = pcc(d0(o)[..., :1, :], ref)
                ttnn.deallocate(o)
                us = traced_us(md, fn, n=7, rep_=10)
                r[vn] = (round(us, 2), round(pc_, 5))
                p(name, vn, r[vn])
            except Exception as e:
                r[vn] = str(e)[:100]
                p(name, vn, "FAIL", str(e)[:100])
        res[name] = r
finally:
    json.dump(res, open(OUT, "w"), indent=1)
    ttnn.close_mesh_device(md)
