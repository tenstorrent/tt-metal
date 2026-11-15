"""pf-integrate S1: N-layer direct-to-pool prefill (Hy4PoolPrefill, HY4_POOL_PF=1) vs the replicated-cache chain prefill (Hy4AttentionPool/Hy4Attention path, then the same head) on ONE 1x8 slot.
HY4_NO_MOE=1 stubs every MoE MLP (y = 0.05 h) on both paths (the MoE block needs the 4x8 mesh); layer 0 is the real dense MLP, attention / indexer / IndexCache carry / iHC / head are real.
Env: NL (layers, 4), LENS (comma list of prompt lengths, 2048), NEEDLE (prompt source .pt, ids[0]), PF_ROWS (512)."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import pcc
from models.demos.blackhole.hy4_preview.tt.hy4_chain import Hy4Chain, dram_view
from models.demos.blackhole.hy4_preview.tt.hy4_pool_decode import PoolCtx
from models.demos.blackhole.hy4_preview.tt.hy4_pool_prefill_model import Hy4PoolPrefill
from models.demos.gpt_oss.config import mesh_1x8
from models.demos.gpt_oss.tt.ccl import CCLManager

R = lambda **kw: print("RES " + json.dumps(kw), flush=True)
NL = int(os.environ.get("NL", "4"))
PF = int(os.environ.get("PF_ROWS", "512"))
LENS = [int(x) for x in os.environ.get("LENS", "2048").split(",")]
os.environ["HY4_NO_MOE"] = "1"


class Shim:  # the part of Hy4PoolDecoder that Hy4PoolPrefill uses
    pass


def build(md, mc, ccl, pf, pages):
    os.environ["HY4_POOL"] = "1"
    os.environ["HY4_POOL_PF"] = "1" if pf else "0"
    ch = Hy4Chain(md, mc, ccl, ckpt="/ckpt", kv_fmt="scaled_fp8")
    ch.add_layers(0, NL, log=lambda s: None)
    ch.build_head()
    if not pf:
        return ch, None
    s = Shim()
    s.ch, s.pages = ch, pages
    s.ctx = PoolCtx(md, mc, ccl, pages, rows=1, S=1, mode="topk", bucket_tokens=pages * 512)
    for ly in ch.layers:
        kv = ttnn.empty(
            [1, 1, pages * 64, 656],
            dtype=ttnn.fp8_e4m3,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=md,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        kp = (
            ttnn.empty(
                [pages, 1, 64, 128],
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=md,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if ly.attn.has_indexer
            else None
        )
        ly.attn.init_pool(s.ctx, kv, kp)
    return ch, s


def rep_prefill(ch, md, ids_list, L):
    """replicated path: chunks of PF rows, then head at the last token. -> (hidden [6144], logits [V])"""
    ch.reset()
    nch = -(-L // PF)
    ibl = {}
    for c in range(nch):
        ids = torch.full((1, PF), 120002, dtype=torch.long)
        seg = ids_list[c * PF : (c + 1) * PF]
        ids[0, : len(seg)] = torch.tensor(seg)
        x4, ibl = ch.prefill(ids, start=c * PF, ids_by_layer=ibl)
        if c == nch - 1:
            hn, logits = ch.head(x4)
            last = L - 1 - c * PF
            h = ttnn.to_torch(ttnn.get_device_tensors(hn)[0]).float().reshape(PF, -1)[last]
            lg = (
                ttnn.to_torch(
                    logits, mesh_composer=ttnn.ConcatMesh2dToTensor(md, dims=(0, 3), mesh_shape=tuple(md.shape))
                )
                .reshape(PF, -1)[last]
                .float()
            )
        ttnn.synchronize_device(md)
        ttnn.deallocate(x4)
    return h, lg


def main():
    d = torch.load(os.environ.get("NEEDLE", "/out/data/needle_64k.pt"))
    ids_all = d["ids"][0]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=200_000_000)
    try:
        mc = mesh_1x8()
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        pages = (-(-max(LENS) // PF)) * PF // 512 + 2
        chA, shim = build(md, mc, ccl, True, pages)
        chB, _ = build(md, mc, ccl, False, pages)
        R(event="built", NL=NL, pages=pages, **{k: v for k, v in dram_view(md).items() if k in ("alloc_per_bank",)})
        pp = Hy4PoolPrefill(
            shim,
            pf_rows=PF,
            log=lambda **kw: R(**kw) if kw.get("c", 0) < 2 else None,
            mem_sample_s=float(os.environ.get("MEMS", "0")),
        )
        ok_all = True
        for L in LENS:
            prompt = list(ids_all[:L])
            hB, lB = rep_prefill(chB, md, prompt, L)
            r1 = pp.prefill([prompt], keep_hidden=True, keep_logits=True)
            r2 = pp.prefill([prompt], keep_hidden=True, keep_logits=True)
            hA, lA = r1.hidden[0], r1.logits[0]
            ph, pl = pcc(hA, hB), pcc(lA, lB)
            det = pcc(r1.hidden[0], r2.hidden[0])
            top1 = int(lA.argmax()) == int(lB.argmax())
            ok = ph >= 0.995 and top1 and det > 0.99999 and r1.first == r2.first
            ok_all &= ok
            R(
                event="gate",
                L=L,
                pcc_hidden=round(ph, 6),
                pcc_logits=round(pl, 6),
                top1_equal=top1,
                first_pf=r1.first,
                first_rep=int(lB.argmax()),
                determinism_pcc=round(det, 8),
                sec_pf=round(r1.total_sec, 1),
                chunk_sec_mean=round(sum(r1.chunk_sec) / len(r1.chunk_sec), 3),
                ok=ok,
                **{k: v for k, v in dram_view(md).items() if k in ("alloc_per_bank",)},
            )
        R(event="verdict", ok=ok_all)
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
