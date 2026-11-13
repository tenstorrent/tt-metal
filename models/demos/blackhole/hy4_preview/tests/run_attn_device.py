# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""S1 device test of the Hy4 attention layer vs the oracle golden (see gen_attn_golden.py), run inside the DSV4 image by run_attn.sh (mesh 4x8).
Env: LAYER (0|1|2), MODE (prefill|decode), KVFMT (bf16|scaled_fp8), CHUNKS (default 6), WDT (bf16|bfp8), GOLD, HY4_CKPT, TIMING (1: repeat the last chunk for a warm latency).
Prints 'RES {json}' lines (one per chunk / decode case) and a 'SUMMARY {json}' line."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.config import Hy4Config
from models.demos.blackhole.hy4_preview.tt.hy4_attention import Hy4Attention
from models.demos.blackhole.hy4_preview.tt.hy4_common import to_host, upload
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

GOLD = os.environ.get("GOLD", "/gold/golden")
CKPT = os.environ.get("HY4_CKPT", "/ckpt")
L = int(os.environ.get("LAYER", "1"))
MODE = os.environ.get("MODE", "prefill")
KVFMT = os.environ.get("KVFMT", "bf16")
CHUNK = 512
NCH = int(os.environ.get("CHUNKS", "6"))
WDT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[os.environ.get("WDT", "bf16")]


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


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


def idx_setmatch(dev_ids, ref_ids):
    """per-row |A & B| / |A| with A = reference set (-1 = unused); returns mean overlap and fraction of rows with identical sets."""
    ov, same = [], 0
    for d, r in zip(dev_ids, ref_ids):
        a = set(r[r >= 0].tolist())
        b = set(d[(d >= 0) & (d < 2**31)].tolist())
        ov.append(len(a & b) / max(1, len(a)))
        same += a == b
    return sum(ov) / len(ov), same / len(ov)


def main():
    cfg = Hy4Config.from_checkpoint(CKPT)
    g = torch.load(f"{GOLD}/L{L}.pt")
    conv = g["conv"]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(4, 8), l1_small_size=16384, trace_region_size=100_000_000)
    try:
        mc = mesh_4x8()
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        sd = load_sd(L)
        t0 = time.time()
        attn = Hy4Attention(
            md,
            mc,
            ccl,
            cfg,
            sd,
            L,
            kv_fmt=KVFMT,
            rope_interleave=conv["rope_interleave"],
            idx_rope_interleave=conv["idx_rope_interleave"],
            wdtype=WDT,
        )
        print(
            f"build {time.time()-t0:.1f}s layer {L} kind {cfg.layer(L).indexer} kv {KVFMT} wdt {os.environ.get('WDT','bf16')}",
            flush=True,
        )
        if MODE == "prefill":
            run_prefill(md, attn, g, cfg)
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def run_prefill(md, attn, g, cfg):
    h_all, out_all = g["h"], g["out"]
    idx_gold = g["idx"]  # [S,2048] int64 (-1 unused), index source = g["idx_source"]
    shared = not attn.has_indexer
    res = []
    for i in range(NCH):
        p0 = i * CHUNK
        h = upload(md, h_all[p0 : p0 + CHUNK].reshape(1, 1, CHUNK, -1).to(torch.bfloat16))
        ids_in = None
        if shared:
            ix = idx_gold[p0 : p0 + CHUNK].to(torch.int32).reshape(1, 1, CHUNK, -1)
            ids_in = upload(md, ix, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        snap = (
            attn.cache.rows,
            attn.cache.length,
            attn.indexer.keys if attn.indexer else None,
            attn.indexer.length if attn.indexer else 0,
        )
        attn.taps = {}
        t = time.perf_counter()
        out, ids = attn.prefill_chunk(h, p0, ids_in)
        ttnn.synchronize_device(md)
        dt_first = time.perf_counter() - t
        o = to_host(out, md).reshape(CHUNK, -1).float()
        ref = out_all[p0 : p0 + CHUNK]
        r = {
            "layer": L,
            "mode": "prefill",
            "kv": KVFMT,
            "chunk": i,
            "p0": p0,
            "pcc": round(pcc(o, ref), 6),
            "max_abs": round((o - ref).abs().max().item(), 5),
            "ref_absmax": round(ref.abs().max().item(), 4),
            "eager_ms_first": round(dt_first * 1e3, 2),
        }
        # intermediates vs golden
        tp = attn.taps
        lat = to_host(tp["lat"], md).reshape(CHUNK, -1).float()
        kpe = to_host(tp["k_pe"], md).reshape(CHUNK, -1).float()
        r["pcc_lat"] = round(pcc(lat, g["lat"][p0 : p0 + CHUNK]), 6)
        r["pcc_k_pe"] = round(pcc(kpe, g["k_pe"][p0 : p0 + CHUNK]), 6)
        r["pcc_qa"] = round(pcc(to_host(tp["qa"], md).reshape(CHUNK, -1).float(), g["qa"][p0 : p0 + CHUNK]), 6)
        if not shared:
            dev_ids = to_host(ids, md).reshape(CHUNK, -1).to(torch.int64)
            dev_ids = torch.where(dev_ids >= 2**31, torch.full_like(dev_ids, -1), dev_ids)
            ov, same = idx_setmatch(dev_ids, idx_gold[p0 : p0 + CHUNK])
            r["idx_overlap"], r["idx_rows_identical"] = round(ov, 5), round(same, 4)
        if os.environ.get("TIMING", "0") == "1" and i == NCH - 1:
            ts = []
            for _ in range(3):
                saved = (
                    attn.cache.rows,
                    attn.cache.length,
                    attn.indexer.keys if attn.indexer else None,
                    attn.indexer.length if attn.indexer else 0,
                )
                attn.cache.rows, attn.cache.length = snap[0], snap[1]
                if attn.indexer:
                    attn.indexer.keys, attn.indexer.length = snap[2], snap[3]
                attn.taps = None
                ttnn.synchronize_device(md)
                t = time.perf_counter()
                out2, _ = attn.prefill_chunk(h, p0, ids_in)
                ttnn.synchronize_device(md)
                ts.append(time.perf_counter() - t)
            r["eager_ms_warm"] = round(min(ts) * 1e3, 2)
        res.append(r)
        print("RES " + json.dumps(r), flush=True)
    print("SUMMARY " + json.dumps({"layer": L, "kv": KVFMT, "min_pcc": min(x["pcc"] for x in res)}), flush=True)


if __name__ == "__main__":
    main()
