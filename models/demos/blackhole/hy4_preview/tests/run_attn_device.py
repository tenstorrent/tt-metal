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
from models.demos.blackhole.hy4_preview.tt.hy4_attn_ref import column_intermediates
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
    md = ttnn.open_mesh_device(ttnn.MeshShape(4, 8), l1_small_size=16384, trace_region_size=800_000_000)
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


def trace_time(md, fn, mk_tabs, reps=5):
    """device-side latency of one call: fn() re-runs the layer with append=False (static shapes) under a trace; returns ms per call."""
    global tabs_t
    tabs_t = mk_tabs()
    fn()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    fn()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    ts = []
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(reps):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        ts.append((time.perf_counter() - t) / reps)
    ttnn.release_trace(md, tid)
    return {"traced_ms": round(min(ts) * 1e3, 3)}


def run_prefill(md, attn, g, cfg):
    global HW
    HW = attn.host_w
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
        ov = None
        if os.environ.get("IDS", "dev") != "dev":
            ix = idx_gold[p0 : p0 + CHUNK].clone()
            if os.environ["IDS"] == "host_sorted":
                ix = torch.where(ix >= 0, ix, torch.full_like(ix, 2**31 - 1)).sort(-1).values
                ix = torch.where(ix == 2**31 - 1, torch.full_like(ix, -1), ix)
            ov = upload(md, ix.to(torch.int32).reshape(1, 1, CHUNK, -1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        out, ids = attn.prefill_chunk(h, p0, ids_in, ids_override=ov)
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
        if os.environ.get("DIAG", "1") == "1":
            n = p0 + CHUNK
            rows = torch.cat([g["lat"][:n], g["k_pe"][:n]], -1)
            ix = idx_gold[p0 : p0 + CHUNK]
            allowed = torch.zeros(CHUNK, n + 1, dtype=torch.bool)
            allowed.scatter_(1, torch.where(ix >= 0, ix, torch.full_like(ix, n)), True)
            allowed = allowed[:, :n] & (torch.arange(n)[None, :] <= torch.arange(p0, n)[:, None])
            for c in (0, 5):
                ci = column_intermediates(cfg, HW, g["qa"][p0:n], rows, allowed, c, p0, g["conv"]["rope_interleave"])
                r[f"pcc_qfull_c{c}"] = round(pcc(to_host(tp["qfull"], md, col=c).float(), ci["qfull"]), 6)
                r[f"pcc_olat_c{c}"] = round(pcc(to_host(tp["o_lat"], md, col=c).float(), ci["o_lat"]), 6)
                r[f"pcc_o_cpu_c{c}"] = round(pcc(ci["o"], g["o"][p0:n, c * 2048 : (c + 1) * 2048]), 7)
        if os.environ.get("IDCHK", "0") == "1" and not shared:
            allid = [to_host(ids, md, col=c).reshape(CHUNK, -1).to(torch.int64) for c in range(8)]
            for c in range(8):
                d = torch.where(allid[c] >= 2**31, torch.full_like(allid[c], -1), allid[c])
                nbad = 0
                for rr in range(CHUNK):
                    v = d[rr][d[rr] >= 0]
                    want = set(idx_gold[p0 + rr][idx_gold[p0 + rr] >= 0].tolist())
                    if set(v.tolist()) != want or len(v) != len(want):
                        nbad += 1
                        if nbad <= 3:
                            print(
                                f"IDBAD col {c} row {rr}: dev n={len(v)} want n={len(want)} extra={sorted(set(v.tolist())-want)[:5]} missing={sorted(want-set(v.tolist()))[:5]}",
                                flush=True,
                            )
                print(f"IDCHK chunk {i} col {c}: rows with wrong id set: {nbad}", flush=True)
        if os.environ.get("DUMP", "0") == "1":
            torch.save(
                {
                    "o_lat_c0": to_host(tp["o_lat"], md, col=0).float(),
                    "ids": to_host(ids, md),
                    "qfull_c0": to_host(tp["qfull"], md, col=0).float(),
                },
                f"/out/dump_L{L}_chunk{i}.pt",
            )
        po = []
        for c in (0, 3, 7):
            oc = to_host(tp["o"], md, col=c).reshape(CHUNK, -1).float()
            po.append(round(pcc(oc, g["o"][p0 : p0 + CHUNK, c * 2048 : (c + 1) * 2048]), 6))
        r["pcc_o_cols_0_3_7"] = po
        if not shared:
            dev_ids = to_host(ids, md).reshape(CHUNK, -1).to(torch.int64)
            dev_ids = torch.where(dev_ids >= 2**31, torch.full_like(dev_ids, -1), dev_ids)
            ov, same = idx_setmatch(dev_ids, idx_gold[p0 : p0 + CHUNK])
            r["idx_overlap"], r["idx_rows_identical"] = round(ov, 5), round(same, 4)
        if os.environ.get("TIMING", "0") == "1" and i == NCH - 1:
            r.update(
                trace_time(
                    md,
                    lambda: attn.prefill_chunk(h, p0, ids_in, append=False, tabs=tabs_t),
                    lambda: attn.rope_tabs(p0, CHUNK),
                )
            )
        res.append(r)
        print("RES " + json.dumps(r), flush=True)
    print("SUMMARY " + json.dumps({"layer": L, "kv": KVFMT, "min_pcc": min(x["pcc"] for x in res)}), flush=True)


if __name__ == "__main__":
    main()
