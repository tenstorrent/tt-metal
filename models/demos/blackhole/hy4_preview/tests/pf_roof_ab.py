# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""prefill-roof phase 3 layer-level A/B: Hy4Attention built with PfOpt variants (Env VARIANTS ';'-separated token lists, 'off' = stock), whole 6-chunk prefill of the golden prompt on a 1x8 slot,
PCC of EVERY chunk output vs the oracle golden, indexer top-k overlap (full layer), traced us of the last chunk + the indexer select alone. Env: LAYER (1|2). Prints AB json lines."""
import json
import os
import time
import statistics

import torch

import ttnn
from models.demos.blackhole.hy4_preview.config import Hy4Config
from models.demos.blackhole.hy4_preview.tests.run_attn_device import idx_setmatch, load_sd, pcc
from models.demos.blackhole.hy4_preview.tt.hy4_attention import Hy4Attention
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload
from models.demos.blackhole.hy4_preview.tt.hy4_pf_opt import PfOpt
from models.demos.gpt_oss.config import mesh_1x8
from models.demos.gpt_oss.tt.ccl import CCLManager

GOLD = os.environ.get("GOLD", "/mnt/tt-data/ssinghal/hy4-logs/layer-attn/golden")
CKPT = os.environ.get("HY4_CKPT", "/mnt/tt-data/Hy4-Preview")
L = int(os.environ.get("LAYER", "1"))
N = 512
LAST = 2560


def tr(md, fn, reps=3, n=5):
    fn()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    for _ in range(reps):
        fn()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        ts.append((time.perf_counter() - t0) / reps * 1e6)
    ttnn.release_trace(md, tid)
    return round(min(ts), 1)


def main():
    cfg = Hy4Config.from_checkpoint(CKPT)
    g = torch.load(f"{GOLD}/L{L}.pt")
    conv = g["conv"]
    sd = load_sd(L)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=800_000_000)
    try:
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        for v in os.environ.get("VARIANTS", "off;all").split(";"):
            pf = PfOpt.parse("" if v == "off" else v)
            a = Hy4Attention(md, mesh_1x8(), ccl, cfg, sd, L, kv_fmt="bf16", rope_interleave=conv["rope_interleave"], idx_rope_interleave=conv["idx_rope_interleave"], wdtype=ttnn.bfloat8_b, pf_opt=pf)
            shared = not a.has_indexer
            pccs, ovs = [], []
            for p0 in range(0, 3072, N):
                h = upload(md, g["h"][p0 : p0 + N].reshape(1, 1, N, -1).to(torch.bfloat16))
                ids_in = upload(md, g["idx"][p0 : p0 + N].to(torch.int32).reshape(1, 1, N, -1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT) if shared else None
                out, ids = a.prefill_chunk(h, p0, ids_in)
                pccs.append(round(pcc(ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(N, -1).float(), g["out"][p0 : p0 + N].float()), 6))
                if not shared:
                    ovs.append(round(idx_setmatch(ttnn.to_torch(ttnn.get_device_tensors(ids)[0]).reshape(N, -1).long(), g["idx"][p0 : p0 + N])[0], 4))
            r = {"layer": L, "variant": v, "pcc_chunks": pccs, "min_pcc": min(pccs)}
            if ovs:
                r["idx_overlap"] = ovs
            tabs = a.rope_tabs(LAST, N)
            r["chunk_us"] = tr(md, lambda: a.prefill_chunk(h, LAST, ids_in, append=False, tabs=tabs))
            if not shared:
                qa_c = a.mesh_config.allgather(__import__("models.demos.blackhole.hy4_preview.tt.hy4_pf_opt", fromlist=["plinear"]).plinear(pf, h, a.w_q_a, a.ckc_pf), a.ccl, axis=1, dim=3)
                qa = a._rms(qa_c, a.q_a_norm)
                r["indexer_us"] = tr(md, lambda: a.indexer.select(h, qa, LAST, tabs[2], tabs[3], None, append=False))
            print("AB " + json.dumps(r), flush=True)
            del a
    finally:
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
