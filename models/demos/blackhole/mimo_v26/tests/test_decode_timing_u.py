# SPDX-License-Identifier: Apache-2.0
"""Decode step-time table vs users per mesh row U in one session (one model build, Generator.reconfigure between U), forward order then reverse, plus a per-part breakdown
(embedding / attention / FFN-MoE / head) from separately captured traces at each U (replay-only device+dispatch time, no host work).
Env: DEC_LAYERS(48) DEC_ORDER(32,1,2,4,8,16,32,16,8,4,2,1,32) GSM_NEW(48 tokens generated per user for the step-time table) DEC_BREAKDOWN(1) DEC_REPS(30)."""
import json
import os
import sys
import time

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_common import gsm_prompts  # noqa: E402
from harness import close_mesh, dram, open_mesh_full  # noqa: E402

from models.demos.blackhole.mimo_v26.reference.prompts import tokenizer  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.decode import HIDDEN  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.generator import Generator  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.kv_pool import KVLayout  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.layer import EPS, _k_norm  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.model import MiMoModel  # noqa: E402

E = os.environ.get
OUT = E("DEC_OUT", "/tmp/dec_timing_u.jsonl")


def rec(case, **kw):
    kw["case"] = case
    s = json.dumps(kw, default=str)
    print("DEC_RESULT " + s, flush=True)
    open(OUT, "a").write(s + "\n")


def parts_forward(dec, parts):
    m = dec.m

    def fwd():
        e = ttnn.embedding(dec.tok, m.embed, layout=ttnn.TILE_LAYOUT)
        h = ttnn.reshape(e, (1, 1, dec.U, HIDDEN))
        for lyr in m.layers:
            h1 = h
            if "attn" in parts:
                xn = lyr.rms(h, lyr.w_in)
                a = lyr.attn.decode(xn, dec.pos["cur"], dec.pos["idx"])
                h1 = ttnn.add(h, a, dtype=ttnn.bfloat16)
                for t in (xn, a):
                    ttnn.deallocate(t)
            h2 = h1
            if "ffn" in parts:
                fn = lyr.rms(h1, lyr.w_post)
                f = lyr.ffn_dense(fn) if lyr.L == 0 else lyr.moe_decode(fn, dec.live.t)
                h2 = ttnn.add(h1, f, dtype=ttnn.bfloat16)
                for t in (fn, f):
                    ttnn.deallocate(t)
            if h1 is not h:
                ttnn.deallocate(h1) if h2 is not h1 else None
            if h2 is not h:
                ttnn.deallocate(h)
            h = h2
        if "head" in parts:
            x = ttnn.rms_norm(h, epsilon=EPS, weight=m.w_norm, compute_kernel_config=_k_norm())
            lg = ttnn.linear(x, m.lm_head, dtype=ttnn.bfloat16)
            vals = ttnn.max(lg, dim=3, keepdim=True)
            return vals
        return h

    return fwd


def time_trace(mesh, fn, reps):
    o = fn()
    ttnn.synchronize_device(mesh)
    os.environ["MIMO_CHECK_CAP"] = "0"
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    o = fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    ts = []
    for _ in range(reps + 3):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        ts.append(time.perf_counter() - t0)
    ttnn.release_trace(mesh, tid)
    ttnn.deallocate(o)
    ts = sorted(ts[3:])
    return round(ts[len(ts) // 2] * 1e3, 2)


def main():
    NL = int(E("DEC_LAYERS", "48"))
    order = [int(v) for v in E("DEC_ORDER", "32,1,2,4,8,16,32,16,8,4,2,1,32").split(",")]
    NEW = int(E("GSM_NEW", "48"))
    reps = int(E("DEC_REPS", "30"))
    tok = tokenizer()
    items = gsm_prompts(128, tok)
    S, max_pos = 256, 512
    mesh = open_mesh_full()
    try:
        model = MiMoModel(
            mesh, n_layers=NL, layout=KVLayout(users=order[0], max_pos=max_pos, chunk=S), log=lambda s: None
        )
        gen = Generator(model, max_pos, trace=True, keep_logits=False, U=order[0])
        rec("env", layers=NL, order=order, dram0=dram(mesh))
        for k, U in enumerate(order):
            if k > 0:
                gen.reconfigure(U, KVLayout(users=U, max_pos=max_pos, chunk=S))
            n = 4 * U
            r = gen.generate([it["prompt_ids"] for it in items[:n]], NEW, S=S)
            t = r["times"]
            st = sorted(t[5:])
            row = dict(
                U=U,
                B=n,
                step0_ms=round(t[0] * 1e3, 1),
                step_ms_median=round(st[len(st) // 2] * 1e3, 2),
                p10_p90=[round(st[len(st) // 10] * 1e3, 2), round(st[9 * len(st) // 10] * 1e3, 2)],
                steps=len(t),
                prefill_s=round(r["prefill_s"], 1),
                tok_s_user=round(1 / st[len(st) // 2], 2),
            )
            if E("DEC_BREAKDOWN", "1") == "1":
                gen.dec.release()
                bd = {}
                for name, parts in (
                    ("embed_only", set()),
                    ("attn", {"attn"}),
                    ("ffn_moe", {"ffn"}),
                    ("attn_ffn", {"attn", "ffn"}),
                    ("head", {"head"}),
                ):
                    bd[name] = time_trace(mesh, parts_forward(gen.dec, parts), reps)
                row["breakdown_replay_ms"] = bd
            row["dram"] = dram(mesh)
            rec("U", **row)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
