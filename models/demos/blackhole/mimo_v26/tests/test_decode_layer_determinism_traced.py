# SPDX-License-Identifier: Apache-2.0
"""TRACED variant of the layer-level determinism test: prefill DEC_N prompts, run DEC_WARM decode steps, then execute the SAME decode step twice (eager, taps) and compare attn_out / ffn_in / ffn_out / h_out
of every layer bitwise between the two executions (per mesh row and slot). The KV write of a repeated step is idempotent, so any difference is nondeterminism."""
import json
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_common import gsm_prompts  # noqa: E402
from harness import close_mesh, open_mesh_full  # noqa: E402

from models.demos.blackhole.mimo_v26.reference.prompts import tokenizer  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.generator import SLOTS, Generator  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.kv_pool import KVLayout  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.model import MiMoModel  # noqa: E402

E = os.environ.get
TAPS = E("DEC_TAPS", "attn_out,ffn_in,ffn_out,h_out").split(",")
OUT = E("DEC_OUT", "/tmp/dec_ldet.jsonl")


def rec(case, **kw):
    kw["case"] = case
    s = json.dumps(kw, default=str)
    print("DEC_RESULT " + s, flush=True)
    open(OUT, "a").write(s + "\n")


def host(mesh, t):
    comp = ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 1), mesh_shape=tuple(mesh.shape))
    return ttnn.to_torch(t, mesh_composer=comp).float()  # [rows, cols, 1?, 32, 4096]


def main():
    NL, N, WARM = int(E("DEC_LAYERS", "48")), int(E("DEC_N", "32")), int(E("DEC_WARM", "5"))
    tok = tokenizer()
    prompts = [it["prompt_ids"] for it in gsm_prompts(N, tok)]
    S = 256
    lay = KVLayout(users=-(-N // 4), max_pos=512, chunk=S)
    mesh = open_mesh_full()
    try:
        model = MiMoModel(mesh, n_layers=NL, layout=lay, log=lambda s: None)
        g = Generator(model, 512, trace=False, keep_logits=False)
        # generate WARM+1 tokens, leaving KV state at position len+WARM for every user
        r = g.generate(prompts, WARM + 1, S=S)
        place = r["place"]
        rows = mesh.shape[0]
        cur = torch.zeros(rows, SLOTS, dtype=torch.long)
        pos = torch.zeros(rows, SLOTS, dtype=torch.long)
        alive = torch.zeros(rows, SLOTS, dtype=torch.bool)
        for i, (rw, s) in enumerate(place):
            cur[rw, s] = r["tokens"][i][-1]
            pos[rw, s] = len(prompts[i]) + WARM
            alive[rw, s] = True
        dec = g.dec
        dec.set_inputs(cur, pos, ~alive)
        m = model
        taps_all = []

        def fwd():
            taps_all.clear()
            e = ttnn.embedding(dec.tok, m.embed, layout=ttnn.TILE_LAYOUT)
            h = ttnn.reshape(e, (1, 1, SLOTS, 4096))
            for lyr in m.layers:
                taps = {}
                h2 = lyr.decode(h, dec.pos["cur"], dec.pos["idx"], taps=taps, live=dec.live.t)
                taps["h_out"] = h2
                taps_all.append(taps)
                h = h2
            return h

        fwd()  # eager warmup (compile); taps leak, fine
        ttnn.synchronize_device(mesh)
        R = int(E("DEC_REPS", "30"))
        RE = int(E("DEC_EAGER", "0"))
        live = alive
        stats = {}
        first = None
        # NO device allocation after the trace capture (it hung the next replay: 'Allocating device buffers is potentially unsafe due to the existence of
        # an active trace'). All compared taps are concatenated INSIDE the trace into one persistent tensor [1,1,32,len(names)*4096]; one host read per replay.
        names = [(L, k) for L, t in enumerate(taps_all) for k in t if k in TAPS]

        def pack():
            return ttnn.concat([taps_all[L][k] for L, k in names], dim=3)

        pack()  # compile
        fwd()
        pack()
        ttnn.synchronize_device(mesh)
        os.environ["MIMO_CHECK_CAP"] = "0"
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        fwd()
        big = pack()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.synchronize_device(mesh)
        comp = ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 1), mesh_shape=tuple(mesh.shape))
        livem = alive.reshape(alive.shape[0], 1, SLOTS, 1)

        def grab(t):
            return ttnn.to_torch(t, mesh_composer=comp).reshape(alive.shape[0], mesh.shape[1], SLOTS, len(names), 4096)

        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        ref = grab(big).view(torch.int16)

        def check(rep, cur, tag):
            nonlocal first
            d = (cur.view(torch.int16) != ref) & livem.unsqueeze(-1)  # [rows, cols, 32, names, 4096]
            if not d.any():
                return
            cnt = d.sum(dim=(2, 4))  # [rows, cols, names]
            for r, c, j in cnt.nonzero().tolist():
                L, k = names[j]
                e = stats.setdefault((L, k), dict(n_reps_bad=0, rows=set(), maxdiff=0.0, devs=set()))
                e["n_reps_bad"] += 1
                e["rows"].add(r)
                e["devs"].add((r, c))
                e["maxdiff"] = max(e["maxdiff"], float(cnt[r, c, j]))  # = number of differing elements
                if first is None or (L, k) < first:
                    first = (L, k)
                print("DEC_GLITCH", tag, rep, f"L{L}.{k}", (r, c), int(cnt[r, c, j]), flush=True)

        t0 = time.time()
        for rep in range(R):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            check(rep, grab(big), "trace")
            if rep % 50 == 0:
                print("DEC_PROGRESS trace", rep, round(time.time() - t0), flush=True)
        for rep in range(RE):  # eager steps compared with the same reference
            fwd()
            check(rep, grab(pack()), "eager")
            if rep % 25 == 0:
                print("DEC_PROGRESS eager", rep, round(time.time() - t0), flush=True)
        R = R + RE
        out = {
            f"L{L}.{k}": dict(n_reps_bad=v["n_reps_bad"], rows=sorted(v["rows"]), maxdiff=v["maxdiff"])
            for (L, k), v in sorted(stats.items())
        }
        rec(
            "layer_determinism_traced_many_replays",
            layers=NL,
            reps=R,
            first=first,
            n_layer_taps_with_live_diff=len(out),
            earliest=dict(list(out.items())[:12]),
        )
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
