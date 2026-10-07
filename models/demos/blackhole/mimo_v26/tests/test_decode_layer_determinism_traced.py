# SPDX-License-Identifier: Apache-2.0
"""TRACED variant of the layer-level determinism test: prefill DEC_N prompts, run DEC_WARM decode steps, then execute the SAME decode step twice (eager, taps) and compare attn_out / ffn_in / ffn_out / h_out
of every layer bitwise between the two executions (per mesh row and slot). The KV write of a repeated step is idempotent, so any difference is nondeterminism."""
import json
import os
import sys

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
        os.environ["MIMO_CHECK_CAP"] = "0"
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        fwd()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.synchronize_device(mesh)
        taps_dev = list(taps_all)
        R = int(E("DEC_REPS", "30"))
        live = alive
        stats = {}
        first = None
        ref = None

        def compare(run, rep):
            nonlocal first
            for L in range(NL):
                for k in ref[L]:
                    d = (ref[L][k] - run[L][k]).abs().amax(dim=-1)  # [rows, cols, 32]
                    dl = d[live.unsqueeze(1).expand_as(d)]
                    if dl.numel() and float(dl.max()) > 0:
                        rows_bad = sorted(
                            {int(r) for r in (d * live.unsqueeze(1)).amax(dim=(1, 2)).nonzero().flatten().tolist()}
                        )
                        e = stats.setdefault((L, k), dict(n_reps_bad=0, rows=set(), maxdiff=0.0))
                        e["n_reps_bad"] += 1
                        e["rows"].update(rows_bad)
                        e["maxdiff"] = max(e["maxdiff"], float(dl.max()))
                        if first is None or (L, k) < first:
                            first = (L, k)
                        print("DEC_GLITCH", rep, L, k, rows_bad, float(dl.max()), flush=True)

        # streaming comparison against replay 0 (no storage of all replays)
        for rep in range(R):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            run = [{k: host(mesh, v) for k, v in t.items() if k in TAPS} for t in taps_dev]
            if ref is None:
                ref = run
            else:
                compare(run, rep)
        RE = int(E("DEC_EAGER", "0"))
        for rep in range(RE):  # eager steps: taps of the eager step vs the trace reference
            fwd()
            run = [{k: host(mesh, v) for k, v in t.items() if k in TAPS} for t in taps_all]
            compare(run, 100000 + rep)
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
