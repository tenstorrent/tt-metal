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
        runs = []
        for rep in range(int(E("DEC_REPS", "3"))):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            runs.append([{k: host(mesh, v) for k, v in t.items()} for t in taps_dev])
        runs = runs[:2] if len(runs) < 3 else runs
        extra = [(0, 1), (1, 2)] if len(runs) == 3 else [(0, 1)]
        live = alive  # [rows, 32]
        first_live, first_any, per_layer = None, None, []
        runs = [runs[0], runs[1]]
        for L in range(NL):
            diffs = {}
            for k in runs[0][L]:
                d = (runs[0][L][k] - runs[1][L][k]).abs().amax(dim=-1)  # [rows, cols, 32]
                dl = d[live.unsqueeze(1).expand_as(d)]
                dd = d[(~live).unsqueeze(1).expand_as(d)]
                if float(d.max()) > 0:
                    diffs[k] = dict(
                        live_maxdiff=float(dl.max()) if dl.numel() else 0.0,
                        live_bad=int((dl > 0).sum()),
                        dead_maxdiff=float(dd.max()) if dd.numel() else 0.0,
                        dead_bad=int((dd > 0).sum()),
                    )
            per_layer.append(diffs)
            if diffs and first_any is None:
                first_any = L
            if first_live is None and any(v["live_bad"] for v in diffs.values()):
                first_live = L
        rec(
            "layer_determinism",
            layers=NL,
            first_layer_any_diff=first_any,
            first_layer_live_diff=first_live,
            n_live_per_row=int(live[0].sum()),
            per_layer={L: d for L, d in enumerate(per_layer) if d and L < (first_any or 0) + 3 or L == first_live},
        )
        rec("summary", n_layers_with_diff=sum(1 for d in per_layer if d))
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
