# SPDX-License-Identifier: Apache-2.0
"""Paired in-process A/B of the MoE column-reduce workaround on the 48-layer traced decode step (race track): for each U (users per mesh row) two traces of the SAME step are captured in one process,
MIMO_MOE_AR=ag_sum (all_gather + sum, new default) and MIMO_MOE_AR=rs64 (old all_reduce of the 64-row partial), and replayed ALTERNATELY (AB_ROUNDS rounds of AB_REPS replays each); reports median ms per
step of each, the delta, and the agreement of ffn_out of layers AB_TAPS (default 1,5,16) and of the final hidden state between the two modes on the same state (PCC, max abs diff, relative L2).
Env: DEC_LAYERS(48) AB_U("1,8,32") AB_ROUNDS(5) AB_REPS(30) AB_TAPS("1,5,16") DEC_OUT."""
import json
import os
import statistics
import sys
import time

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_common import gsm_prompts  # noqa: E402
from harness import close_mesh, open_mesh_full  # noqa: E402

from models.demos.blackhole.mimo_v26.reference.prompts import tokenizer  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.decode import HIDDEN  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.generator import Generator  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.kv_pool import KVLayout  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.model import MiMoModel  # noqa: E402

E = os.environ.get
OUT = E("DEC_OUT", "/tmp/dec_ab.jsonl")


def rec(case, **kw):
    kw["case"] = case
    s = json.dumps(kw, default=str)
    print("AB_RESULT " + s, flush=True)
    open(OUT, "a").write(s + "\n")


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def main():
    NL = int(E("DEC_LAYERS", "48"))
    Us = [int(v) for v in E("AB_U", "1,8,32").split(",")]
    ROUNDS, REPS = int(E("AB_ROUNDS", "5")), int(E("AB_REPS", "30"))
    TAPL = [int(v) for v in E("AB_TAPS", "1,5,16").split(",")]
    tok = tokenizer()
    items = gsm_prompts(128, tok)
    S, max_pos = 256, 512
    mesh = open_mesh_full()
    try:
        model = MiMoModel(mesh, n_layers=NL, layout=KVLayout(users=Us[0], max_pos=max_pos, chunk=S), log=lambda s: None)
        gen = Generator(model, max_pos, trace=True, keep_logits=False, U=Us[0])
        comp = ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 1), mesh_shape=tuple(mesh.shape))
        for k, U in enumerate(Us):
            if k > 0:
                gen.reconfigure(U, KVLayout(users=U, max_pos=max_pos, chunk=S))
            gen.generate([it["prompt_ids"] for it in items[: 4 * U]], 6, S=S)
            gen.dec.release()
            dec, m = gen.dec, model

            def make(mode):
                os.environ["MIMO_MOE_AR"] = mode
                taps_l = {}

                def fwd():
                    e = ttnn.embedding(dec.tok, m.embed, layout=ttnn.TILE_LAYOUT)
                    h = ttnn.reshape(e, (1, 1, dec.U, HIDDEN))
                    for i, lyr in enumerate(m.layers):
                        taps = {} if i in TAPL else None
                        h2 = lyr.decode(h, dec.pos["cur"], dec.pos["idx"], taps=taps, live=dec.live.t)
                        if taps is not None:
                            taps_l[i] = taps["ffn_out"]
                        h = h2
                    taps_l["final"] = h
                    return h

                fwd()  # compile
                ttnn.synchronize_device(mesh)
                os.environ["MIMO_CHECK_CAP"] = "0"
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                fwd()
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                ttnn.synchronize_device(mesh)
                return tid, dict(taps_l)

            tA, oA = make("ag_sum")
            tB, oB = make("rs64")
            tms = {"ag_sum": [], "rs64": []}
            for rd in range(ROUNDS):
                for name, tid in (("ag_sum", tA), ("rs64", tB)) if rd % 2 == 0 else (("rs64", tB), ("ag_sum", tA)):
                    ts = []
                    for _ in range(REPS + 2):
                        t0 = time.perf_counter()
                        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                        ts.append(time.perf_counter() - t0)
                    tms[name].append(statistics.median(ts[2:]) * 1e3)
            # agreement on the same state
            ttnn.execute_trace(mesh, tA, cq_id=0, blocking=True)
            ha = {k2: ttnn.to_torch(v, mesh_composer=comp).float() for k2, v in oA.items()}
            ttnn.execute_trace(mesh, tB, cq_id=0, blocking=True)
            hb = {k2: ttnn.to_torch(v, mesh_composer=comp).float() for k2, v in oB.items()}
            agree = {}
            for k2 in ha:
                a, b = (
                    ha[k2],
                    hb[k2],
                )  # [4, 8, rows, 4096]; every column identical after the reduce -> use col 0, live rows
                a, b = a[:, 0, : dec.U], b[:, 0, : dec.U]
                agree[str(k2)] = dict(
                    pcc=round(pcc(a, b), 7),
                    max_abs=float((a - b).abs().max()),
                    rel_l2=float((a - b).norm() / (b.norm() + 1e-30)),
                    alpha_ag_over_rs=float((a.double() * b.double()).sum() / ((b.double() ** 2).sum() + 1e-30)),
                )
            medA, medB = statistics.median(tms["ag_sum"]), statistics.median(tms["rs64"])
            rec(
                "U",
                U=U,
                layers=NL,
                ag_sum_ms=round(medA, 3),
                rs64_ms=round(medB, 3),
                delta_ms=round(medA - medB, 3),
                delta_pct=round(100 * (medA - medB) / medB, 2),
                per_round_ag=[round(x, 3) for x in tms["ag_sum"]],
                per_round_rs=[round(x, 3) for x in tms["rs64"]],
                agree=agree,
                session="one process, both traces alive, alternating rounds of %d replays, host-timed blocking replay"
                % REPS,
            )
            ttnn.release_trace(mesh, tA)
            ttnn.release_trace(mesh, tB)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
