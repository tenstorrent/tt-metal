# SPDX-License-Identifier: Apache-2.0
"""Decode at U users per mesh row in {32,1,2,4,8,16,32} (B = 4U) in ONE process with Generator.reconfigure (weights never reloaded).
Phase 1 (DEC_LAYERS=4, oracle DEC_ORACLE with >= 4*32 prompts): teacher-forced + free-running tokens vs the CPU oracle at every U (exact token equality), ragged prompts.
Phase 2 (any layers): step-time table per U (median steady / p10-p90 / step 0), DRAM per U, identical DRAM after returning to the same U.
Env: DEC_ORDER=32,1,2,4,8,16,32  DEC_ORACLE  DEC_STEPS(12)  DEC_TRACE(1)  DEC_TIMING_STEPS(40)."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attn_ref import pcc  # noqa: E402
from harness import close_mesh, dram, open_mesh_full  # noqa: E402

from models.demos.blackhole.mimo_v26.tt.generator import Generator  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.kv_pool import KVLayout  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.model import MiMoModel  # noqa: E402

E = os.environ.get
OUT = E("DEC_OUT", "/tmp/dec_reconf.jsonl")
ROWS = 4


def rec(case, **kw):
    kw["case"] = case
    s = json.dumps(kw, default=str)
    print("DEC_RESULT " + s, flush=True)
    open(OUT, "a").write(s + "\n")


def main():
    NL = int(E("DEC_LAYERS", "4"))
    order = [int(v) for v in E("DEC_ORDER", "32,1,2,4,8,16,32").split(",")]
    orc = torch.load(E("DEC_ORACLE", "/mnt/tt-data/ssinghal/mimo-dumps/decode_chain4_n128.pt"))
    items = orc["items"]
    steps = orc["steps"]
    S = 256
    max_pos = 512
    tsteps = int(E("DEC_TIMING_STEPS", "0"))
    mesh = open_mesh_full()
    try:
        lay = KVLayout(users=order[0], max_pos=max_pos, chunk=S)
        model = MiMoModel(mesh, n_layers=NL, layout=lay, log=lambda s: None)
        gen = Generator(model, max_pos, trace=E("DEC_TRACE", "1") == "1", keep_logits=True, U=order[0])
        rec("env", layers=NL, order=order, dram0=dram(mesh))
        dram_by_u = {}
        for k, U in enumerate(order):
            if k > 0:
                gen.reconfigure(U, KVLayout(users=U, max_pos=max_pos, chunk=S))
            n = ROWS * U
            sub = items[:n]
            prompts = [it["prompt_ids"] for it in sub]
            teacher = [it["tokens"] for it in sub]
            r = gen.generate(prompts, steps, S=S, teacher=teacher)
            place = r["place"]
            fl = r["first_logits"]
            first_pcc = min(pcc(fl[rw, s], sub[i]["logits"][0].float()) for i, (rw, s) in enumerate(place))
            step_pcc = min(
                pcc(r["step_logits"][i][kk], sub[i]["logits"][kk + 1].float())
                for i in range(n)
                for kk in range(steps - 1)
            )
            agree = sum(int(r["tokens"][i][kk] == sub[i]["tokens"][kk]) for i in range(n) for kk in range(steps)) / (
                n * steps
            )
            # free running (no logits readback) for the step-time table
            gen.keep_logits = False
            gen.dec.keep_logits = False
            gen.dec.release()
            f = gen.generate(prompts, steps, S=S)
            free_eq = sum(int(f["tokens"][i][kk] == sub[i]["tokens"][kk]) for i in range(n) for kk in range(steps)) / (
                n * steps
            )
            gen.keep_logits = True
            gen.dec.keep_logits = True
            gen.dec.release()
            d = dram(mesh)
            dram_by_u.setdefault(U, []).append(d["alloc_mb_per_bank"])
            rec(
                "U",
                U=U,
                B=n,
                first_logits_pcc_min=round(first_pcc, 5),
                step_logits_pcc_min=round(step_pcc, 5),
                teacher_token_agree=agree,
                free_running_equal_oracle=free_eq,
                dram=d,
            )
        rec("dram_identical_on_return", by_U=dram_by_u)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
