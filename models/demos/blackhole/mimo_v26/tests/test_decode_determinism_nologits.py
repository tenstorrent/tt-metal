# SPDX-License-Identifier: Apache-2.0
"""Determinism without logits readback: 3 traced teacher-forced runs (and optionally eager runs) compare tokens and the max logit of every step bitwise."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_common import gsm_prompts  # noqa: E402
from harness import close_mesh, open_mesh_full  # noqa: E402

from models.demos.blackhole.mimo_v26.reference.prompts import tokenizer  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.generator import Generator  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.kv_pool import KVLayout  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.model import MiMoModel  # noqa: E402

E = os.environ.get
OUT = E("DEC_OUT", "/tmp/dec_det2.jsonl")


def main():
    NL, N, STEPS = int(E("DEC_LAYERS", "48")), int(E("DEC_N", "32")), int(E("DEC_STEPS", "24"))
    tok = tokenizer()
    prompts = [it["prompt_ids"] for it in gsm_prompts(N, tok)]
    S = 256
    lay = KVLayout(users=-(-N // 4), max_pos=512, chunk=S)
    mesh = open_mesh_full()
    try:
        model = MiMoModel(mesh, n_layers=NL, layout=lay, log=lambda s: None)
        mode = E("DEC_MODES", "traced,traced,traced,eager,eager").split(",")
        gens = {
            "traced": Generator(model, 512, trace=True, keep_logits=False),
            "eager": Generator(model, 512, trace=False, keep_logits=False),
        }
        runs = []
        ta = None
        for m in mode:
            r = gens[m].generate(prompts, STEPS, S=S, teacher=ta)
            ta = ta or r["tokens"]
            runs.append((m, r))
        res = {}
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                a, b = runs[i][1], runs[j][1]
                bad = [
                    (u, k)
                    for u in range(N)
                    for k, (x, y) in enumerate(zip(a["step_best"][u], b["step_best"][u]))
                    if x != y
                ]
                res[f"run{i}({runs[i][0]})_vs_run{j}({runs[j][0]})"] = dict(
                    n_bad_user_steps=len(bad),
                    bad_users=sorted({u for u, _ in bad}),
                    bad_rows=sorted({u % 4 for u, _ in bad}),
                    first=min(bad, key=lambda t: t[1]) if bad else None,
                    tokens_equal=a["tokens"] == b["tokens"],
                )
        line = json.dumps(dict(case="determinism_nologits", layers=NL, n=N, steps=STEPS, modes=mode, res=res))
        print("DEC_RESULT " + line, flush=True)
        open(OUT, "a").write(line + "\n")
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
