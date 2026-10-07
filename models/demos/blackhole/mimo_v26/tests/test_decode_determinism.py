# SPDX-License-Identifier: Apache-2.0
"""Where does run-to-run nondeterminism enter? Same prompts, same process, DEC_LAYERS layers: run A free (tokens TA); runs B and C teacher forced on TA with logits kept.
Compares the prefill first-token logits and every decode step's logits of B vs C bitwise (+ max abs diff, first differing step/user).
Env: DEC_LAYERS (48), DEC_N (32), DEC_STEPS (24), DEC_TRACE (1)."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_common import gsm_prompts  # noqa: E402
from harness import close_mesh, open_mesh_full  # noqa: E402

from models.demos.blackhole.mimo_v26.reference.prompts import tokenizer  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.generator import Generator  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.kv_pool import KVLayout  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.model import MiMoModel  # noqa: E402

E = os.environ.get
OUT = E("DEC_OUT", "/tmp/dec_det.jsonl")


def rec(case, **kw):
    kw["case"] = case
    s = json.dumps(kw, default=str)
    print("DEC_RESULT " + s, flush=True)
    open(OUT, "a").write(s + "\n")


def main():
    NL, N, STEPS = int(E("DEC_LAYERS", "48")), int(E("DEC_N", "32")), int(E("DEC_STEPS", "24"))
    tok = tokenizer()
    prompts = [it["prompt_ids"] for it in gsm_prompts(N, tok)]
    S = 256
    lay = KVLayout(users=-(-N // 4), max_pos=512, chunk=S)
    mesh = open_mesh_full()
    try:
        model = MiMoModel(mesh, n_layers=NL, layout=lay, log=lambda s: None)
        g = Generator(model, 512, trace=E("DEC_TRACE", "1") == "1", keep_logits=False)
        a = g.generate(prompts, STEPS, S=S)
        ta = a["tokens"]
        g.keep_logits = True
        g.dec.keep_logits = True
        # the traced graph was captured without logits outputs: rebuild decoder with logits
        g.dec.release()
        from models.demos.blackhole.mimo_v26.tt.decode import Decoder

        g.dec = Decoder(model, trace=E("DEC_TRACE", "1") == "1", keep_logits=True)
        ge = Generator(model, 512, trace=False, keep_logits=True)
        e = ge.generate(prompts, STEPS, S=S, teacher=ta)
        runs = {"E": e}
        for name in ("B", "C"):
            runs[name] = g.generate(prompts, STEPS, S=S, teacher=ta)
        runs["E2"] = ge.generate(prompts, STEPS, S=S, teacher=ta)

        def cmp(x, y):
            bad = [
                (i, k, float((p - q).abs().max()))
                for i in range(N)
                for k, (p, q) in enumerate(zip(runs[x]["step_logits"][i], runs[y]["step_logits"][i]))
                if not torch.equal(p, q)
            ]
            first = min(bad, key=lambda t: t[1]) if bad else None
            return dict(
                first_logits_equal=bool(torch.equal(runs[x]["first_logits"], runs[y]["first_logits"])),
                n_bad_user_steps=len(bad),
                n_bad_users=len({t[0] for t in bad}),
                first_bad=first,
                tokens_equal=runs[x]["tokens"] == runs[y]["tokens"],
            )

        rec(
            "determinism",
            layers=NL,
            n=N,
            steps=STEPS,
            **{f"{x}_vs_{y}": cmp(x, y) for x, y in (("E", "E2"), ("E", "B"), ("E", "C"), ("B", "C"))},
        )
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
