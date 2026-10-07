# SPDX-License-Identifier: Apache-2.0
"""Device-only decode smoke (no oracle): DEC_LAYERS layers, ragged prompts of GSM8K, eager vs traced decode must give identical tokens/logits; dead slots; per-row positions."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_common import gsm_prompts  # noqa: E402
from harness import close_mesh, dram, open_mesh_full  # noqa: E402

from models.demos.blackhole.mimo_v26.reference.prompts import tokenizer  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.generator import Generator  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.kv_pool import KVPoolLayout  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.model import MiMoModel  # noqa: E402

E = os.environ.get
OUT = E("DEC_OUT", "/tmp/dec_smoke.jsonl")


def rec(case, **kw):
    kw["case"] = case
    s = json.dumps(kw, default=str)
    print("DEC_RESULT " + s, flush=True)
    open(OUT, "a").write(s + "\n")


def main():
    NL, N, STEPS = int(E("DEC_LAYERS", "2")), int(E("DEC_N", "6")), int(E("DEC_STEPS", "8"))
    tok = tokenizer()
    prompts = [it["prompt_ids"] for it in gsm_prompts(N, tok)]
    pool = KVPoolLayout(max_pos=512, ring_blocks=12, slots=32)
    mesh = open_mesh_full()
    try:
        model = MiMoModel(
            mesh, n_layers=NL, max_pos=512, users=32, ring_blocks=12, pool=pool, log=lambda s: print(s, flush=True)
        )
        outs = {}
        for name, trace in (("eager", False), ("traced", True)):
            g = Generator(model, 512, trace=trace, keep_logits=True)
            r = g.generate(prompts, STEPS, S=256)
            outs[name] = r
            rec(
                name,
                lens=[len(p) for p in prompts],
                tokens=r["tokens"],
                times_ms=[round(t * 1e3, 1) for t in r["times"]],
                prefill_s=round(r["prefill_s"], 1),
                dram=dram(mesh),
            )
            g.dec.release()
        same_tok = outs["eager"]["tokens"] == outs["traced"]["tokens"]
        dl = max(
            float((a - b).abs().max())
            for i in range(N)
            for a, b in zip(outs["eager"]["step_logits"][i], outs["traced"]["step_logits"][i])
        )
        rec("eager_vs_traced", same_tokens=same_tok, max_logit_diff=dl)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
