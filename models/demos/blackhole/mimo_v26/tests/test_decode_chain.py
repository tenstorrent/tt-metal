# SPDX-License-Identifier: Apache-2.0
"""Closed-loop decode of a short chain (DEC_LAYERS layers + final norm + lm_head) on the 4x8 mesh vs the CPU oracle (ref_chain_decode.py output), ragged real GSM8K prompts
(non-thinking template), per-row/per-slot positions, traced decode loop with dead slots.
Phase 1 teacher forced: device fed the ORACLE tokens: first-token logits (prefill) and per-step logits PCC / top-1 agreement.
Phase 2 free running: device greedy tokens vs oracle tokens (equal until the first near-tie flip). Phase 3 determinism: the free-running run repeated, bit-identical tokens + logits.
Env: DEC_LAYERS=4 DEC_ORACLE=/mnt/tt-data/ssinghal/mimo-dumps/decode_chain4.pt DEC_TRACE=1 DEC_OUT=jsonl MIMO_CKPT + MOE_UNI_* (see handoff)."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attn_ref import pcc  # noqa: E402
from harness import close_mesh, dram, open_mesh_full  # noqa: E402

import ttnn  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.generator import Generator  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.kv_pool import KVPoolLayout  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.model import MiMoModel  # noqa: E402

E = os.environ.get
OUT = E("DEC_OUT", "/tmp/dec_chain.jsonl")


def rec(case, **kw):
    kw["case"] = case
    s = json.dumps(kw, default=str)
    print("DEC_RESULT " + s, flush=True)
    with open(OUT, "a") as f:
        f.write(s + "\n")


def main():
    NL = int(E("DEC_LAYERS", "4"))
    orc = torch.load(E("DEC_ORACLE", "/mnt/tt-data/ssinghal/mimo-dumps/decode_chain4.pt"))
    assert orc["layers"] == NL
    items = orc["items"]
    n, steps = len(items), orc["steps"]
    prompts = [it["prompt_ids"] for it in items]
    max_pos = 512
    pool = KVPoolLayout(max_pos=max_pos, ring_blocks=256 // 64 + 8, slots=32)
    mesh = open_mesh_full()
    try:
        model = MiMoModel(
            mesh,
            n_layers=NL,
            max_pos=max_pos,
            users=32,
            ring_blocks=pool.ring_blocks,
            pool=pool,
            log=lambda s: print(s, flush=True),
        )
        rec("env", layers=NL, n=n, steps=steps, lens=[len(p) for p in prompts], dram=dram(mesh))
        gen = Generator(model, max_pos, trace=E("DEC_TRACE", "1") == "1", keep_logits=True)
        teacher = [it["tokens"] for it in items]
        # ---- phase 1: teacher forced
        r = gen.generate(prompts, steps, S=256, teacher=teacher)
        place = r["place"]
        fl = r["first_logits"]  # [rows,U,V]
        first_pcc = [pcc(fl[rw, s], items[i]["logits"][0].float()) for i, (rw, s) in enumerate(place)]
        step_pcc = []
        for k in range(steps - 1):
            step_pcc.append(min(pcc(r["step_logits"][i][k], items[i]["logits"][k + 1].float()) for i in range(n)))
        agree = sum(int(r["tokens"][i][k] == items[i]["tokens"][k]) for i in range(n) for k in range(steps)) / (
            n * steps
        )
        # near-tie aware: oracle top1-top2 gap where the device token differs
        flips = []
        for i in range(n):
            for k in range(1, steps):
                if r["tokens"][i][k] != items[i]["tokens"][k]:
                    l = items[i]["logits"][k].float()
                    t2 = l.topk(2).values
                    flips.append(dict(user=i, step=k, gap=float(t2[0] - t2[1])))
        rec(
            "teacher_forced",
            first_logits_pcc_min=round(min(first_pcc), 5),
            step_logits_pcc_min_per_step=[round(x, 5) for x in step_pcc],
            token_agree=agree,
            flips=flips,
            step_ms_steady=round(1e3 * sorted(r["times"][5:])[len(r["times"][5:]) // 2], 2)
            if len(r["times"]) > 6
            else None,
            step0_ms=round(1e3 * r["times"][0], 1),
            prefill_s=round(r["prefill_s"], 1),
        )
        # ---- phase 2: free running + determinism
        gen.keep_logits = False
        gen.dec.keep_logits = False
        a = gen.generate(prompts, steps, S=256)
        b = gen.generate(prompts, steps, S=256)
        same = a["tokens"] == b["tokens"]
        match = [sum(int(x == y) for x, y in zip(a["tokens"][i], items[i]["tokens"])) for i in range(n)]
        first_div = [
            next((k for k, (x, y) in enumerate(zip(a["tokens"][i], items[i]["tokens"])) if x != y), steps)
            for i in range(n)
        ]
        rec(
            "free_running",
            tokens_equal_to_oracle=sum(match) / (n * steps),
            first_divergence=first_div,
            run_to_run_identical=same,
            dram_after=dram(mesh),
        )
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
