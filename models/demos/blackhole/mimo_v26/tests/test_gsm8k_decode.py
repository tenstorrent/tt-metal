# SPDX-License-Identifier: Apache-2.0
"""Full (48-layer) model greedy GSM8K on the 4x8 mesh: prefill + traced decode loop, non-thinking chat template (enable_thinking=False), scored with reference/score_gsm8k.py.
Env: GSM_N prompts (spread over rows: B = GSM_N users, GSM_N/4 per row), GSM_NEW max new tokens, GSM_START first test row, GSM_LAYERS (48), GSM_TRACE (1), DEC_OUT jsonl,
GSM_TEXTS path of the outputs json (idx -> completion). One process = one batch-size scenario."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_common import gsm_prompts  # noqa: E402
from harness import close_mesh, dram, open_mesh_full  # noqa: E402

from models.demos.blackhole.mimo_v26.reference.prompts import tokenizer  # noqa: E402
from models.demos.blackhole.mimo_v26.reference.score_gsm8k import extract, wilson  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.generator import Generator  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.kv_pool import KVPoolLayout  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.model import MiMoModel  # noqa: E402

E = os.environ.get
OUT = E("DEC_OUT", "/tmp/gsm8k_decode.jsonl")


def rec(case, **kw):
    kw["case"] = case
    s = json.dumps(kw, default=str)
    print("GSM_RESULT " + s, flush=True)
    with open(OUT, "a") as f:
        f.write(s + "\n")


def main():
    N, NEW, START, NL = (
        int(E("GSM_N", "64")),
        int(E("GSM_NEW", "512")),
        int(E("GSM_START", "0")),
        int(E("GSM_LAYERS", "48")),
    )
    tok = tokenizer()
    items = gsm_prompts(N, tok, START)
    maxlen = max(len(it["prompt_ids"]) for it in items)
    S = max(256, -(-maxlen // 128) * 128)
    max_pos = -(-(S + NEW + 8) // 256) * 256
    pool = KVPoolLayout(max_pos=max_pos, ring_blocks=S // 64 + 8, slots=32)
    mesh = open_mesh_full()
    try:
        t0 = time.time()
        model = MiMoModel(
            mesh,
            n_layers=NL,
            max_pos=max_pos,
            users=32,
            ring_blocks=pool.ring_blocks,
            pool=pool,
            log=lambda s: print(s, flush=True) if "layer 47" in s or "layer 0 " in s else None,
        )
        rec(
            "env",
            layers=NL,
            n=N,
            new=NEW,
            S=S,
            max_pos=max_pos,
            build_s=round(time.time() - t0, 1),
            expert_op=E("MIMO_EXPERT_OP"),
            dram=dram(mesh),
            cache=model.wc.stats,
        )
        gen = Generator(model, max_pos, trace=E("GSM_TRACE", "1") == "1", keep_logits=False)
        r = gen.generate([it["prompt_ids"] for it in items], NEW, S=S)
        times = r["times"]
        steady = sorted(times[5:])
        texts, ok = {}, []
        for i, it in enumerate(items):
            txt = tok.decode(r["tokens"][i], skip_special_tokens=True)
            texts[it["idx"]] = txt
            if extract(txt) == it["gold"]:
                ok.append(it["idx"])
        lo, hi = wilson(len(ok), N)
        rec(
            "result",
            n=N,
            correct=len(ok),
            acc=round(len(ok) / N, 4),
            wilson95=[round(lo, 3), round(hi, 3)],
            prefill_s=round(r["prefill_s"], 1),
            decode_steps=len(times),
            step0_ms=round(times[0] * 1e3, 1),
            step_ms_median_steady=round(steady[len(steady) // 2] * 1e3, 2) if steady else None,
            step_ms_p10_p90=[round(steady[len(steady) // 10] * 1e3, 2), round(steady[9 * len(steady) // 10] * 1e3, 2)]
            if steady
            else None,
            gen_lens=[len(t) for t in r["tokens"]],
            finished_eos=sum(int(any(x in gen.eos for x in t)) for t in r["tokens"]),
            session="dedicated single-scenario process, trace=%s, unpaired, host-timed replay+candidate readback"
            % E("GSM_TRACE", "1"),
            dram_after=dram(mesh),
        )
        json.dump(texts, open(E("GSM_TEXTS", OUT.replace(".jsonl", "_texts.json")), "w"), ensure_ascii=False)
        rec(
            "samples",
            first=[(items[i]["idx"], items[i]["gold"], texts[items[i]["idx"]][:300]) for i in range(min(3, N))],
        )
        gen.dec.release()
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
