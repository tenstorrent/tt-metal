# SPDX-License-Identifier: Apache-2.0
"""Paired A/B of the host loop: Decoder feedback off vs on (on-device argmax + position increment, tokens-only readback), one process, Generator.generate end to end
(step time INCLUDES set_inputs and readback), alternating rounds, 48 layers. Also checks both modes generate identical tokens.
Env: DEC_US (1,8,32) DEC_LAYERS (48) DEC_STEPS (40) DEC_ROUNDS (3)."""
import json
import os
import sys
import time

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_common import gsm_prompts  # noqa: E402
from harness import close_mesh, dram, open_mesh_full  # noqa: E402

from models.demos.blackhole.mimo_v26.reference.prompts import tokenizer  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.generator import Generator  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.kv_pool import KVLayout  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.model import MiMoModel  # noqa: E402

E = os.environ.get
OUT = E("DEC_OUT", "/tmp/dec_fb.jsonl")


def main():
    Us = [int(v) for v in E("DEC_US", "1,8,32").split(",")]
    NL = int(E("DEC_LAYERS", "48"))
    steps = int(E("DEC_STEPS", "40"))
    rounds = int(E("DEC_ROUNDS", "3"))
    tok = tokenizer()
    items = gsm_prompts(4 * max(Us), tok)
    S, max_pos = 256, 512
    mesh = open_mesh_full()
    try:
        model = MiMoModel(mesh, n_layers=NL, layout=KVLayout(users=Us[0], max_pos=max_pos, chunk=S), log=lambda s: None)
        gen = Generator(model, max_pos, trace=True, keep_logits=False, U=Us[0])
        for k, U in enumerate(Us):
            if k > 0:
                gen.reconfigure(U, KVLayout(users=U, max_pos=max_pos, chunk=S))
            prompts = [it["prompt_ids"] for it in items[: 4 * U]]
            ms, toks = {False: [], True: []}, {}
            extra = [int(v) for v in E("FB_EOS_EXTRA", "").split(",") if v]
            if E("FB_EOS_AUTO"):  # a frequent token of a plain run becomes an extra eos
                gen.dec.set_feedback(False)
                from collections import Counter

                cnt = Counter(t for o in gen.generate(prompts, steps, S=S)["tokens"] for t in o[8:])
                extra = [t for t, _ in cnt.most_common(int(E("FB_EOS_AUTO")) + 1)][1:]
            if extra:  # make users finish early so the device-side eos / dead handling is exercised
                gen.eos = set(gen.eos) | set(extra)
                gen.dec.eos = tuple(sorted(gen.eos))
            for rd in range(rounds):
                for fb in (False, True):
                    gen.dec.set_feedback(fb)
                    r = gen.generate(prompts, steps, S=S)
                    t = r["times"][3:]
                    ms[fb].append(round(sorted(t)[len(t) // 2] * 1e3, 3))
                    toks[fb] = r["tokens"]
            # device-bound reference: back-to-back replays of the feedback trace, one sync at the end (no ring read, no host work between enqueues)
            gen.dec.set_feedback(True)
            gen.generate(prompts, 8, S=S)
            tr = gen.dec._tr
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(20):
                ttnn.execute_trace(mesh, tr, cq_id=0, blocking=False)
            enq = (time.perf_counter() - t0) / 20
            ttnn.synchronize_device(mesh)
            pipe = (time.perf_counter() - t0) / 20
            diff = None
            for i, (x, y) in enumerate(zip(toks[False], toks[True])):
                if x != y:
                    j = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
                    diff = [i, j, len(x), len(y)]
                    break
            line = json.dumps(
                dict(
                    case="fb_ab",
                    U=U,
                    layers=NL,
                    steps=steps,
                    nofeedback_ms=ms[False],
                    feedback_ms=ms[True],
                    tokens_equal=toks[False] == toks[True],
                    first_diff_user_step_lens=diff,
                    trace_pipelined_ms=round(pipe * 1e3, 3),
                    enqueue_ms_per_step=round(enq * 1e3, 3),
                )
            )
            print("FB_RESULT " + line, flush=True)
            open(OUT, "a").write(line + "\n")
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
