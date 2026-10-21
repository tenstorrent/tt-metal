# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-model performance (profiler off): TTFT, traced token-out decode, logits-only decode, sampler cost.

Workload of the primary single-user profile: a 128-token prompt, 128 generated tokens, batch 1 (and batch 32).
Every number is the median of warm repetitions, host wall clock.

* ``ttft_ms``: ``prefill_forward(sample_on_device=True)`` for the 128-token prompt, including the first token's
  on-device sampling and its readback (what a caller waits for). Single-chunk prompts replay their bucket's
  prefill trace.
* ``token_out``: free-running ``generator.generate`` (pipelined: model + sampling replays with on-device token
  feedback, every token copied to the host asynchronously, the host waiting only for the step issued
  ``PIPELINE_LAG`` steps earlier): decode wall time from the first decode issue to the last token on the host,
  divided by the decode steps (median of 3 runs); ``token_out_replay``: the same two traces replayed back to back
  with no readback at all (device-bound token-out rate).
* ``teacher_forcing``: ``generate(next_input=...)`` (the readiness teacher-forcing path: each step blocks on its
  token, then the forced input is compared with the device buffer and written only when it differs): median
  per-step time.
* ``logits_only``: the model trace alone (no sampling, no token feedback), the PERF.md-style teacher-forcing
  comparison; ``sampling_trace``: the sampling trace alone.

  python models/autoports/google_gemma_4_12b_it/tests/perf_full_model.py --out doc/full_model/logs/perf.json
"""

import argparse
import json
import statistics
import time

import torch

import ttnn
from models.autoports.google_gemma_4_12b_it.tt.generator import build_generator
from models.autoports.google_gemma_4_12b_it.tt.multichip_decoder import close_qb2_mesh, open_qb2_mesh


def median_ms(fn, reps):
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1e3)
    return round(statistics.median(out), 3), round(min(out), 3), round(max(out), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=int, default=128)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--batches", default="1,32")
    ap.add_argument("--prefill_lengths", default="128,1024,4096,8192")
    ap.add_argument("--build-kwargs", default="{}")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(16)
    mesh = open_qb2_mesh(trace_region_size=0)
    res = {"workload": f"prompt {a.prompt}, generate {a.gen}, greedy, profiler off"}
    try:
        t0 = time.time()
        gen = build_generator(None, mesh, **json.loads(a.build_kwargs))
        res["build_and_prepare_s"] = round(time.time() - t0, 1)
        tok = gen.tokenizer
        text = open("tech_reports/LLMs/llms.md").read()
        base = tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=False, tokenize=True, return_dict=False
        )

        # ---- TTFT (prefill + first token on device), several prompt lengths
        res["ttft_ms"] = {}
        for n in [int(v) for v in a.prefill_lengths.split(",")]:
            ids = (base * (1 + n // len(base)))[:n]
            row = gen._alloc_row(n + 8)

            def prefill():
                gen.prefill_forward(torch.tensor([ids]), page_table=row[None], prompt_lens=[n], sample_on_device=True)

            prefill()
            res["ttft_ms"][n] = median_ms(prefill, 10 if n <= 4096 else 5)
            gen._release(row)
            gen.reset()
            print("TTFT", n, res["ttft_ms"][n], flush=True)

        # ---- decode at batch 1 and 32
        ids = base[: a.prompt]
        for B in [int(v) for v in a.batches.split(",")]:
            r = {}
            prompts = [ids] * B
            # token-out through free-running generate(): decode wall time per token (pipelined, async readback)
            gen.generate(prompts if B > 1 else ids, a.gen, stop_on_eos=False)  # warm
            gen.reset()
            per_tok = []
            for rep_i in range(3):
                c0 = dict(gen.counters)
                gen.generate(prompts if B > 1 else ids, a.gen, stop_on_eos=False)
                t = gen.last_generate_timing
                per_tok.append((t["t_done"] - t["t_first_issue"]) * 1e3 / t["decode_steps"])
                if rep_i == 0:
                    r["generate_counters"] = {k: gen.counters[k] - c0[k] for k in c0}
                gen.reset()
            r["token_out_generate_ms_per_token"] = round(statistics.median(per_tok), 3)
            r["token_out_generate_t_s_u"] = round(1e3 / statistics.median(per_tok), 2)
            r["token_out_generate_runs_ms"] = [round(v, 3) for v in per_tok]
            if B == 1:
                # teacher forcing through generate(next_input): blocking per step (the readiness TF path)
                stamps = []
                orig = gen.decode_forward

                def timed(*args, **kw):
                    out = orig(*args, **kw)
                    stamps.append(time.perf_counter())
                    return out

                gen.decode_forward = timed
                # forced input = the prediction (as in the ~95% of readiness steps where they agree): no token writes
                gen.generate(ids, a.gen, next_input=lambda step, pred: pred)
                gen.decode_forward = orig
                steps = [(y - x) * 1e3 for x, y in zip(stamps, stamps[1:])]
                r["teacher_forcing_generate_ms_per_step"] = round(statistics.median(steps), 3)
                r["teacher_forcing_generate_t_s_u"] = round(1e3 / statistics.median(steps), 2)
                gen.reset()
            # pure replay loops from a prefilled state (positions continue on device)
            rows = [gen._alloc_row(a.prompt + 3 * a.gen + 64) for _ in range(B)]
            table = torch.stack(rows)
            first = gen.prefill_forward(
                torch.tensor(prompts), page_table=table, prompt_lens=[a.prompt] * B, sample_on_device=True
            )
            gen.decode_forward(first, torch.tensor([a.prompt] * B), page_table=table)
            bucket = gen._bucket(B)
            split = bucket % 4 == 0 and gen.model.use_user_split(bucket, a.prompt + 1)
            model_tid = gen.traces[("decode", bucket, split)]
            samp_tid = gen.traces[("sample",)]

            start = gen.positions[bucket]
            pos0 = torch.tensor([a.prompt + 1] * B, dtype=torch.int32)

            def loop(n, sample=True, model=True):
                gen._write(start, pos0)  # every repetition decodes the same positions (prompt + 1 ...)
                for _ in range(n):
                    if model:
                        ttnn.execute_trace(mesh, model_tid, cq_id=0, blocking=False)
                    if sample:
                        ttnn.execute_trace(mesh, samp_tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)

            n = a.gen // 4
            r["token_out_replay_ms_per_token"] = round(median_ms(lambda: loop(n), 5)[0] / n, 3)
            r["logits_only_replay_ms_per_step"] = round(median_ms(lambda: loop(n, sample=False), 5)[0] / n, 3)
            r["sampling_trace_ms"] = round(median_ms(lambda: loop(n, model=False), 5)[0] / n, 3)
            r["token_out_replay_t_s_u"] = round(1e3 / r["token_out_replay_ms_per_token"], 2)
            r["user_split_trace"] = split
            for rr in rows:
                gen._release(rr)
            gen.reset()
            res[f"decode_b{B}"] = r
            print("DECODE", B, r, flush=True)
        res["counters_total"] = dict(gen.counters)
        gen.teardown()
    finally:
        close_qb2_mesh(mesh)
    print("PERF", json.dumps(res), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
