# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fast reduced full-model probe (debugging only; final evidence comes from the 48-layer model).

Builds ``Gemma4Model`` with one real layer of each kind (default layers 0 and 5), the real embedding / final norm /
LM head / sampler and the full generator (prepare + traces), then checks prefill logits against the matching reduced
HF model, traced decode logits (teacher-forced on HF's greedy tokens), on-device greedy token feedback and the
refresh counters.

  python models/autoports/google_gemma_4_12b_it/tests/full_model_probe.py --layers 0,5 --context 16384
"""

import argparse
import json
import time

import torch

import ttnn
from models.autoports.google_gemma_4_12b_it.tests.full_model_ref import hf_causal_lm, hf_greedy, hf_logits
from models.autoports.google_gemma_4_12b_it.tests.hf_reference import pcc
from models.autoports.google_gemma_4_12b_it.tt.generator import Gemma4Generator
from models.autoports.google_gemma_4_12b_it.tt.model import Gemma4Model
from models.autoports.google_gemma_4_12b_it.tt.multichip_decoder import close_qb2_mesh, open_qb2_mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,5")
    ap.add_argument("--context", type=int, default=16384)
    ap.add_argument("--blocks", type=int, default=600)
    ap.add_argument("--buckets", default="1,4,8,16,32")
    ap.add_argument("--lengths", default="17,300")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--no_hf", action="store_true")
    ap.add_argument("--hold", type=float, default=0.0, help="seconds to keep the mesh open at the end (watcher polls)")
    a = ap.parse_args()
    torch.set_num_threads(16)
    layers = [int(v) for v in a.layers.split(",")]
    res = {}
    mesh = open_qb2_mesh(trace_region_size=0)
    try:
        t0 = time.time()
        model = Gemma4Model(mesh, layer_indices=layers, max_context=a.context, num_blocks=a.blocks, log=lambda *_: None)
        res["build_s"] = round(time.time() - t0, 1)
        t0 = time.time()
        gen = Gemma4Generator(model, decode_buckets=[int(b) for b in a.buckets.split(",")])
        res["prepare_s"] = round(time.time() - t0, 1)
        print("BUILD", res, flush=True)
        hf = None if a.no_hf else hf_causal_lm(layers)
        prompt = gen.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Write a short poem about the sea and explain its meter."}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
        base = (prompt * 1000)[:8300]
        for n in [int(v) for v in a.lengths.split(",")]:
            ids = base[:n]
            t0 = time.time()
            got = gen.prefill_logits(ids)[0]
            dt = time.time() - t0
            row = {"seconds": round(dt, 2)}
            if hf is not None:
                want = hf_logits(hf, ids)
                row["pcc_all"] = round(pcc(want, got), 5)
                row["pcc_last"] = round(pcc(want[-1], got[-1]), 5)
                row["top1_agree"] = round(float((want.argmax(-1) == got.argmax(-1)).float().mean()), 4)
            res[f"prefill_{n}"] = row
            print("PREFILL", n, row, flush=True)
            gen.reset()
        # greedy token-out (on-device split sampling, traced) vs HF greedy
        ids = prompt
        steps = a.steps
        t0 = time.time()
        tt_tokens = gen.generate(ids, steps, stop_on_eos=False)
        res["generate_s"] = round(time.time() - t0, 2)
        res["tt_tokens"] = tt_tokens
        gen.reset()
        # traced single-chunk prefill + on-device first token, a pipelined batch (bucket 4, mixed prompt lengths) and a
        # seeded top-k / top-p request (the optimized full model's token-out paths)
        traced = []
        for n in [int(v) for v in a.lengths.split(",")]:
            r = gen._alloc_row(n + 4)
            traced.append(
                int(
                    gen.prefill_forward(
                        torch.tensor([base[:n]]), page_table=r[None], prompt_lens=[n], sample_on_device=True
                    )[0]
                )
            )
            gen._release(r)
            gen.reset()
        res["traced_prefill_first_tokens"] = traced
        if 4 in gen.buckets:
            res["batch4_lengths"] = [len(t) for t in gen.generate([ids, prompt[:9], ids[:-3], prompt[:20]], steps)]
            gen.reset()
        from models.autoports.google_gemma_4_12b_it.tt.generator import SamplingParams

        res["sampled_tokens"] = gen.generate(
            ids, steps, sampling_params=SamplingParams(top_k=32, top_p=0.9, seed=3), stop_on_eos=False
        )
        gen.reset()
        gen.set_sampling_params()
        if hf is not None:
            hf_tokens = hf_greedy(hf, ids, steps)
            res["hf_tokens"] = hf_tokens
            res["greedy_match"] = sum(int(x == y) for x, y in zip(tt_tokens, hf_tokens))
            # teacher-forced decode logits through the traced model (return_logits path) vs HF on the same tokens
            row = gen._alloc_row(len(ids) + steps)
            first = gen.prefill_forward(torch.tensor([ids]), page_table=row[None], prompt_lens=[len(ids)])[0, 0]
            seq = ids + hf_tokens
            want = hf_logits(hf, seq)
            pccs = [pcc(want[len(ids) - 1], first)]
            for i in range(steps - 1):
                lg = gen.decode_forward(
                    torch.tensor([hf_tokens[i]]), torch.tensor([len(ids) + i]), page_table=row[None], return_logits=True
                )[0]
                pccs.append(pcc(want[len(ids) + i], lg))
            res["decode_logit_pcc"] = [round(v, 5) for v in pccs]
            gen._release(row)
            gen.reset()
        res["counters"] = dict(gen.counters)
        print("RESULT", json.dumps(res), flush=True)
        gen.teardown()
        if a.hold:
            ttnn.synchronize_device(mesh)
            time.sleep(a.hold)
    finally:
        close_qb2_mesh(mesh)


if __name__ == "__main__":
    main()
