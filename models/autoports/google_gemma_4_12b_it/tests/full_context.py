# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-context check of the public generator plus the DRAM accounting of the context contract.

* builds the 48-layer generator with ``--blocks`` KV blocks per layer (default: one 262144-token request + the 2
  reserved blocks) and records per-chip DRAM after the build, after ``prepare`` (traces) and after the request;
* runs one chat request whose prompt is ``--tokens`` long (default 262110: not a multiple of any chunk, page or tile; prompt + answer stay inside 262144)
  with a pass-key sentence near the start, and generates the answer through traced token-out decode at the end of
  the context (needle retrieval across ~262K tokens);
* records prefill time and decode time per token at that depth.

  python models/autoports/google_gemma_4_12b_it/tests/full_context.py --out doc/full_model/logs/full_context.json
"""

import argparse
import json
import time

import torch

import ttnn
from models.autoports.google_gemma_4_12b_it.tt.generator import build_generator
from models.autoports.google_gemma_4_12b_it.tt.multichip_decoder import close_qb2_mesh, open_qb2_mesh


def dram(mesh):
    v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return dict(
        allocated_per_chip=v.total_bytes_allocated_per_bank * v.num_banks,
        free_per_chip=v.total_bytes_free_per_bank * v.num_banks,
        total_per_chip=v.total_bytes_per_bank * v.num_banks,
        largest_free_block_per_bank=v.largest_contiguous_bytes_free_per_bank,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=262110)
    ap.add_argument("--blocks", type=int, default=0)
    ap.add_argument("--gen", type=int, default=24)
    ap.add_argument("--passkey", default="68395")
    ap.add_argument("--skip_request", action="store_true")
    ap.add_argument("--build-kwargs", default="{}", help="JSON kwargs for build_generator (e.g. a precision config)")
    ap.add_argument(
        "--logit_trace",
        type=int,
        default=0,
        help="decode the answer greedily from host logits instead and record the top-5 tokens / logits of the first "
        "N steps (margin diagnostics; the default path samples on device)",
    )
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(16)
    mesh = open_qb2_mesh(trace_region_size=0)
    res = {"dram_open": dram(mesh)}
    try:
        t0 = time.time()
        kw = {"prepare": False, **json.loads(a.build_kwargs)}
        if a.blocks:
            kw["num_blocks"] = a.blocks
        gen = build_generator(None, mesh, **kw)
        res["build_s"] = round(time.time() - t0, 1)
        res["num_blocks"] = gen.model.num_blocks
        res["precision_config"] = gen.model.precision.config_id
        res["lm_head"] = {k: str(v) for k, v in gen.model.lm_policy.items()}
        res["dram_after_build"] = dram(mesh)
        t0 = time.time()
        gen.prepare()
        res["prepare_s"] = round(time.time() - t0, 1)
        res["dram_after_prepare"] = dram(mesh)
        print("READY", json.dumps(res), flush=True)
        if not a.skip_request:
            tok = gen.tokenizer
            filler = open("tech_reports/LLMs/llms.md").read()
            question = (
                "\n\nWhat is the secret pass key mentioned near the start of this text? Answer with the number only."
            )
            needle = f"\nThe secret pass key is {a.passkey}. Remember it.\n"

            def render(reps):
                body = filler[:2000] + needle + filler[2000:] + filler * (reps - 1) + question
                return tok.apply_chat_template(
                    [{"role": "user", "content": body}], add_generation_prompt=True, tokenize=True, return_dict=False
                )

            n_fill = len(tok.encode(filler, add_special_tokens=False))
            reps = max(1, a.tokens // n_fill)
            ids = render(reps)
            while len(ids) > a.tokens:
                reps -= 1
                ids = render(reps)
            # top up to the exact length with filler text placed before the needle-free tail
            extra = a.tokens - len(ids)
            pad = tok.encode(filler, add_special_tokens=False)[:extra]
            cut = (
                len(ids)
                - len(
                    tok.apply_chat_template(
                        [{"role": "user", "content": question}],
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=False,
                    )
                )
                + 3
            )
            ids = ids[:cut] + pad + ids[cut:]
            res["prompt_tokens"] = len(ids)
            row = gen._alloc_row(len(ids) + a.gen)
            if a.logit_trace:
                tok_ = gen.tokenizer
                trace = []

                def top5(lg):
                    v, i = lg.float().topk(5)
                    return [[tok_.decode([int(x)]), round(float(y), 3)] for x, y in zip(i, v)]

                lg = gen.prefill_forward(torch.tensor([ids]), page_table=row[None], prompt_lens=[len(ids)])[0, -1]
                out = [int(lg.argmax())]
                trace.append(top5(lg))
                pos = len(ids)
                for _ in range(a.logit_trace - 1):
                    lg = gen.decode_forward(
                        torch.tensor([out[-1]]), torch.tensor([pos]), page_table=row[None], return_logits=True
                    ).reshape(-1)
                    trace.append(top5(lg))
                    out.append(int(lg.argmax()))
                    pos += 1
                res["logit_trace"] = trace
                res["answer"] = tok_.decode(out, skip_special_tokens=False)
                res["passkey_found_within_trace_steps"] = a.passkey in res["answer"]
                print("ANSWER", repr(res["answer"]), flush=True)
                for k, t in enumerate(trace):
                    print("STEP", k, t, flush=True)
                gen.teardown()
                return finish(res, a)
            t0 = time.perf_counter()
            first = gen.prefill_forward(
                torch.tensor([ids]), page_table=row[None], prompt_lens=[len(ids)], sample_on_device=True
            )
            res["prefill_s"] = round(time.perf_counter() - t0, 2)
            out = [int(first[0])]
            pos = len(ids)
            stamps = []
            for _ in range(a.gen - 1):
                nxt = gen.decode_forward(torch.tensor([out[-1]]), torch.tensor([pos]), page_table=row[None])
                stamps.append(time.perf_counter())
                out.append(int(nxt[0]))
                pos += 1
                if out[-1] in gen.eos_token_ids:
                    break
            steps = [(y - x) * 1e3 for x, y in zip(stamps, stamps[1:])]
            res["decode_ms_per_token_at_depth"] = round(sorted(steps)[len(steps) // 2], 3) if steps else None
            res["answer_tokens"] = out
            res["answer"] = tok.decode(out, skip_special_tokens=False)
            res["passkey_found"] = a.passkey in res["answer"]
            res["dram_after_request"] = dram(mesh)
            res["counters"] = dict(gen.counters)
            print("ANSWER", repr(res["answer"]), flush=True)
        gen.teardown()
    finally:
        close_qb2_mesh(mesh)
    finish(res, a)


def finish(res, a):
    print("FULLCTX", json.dumps(res), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
