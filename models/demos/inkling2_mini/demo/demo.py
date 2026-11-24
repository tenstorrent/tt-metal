# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Mini on one p100a: DRAM report, trace checks, greedy generation, decode speed.

    python -m models.demos.inkling2_mini.demo.demo --prompt "Write a haiku about a lake." [--gen 64] [--fast]
    python -m models.demos.inkling2_mini.demo.demo --golden DIR --max-ctx 131072 \
        [--skip-app-trace] [--gen 64] [--perf-pos 2048 32768] [--json out.json]

Weights come from ``$INKLING_CACHE`` (default: the run mode's cache, see modes.py; ``--fast`` or
``INKLING_MODE=fast`` picks the faster, less exact own3-l8 experts). ``--prompt`` is wrapped in the model's chat template (tokenizer from
``--tokenizer``, default ``$INKLING_CKPT``) and replaces the golden prompts; without it the golden prompts
(``--golden``, default ``$INKLING_GOLDEN``) are generated and prompt 0 is compared with the golden greedy tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

import ttnn
from models.demos.inkling2_mini import modes
from models.demos.inkling2_mini.tt.device import open_device
from models.demos.inkling2_mini.tt.generator import Generator
from models.demos.inkling2_mini.tt.model import Inkling2Model

GiB = 2**30


def dram_free(dev):
    v = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
    return v.num_banks * v.total_bytes_free_per_bank / GiB, v.num_banks * v.largest_contiguous_bytes_free_per_bank / GiB


def trace_one_app(model, L=11, p=1, pos=5):
    """Capture one layer application, replay twice from the same state: bit-equal?"""
    dev = model.device
    h = ttnn.from_torch(torch.randn(1, 1, 1, 2304), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    stack = model.rows.seal(None, h, 0)  # depth-2 attn-res for L11
    model.reset()
    model.positions.update(pos)
    model.positions.prepare()
    eager, _, _ = model._app(L, p, h, stack, 1, None)
    eager = ttnn.to_torch(eager)
    model.reset()
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    model.positions.prepare()
    out, _, _ = model._app(L, p, h, stack, 1, None)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    runs = []
    for _ in range(2):
        model.reset()
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        runs.append(ttnn.to_torch(out))
    ttnn.release_trace(dev, tid)
    return {
        "app_trace_bit_equal": bool(torch.equal(runs[0], runs[1])),
        "app_trace_eq_eager": bool(torch.equal(runs[0], eager)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--max-ctx",
        type=int,
        default=196608,
        help="context limit. The far KV cache takes 19,584 B per token of it: 196608 leaves ~2.4 GiB of DRAM free "
        "(decode only); 327680 is the largest that fits (0.05 GiB free). Decode cost follows the position, not this.",
    )
    ap.add_argument("--trace-mib", type=int, default=512)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--prompts", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--perf-pos", type=int, nargs="*", default=[2048, 32768])
    ap.add_argument("--perf-steps", type=int, default=20)
    ap.add_argument("--skip-app-trace", action="store_true")
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--prompt", type=str, help="user message to answer (replaces the golden prompts)")
    ap.add_argument("--golden", type=Path, default=os.environ.get("INKLING_GOLDEN"), help="golden dir")
    ap.add_argument("--tokenizer", default=os.environ.get("INKLING_CKPT"), help="HF tokenizer dir (checkpoint dir)")
    ap.add_argument(
        "--fast",
        action="store_true",
        help="fast mode (also INKLING_MODE=fast): own3-l8 experts, ~1.5 ms/token faster decode, KL .0096 vs .0067",
    )
    args = ap.parse_args()
    if args.prompt is None and args.golden is None:
        ap.error("give --prompt, or --golden / INKLING_GOLDEN for the golden prompts")
    if args.prompt is not None and args.tokenizer is None:
        ap.error("--prompt needs --tokenizer or INKLING_CKPT")
    res = {"max_ctx": args.max_ctx, "trace_mib": args.trace_mib}
    t0 = time.time()
    dev = open_device(trace_region_size=args.trace_mib << 20)
    try:
        res["dram_free_open_gib"] = dram_free(dev)[0]
        res["cache"] = str(modes.roots(args.fast)[0])
        model = Inkling2Model(dev, root=res["cache"], max_ctx=args.max_ctx, log=lambda s: None)
        res["load_s"] = round(time.time() - t0, 1)
        res["dram_free_after_load_gib"], res["dram_largest_after_load_gib"] = dram_free(dev)
        print(json.dumps(res), flush=True)

        if not args.skip_app_trace:
            res.update(trace_one_app(model))
            print(json.dumps(res), flush=True)

        gen = Generator(model)
        t1 = time.time()
        gen.capture()
        res["capture_s"] = round(time.time() - t1, 1)
        tv = ttnn.get_memory_view(dev, ttnn.BufferType.TRACE)
        res["trace_used_mib"] = tv.num_banks * tv.total_bytes_allocated_per_bank / 2**20
        res["dram_free_after_trace_gib"] = dram_free(dev)[0]
        # full-step trace: replay twice from the same state -> bit-equal; and vs eager
        outs = []
        for _ in range(2):
            model.reset()
            o = gen.decode(200005, 3, traced=True)
            outs.append(ttnn.to_torch(o["logits"]).clone())
        model.reset()
        e = ttnn.to_torch(gen.decode(200005, 3, traced=False)["logits"])
        res["step_trace_bit_equal"] = bool(torch.equal(outs[0], outs[1]))
        res["step_trace_eq_eager"] = bool(torch.equal(outs[0], e))
        res["step_trace_vs_eager_maxdiff"] = float((outs[0] - e).abs().max())
        print(json.dumps(res), flush=True)

        tok = None
        if args.tokenizer is not None:
            try:
                from transformers import AutoTokenizer

                tok = AutoTokenizer.from_pretrained(args.tokenizer)
            except Exception as ex:  # noqa: BLE001
                if args.prompt is not None:
                    raise
                res["tokenizer_error"] = str(ex)
        if args.prompt is not None:
            ids = tok.apply_chat_template([{"role": "user", "content": args.prompt}], tokenize=True)
            prompts = [("prompt", ids["input_ids"] if hasattr(ids, "keys") else list(ids))]
            gold = None
        else:
            gold = json.loads((args.golden / "greedy.json").read_text())
            prompts = [(k, torch.load(args.golden / f"prompt{k}" / "input_ids.pt")[0].tolist()) for k in args.prompts]
        res["gen"] = []
        for k, ids in prompts:
            t1 = time.time()
            new, times = gen.generate(ids, args.gen, traced=True)
            g = {"prompt": k, "ids": new, "seconds": round(time.time() - t1, 1)}
            if times:
                g["decode_tok_s"] = len(times) / sum(times)
            if gold is not None and k == 0:
                ref = gold["new_ids"]
                n = min(len(ref), len(new))
                g["match_golden_first16"] = sum(int(a == b) for a, b in zip(new[:n], ref[:n]))
                g["first_mismatch"] = next((i for i in range(n) if new[i] != ref[i]), None)
            if tok is not None:
                g["text"] = tok.decode(new, skip_special_tokens=False)
            if args.prompt is not None:
                print(g["text"], flush=True)
            res["gen"].append(g)
            print(json.dumps(g, ensure_ascii=False), flush=True)

        # decode speed at a given position (v1 cost depends on max_ctx, not on content)
        res["perf"] = []
        for pos in args.perf_pos:
            if pos >= args.max_ctx:
                continue
            gen.decode(1, pos)
            ttnn.synchronize_device(dev)
            t1 = time.perf_counter()
            for i in range(args.perf_steps):
                gen.decode(1, pos + i)
            ttnn.synchronize_device(dev)
            dt = (time.perf_counter() - t1) / args.perf_steps
            t1 = time.perf_counter()
            for i in range(args.perf_steps):
                o = gen.decode(1, pos + i)
                gen.read_token(o)
            dt_sync = (time.perf_counter() - t1) / args.perf_steps
            res["perf"].append(
                {
                    "pos": pos,
                    "ms": dt * 1e3,
                    "tok_s": 1 / dt,
                    "ms_with_token_read": dt_sync * 1e3,
                    "tok_s_with_read": 1 / dt_sync,
                }
            )
            print(json.dumps(res["perf"][-1]), flush=True)
        gen.release()
    finally:
        ttnn.close_device(dev)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
