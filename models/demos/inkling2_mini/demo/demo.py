# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Mini on one p100a: DRAM report, trace checks, greedy generation, decode speed.

    flock /tmp/tt-card.lock python -m models.demos.inkling2_mini.demo.demo --max-ctx 131072 \
        [--skip-app-trace] [--gen 64] [--perf-pos 2048 32768] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from models.demos.inkling2_mini.tt.generator import Generator
from models.demos.inkling2_mini.tt.model import Inkling2Model

GOLDEN = Path("/work/inkling-bringup/golden")
TOKENIZER = "/work/models/Inkling-2-Mini/tokenizer.json"
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
    ap.add_argument("--max-ctx", type=int, default=131072)
    ap.add_argument("--trace-mib", type=int, default=512)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--prompts", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--perf-pos", type=int, nargs="*", default=[2048, 32768])
    ap.add_argument("--perf-steps", type=int, default=20)
    ap.add_argument("--skip-app-trace", action="store_true")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()
    res = {"max_ctx": args.max_ctx, "trace_mib": args.trace_mib}
    t0 = time.time()
    dev = ttnn.open_device(device_id=0, trace_region_size=args.trace_mib << 20)
    try:
        res["dram_free_open_gib"] = dram_free(dev)[0]
        model = Inkling2Model(dev, max_ctx=args.max_ctx, log=lambda s: None)
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

        try:
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(TOKENIZER)
        except Exception as ex:  # noqa: BLE001
            tok = None
            res["tokenizer_error"] = str(ex)
        gold = json.loads((GOLDEN / "greedy.json").read_text())
        res["gen"] = []
        for k in args.prompts:
            ids = torch.load(GOLDEN / f"prompt{k}" / "input_ids.pt")[0].tolist()
            t1 = time.time()
            new, times = gen.generate(ids, args.gen, traced=True)
            g = {"prompt": k, "ids": new, "seconds": round(time.time() - t1, 1)}
            if times:
                g["decode_tok_s"] = len(times) / sum(times)
            if k == 0:
                ref = gold["new_ids"]
                n = min(len(ref), len(new))
                g["match_golden_first16"] = sum(int(a == b) for a, b in zip(new[:n], ref[:n]))
                g["first_mismatch"] = next((i for i in range(n) if new[i] != ref[i]), None)
            if tok is not None:
                g["text"] = tok.decode(new, skip_special_tokens=False)
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
