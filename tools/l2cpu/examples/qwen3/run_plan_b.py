#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end Plan B demo (host-free decode loop with x280 sampling), in this host order:
open device -> allocate arena -> load firmware -> clock/reset -> wait READY -> load model -> prefill -> capture
trace -> enqueue all decode steps -> stream the text by draining the arena output ring while the device runs.
The host never reads logits or token tensors during decode; a monitor thread watches heartbeats / errors / the
wait op's status word and exits nonzero (17) on trouble.

    tools/l2cpu/scripts/l2cpu_run.sh "demo" $PY tools/l2cpu/examples/qwen3/run_plan_b.py --model Qwen/Qwen3-8B --batch 1 --max-new-tokens 256 --temperature 0.7 --top-k 50 --top-p 0.9 --seed 1234
    ... --batch 32 --per-user-mix            # sampling_mix.py settings per user
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import deps  # noqa: E402,F401  (bindings to the other l2cpu components, see deps.py)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--batch", type=int, default=1, choices=[1, 32])
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--per-user-mix", action="store_true", help="per-user settings/seeds from sampling_mix.py")
    ap.add_argument(
        "--prompts-file",
        default=None,
        help='demo-style JSON list of {"prompt": ...} (default: the demo\'s 128-token questions)',
    )
    ap.add_argument("--input-tokens", type=int, default=None, help="force every prompt to exactly this many tokens")
    ap.add_argument("--quiet", action="store_true", help="no streaming, only the final texts")
    a = ap.parse_args()
    import decode_harness as dh

    home = dh.check_env()
    os.chdir(home)
    os.environ["HF_MODEL"] = a.model
    os.environ["TT_CACHE_PATH"] = deps.weights_cache(a.model)
    import torch

    torch.manual_seed(213919)
    from plan_b import PlanB, prepare_group_planb
    from l2cpu_monitor import SamplingMonitor
    from l2cpu_sampler import session

    B, N = a.batch, a.max_new_tokens

    class NS(types.SimpleNamespace):
        def __getattr__(self, k):
            return None

    t_start = time.time()
    # open device, allocate arena, load firmware, clock/reset, READY (x280=True), then load the model
    h = dh.DecodeHarness(
        NS(
            trace_region_size=200_000_000,
            precision="performance",
            batch=B,
            max_seq_len=1024,
            prefill_mode="batched",
            convert="oneshot",
            one_trace=False,
            input_tokens=a.input_tokens,
            clear_kv=1,
            max_new_tokens=N,
            hash_rows=0,
            x280=True,
        )
    )
    fw = session()["fw"]
    mon = SamplingMonitor(fw, log=lambda *x: dh.log("[monitor]", *x)).start()
    pb = PlanB(h.mesh, session=session(), uncached=B > 1, log=dh.log)
    if a.prompts_file:
        prompts = [d["prompt"] if isinstance(d, dict) else str(d) for d in json.load(open(a.prompts_file))]
        while len(prompts) < B:
            prompts = prompts + prompts
        prompts = prompts[:B]
    else:
        prompts = dh.load_prompts(home, B)
    if a.per_user_mix:
        from sampling_mix import user_params

        up = user_params(B)
        params = [dict(temperature=t, top_k=k, top_p=p) for (t, k, p, _) in up]
        seeds = [sd for (*_, sd) in up]
    else:
        params = [dict(temperature=a.temperature, top_k=a.top_k, top_p=a.top_p)] * B
        seeds = [a.seed] * B
    g = prepare_group_planb(h, pb, prompts, params, seeds)  # prefill, first token, firmware config, capture, inputs
    dh.log(f"ready to decode after {time.time() - t_start:.1f} s (prefill {g['prefill_s']:.2f} s)")
    tok = h.tokenizer
    toks = [[t] for t in g["first"]]
    shown = [""] * B
    stop_ids = set(getattr(tok, "stop_tokens", None) or [tok.eos_token_id])
    ring_seen = [0]

    def drain(done):
        if done <= ring_seen[0]:
            return
        new = pb.ring_tokens_range(ring_seen[0], done, B)
        ring_seen[0] = done
        for row in new:
            for b in range(B):
                toks[b].append(row[b])
        if a.quiet:
            return
        if B == 1:
            text = tok.decode(toks[0])
            sys.stdout.write(text[len(shown[0]) :])
            sys.stdout.flush()
            shown[0] = text
        elif done % 32 == 0 or done == N:
            for b in range(B):
                text = tok.decode(toks[b])
                if text != shown[b]:
                    print(f"[user {b:2d}] +{text[len(shown[b]):]!r}", flush=True)
                    shown[b] = text

    dt, t_enq, nring = pb.run(N, poll=True, poll_cb=drain)
    drain(N)
    print()
    pb.check_fw()
    for b in range(B):
        gen = toks[b]
        cut = next((i for i, t in enumerate(gen) if t in stop_ids), None)
        print(
            f"=== user {b} ({params[b]['temperature']}/{params[b]['top_k']}/{params[b]['top_p']} seed {seeds[b]}): "
            f"{tok.decode(gen[:cut] if cut is not None else gen)!r}"
        )
    ms = dt / N * 1e3
    print(
        f"\nPlan B: {N} tokens per user x {B} users in {dt:.2f} s: {ms:.2f} ms/token, {1e3 / ms:.2f} tokens/s/user, "
        f"{B * 1e3 / ms:.0f} tokens/s total; host device ops during decode: {N} enqueues + ring polls; monitor {mon.stop()}"
    )
    h.ttnn.close_mesh_device(h.mesh)
    return 0


if __name__ == "__main__":
    try:
        rc = main()
        sys.stdout.flush()
        os._exit(rc)
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(3)
