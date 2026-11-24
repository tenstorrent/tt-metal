# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4-Preview text demo: IN-PROCESS ``Hy4Engine`` on the 4x8 Blackhole galaxy, no vLLM / plugin / scheduler / client (step 0 of program/HY4_PORT_PLAN.md).

Boot ONCE (engine build + warm, ~15-18 min), then run any number of scenarios on that boot:
    python -m models.demos.blackhole.hy4_preview.demo.text_demo --text                       # a small chat prompt set: prefill + spec decode, prints text, TTFT, ms/round
    python -m models.demos.blackhole.hy4_preview.demo.text_demo --scenarios multirow,masked,cap0,pfx_hit_cold --check
    python -m models.demos.blackhole.hy4_preview.demo.text_demo --scenarios multirow --record # (re)write the greedy goldens for this config
    python -m models.demos.blackhole.hy4_preview.demo.text_demo --perf                        # prefill ms/token (1 and 4 rows), decode ms/round, tokens/round
Scenarios: see demo/scenarios.py (multirow, masked, cap0, pfx_hit_cold, xrow, stash, chunked_pf, tailpark, long, perf) and demo/scenario_sched.py (sched_xrow5, sched_cached, sched_victim). A scenario whose engine feature is not built or not enabled by env reports SKIP.
Environment: the PRODUCT flags (program/product17/FINAL_FLAGS.txt, models/demos/blackhole/hy4_preview/FINAL_FLAGS.txt) must be exported exactly as for the server, plus HY4_PFX=1 (prefix scenarios), HY4_XROW=1,
HY4_STASH=1 HY4_BLOCKS=1 (stash), HY4_PFX_LOSS_LOG=1 (optional). Run inside the product container with the b46 libs mounted (product17/final_launch.sh shows the mounts); weights from HY4_CKPT.
Env of the demo: K (spec draft length, default 3; the demo needs K >= 1), MAXLEN (default 262144; use 1048576 for the production-size final check), HY4_SERVE_LAYERS (e.g. 4 for a plumbing check), TRACE_BYTES,
DEMO_LONG (comma list of prompt lengths for the 'long' scenario), DEMO_GOLDEN (golden json path; default demo/golden/hy4_goldens.json), HY4_TOKENIZER (default /mnt/tt-data/Hy4-Preview).
Output lines: ``DEMO_GATE <scenario> <gate> PASS|FAIL|SKIP <value>``, ``DEMO_PERF {json}``, ``DEMO_RESULT pass=<n> fail=<n> skip=<n>``. Exit code 1 on any FAIL. Never SIGKILL: stop with TERM only.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "Explain in two sentences why the sky is blue.",
    "List three prime numbers greater than 100.",
]


def load_golden(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def config_tag(K, maxlen):
    return f"L{os.environ.get('HY4_SERVE_LAYERS', '78')}_K{K}_S{maxlen}"


def boot(K, maxlen):
    from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh
    from models.demos.blackhole.hy4_preview.tt.hy4_serve_engine import Hy4Engine

    md = open_mesh(trace=int(float(os.environ.get("TRACE_BYTES", "1900e6"))))
    t0 = time.time()
    eng = Hy4Engine(md, 4, maxlen, k=K)
    print(f"DEMO_BOOT built {time.time() - t0:.1f}s pages={eng.pages} K={K}", flush=True)
    t0 = time.time()
    eng.warm()
    print(f"DEMO_BOOT warm {time.time() - t0:.1f}s", flush=True)
    return md, eng, close_mesh


def text_mode(ctx):
    """chat prompts through the real tokenizer + chat template (reasoning_effort no_think), spec decode, greedy; prints the text and md5s. Falls back to the built-in token prompts if no tokenizer."""
    from transformers import AutoTokenizer

    from models.demos.blackhole.hy4_preview.demo import scenarios as S

    tok = AutoTokenizer.from_pretrained(
        os.environ.get("HY4_TOKENIZER", "/mnt/tt-data/Hy4-Preview"), trust_remote_code=True
    )
    ids = []
    for p in PROMPTS:
        try:
            x = tok.apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True, reasoning_effort="no_think"
            )
        except Exception:  # noqa: BLE001
            x = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True)
        x = x["input_ids"] if hasattr(x, "keys") else x
        ids.append([int(t) for t in x])
    S.release_all(ctx.eng)
    first, md5, dt, _ = S.prefill(ctx, {r: ids[r] for r in range(4)})
    n = max(len(i) for i in ids)
    print(
        f"DEMO_TEXT TTFT {dt:.2f}s for {sum(len(i) for i in ids)} prompt tokens (call cost follows the longest row: {n})",
        flush=True,
    )
    st, ms, acc = S.decode(ctx, first, 32)
    for r in range(4):
        print(
            f"--- user {r}: {PROMPTS[r]}\n{tok.decode(st[r], skip_special_tokens=True)}\n    lg_md5={md5.get(r)} tokens_md5={S.md5_ints(st[r])}",
            flush=True,
        )
        ctx.golden_check("text", f"lg_md5_row{r}", md5.get(r))
        ctx.golden_check("text", f"stream_md5_row{r}", S.md5_ints(st[r]))
    print(
        f"DEMO_TEXT decode {sum(ms) / len(ms):.1f} ms/round, {sum(acc) / len(acc):.2f} tokens/round/user (all rows)",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", action="store_true")
    ap.add_argument("--perf", action="store_true")
    ap.add_argument("--scenarios", default="")
    ap.add_argument(
        "--record", action="store_true", help="write the greedy goldens of this config instead of comparing"
    )
    ap.add_argument("--check", action="store_true", help="a missing golden is a FAIL (default: SKIP)")
    a = ap.parse_args()
    K = int(os.environ.get("K", "3"))
    maxlen = int(os.environ.get("MAXLEN", "262144"))
    if K < 1:
        sys.exit("the demo needs spec mode (K >= 1)")
    from models.demos.blackhole.hy4_preview.demo import scenarios as S

    gpath = os.environ.get("DEMO_GOLDEN", os.path.join(HERE, "golden", "hy4_goldens.json"))
    allg = load_golden(gpath)
    tag = config_tag(K, maxlen)
    md, eng, close_mesh = boot(K, maxlen)
    rc = 1
    try:
        ctx = S.Ctx(eng, golden=allg.setdefault(tag, {}), record=a.record)
        names = [s for s in a.scenarios.split(",") if s]
        if a.perf and "perf" not in names:
            names.append("perf")
        if a.text:
            text_mode(ctx)
        for n in names or ([] if a.text else S.DEFAULT):
            fn = S.SCENARIOS[n]
            kw = {}
            if n == "long" and os.environ.get("DEMO_LONG"):
                kw["lens"] = tuple(int(x) for x in os.environ["DEMO_LONG"].split(","))
            t0 = time.time()
            try:
                fn(ctx, **kw)
            except Exception as e:  # noqa: BLE001
                ctx.gate(n, "exception", False, f"{type(e).__name__}: {str(e)[:300]}")
            print(f"DEMO_SCENARIO {n} {time.time() - t0:.1f}s", flush=True)
        for p in ctx.perf:
            print("DEMO_PERF " + json.dumps(p), flush=True)
        if a.record:
            os.makedirs(os.path.dirname(gpath), exist_ok=True)
            with open(gpath, "w") as f:
                json.dump(allg, f, indent=1, sort_keys=True)
            print(f"DEMO_GOLDEN written {gpath} [{tag}]", flush=True)
        npass = sum(g[2] == "PASS" for g in ctx.gates)
        nfail = sum(g[2] == "FAIL" or (a.check and g[2] == "SKIP" and g[1].startswith("golden:")) for g in ctx.gates)
        nskip = sum(g[2] == "SKIP" for g in ctx.gates)
        print(f"DEMO_RESULT pass={npass} fail={nfail} skip={nskip}", flush=True)
        rc = 1 if nfail else 0
    finally:
        close_mesh(md)
    sys.exit(rc)


if __name__ == "__main__":
    main()
