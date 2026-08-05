#!/usr/bin/env python3
"""Spec-decode latency sweep (batch-1, greedy), prefix-cache-aware.
Methodology: for each context, WARM the prompt (short call → caches the prefill + compiles decode), then a
MEASURED call whose prefill is a cache hit, so its E2EL ≈ pure decode. decode t/s/u = OSL / E2EL_measured.
HIGH-COPY prompt (reproduce-a-block) so the ngram drafter hits (where agentic edits live). Reports
ISL/OSL/E2EL + t/s/u (never ms/tok). Run once spec-ON, once spec-OFF; compare."""
import json
import sys
import time
import urllib.request

URL = "http://localhost:8000/v1/completions"
MODEL = "poolside/Laguna-XS-2.1"
TAG = sys.argv[1] if len(sys.argv) > 1 else "run"
N = 160


def build_prompt(approx_tokens):
    nlines = max(20, approx_tokens // 12)
    block = "\n".join(
        f"    self.layer_{i} = nn.Linear({64 + i}, {128 + i})  # projection block number {i}" for i in range(nlines)
    )
    return f"Reproduce this class body exactly, twice.\n\nclass Net(nn.Module):\n  def __init__(self):\n{block}\n\nclass Net(nn.Module):\n  def __init__(self):\n"


def call(prompt, max_tokens):
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0}).encode()
    t0 = time.time()
    r = json.load(
        urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=900)
    )
    return time.time() - t0, r["usage"]["prompt_tokens"], r["usage"]["completion_tokens"]


print(f"[{TAG}] ctx | ISL | OSL | E2EL(decode-only, cache-warm) | DECODE t/s/u", flush=True)
for ctx in (512, 2048, 8192):
    try:
        p = build_prompt(ctx)
        call(p, 8)  # WARM: cache the prefill + compile decode (timing discarded)
        e2e, isl, osl = call(p, N)  # MEASURED: prefill is a cache hit → E2EL ≈ pure decode
        dtps = osl / e2e if e2e > 0 else 0.0
        print(f"[{TAG}] ctx={ctx:6d} | ISL={isl:6d} | OSL={osl:4d} | E2EL={e2e:6.3f}s | {dtps:6.1f} t/s/u", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"[{TAG}] ctx={ctx:6d} | ERROR {type(ex).__name__}: {ex}", flush=True)
print(f"[{TAG}] done", flush=True)
