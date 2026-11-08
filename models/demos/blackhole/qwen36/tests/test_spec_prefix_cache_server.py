# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SERVER-level checks of spec prefix caching (needs a vLLM TT server with the MTP spec class, --enable-prefix-caching,
--block-size 64, speculative decoding on; NOT started by this test). Skipped unless SPEC_PC_URL is set.

  SPEC_PC_URL=http://localhost:8000 SPEC_PC_MODEL=<served model name> pytest .../test_spec_prefix_cache_server.py -svq

 1. lossless: greedy output of B run cold (before A) == greedy output of B after A (prefix hit), token for token over
    the first MATCH_MIN tokens (bf16 near-ties may flip later).
 2. divergence exactly on a 128 boundary: B' shares 4352 tokens (34 x 128) with A.
 3. acceptance: mean accepted tokens/step (vLLM /metrics vllm:spec_decode_num_accepted_tokens_total / drafts_total)
    for the cold run vs the hit run must agree within ACC_TOL (hit must not degrade the drafter).
Run the same file against a server started with QWEN36_SPEC_PREFIX_CACHE=0 to get the miss-only baseline.
"""
import os
import re

import pytest

requests = pytest.importorskip("requests")
URL = os.environ.get("SPEC_PC_URL")
MODEL = os.environ.get("SPEC_PC_MODEL", "Qwen/Qwen3.8-27B")
MATCH_MIN = 24
ACC_TOL = 0.25
pytestmark = pytest.mark.skipif(not URL, reason="set SPEC_PC_URL to a running vLLM TT server")


def _complete(prompt_ids, n=64):
    r = requests.post(
        f"{URL}/v1/completions",
        json={
            "model": MODEL,
            "prompt": prompt_ids,
            "max_tokens": n,
            "temperature": 0,
            "return_tokens_as_token_ids": True,
        },
        timeout=1800,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["text"]


def _metric(name):
    txt = requests.get(f"{URL}/metrics", timeout=60).text
    return sum(float(m.group(1)) for m in re.finditer(rf"^{name}(?:{{[^}}]*}})? ([0-9.e+]+)$", txt, re.M))


def _accept_rate(fn):
    a0, d0 = _metric("vllm:spec_decode_num_accepted_tokens_total"), _metric("vllm:spec_decode_num_drafts_total")
    out = fn()
    a1, d1 = _metric("vllm:spec_decode_num_accepted_tokens_total"), _metric("vllm:spec_decode_num_drafts_total")
    return out, (a1 - a0) / max(d1 - d0, 1)


def _prompts():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.environ.get("HF_MODEL", MODEL), trust_remote_code=True)
    import glob

    text = "\n\n".join(
        open(f, errors="ignore").read() for f in sorted(glob.glob("tech_reports/**/*.md", recursive=True))[:60]
    )
    ids = tok(text, add_special_tokens=False)["input_ids"]
    assert len(ids) > 20000
    return ids


@pytest.mark.parametrize("shared_len", [4352, 4288])  # 128-boundary divergence, 64-block divergence
def test_hit_is_lossless_and_keeps_acceptance(shared_len):
    ids = _prompts()
    shared = ids[:shared_len]
    A = shared + ids[10000:10700]
    B = shared + ids[12000:12600]
    cold, acc_cold = _accept_rate(lambda: _complete(B))  # B first: cold (may itself be a miss on a fresh server)
    _complete(A)  # populates vLLM blocks + the GDN snapshots (and learns for the 3rd request)
    _complete(B)  # 2nd request with this prefix: learn point
    hit, acc_hit = _accept_rate(lambda: _complete(B))
    n = min(len(cold), len(hit), MATCH_MIN)
    assert cold[:n] == hit[:n], f"hit output diverged from cold output:\ncold={cold!r}\nhit ={hit!r}"
    assert acc_hit >= acc_cold - ACC_TOL, f"acceptance dropped on a hit: cold {acc_cold:.2f} hit {acc_hit:.2f}"
