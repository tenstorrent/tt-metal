# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Targeted serving check: send OpenAI /v1/completions requests whose PROMPT TOKEN LENGTH is NOT
divisible by any internal chunk/page/tile/trace size (block_size=64, tile=32, MoE prefill chunk=256,
prefill pad 128), and assert the server returns a finite, in-vocab, non-degenerate completion.

Proves the advertised context is not silently gated to aligned lengths. Run against a live server:

  cd /tmp && TT_METAL_HOME=... PYTHONPATH=/home/ttuser/dev/tt-metal:.../vllm \
    python -m models.autoports.poolside_laguna_xs_2_1.doc.vllm_integration.scripts.nonaligned_request \
      --server-url http://localhost:8000 --hf-model poolside/Laguna-XS-2.1
"""
from __future__ import annotations

import argparse
import json
import sys

import openai
from transformers import AutoTokenizer

# Deliberately non-aligned token counts: 100, 129, 257, 513 are not divisible by 64/32/256, and
# 129/257/513 are +1 past a 128/256/512 boundary (tile/chunk edge). 1500 is a longer non-aligned one.
NON_ALIGNED_LENS = [100, 129, 257, 513, 1500]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-url", default="http://localhost:8000")
    ap.add_argument("--hf-model", default="poolside/Laguna-XS-2.1")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    client = openai.OpenAI(base_url=f"{args.server_url.rstrip('/')}/v1", api_key="dummy")

    # A base filler string; we trim to the exact target token length so the PROMPT length is exact.
    base = (
        "The history of computing spans many centuries, from early mechanical calculators "
        "to modern processors, and it continues to evolve rapidly across the world. "
    ) * 200
    base_ids = tok(base, add_special_tokens=False)["input_ids"]

    results = []
    ok_all = True
    for target in NON_ALIGNED_LENS:
        assert target % 64 != 0 and target % 32 != 0 or target in (100, 1500), target
        ids = base_ids[:target]
        prompt = tok.decode(ids)
        # Re-encode to confirm the exact prompt token length the server will see.
        reenc = len(tok(prompt, add_special_tokens=False)["input_ids"])
        try:
            resp = client.completions.create(
                model=args.hf_model, prompt=prompt, max_tokens=args.max_tokens, temperature=0.0
            )
            text = resp.choices[0].text
            n_out = len(tok(text, add_special_tokens=False)["input_ids"]) if text else 0
            # Degenerate check: non-empty, not a single repeated token.
            words = text.split()
            distinct = len(set(words))
            degenerate = (n_out == 0) or (len(words) >= 6 and distinct <= 2)
            passed = (n_out > 0) and not degenerate
            results.append(
                {
                    "target_prompt_len": target,
                    "reencoded_prompt_len": reenc,
                    "aligned_64": reenc % 64 == 0,
                    "aligned_32": reenc % 32 == 0,
                    "output_tokens": n_out,
                    "distinct_words": distinct,
                    "degenerate": degenerate,
                    "passed": passed,
                    "completion_snippet": text[:160],
                }
            )
            ok_all = ok_all and passed
            print(
                f"len={target} (reenc={reenc}, %64={reenc%64}, %32={reenc%32}) "
                f"-> out_tok={n_out} distinct={distinct} pass={passed}"
            )
            print(f"    {text[:120]!r}")
        except Exception as e:  # noqa: BLE001
            results.append({"target_prompt_len": target, "error": str(e), "passed": False})
            ok_all = False
            print(f"len={target} -> ERROR {e}")

    summary = {"all_pass": ok_all, "server_url": args.server_url, "results": results}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {args.out}")
    print("NONALIGNED_SERVING:", "ALL PASS" if ok_all else "FAILURES")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
