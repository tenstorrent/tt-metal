# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Targeted live-server shape/page-boundary requests; reduced runs are not quality evidence."""

import argparse
import concurrent.futures
import json
from pathlib import Path

import requests
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--lengths", default="31,33,127,129,31,33")
    parser.add_argument("--generation", type=int, default=70)
    parser.add_argument("--concurrent", action="store_true")
    args = parser.parse_args()
    models = requests.get(args.url + "/v1/models", timeout=10).json()
    model = models["data"][0]["id"]
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B", local_files_only=True)
    base = tokenizer.encode("A quiet library contains shelves of books. " * 64)
    jobs = list(enumerate(int(n) for n in args.lengths.split(",")))

    def request(job):
        index, length = job
        prompt = ([base[index + 1]] + base)[:length]
        payload = dict(
            model=model, prompt=prompt, max_tokens=args.generation, temperature=0.0, ignore_eos=True, seed=17
        )
        response = requests.post(args.url + "/v1/completions", json=payload, timeout=600)
        row = dict(
            index=index,
            prompt_length=length,
            prompt_token_ids=prompt,
            request=payload,
            status=response.status_code,
            response=response.json(),
        )
        if response.status_code == 200:
            row["complete"] = response.json().get("usage", {}).get("completion_tokens") == args.generation
        return row

    before = requests.get(args.url + "/metrics", timeout=10).text
    if args.concurrent:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            rows = list(pool.map(request, jobs))
    else:
        rows = [request(job) for job in jobs]
    after = requests.get(args.url + "/metrics", timeout=10).text
    args.output.write_text(
        json.dumps(dict(concurrent=args.concurrent, rows=rows, metrics_before=before, metrics_after=after), indent=2)
    )
    assert all(row.get("complete") for row in rows), rows
    print("LIFECYCLE_PASS", [(r["prompt_length"], r["response"]["usage"]) for r in rows])


if __name__ == "__main__":
    main()
