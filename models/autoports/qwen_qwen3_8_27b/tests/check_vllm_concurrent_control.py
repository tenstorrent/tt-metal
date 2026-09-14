# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare diverse native async streams with synchronous greedy logprob controls."""

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
    args = parser.parse_args()
    model = requests.get(args.url + "/v1/models", timeout=10).json()["data"][0]["id"]
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    prompts = [
        "Explain why the Moon has phases. Use a clear example involving sunlight.",
        "Describe how to bake bread from flour, water, yeast and salt, in order.",
        "Explain how a bicycle changes gears and why low gears help uphill.",
        "Tell a story about a botanist finding a blue flower on a snowy mountain.",
    ]

    def request(index, host_control):
        messages = [{"role": "user", "content": prompts[index]}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
        payload = dict(
            model=model, messages=messages, max_tokens=100, temperature=0.0, ignore_eos=True, return_token_ids=True
        )
        if host_control:
            # Logprobs disables overlap and uses the plugin's existing host sampler.
            payload.update(logprobs=True, top_logprobs=1)
        response = requests.post(args.url + "/v1/chat/completions", json=payload, timeout=900)
        response.raise_for_status()
        data = response.json()
        assert data["usage"]["completion_tokens"] == 100, data
        assert data["usage"]["prompt_tokens"] == len(prompt_ids), data["usage"]
        assert len(data["choices"][0]["token_ids"]) == 100, data
        text = data["choices"][0]["message"]["content"]
        return dict(
            prompt_id=index,
            host_control=host_control,
            prompt_token_ids=prompt_ids,
            request=payload,
            response=data,
            text=text,
            token_ids=data["choices"][0]["token_ids"],
            decode_page_boundaries=list(range(((len(prompt_ids) + 31) // 32) * 32, len(prompt_ids) + 99, 32)),
        )

    before = requests.get(args.url + "/metrics", timeout=10).text
    rows = []
    for host_control, order in ((False, [0, 1, 2, 3]), (True, [3, 2, 1, 0]), (False, [2, 0, 3, 1])):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            rows.extend(pool.map(lambda i: request(i, host_control), order))
    baseline = {row["prompt_id"]: row["text"] for row in rows[:4]}
    matches = [row["text"] == baseline[row["prompt_id"]] for row in rows]
    baseline_ids = {row["prompt_id"]: row["token_ids"] for row in rows[:4]}
    token_matches = [row["token_ids"] == baseline_ids[row["prompt_id"]] for row in rows]
    report = dict(
        mode="native async versus logprob-forced synchronous host greedy; reordered concurrent HTTP submission",
        scheduler_row_placement="not forced by HTTP ordering",
        rows=rows,
        exact_matches=matches,
        exact_token_matches=token_matches,
        distinct_native_streams=len(set(baseline.values())),
        metrics_before=before,
        metrics_after=requests.get(args.url + "/metrics", timeout=10).text,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    assert all(matches), "Native/control/reordered streams differ; inspect saved responses"
    assert all(token_matches), "Native/control/reordered token IDs differ; inspect saved responses"
    assert len(set(baseline.values())) == 4, "Control requires distinct request-sensitive streams"
    assert all(len(set(tokenizer.encode(t))) > 10 for t in baseline.values()), "Degenerate control stream"
    print("CONCURRENT_CONTROL_PASS", [len(r["prompt_token_ids"]) for r in rows[:4]], matches)


if __name__ == "__main__":
    main()
