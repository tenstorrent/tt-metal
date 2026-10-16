# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Live single-slot routing, penalties, request isolation and unsupported-input checks."""

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

import httpx


async def validate(url, output):
    rows = []
    checks = []
    output.mkdir(parents=True, exist_ok=False)
    completed = False
    try:
        async with httpx.AsyncClient(base_url=url, timeout=900) as client:

            async def run(label, **fields):
                payload = dict(
                    model="Qwen/Qwen3.8-Flash-Next",
                    prompt="The red fox fox fox walks. Continue the story: ",
                    max_tokens=16,
                    temperature=0,
                    seed=42,
                    ignore_eos=True,
                    return_token_ids=True,
                    return_tokens_as_token_ids=True,
                )
                payload.update(fields)
                start = time.monotonic()
                r = await client.post("/v1/completions", json=payload)
                row = dict(
                    label=label,
                    payload=payload,
                    status=r.status_code,
                    elapsed_s=time.monotonic() - start,
                    response=r.json(),
                )
                rows.append(row)
                with (output / "requests.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                return row

            def ids(row):
                assert row["status"] == 200, (row["label"], row["response"])
                r = row["response"]
                tokens = r["choices"][0]["token_ids"]
                assert len(tokens) == r["usage"]["completion_tokens"] and len(tokens) > 0
                return tokens

            def check(label, condition):
                checks.append(dict(label=label, passed=bool(condition)))
                assert condition, label

            first = await run("greedy-model-A0")
            ids(first)
            other = await run(
                "intervening-B",
                prompt="Describe three imaginary planets: ",
                temperature=0.8,
                top_k=30,
                top_p=0.9,
                seed=7,
            )
            ids(other)
            last = await run("greedy-model-A1")
            check("sequential A/B/A isolation", ids(first) == ids(last))
            host = await run("greedy-host-logprobs0", logprobs=0)
            check("greedy model/host route equality", ids(first) == ids(host))
            for n in (0, 1, 5):
                r = host if n == 0 else await run(f"logprobs{n}", logprobs=n)
                ids(r)
                lp = r["response"]["choices"][0]["logprobs"]
                check(
                    f"logprobs{n} finite",
                    len(lp["token_logprobs"]) == len(ids(r))
                    and all(v is not None and math.isfinite(v) for v in lp["token_logprobs"]),
                )
                check(f"logprobs{n} alternatives", all(n <= len(v) <= n + 1 for v in lp["top_logprobs"]))
            policies = [
                dict(presence_penalty=1.5),
                dict(frequency_penalty=1.5),
                dict(repetition_penalty=1.5),
                dict(presence_penalty=1.1, frequency_penalty=0.5, repetition_penalty=1.3),
                dict(presence_penalty=-0.7, frequency_penalty=-0.5, repetition_penalty=0.8),
            ]
            for index, policy in enumerate(policies):
                x = await run(f"penalty{index}-model", **policy)
                y = await run(f"penalty{index}-host", logprobs=0, **policy)
                check(f"penalty{index} model/host equality", ids(x) == ids(y))
            params = [
                dict(temperature=0),
                dict(temperature=0.7, top_k=20, top_p=0.85, seed=42),
                dict(temperature=0.9, seed=91, frequency_penalty=0.7),
                dict(temperature=0.7, seed=7, logprobs=0, min_p=0.2),
            ]
            sequential = [await run(f"queued-reference{i}", **policy) for i, policy in enumerate(params)]
            queued = await asyncio.gather(
                *(run(f"queued-mixed{i}", **policy) for i, policy in reversed(list(enumerate(params))))
            )
            by_label = {r["label"]: r for r in queued}
            for i in range(len(params)):
                check(f"queued seeded isolation{i}", ids(sequential[i]) == ids(by_label[f"queued-mixed{i}"]))
            for label, policy in [
                ("min_p", dict(min_p=0.2)),
                ("min_tokens", dict(min_tokens=8, max_tokens=8, ignore_eos=False)),
                ("allowed_tokens", dict(allowed_token_ids=[16, 17, 18])),
                ("logit_bias", dict(logit_bias={"16": 100})),
            ]:
                r = await run(label, **policy)
                tokens = ids(r)
                if label == "allowed_tokens":
                    check(label, set(tokens) <= set(policy["allowed_token_ids"]))
                if label == "min_tokens":
                    check(label, len(tokens) == 8)
                if label == "logit_bias":
                    check(label, set(tokens) == {16})
            structured = await run(
                "structured-host-regex", structured_outputs={"regex": "[0-9]{3}"}, ignore_eos=False, max_tokens=16
            )
            ids(structured)
            check(
                "structured output uses full-logit grammar route",
                len(structured["response"]["choices"][0]["text"]) == 3
                and structured["response"]["choices"][0]["text"].isdigit(),
            )
            for label, fields in [
                ("unsupported_prompt_logprobs", dict(prompt_logprobs=1)),
                ("invalid_penalty", dict(repetition_penalty=0)),
                ("invalid_top_p", dict(top_p=1.5)),
            ]:
                r = await run(label, **fields)
                check(label, 400 <= r["status"] < 500)
            final = await run("post-rejection-A")
            check("rejections leave slot serviceable", ids(final) == ids(first))
        completed = True
    finally:
        result = dict(
            requests=len(rows), checks=checks, passed=completed and bool(checks) and all(c["passed"] for c in checks)
        )
        (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = asyncio.run(validate(a.url, a.output))
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
