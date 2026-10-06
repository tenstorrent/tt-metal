# SPDX-License-Identifier: Apache-2.0
"""Compare API top-logprobs across row orderings and canonical-generator controls."""

import argparse
import json
from pathlib import Path

import requests

p = argparse.ArgumentParser()
p.add_argument("--control", required=True)
p.add_argument("--model", required=True)
p.add_argument("--output", required=True)
p.add_argument("--url", default="http://localhost:8000")
a = p.parse_args()
control = json.loads(Path(a.control).read_text())
records = [x for x in control["records"] if x["repeat"] == 0]
runs = []
for order in (list(range(len(records))), list(reversed(range(len(records)))), list(range(len(records)))):
    response = requests.post(
        a.url + "/v1/completions",
        json=dict(
            model=a.model,
            prompt=[records[i]["prompt_ids"] for i in order],
            max_tokens=1,
            temperature=0.0,
            logprobs=20,
            return_tokens_as_token_ids=True,
            return_token_ids=True,
        ),
        timeout=900,
    )
    response.raise_for_status()
    data = response.json()
    runs.append(dict(order=order, response=data))
result = dict(status="recorded-not-validated", control=a.control, runs=runs)
Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
observed = {}
for run in runs:
    for choice in run["response"]["choices"]:
        index = run["order"][choice["index"]]
        ref = records[index]
        top = {int(k.split(":")[-1]): v for k, v in choice["logprobs"]["top_logprobs"][0].items()}
        expected = dict(zip(ref["top20_token_ids"], ref["top20_logprobs"]))
        assert choice["token_ids"][0] == ref["token"], (ref["id"], choice, ref)
        common = top.keys() & expected.keys()
        assert len(common) >= 19, (top, expected)
        error = max(abs(top[i] - expected[i]) for i in common)
        assert error < 1e-5, (ref["id"], error, top, expected)
        if index in observed:
            assert top == observed[index], (ref["id"], top, observed[index])
        observed[index] = top
result["status"] = "pass"
result["cases"] = len(records)
result["orders"] = len(runs)
Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
print("API_LOGIT_DETERMINISM_PASS", len(records), len(runs))
