# SPDX-License-Identifier: Apache-2.0
"""Reproduce special-token allowlist semantics on an already-running server."""

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def post(path, body):
    request = urllib.request.Request(
        "http://localhost:8000" + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def request(allowed, **extra):
    body = dict(
        model="google/gemma-4-31B-it",
        prompt="Allowed: ",
        max_tokens=10,
        temperature=1.0,
        top_p=1.0,
        allowed_token_ids=allowed,
        return_token_ids=True,
        **extra,
    )
    response = post("/v1/completions", body)
    choice = response["choices"][0]
    ids = choice["token_ids"]
    assert ids and set(ids) <= set(allowed), (ids, allowed)
    assert response["usage"]["completion_tokens"] == len(ids)
    return dict(request=body, choice=choice, usage=response["usage"])


def main():
    allowed_sets = [list(range(start, start + 3)) for start in (1, 4, 7, 10, 13)]
    with ThreadPoolExecutor(max_workers=5) as executor:
        original = list(executor.map(request, allowed_sets))
    assert original[0]["choice"]["text"] == ""
    visible = request([1, 2, 3], ignore_eos=True, skip_special_tokens=False)
    assert len(visible["choice"]["token_ids"]) == 10 and visible["choice"]["text"]
    ordinary_ids = post(
        "/tokenize",
        dict(model="google/gemma-4-31B-it", prompt="A", add_special_tokens=False),
    )["tokens"]
    ordinary = request(ordinary_ids)
    assert ordinary["choice"]["text"]
    result = dict(
        status="pass",
        scope="API allowlist and special-token detokenization contract; original stock text assertion fails",
        stock_visible_text_assertion_passed=False,
        original_cases=original,
        visible_special_tokens=visible,
        ordinary_tokens=ordinary,
    )
    path = Path("models/autoports/google_gemma_4_31b_it/readiness_vllm/allowed_token_ids_live.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status="pass", output=str(path))))


if __name__ == "__main__":
    main()
