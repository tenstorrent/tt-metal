# SPDX-License-Identifier: Apache-2.0
"""Conditional served logits and native draws at an observed language-switch prefix."""
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("models/autoports/google_gemma_4_31b_it/readiness_vllm")


def request(body, path="/v1/completions"):
    payload = dict(model="google/gemma-4-31B-it", max_tokens=1, return_token_ids=True)
    payload.update(body)
    req = urllib.request.Request(
        "http://localhost:8000" + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        result = json.load(response)
    return dict(request=payload, choice=result["choices"][0], usage=result["usage"])


def main():
    source = json.loads((ROOT / "qualitative_language_hf_probe.json").read_text())
    tokens = source["prompt_token_ids"] + source["completion_prefix_token_ids"]
    reference = request(dict(prompt=tokens, temperature=0.0, logprobs=20, top_k=32, top_p=0.9))
    waves = []
    for p in (0.9, 0.01):
        with ThreadPoolExecutor(max_workers=32) as pool:
            rows = list(
                pool.map(
                    lambda seed: request(dict(prompt=tokens, temperature=0.7, top_k=32, top_p=p, seed=seed)), range(32)
                )
            )
        waves.append(dict(top_p=p, rows=rows))
    chat_repeats = []
    for seed in (42, 0, None):
        body = dict(
            messages=[{"role": "user", "content": source["prompt"]}],
            max_tokens=256,
            temperature=0.7,
            top_k=32,
            top_p=0.9,
        )
        if seed is not None:
            body["seed"] = seed
        chat_repeats.append(request(body, "/v1/chat/completions"))
    report = dict(
        chat_repeats=chat_repeats,
        scope="Conditional logits and seeded native draws; not a performance benchmark",
        reference=reference,
        waves=waves,
        source="qualitative_language_hf_probe.json",
    )
    (ROOT / "qualitative_language_served_probe.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "reference": reference["choice"],
                "draws": [[r["choice"]["token_ids"] for r in wave["rows"]] for wave in waves],
            }
        )
    )


if __name__ == "__main__":
    main()
