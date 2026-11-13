#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Record long native-chat samples and the shared prompt regression suite."""

import collections
import json
import os
import sys
import time
import urllib.request
from collections.abc import Mapping
from pathlib import Path

MODEL = Path(__file__).resolve().parents[1]
BASE = MODEL / "doc/benchmark"
SKILL = Path(os.environ["TT_MODEL_BRINGUP_ROOT"])
sys.path.insert(0, str(SKILL / "scripts"))
from reply_digest import reply_row, reply_text


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def request(payload):
    req = urllib.request.Request(
        "http://127.0.0.1:8000/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5400) as response:
        return json.load(response)


def main():
    from serving_paths import snapshot_path
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot_path()), revision="842da3794eaa0b77d5f08bae87a17459d91ff475", local_files_only=True
    )

    def prompt_format(payload, body):
        kwargs = payload.get("chat_template_kwargs", {})
        token_ids = tokenizer.apply_chat_template(
            payload["messages"], tokenize=True, add_generation_prompt=True, **kwargs
        )
        if isinstance(token_ids, Mapping):
            token_ids = token_ids["input_ids"]
        return {
            "rendered_prompt": tokenizer.apply_chat_template(
                payload["messages"], tokenize=False, add_generation_prompt=True, **kwargs
            ),
            "prompt_token_ids": token_ids,
            "reference_prompt_tokens": len(token_ids),
            "served_prompt_tokens": body["usage"]["prompt_tokens"],
            "count_matches": len(token_ids) == body["usage"]["prompt_tokens"],
            "basis": "Pinned local tokenizer renders the submitted messages and identical template kwargs; request sends messages once through native chat endpoint.",
        }

    dest = BASE / "anomaly_replies"
    dest.mkdir(exist_ok=True)
    prompts = [
        "Develop a rigorous design for a fault-tolerant library lending system. Explain the data model, concurrency invariants, recovery after partial failures, test cases, and tradeoffs. Work through concrete examples and check the reasoning carefully.",
        "Explain how to compare three competing scientific explanations using limited noisy observations. Construct an original quantitative example, derive the calculations, examine confounding and uncertainty, and discuss what further experiment would distinguish the explanations.",
        "Design an algorithm for planning deliveries when travel times vary and some destinations have time windows. Derive the approach from first principles, work through a substantial example, consider failure cases and prove what guarantees do and do not hold.",
    ]
    prompts.append(
        "Write a self-contained, approximately 8000-word engineering handbook on designing a reliable community observatory's data collection and analysis system. Develop twelve substantial sections covering requirements, mathematical models of measurement error, storage design, concurrent ingestion, calibration, missing observations, anomaly detection, recovery, reproducibility, access control, worked numerical examples, and verification. Include original concrete examples, pseudocode, derivations, failure scenarios and their remedies throughout. Fully develop each section rather than providing an outline or a summary. Finish with a rigorous end-to-end worked case that connects all twelve sections."
    )
    result = {
        "replies": [],
        "anomalies": 0,
        "source": "Native /v1/chat/completions; generic prompts without eval content.",
        "sampling": {"temperature": 1, "top_p": 0.95, "top_k": 32},
        "output_cap": 32768,
        "output_cap_basis": "Main SWE reference output cap32768; GPQA cap126976 is larger. Long check covers tool-eval generation scale, not a GPQA full-cap soak.",
        "thinking": "checkpoint chat-template default (no request override)",
        "started_unix": time.time(),
    }
    for i, prompt in enumerate(prompts):
        payload = {
            "model": "google/gemma-4-31B-it",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32768,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 32,
            "seed": [17, 29, 43, 59][i],
            "stream": False,
        }
        response_path = dest / f"{i}-response.json"
        timing_path = dest / f"{i}-timing.json"
        if response_path.exists():
            assert json.loads((dest / f"{i}-request.json").read_text()) == payload
            body = json.loads(response_path.read_text())
            elapsed = json.loads(timing_path.read_text())["elapsed_seconds"] if timing_path.exists() else None
        else:
            save(dest / f"{i}-request.json", payload)
            start = time.monotonic()
            body = request(payload)
            elapsed = time.monotonic() - start
            save(response_path, body)
            save(timing_path, {"elapsed_seconds": elapsed})
        digest = reply_row(body, 32768)
        text = reply_text(body)
        words = text[-4096:].split()
        grams = collections.Counter(tuple(words[k : k + 3]) for k in range(max(0, len(words) - 2)))
        common = grams.most_common(1)
        row = {
            "prompt": prompt,
            "prompt_format": prompt_format(payload, body),
            "seed": payload["seed"],
            "output_tokens": body["usage"]["completion_tokens"],
            "distinct_tail_chars": digest["tail_distinct"],
            "repeated_ngram": {"ngram": list(common[0][0]), "count": common[0][1]} if common else None,
            "anomaly": digest["flag"],
            "digest": digest,
            "elapsed_seconds": elapsed,
            "timing_note": (
                "Original wall time unavailable after metadata serialization failure; saved response reused."
                if elapsed is None
                else "HTTP wall time"
            ),
            "response_id": body["id"],
            "evidence": f"anomaly_replies/{i}-response.json",
        }
        result["replies"].append(row)
        result["anomalies"] = sum(r["anomaly"] for r in result["replies"])
        save(BASE / "anomaly_check.json", result)
    existing_shared = (
        json.loads((BASE / "qualitative_outputs.json").read_text())
        if (BASE / "qualitative_outputs.json").exists()
        else []
    )
    shared = []
    for i, prompt in enumerate((SKILL / "runtime/readiness_check/vllm_prompts.txt").read_text().splitlines()):
        if not prompt.strip() or prompt.startswith("#"):
            continue
        payload = {
            "model": "google/gemma-4-31B-it",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 128,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": False,
        }
        existing = next((row for row in existing_shared if row["request"] == payload), None)
        body = existing["response"] if existing else request(payload)
        shared.append(
            {
                "id": f"shared_{len(shared)}",
                "prompt": prompt,
                "request": payload,
                "prompt_format": prompt_format(payload, body),
                "response": body,
                "completion": reply_text(body),
                "digest": reply_row(body, 128),
            }
        )
        save(BASE / "qualitative_outputs.json", shared)
    save(
        BASE / "qualitative_prompt_format.json",
        {
            "hf_model": "google/gemma-4-31B-it",
            "revision": "842da3794eaa0b77d5f08bae87a17459d91ff475",
            "chat_template_present": True,
            "prompt_mode": "chat",
            "endpoint": "/v1/chat/completions",
            "shared_suite": str(SKILL / "runtime/readiness_check/vllm_prompts.txt"),
            "shared_control": "doc/datatype_sweep/selected_default/qualitative_hf.json; same revision/native default nonthinking/greedy128. Long samples are anomaly detection only, not an accuracy verdict.",
        },
    )
    print(json.dumps({"anomalies": result["anomalies"], "shared_prompts": len(shared)}))


if __name__ == "__main__":
    main()
