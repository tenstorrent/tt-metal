# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import json
import statistics
import time
import urllib.request
from pathlib import Path

out = Path("models/autoports/qwen_qwen3_6_27b/doc/prefill_device_analysis/artifacts/serving_ttft_controls.json")
result = {
    "model": "Qwen/Qwen3.8-27B",
    "endpoint": "/v1/completions",
    "prompt_token_ids": [1000] * 128,
    "scope": "Synthetic 128-token prompt, no prefix caching. First nonempty streamed text; one discarded warmup then three samples per output length.",
    "cases": [],
}
for output_length in [1, 2, 8]:
    for repetition in range(4):
        body = {
            "model": result["model"],
            "prompt": result["prompt_token_ids"],
            "max_tokens": output_length,
            "temperature": 0,
            "top_k": 1,
            "ignore_eos": True,
            "stream": True,
        }
        req = urllib.request.Request(
            "http://localhost:8023/v1/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        first = None
        text = ""
        chunks = []
        with urllib.request.urlopen(req, timeout=180) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    break
                chunk = json.loads(data)
                chunks.append(chunk)
                token = chunk.get("choices", [{}])[0].get("text", "")
                if token:
                    if first is None:
                        first = time.perf_counter()
                    text += token
        case = {
            "max_tokens": output_length,
            "repetition": repetition,
            "warmup": repetition == 0,
            "ttft_ms": None if first is None else (first - start) * 1000,
            "elapsed_ms": (time.perf_counter() - start) * 1000,
            "text": text,
            "chunks": chunks,
        }
        result["cases"].append(case)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(output_length, repetition, case["ttft_ms"], case["elapsed_ms"], flush=True)
result["warm_medians_ms"] = {
    str(n): statistics.median(c["ttft_ms"] for c in result["cases"] if c["max_tokens"] == n and not c["warmup"])
    for n in [1, 2, 8]
}
out.write_text(json.dumps(result, indent=2) + "\n")
