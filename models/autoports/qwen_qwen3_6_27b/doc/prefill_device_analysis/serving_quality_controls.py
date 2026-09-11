# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import json
import time
import urllib.request
from pathlib import Path

out = Path("models/autoports/qwen_qwen3_6_27b/doc/prefill_device_analysis/artifacts/serving_quality_controls.json")
prompt = "Explain the difference between supervised and unsupervised learning in simple terms."
result = {
    "model": "Qwen/Qwen3.8-27B",
    "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "endpoint": "/v1/chat/completions",
    "prompt": prompt,
    "cases": [],
}
for mode in ["device_unseeded", "device_seed42", "host_logprobs_seed42"]:
    for repetition in range(2):
        body = {
            "model": result["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 160,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 20,
        }
        if mode != "device_unseeded":
            body["seed"] = 42
        if mode.startswith("host"):
            body.update(logprobs=True, top_logprobs=1)
        start = time.perf_counter()
        req = urllib.request.Request(
            "http://localhost:8023/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=240) as response:
                response = json.load(response)
            case = {
                "mode": mode,
                "repetition": repetition,
                "request": body,
                "elapsed_s": time.perf_counter() - start,
                "response": response,
            }
        except Exception as exc:
            case = {
                "mode": mode,
                "repetition": repetition,
                "request": body,
                "elapsed_s": time.perf_counter() - start,
                "error": repr(exc),
            }
        result["cases"].append(case)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(mode, repetition, case["elapsed_s"], case.get("error", "ok"), flush=True)
