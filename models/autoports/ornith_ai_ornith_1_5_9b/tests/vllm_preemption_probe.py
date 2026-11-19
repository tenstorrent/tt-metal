# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Force real scheduler preemption using upstream vLLM's development API."""

import argparse
import json
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-url", default="http://localhost:8000")
    args = parser.parse_args()
    results = []
    for name, extra in (
        ("device_greedy", {"temperature": 0}),
        ("device_seeded", {"temperature": 0.8, "top_k": 32, "top_p": 0.9, "seed": 42}),
        ("host_penalties", {"temperature": 0, "logprobs": 1}),
    ):
        for force in (False, True):
            payload = {
                "model": "ornith-ai/Ornith-1.5-9B",
                "prompt": [9707 + i % 13 for i in range(129)],
                "max_tokens": 100,
                "ignore_eos": True,
                "stream": True,
                "return_token_ids": True,
                "presence_penalty": 0.75,
                "frequency_penalty": 0.5,
                "repetition_penalty": 1.1,
                **extra,
            }
            tokens, chunks, reset = [], [], None
            with requests.post(args.server_url + "/v1/completions", json=payload, stream=True, timeout=600) as response:
                response.raise_for_status()
                for line in response.iter_lines(chunk_size=1):
                    if not line.startswith(b"data: ") or line == b"data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    chunks.append(chunk)
                    for choice in chunk.get("choices", []):
                        tokens.extend(choice.get("token_ids") or [])
                    if force and len(tokens) >= 20 and reset is None:
                        paused = requests.post(args.server_url + "/pause?mode=keep&clear_cache=false", timeout=120)
                        paused.raise_for_status()
                        try:
                            reply = requests.post(
                                args.server_url + "/reset_prefix_cache?reset_running_requests=true", timeout=120
                            )
                            reply.raise_for_status()
                            reset = {"at_client_tokens": len(tokens), "response": reply.json()}
                        finally:
                            requests.post(args.server_url + "/resume", timeout=120).raise_for_status()
            results.append(
                {"name": name, "forced": force, "request": payload, "tokens": tokens, "chunks": chunks, "reset": reset}
            )
            args.output.write_text(json.dumps({"status": "running", "cases": results}, indent=2) + "\n")
            print(name, force, len(tokens), reset, flush=True)
    checks = {}
    for baseline, resumed in zip(results[::2], results[1::2]):
        name = baseline["name"]
        checks[name + "_complete"] = len(baseline["tokens"]) == len(resumed["tokens"]) == 100
        checks[name + "_reset"] = bool(resumed["reset"] and resumed["reset"]["response"].get("success"))
        checks[name + "_exact_tokens"] = baseline["tokens"] == resumed["tokens"]
    report = {"status": "pass" if all(checks.values()) else "fail", "checks": checks, "cases": results}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], checks, flush=True)
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
