# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One served maximum-context capacity request; not a quality benchmark."""

import argparse
import json
import time
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model_dir = Path(__file__).resolve().parents[1]
    contract = json.loads((model_dir / "doc/context_contract.json").read_text())
    length = contract["supported_context"]
    payload = {
        "model": "ornith-ai/Ornith-1.5-9B",
        "prompt": [9707 + i % 13 for i in range(length - 1)],
        "max_tokens": 1,
        "ignore_eos": True,
        "temperature": 0,
        "return_token_ids": True,
    }
    started = time.perf_counter()
    response = requests.post(args.server_url + "/v1/completions", json=payload, timeout=7200)
    result = response.json()
    usage = result.get("usage", {})
    checks = {
        "http_ok": response.status_code == 200,
        "exact_prompt_length": usage.get("prompt_tokens") == length - 1,
        "one_output": usage.get("completion_tokens") == 1,
        "total_matches_context": usage.get("total_tokens") == length,
    }
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "scope": "capacity acceptance only; synthetic continuation is not a quality prompt",
        "context_contract": str(model_dir / "doc/context_contract.json"),
        "request": {**payload, "prompt": {"length": length - 1, "formula": "9707 + i % 13"}},
        "response": result,
        "seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"], checks, report["seconds"], flush=True)
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
