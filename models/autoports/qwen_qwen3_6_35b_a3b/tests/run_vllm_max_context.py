# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""One served maximum-context capacity acceptance request, without a sweep."""

import argparse
import json
import time
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    model_root = Path(__file__).resolve().parents[1]
    contract = json.loads((model_root / "doc/context_contract.json").read_text())
    context = int(contract["supported_context"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError("Acceptance artifact already exists; inspect it before considering another maximum run")
    result = dict(
        status="RUNNING",
        context_contract=contract,
        max_model_len=context,
        prompt_tokens=context - 1,
        output_tokens=1,
        started=time.time(),
        scope="capacity only; repeated valid token IDs are not qualitative evidence",
    )

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    started = time.monotonic()
    try:
        response = requests.post(
            args.url + "/v1/completions",
            timeout=10800,
            json=dict(
                model="Qwen/Qwen3.6-35B-A3B",
                prompt=[9707] * (context - 1),
                max_tokens=1,
                temperature=0.0,
                top_k=1,
                ignore_eos=True,
                return_token_ids=True,
            ),
        )
        raw_path = args.output.with_suffix(".response.raw.log")
        raw_path.write_text(response.text + "\n")
        body = response.json()
        result.update(http_status=response.status_code, raw_response_path=str(raw_path))
        response.raise_for_status()
        usage = body["usage"]
        result["usage"] = usage
        result["choices"] = body["choices"]
        result["returned_prompt_token_count"] = len(body.get("prompt_token_ids") or [])
        if usage["prompt_tokens"] != context - 1 or usage["completion_tokens"] != 1:
            raise AssertionError("The full requested context was not consumed and decoded")
        result["status"] = "PASS"
    except BaseException as error:
        result.update(status="FAIL", error=repr(error))
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        save()
    print(json.dumps({key: result[key] for key in ("status", "prompt_tokens", "output_tokens", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
