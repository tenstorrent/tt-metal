# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One guarded advertised-context serving acceptance request; CPU HTTP only.

Run under timeout21600. The request read timeout is21000 seconds. The persistent
attempt marker is reserved before any HTTP call and survives failure/interruption;
this helper never retries or deletes evidence. A timeout does not establish a
physical capacity limit. Raw continuation stress has no text-quality claim.
"""

import argparse
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import requests


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=os.environ.get(
            "QWEN_CHECKPOINT",
            "/home/ttuser/qwen38-flash/hf/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/812d96e10a0dd82b69d749e68693698cf6c36daa",
        ),
    )
    args = parser.parse_args()
    contract_path = args.model_dir / "doc/context_contract.json"
    contract = json.loads(contract_path.read_text())
    context = contract["supported_context"]
    if type(context) is not int or context != 262144:
        parser.error("This acceptance workload requires supported_context=262144; inspect a changed contract")
    config_path = args.checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    vocabulary = config.get("text_config", config)["vocab_size"]
    if type(vocabulary) is not int or vocabulary <= 785:
        parser.error("Checkpoint vocab_size does not admit the stress prompt token785")
    attempt_path = args.model_dir / "readiness_vllm/max_context_acceptance.attempt.json"
    if args.output.resolve() == attempt_path.resolve():
        parser.error("Output and attempt marker must be distinct paths")
    if args.output.exists() or attempt_path.exists():
        parser.error(f"Preserve existing evidence: output={args.output}, attempt={attempt_path}; no request sent")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt = dict(
        status="reserved_before_http",
        created_utc=datetime.now(timezone.utc).isoformat(),
        pid=os.getpid(),
        output=str(args.output.resolve()),
        server_url=args.server_url,
        supported_context=context,
        prompt_tokens=context - 8,
        output_tokens=8,
        automatic_retry=False,
    )
    # Exclusive creation prevents two helpers from admitting the costly request,
    # even if they choose different output filenames under the same model.
    with attempt_path.open("x") as handle:
        json.dump(attempt, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    report = dict(
        passed=False,
        status="preflight",
        attempt_marker=str(attempt_path),
        context_contract=str(contract_path),
        context_contract_field="supported_context",
        model_id=contract["model_id"],
        revision=contract["hf_revision"],
        checkpoint_config=str(config_path),
        vocab_size=vocabulary,
        supported_context=context,
        prompt_tokens=context - 8,
        max_tokens=8,
        workload="262136 input tokens plus8 output tokens; one request; continuation stress",
        prompt_recipe=dict(token_id=785, repetitions=context - 8),
        non_aligned_prompt=(context - 8) % 32 != 0,
        quality_claim=False,
        timeout_seconds=dict(caller_required=21600, connect=30, read=21000),
        server_url=args.server_url,
    )
    started = time.monotonic()
    output_reserved = False
    try:
        # Reserve the result too; do not replace a file created concurrently.
        with args.output.open("x") as handle:
            output_reserved = True
            json.dump(report, handle, indent=2)
            handle.write("\n")
        url = args.server_url.rstrip("/")
        metadata_response = requests.get(url + "/v1/models", timeout=30)
        metadata_response.raise_for_status()
        metadata = metadata_response.json()
        report["models_response"] = metadata
        models = metadata["data"]
        if len(models) != 1:
            raise AssertionError("Expected exactly one served model for this pinned acceptance workload")
        model = models[0]
        if model.get("max_model_len") != context:
            raise AssertionError(f"Served max_model_len {model.get('max_model_len')} != contract {context}")
        body = dict(
            model=model["id"],
            prompt=[785] * (context - 8),
            max_tokens=8,
            temperature=0,
            top_p=1.0,
            top_k=1,
            ignore_eos=True,
            return_token_ids=True,
        )
        report["request_settings"] = {key: value for key, value in body.items() if key != "prompt"}
        report["status"] = "request_pending"
        attempt["status"] = "post_started"
        attempt["post_started_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(attempt_path, attempt)
        write_json(args.output, report)
        response = requests.post(url + "/v1/completions", json=body, timeout=(30, 21000))
        report["http_status"] = response.status_code
        try:
            payload = response.json()
        except ValueError:
            payload = {"raw_body": response.text}
        report["response"] = payload  # Preserve the full successful or failed response.
        write_json(args.output, report)
        response.raise_for_status()
        if "error" in payload:
            raise AssertionError(f"Server returned error payload: {payload['error']}")
        assert payload["usage"]["prompt_tokens"] == context - 8
        assert payload["usage"]["completion_tokens"] == 8
        assert payload["usage"]["total_tokens"] == context
        assert len(payload["choices"]) == 1
        choice = payload["choices"][0]
        assert choice["index"] == 0 and choice["finish_reason"] == "length"
        tokens = choice["token_ids"]
        assert isinstance(tokens, list) and len(tokens) == 8, "Missing eight exact output token IDs"
        assert all(type(token) is int and 0 <= token < vocabulary for token in tokens), "OOV/invalid output token"
        report["out_of_vocabulary"] = 0
        report["passed"] = True
        report["status"] = "completed"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        if output_reserved:
            write_json(args.output, report)
        attempt.update(status=report["status"], passed=report["passed"], seconds=report["seconds"])
        write_json(attempt_path, attempt)
    print(json.dumps(dict(passed=True, output=str(args.output), seconds=report["seconds"])))


if __name__ == "__main__":
    main()
