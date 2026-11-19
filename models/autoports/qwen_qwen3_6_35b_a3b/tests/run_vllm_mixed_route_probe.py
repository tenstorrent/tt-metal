# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""One instrumentable pair of canonical mixed-parameter HTTP batches.

This preserves canonical sampling parameters and text-equality assertions. The
only extra request fields are request_id and return_token_ids (no logprobs).
It neither chooses a sampler nor infers routes from different output strings.
"""

import argparse
import asyncio
import hashlib
import json
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx


def now():
    return {"utc": datetime.now(timezone.utc).isoformat(), "monotonic_ns": time.monotonic_ns()}


def canonical_configs():
    # Keep in canonical test_mixed_params_batch order. Omitted top_k inherits
    # the server's generation config, exactly as tests.tt.utils.send_request.
    return [
        dict(prompt="Count: ", max_tokens=5, temperature=0),
        dict(prompt="Random: ", max_tokens=5, temperature=2.0, top_k=32, seed=42),
        dict(prompt="test test. Say: ", max_tokens=5, temperature=0.5, repetition_penalty=3.0, seed=42),
        dict(prompt="List: ", max_tokens=10, temperature=0.5, presence_penalty=2.0, seed=42),
        dict(prompt="Word: ", max_tokens=3, temperature=2.0, top_k=1, seed=99),
        dict(prompt="Letter: ", max_tokens=3, temperature=0.01, seed=42),
        dict(prompt="Number: ", max_tokens=3, temperature=2.0, top_k=5, seed=42),
        dict(prompt="a a a. Next: ", max_tokens=5, temperature=0.5, frequency_penalty=2.0, seed=42),
        dict(
            prompt="All: ",
            max_tokens=8,
            temperature=0.5,
            repetition_penalty=1.5,
            presence_penalty=1.0,
            frequency_penalty=1.0,
            seed=6,
        ),
        dict(prompt="TopP: ", max_tokens=8, temperature=1.0, top_p=0.5, seed=7),
    ]


def token_digest(tokens):
    if not isinstance(tokens, list):
        return None
    # Adapter diagnostic events can use this encoding on existing CPU inputs.
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def save(path, result):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)


async def collect(args, result):
    configs = result["canonical_configs"]
    for batch_index, order in enumerate(result["orders"]):
        batch = dict(batch_index=batch_index, order=order, started=now(), requests=[])
        result["batches"].append(batch)
        save(args.output, result)
        # Canonical run_concurrent_batch creates a fresh async client per batch.
        async with httpx.AsyncClient(timeout=args.request_timeout, trust_env=False) as client:

            async def send(case_index, submit_index):
                request_id = f"mixed-route-{result['run_id']}-b{batch_index}-c{case_index}"
                payload = dict(
                    model=args.model,
                    top_p=1.0,
                    seed=None,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    logprobs=None,
                )
                payload.update(configs[case_index])
                payload.update(request_id=request_id, return_token_ids=True)
                record = dict(
                    case_index=case_index,
                    submit_index=submit_index,
                    client_request_id=request_id,
                    payload=payload,
                    submitted=now(),
                )
                batch["requests"].append(record)
                save(args.output, result)
                try:
                    response = await client.post(args.url.rstrip("/") + "/v1/completions", json=payload)
                    record.update(http_status=response.status_code, response_headers=dict(response.headers))
                    try:
                        body = response.json()
                    except ValueError:
                        record["raw_response"] = response.text
                        raise
                    record["response"] = body
                    record["response_request_id"] = body.get("id")
                    response.raise_for_status()
                    choice = body["choices"][0]
                    record["text"] = choice["text"]
                    record["token_ids"] = choice.get("token_ids")
                    record["prompt_token_ids"] = choice.get("prompt_token_ids") or body.get("prompt_token_ids")
                    record["prompt_token_digest"] = token_digest(record["prompt_token_ids"])
                    if not isinstance(record["token_ids"], list):
                        raise AssertionError(
                            "Server did not return output token IDs; do not enable logprobs as fallback"
                        )
                    if not isinstance(record["prompt_token_ids"], list):
                        raise AssertionError("Missing prompt token IDs needed for adapter slot attribution")
                    if not record["response_request_id"]:
                        raise AssertionError("Missing actual response request ID")
                    record["status"] = "PASS"
                except Exception as error:
                    record.update(status="FAIL", error=repr(error))
                finally:
                    record["completed"] = now()
                    save(args.output, result)

            await asyncio.gather(*(send(case_index, index) for index, case_index in enumerate(order)))
        batch["finished"] = now()
        if any(request["status"] != "PASS" for request in batch["requests"]):
            raise AssertionError("Request or evidence collection failed; second batch is not retried")

    first, second = [{request["case_index"]: request for request in batch["requests"]} for batch in result["batches"]]
    comparisons = []
    # Canonical assertions are evaluated in the second batch's shuffled order.
    for index in result["orders"][1]:
        left, right = first[index], second[index]
        deterministic = configs[index]["temperature"] == 0 or configs[index].get("seed") is not None
        tokens_left, tokens_right = left["token_ids"], right["token_ids"]
        mismatch = next(
            (offset for offset, pair in enumerate(zip(tokens_left, tokens_right)) if pair[0] != pair[1]), None
        )
        if mismatch is None and len(tokens_left) != len(tokens_right):
            mismatch = min(len(tokens_left), len(tokens_right))
        comparisons.append(
            dict(
                case_index=index,
                prompt=configs[index]["prompt"],
                canonical_assertion_applies=deterministic,
                text_equal=left["text"] == right["text"],
                token_ids_equal=tokens_left == tokens_right,
                first_divergent_token_index=mismatch,
                first_response_request_id=left["response_request_id"],
                second_response_request_id=right["response_request_id"],
            )
        )
    result["comparisons"] = comparisons
    failures = [item for item in comparisons if item["canonical_assertion_applies"] and not item["text_equal"]]
    result["canonical_text_assertion_failures"] = failures
    if failures:
        raise AssertionError(
            f"Canonical seeded/greedy text equality failed for cases {[item['case_index'] for item in failures]}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--request-timeout", type=float, default=180)
    parser.add_argument(
        "--shuffle-seed", type=int, default=7, help="Client ordering only; request seeds remain canonical"
    )
    parser.add_argument("--second-order", help="Optional exact comma-separated permutation of indices 0 through 9")
    parser.add_argument("--plan-only", action="store_true", help="Write the exact two-batch plan without HTTP requests")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; preserve prior evidence and choose another path")
    if args.request_timeout <= 0:
        parser.error("Request timeout must be positive")
    order = list(range(10))
    random.Random(args.shuffle_seed).shuffle(order)
    if args.second_order is not None:
        order = [int(value) for value in args.second_order.split(",")]
        if sorted(order) != list(range(10)):
            parser.error("Second order must be an exact permutation of indices 0 through 9")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plugin_test = Path(__file__).resolve().parents[4].parent / "vllm-tt-plugin/tests/tt/test_request_isolation.py"
    result = dict(
        status="PLAN_ONLY" if args.plan_only else "RUNNING",
        run_id=uuid.uuid4().hex[:12],
        command=[sys.executable, *sys.argv],
        started=now(),
        url=args.url,
        canonical_node="tests/tt/test_request_isolation.py::TestBatchIsolation::test_mixed_params_batch",
        canonical_source=str(plugin_test),
        canonical_source_sha256=hashlib.sha256(plugin_test.read_bytes()).hexdigest() if plugin_test.exists() else None,
        canonical_configs=canonical_configs(),
        orders=[list(range(10)), order],
        shuffle_seed=args.shuffle_seed,
        expected_server_max_num_seqs=32,
        request_timeout_seconds=args.request_timeout,
        sampling_policy="Unmodified server auto policy; no sampler selection, retries, logprobs, or parameter clamping",
        evidence_limit="Submission order is not engine admission order. Routes require correlated adapter events.",
        batches=[],
    )
    save(args.output, result)
    if args.plan_only:
        print(json.dumps({"status": result["status"], "orders": result["orders"], "output": str(args.output)}))
        return
    try:
        asyncio.run(collect(args, result))
        result["status"] = "PASS"
    except BaseException as error:
        result.update(status="FAIL", error=repr(error))
        raise
    finally:
        result["finished"] = now()
        save(args.output, result)
        print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
