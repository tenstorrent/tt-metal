"""Fallback-only offline scoring and received-response proof; never infer.

The successful benchmark path remains unchanged. The enclosing fallback runs
this helper on the original invocation clock, then performs the context/evidence
checks and final report. Missing responses never become fabricated samples.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
REPO = MODEL_DIR.parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def rows(path):
    if not Path(path).exists():
        return []
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def preserve_inputs(run, config):
    from benchmark_stage.evaluate import evaluate_groups
    from lm_eval.models.openai_completions import LocalChatCompletion

    manifest = read(run / "manifest.json")

    class Saved(Exception):
        pass

    def capture(self, requests, **kwargs):
        inputs = []
        for request in requests:
            inputs.append(
                {
                    "task": request.task_name,
                    "doc_id": manifest["tasks"][request.task_name]["indices"][request.doc_id],
                    "doc": request.doc,
                    "doc_sha256": digest(request.doc),
                    "arguments": request.args,
                    "messages": self.create_message([request.args[0]]),
                    "request_sha256": digest([request.args[0], request.args[1]]),
                }
            )
        with (run / "benchmark-inputs.jsonl").open("w") as stream:
            for row in inputs:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        write(
            run / "input-reconstruction.json",
            {
                "requests": len(inputs),
                "inference_calls": 0,
                "method": "Pinned upstream request construction intercepted before HTTP; full frozen tasks and examples preserved",
            },
        )
        raise Saved

    original = LocalChatCompletion.generate_until
    LocalChatCompletion.generate_until = capture
    try:
        try:
            evaluate_groups(
                model=config["model"],
                base_url=config["base_url"],
                manifest_path=run / "manifest.json",
                groups=config["tasks"],
                output=run / "partial-input-reconstruction",
                generation=config["generation"][config["tasks"][0]],
                shared=True,
            )
        except Saved:
            return
        raise RuntimeError("Upstream reconstruction did not reach the no-HTTP interception")
    finally:
        LocalChatCompletion.generate_until = original


def joined_received(run, config, manifest, inputs):
    """Validate full input coverage, then join only actual completed responses."""
    by_hash = {row["request_sha256"]: row for row in inputs}
    expected = sum(manifest["groups"][task]["sample_count"] for task in config["tasks"])
    require(len(inputs) == len(by_hash) == expected, "Frozen input coverage/uniqueness mismatch")
    joined = {}
    for name in config["tasks"]:
        frozen = manifest["tasks"][name]
        task_inputs = [row for row in inputs if row["task"] == name]
        require(sorted(row["doc_id"] for row in task_inputs) == frozen["indices"], f"{name}: frozen input IDs differ")
        hashes = dict(zip(frozen["indices"], frozen["document_sha256"]))
        for row in task_inputs:
            require(
                digest(row["doc"]) == row["doc_sha256"] == hashes[row["doc_id"]], f"{name}: input document hash differs"
            )
            require(
                digest([row["arguments"][0], row["arguments"][1]]) == row["request_sha256"],
                f"{name}: input request hash differs",
            )
        raw = rows(run / name / "responses.jsonl")
        responses = {response["id"]: response for response in raw}
        links = rows(run / name / "request_links.jsonl")
        require(len(raw) == len(responses) == len(links), f"{name}: raw response/link coverage differs")
        require(len({link["response_id"] for link in links}) == len(links), f"{name}: duplicate linked response")
        require(len({link["request_sha256"] for link in links}) == len(links), f"{name}: duplicate linked request")
        require(set(responses) == {link["response_id"] for link in links}, f"{name}: unmatched raw response")
        joined[name] = []
        for link in links:
            row = by_hash[link["request_sha256"]]
            require(
                row["task"] == link["task"] == name and row["doc_id"] == link["doc_id"],
                f"{name}: response document identity differs",
            )
            joined[name].append((row, responses[link["response_id"]]))
    return joined


def score_received(run, config, manifest, inputs):
    import importlib.metadata

    from benchmark_stage.gpqa import task_spec
    from benchmark_stage.responses import scoring_response
    from benchmark_stage.subsets import flatten
    from lm_eval.api.instance import Instance
    from lm_eval.evaluator_utils import _compute_task_aggregations
    from lm_eval.tasks import get_task_dict

    require(importlib.metadata.version("lm_eval") == manifest["harness_version"], "Harness version differs")
    require(
        digest({key: value for key, value in manifest.items() if key != "manifest_sha256"})
        == manifest["manifest_sha256"],
        "Manifest digest differs",
    )
    joined = joined_received(run, config, manifest, inputs)
    scored, settings, completed = {}, {}, {}
    for name in config["tasks"]:
        task = flatten(get_task_dict([task_spec(name)]))[name]
        frozen = manifest["tasks"][name]
        require(digest(list(task.eval_docs)) == frozen["population_sha256"], f"{name}: upstream dataset changed")
        require((task.config.num_fewshot or 0) == frozen["num_fewshot"], f"{name}: few-shot count changed")
        if frozen["num_fewshot"]:
            require(digest(list(task.fewshot_docs())) == frozen["fewshot_sha256"], f"{name}: few-shot examples changed")
        instances = []
        for row, response in joined[name]:
            normalized, empty_at_limit = scoring_response(response)
            answer = normalized["choices"][0]["message"]["content"]
            instances.append(
                Instance(
                    "generate_until",
                    row["doc"],
                    tuple(row["arguments"]),
                    0,
                    metadata=(name, row["doc_id"], 1),
                    resps=[answer],
                )
            )
            scored[(name, row["doc_id"])] = {
                "status": "received_and_scored",
                "response_id": response["id"],
                "finish_reason": response["choices"][0]["finish_reason"],
                "stop_reason": response["choices"][0].get("stop_reason"),
                "empty_final_at_length": empty_at_limit,
                "usage": response.get("usage"),
            }
        raw_metrics = defaultdict(list)
        if instances:
            task._instances = instances
            task.apply_filters()
            for instance in instances:
                scores = {}
                for filter_name, value in instance.filtered_resps.items():
                    metrics = task.process_results(instance.doc, [value])
                    scores[filter_name] = metrics
                    for metric, score in metrics.items():
                        raw_metrics[(metric, filter_name)].append(score)
                scored[(name, instance.doc_id)].update(
                    filtered_responses=instance.filtered_resps, upstream_scores=scores
                )
        settings[name] = {
            "version": task.VERSION,
            "num_fewshot": task.config.num_fewshot or 0,
            "filter_list": task.config.filter_list,
            "scored_questions": len(instances),
            "expected_questions": len(frozen["indices"]),
            "aggregate_available": False,
        }
        if sorted(instance.doc_id for instance in instances) == frozen["indices"]:
            # This is the exact pinned evaluator aggregation and stderr path.
            aggregate, count = _compute_task_aggregations(task, raw_metrics, bootstrap_iters=100000)
            require(count == len(frozen["indices"]), f"{name}: upstream aggregate count differs")
            result_path = run / name / "offline-results.json"
            write(
                result_path,
                {
                    "results": {name: aggregate},
                    "versions": {name: task.VERSION},
                    "n-shot": {name: frozen["num_fewshot"]},
                    "n-samples": {name: {"original": frozen["population"], "effective": count}},
                    "configs": {name: task.dump_config()},
                    "benchmark_stage": {
                        "model": config["model"],
                        "group": name,
                        "subset_sha256": manifest["manifest_sha256"],
                        "expected_samples": count,
                        "responses": count,
                        "concurrency": 32,
                        "generation_overrides": config["generation"][name],
                        "inference_calls": 0,
                        "method": "Offline upstream apply_filters/process_results/_compute_task_aggregations over every frozen response; original inference retained",
                    },
                },
            )
            completed[name] = {"results_path": str(result_path.relative_to(run)), "samples": count, "score": aggregate}
            settings[name]["aggregate_available"] = True
    with (run / "upstream-question-scores.jsonl").open("w") as stream:
        for row in inputs:
            record = {key: row[key] for key in ("task", "doc_id", "doc_sha256", "request_sha256")}
            record.update(
                scored.get((row["task"], row["doc_id"]), {"status": "no_completed_response", "upstream_scores": None})
            )
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    evidence = {
        "harness_version": manifest["harness_version"],
        "manifest_sha256": manifest["manifest_sha256"],
        "intended_questions": len(inputs),
        "received_and_scored": len(scored),
        "settings": settings,
        "completed_tasks": completed,
        "aggregation": "Unmodified upstream aggregation only when every frozen document has one received response; incomplete tasks have per-question results only",
        "inference_calls": 0,
    }
    write(run / "upstream-question-scoring.json", evidence)
    return evidence


def verify_received(run, config, manifest, inputs):
    from transformers import AutoTokenizer

    from models.demos.k2_horizon_7b_qb2.tests.verify_benchmark_chat_tokens import (
        SNAPSHOT,
        K2HorizonBenchmarkReasoningParser,
        verify_response,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(SNAPSHOT), trust_remote_code=True, local_files_only=True)
    parser = K2HorizonBenchmarkReasoningParser(tokenizer)
    eos = read(SNAPSHOT / "generation_config.json")["eos_token_id"]
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    verified, coverage = [], {}
    for name, received in joined_received(run, config, manifest, inputs).items():
        actual = {row["doc_id"] for row, _ in received}
        missing = sorted(set(manifest["tasks"][name]["indices"]) - actual)
        coverage[name] = {
            "verified_responses": len(received),
            "expected_responses": len(manifest["tasks"][name]["indices"]),
            "missing_doc_ids": missing,
            "complete_task_coverage": not missing,
        }
        for row, response in received:
            payload = {"model": config["model"], "messages": row["messages"], **config["generation"][name]}
            verified.append(
                verify_response(
                    tokenizer,
                    parser,
                    eos_ids,
                    payload,
                    response,
                    {
                        "task": name,
                        "doc_id": row["doc_id"],
                        "doc_sha256": row["doc_sha256"],
                        "request_sha256": row["request_sha256"],
                    },
                )
            )
    evidence = {
        "status": "received_responses_verified",
        "scope": "Every retained complete response; missing responses are unverified and overall stage remains incomplete",
        "verified_responses": len(verified),
        "expected_responses": len(inputs),
        "complete_coverage": len(verified) == len(inputs),
        "coverage": coverage,
        "literal_close_responses": sum(row["literal_close_markers_in_content"] > 0 for row in verified),
        "inference_calls": 0,
        "requests": verified,
    }
    write(run / "chat-token-verification-partial.json", evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=MODEL_DIR / "doc/benchmark/run")
    parser.add_argument("--verify-received", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    config, manifest = read(run / "run_config.json"), read(run / "manifest.json")
    invocation = read(run.parent / "invocation.json")
    started = invocation["started_monotonic"]
    deadline = started + min(3600, config["budget_seconds"])
    require(0 <= time.monotonic() - started < min(3600, config["budget_seconds"]), "Original clock expired or changed")
    if args.verify_received:
        print(
            json.dumps(
                {
                    key: value
                    for key, value in verify_received(
                        run, config, manifest, rows(run / "benchmark-inputs.jsonl")
                    ).items()
                    if key != "requests"
                }
            )
        )
        return
    summary = read(run / "summary.json")
    require(summary["status"] == "failed", "Fallback requires retained failed-stage summary")
    if not (run / "benchmark-inputs.jsonl").exists():
        preserve_inputs(run, config)
    evidence = score_received(run, config, manifest, rows(run / "benchmark-inputs.jsonl"))
    summary.update(status="failed", offline_accuracy=evidence, completed_accuracy=evidence["completed_tasks"])
    for name, complete in evidence["completed_tasks"].items():
        summary["incomplete_accuracy"][name].update(
            scoring_status="complete frozen task scored offline with upstream code", score=complete["score"]
        )
    write(run / "summary.json", summary)
    remaining = deadline - time.monotonic()
    require(remaining > 0, "Original deadline exhausted before token verification")
    with (run / "verify-received-chat-tokens.log").open("w") as log:
        subprocess.run(
            [
                str(REPO / "python_env/bin/python"),
                str(Path(__file__).resolve()),
                "--run-dir",
                str(run),
                "--verify-received",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=remaining,
            env=os.environ.copy(),
        )
    expected = (run.parent / "setup/context-contract-original.sha256").read_text().strip()
    actual = hashlib.sha256((MODEL_DIR / "doc/context_contract.json").read_bytes()).hexdigest()
    require(actual == expected, "Original context contract bytes changed")
    summary.update(
        context_contract_bytes_unchanged=True,
        elapsed_seconds=time.monotonic() - started,
        within_original_budget=time.monotonic() < deadline,
    )
    summary["received_token_verification"] = {
        key: value for key, value in read(run / "chat-token-verification-partial.json").items() if key != "requests"
    }
    write(run / "summary.json", summary)
    print(
        json.dumps(
            {
                "status": "failed",
                "completed_tasks": list(evidence["completed_tasks"]),
                "received_and_scored": evidence["received_and_scored"],
                "intended_questions": evidence["intended_questions"],
                "elapsed_seconds": summary["elapsed_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
