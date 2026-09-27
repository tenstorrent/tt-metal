# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Retain upstream per-question scores for incomplete accuracy; never aggregate."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

from benchmark_stage.responses import read_jsonl, scoring_response
from benchmark_stage.subsets import digest, flatten
from lm_eval.api.instance import Instance
from lm_eval.models.openai_completions import LocalChatCompletion
from lm_eval.tasks import get_task_dict


def score(root):
    manifest = json.loads((root / "manifest.json").read_text())
    config = json.loads((root / "run_config.json").read_text())
    if version("lm_eval") != manifest["harness_version"]:
        raise ValueError("Harness version differs from original frozen manifest")
    if digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest["manifest_sha256"]:
        raise ValueError("Manifest hash mismatch")
    inputs = read_jsonl(root / "benchmark-inputs.jsonl")
    indexed = {}
    for row in inputs:
        key = (row["task"], row["doc_id"])
        if key in indexed:
            raise ValueError("Duplicate retained question")
        frozen = manifest["tasks"][row["task"]]
        offset = frozen["indices"].index(row["doc_id"])
        if digest(row["doc"]) != row["document_sha256"] or row["document_sha256"] != frozen["document_sha256"][offset]:
            raise ValueError("Document hash mismatch")
        if digest(row["arguments"]) != row["request_sha256"]:
            raise ValueError("Retained request hash mismatch")
        indexed[key] = row
    expected = {
        (task, index)
        for group in config["tasks"]
        for task in manifest["groups"][group]["tasks"]
        for index in manifest["tasks"][task]["indices"]
    }
    if set(indexed) != expected:
        raise ValueError("Retained questions do not cover every selected frozen document")
    tasks = flatten(get_task_dict(config["tasks"]))
    for name, task in tasks.items():
        frozen = manifest["tasks"][name]
        if digest(list(task.eval_docs)) != frozen["population_sha256"]:
            raise ValueError("Upstream document population changed")
        if (task.config.num_fewshot or 0) != frozen["num_fewshot"]:
            raise ValueError("Upstream few-shot count changed")
        if frozen["num_fewshot"] and digest(list(task.fewshot_docs())) != frozen["fewshot_sha256"]:
            raise ValueError("Upstream few-shot content changed")
        task._instances = []
    backend = LocalChatCompletion(
        model=config["model"], base_url="http://127.0.0.1:1/unused", tokenizer_backend=None, tokenized_requests=False
    )
    completed = {}
    for group in config["tasks"]:
        response_path = root / group / "responses.jsonl"
        link_path = root / group / "request_links.jsonl"
        responses = read_jsonl(response_path) if response_path.exists() else []
        links = read_jsonl(link_path) if link_path.exists() else []
        by_id = {row["id"]: row for row in responses}
        if len(by_id) != len(responses) or len(links) != len(responses):
            raise ValueError("Duplicate responses or response/link count mismatch")
        linked = set()
        for link in links:
            key = (link["task"], link["doc_id"])
            source = indexed[key]
            if source["group"] != group or link["group"] != group or source["request_sha256"] != link["request_sha256"]:
                raise ValueError("Completed request differs from frozen retained input")
            if key in completed or link["response_id"] in linked:
                raise ValueError("Duplicate completed question/response link")
            raw = by_id[link["response_id"]]
            linked.add(link["response_id"])
            normalized, empty_final_length = scoring_response(raw)
            answer = backend.parse_generations(normalized)[0]
            instance = Instance(
                request_type="generate_until",
                doc=source["doc"],
                arguments=tuple(source["arguments"]),
                idx=0,
                metadata=(link["task"], link["doc_id"], 1),
                resps=[answer],
            )
            tasks[link["task"]]._instances.append(instance)
            completed[key] = {
                "instance": instance,
                "response_id": link["response_id"],
                "finish_reason": raw["choices"][0]["finish_reason"],
                "usage": raw.get("usage"),
                "empty_final_length": empty_final_length,
            }
        if linked != set(by_id):
            raise ValueError("Unlinked response evidence")
    for task in tasks.values():
        if task._instances:
            task.apply_filters()
    rows = []
    group_counts = defaultdict(Counter)
    for source in inputs:
        key = (source["task"], source["doc_id"])
        row = {k: source[k] for k in ("group", "task", "doc_id", "document_sha256", "request_sha256")}
        if key not in completed:
            row.update(
                status="missing_response",
                upstream_scores=None,
                missing_reason="Accuracy client interrupted before complete response; no score assigned.",
            )
            group_counts[source["group"]]["missing_samples"] += 1
        else:
            result = completed[key]
            instance = result["instance"]
            task = tasks[source["task"]]
            scores = {
                filter_name: task.process_results(source["doc"], [value])
                for filter_name, value in instance.filtered_resps.items()
            }
            row.update(
                status="completed_response_scored",
                response_id=result["response_id"],
                filtered_responses=instance.filtered_resps,
                upstream_scores=scores,
                finish_reason=result["finish_reason"],
                usage=result["usage"],
                empty_final_length=result["empty_final_length"],
            )
            count = group_counts[source["group"]]
            count["completed_samples"] += 1
            count[result["finish_reason"] + "_responses"] += 1
            count["empty_final_length_responses"] += int(result["empty_final_length"])
        rows.append(row)
    questions = root / "partial-accuracy-questions.jsonl"
    questions.write_text("".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows))
    groups = {}
    for group in config["tasks"]:
        frozen = manifest["groups"][group]
        groups[group] = {
            key: group_counts[group][key]
            for key in (
                "completed_samples",
                "missing_samples",
                "stop_responses",
                "length_responses",
                "empty_final_length_responses",
            )
        }
        groups[group].update(
            selected_samples=frozen["sample_count"],
            dataset_samples=frozen["population"],
            scope="incomplete_frozen_subset",
            aggregate_score=None,
        )
    summary = {
        "status": "incomplete_accuracy_per_question_only",
        "harness_version": version("lm_eval"),
        "manifest_sha256": manifest["manifest_sha256"],
        "groups": groups,
        "retained_inputs": len(inputs),
        "verified_completed_request_hashes": len(completed),
        "questions_file": questions.name,
        "questions_sha256": hashlib.sha256(questions.read_bytes()).hexdigest(),
        "scoring": "Actual upstream task.apply_filters() and task.process_results(); no copied filters or scorer.",
        "limitations": [
            "No aggregate score: completed answers are a completion-biased subset.",
            "Missing answers remain explicitly unscored; this cannot satisfy the benchmark stage.",
            "Original task results/sample artifacts are not created or overwritten.",
        ],
    }
    (root / "partial-accuracy.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    score(parser.parse_args().run_dir)
