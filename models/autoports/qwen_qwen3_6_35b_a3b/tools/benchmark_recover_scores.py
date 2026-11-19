#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Recover upstream scores from retained responses, with network access forbidden.

Run with the original lm-eval environment and benchmark_stage on PYTHONPATH.
This does not resume inference or turn an incomplete benchmark stage into a pass.
Use --inputs-only to export verified contexts without touching scoring evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forbid_network(*args, **kwargs):
    raise RuntimeError("Offline score recovery forbids network connections and model requests")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--inputs-only", action="store_true", help="export verified inputs without scoring or backups")
    args = parser.parse_args()
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    socket.socket.connect = forbid_network
    socket.socket.connect_ex = forbid_network
    socket.create_connection = forbid_network

    from benchmark_stage.answers import answer_statistics, output_cap
    from benchmark_stage.gpqa import processing_policy, task_spec
    from benchmark_stage.responses import read_jsonl, scoring_response
    from benchmark_stage.subsets import digest, flatten
    from lm_eval import evaluator
    from lm_eval.models.openai_completions import LocalChatCompletion
    from lm_eval.tasks import get_task_dict

    root = args.run_dir.resolve()
    if not args.inputs_only:
        preserved = root / "failed-execution"
        preserved.mkdir(exist_ok=True)
        for name in ("summary.json", "REPORT.md"):
            if (root / name).exists() and not (preserved / name).exists():
                shutil.copy2(root / name, preserved / name)
        summary = json.loads((preserved / "summary.json").read_text())
    config = json.loads((root / "run_config.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    assert version("lm_eval") == manifest["harness_version"]
    assert digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}) == manifest["manifest_sha256"]
    groups = config["tasks"]
    assert len({digest(config["generation"][g]) for g in groups}) == 1
    tasks = flatten(get_task_dict([task_spec(g) for g in groups]))
    membership = {task: group for group in groups for task in manifest["groups"][group]["tasks"]}
    assert set(tasks) == set(membership)
    documents, samples = {}, {}
    for name, task in tasks.items():
        docs = list(task.eval_docs)
        frozen = manifest["tasks"][name]
        assert frozen.get("document_processing") == processing_policy(name)
        assert len(docs) == frozen["population"] and digest(docs) == frozen["population_sha256"]
        assert (task.config.num_fewshot or 0) == frozen["num_fewshot"]
        if frozen["num_fewshot"]:
            assert digest(list(task.fewshot_docs())) == frozen["fewshot_sha256"]
        assert [digest(docs[i]) for i in frozen["indices"]] == frozen["document_sha256"]
        documents[name], samples[name] = docs, frozen["indices"]

    responses, links, raw_hashes = {}, {}, {}
    for group in groups:
        for filename in ("responses.jsonl", "request_links.jsonl"):
            raw_hashes[f"{group}/{filename}"] = sha(root / group / filename)
        group_responses = read_jsonl(root / group / "responses.jsonl")
        group_links = read_jsonl(root / group / "request_links.jsonl")
        assert len(group_responses) == len(group_links)
        for response in group_responses:
            assert response["id"] not in responses
            scoring_response(response)
            responses[response["id"]] = response
        for link in group_links:
            identity = link["request_instance_sha256"]
            assert identity not in links and link["group"] == group
            assert link["response_id"] in responses
            links[identity] = link
    assert len({link["response_id"] for link in links.values()}) == len(responses)

    captured, used = {}, set()

    class CapturedInputs(Exception):
        pass

    class Replay(LocalChatCompletion):
        model_call = forbid_network
        amodel_call = forbid_network

        def __init__(self, selected, capture=False):
            super().__init__(
                model=config["model"],
                base_url="http://127.0.0.1:0/FORBIDDEN",
                tokenized_requests=False,
                tokenizer_backend=None,
                num_concurrent=32,
                max_gen_toks=2048,
            )
            self.selected, self.capture = selected, capture

        def generate_until(self, requests, **kwargs):
            answers = []
            for req in requests:
                assert req.repeats == 1
                doc_id = self.selected[req.task_name][req.doc_id]
                key = digest([req.args[0], req.args[1]])
                identity = digest([req.task_name, doc_id, key])
                group = membership[req.task_name]
                assert digest(req.doc) == digest(documents[req.task_name][doc_id])
                link = links.get(identity)
                if link is not None:
                    assert link["task"] == req.task_name and link["doc_id"] == doc_id
                    assert link["request_sha256"] == key and link["group"] == group
                if self.capture:
                    assert identity not in captured
                    messages = self.create_message([req.args[0]])
                    assert isinstance(messages, list) and all(isinstance(m, dict) for m in messages)
                    captured[identity] = dict(
                        group=group,
                        task=req.task_name,
                        original_doc_id=doc_id,
                        document=req.doc,
                        document_sha256=digest(req.doc),
                        request_sha256=key,
                        request_instance_sha256=identity,
                        structured_messages=messages,
                        serialized_context=req.args[0].prompt,
                        request_context_hash_representation=req.args[0],
                        generation_kwargs=req.args[1],
                        num_fewshot=manifest["tasks"][req.task_name]["num_fewshot"],
                        fewshot_as_multiturn=True,
                        response_id=None if link is None else link["response_id"],
                        response_retained=link is not None,
                        template="Structured messages; original server applied native HF template exactly once. No template applied during replay.",
                        provenance="Regenerated with original recipe/seeds; retained request hashes verified where responses exist.",
                    )
                    continue
                if link is None:
                    raise RuntimeError(f"Missing retained response: {req.task_name}/{doc_id}")
                assert identity not in used and identity in captured
                used.add(identity)
                answer = self.parse_generations(scoring_response(responses[link["response_id"]])[0])
                assert len(answer) == 1 and answer[0] is not None
                answers.append(answer[0])
            if self.capture:
                raise CapturedInputs()
            return answers

    def evaluate(backend, selected, selected_groups):
        return evaluator.simple_evaluate(
            model=backend,
            tasks=[task_spec(g) for g in selected_groups],
            samples=selected,
            apply_chat_template=True,
            fewshot_as_multiturn=True,
            log_samples=True,
            gen_kwargs=config["generation"][selected_groups[0]],
            random_seed=0,
            numpy_random_seed=1234,
            torch_random_seed=1234,
            fewshot_random_seed=1234,
        )

    try:
        evaluate(Replay(samples, capture=True), samples, groups)
    except CapturedInputs:
        pass
    assert len(captured) == sum(len(ids) for ids in samples.values())
    assert set(links) <= set(captured)
    lines = [json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in captured.values()]
    input_files = []
    if sum(len(line.encode()) for line in lines) <= 500_000:
        target = root / "benchmark-inputs.jsonl"
        target.write_text("".join(lines))
        input_files.append(target.name)
    else:
        parts, current, size = [], [], 0
        for line in lines:
            if current and size + len(line.encode()) > 500_000:
                parts.append(current)
                current, size = [], 0
            current.append(line)
            size += len(line.encode())
        if current:
            parts.append(current)
        for index, part in enumerate(parts, 1):
            target = root / f"benchmark-inputs-{index:03d}.jsonl"
            target.write_text("".join(part))
            input_files.append(target.name)

    if args.inputs_only:
        for filename, expected_sha in raw_hashes.items():
            assert sha(root / filename) == expected_sha, "Input export altered raw inference evidence"
        provenance = dict(
            mode="inputs_only",
            started_utc=started_utc,
            finished_utc=datetime.now(timezone.utc).isoformat(),
            offline_input_export_seconds=time.monotonic() - started,
            command=[sys.executable, *sys.argv],
            pythonpath=os.environ.get("PYTHONPATH"),
            tool_sha256=sha(Path(__file__)),
            lm_eval_version=version("lm_eval"),
            manifest_sha256=manifest["manifest_sha256"],
            run_configuration_sha256=sha(root / "run_config.json"),
            raw_artifact_sha256=raw_hashes,
            input_files={filename: sha(root / filename) for filename in input_files},
            regenerated_inputs=len(captured),
            verified_retained_request_hashes=len(links),
            missing_response_inputs=[
                {"task": row["task"], "original_doc_id": row["original_doc_id"]}
                for row in captured.values()
                if not row["response_retained"]
            ],
            network="Socket connections and model request methods explicitly forbidden; HF offline modes enabled",
            scope="Input regeneration and identity verification only; no scoring, statistics, summary, or failed-execution writes",
        )
        save(root / "input-export-provenance.json", provenance)
        print(json.dumps({"inputs": len(captured), "verified_responses": len(links), "mode": "inputs_only"}))
        return

    recovered = {}
    for group in groups:
        children = manifest["groups"][group]["tasks"]
        retained = {(link["task"], link["doc_id"]) for link in links.values() if link["group"] == group}
        selected = {name: [i for i in samples[name] if (name, i) in retained] for name in children}
        assert all(selected.values()), "A child task has no retained response"
        expected = manifest["groups"][group]["sample_count"]
        partial = len(retained) != expected
        result = evaluate(Replay(selected), selected, [group])
        assert result is not None
        scored_samples = result.pop("samples")
        for name, rows in scored_samples.items():
            for row in rows:
                assert row["doc_id"] in selected[name]
                assert digest(row["doc"]) == digest(documents[name][row["doc_id"]])
                row["recovery_original_doc_id"] = row["doc_id"]
                row["recovery_partial_frozen_task"] = partial
            target = root / group / f"samples_{name}.jsonl"
            target.write_text("".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows))
        group_responses = [responses[link["response_id"]] for link in links.values() if link["group"] == group]
        missing = {name: [i for i in samples[name] if (name, i) not in retained] for name in children}
        result["benchmark_stage"] = dict(
            model=config["model"],
            group=group,
            subset_sha256=manifest["manifest_sha256"],
            expected_samples=expected,
            responses=len(retained),
            partial_frozen_task=partial,
            missing_original_doc_ids=missing,
            concurrency=summary["accuracy_concurrency"],
            finish_reasons=dict(Counter(r["choices"][0]["finish_reason"] for r in group_responses)),
            generation_overrides=config["generation"][group],
            fewshot_as_multiturn=True,
            empty_final_length_responses=sum(scoring_response(r)[1] for r in group_responses),
            inference_timing_scope="Original failed shared accuracy pass; per-task wall time not separately measured",
            original_failed_client_seconds=summary["client_seconds"],
            request_timeout_seconds=3600,
            timing_scope="Offline upstream scoring recovery; no new inference",
            shared_groups=groups,
            template="Original native server chat template; structured messages exactly once",
            recovery_provenance="../recovery-provenance.json",
        )
        filename = "partial-results.json" if partial else "results.json"
        save(root / group / filename, result)
        statistics = answer_statistics(root / group, children, output_cap(config, group, result))
        statistics.update(
            partial_frozen_task=partial,
            expected_frozen_answers=expected,
            missing_original_doc_ids=missing,
            provenance="Offline recovery from original retained responses",
        )
        save(root / group / ("partial-answer-statistics.json" if partial else "answer-statistics.json"), statistics)
        recovered[group] = dict(
            file=f"{group}/{filename}",
            responses=len(retained),
            expected_samples=expected,
            partial_frozen_task=partial,
            missing_original_doc_ids=missing,
            metrics=result["results"].get(group, result["results"]),
        )
    assert used == set(links)
    for filename, expected_sha in raw_hashes.items():
        assert sha(root / filename) == expected_sha, "Recovery altered raw inference evidence"
    provenance = dict(
        started_utc=started_utc,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        offline_recovery_seconds=time.monotonic() - started,
        command=[sys.executable, *sys.argv],
        pythonpath=os.environ.get("PYTHONPATH"),
        tool_sha256=sha(Path(__file__)),
        lm_eval_version=version("lm_eval"),
        manifest_sha256=manifest["manifest_sha256"],
        raw_artifact_sha256=raw_hashes,
        input_files=input_files,
        regenerated_inputs=len(captured),
        verified_retained_request_hashes=len(links),
        missing_response_inputs=[
            {"task": row["task"], "original_doc_id": row["original_doc_id"]}
            for row in captured.values()
            if not row["response_retained"]
        ],
        network="Socket connections and model request methods explicitly forbidden; HF offline modes enabled",
        inference_clock={
            key: summary.get(key)
            for key in (
                "status",
                "error",
                "client_seconds",
                "setup_seconds",
                "elapsed_seconds",
                "client_budget_seconds",
            )
        },
        clock_note="Offline recovery is separately timed after the failed execution; it does not erase or reset the original clocks or establish stage completion.",
        results=recovered,
    )
    save(root / "recovery-provenance.json", provenance)
    print(
        json.dumps(
            {"recovered": recovered, "inputs": len(captured), "offline_seconds": provenance["offline_recovery_seconds"]}
        )
    )


if __name__ == "__main__":
    main()
