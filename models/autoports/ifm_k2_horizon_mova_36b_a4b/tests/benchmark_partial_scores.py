# SPDX-License-Identifier: Apache-2.0
"""Preserve upstream scores for received answers without inventing missing answers.

Run after interrupting the accuracy client (or to take a read-only snapshot).
Does not issue HTTP requests, change inputs, or overwrite upstream result files.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

RUNTIME = Path(
    "/home/ttuser/k2-horizon-mova/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.20/runtime"
)
PRIVATE_NLTK = Path("/home/ttuser/.k2_benchmark_nltk")


def snapshot_jsonl(path):
    """A killed writer may leave one incomplete last line; record it explicitly."""
    if not path.exists():
        return [], None
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    records = []
    for i, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if i != len(lines) - 1 or line.endswith(b"\n"):
                raise
            return records, {"path": str(path), "trailing_incomplete_bytes": len(line)}
    return records, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=Path(__file__).parents[1] / "doc/benchmark/run")
    args = ap.parse_args()
    run = args.run.resolve()
    os.environ["VLLM_PLUGINS"] = ""
    os.environ["NLTK_DATA"] = str(PRIVATE_NLTK)
    sys.path.insert(0, str(RUNTIME))
    import nltk

    nltk.data.path.insert(0, str(PRIVATE_NLTK))
    nltk.data.find("tokenizers/punkt_tab")
    from benchmark_stage.gpqa import task_spec
    from benchmark_stage.responses import scoring_response
    from benchmark_stage.subsets import digest, flatten
    from lm_eval.api.instance import Instance
    from lm_eval.models.openai_completions import LocalChatCompletion
    from lm_eval.tasks import get_task_dict

    manifest = json.loads((run / "manifest.json").read_text())
    assert version("lm_eval") == manifest["harness_version"]
    assert digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}) == manifest["manifest_sha256"]
    inputs, tail = snapshot_jsonl(run / "benchmark_inputs.jsonl")
    assert tail is None
    lookup = {(row["task"], row["doc_id"]): row for row in inputs}
    assert len(lookup) == len(inputs) == sum(g["sample_count"] for g in manifest["groups"].values())
    for task, frozen in manifest["tasks"].items():
        for doc_id, expected_hash in zip(frozen["indices"], frozen["document_sha256"]):
            row = lookup[task, doc_id]
            assert digest(row["document"]) == row["document_sha256"] == expected_hash
    tasks = {}
    for group in manifest["groups"]:
        tasks.update(flatten(get_task_dict([task_spec(group)])))
    linked = {}
    integrity_notes = []
    orphan_ids = []
    for group in manifest["groups"]:
        responses, tail = snapshot_jsonl(run / group / "responses.jsonl")
        if tail:
            integrity_notes.append(tail)
        links, tail = snapshot_jsonl(run / group / "request_links.jsonl")
        if tail:
            integrity_notes.append(tail)
        by_id = {r["id"]: r for r in responses}
        assert len(by_id) == len(responses), "Duplicate raw response IDs"
        used = set()
        for link in links:
            key = link["task"], link["doc_id"]
            assert key in lookup and key not in linked, f"Duplicate/unknown link {key}"
            response = by_id.get(link["response_id"])
            if response is None:
                integrity_notes.append({"link_without_response": link})
                continue
            linked[key] = (link, response)
            used.add(link["response_id"])
        orphan_ids.extend(sorted(set(by_id) - used))
    scored = {}
    aggregates = defaultdict(lambda: defaultdict(list))
    for name, task in tasks.items():
        instances = []
        for key, (link, response) in linked.items():
            if key[0] != name:
                continue
            normalized, exhausted = scoring_response(response)
            # Use the exact installed backend text postprocessor as evaluate.py.
            text = LocalChatCompletion.parse_generations(SimpleNamespace(think_end_token=None), normalized)[0]
            source = lookup[key]
            instance = Instance(
                request_type="generate_until",
                doc=source["document"],
                arguments=(source["messages"], source["generation"]),
                idx=0,
                metadata=(name, key[1], 1),
                resps=[text],
            )
            instances.append(instance)
            scored[key] = {
                "status": "completed",
                "response_id": response["id"],
                "request_sha256": link["request_sha256"],
                "raw_response": response,
                "scoring_text": text,
                "empty_final_length_response": exhausted,
                "finish_reason": response["choices"][0]["finish_reason"],
                "filtered_responses": {},
                "upstream_scores": {},
            }
        if not instances:
            continue
        for pipeline in task._filters:
            pipeline.apply(instances)
        for inst in instances:
            item = scored[name, inst.doc_id]
            item["filtered_responses"] = inst.filtered_resps
            for filter_name, filtered in inst.filtered_resps.items():
                metrics = task.process_results(inst.doc, [filtered])
                item["upstream_scores"][filter_name] = metrics
                for metric, value in metrics.items():
                    aggregates[name][metric + "," + filter_name].append(value)
    records = []
    for source in inputs:
        key = source["task"], source["doc_id"]
        row = dict(source)
        row.update(
            scored.get(
                key, {"status": "missing_response", "upstream_scores": None, "scoring_text": None, "raw_response": None}
            )
        )
        records.append(row)
    summaries = {}
    for name, task in tasks.items():
        rows = [r for r in records if r["task"] == name]
        complete = [r for r in rows if r["status"] == "completed"]
        diagnostics = {key: task.aggregation()[key.split(",")[0]](values) for key, values in aggregates[name].items()}
        summaries[name] = dict(
            selected_samples=len(rows),
            full_population=manifest["tasks"][name]["population"],
            completed_responses=len(complete),
            missing_responses=len(rows) - len(complete),
            selected_set_scores=diagnostics if len(complete) == len(rows) else None,
            completed_only_diagnostic_scores=diagnostics,
            finish_reasons=dict(Counter(r["finish_reason"] for r in complete)),
            empty_final_length_responses=sum(r["empty_final_length_response"] for r in complete),
            fewshot=task.config.num_fewshot or 0,
            upstream_filter_config=task.config.filter_list,
            upstream_metric_config=task.config.metric_list,
        )
    assert not any(n == "ttnn" or n.startswith("ttnn.") for n in sys.modules), "Unexpected TTNN import"
    summary = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        status="complete_response_set" if len(scored) == len(inputs) else "incomplete_response_set",
        selected_question_count=len(inputs),
        completed_response_count=len(scored),
        missing_response_count=len(inputs) - len(scored),
        manifest_sha256=manifest["manifest_sha256"],
        harness_version=version("lm_eval"),
        tasks=summaries,
        integrity_notes=integrity_notes,
        unlinked_raw_response_ids=orphan_ids,
        nltk_data=str(PRIVATE_NLTK),
        ttnn_imported=False,
        interpretation="Completed-only diagnostics are completion-biased and are not selected-set benchmark scores. Missing responses have null scores, not fabricated failures. Received token-limited answers are scored normally and retained.",
        scope="Host-only snapshot; no generation, input changes, or accuracy acceptance verdict.",
    )
    (run / "partial_scores.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in records)
    )
    (run / "partial_scores_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
