"""Retain upstream per-question scores without aggregating incomplete subsets."""

import importlib.metadata
import json
from pathlib import Path

from benchmark_stage.gpqa import task_spec
from benchmark_stage.responses import scoring_response
from benchmark_stage.subsets import digest, flatten
from lm_eval.api.instance import Instance
from lm_eval.tasks import get_task_dict

RUN = Path(__file__).resolve().parents[1] / "doc/benchmark/run"


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def main():
    manifest = json.loads((RUN / "manifest.json").read_text())
    groups = json.loads((RUN / "run_config.json").read_text())["tasks"]
    assert importlib.metadata.version("lm_eval") == manifest["harness_version"]
    inputs = read_rows(RUN / "benchmark-inputs.jsonl")
    assert len(inputs) == sum(manifest["groups"][name]["sample_count"] for name in groups)
    scored = {}
    settings = {}
    for name in groups:
        task = flatten(get_task_dict([task_spec(name)]))[name]
        frozen = manifest["tasks"][name]
        assert digest(list(task.eval_docs)) == frozen["population_sha256"]
        responses = {row["id"]: row for row in read_rows(RUN / name / "responses.jsonl")}
        linked = {row["request_sha256"]: row for row in read_rows(RUN / name / "request_links.jsonl")}
        instances = []
        evidence = {}
        for row in inputs:
            if row["task"] != name or row["request_sha256"] not in linked:
                continue
            link = linked[row["request_sha256"]]
            assert link["doc_id"] == row["doc_id"]
            response = responses[link["response_id"]]
            normalized, empty_at_limit = scoring_response(response)
            answer = normalized["choices"][0]["message"]["content"]
            instance = Instance(
                "generate_until",
                row["doc"],
                tuple(row["arguments"]),
                0,
                metadata=(name, row["doc_id"], 1),
                resps=[answer],
            )
            instances.append(instance)
            evidence[row["doc_id"]] = {
                "response_id": response["id"],
                "finish_reason": response["choices"][0]["finish_reason"],
                "stop_reason": response["choices"][0].get("stop_reason"),
                "empty_final_at_length": empty_at_limit,
                "usage": response.get("usage"),
            }
        if instances:
            task._instances = instances
            task.apply_filters()
            for instance in instances:
                scores = {
                    key: task.process_results(instance.doc, [value]) for key, value in instance.filtered_resps.items()
                }
                scored[(name, instance.doc_id)] = {
                    **evidence[instance.doc_id],
                    "status": "received_and_scored",
                    "filtered_responses": instance.filtered_resps,
                    "upstream_scores": scores,
                }
        settings[name] = {
            "version": task.VERSION,
            "num_fewshot": task.config.num_fewshot or 0,
            "filter_list": task.config.filter_list,
            "scored_questions": len(instances),
        }
    with (RUN / "upstream-question-scores.jsonl").open("w") as stream:
        for row in inputs:
            result = {key: row[key] for key in ("task", "doc_id", "doc_sha256", "request_sha256")}
            result.update(
                scored.get(
                    (row["task"], row["doc_id"]),
                    {
                        "status": "no_completed_response",
                        "upstream_scores": None,
                    },
                )
            )
            stream.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
    (RUN / "upstream-question-scoring.json").write_text(
        json.dumps(
            {
                "harness_version": manifest["harness_version"],
                "manifest_sha256": manifest["manifest_sha256"],
                "intended_questions": len(inputs),
                "received_and_scored": len(scored),
                "settings": settings,
                "aggregation": "None: incomplete response coverage is not an accuracy estimate. All frozen questions remain listed; unanswered questions have null scores, never fabricated model responses.",
                "method": "Unmodified upstream FilterEnsembles and task.process_results over each complete raw final answer; no inference",
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    print({"questions": len(inputs), "received_and_scored": len(scored), "aggregate_scores": "not calculated"})


if __name__ == "__main__":
    main()
