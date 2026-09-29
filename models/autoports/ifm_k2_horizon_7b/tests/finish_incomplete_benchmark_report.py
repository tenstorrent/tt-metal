"""Offline final report after the separate IFEval continuation; never infer.

Run only after inference/client shutdown and any completed IFEval directory has
been preserved under run/ifeval. Partial transcripts never become scored subsets.
The original invocation clock and failed overall stage status are mandatory.
"""

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from finish_incomplete_benchmark import MODEL_DIR, partial_accuracy, read, render_report, write

IFEVAL_METRICS = (
    "prompt_level_strict_acc,none",
    "inst_level_strict_acc,none",
    "prompt_level_loose_acc,none",
    "inst_level_loose_acc,none",
)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validated_ifeval(run, config, manifest):
    """Accept only the entire frozen set with upstream scores and sample proof."""
    from benchmark_stage.evidence import benchmark_rows
    from benchmark_stage.responses import scoring_response
    from benchmark_stage.subsets import digest

    path = run / "ifeval/results.json"
    if not path.exists():
        return None
    result = read(path)
    meta = result["benchmark_stage"]
    frozen = manifest["tasks"]["ifeval"]
    expected = manifest["groups"]["ifeval"]["sample_count"]
    for key, value in {
        "model": config["model"],
        "group": "ifeval",
        "subset_sha256": manifest["manifest_sha256"],
        "concurrency": 32,
        "expected_samples": expected,
        "responses": expected,
        "generation_overrides": config["generation"]["ifeval"],
    }.items():
        if meta.get(key) != value:
            raise ValueError(f"IFEval incomplete or mismatched {key}")
    responses = jsonl(run / "ifeval/responses.jsonl")
    if len(responses) != expected or len({row["id"] for row in responses}) != expected:
        raise ValueError("IFEval response population/identity mismatch")
    reasons = Counter(response["choices"][0].get("finish_reason", "missing") for response in responses)
    if set(reasons) - {"stop", "length"} or dict(reasons) != meta["finish_reasons"]:
        raise ValueError("IFEval stop/truncation evidence mismatch")
    normalized = [scoring_response(response) for response in responses]
    if sum(empty for _, empty in normalized) != meta["empty_final_length_responses"]:
        raise ValueError("IFEval empty-final evidence mismatch")
    samples = jsonl(run / "ifeval/samples_ifeval.jsonl")
    if len(samples) != expected or sorted(row["doc_id"] for row in samples) != frozen["indices"]:
        raise ValueError("IFEval scored document set is incomplete")
    hashes = dict(zip(frozen["indices"], frozen["document_sha256"]))
    inputs = {row["doc_id"]: row for row in jsonl(run / "benchmark-inputs.jsonl") if row["task"] == "ifeval"}
    for sample in samples:
        index = sample["doc_id"]
        if digest(sample["doc"]) != hashes[index] or inputs[index]["doc_sha256"] != hashes[index]:
            raise ValueError("IFEval scored/reconstructed document hash mismatch")
        arguments = sample.get("arguments", [])
        if len(arguments) != 1 or digest(arguments[0]) != inputs[index]["request_sha256"]:
            raise ValueError("IFEval scored request differs from preserved exact benchmark input")
    # Ensure scored response contents exhaust the complete raw response set;
    # evaluation completion order is independent of frozen sample order.
    scored = Counter(json.dumps(row["resps"], ensure_ascii=False, sort_keys=True) for row in samples)
    completed = Counter(
        json.dumps([[row["choices"][0]["message"]["content"]]], ensure_ascii=False, sort_keys=True)
        for row, _ in normalized
    )
    if scored != completed:
        raise ValueError("IFEval scored answers differ from complete retained responses")
    score_config = copy.deepcopy(config)
    score_config.setdefault("metrics", {})["ifeval"] = list(IFEVAL_METRICS)
    rows = benchmark_rows(score_config, manifest, "ifeval", result)
    if len(rows) != 4:
        raise ValueError("IFEval must preserve all four upstream accuracy metrics")
    return {
        "result": result,
        "rows": rows,
        "validation": "full frozen population, exact docs/requests, raw answers and four upstream scores verified",
    }


def final_report(run, config, manifest, summary, started, ifeval):
    render_report(run, config, manifest, summary, started)
    path = run / "REPORT.md"
    text = path.read_text()
    empty_accuracy_table = (
        "| Benchmark | Responses | Token-limited | Empty final at token limit | Wall seconds |\n"
        "|---|---:|---:|---:|---:|\n\n"
        "Accuracy tasks share one request pool; their wall times refer to the same interval.\n\n"
    )
    text = text.replace(empty_accuracy_table, "")
    text += (
        "\nAccuracy execution had two intervals: the original shared pool was interrupted after 1109.3 seconds, "
        "then IFEval was attempted separately after both serving profiles. Its bounded attempt is retained in "
        "[the continuation record](accuracy-resume/attempt.json). Final response counts cover these separate intervals.\n"
    )
    if ifeval:
        replacement = []
        count = manifest["groups"]["ifeval"]
        for metric, score, reference in ifeval["rows"]:
            published = f"{reference['score']:.2f}" if reference else "Unavailable"
            delta = f"{score-reference['score']:+.2f}" if reference else "N/A"
            link = f"[reference]({reference['source_url']})" if reference else "No matching published figure"
            replacement.append(
                f"| ifeval | {count['sample_count']} completed / {count['sample_count']} frozen / {count['population']} full | {metric} | {score:.2f} | {published} | {delta} | {link} |"
            )
        lines = text.splitlines()
        old_row = next(
            index
            for index, line in enumerate(lines)
            if line.startswith("| ifeval |") and "Upstream scoring incomplete" in line
        )
        lines[old_row : old_row + 1] = replacement
        text = "\n".join(lines) + "\n"
        text += "\n## Completed IFEval continuation\n\n"
        meta = ifeval["result"]["benchmark_stage"]
        text += f"IFEval completed all {meta['responses']} frozen questions at concurrency 32 in {meta['elapsed_seconds']:.3f} seconds, using the unchanged generation settings and upstream scorer. All four scores above include every question, including token-limited responses. [Upstream results](ifeval/results.json), [scored samples](ifeval/samples_ifeval.jsonl), [raw responses](ifeval/responses.jsonl). GPQA remains incomplete; the overall stage remains failed.\n\n"
        text += f"Actual continuation scope: `{meta['timing_scope']}`; shared groups: `{json.dumps(meta['shared_groups'])}`. This individual task continuation is recorded separately from the interrupted original shared pass.\n"
    elif summary.get("ifeval_validation_error"):
        text += (
            "\nIFEval score validation failed; no aggregate is reported: " + summary["ifeval_validation_error"] + "\n"
        )
    text += "\n[Exact preserved benchmark inputs](benchmark-inputs.jsonl) include all 384 intended questions, document hashes, structured messages, generation settings and request hashes. [Input reconstruction evidence](input-reconstruction.json). No inference is performed by this reporting helper.\n\n"
    question_scores = run / "upstream-question-scoring.json"
    if question_scores.exists():
        scoring = read(question_scores)
        text += (
            f"Unmodified upstream filters and per-question scorers were run for all {scoring['received_and_scored']} received final answers. "
            "[Per-question scores](upstream-question-scores.jsonl) list all 384 intended questions, with null scores for missing responses; "
            "no aggregate is calculated over an incomplete subset. [Scorer settings and method](upstream-question-scoring.json).\n\n"
        )
    text += "[Performance prompt reconstruction](performance-input-reconstruction.json) verifies distinct seeded prompts and 4096 actual input tokens for every measured and warmup request.\n\n"
    text += "Large raw files, logs and verbatim source copies are committed in [lossless evidence archives](archives/README.md), with original files retained in the working checkout and every byte bound by the archive manifest.\n\n"
    text += "The pinned [checkpoint evaluation guidance](https://huggingface.co/IFM/K2-Horizon-7B/blob/036114ce8d46c32b24c15423211069abb9c5d25e/README.md) supplies the high-reasoning generation policy; it provides no matching GPQA/IFEval score for this checkpoint.\n\n"
    server = summary.get("accuracy_resume_server")
    if server:
        text += f"The additional 32-slot restart used [its own recorded server identity](accuracy-resume/server.json), separate from performance identities, with startup {server['startup_seconds']:.3f} seconds. Its launch, reload/compilation and accuracy work count toward the same original stage clock. Retained identity SHA256: `{server['identity_sha256']}`.\n\n"
    text += f"Final status: failed. Total original client-stage elapsed time including final report: {time.monotonic()-started:.3f} seconds; budget {summary['budget_seconds']:.0f} seconds.\n"
    path.write_text(text)
    summary["elapsed_seconds"] = time.monotonic() - started
    summary["within_original_budget"] = summary["elapsed_seconds"] < summary["budget_seconds"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=MODEL_DIR / "doc/benchmark/run")
    parser.add_argument(
        "--reason",
        default="GPQA did not finish its frozen subset before client interruption; final benchmark stage is incomplete",
    )
    args = parser.parse_args()
    from benchmark_stage.evidence import validate_server
    from benchmark_stage.subsets import digest

    run = args.run_dir.resolve()
    config, manifest = read(run / "run_config.json"), read(run / "manifest.json")
    invocation = read(run.parent / "invocation.json")
    started = float(invocation["started_monotonic"])
    budget = min(3600, float(config.get("budget_seconds", 3600)))
    deadline = started + budget
    summary = read(run / "summary.json")
    if summary.get("status") != "failed" or time.monotonic() < started:
        raise RuntimeError("Expected original failed stage and unchanged monotonic clock")
    for name in ("summary.json", "REPORT.md", "evidence-check.log", "context-check.log"):
        source, target = run / name, run / ("before-accuracy-resume-" + name)
        if source.exists() and not target.exists():
            shutil.copyfile(source, target)
    summary.update(
        status="failed",
        error=args.reason,
        budget_seconds=budget,
        accuracy={},
        incomplete_accuracy=partial_accuracy(run, config, manifest),
    )
    try:
        ifeval = validated_ifeval(run, config, manifest)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        ifeval = None
        summary["ifeval_validation_error"] = f"{type(exc).__name__}: {exc}"
    if ifeval:
        summary["accuracy"]["ifeval"] = ifeval["result"]["benchmark_stage"]
        summary["incomplete_accuracy"]["ifeval"].update(
            scoring_status="completed upstream over all frozen questions",
            score={key: score for key, score, _ in ifeval["rows"]},
        )
        summary["ifeval_evidence_validation"] = ifeval["validation"]
        summary[
            "accuracy_continuation_execution"
        ] = "ifeval individually after interrupted shared accuracy and both performance profiles"
    server_path = run / "accuracy-resume/server.json"
    if server_path.exists():
        server = read(server_path)
        validate_server(
            server,
            32,
            config["model"],
            config["base_url"],
            baseline=read(run / "perf-b32-server.json"),
            output=server_path.parent,
        )
        summary["accuracy_resume_server"] = {
            "path": "accuracy-resume/server.json",
            "identity_sha256": digest(server),
            "startup_seconds": server["process"]["startup_seconds"],
            "max_num_seqs": 32,
        }
    write(run / "incomplete-accuracy.json", summary["incomplete_accuracy"])
    final_report(run, config, manifest, summary, started, ifeval)
    write(run / "summary.json", summary)
    plugin = Path(os.environ["TT_MODEL_BRINGUP_ROOT"])
    checks = {
        "evidence-check": [
            sys.executable,
            "-m",
            "benchmark_stage.check",
            "--model-dir",
            str(MODEL_DIR),
            "--hf-model",
            config["model"],
        ],
        "context-check": [
            sys.executable,
            str(plugin / "scripts/check_context_contract.py"),
            "--model-dir",
            str(MODEL_DIR),
            "--hf-model",
            config["model"],
            "--stage",
            "benchmark",
            "--require-contract",
        ],
    }
    for label, argv in checks.items():
        result = {"command": argv, "exit_code": None, "status": "not run: original deadline exhausted"}
        remaining = deadline - time.monotonic()
        if remaining > 0:
            try:
                with (run / f"{label}.log").open("w") as log:
                    log.write(json.dumps(argv) + "\n")
                    log.flush()
                    completed = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=remaining)
                result.update(
                    exit_code=completed.returncode, status="passed" if completed.returncode == 0 else "failed"
                )
            except subprocess.TimeoutExpired:
                result["status"] = "timed out at original deadline"
        summary.setdefault("checks", {})[label] = result
    summary["checks"]["evidence-check"]["expected_failure"] = "GPQA incomplete; stage remains failed"
    expected = (run.parent / "setup/context-contract-original.sha256").read_text().strip()
    summary["context_contract_bytes_unchanged"] = (
        hashlib.sha256((MODEL_DIR / "doc/context_contract.json").read_bytes()).hexdigest() == expected
    )
    final_report(run, config, manifest, summary, started, ifeval)
    write(run / "summary.json", summary)
    print(
        json.dumps(
            {
                "status": "failed",
                "ifeval": "complete" if ifeval else "incomplete/unvalidated",
                "gpqa": "incomplete",
                "elapsed_seconds": summary["elapsed_seconds"],
            }
        )
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
