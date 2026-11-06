"""Offline preparation only; retain native content privately, never invoke tools."""

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

MODEL_ROOT = Path("models/autoports/google_gemma_4_31b_it")
EXTRACTOR = MODEL_ROOT / "doc/agentic_evals/runs/harbor-replay-extraction/extract.py"
PIN = "842da3794eaa0b77d5f08bae87a17459d91ff475"
SELECTED = ["pylint-dev__pylint-7080", "django__django-11265", "psf__requests-1766", "pydata__xarray-4966"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.collected.is_dir(), "Await actual collected nightly artifacts"
    assert args.output.resolve().is_relative_to(Path("bringup/artifacts").resolve())
    assert not args.output.exists(), "Never overwrite earlier extraction evidence"
    cohorts = json.loads((MODEL_ROOT / "doc/agentic_evals/cohorts.json").read_text())
    assert cohorts["swe_bench_verified"]["rungs"]["nightly"][:4] == SELECTED
    spec = importlib.util.spec_from_file_location("native_extract", EXTRACTOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prepared = []
    for task in SELECTED:
        matches = [
            p
            for p in args.collected.rglob("mini-swe-agent.trajectory.json")
            if p.parent.parent.name.startswith(task + "__")
        ]
        assert len(matches) == 1, f"{task}: need exactly one recorded trial, got {len(matches)}"
        source = matches[0]
        data = json.loads(source.read_text())
        rows, omitted = module.reconstruct(data, allow_recorded_subset=False)
        assistants = [m for m in data["messages"] if m.get("role") == "assistant"]
        assert len(rows) == len(assistants) >= 3
        usage = []
        for i, (row, message) in enumerate(zip(rows, assistants)):
            response = message["extra"]["response"]
            u = response.get("usage")
            assert u and all(
                isinstance(u.get(k), int) for k in ["prompt_tokens", "completion_tokens"]
            ), f"{task}: missing usage at{i}"
            # Native content/tool schemas come from stock formatter, no JSON synthesis.
            assert row["tools"] and row["messages"]
            assert row.get("max_tokens", row.get("max_completion_tokens")) == 32768
            assert row.get("temperature") == 1.0 and row.get("top_p") == 0.95
            assert row.get("extra_body", {}).get("top_k") == 20
            usage.append({"recorded_call_index": i, "usage": copy.deepcopy(u)})
        http = []
        for row in rows[-3:]:
            body = copy.deepcopy(row)
            extra = body.pop("extra_body", {})
            assert not (set(extra) & set(body)), "Ambiguous provider extra_body collision"
            body.update(extra)
            http.append(body)
        prepared.append((task, source, rows, http, usage, omitted))
    # Write only after all four histories pass; no score/result fields are inspected.
    args.output.mkdir(parents=True)
    manifest = {
        "run_id": 42210467509,
        "kind": "reconstructed_stock_LiteLLM_inputs_not_wire_capture",
        "selection": "first four frozen nightly task IDs; scores not inspected",
        "tasks": [],
        "tokenizer_revision_for_future_parity": PIN,
        "capacity_only": True,
        "exact_output_confirmation": False,
        "limitations": [
            "Strict recorded call count; unrecorded/retried transport calls cannot be recreated.",
            "Stock replay overrides output1/temperature0/streamfalse and discards choices.",
            "No model template token-count or actual API parity claim until separately checked.",
        ],
    }
    for task, source, rows, http, usage, omitted in prepared:
        for group, values in [("native-full", rows), ("native-peak3", rows[-3:]), ("http-peak3", http)]:
            target = args.output / group / (task + ".jsonl")
            target.parent.mkdir(exist_ok=True)
            target.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in values))
            target.chmod(0o600)
        target = args.output / "usage" / (task + ".json")
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(usage, indent=2) + "\n")
        target.chmod(0o600)
        manifest["tasks"].append(
            {
                "task": task,
                "source": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "recorded_calls": len(rows),
                "api_calls": len(rows),
                "missing_recorded_calls": 0,
                "selected_call_indices": list(range(len(rows) - 3, len(rows))),
                "recorded_peak3_usage": usage[-3:],
                "omitted_transport_field_names": omitted,
                "native_full_sha256": hashlib.sha256(
                    (args.output / "native-full" / (task + ".jsonl")).read_bytes()
                ).hexdigest(),
                "http_peak3_sha256": hashlib.sha256(
                    (args.output / "http-peak3" / (task + ".jsonl")).read_bytes()
                ).hexdigest(),
            }
        )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Prepared four distinct histories, twelve final recorded requests; no network/device access")


if __name__ == "__main__":
    main()
