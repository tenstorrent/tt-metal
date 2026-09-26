# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Check default decoder preservation on real-weight 1025/512 traced streams.

Raw HF results remain separate: inherited baseline failures do not become HF
passes when optimized-versus-fused preservation succeeds.
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from models.autoports.google_gemma_4_26b_a4b_it.tests.compare_decoder_outputs import THRESHOLD, compare

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
MODULE = "models.autoports.google_gemma_4_26b_a4b_it.tests"
LENGTH, STEPS = 1025, 512
LAYERS = {0: "sliding_attention", 5: "full_attention"}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes():
    paths = [ROOT / "tt" / name for name in ("functional_decoder.py", "fused_decoder.py", "optimized_decoder.py")] + [
        ROOT / "tests" / name
        for name in ("run_decoder.py", "run_optimized_contract.py", "compare_decoder_outputs.py", Path(__file__).name)
    ]
    return {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def validate_report(report, decoder, layer, fixture):
    """Reject incomplete runs and contradictory gate metadata before comparison."""
    decode = report.get("decode", {})
    rows = decode.get("checks", [])
    expected_positions = set(range(LENGTH, LENGTH + STEPS))
    checks = {
        "decoder": report.get("decoder") == decoder,
        "real_weights": report.get("real_weights") is True,
        "layer_type": report.get("layer_type") == LAYERS[layer],
        "length": report.get("length") == LENGTH and report.get("prefix_length") == 0,
        "prefill_hf": report.get("passed") is True
        and isinstance(report.get("pcc"), (int, float))
        and math.isfinite(report["pcc"])
        and report["pcc"] >= THRESHOLD,
        "prefill_runtime_audit": report.get("runtime_prefill_audit") == "clean",
        "decode_runtime_audit": decode.get("runtime_decode_audit") == "clean",
        "program_cache_guard": report.get("program_cache_miss_guard") is True
        and report.get("prefill_cache_entries", 0) > 0,
        "traced": decode.get("traced") is True,
        "deterministic": decode.get("repeated_equal") is True,
        "steps": decode.get("steps") == STEPS,
        "position_range": decode.get("positions") == [LENGTH, LENGTH + STEPS - 1],
        "all_position_checks": {row.get("position") for row in rows} == expected_positions,
    }
    numerical_rows = all(
        isinstance(row.get("pcc"), (int, float))
        and math.isfinite(row["pcc"])
        and row.get("passed") is (row["pcc"] >= THRESHOLD)
        for row in rows
    )
    checks["hf_flags_match_pcc"] = bool(rows) and numerical_rows
    checks["hf_decode_status_consistent"] = bool(rows) and decode.get("passed") is all(
        row.get("passed") is True for row in rows
    )
    checks["hf_minimum_consistent"] = (
        bool(rows)
        and numerical_rows
        and isinstance(decode.get("min_pcc"), (int, float))
        and math.isclose(decode["min_pcc"], min(row["pcc"] for row in rows), rel_tol=0, abs_tol=1e-12)
    )
    if decoder == "optimized":
        checks["functional_fallback_forbidden"] = report.get("functional_fallback") == "forbidden"
        checks["default_contract"] = report.get("contract") == "run_decoder" and "candidate" not in report
    checks["input_fixture"] = (
        Path(report.get("input_fixture", "")).resolve() == fixture
        if fixture is not None
        else "input_fixture" not in report
    )
    return checks


def expected_process_result(returncode, report, log):
    """Only the harness's final decode-HF assertion may accompany complete results."""
    if report.get("decode", {}).get("passed") is True:
        return returncode == 0
    # Other exceptions, including failures during device teardown, must not be
    # mistaken for an inherited numerical failure merely because files exist.
    final_traceback = log.rsplit("Traceback (most recent call last):", 1)[-1]
    return (
        returncode == 1
        and report.get("decode", {}).get("passed") is False
        and "assert dpass and torch.equal(da, repeat), result" in final_traceback
        and any(line.startswith("AssertionError:") for line in final_traceback.splitlines())
    )


def hf_status(report):
    decode = report["decode"]
    return {
        "prefill_passed": report["passed"],
        "prefill_pcc": report["pcc"],
        "decode_passed": decode["passed"],
        "minimum_decode_pcc": decode["min_pcc"],
        "failed_positions": sorted({row["position"] for row in decode["checks"] if not row["passed"]}),
    }


def parse_fixtures(parser, specifications):
    fixtures = {}
    for specification in specifications:
        layer_text, separator, filename = specification.partition("=")
        if not separator or layer_text not in ("0", "5") or not filename:
            parser.error("--input-fixture must be LAYER=PATH, with layer 0 or 5")
        layer = int(layer_text)
        if layer in fixtures:
            parser.error(f"Duplicate input fixture for layer {layer}")
        path = Path(filename).resolve()
        if not path.is_file():
            parser.error(f"Input fixture does not exist: {path}")
        fixtures[layer] = path
    if fixtures and set(fixtures) != set(LAYERS):
        parser.error("Recorded-input stress requires an --input-fixture for each of layers 0 and 5")
    return fixtures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "doc/optimized_decoder/stress_default")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "--input-fixture", action="append", default=[], metavar="LAYER=PATH", help="Repeat for layers 0 and 5"
    )
    inputs.add_argument(
        "--input-fixture-template",
        help="Recorded-input path containing {layer}, expanded separately for layers 0 and 5",
    )
    args = parser.parse_args()
    specifications = args.input_fixture
    if args.input_fixture_template:
        if "{layer}" not in args.input_fixture_template:
            parser.error("--input-fixture-template must contain {layer}")
        try:
            specifications = [f"{layer}={args.input_fixture_template.format(layer=layer)}" for layer in LAYERS]
        except (KeyError, ValueError) as error:
            parser.error(f"Invalid input-fixture template: {error}")
    fixtures = parse_fixtures(parser, specifications)
    torch.set_num_threads(2)
    output_dir = args.output_dir.resolve()
    input_mode = "recorded" if fixtures else "gaussian"
    run_dir = output_dir / input_mode / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir.mkdir(parents=True)
    journal_path, summary_path = run_dir / "commands.json", run_dir / "summary.json"
    journal = []
    initial_hashes = source_hashes()
    fixture_hashes = {str(layer): hashlib.sha256(path.read_bytes()).hexdigest() for layer, path in fixtures.items()}
    summary = {
        "contract": "optimized_preserves_fused_1025_prefill_512_traced_decode",
        "threshold": THRESHOLD,
        "length": LENGTH,
        "steps": STEPS,
        "real_weights": True,
        "input_mode": input_mode,
        "input_kind": "recorded layer activations" if fixtures else "seed-42 Gaussian activations",
        "evidence_scope": "recorded-input preservation" if fixtures else "diagnostic Gaussian preservation",
        "input_fixtures": {str(layer): str(path) for layer, path in fixtures.items()},
        "input_fixture_sha256": fixture_hashes,
        "source_sha256": initial_hashes,
        "run_directory": str(run_dir),
        "command_journal": str(journal_path),
        "orchestrator_command": [sys.executable, "-m", f"{MODULE}.run_optimized_stress", *sys.argv[1:]],
        "preservation_passed": False,
        "exact_hf_passed": False,
        "interpretation": "A preservation pass retains every raw HF failure; it does not assert exact-HF parity",
        "layers": {},
        "errors": [],
    }
    write_json(summary_path, summary)
    try:
        for layer in LAYERS:
            reports, prefixes = {}, {}
            record = {"checks": {}, "raw_hf": {}, "reports": {}}
            summary["layers"][str(layer)] = record
            for decoder in ("fused", "optimized"):
                prefix = run_dir / f"{decoder}_layer{layer}"
                prefixes[decoder] = prefix
                report_path, tensor_path, log_path = (prefix.with_suffix(suffix) for suffix in (".json", ".pt", ".log"))
                module = "run_decoder" if decoder == "fused" else "run_optimized_contract"
                selector = ["--decoder", "fused"] if decoder == "fused" else ["--contract", "run_decoder"]
                command = [
                    sys.executable,
                    "-m",
                    f"{MODULE}.{module}",
                    *selector,
                    "--layer",
                    str(layer),
                    "--length",
                    str(LENGTH),
                    "--real",
                    "--decode",
                    "--steps",
                    str(STEPS),
                    "--verify-program-cache",
                    "--output",
                    str(report_path),
                    "--save-output-tensors",
                    str(tensor_path),
                ]
                fixture = fixtures.get(layer)
                if fixture is not None:
                    command.extend(("--input-fixture", str(fixture)))
                entry = {"command": command, "cwd": str(REPO), "log": str(log_path), "exit_code": None}
                journal.append(entry)
                write_json(journal_path, journal)
                print(f"Running {decoder} layer {layer}; log: {log_path}", flush=True)
                start = time.monotonic()
                with log_path.open("w") as log:
                    result = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=False)
                entry.update(exit_code=result.returncode, elapsed_seconds=time.monotonic() - start)
                write_json(journal_path, journal)
                report = json.loads(report_path.read_text())
                reports[decoder] = report
                record["reports"][decoder] = str(report_path)
                checks = validate_report(report, decoder, layer, fixture)
                checks["expected_process_result"] = expected_process_result(
                    result.returncode, report, log_path.read_text()
                )
                checks["saved_outputs_exist"] = tensor_path.is_file()
                record["checks"][decoder] = checks
                if all(key in report.get("decode", {}) for key in ("passed", "min_pcc", "checks")):
                    record["raw_hf"][decoder] = hf_status(report)
                if not all(checks.values()):
                    failed = [name for name, passed in checks.items() if not passed]
                    raise AssertionError(f"Layer {layer} {decoder} run checks failed: {failed}")
                write_json(summary_path, summary)
            comparison = compare(prefixes["fused"], prefixes["optimized"])
            comparison_path = run_dir / f"comparison_layer{layer}.json"
            write_json(comparison_path, comparison)
            record["comparison"] = str(comparison_path)
            record["checks"]["preservation"] = {
                "all_outputs_direct_pcc": comparison["direct_passed"],
                "no_new_hf_failures": not comparison["new_hf_failed_positions"],
                "matching_input_metadata": reports["fused"].get("input_source")
                == reports["optimized"].get("input_source"),
            }
            record["minimum_direct_decode_pcc"] = comparison["minimum_direct_decode_pcc"]
            record["direct_failed_positions"] = comparison["direct_failed_positions"]
            record["new_hf_failed_positions"] = comparison["new_hf_failed_positions"]
            record["shared_hf_failed_positions"] = comparison["shared_hf_failed_positions"]
            record["recovered_hf_positions"] = comparison["recovered_hf_positions"]
            write_json(summary_path, summary)
    except (AssertionError, KeyError, OSError, RuntimeError, ValueError) as error:
        summary["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        summary["sources_unchanged"] = source_hashes() == initial_hashes
        summary["input_fixtures_unchanged"] = all(
            hashlib.sha256(path.read_bytes()).hexdigest() == fixture_hashes[str(layer)]
            for layer, path in fixtures.items()
        )
        complete = all("preservation" in summary["layers"].get(str(layer), {}).get("checks", {}) for layer in LAYERS)
        summary["preservation_passed"] = (
            complete
            and not summary["errors"]
            and summary["sources_unchanged"]
            and summary["input_fixtures_unchanged"]
            and all(
                passed
                for record in summary["layers"].values()
                for checks in record["checks"].values()
                for passed in checks.values()
            )
        )
        summary["exact_hf_passed"] = complete and all(
            raw["prefill_passed"] and raw["decode_passed"]
            for record in summary["layers"].values()
            for raw in record["raw_hf"].values()
        )
        write_json(summary_path, summary)
        write_json(output_dir / "summary.json", summary)
        write_json(output_dir / f"{input_mode}_summary.json", summary)
        print(
            json.dumps(
                {
                    "summary": str(summary_path),
                    "preservation_passed": summary["preservation_passed"],
                    "exact_hf_passed": summary["exact_hf_passed"],
                    "errors": summary["errors"],
                }
            ),
            flush=True,
        )
    return 0 if summary["preservation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
