# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Validates benchmark data files produced by the models-CI pipeline.

Supports two file types:
  - partial_run_*.pkl  — written by save_partial_run_json() inside each demo script;
                         device_info is intentionally absent (added by enrichment step).
  - complete_run_*.json — written by create_benchmark_with_environment_json.py after
                          the enrichment step; device_info (including sku) must be present.

Usage:
    python tools/scripts/validate_benchmark_pkl.py path/to/partial_run_*.pkl [...]
    python tools/scripts/validate_benchmark_pkl.py path/to/complete_run_*.json [...]

Exit code 0 = all files pass, 1 = one or more files fail.
"""

# TODO: integrate as a CI step after e2e perf/accuracy tests to catch regressions automatically.

import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

# Measurements required for perf/generate runs (not for pure accuracy runs)
REQUIRED_PERF_MEASUREMENTS = {
    "tokens/s",  # decode_t/s
    "tokens/s/user",  # decode_t/s/u
    "time_to_token",  # TTFT
}

# Run-level fields required in both partial and complete runs
REQUIRED_RUN_FIELDS_ALWAYS = [
    "ml_model_name",
    "batch_size",
    "input_sequence_length",
    "run_type",
    "run_start_ts",
    "run_end_ts",
]

# Additional fields required only in complete (enriched) JSON runs
REQUIRED_RUN_FIELDS_COMPLETE = [
    "device_info",
    "git_commit_hash",
    "git_branch_name",
    "github_pipeline_id",
    "device_hostname",
]

# Fields that are expected but only emit a warning when missing
OPTIONAL_RUN_FIELDS = [
    "precision",
    "output_sequence_length",
    "num_layers",
    "config_params",
]

# Valid run_type values — enforced as a warning (not hard fail) to allow future additions
VALID_RUN_TYPES = {"demo_perf", "demo_accuracy", "demo_generate", "demo_perf_8chip"}

# Accuracy measurements expected when run_type == "demo_accuracy"
ACCURACY_MEASUREMENTS = {"top1_token_accuracy", "top5_token_accuracy"}


def _get(obj, field):
    """Attribute access for Pydantic/dataclass objects and plain dicts alike."""
    if isinstance(obj, dict):
        return obj.get(field)
    return getattr(obj, field, None)


def _load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.loads(f.read()), "partial"


def _load_json(path: Path):
    with open(path) as f:
        data = json.load(f)
    # Wrap measurements list so field access is uniform
    measurements = []
    for m in data.get("measurements", []):
        measurements.append(SimpleNamespace(**m))
    data["measurements"] = measurements
    return data, "complete"


def validate(path: Path) -> bool:
    print(f"\n{'='*60}")
    print(f"Validating: {path}")

    suffix = path.suffix.lower()
    try:
        if suffix == ".pkl":
            run, run_kind = _load_pkl(path)
        elif suffix == ".json":
            run, run_kind = _load_json(path)
        else:
            print(f"  ERROR: unsupported file extension '{suffix}' (expected .pkl or .json)")
            return False
    except Exception as e:
        print(f"  ERROR: failed to load file: {e}")
        return False

    print(f"  Kind: {run_kind} run")
    passed = True

    # --- Fields always required ---
    for field in REQUIRED_RUN_FIELDS_ALWAYS:
        val = _get(run, field)
        if val is None:
            print(f"  FAIL  required field '{field}' is None")
            passed = False
        else:
            print(f"  OK    {field} = {val!r}")

    # --- Fields required only in complete (enriched) runs ---
    if run_kind == "complete":
        for field in REQUIRED_RUN_FIELDS_COMPLETE:
            val = _get(run, field)
            if val is None:
                print(f"  FAIL  required field '{field}' is None in complete run")
                passed = False
            else:
                print(f"  OK    {field} = {val!r}")

        # Validate device_info["sku"] is present
        device_info = _get(run, "device_info")
        if isinstance(device_info, dict):
            sku = device_info.get("sku")
            card_type = device_info.get("card_type")
            if not sku:
                print("  FAIL  device_info['sku'] is absent — SKU was not passed to enrichment script")
                passed = False
            else:
                print(f"  OK    device_info['sku'] = {sku!r}")
            if not card_type:
                print("  WARN  device_info['card_type'] is absent")
            else:
                print(f"  OK    device_info['card_type'] = {card_type!r}")

    else:
        # Partial run — device_info populated by enrichment step, absence is expected
        device_info = _get(run, "device_info")
        if device_info is None:
            print("  INFO  device_info is None in partial run (populated by enrichment step)")
        else:
            print(f"  OK    device_info = {device_info!r}")

    # --- Optional run-level fields (warnings only) ---
    for field in OPTIONAL_RUN_FIELDS:
        val = _get(run, field)
        if val is None:
            print(f"  WARN  optional field '{field}' is None")
        else:
            print(f"  OK    {field} = {val!r}")

    # --- run_type validation ---
    run_type = _get(run, "run_type")
    if run_type and run_type not in VALID_RUN_TYPES:
        print(f"  WARN  run_type '{run_type}' not in expected set {VALID_RUN_TYPES}")

    is_accuracy_run = run_type == "demo_accuracy"

    # --- Required measurements (perf vs accuracy runs differ) ---
    measurements = _get(run, "measurements") or []
    found_names = {m.name for m in measurements}

    if is_accuracy_run:
        # Pure accuracy runs (e.g. test_qwen_accuracy.py): only accuracy measurements required
        for name in ACCURACY_MEASUREMENTS:
            if name not in found_names:
                print(f"  FAIL  accuracy run missing measurement '{name}'")
                passed = False
            else:
                value = next(m.value for m in measurements if m.name == name)
                print(f"  OK    {name} = {value:.2f}%")
        # Warn (not fail) if standard perf measurements are also present — that's fine (e.g. Gemma-3)
        for name in REQUIRED_PERF_MEASUREMENTS:
            if name in found_names:
                value = next(m.value for m in measurements if m.name == name)
                print(f"  INFO  perf measurement '{name}' = {value} (present in accuracy run — OK)")
    else:
        # Perf / generate runs: standard decode throughput measurements required
        for name in REQUIRED_PERF_MEASUREMENTS:
            if name not in found_names:
                print(f"  FAIL  required measurement '{name}' not found")
                passed = False
            else:
                value = next(m.value for m in measurements if m.name == name)
                print(f"  OK    measurement '{name}' = {value}")

    if is_accuracy_run:
        # Warn if checkpoint measurements are present (fine for Gemma-3 which calls create_benchmark_data)
        if any(n.startswith("decode_latency_ms_token_") for n in found_names):
            print("  WARN  accuracy run contains perf checkpoint measurements (unexpected)")
        if "avg_decode_time_first_128" in found_names:
            print("  WARN  accuracy run contains avg_decode_time_first_128 (unexpected)")
    else:
        # Perf runs: checkpoints and avg_128 are expected
        has_any_checkpoint = any(n.startswith("decode_latency_ms_token_") for n in found_names)
        if not has_any_checkpoint:
            print("  WARN  no 'decode_latency_ms_token_*' checkpoint measurements found")
        else:
            checkpoints = sorted(n for n in found_names if n.startswith("decode_latency_ms_token_"))
            print(f"  OK    decode_latency_ms_token checkpoints: {checkpoints}")

        has_avg_128 = "avg_decode_time_first_128" in found_names
        if not has_avg_128:
            print("  WARN  'avg_decode_time_first_128' measurement not found")
        else:
            value = next(m.value for m in measurements if m.name == "avg_decode_time_first_128")
            print(f"  OK    avg_decode_time_first_128 = {value:.3f} ms")

        if ACCURACY_MEASUREMENTS.issubset(found_names):
            print("  WARN  perf run unexpectedly contains accuracy measurements (top1/top5)")

    print(f"  Total measurements: {len(measurements)}")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} partial_run_*.pkl|complete_run_*.json [...]")
        sys.exit(1)

    paths = [Path(p) for p in sys.argv[1:]]
    results = [validate(p) for p in paths]

    print(f"\n{'='*60}")
    print(f"Summary: {sum(results)}/{len(results)} files passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
