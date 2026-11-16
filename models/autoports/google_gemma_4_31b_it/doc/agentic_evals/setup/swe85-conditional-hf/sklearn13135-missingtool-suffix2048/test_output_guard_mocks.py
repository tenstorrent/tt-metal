"""Host-only mocks for terminal output validation; imports no model/runtime code."""
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRIVER = HERE / "run_serial_candidate.py"
spec = importlib.util.spec_from_file_location("sklearn_suffix2048_driver_mock", DRIVER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def make_case(root, mode, invalid=False):
    ids = list(range(2048))
    chosen_ids = list(range(64))
    fixture_paths = []
    jobs = []
    output = root / "runs" / mode
    output.mkdir(parents=True)
    cfg_path = root / "precision.json"
    cfg = {"config_id": "stage12_head_bfp8_lofi"}
    cfg_path.write_text(json.dumps(cfg))
    for seed in (42, 43, 44):
        fixture = {
            "prompt_token_ids": ids,
            "continuations": {"chosen": chosen_ids},
            "metadata": {
                "seed": seed,
                "diagnostic_seed": seed,
                "diagnostic_mode": mode,
                "prompt_suffix_start_index": 15347,
                "prompt_suffix_end_index_exclusive": 17395,
                "prompt_suffix_token_ids_sha256": digest(ids),
            },
        }
        fp = root / f"fixture-{seed}.private.json"
        fp.write_text(json.dumps(fixture))
        fixture_paths.append(fp)
        name = f"case.seed{seed}"
        jobs.append(
            {
                "name": name,
                "case": str(fp.relative_to(root)),
                "free_run_tokens": 1024,
                "free_run_trials": 1,
                "branches": ["chosen"],
            }
        )
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(jobs))
    provenance = {"suffix_token_ids_sha256": digest(ids), "chosen_token_ids_sha256": digest(chosen_ids)}
    module.HERE = root
    module.ROOT = root
    (root / "prefix-provenance.json").write_text(json.dumps(provenance))
    (root / "admission.json").write_text("{}")
    (root / "forecast.json").write_text(
        json.dumps({"mode_estimates": {mode: {"primary_forecast": {"point_seconds": 1, "range_seconds": [0, 2]}}}})
    )
    for seed, fp in zip((42, 43, 44), fixture_paths):
        fixture = json.loads(fp.read_text())
        conditioning = fixture["prompt_token_ids"]
        if invalid and seed == 44:
            conditioning = conditioning[:-1]
        row = {
            "case": str(fp.relative_to(root)),
            "metadata": {"diagnostic_mode": mode, "diagnostic_seed": seed},
            "runtime_precision": cfg,
            "continuations": {"chosen": {"tokens": 64, "positions": [{}] * 64, "conditioning_token_ids": conditioning}},
            "free_running": [{"sampling": {"seed": seed}, "token_ids": [1]}],
        }
        (output / f"fixture-{seed}.json").write_text(json.dumps(row))
    (output / "load.json").write_text(json.dumps({"build_kwargs": {"precision_config": str(cfg_path)}}))
    spec_doc = {
        "manifest": str(manifest.relative_to(root)),
        "output_dir": str(output.relative_to(root)),
        "precision_config": str(cfg_path),
    }
    return output, {mode: spec_doc}


def test_success():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _, commands = make_case(root, "served")
        launch = root / "launch-served.json"
        launch.write_text("{}")
        rc = module.finalize_completion("served", 0, 1.25, launch, commands)
        completion = json.loads((root / "completion-served.json").read_text())
        assert rc == 0 and completion["status"] == "completed"
        assert completion["process_exit_code"] == 0 and completion["completed_unix"] > 0
        assert completion["load_sha256"] and len(completion["output_hashes"]) == 4
        assert set(completion["input_fingerprints"]) == {"42", "43", "44"}


def test_invalid_output_is_terminal_and_nonzero():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _, commands = make_case(root, "served", invalid=True)
        launch = root / "launch-served.json"
        launch.write_text("{}")
        rc = module.finalize_completion("served", 0, 1.25, launch, commands)
        completion = json.loads((root / "completion-served.json").read_text())
        assert rc != 0 and completion["status"] == "invalid_outputs"
        assert completion["process_exit_code"] == 0
        assert completion["output_validation_error"]
        assert completion["completed_unix"] > 0 and completion["output_hashes"]


if __name__ == "__main__":
    test_success()
    test_invalid_output_is_terminal_and_nonzero()
    print("host output guard mocks passed: success and invalid identity")
