# SPDX-License-Identifier: Apache-2.0
"""Serialize fresh selected-default validation and representative tracked checks."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("all", "full", "tracked"), default="all")
    args = parser.parse_args()
    model = Path(__file__).resolve().parents[1]
    stage = model / "doc/datatype_sweep"
    selected = json.loads((stage / "selected_precision_config.json").read_text())
    report = json.loads((stage / "sweep_results.json").read_text())
    assert report["final_selection"] and report["selected_config_id"] == selected["config_id"]
    package = "models.autoports.aleph_alpha_kolibri_1_bf16.tests"
    jobs = []
    if args.phase in ("all", "full"):
        output = stage / "selected_default"
        output.mkdir(parents=True, exist_ok=True)
        controls = {}
        for name in (
            "qualitative_prompts.json",
            "qualitative_prompt_format.json",
            "qualitative_hf.json",
            "reference_metadata.json",
        ):
            source = model / "doc/optimized_full_model" / name
            shutil.copyfile(source, output / name)
            controls[name] = dict(source=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        (output / "inherited_controls.json").write_text(json.dumps(controls, indent=2) + "\n")
        jobs.append(
            (
                output,
                "0",
                "run",
                "datatype_run",
                [
                    "--token-out",
                    "--token-out-repeats",
                    "5",
                    "--qualitative",
                    "--capability",
                    "--long-prefix",
                    "--batch-reconfigure",
                ],
            )
        )
        measured = [json.loads(path.read_text()) for path in (stage / "candidates").glob("*/result.json")]
        token_out_leader = max(
            (row for row in measured if row["status"] == "pass" and "token_out" in row),
            key=lambda row: row["token_out"]["decode_t_s_u"],
        )
        if token_out_leader["config_id"] != selected["config_id"]:
            jobs.append(
                (
                    stage / "token_out_rank_control",
                    "0",
                    "run",
                    "datatype_run",
                    [
                        "--config",
                        str(stage / "configs" / (token_out_leader["config_id"] + ".json")),
                        "--token-out",
                        "--token-out-repeats",
                        "5",
                    ],
                )
            )
    if args.phase in ("all", "tracked"):
        output = stage / "selected_tracked"
        jobs.extend(
            [
                (output, "1", "run", "datatype_run", ["--reduced", "--capability", "--batch-reconfigure"]),
                (output, "1", "trace_b1", "full_trace_contract", ["--capacity", "1048576"]),
                (output, "1", "trace_b32", "full_trace_contract", ["--batch", "32", "--capacity", "8192"]),
                (output, "1", "trace_b2", "full_trace_contract", ["--batch", "2", "--capacity", "8192"]),
                (output, "1", "trace_b31", "full_trace_contract", ["--batch", "31", "--capacity", "8192"]),
            ]
        )
    for output, tracking, label, module, flags in jobs:
        assert not (output / (label + ".command.json")).exists(), "Preserve previous evidence before an explicit rerun"
        env = dict(os.environ, FULL_ARTIFACT_DIR=str(output))
        env.pop("KOLIBRI_PRECISION_CONFIG", None)
        env["TT_METAL_TRACE_ALLOC_TRACKING"] = tracking
        env["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] = "0"
        command = [sys.executable, "-m", package + "." + module, *flags]
        print("SELECTED_VALIDATION_START", output.name, label, flush=True)
        subprocess.run([sys.executable, "-m", package + ".optimized_full_run", label, *command], env=env, check=True)
        print("SELECTED_VALIDATION_END", output.name, label, flush=True)


if __name__ == "__main__":
    main()
