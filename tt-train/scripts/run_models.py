# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Run tt-train model binaries, capture logs, run analysis, and export to JSON.
This script implies tt-metal has already been built with mostly default options.
"""

import argparse
import os
import subprocess
import time
from pathlib import Path

import yaml
import git
import tt_train_metrics
import analyze_memory
import analyze_steps


def get_env(name: str, required=False) -> str:
    """Get an environment variable. Exit with error if missing and required is True."""
    value = os.environ.get(name)
    if required and not value:
        raise Exception(f"{name} is not set")
    return value


def get_git_commit_hash() -> str:
    """Return current git HEAD commit hash, or empty string if not in a repo."""
    try:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        return sha
    except Exception:
        return ""


def run_and_save_log(args: list[str], log_path: Path):
    """Run a command, writing stdout to log_path and to this process's stdout. Return exit code."""
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
            print(line, end="")
        proc.communicate()
        return proc.returncode


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    tt_metal_runtime_root = get_env("TT_METAL_RUNTIME_ROOT", required=True)
    parser = argparse.ArgumentParser(
        description="Run tt-train model binaries, capture logs, run analysis, and export to JSON."
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default=f"{tt_metal_runtime_root}/tt-train/scripts/run_models_config.yaml",
        help="Path to run_models_config.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_metrics",
        help="Directory for generated logs and JSON (default: generated_metrics)",
    )
    return parser.parse_args()


def main() -> int:
    parsed_args = parse_args()
    model_config = parsed_args.model_config
    output_dir = Path(parsed_args.output_dir)

    # Check for required environment variables
    tt_metal_runtime_root = Path(get_env("TT_METAL_RUNTIME_ROOT", required=True))
    # Turn off tt-logger to reduce log noise
    os.environ["TT_LOGGER_LEVEL"] = "off"

    # Common parent directories
    tt_train_path = tt_metal_runtime_root / "tt-train"
    build_examples = tt_metal_runtime_root / "build" / "tt-train" / "sources" / "examples"

    # Quick sanity checks
    if not tt_train_path.is_dir():
        raise Exception(f"{str(tt_train_path)} does not exist or not a directory")
    if not build_examples.is_dir():
        raise Exception(f"{str(build_examples)} does not exist or not a directory")

    # Create output directory to store metrics
    output_dir.mkdir(parents=True, exist_ok=True)

    # Report failing models
    failing_models = []

    with open(model_config) as f:
        models = yaml.safe_load(f)

    # Run each model from config: execute binary, analyze logs, write metrics to JSON
    for model in models["models"]:
        model_name = model["name"]
        model_filename = model["filename"]
        binary = str(build_examples / model["binary"] / model["binary"])
        args = [arg.replace("{TT_METAL_RUNTIME_ROOT}", str(tt_metal_runtime_root)) for arg in model["args"]]

        # Microseconds since epoch (same as shell: date +%s%N | cut -b1-16)
        current_time = int(time.time_ns() // 1_000)

        log_basename = f"{model_filename}_memory_analysis_{current_time}"
        log_path = output_dir / f"{log_basename}.log"

        cmd = [binary] + args
        print(f"Running {model_filename}: {binary} {args}")
        ret_code = run_and_save_log(cmd, log_path)

        # Record failing model run but continue to run remaining models
        if ret_code != 0:
            failing_models.append(str(log_path))
            print(f"Subprocess {binary} failed. Return code {ret_code}")
            continue

        # Extract memory and step metrics from the run log
        memory_data = analyze_memory.main(["--logs", str(log_path)])
        print(memory_data)

        step_data = analyze_steps.main(["--logs", str(log_path)])
        print(step_data)

        if all(x is not None for x in (memory_data, step_data)):
            git_commit_hash = get_git_commit_hash()

            # Build metrics payload and write JSON alongside the log
            pydantic_data = tt_train_metrics.TtTrainMetricsData(
                test_ts=current_time,
                model_name=model_name,
                model_filename=model_filename,
                binary_name=binary,
                args=" ".join(args),
                git_commit_hash=git_commit_hash,
                model_dram_mb=memory_data["model"],
                optimizer_dram_mb=memory_data["optimizer"],
                activations_dram_mb=memory_data["activations"],
                gradients_dram_mb=memory_data["gradients_overhead"],
                unaccounted_dram_mb=memory_data["other"],
                total_dram_mb=memory_data["total"],
                device_memory_mb=memory_data["device_memory"],
                last_loss=step_data["last_loss"],
                average_iteration_time_ms=step_data["average_iteration_time_ms"],
            )
            print(pydantic_data)

            output_filename = output_dir / log_path.with_suffix(".json").name
            tt_train_metrics.write_json(pydantic_data, output_filename)

    if failing_models:
        # Fail the run if any model binary exited non-zero
        raise Exception(
            "{} model(s) failed with error code != 0. Check the following logs: \n{}".format(
                len(failing_models), "\n".join(failing_models)
            )
        )


if __name__ == "__main__":
    main()
