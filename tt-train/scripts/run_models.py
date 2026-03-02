# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Run tt-train model binaries, capture logs, run memory analysis, and annotate CSV with metadata.
Python equivalent of run_models.sh.
"""

import os
import subprocess
import sys
import time
from pathlib import Path
import git
import process_logs
import analyze_memory


def require_env(name: str) -> str:
    """Require an environment variable to be set; exit with error if missing."""
    value = os.environ.get(name)
    if not value:
        print(f"{name} is not set", file=sys.stderr)
        sys.exit(1)
    return value


def get_git_commit_hash() -> str:
    """Return current git HEAD commit hash, or empty string if not in a repo."""
    try:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        return sha
    except:
        return ""


def run_with_tee(args: list[str], log_path: Path) -> int:
    """Run a command, writing stdout to log_path and to this process's stdout. Return exit code."""
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
            print(line, end="")
        proc.wait()
        return proc.returncode


def prepend_metadata_to_csv(
    csv_path: Path, metadata_header: str, metadata_row: str
) -> None:
    """Prepend metadata columns to the first two lines of the CSV file."""
    with open(csv_path) as f:
        lines = f.readlines()
    if len(lines) < 2:
        return
    lines[0] = metadata_header + lines[0]
    lines[1] = metadata_row + lines[1]
    with open(csv_path, "w") as f:
        f.writelines(lines)


def main() -> None:
    tt_metal_home = require_env("TT_METAL_HOME")
    require_env("TT_METAL_RUNTIME_ROOT")

    os.environ["TT_LOGGER_LEVEL"] = "off"

    tt_train = Path(tt_metal_home) / "tt-train"
    build_examples = Path(tt_metal_home) / "build" / "tt-train" / "sources" / "examples"
    nano_gpt_bin = build_examples / "nano_gpt" / "nano_gpt"
    linear_regression_tp_dp_bin = (
        build_examples / "linear_regression_tp_dp" / "linear_regression_tp_dp"
    )

    # (model_name, model_filename, binary_path, args)
    models = [
        (
            "NanoGPT Shakespeare",
            "nanogpt",
            str(nano_gpt_bin),
            [
                "-c",
                f"{tt_train}/configs/training_configs/training_shakespeare_nanogpt.yaml",
            ],
        ),
        # ("NanoLlama3 Shakespeare", "nanollama", str(nano_gpt_bin),
        #  f"-c {tt_train}/configs/training_configs/training_shakespeare_nanollama3.yaml"),
        # ("Linear Regression TP+DP", "linear_regression_tp_dp", str(linear_regression_tp_dp_bin),
        #  "--mesh_shape=8x4"),
    ]

    metadata_header = "date,model_name,model_filename,binary_name,args,git_commit_hash,"

    for model_name, model_filename, binary, args in models:
        # Microseconds since epoch (same as shell: date +%s%N | cut -b1-16)
        current_time = int(time.time() * 1_000_000)

        out_basename = f"{model_filename}_memory_analysis_{current_time}"
        log_path = Path(f"{out_basename}.log")

        cmd = [binary] + args
        print(f"Running {model_filename}: {binary} {args}")
        run_with_tee(cmd, log_path)

        data = analyze_memory._main(["--logs", str(log_path)])
        if data is not None:
            git_commit_hash = get_git_commit_hash()

            run_metadata = process_logs.RunMetadata(
                test_ts=current_time,
                model_name=model_name,
                model_filename=model_filename,
                binary_name=binary,
                args=" ".join(args),
                git_commit_hash=git_commit_hash,
            )
            print(run_metadata)

            memory_tracking_data = process_logs.MemoryTracking(
                metadata=run_metadata,
                model_dram_mb=data["model"],
                optimizer_dram_mb=data["optimizer"],
                activations_dram_mb=data["activations"],
                gradients_dram_mb=data["gradients_overhead"],
                unaccounted_dram_mb=data["other"],
                total_dram_mb=data["total"],
                device_memory_mb=data["device_memory"],
            )
            print(memory_tracking_data)

            log_filename = log_path.stem
            process_logs.write_csv(memory_tracking_data, log_filename)


if __name__ == "__main__":
    main()
