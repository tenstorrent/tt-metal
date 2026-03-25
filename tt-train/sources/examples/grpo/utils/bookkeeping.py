from dataclasses import dataclass
import socket, getpass, subprocess, shutil, json
from datetime import datetime, timezone
import sys, logging
import csv
import sys
import os
import argparse
import ttnn
from safetensors.numpy import save_file
from ttml.common.utils import get_tt_metal_home


@dataclass
class RunContext:
    output_dir: str
    checkpoint_dir: str
    logger: logging.Logger

    def save_checkpoint(self, model, step: int) -> str:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        tensors = {}
        for name, param in model.parameters().items():
            tensors[name] = param.to_numpy(ttnn.DataType.FLOAT32)

        filepath = os.path.join(self.checkpoint_dir, f"grpo_step_{step}.safetensors")
        save_file(tensors, filepath)
        self.logger.info(f"Saved checkpoint ({len(tensors)} tensors) to {filepath}")
        return filepath


def setup_training_run() -> RunContext:
    args = _parse_args()
    repo_root = get_tt_metal_home()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = os.path.join(repo_root, "generated/tt-train/grpo_training_runs", f"{args.run_name}_{timestamp}")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    _archive_configs(output_dir, repo_root)
    _save_metadata(output_dir, args.run_name, timestamp, repo_root)
    logger = _setup_logging(output_dir)

    return RunContext(output_dir=output_dir, checkpoint_dir=checkpoint_dir, logger=logger)


def _parse_args(default_run_name: str = "grpo_training"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=default_run_name)
    args, _ = parser.parse_known_args()
    return args


def _archive_configs(output_dir: str, repo_root: str):
    shutil.copy2(__file__, output_dir)
    for cfg_dir in ["tt-train/configs/model_configs", "tt-train/configs/training_configs"]:
        src = os.path.join(repo_root, cfg_dir)
        dst = os.path.join(output_dir, os.path.basename(cfg_dir))
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)


def _save_metadata(output_dir: str, run_name: str, timestamp: str, repo_root: str):
    def _git(cmd):
        try:
            return subprocess.check_output(["git", "-C", repo_root] + cmd, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return "unknown"

    metadata = {
        "run_name": run_name,
        "timestamp_utc": timestamp,
        "started_at_human_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "host": socket.gethostname(),
        "user": getpass.getuser(),
        "repo_root": repo_root,
        "python": sys.executable,
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
    }
    with open(os.path.join(output_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def _setup_logging(output_dir: str, log_filename: str = "grpo_training.log"):
    log_path = os.path.join(output_dir, log_filename)

    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger("grpo")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class TrainingMetricsTracker:
    def __init__(self, run_dir: str):
        self.csv_path = os.path.join(run_dir, "metrics.csv")
        self._file = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=[
                "step",
                "batch",
                "mini_epoch",
                "reward_mean",
                "reward_std",
                "batch_elapsed_s",
                "lr",
                "timestamp",
            ],
        )

        self._writer.writeheader()

    def log_step(self, **kwargs):
        kwargs["timestamp"] = datetime.now(timezone.utc).isoformat()
        self._writer.writerow(kwargs)
        self._file.flush()

    def close(self):
        self._file.close()


def setup_accuracy_run() -> RunContext:
    args = _parse_args(default_run_name="grpo_model_accuracy")
    repo_root = get_tt_metal_home()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = os.path.join(repo_root, "generated/tt-train/grpo_model_accuracy_runs", f"{args.run_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    _archive_configs(output_dir, repo_root)
    _save_metadata(output_dir, args.run_name, timestamp, repo_root)
    logger = _setup_logging(output_dir, log_filename="grpo_model_accuracy.log")

    return RunContext(output_dir=output_dir, checkpoint_dir="", logger=logger)


class AccuracyMetricsTracker:
    def __init__(self, run_dir: str):
        self.csv_path = os.path.join(run_dir, "grpo_accuracy_results.csv")
        self._file = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=[
                "question_idx",
                "prompt",
                "correct",
                "golden_answer",
                "model_answer",
                "correct_so_far",
                "total_so_far",
                "running_accuracy",
                "timestamp",
            ],
        )
        self._writer.writeheader()

    def log_result(self, **kwargs):
        kwargs["timestamp"] = datetime.now(timezone.utc).isoformat()
        self._writer.writerow(kwargs)
        self._file.flush()

    def close(self):
        self._file.close()
