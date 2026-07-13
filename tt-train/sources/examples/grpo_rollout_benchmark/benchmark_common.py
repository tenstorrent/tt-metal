# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared harness for the GRPO rollout benchmark.

Everything that must be identical across the two backends lives here: the model,
the BoolQ prompt/reward, the balanced GRPO config, and the CSV monitor. The
backends differ ONLY in the generation path (the completer), so any difference in
``gen_time_s`` / ``tok_per_s`` is attributable to generation, not the setup.

ttml/datasets imports are deliberately lazy (inside functions): the two-rank
``bench_ttt.py`` must pin FABRIC_2D before anything touches ttml/ttnn, and it
imports this module at the top.
"""

from __future__ import annotations

import csv
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Tuple

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

# Benchmark invariants. The generation batch per step is COMPLETIONS_PER_STEP
# completions (COMPLETIONS_PER_STEP // NUM_GENERATIONS prompts), each up to
# MAX_COMPLETION_LENGTH tokens. Holding these constant across device counts and
# backends is what makes gen_time_s comparable. COMPLETIONS_PER_STEP must divide
# by every supported ttml device count (1/2/4/8) and by NUM_GENERATIONS.
COMPLETIONS_PER_STEP = 32
NUM_GENERATIONS = 4
MAX_COMPLETION_LENGTH = 256
TEMPERATURE = 1.0
DATASET_SEED = 42

SYSTEM_PROMPT = (
    "Answer the question. Your answer should begin with either a Yes or a No. "
    "Then, explain why you answered Yes or No."
)

REPO_ROOT = Path(__file__).resolve().parents[4]


def prompts_per_step() -> int:
    return COMPLETIONS_PER_STEP // NUM_GENERATIONS


def benchmark_csv_path(backend: str, ttml_devices: int, ttt_devices: int) -> str:
    """Stable CSV path per (backend, device split). All --repeats append here."""
    name = f"grpo_bench_{backend}_{ttml_devices}x{ttt_devices}.csv"
    return str(REPO_ROOT / "generated" / "tt-train" / "grpo_bench" / name)


def build_boolq_dataset(tokenizer: Any) -> Any:
    from datasets import load_dataset

    def _format(example: dict) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {example['question']}? Context: {example['passage']}"},
        ]
        return {
            "prompt": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            "answer": "yes" if example["answer"] else "no",
        }

    return load_dataset("google/boolq", split="train").shuffle(seed=DATASET_SEED).map(_format)


def boolq_reward(completions: List[str], answer: List[str], **kwargs: Any) -> List[float]:
    rewards = []
    for text, ground_truth in zip(completions, answer):
        clean = text.strip().lower()
        accuracy = 2.0 if clean.startswith(ground_truth.lower()) else -1.0
        brevity = -0.1 * (len(text) / 20) ** 2
        rewards.append(accuracy + brevity)
    return rewards


def load_balanced_config(device_config_path: str, steps: int) -> Tuple[Any, Any, Any, dict, int]:
    """Load a device YAML and rewrite its grpo_config so the per-step generation
    workload is identical across device counts and the run stops after ``steps``.

    The ttml device count is read from the YAML mesh itself; per-device batch is
    balanced against it so every count generates the same COMPLETIONS_PER_STEP.

    Returns ``(device_config, grpo_config, transformer_config, optimizer_dict, ttml_devices)``.
    """
    from ttml.common.config import DeviceConfig, get_model_config, load_config
    from ttml.trainers import get_grpo_config

    if COMPLETIONS_PER_STEP % NUM_GENERATIONS:
        raise ValueError(
            f"COMPLETIONS_PER_STEP={COMPLETIONS_PER_STEP} is not divisible by NUM_GENERATIONS={NUM_GENERATIONS}"
        )

    raw = load_config(device_config_path)
    device_config = DeviceConfig(raw)
    ttml_devices = device_config.total_devices()
    if COMPLETIONS_PER_STEP % ttml_devices:
        raise ValueError(
            f"COMPLETIONS_PER_STEP={COMPLETIONS_PER_STEP} is not divisible by the YAML device count "
            f"{ttml_devices} ({device_config_path})"
        )

    # Pin the benchmark's generation workload; leave optimizer / model untouched.
    grpo = raw["training_config"]["grpo_config"]
    grpo.update(
        {
            "per_device_train_batch_size": COMPLETIONS_PER_STEP // ttml_devices,
            "gradient_accumulation_steps": 1,
            "num_generations": NUM_GENERATIONS,
            "max_completion_length": MAX_COMPLETION_LENGTH,
            "temperature": TEMPERATURE,
            "prompts_to_train": steps * prompts_per_step(),
            "checkpointing": False,
        }
    )

    output_dir = str(REPO_ROOT / "generated" / "tt-train" / "grpo_bench_runs" / _timestamp())
    grpo_config = get_grpo_config(raw, output_dir=output_dir)
    transformer_config = get_model_config(raw["training_config"]["model_config"])
    optimizer_dict = raw["training_config"]["optimizer"]
    return device_config, grpo_config, transformer_config, optimizer_dict, ttml_devices


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


class BenchmarkMonitor:
    """GRPO callback that appends per-step generation metrics to a CSV.

    Plain duck-typed callback (not a ttml TrainerCallback subclass) so importing
    this module never pulls in ttml — it implements exactly the hooks GRPOTrainer
    invokes. Every ``--repeats`` run of a (backend, split) appends to ONE csv,
    tagged by the leading ``run`` column.
    """

    _HEADER = [
        "run",
        "step",
        "backend",
        "ttml_devices",
        "ttt_devices",
        "reward",
        "avg_len",
        "gen_time_s",
        "gen_tokens",
        "tok_per_s",
        "step_time_s",
    ]

    def __init__(
        self,
        csv_path: str,
        *,
        backend: str,
        run_index: int,
        ttml_devices: int,
        ttt_devices: int,
    ) -> None:
        self.csv_path = csv_path
        self.backend = backend
        self.run_index = run_index
        self.ttml_devices = ttml_devices
        self.ttt_devices = ttt_devices
        self._gen_time_s: List[float] = []
        self._tok_per_s: List[float] = []
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(self._HEADER)

    def on_train_begin(self, trainer: Any) -> None:
        pass

    def on_before_optimizer_step(self, trainer: Any) -> None:
        pass

    def on_save(self, trainer: Any, step: int, path: str) -> None:
        pass

    def on_step_end(self, trainer: Any, step: int, *args: Any, **kwargs: Any) -> None:
        reward = float(kwargs.get("reward_mean", float("nan")))
        avg_len = float(kwargs.get("mean_completion_len", float("nan")))
        gen_time_s = float(kwargs.get("generation_time_s", float("nan")))
        step_time_s = float(kwargs.get("step_time_s", float("nan")))
        gen_tokens = COMPLETIONS_PER_STEP * avg_len
        tok_per_s = gen_tokens / gen_time_s if gen_time_s > 0 else float("nan")
        self._gen_time_s.append(gen_time_s)
        self._tok_per_s.append(tok_per_s)
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    self.run_index,
                    step,
                    self.backend,
                    self.ttml_devices,
                    self.ttt_devices,
                    f"{reward:.4f}",
                    f"{avg_len:.2f}",
                    f"{gen_time_s:.3f}",
                    f"{gen_tokens:.0f}",
                    f"{tok_per_s:.1f}",
                    f"{step_time_s:.3f}",
                ]
            )
        print(
            f"[bench:{self.backend} run{self.run_index}] step {step} | "
            f"gen {gen_time_s:.2f}s ({tok_per_s:.0f} tok/s) | step {step_time_s:.2f}s | reward {reward:.3f}",
            flush=True,
        )

    def on_train_end(self, trainer: Any) -> None:
        # Median over steps, excluding step 0 (device/trace warmup).
        gen = [g for g in self._gen_time_s[1:] if g == g]
        tps = [t for t in self._tok_per_s[1:] if t == t]
        if not gen:
            return
        print(
            f"[bench:{self.backend} run{self.run_index}] median gen_time_s={statistics.median(gen):.2f} "
            f"median tok/s={statistics.median(tps):.0f} over {len(gen)} steps (excl. warmup) -> {self.csv_path}",
            flush=True,
        )


class WeightSyncCallback:
    """Push fresh policy weights to the TTT generation worker every ``every``
    steps. Used only by the ttt backend; the caller does the initial push before
    ``trainer.train()``."""

    def __init__(self, completer: Any, every: int = 1) -> None:
        if every < 1:
            raise ValueError(f"WeightSyncCallback: 'every' must be >= 1 (got {every})")
        self.completer = completer
        self.every = every

    def on_train_begin(self, trainer: Any) -> None:
        pass

    def on_before_optimizer_step(self, trainer: Any) -> None:
        pass

    def on_step_end(self, trainer: Any, step: int, *args: Any, **kwargs: Any) -> None:
        if (step + 1) % self.every == 0:
            self.completer.push_weights()

    def on_save(self, trainer: Any, step: int, path: str) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        pass
