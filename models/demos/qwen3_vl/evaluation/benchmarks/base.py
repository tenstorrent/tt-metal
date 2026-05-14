# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Base class for VQA benchmarks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import json
import time
from pathlib import Path
from loguru import logger


@dataclass
class BenchmarkResult:
    name: str
    metric: str
    score: float
    num_samples: int
    elapsed_sec: float
    per_sample: list[dict] = field(default_factory=list)

    def __str__(self):
        return (
            f"{self.name}: {self.metric}={self.score*100:.1f}%  "
            f"({self.num_samples} samples, {self.elapsed_sec:.1f}s)"
        )


class BaseBenchmark(ABC):
    """Abstract base benchmark.

    Subclasses implement:
      - name / metric_name properties
      - load_dataset() -> list of samples
      - build_messages(sample) -> list[dict]  (HF-style chat messages)
      - score_sample(prediction, sample) -> float  (0.0 – 1.0)
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def metric_name(self) -> str: ...

    @abstractmethod
    def load_dataset(self, num_samples: int | None = None) -> list[dict]:
        """Return list of sample dicts."""
        ...

    @abstractmethod
    def build_messages(self, sample: dict) -> list[dict]:
        """Convert sample to HF-style messages for the model."""
        ...

    @abstractmethod
    def score_sample(self, prediction: str, sample: dict) -> float:
        """Return per-sample score in [0, 1]."""
        ...

    def postprocess_prediction(self, prediction: str, sample: dict) -> str:
        """Optional: clean/extract the answer from raw model output."""
        return prediction

    def run(self, runner, num_samples: int | None = None, max_new_tokens: int = 50) -> BenchmarkResult:
        """Run full evaluation loop.

        Args:
            runner: Qwen3VL2BRunner instance (already set up)
            num_samples: Evaluate only the first N samples (None = all)
            max_new_tokens: Max tokens to generate per sample
        """
        samples = self.load_dataset(num_samples)
        logger.info(f"[{self.name}] Evaluating {len(samples)} samples …")

        scores = []
        per_sample = []
        t0 = time.time()

        for i, sample in enumerate(samples):
            messages = self.build_messages(sample)
            try:
                raw = runner.generate(messages, max_new_tokens=max_new_tokens)
            except Exception as e:
                logger.warning(f"[{self.name}] sample {i} failed: {e}")
                raw = ""

            pred = self.postprocess_prediction(raw, sample)
            score = self.score_sample(pred, sample)
            scores.append(score)

            if (i + 1) % 50 == 0 or i == 0:
                running_avg = sum(scores) / len(scores)
                logger.info(
                    f"[{self.name}] {i+1}/{len(samples)}  "
                    f"running {self.metric_name}={running_avg*100:.1f}%"
                )

            per_sample.append({"idx": i, "pred": pred, "score": score})

        elapsed = time.time() - t0
        final_score = sum(scores) / len(scores) if scores else 0.0
        result = BenchmarkResult(
            name=self.name,
            metric=self.metric_name,
            score=final_score,
            num_samples=len(samples),
            elapsed_sec=elapsed,
            per_sample=per_sample,
        )
        logger.info(f"[{self.name}] FINAL: {result}")
        return result

    def save_results(self, result: BenchmarkResult, output_dir: str):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = Path(output_dir) / f"{self.name}.json"
        with open(path, "w") as f:
            json.dump(
                {
                    "name": result.name,
                    "metric": result.metric,
                    "score": result.score,
                    "num_samples": result.num_samples,
                    "elapsed_sec": result.elapsed_sec,
                    "per_sample": result.per_sample,
                },
                f,
                indent=2,
            )
        logger.info(f"Saved results to {path}")
