# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ScienceQA benchmark — visual scientific knowledge reasoning."""

from models.demos.qwen3_vl.evaluation.benchmarks.base import BaseBenchmark
from models.demos.qwen3_vl.evaluation.metrics import extract_mcq_answer


class ScienceQABenchmark(BaseBenchmark):
    """ScienceQA — Visual Scientific Knowledge Reasoning.

    Dataset: derek-thomas/ScienceQA (test split, image subset only)
    Metric: Accuracy (multiple choice)
    Reference score (Phi-3.5-vision): 91.3
    """

    @property
    def name(self) -> str:
        return "ScienceQA"

    @property
    def metric_name(self) -> str:
        return "Accuracy"

    def load_dataset(self, num_samples=None):
        from datasets import load_dataset

        ds = load_dataset("derek-thomas/ScienceQA", split="test")
        samples = [s for s in ds if s.get("image") is not None]
        return samples[:num_samples] if num_samples else samples

    def build_messages(self, sample):
        image = sample["image"]
        question = sample["question"]
        choices = sample["choices"]
        choice_text = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
        prompt = f"{question}\n{choice_text}\nAnswer with the letter of the correct option only."
        return [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]

    def postprocess_prediction(self, prediction, sample):
        choices = sample["choices"]
        return extract_mcq_answer(prediction, choices, num_choices=len(choices))

    def score_sample(self, prediction, sample):
        gt_idx = sample["answer"]
        gt_letter = chr(65 + gt_idx)
        return 1.0 if prediction.strip().upper() == gt_letter else 0.0
