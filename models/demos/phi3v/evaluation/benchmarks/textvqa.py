# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TextVQA benchmark — reading text in images."""

from models.demos.qwen3_vl.evaluation.benchmarks.base import BaseBenchmark
from models.demos.qwen3_vl.evaluation.metrics import vqa_accuracy


class TextVQABenchmark(BaseBenchmark):
    """TextVQA — Reading Text in Images.

    Dataset: facebook/textvqa (validation split, ~5000 examples)
    Metric: VQA accuracy (min(matching_annotators/3, 1.0))
    Reference score (Phi-3.5-vision): 72.0
    """

    @property
    def name(self) -> str:
        return "TextVQA"

    @property
    def metric_name(self) -> str:
        return "VQA Acc"

    def load_dataset(self, num_samples=None):
        from datasets import load_dataset

        ds = load_dataset("facebook/textvqa", split="validation")
        samples = list(ds)
        return samples[:num_samples] if num_samples else samples

    def build_messages(self, sample):
        image = sample["image"]
        question = sample["question"]
        prompt = f"{question}\nAnswer with a short phrase or single word."
        return [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]

    def score_sample(self, prediction, sample):
        answers = sample["answers"]
        return vqa_accuracy(prediction, answers)
