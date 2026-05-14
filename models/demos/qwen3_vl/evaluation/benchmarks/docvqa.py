# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
from .base import BaseBenchmark
from ..metrics import anls


class DocVQABenchmark(BaseBenchmark):
    """DocVQA – Document Visual Question Answering.

    Dataset: HuggingFaceM4/DocumentVQA  (validation split, ~5349 examples)
    Metric: ANLS (Average Normalized Levenshtein Similarity)
    Reference score (Qwen3-VL-2B): 93.3
    """

    @property
    def name(self) -> str:
        return "DocVQA"

    @property
    def metric_name(self) -> str:
        return "ANLS"

    def load_dataset(self, num_samples=None):
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceM4/DocumentVQA", split="validation")
        samples = list(ds)
        return samples[:num_samples] if num_samples else samples

    def build_messages(self, sample):
        image = sample["image"]  # PIL.Image
        question = sample["question"]
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Answer the question using a single word or phrase.\n{question}"},
                ],
            }
        ]

    def score_sample(self, prediction, sample):
        ground_truths = sample["answers"]
        return anls(prediction, ground_truths)


class InfoVQABenchmark(BaseBenchmark):
    """InfoVQA – Infographics Visual QA.

    Dataset: LIME-DATA/infovqa  (train split, 1200 examples)
    Metric: ANLS
    Reference score (Qwen3-VL-2B): 72.4
    """

    @property
    def name(self) -> str:
        return "InfoVQA"

    @property
    def metric_name(self) -> str:
        return "ANLS"

    def load_dataset(self, num_samples=None):
        from datasets import load_dataset
        ds = load_dataset("LIME-DATA/infovqa", split="train")
        samples = list(ds)
        return samples[:num_samples] if num_samples else samples

    def build_messages(self, sample):
        image = sample["image"]
        question = sample["question"]
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Answer the question using a single word or phrase.\n{question}"},
                ],
            }
        ]

    def score_sample(self, prediction, sample):
        return anls(prediction, sample["answers"])
