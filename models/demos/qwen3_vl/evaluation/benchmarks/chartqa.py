# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
from .base import BaseBenchmark
from ..metrics import relaxed_accuracy


class ChartQABenchmark(BaseBenchmark):
    """ChartQA – Chart Understanding.

    Dataset: HuggingFaceM4/ChartQA  (test split, 2500 examples)
    Metric: Relaxed accuracy (±5% tolerance for numeric answers)
    Reference score (Qwen3-VL-2B): 72.8
    """

    @property
    def name(self) -> str:
        return "ChartQA"

    @property
    def metric_name(self) -> str:
        return "RelaxedAcc"

    def load_dataset(self, num_samples=None):
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceM4/ChartQA", split="test")
        samples = list(ds)
        return samples[:num_samples] if num_samples else samples

    def build_messages(self, sample):
        image = sample["image"]
        question = sample["query"]
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            "Answer the question with a short phrase or number. "
                            "Do not explain.\n" + question
                        ),
                    },
                ],
            }
        ]

    def score_sample(self, prediction, sample):
        ground_truth = sample["label"]
        # ground_truth may be a list or a string
        gts = ground_truth if isinstance(ground_truth, list) else [ground_truth]
        return relaxed_accuracy(prediction, [str(g) for g in gts])
