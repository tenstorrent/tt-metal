# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""POPE benchmark — object visual presence verification."""

from models.demos.qwen3_vl.evaluation.benchmarks.base import BaseBenchmark


class POPEBenchmark(BaseBenchmark):
    """POPE — Polling-based Object Probing Evaluation.

    Dataset: lmms-lab/POPE (test split)
    Metric: Accuracy (Yes/No binary classification)
    Reference score (Phi-3.5-vision): 86.1
    """

    @property
    def name(self) -> str:
        return "POPE"

    @property
    def metric_name(self) -> str:
        return "Accuracy"

    def load_dataset(self, num_samples=None):
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download

        parquet_files = [
            hf_hub_download("lmms-lab/POPE", f"data/test-{i:05d}-of-00003.parquet", repo_type="dataset")
            for i in range(3)
        ]
        ds = load_dataset("parquet", data_files=parquet_files, split="train")
        samples = list(ds)
        return samples[:num_samples] if num_samples else samples

    def build_messages(self, sample):
        image = sample["image"]
        question = sample["question"]
        prompt = f"{question}\nAnswer with Yes or No only."
        return [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]

    def postprocess_prediction(self, prediction, sample):
        pred = prediction.strip().lower()
        if pred.startswith("yes"):
            return "yes"
        if pred.startswith("no"):
            return "no"
        for word in pred.split():
            if word in ("yes", "no"):
                return word
        return pred

    def score_sample(self, prediction, sample):
        gt = sample["answer"].strip().lower()
        return 1.0 if prediction.strip().lower() == gt else 0.0
