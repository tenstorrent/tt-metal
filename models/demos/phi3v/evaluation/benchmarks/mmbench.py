# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MMBench benchmark — general multimodal understanding."""

from models.demos.qwen3_vl.evaluation.benchmarks.base import BaseBenchmark
from models.demos.qwen3_vl.evaluation.metrics import extract_mcq_answer


class MMBenchBenchmark(BaseBenchmark):
    """MMBench — General Multimodal Understanding.

    Dataset: lmms-lab/MMBench (dev_en split)
    Metric: Accuracy (MCQ A/B/C/D)
    Reference score (Phi-3.5-vision): 81.9
    """

    @property
    def name(self) -> str:
        return "MMBench"

    @property
    def metric_name(self) -> str:
        return "Accuracy"

    def load_dataset(self, num_samples=None):
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download

        parquet_file = hf_hub_download("lmms-lab/MMBench", "en/dev-00000-of-00001.parquet", repo_type="dataset")
        ds = load_dataset("parquet", data_files=parquet_file, split="train")
        samples = list(ds)
        return samples[:num_samples] if num_samples else samples

    def build_messages(self, sample):
        image = sample["image"]
        question = sample["question"]
        choices = []
        for key in ["A", "B", "C", "D"]:
            val = sample.get(key, "")
            if val:
                choices.append(val)
        choice_text = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
        prompt = f"{question}\n{choice_text}\nAnswer with the letter of the correct option only."
        return [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]

    def postprocess_prediction(self, prediction, sample):
        choices = [sample.get(k, "") for k in ["A", "B", "C", "D"] if sample.get(k)]
        return extract_mcq_answer(prediction, choices)

    def score_sample(self, prediction, sample):
        gt = sample["answer"].strip().upper()
        return 1.0 if prediction.strip().upper() == gt else 0.0
