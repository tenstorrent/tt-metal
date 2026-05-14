# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multiple-choice VQA benchmarks: MMMU, MMStar, MathVista, AI2D, RealWorldQA, OCRBench."""

from .base import BaseBenchmark
from ..metrics import extract_mcq_answer, exact_match, anls


class MMMUBenchmark(BaseBenchmark):
    """MMMU – Massive Multidiscipline Multimodal Understanding.

    Dataset: MMMU/MMMU  (validation split, 900 examples)
    Metric: Accuracy (multiple choice A/B/C/D)
    Reference score (Qwen3-VL-2B): 53.4
    """

    @property
    def name(self) -> str:
        return "MMMU"

    @property
    def metric_name(self) -> str:
        return "Accuracy"

    def load_dataset(self, num_samples=None):
        from datasets import load_dataset, concatenate_datasets
        # MMMU has per-subject splits; load all validation subjects
        subjects = [
            "Accounting", "Agriculture", "Architecture_and_Engineering", "Art",
            "Art_Theory", "Basic_Medical_Science", "Biology", "Chemistry",
            "Clinical_Medicine", "Computer_Science", "Design", "Diagnostics_and_Laboratory_Medicine",
            "Economics", "Electronics", "Energy_and_Power", "Finance", "Geography",
            "History", "Literature", "Management", "Marketing", "Materials_Science",
            "Math", "Mechanical_Engineering", "Music", "Pharmacy", "Physics",
            "Psychology", "Public_Health", "Sociology",
        ]
        splits = []
        for subj in subjects:
            try:
                ds = load_dataset("MMMU/MMMU", subj, split="validation", ignore_verifications=True)
                splits.append(ds)
            except Exception:
                pass
        if not splits:
            raise RuntimeError("Could not load MMMU dataset.")
        combined = concatenate_datasets(splits)
        samples = list(combined)
        return samples[:num_samples] if num_samples else samples

    def build_messages(self, sample):
        question = sample["question"]
        choices = [sample.get(f"option_{c}", "") for c in ["A", "B", "C", "D"] if sample.get(f"option_{c}")]
        choice_text = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))

        content = []
        # Attach all images
        for img_key in ["image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7"]:
            if sample.get(img_key) is not None:
                content.append({"type": "image", "image": sample[img_key]})

        content.append({
            "type": "text",
            "text": f"{question}\n{choice_text}\nAnswer with the letter of the correct option only.",
        })
        return [{"role": "user", "content": content}]

    def postprocess_prediction(self, prediction, sample):
        choices = [sample.get(f"option_{c}", "") for c in ["A", "B", "C", "D"] if sample.get(f"option_{c}")]
        return extract_mcq_answer(prediction, choices)

    def score_sample(self, prediction, sample):
        gt = sample["answer"].strip().upper()
        return 1.0 if prediction.strip().upper() == gt else 0.0


class MMStarBenchmark(BaseBenchmark):
    """MMStar – General VQA with 6 core categories.

    Dataset: Lin-Chen/MMStar  (val split, 1500 examples)
    Metric: Accuracy
    Reference score (Qwen3-VL-2B): 58.3
    """

    @property
    def name(self) -> str:
        return "MMStar"

    @property
    def metric_name(self) -> str:
        return "Accuracy"

    def load_dataset(self, num_samples=None):
        from io import BytesIO
        from PIL import Image
        from huggingface_hub import hf_hub_download
        from datasets import load_dataset

        path = hf_hub_download("Lin-Chen/MMStar", "mmstar.parquet", repo_type="dataset")
        ds = load_dataset("parquet", data_files=path, split="train")
        samples = []
        for row in ds:
            img_data = row["image"]
            if isinstance(img_data, bytes):
                img_data = Image.open(BytesIO(img_data)).convert("RGB")
            samples.append({**row, "image": img_data})
        return samples[:num_samples] if num_samples else samples

    def build_messages(self, sample):
        image = sample["image"]
        question = sample["question"]
        prompt = question + "\nAnswer with the letter of the correct option only."
        return [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]

    def postprocess_prediction(self, prediction, sample):
        return extract_mcq_answer(prediction)

    def score_sample(self, prediction, sample):
        gt = str(sample["answer"]).strip().upper()
        return 1.0 if prediction.strip().upper() == gt else 0.0


class MathVistaBenchmark(BaseBenchmark):
    """MathVista – Mathematical Reasoning in Visual Contexts.

    Dataset: AI4Math/MathVista  (testmini split, 1000 examples)
    Metric: Accuracy
    Reference score (Qwen3-VL-2B): 61.3
    """

    @property
    def name(self) -> str:
        return "MathVista"

    @property
    def metric_name(self) -> str:
        return "Accuracy"

    def load_dataset(self, num_samples=None):
        from datasets import load_dataset
        ds = load_dataset("AI4Math/MathVista", split="testmini")
        samples = list(ds)
        return samples[:num_samples] if num_samples else samples

    def build_messages(self, sample):
        # Use decoded_image (PIL) instead of image (URL string)
        image = sample.get("decoded_image") or sample.get("image")
        question = sample.get("query") or sample.get("question", "")
        question_type = sample.get("question_type", "free_form")

        if question_type == "multi_choice":
            choices = sample.get("choices", [])
            choice_text = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
            prompt = f"{question}\n{choice_text}\nAnswer with the letter of the correct option only."
        else:
            prompt = f"{question}\nAnswer with a short phrase or number only."

        return [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]

    def postprocess_prediction(self, prediction, sample):
        if sample.get("question_type") == "multi_choice":
            choices = sample.get("choices", [])
            return extract_mcq_answer(prediction, choices, num_choices=len(choices))
        return prediction

    def score_sample(self, prediction, sample):
        from ..metrics import relaxed_accuracy
        gt = str(sample.get("answer", "")).strip()
        if sample.get("question_type") == "multi_choice":
            choices = sample.get("choices", [])
            pred_upper = prediction.strip().upper()
            # Fix: map letter → choice text, then compare to gt (which is the text, not the letter)
            if len(pred_upper) == 1 and ord('A') <= ord(pred_upper) <= ord('A') + len(choices) - 1:
                idx = ord(pred_upper) - ord('A')
                pred_text = str(choices[idx]).strip()
                return 1.0 if pred_text.lower() == gt.lower() else 0.0
            # Fallback: direct text comparison (model output the text directly)
            return 1.0 if pred_upper == gt.upper() else 0.0
        return relaxed_accuracy(prediction, [gt])


class AI2DBenchmark(BaseBenchmark):
    """AI2D – Science Diagram Understanding.

    Dataset: lmms-lab/ai2d  (test split, ~3088 examples)
    Metric: Accuracy
    Reference score (Qwen3-VL-2B): 76.9
    """

    @property
    def name(self) -> str:
        return "AI2D"

    @property
    def metric_name(self) -> str:
        return "Accuracy"

    def load_dataset(self, num_samples=None):
        from datasets import load_dataset
        ds = load_dataset("lmms-lab/ai2d", split="test")
        samples = list(ds)
        return samples[:num_samples] if num_samples else samples

    def build_messages(self, sample):
        image = sample["image"]
        question = sample["question"]
        # options is a list of choice strings
        choices = sample.get("options", sample.get("choices", []))
        choice_text = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
        prompt = f"{question}\n{choice_text}\nAnswer with the letter of the correct option only."
        return [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]

    def postprocess_prediction(self, prediction, sample):
        choices = sample.get("options", sample.get("choices", []))
        return extract_mcq_answer(prediction, choices)

    def score_sample(self, prediction, sample):
        # answer is a numeric string "0","1","2","3" (index into options)
        raw_answer = str(sample.get("answer", "")).strip()
        try:
            gt_idx = int(raw_answer)
            choices = sample.get("options", sample.get("choices", []))
            gt_letter = chr(65 + gt_idx) if 0 <= gt_idx < len(choices) else raw_answer.upper()
        except ValueError:
            gt_letter = raw_answer.upper()
        return 1.0 if prediction.strip().upper() == gt_letter else 0.0


class RealWorldQABenchmark(BaseBenchmark):
    """RealWorldQA – Real-World Understanding.

    Dataset: xai-org/RealworldQA  (test split, 765 examples)
    Metric: Accuracy
    Reference score (Qwen3-VL-2B): 63.9
    """

    @property
    def name(self) -> str:
        return "RealWorldQA"

    @property
    def metric_name(self) -> str:
        return "Accuracy"

    def load_dataset(self, num_samples=None):
        from datasets import load_dataset
        ds = load_dataset("xai-org/RealworldQA", split="test")
        samples = list(ds)
        return samples[:num_samples] if num_samples else samples

    def build_messages(self, sample):
        image = sample["image"]
        # Question already contains the choices (e.g., "A. xxx\nB. yyy") embedded in the text
        question = sample["question"]
        prompt = question + "\nAnswer with only the letter or value (e.g. A, B, Yes, No)."
        return [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]

    def postprocess_prediction(self, prediction, sample):
        gt = str(sample.get("answer", "")).strip()
        # If gt is a single letter A-E, try to extract letter from prediction
        if len(gt) == 1 and gt.upper() in "ABCDE":
            return extract_mcq_answer(prediction)
        return prediction.strip()

    def score_sample(self, prediction, sample):
        gt = str(sample.get("answer", "")).strip()
        pred = prediction.strip()
        if gt.upper() == pred.upper():
            return 1.0
        # Fuzzy: gt is single letter, pred might be longer
        if len(gt) == 1 and pred.upper().startswith(gt.upper()):
            return 1.0
        return 0.0


class OCRBenchBenchmark(BaseBenchmark):
    """OCRBench – OCR Understanding (score out of 1000).

    Dataset: echo840/OCRBench  (test split, 1000 examples)
    Metric: Score (0–1000); reported as a raw integer in the original paper
    Reference score (Qwen3-VL-2B): 881 (= 88.1%)
    """

    @property
    def name(self) -> str:
        return "OCRBench"

    @property
    def metric_name(self) -> str:
        return "Score(×1000)"

    def load_dataset(self, num_samples=None):
        from datasets import load_dataset
        ds = load_dataset("echo840/OCRBench", split="test")
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
        # answer field is a list of strings
        answers = sample.get("answer", [])
        if isinstance(answers, str):
            answers = [answers]
        return anls(prediction, [str(a) for a in answers], threshold=0.5)
