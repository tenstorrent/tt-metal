# tests/test_deepseek_token_accuracy.py
import os
from pathlib import Path

import pytest

from models.demos.deepseek_v3.demo.demo import run_demo

MODEL_PATH = Path(os.getenv("DEEPSEEK_V3_HF_MODEL", "/proj_sw/user_dev/deepseek-ai/DeepSeek-R1-0528"))
CACHE_DIR = Path(os.getenv("DEEPSEEK_V3_CACHE", "/proj_sw/user_dev/deepseek-v3-cache"))
REFERENCE_FILE = Path(os.getenv("DEEPSEEK_V3_REF", "models/demos/deepseek_v3/reference/example.refpt"))

MIN_TOP1 = float(os.getenv("MIN_TOP1", "0.80"))
MIN_TOP5 = float(os.getenv("MIN_TOP5", "0.95"))


@pytest.mark.skipif(not REFERENCE_FILE.exists(), reason="No reference .refpt/.pt found for token-accuracy test")
@pytest.mark.parametrize("repeat_batches", [1])
def test_teacher_forced_token_accuracy(repeat_batches):
    """
    Runs the demo in teacher-forced token-accuracy mode for a single synthesized prompt
    derived from the reference file. Checks that top-1/top-5 meet thresholds.
    """
    results = run_demo(
        prompts=None,
        model_path=str(MODEL_PATH),
        max_new_tokens=256,
        cache_dir=str(CACHE_DIR),
        random_weights=False,
        token_accuracy=True,
        reference_file=str(REFERENCE_FILE),
        tf_prompt_len=None,
        early_print_first_user=False,
        repeat_batches=repeat_batches,
    )

    # Expect one generation
    assert "generations" in results and len(results["generations"]) >= 1
    g0 = results["generations"][0]

    # Accuracy is reported on the first generation
    top1 = float(g0.get("accuracy_top1", 0.0))
    top5 = g0.get("accuracy_top5", None)

    assert top1 >= MIN_TOP1, f"Top-1 accuracy {top1:.3f} < {MIN_TOP1:.3f}"
    if top5 is not None:
        assert float(top5) >= MIN_TOP5, f"Top-5 accuracy {top5:.3f} < {MIN_TOP5:.3f}"
