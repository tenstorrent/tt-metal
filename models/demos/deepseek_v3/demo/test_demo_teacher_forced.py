# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest
import torch

from models.demos.deepseek_v3.demo.demo import run_demo

MODEL_PATH = Path(
    os.getenv(
        "DEEPSEEK_V3_HF_MODEL",
        "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528",
    )
)
CACHE_DIR = Path(
    os.getenv(
        "DEEPSEEK_V3_CACHE",
        "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache",
    )
)

# Must match the path used in generate_teacher_forced_file.py
REFERENCE_FILE = Path(__file__).with_name("deepseek_v3_teacher_forcing.refpt")


@pytest.mark.parametrize("reference_file", [REFERENCE_FILE])
def test_demo_teacher_forcing_accuracy(reference_file: Path):
    """
    Test DeepSeek v3 demo with teacher forcing to verify accuracy.

    This test assumes the reference file has been generated beforehand by
    running `generate_teacher_forced_file.py`:

        python generate_teacher_forced_file.py

    Phase 1 (offline, in generate_teacher_forced_file.py):
      - Run HuggingFace model to generate tokens and save as reference.
      - Saves: reference_tokens (prompt + generated), top5_tokens, tf_prompt_len

    Phase 2 (this test):
      - Run TT model with teacher forcing using the reference tokens.
      - Compares TT model's predicted tokens against reference's generated tokens.

    The comparison:
      - TokenAccuracy splits reference_tokens at tf_prompt_len into prompt and gt_tokens.
      - gt_tokens are the generated tokens from the reference file.
      - TT model predictions are compared against gt_tokens for accuracy.

    Expected accuracy: 100% if prefill is deterministic (all tokens should match).
    """

    if not REFERENCE_FILE.exists():
        pytest.fail(
            f"Reference file not found at {REFERENCE_FILE}. "
            "Generate it first by running "
            "`python generate_teacher_forced_file.py`."
        )

    payload = torch.load(REFERENCE_FILE)
    assert "reference_tokens" in payload, "Reference file missing 'reference_tokens'"
    assert "tf_prompt_len" in payload, "Reference file missing 'tf_prompt_len'"

    reference_tokens = payload["reference_tokens"]
    tf_prompt_len = int(payload["tf_prompt_len"])
    saved_max_new_tokens = int(payload.get("max_new_tokens", 32))

    # Verify reference file structure: reference_tokens should be [1, T] where T = prompt_len + generated_len
    assert reference_tokens.dim() == 2, f"Expected reference_tokens to be 2D [1, T], got shape {reference_tokens.shape}"
    assert reference_tokens.shape[0] == 1, f"Expected batch size 1, got {reference_tokens.shape[0]}"
    total_ref_tokens = reference_tokens.shape[1]
    assert (
        total_ref_tokens > tf_prompt_len
    ), f"Total tokens ({total_ref_tokens}) should be > prompt_len ({tf_prompt_len})"

    # Verify there are generated tokens to compare against
    num_generated_ref = total_ref_tokens - tf_prompt_len
    print(
        f"Reference file: {tf_prompt_len} prompt tokens + {num_generated_ref} generated tokens = {total_ref_tokens} total"
    )

    # Use the saved max_new_tokens from the reference file
    max_new_tokens = saved_max_new_tokens

    total_tokens = int(reference_tokens.shape[-1])
    print(f"\n=== Phase 2: Run teacher forcing ===")
    print(f"Loaded reference from: {REFERENCE_FILE}")
    print(f"Total reference tokens: {total_tokens}, prompt length: {tf_prompt_len}")
    print(f"Using max_new_tokens={max_new_tokens}")

    # Run the demo with teacher forcing
    results = run_demo(
        prompts=None,
        model_path=MODEL_PATH,
        cache_dir=CACHE_DIR,
        random_weights=False,
        max_new_tokens=max_new_tokens,
        repeat_batches=1,
        token_accuracy=True,
        reference_file=REFERENCE_FILE,
        tf_prompt_len=tf_prompt_len,
    )

    # Check results
    assert "generations" in results
    assert len(results["generations"]) > 0

    first_gen = results["generations"][0]

    assert "accuracy_top1" in first_gen, "Top-1 accuracy should be present in results"
    assert first_gen["accuracy_top1"] is not None, "Top-1 accuracy should not be None"

    top1_acc = first_gen["accuracy_top1"]
    assert isinstance(top1_acc, (int, float)), "Top-1 accuracy should be a number"
    assert 0.0 <= top1_acc <= 1.0, f"Top-1 accuracy should be between 0 and 1, got {top1_acc}"

    print(f"\nTop-1 accuracy: {top1_acc:.4f} ({top1_acc * 100:.2f}%)")

    # Verify tokens were generated
    assert "tokens" in first_gen
    generated_tokens = first_gen["tokens"]
    assert len(generated_tokens) > 0, "Should generate at least some tokens"

    # Verify we're comparing against the right number of tokens
    # TokenAccuracy compares TT model predictions against reference's gt_tokens (generated tokens)
    print(f"TT model generated {len(generated_tokens)} tokens")
    print(f"Comparing against {num_generated_ref} reference generated tokens")

    min_expected_accuracy = 1.0
    assert top1_acc >= min_expected_accuracy, (
        f"Top-1 accuracy {top1_acc:.4f} is below minimum expected "
        f"{min_expected_accuracy:.2f}. This may indicate a bug in teacher forcing "
        f"or model non-determinism. Generated {len(generated_tokens)} tokens, "
        f"reference has {num_generated_ref} generated tokens."
    )

    print(f"\n=== Teacher forcing test PASSED with {top1_acc * 100:.2f}% accuracy ===")
