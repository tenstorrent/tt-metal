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

# Must match the path used in generate_deepseek_v3_teacher_forcing_reference.py
REFERENCE_FILE = Path(__file__).with_name("deepseek_v3_teacher_forcing.refpt")


@pytest.mark.parametrize("max_new_tokens", [32])
def test_demo_teacher_forcing_accuracy(max_new_tokens: int):
    """
    Test DeepSeek v3 demo with teacher forcing to verify accuracy.

    This test assumes the reference file has been generated beforehand by
    running `generate_deepseek_v3_teacher_forcing_reference.py`:

        python generate_deepseek_v3_teacher_forcing_reference.py

    Phase 1 (offline, in the generator script):
      - Run model normally to generate tokens and save them as reference.

    Phase 2 (this test):
      - Use teacher forcing with the saved reference tokens.

    After fixing the generator to NOT force the prefill output, the contexts align:
      - Phase 1: prefill→A, decode(A)→B, decode(B)→C, ...
      - Phase 2: prefill→A', decode(A')→B', decode(B_forced)→C', ...

    If A' = A (deterministic prefill), then B' = B. And since we force B for
    decode 1, C' should equal C. All subsequent predictions should match.

    Expected accuracy: >80% (only the first token may differ if there is
    prefill non-determinism). Uses only 5 layers (override_num_layers=5)
    for faster CI execution.
    """

    if not REFERENCE_FILE.exists():
        pytest.fail(
            f"Reference file not found at {REFERENCE_FILE}. "
            "Generate it first by running "
            "`python generate_deepseek_v3_teacher_forcing_reference.py`."
        )

    payload = torch.load(REFERENCE_FILE)
    assert "reference_tokens" in payload, "Reference file missing 'reference_tokens'"

    reference_tokens = payload["reference_tokens"]
    tf_prompt_len = int(payload["tf_prompt_len"])
    saved_max_new_tokens = int(payload.get("max_new_tokens", max_new_tokens))

    # Keep the test parameter as the default, but prefer the value used to
    # generate the reference (for strict consistency).
    max_new_tokens = saved_max_new_tokens

    total_tokens = int(reference_tokens.shape[-1])
    print(f"\n=== Phase 2: Run teacher forcing ===")
    print(f"Loaded reference from: {REFERENCE_FILE}")
    print(f"Total reference tokens: {total_tokens}, prompt length: {tf_prompt_len}")
    print(f"Using max_new_tokens={max_new_tokens}")

    results = run_demo(
        prompts=None,
        model_path=MODEL_PATH,
        cache_dir=CACHE_DIR,
        random_weights=False,
        max_new_tokens=max_new_tokens,
        override_num_layers=5,
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
    assert len(first_gen["tokens"]) > 0, "Should generate at least some tokens"

    # With the generator fix (no forcing after prefill), we expect high accuracy:
    #   - If prefill is deterministic: A' = A, so all predictions should match.
    #   - If prefill has some variance: only first prediction might differ.
    #   - All subsequent predictions use forced inputs, so they should match.
    #
    # With 5 layers there may be some numerical non-determinism, so we allow
    # a threshold, but expect significantly better than the old ~68%.
    min_expected_accuracy = 0.8  # Expect at least 80% accuracy
    assert top1_acc >= min_expected_accuracy, (
        f"Top-1 accuracy {top1_acc:.4f} is below minimum expected "
        f"{min_expected_accuracy:.2f}. This may indicate a bug in teacher forcing "
        f"or model non-determinism."
    )

    print(f"\n=== Teacher forcing test PASSED with {top1_acc * 100:.2f}% accuracy ===")
