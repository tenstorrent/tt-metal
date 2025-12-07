# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from models.demos.deepseek_v3.demo.demo import run_demo

MODEL_PATH = Path(os.getenv("DEEPSEEK_V3_HF_MODEL", "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528"))
CACHE_DIR = Path(os.getenv("DEEPSEEK_V3_CACHE", "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache"))


@pytest.mark.parametrize("max_new_tokens", [32])
def test_demo_teacher_forcing_accuracy(max_new_tokens):
    """
    Test DeepSeek v3 demo with teacher forcing to verify accuracy.

    This test uses a two-phase approach:
    1. Phase 1: Generate tokens normally with the TT model (creates reference)
    2. Phase 2: Use teacher forcing with those tokens as ground truth

    After fixing the generator to NOT force the prefill output, the contexts align:
    - Phase 1: prefill→A, decode(A)→B, decode(B)→C, ...
    - Phase 2: prefill→A', decode(A')→B', decode(B_forced)→C', ...

    If A'=A (deterministic prefill), then B'=B. And since we force B for decode 1,
    C' should equal C. All subsequent predictions should match.

    Expected accuracy: >90% if model is deterministic (only first token may differ
    if there's any prefill non-determinism).

    Uses only 5 layers (override_num_layers=5) for faster CI execution.
    """
    # Use a simple test prompt
    test_prompt = "What is the capital of France? Please provide a brief answer."

    print(f"\n=== Phase 1: Generate reference tokens with TT model ===")

    # Phase 1: Run model normally to generate reference tokens
    results_phase1 = run_demo(
        prompts=[test_prompt],
        model_path=MODEL_PATH,
        cache_dir=CACHE_DIR,
        random_weights=False,
        max_new_tokens=max_new_tokens,
        override_num_layers=5,
        repeat_batches=1,
        token_accuracy=False,  # Normal generation, no teacher forcing
    )

    assert "generations" in results_phase1
    assert len(results_phase1["generations"]) > 0

    generated_tokens = results_phase1["generations"][0]["tokens"]
    print(f"Phase 1: Generated {len(generated_tokens)} tokens")

    # Create reference file
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    raw_prompt_tokens = tokenizer.encode(test_prompt, add_special_tokens=False)

    # gt_tokens = Phase 1's decode outputs
    # With the generator fix, teacher forcing starts at decode step 0:
    # - Decode 0: input is natural prefill output, predict and collect, force gt[0]
    # - Decode 1: input is gt[0], predict and collect, force gt[1]
    # - ...
    full_sequence = raw_prompt_tokens + generated_tokens
    reference_tokens_tensor = torch.tensor(full_sequence, dtype=torch.long).unsqueeze(0)

    temp_reference_file = Path("/tmp/deepseek_v3_teacher_forcing_test.refpt")
    temp_reference_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"reference_tokens": reference_tokens_tensor}, temp_reference_file)

    print(f"Phase 2: Created reference file with {len(full_sequence)} tokens")
    print(f"  raw_prompt: {len(raw_prompt_tokens)}, generated: {len(generated_tokens)}")

    print(f"\n=== Phase 2: Run teacher forcing ===")

    # Phase 2: Run with teacher forcing
    results_phase2 = run_demo(
        prompts=None,
        model_path=MODEL_PATH,
        cache_dir=CACHE_DIR,
        random_weights=False,
        max_new_tokens=max_new_tokens,
        override_num_layers=5,
        repeat_batches=1,
        token_accuracy=True,
        reference_file=temp_reference_file,
        tf_prompt_len=len(raw_prompt_tokens),
    )

    # Cleanup temp file
    if temp_reference_file.exists():
        temp_reference_file.unlink()

    # Check results
    assert "generations" in results_phase2
    assert len(results_phase2["generations"]) > 0

    first_gen = results_phase2["generations"][0]

    assert "accuracy_top1" in first_gen, "Top-1 accuracy should be present in results"
    assert first_gen["accuracy_top1"] is not None, "Top-1 accuracy should not be None"

    top1_acc = first_gen["accuracy_top1"]
    assert isinstance(top1_acc, (int, float)), "Top-1 accuracy should be a number"
    assert 0.0 <= top1_acc <= 1.0, f"Top-1 accuracy should be between 0 and 1, got {top1_acc}"

    print(f"\nTop-1 accuracy: {top1_acc:.4f} ({top1_acc*100:.2f}%)")

    # Verify tokens were generated
    assert "tokens" in first_gen
    assert len(first_gen["tokens"]) > 0, "Should generate at least some tokens"

    # With the generator fix (no forcing after prefill), we expect high accuracy:
    # - If prefill is deterministic: A' = A, so all predictions should match
    # - If prefill has some variance: only first prediction might differ
    # - All subsequent predictions use forced inputs, so they should match
    #
    # With 5 layers there may be some numerical non-determinism, so we allow
    # a lower threshold, but expect significantly better than the old 68%.
    min_expected_accuracy = 1  # Expect at least 80% accuracy
    assert top1_acc >= min_expected_accuracy, (
        f"Top-1 accuracy {top1_acc:.4f} is below minimum expected {min_expected_accuracy}. "
        f"This may indicate a bug in teacher forcing or model non-determinism."
    )

    print(f"\n=== Teacher forcing test PASSED with {top1_acc*100:.2f}% accuracy ===")
