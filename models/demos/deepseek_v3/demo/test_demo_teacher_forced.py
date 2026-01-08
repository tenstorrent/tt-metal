# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest
import torch

from models.demos.deepseek_v3.demo.demo import run_demo
from models.demos.deepseek_v3.utils.hf_model_utils import load_tokenizer

MODEL_PATH = Path(
    os.getenv(
        "DEEPSEEK_V3_HF_MODEL",
        "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528",
    )
)
CACHE_DIR = Path(
    os.getenv(
        "DEEPSEEK_V3_CACHE",
        "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/dev/",
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
      - Saves: reference_tokens, prompt_tokens, generated_tokens, top5_tokens, tf_prompt_len

    Phase 2 (this test):
      - Run TT model with teacher forcing using the reference tokens.
      - Compares TT model's predicted tokens against HF model's top5_tokens.

    Accuracy comparison (top5_tokens alignment):
      - reference_tokens[0, i] is the *actual* token at position i
      - top5_tokens[i] is HF model's top-5 prediction for token at position i,
        given context tokens [0..i-1]
      - top5_tokens[0] is zeros (no prediction for the first token)
      - TT model's prediction at position i is compared against top5_tokens[i]

    Expected accuracy: ~100% if TT model matches HF model behavior.
    """

    if not REFERENCE_FILE.exists():
        pytest.fail(
            f"Reference file not found at {REFERENCE_FILE}. "
            "Generate it first by running "
            "`python generate_teacher_forced_file.py`."
        )

    payload = torch.load(REFERENCE_FILE, weights_only=False)
    assert "reference_tokens" in payload, "Reference file missing 'reference_tokens'"
    assert "prompt_tokens" in payload, "Reference file missing 'prompt_tokens'"
    assert "generated_tokens" in payload, "Reference file missing 'generated_tokens'"
    assert "top5_tokens" in payload, "Reference file missing 'top5_tokens'"
    assert "tf_prompt_len" in payload, "Reference file missing 'tf_prompt_len'"

    reference_tokens = payload["reference_tokens"]  # [1, L] full sequence
    prompt_tokens = payload["prompt_tokens"]  # [1, prompt_len]
    generated_tokens = payload["generated_tokens"]  # [1, gen_len]
    top5_tokens = payload["top5_tokens"]  # [L, 5] HF model's predictions
    tf_prompt_len = int(payload["tf_prompt_len"])
    saved_max_new_tokens = int(payload.get("max_new_tokens", 32))

    # Print token ID metadata if available
    if "token_ids_meta" in payload:
        meta = payload["token_ids_meta"]
        print(f"Token ID metadata: bos={meta.get('bos_id')}, eos={meta.get('eos_id')}, pad={meta.get('pad_id')}")

    # Verify reference file structure
    assert reference_tokens.dim() == 2, f"Expected reference_tokens to be 2D [1, T], got shape {reference_tokens.shape}"
    assert reference_tokens.shape[0] == 1, f"Expected batch size 1, got {reference_tokens.shape[0]}"
    total_ref_tokens = reference_tokens.shape[1]

    # Verify prompt + generated = total
    prompt_len = prompt_tokens.shape[1] if prompt_tokens.dim() == 2 else len(prompt_tokens)
    gen_len = generated_tokens.shape[1] if generated_tokens.dim() == 2 else len(generated_tokens)
    assert prompt_len == tf_prompt_len, f"prompt_tokens length {prompt_len} != tf_prompt_len {tf_prompt_len}"
    assert (
        prompt_len + gen_len == total_ref_tokens
    ), f"prompt ({prompt_len}) + generated ({gen_len}) != total ({total_ref_tokens})"

    print(f"\nReference file structure:")
    print(f"  prompt_tokens: {prompt_len} tokens")
    print(f"  generated_tokens: {gen_len} tokens")
    print(f"  reference_tokens: {total_ref_tokens} tokens (prompt + generated)")
    print(f"  top5_tokens: {tuple(top5_tokens.shape)} (HF model predictions)")

    # Use the saved max_new_tokens from the reference file
    max_new_tokens = saved_max_new_tokens

    print(f"\n=== Phase 2: Run teacher forcing ===")
    print(f"Loaded reference from: {REFERENCE_FILE}")
    print(f"Total reference tokens: {total_ref_tokens}, prompt length: {tf_prompt_len}")
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

    # Decode and print for comparison
    tokenizer = load_tokenizer(MODEL_PATH)

    # Get tokens from payload (already properly shaped)
    prompt_ids = prompt_tokens[0] if prompt_tokens.dim() == 2 else prompt_tokens
    ref_gen_ids = generated_tokens[0] if generated_tokens.dim() == 2 else generated_tokens

    # When teacher forcing is on:
    # - 'tokens' are the FORCED (ground-truth) tokens fed back into TT decode (should match HF generated_tokens)
    # - 'predicted_tokens' are TT's raw predictions before forcing (used for accuracy vs HF top5_tokens)
    tt_forced_ids = torch.tensor(first_gen["tokens"])
    tt_pred_ids = torch.tensor(first_gen.get("predicted_tokens", first_gen["tokens"]))

    prompt_text = tokenizer.decode(prompt_ids.tolist(), skip_special_tokens=False)
    ref_text = tokenizer.decode(ref_gen_ids.tolist(), skip_special_tokens=False)
    tt_forced_text = tokenizer.decode(tt_forced_ids.tolist(), skip_special_tokens=False)
    tt_pred_text = tokenizer.decode(tt_pred_ids.tolist(), skip_special_tokens=False)

    print("\n\n=== Comparison ===")
    print(f"Prompt Text:\n{prompt_text}\n")
    print(f"HF Reference Output (Tokens: {len(ref_gen_ids)}):\n{ref_text}\n")
    print(f"HF Reference Token IDs:\n{ref_gen_ids.tolist()}\n")
    print(f"TT Forced Output (Tokens: {len(tt_forced_ids)}):\n{tt_forced_text}\n")
    print(f"TT Forced Token IDs:\n{tt_forced_ids.tolist()}\n")
    print(f"TT Predicted Output (Tokens: {len(tt_pred_ids)}):\n{tt_pred_text}\n")
    print(f"TT Predicted Token IDs:\n{tt_pred_ids.tolist()}\n")

    # Show HF top5 predictions for first few generated positions
    print("HF Model Top-5 Predictions (for generated token positions):")
    for i in range(min(5, gen_len)):
        pos = tf_prompt_len + i  # position in full sequence
        hf_top5 = top5_tokens[pos].tolist()
        actual_token = ref_gen_ids[i].item()
        hf_top5_decoded = [tokenizer.decode([t], skip_special_tokens=False) for t in hf_top5]
        actual_decoded = tokenizer.decode([actual_token], skip_special_tokens=False)
        in_top5 = "✓" if actual_token in hf_top5 else "✗"
        print(f"  Position {pos}: top5={hf_top5_decoded}, actual={actual_decoded!r} (id={actual_token}) {in_top5}")
    print("==================\n")

    # check top1 accuracy
    assert "accuracy_top1" in first_gen, "Top-1 accuracy should be present in results"
    assert first_gen["accuracy_top1"] is not None, "Top-1 accuracy should not be None"
    top1_acc = first_gen["accuracy_top1"]
    assert isinstance(top1_acc, (int, float)), "Top-1 accuracy should be a number"
    assert 0.0 <= top1_acc <= 1.0, f"Top-1 accuracy should be between 0 and 1, got {top1_acc}"
    print(f"\nTop-1 accuracy: {top1_acc:.4f} ({top1_acc * 100:.2f}%)")

    # check top5 accuracy
    assert "accuracy_top5" in first_gen, "Top-5 accuracy should be present in results"
    assert first_gen["accuracy_top5"] is not None, "Top-5 accuracy should not be None"
    top5_acc = first_gen["accuracy_top5"]
    assert isinstance(top5_acc, (int, float)), "Top-5 accuracy should be a number"
    assert 0.0 <= top5_acc <= 1.0, f"Top-5 accuracy should be between 0 and 1, got {top5_acc}"
    print(f"Top-5 accuracy: {top5_acc:.4f} ({top5_acc * 100:.2f}%)")

    # Verify tokens were generated
    assert "tokens" in first_gen
    tt_generated_tokens = first_gen["tokens"]
    assert len(tt_generated_tokens) > 0, "Should generate at least some tokens"

    # Teacher forcing correctness check: forced tokens must exactly equal the HF reference generated tokens.
    expected_forced = ref_gen_ids.tolist()
    assert tt_generated_tokens == expected_forced, (
        "Teacher forcing failed: TT forced tokens do not match HF reference generated tokens.\n"
        f"First 20 expected: {expected_forced[:20]}\n"
        f"First 20 got     : {tt_generated_tokens[:20]}"
    )

    if "predicted_tokens" in first_gen:
        assert len(first_gen["predicted_tokens"]) == len(
            tt_generated_tokens
        ), "predicted_tokens length should match forced token length under teacher forcing"

    # Compare the number of generated TT model tokens against the number of generated reference model tokens
    print(f"\nTT model generated {len(tt_generated_tokens)} tokens")
    print(f"Comparing against {gen_len} HF reference generated tokens")

    # Token-by-token comparison: TT predictions vs HF top5
    if "predicted_tokens" in first_gen:
        tt_preds = first_gen["predicted_tokens"]
        matches_top1 = 0
        matches_top5 = 0
        print("\nDetailed token comparison (TT prediction vs HF top5):")
        total_compared = min(len(tt_preds), gen_len)
        for i in range(total_compared):
            pos = tf_prompt_len + i
            tt_pred = tt_preds[i]
            hf_top5 = top5_tokens[pos].tolist()
            hf_top1 = hf_top5[0]
            is_top1 = tt_pred == hf_top1
            is_top5 = tt_pred in hf_top5
            if is_top1:
                matches_top1 += 1
            if is_top5:
                matches_top5 += 1
            status = "✓ top1" if is_top1 else ("✓ top5" if is_top5 else "✗")
            if i < 10 or not is_top5:  # Print first 10 or any mismatches
                tt_decoded = tokenizer.decode([tt_pred], skip_special_tokens=False)
                hf_top1_decoded = tokenizer.decode([hf_top1], skip_special_tokens=False)
                print(
                    f"  [{i:3d}] TT={tt_pred:6d} ({tt_decoded!r}), HF top1={hf_top1:6d} ({hf_top1_decoded!r}) {status}"
                )

        computed_top1 = matches_top1 / total_compared if total_compared else 0
        computed_top5 = matches_top5 / total_compared if total_compared else 0
        print(f"\nComputed accuracy from predictions:")
        print(f"  Top-1: {matches_top1}/{total_compared} = {computed_top1:.4f} ({computed_top1 * 100:.2f}%)")
        print(f"  Top-5: {matches_top5}/{total_compared} = {computed_top5:.4f} ({computed_top5 * 100:.2f}%)")

    min_expected_accuracy = 0.90
    assert top1_acc >= min_expected_accuracy, (
        f"Top-1 accuracy {top1_acc:.4f} is below minimum expected "
        f"{min_expected_accuracy:.2f}. This may indicate a bug in teacher forcing "
        f"or model non-determinism. Generated {len(tt_generated_tokens)} tokens, "
        f"reference has {gen_len} generated tokens."
    )

    assert top5_acc >= min_expected_accuracy, (
        f"Top-5 accuracy {top5_acc:.4f} is below minimum expected "
        f"{min_expected_accuracy:.2f}. This may indicate a bug in teacher forcing "
        f"or model non-determinism. Generated {len(tt_generated_tokens)} tokens, "
        f"reference has {gen_len} generated tokens."
    )

    print(f"\n=== Teacher forcing test PASSED ===")
    print(f"Top-1 accuracy: {top1_acc:.4f} ({top1_acc * 100:.2f}%)")
    print(f"Top-5 accuracy: {top5_acc:.4f} ({top5_acc * 100:.2f}%)")
