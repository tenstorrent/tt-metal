# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.demos.deepseek_v3.demo.demo import run_demo
from models.demos.deepseek_v3.tt.generator import MAX_SEQ_LEN as GENERATOR_MAX_SEQ_LEN
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW
from models.demos.deepseek_v3.utils.hf_model_utils import load_tokenizer
from models.demos.deepseek_v3.utils.test_utils import system_name_to_mesh_shape

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
# REFERENCE_FILE = Path(__file__).with_name("deepseek_v3_teacher_forcing.refpt")
REFERENCE_FILE = Path(__file__).with_name("gpqa_diamond_racemic.refpt")


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("reference_file", [REFERENCE_FILE])
@pytest.mark.parametrize("max_new_tokens", [128, 2048, 8192], ids=["128", "2048", "8192"])
def test_demo_teacher_forcing_accuracy(reference_file: Path, max_new_tokens: int, is_ci_env: bool):
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

    if is_ci_env and max_new_tokens != 128:
        pytest.skip("CI runs only the 128-token teacher forcing test to keep runtime manageable.")

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
    saved_max_new_tokens = int(payload.get("max_new_tokens"))

    max_supported_new_tokens = GENERATOR_MAX_SEQ_LEN - tf_prompt_len
    if max_supported_new_tokens <= 0:
        pytest.skip(f"Prompt length {tf_prompt_len} exceeds max_seq_len {GENERATOR_MAX_SEQ_LEN}.")
    if max_new_tokens > max_supported_new_tokens:
        pytest.skip(
            f"Requested max_new_tokens={max_new_tokens} exceeds generator capacity: "
            f"max_seq_len={GENERATOR_MAX_SEQ_LEN}, prompt_len={tf_prompt_len} -> max_new_tokens<={max_supported_new_tokens}."
        )

    requested_system_name = os.getenv("MESH_DEVICE")
    if requested_system_name is None:
        pytest.fail("Environment variable $MESH_DEVICE is not set. Please set it to DUAL, QUAD, TG, or T3K.")
    mesh_shape = system_name_to_mesh_shape(requested_system_name.upper())
    num_users = USERS_PER_ROW * mesh_shape[0]
    prompt_text_for_users = payload.get("prompt", "")

    # Print token ID metadata if available
    if "token_ids_meta" in payload:
        meta = payload["token_ids_meta"]
        logger.info(
            "Token ID metadata: bos={}, eos={}, pad={}",
            meta.get("bos_id"),
            meta.get("eos_id"),
            meta.get("pad_id"),
        )

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

    logger.info("Reference file structure:")
    logger.info("  prompt_tokens: {} tokens", prompt_len)
    logger.info("  generated_tokens: {} tokens", gen_len)
    logger.info("  reference_tokens: {} tokens (prompt + generated)", total_ref_tokens)
    logger.info("  top5_tokens: {} (HF model predictions)", tuple(top5_tokens.shape))

    # Ensure the requested length is available in the reference payload.
    available_gen_tokens = generated_tokens[0].shape[-1]
    if max_new_tokens > saved_max_new_tokens:
        pytest.fail(
            f"Requested max_new_tokens={max_new_tokens} exceeds reference file max_new_tokens={saved_max_new_tokens} "
            f"({REFERENCE_FILE}). Regenerate the reference with a larger max_new_tokens."
        )
    if max_new_tokens > available_gen_tokens:
        pytest.fail(
            f"Requested max_new_tokens={max_new_tokens} exceeds available generated tokens={available_gen_tokens} "
            f"in {REFERENCE_FILE}. Regenerate the reference with a larger max_new_tokens."
        )

    logger.info("=== Phase 2: Run teacher forcing ===")
    logger.info("Loaded reference from: {}", REFERENCE_FILE)
    logger.info("Total reference tokens: {}, prompt length: {}", total_ref_tokens, tf_prompt_len)
    logger.info("Using max_new_tokens={}", max_new_tokens)

    # Run the demo with teacher forcing
    results = run_demo(
        prompts=[prompt_text_for_users] * num_users,
        model_path=MODEL_PATH,
        cache_dir=CACHE_DIR,
        random_weights=False,
        max_new_tokens=max_new_tokens,
        repeat_batches=1,
        token_accuracy=True,
        reference_file=REFERENCE_FILE,
        tf_prompt_len=tf_prompt_len,
        enable_trace=True,
    )

    # Check results
    assert "generations" in results
    assert len(results["generations"]) == num_users, (
        f"Expected {num_users} generations (USERS_PER_ROW={USERS_PER_ROW}, rows={mesh_shape[0]}), "
        f"got {len(results['generations'])}"
    )

    first_gen = results["generations"][0]

    # Decode and print for comparison
    tokenizer = load_tokenizer(MODEL_PATH)

    # Get tokens from payload (already properly shaped)
    reference_ids = reference_tokens[0] if reference_tokens.dim() == 2 else reference_tokens
    ref_gen_ids = generated_tokens[0] if generated_tokens.dim() == 2 else generated_tokens

    # When teacher forcing is on:
    # - 'tokens' are the FORCED (ground-truth) tokens fed back into TT decode (should match HF generated_tokens)
    # - 'predicted_tokens' are TT's raw predictions before forcing (used for accuracy vs HF top5_tokens)

    # check accuracy is present
    assert "accuracy_top1" in first_gen, "Top-1 accuracy should be present in results"
    assert "accuracy_top5" in first_gen, "Top-5 accuracy should be present in results"

    # Verify tokens were generated
    assert "tokens" in first_gen
    tt_generated_tokens = first_gen["tokens"]
    assert len(tt_generated_tokens) > 0, "Should generate at least some tokens"

    # Teacher forcing correctness check: forced tokens must exactly equal the HF reference generated tokens.
    expected_forced = ref_gen_ids.tolist()[:max_new_tokens]
    assert tt_generated_tokens == expected_forced, (
        "Teacher forcing failed: TT forced tokens do not match HF reference generated tokens.\n"
        f"First 20 expected: {expected_forced[:20]}\n"
        f"First 20 got     : {tt_generated_tokens[:20]}"
    )

    expected_tokens = results["generations"][0]["tokens"]
    for idx, gen in enumerate(results["generations"][1:], start=1):
        tokens = gen["tokens"]
        if tokens != expected_tokens:
            first_diff = next((i for i, (a, b) in enumerate(zip(expected_tokens, tokens)) if a != b), None)
            if first_diff is None:
                detail = f"length mismatch: user0={len(expected_tokens)}, user{idx}={len(tokens)}"
            else:
                detail = (
                    f"first mismatch at token {first_diff}: "
                    f"user0={expected_tokens[first_diff]}, user{idx}={tokens[first_diff]}"
                )
            pytest.xfail(f"User outputs diverged across batch (issue #35509). {detail}")

    if "predicted_tokens" in first_gen:
        assert len(first_gen["predicted_tokens"]) == len(
            tt_generated_tokens
        ), "predicted_tokens length should match forced token length under teacher forcing"

    # Compare the number of generated TT model tokens against the number of generated reference model tokens
    logger.info("TT model generated {} tokens", len(tt_generated_tokens))
    logger.info("Comparing against {} HF reference generated tokens", min(gen_len, max_new_tokens))

    # Token-by-token comparison: TT predictions vs HF top5
    assert "predicted_tokens" in first_gen, "predicted_tokens missing; token_accuracy must provide predictions"
    tt_preds = first_gen["predicted_tokens"]
    total_compared = min(len(tt_preds), gen_len, max_new_tokens)
    assert total_compared > 0, "No tokens to compare"

    logger.info(f"{'Progress':<15}{'Correct':<8}{'True':<15}{'Actual':<15}{'Top 5 Predictions':<75}")
    logger.info("-" * 113)

    top1_correct = []
    top5_correct = []
    errors = []

    sanitize = lambda x: repr(x)[1:-1]  # Use repr() and remove the outer quotes

    for i in range(total_compared):
        pos = tf_prompt_len + i
        tt_pred = int(tt_preds[i])
        hf_top5 = top5_tokens[pos].tolist()
        hf_top1 = hf_top5[0]
        true_token = int(reference_ids[pos].item())

        top1_match = tt_pred == hf_top1
        top5_match = tt_pred in hf_top5
        true_match = tt_pred == true_token

        top1_correct.append(top1_match)
        top5_correct.append(top5_match)

        if not top5_match:
            context_start = max(0, pos - 9)
            context_tokens = reference_ids[context_start:pos]
            context_text = tokenizer.decode(context_tokens.tolist(), skip_special_tokens=False)
            incorrect_token = tokenizer.decode([tt_pred], skip_special_tokens=False)
            expected_tokens = [tokenizer.decode([t], skip_special_tokens=False) for t in hf_top5]
            errors.append(
                {
                    "position": pos,
                    "context": context_text,
                    "incorrect": incorrect_token,
                    "expected": expected_tokens,
                    "predicted_id": tt_pred,
                    "expected_ids": hf_top5,
                    "true_id": true_token,
                }
            )

        true_text = sanitize(tokenizer.decode([true_token], skip_special_tokens=False))
        tt_text = sanitize(tokenizer.decode([tt_pred], skip_special_tokens=False))
        ref_top5_text = [tokenizer.decode([t], skip_special_tokens=False) for t in hf_top5]
        ref_top5_str = " ".join(f"{sanitize(t):<14}" for t in ref_top5_text)

        progress_str = f"{i+1}/{total_compared}"
        correct = "x" if top1_match else ("-" if top5_match else ("!" if true_match else " "))
        logger.info(f"{progress_str:<15}{correct:<8}{true_text:<15}{tt_text:<15}{ref_top5_str}")

    # Compute accuracies over every 100 tokens
    num_tokens = len(top1_correct)
    num_segments = (num_tokens + 99) // 100
    for seg in range(num_segments):
        start = seg * 100
        end = min(start + 100, num_tokens)
        seg_top1 = 100 * sum(top1_correct[start:end]) / (end - start)
        seg_top5 = 100 * sum(top5_correct[start:end]) / (end - start)
        max_width = len(str(total_compared))
        logger.info(
            f"Tokens {start:{max_width}d}-{end:{max_width}d}: Top-1 accuracy: {seg_top1:3.0f} %, Top-5 accuracy: {seg_top5:3.0f} %"
        )

    # Report total accuracies
    total_top1 = sum(top1_correct) / num_tokens
    total_top5 = sum(top5_correct) / num_tokens
    logger.info(
        f"Total tokens {num_tokens}: Top-1 accuracy: {100 * total_top1:3.1f} %, Top-5 accuracy: {100 * total_top5:3.1f} %"
    )

    # Only show error summary when HF top-1 matches the true token (more actionable)
    logger.info("\nError Summary (only showing errors where reference top-1 matches true token):")
    logger.info("-" * 120)
    for error in errors:
        if error["expected_ids"][0] == error["true_id"]:
            context = sanitize(error["context"])
            incorrect = sanitize(error["incorrect"])
            expected = " | ".join(sanitize(t) for t in error["expected"])
            true_word = sanitize(tokenizer.decode([error["true_id"]], skip_special_tokens=False))
            logger.info(f"{error['position']}: {context}[{incorrect}] != [{expected}], true: [{true_word}]")

    # Sanity-check token_accuracy's computed values vs our computed totals
    if isinstance(first_gen.get("accuracy_top1"), (int, float)):
        if abs(first_gen["accuracy_top1"] - total_top1) > 1e-4:
            logger.warning(
                "TokenAccuracy top-1 {:.6f} != computed {:.6f}",
                first_gen["accuracy_top1"],
                total_top1,
            )
    if isinstance(first_gen.get("accuracy_top5"), (int, float)):
        if abs(first_gen["accuracy_top5"] - total_top5) > 1e-4:
            logger.warning(
                "TokenAccuracy top-5 {:.6f} != computed {:.6f}",
                first_gen["accuracy_top5"],
                total_top5,
            )

    logger.info(f"Top-1: {100 * total_top1:.0f}% | Top-5: {100 * total_top5:.0f}%")

    min_expected_top1 = 0.90
    min_expected_top5 = 0.99
    ref_compared = min(gen_len, max_new_tokens)
    assert total_top1 >= min_expected_top1, (
        f"Top-1 accuracy {total_top1:.4f} is below minimum expected "
        f"{min_expected_top1:.2f}. Generated {len(tt_generated_tokens)} tokens, "
        f"reference has {ref_compared} generated tokens."
    )

    assert total_top5 >= min_expected_top5, (
        f"Top-5 accuracy {total_top5:.4f} is below minimum expected "
        f"{min_expected_top5:.2f}. Generated {len(tt_generated_tokens)} tokens, "
        f"reference has {ref_compared} generated tokens."
    )
