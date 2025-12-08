# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest

from models.demos.deepseek_v3.demo.demo import run_demo
from models.demos.deepseek_v3.utils.hf_model_utils import load_tokenizer

MODEL_PATH = Path(os.getenv("DEEPSEEK_V3_HF_MODEL", "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528"))
CACHE_DIR = Path(os.getenv("DEEPSEEK_V3_CACHE", "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache"))


def create_prompt_of_length(target_token_length: int, tokenizer) -> str:
    """
    Create a prompt that tokenizes to approximately the target token length.
    Uses a repeating pattern to reach the desired length.
    """
    # Base text that will be repeated
    base_text = (
        "The quick brown fox jumps over the lazy dog. "
        "This is a test of long context sequences. "
        "We need to test how the model handles increasingly longer input prompts. "
    )

    # First, measure the chat template overhead by tokenizing an empty prompt
    empty_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True
    )
    template_overhead = len(empty_tokens)

    # Tokenize base text to get tokens per repetition (content only)
    base_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": base_text}], add_generation_prompt=True, tokenize=True
    )
    tokens_per_repetition = len(base_tokens) - template_overhead

    # Calculate how many repetitions we need
    # Subtract template overhead from target
    content_tokens_needed = target_token_length - template_overhead
    if tokens_per_repetition <= 0:
        tokens_per_repetition = 1
    num_repetitions = max(1, content_tokens_needed // tokens_per_repetition)

    # Create the prompt
    prompt = base_text * num_repetitions

    # Verify and adjust the actual token length
    actual_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
    )
    actual_length = len(actual_tokens)

    # If we're still too short, add more text word by word
    if actual_length < target_token_length:
        words = base_text.split()
        word_idx = 0
        max_iterations = (target_token_length - actual_length) * 2  # Safety limit
        iteration = 0
        while actual_length < target_token_length and iteration < max_iterations:
            prompt += words[word_idx % len(words)] + " "
            actual_tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
            )
            actual_length = len(actual_tokens)
            word_idx += 1
            iteration += 1

    return prompt


@pytest.mark.parametrize("target_prompt_tokens", [1024, 2048, 3145, 8192])
def test_long_context_input_sequences(target_prompt_tokens):
    """
    Test with varying input prompt lengths to test long context sequence handling.
    Uses progressively longer input prompts to test prefill performance and KV cache management.
    Generates a fixed small number of output tokens to focus on input sequence length testing.
    Uses only 5 layers (override_num_layers=5) for faster CI execution.
    """
    # Load tokenizer to create prompts of specific lengths
    tokenizer = load_tokenizer(MODEL_PATH)

    # Create a prompt of approximately the target token length
    prompt = create_prompt_of_length(target_prompt_tokens, tokenizer)

    # Verify the prompt length is close to target (within 20% tolerance)
    actual_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
    )
    actual_prompt_length = len(actual_tokens)

    # Use a fixed small number of output tokens to focus on input length testing
    max_new_tokens = 32

    # Run demo with long input prompt
    results = run_demo(
        prompts=[prompt],
        model_path=MODEL_PATH,
        cache_dir=CACHE_DIR,
        random_weights=False,
        max_new_tokens=max_new_tokens,
        override_num_layers=5,
        repeat_batches=1,
        early_print_first_user=False,
    )

    # Verify results are returned
    assert len(results["generations"]) > 0, "No generations returned"
    assert (
        len(results["generations"][0]["tokens"]) == max_new_tokens
    ), f"Expected {max_new_tokens} tokens generated, got {len(results['generations'][0]['tokens'])}"

    # Verify statistics are present and contain prefill metrics
    assert "statistics" in results, "Statistics not found in results"
    stats = results["statistics"]
    assert "prefill_t/s" in stats, "Prefill tokens/sec metric not found"
    assert "inference_prefill" in stats, "Prefill inference time not found"

    # Log performance metrics for long context sequences
    prefill_tps = stats.get("prefill_t/s", 0)
    prefill_time_ms = stats.get("inference_prefill", 0) * 1000
    print(f"\nInput prompt length: {actual_prompt_length} tokens (target: {target_prompt_tokens})")
    print(f"Prefill performance: {prefill_tps:.2f} tokens/sec")
    print(f"Prefill time: {prefill_time_ms:.2f}ms")

    # Verify prompt length is reasonable (within 20% of target)
    assert target_prompt_tokens > 0, "Target prompt tokens must be greater than 0"
    assert (
        abs(actual_prompt_length - target_prompt_tokens) / target_prompt_tokens < 0.2
    ), f"Prompt length {actual_prompt_length} is not close to target {target_prompt_tokens}"
