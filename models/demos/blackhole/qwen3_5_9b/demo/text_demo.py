# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5-9B end-to-end text generation test on Blackhole P150.

Consolidates all e2e demo tests into a single parametrized test that covers:
  - Short prompt text generation (128 tokens)
  - Medium prefill with traced decode (2048 tokens)
  - Long-context prefill + decode (4k, 8k tokens)

Run all:    pytest models/demos/blackhole/qwen3_5_9b/demo/text_demo.py -v -s
Run short:  pytest models/demos/blackhole/qwen3_5_9b/demo/text_demo.py -v -s -k "prefill_128"
Run 2k:     pytest models/demos/blackhole/qwen3_5_9b/demo/text_demo.py -v -s -k "prefill_2k"
"""

import json
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen3_5_9b.tt.qwen35_model import Qwen35Model

CHECKPOINT_DIR = "/localdev/atupe/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2"

DEVICE_PARAMS = [{"l1_small_size": 24576, "num_command_queues": 2}]

SAMPLE_PROMPTS_DIR = "models/demos/blackhole/qwen3_5_9b/demo/sample_prompts"

PERF_TARGETS = {
    128: {"min_decode_tok_s": 5.0, "max_ttft_s": 15.0},
    2048: {"min_decode_tok_s": 4.0, "max_ttft_s": 20.0},
    4096: {"min_decode_tok_s": 2.0, "max_ttft_s": 40.0},
    8192: {"min_decode_tok_s": 1.5, "max_ttft_s": 80.0},
}


def _get_prompt(seqlen, tokenizer):
    """Load or generate a prompt of approximately seqlen tokens."""
    if seqlen <= 128:
        prompt = "<|im_start|>user\n" "What is the capital of France?<|im_end|>\n" "<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt")
        return inputs["input_ids"]

    size_label = f"{seqlen // 1024}k" if seqlen >= 1024 else str(seqlen)
    path = f"{SAMPLE_PROMPTS_DIR}/input_data_long_{size_label}.json"
    with open(path) as f:
        data = json.load(f)
    prompt_text = data[0]["prompt"]
    inputs = tokenizer(prompt_text, return_tensors="pt")
    return inputs["input_ids"][:, :seqlen]


@run_for_blackhole()
@pytest.mark.timeout(900)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize(
    "seqlen, max_seq_len, max_generated_tokens, use_trace",
    [
        (128, 2048, 50, False),
        (2048, 4096, 30, True),
        (4096, 8192, 30, True),
        (8192, 16384, 20, True),
    ],
    ids=["prefill_128", "prefill_2k", "prefill_4k", "prefill_8k"],
)
def test_demo_text(
    device,
    seqlen,
    max_seq_len,
    max_generated_tokens,
    use_trace,
):
    """End-to-end text generation: prefill + decode with performance validation."""
    from transformers import PreTrainedTokenizerFast

    device.enable_program_cache()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(CHECKPOINT_DIR)

    t0 = time.time()
    model = Qwen35Model.from_pretrained(
        device,
        CHECKPOINT_DIR,
        max_batch_size=1,
        max_seq_len=max_seq_len,
    )
    logger.info(f"Model load: {time.time() - t0:.1f}s")

    token_ids = _get_prompt(seqlen, tokenizer)
    actual_len = token_ids.shape[1]
    logger.info(f"Prompt: {actual_len} tokens (target: {seqlen}, max_seq_len: {max_seq_len})")

    if use_trace:
        generated, perf = _run_traced_generation(
            model,
            tokenizer,
            device,
            token_ids,
            max_generated_tokens,
        )
    else:
        generated, perf = _run_eager_generation(
            model,
            tokenizer,
            device,
            token_ids,
            max_generated_tokens,
        )

    text = tokenizer.decode(generated, skip_special_tokens=True)
    _log_results(perf, actual_len, len(generated), text)
    _assert_results(perf, actual_len, len(generated))


def _run_eager_generation(model, tokenizer, device, token_ids, max_generated_tokens):
    """Prefill + eager decode loop. Returns (generated_tokens, perf_dict)."""
    prompt_len = token_ids.shape[1]
    model.reset_state(batch_size=1)

    t0 = time.time()
    logits = model.prefill(token_ids)
    ttnn.synchronize_device(device)
    ttft = time.time() - t0

    logits_torch = ttnn.to_torch(logits).squeeze()
    assert not torch.isnan(logits_torch).any(), "NaN in prefill logits"
    next_token = logits_torch.argmax().item()

    generated = [next_token]
    decode_times = []

    for i in range(max_generated_tokens - 1):
        next_input = torch.tensor([[next_token]], dtype=torch.long)

        t_step = time.time()
        logits = model.decode(next_input, current_pos=prompt_len + i)
        ttnn.synchronize_device(device)
        decode_times.append(time.time() - t_step)

        logits_torch = ttnn.to_torch(logits).squeeze()
        assert not torch.isnan(logits_torch).any(), f"NaN in decode logits at step {i}"
        next_token = logits_torch.argmax().item()

        if next_token == tokenizer.eos_token_id:
            break
        generated.append(next_token)

    avg_decode = sum(decode_times) / len(decode_times) if decode_times else float("inf")
    return generated, {"ttft": ttft, "avg_decode_s": avg_decode, "decode_steps": len(decode_times)}


def _run_traced_generation(model, tokenizer, device, token_ids, max_generated_tokens):
    """Prefill + traced decode loop. Returns (generated_tokens, perf_dict)."""
    T = token_ids.shape[1]
    model.enable_trace(batch_size=1)

    t0 = time.time()
    logits = model._prefill_for_trace(token_ids)
    ttnn.synchronize_device(device)
    ttft = time.time() - t0

    logits_torch = ttnn.to_torch(logits).squeeze()
    assert not torch.isnan(logits_torch).any(), "NaN in prefill logits"
    next_token = logits_torch.argmax().item()

    # Warmup decode (populates program cache before trace capture)
    warmup_input = torch.tensor([[next_token]], dtype=torch.long)
    token_ids_ttnn = ttnn.from_torch(warmup_input, dtype=ttnn.uint32, device=device)
    x = ttnn.embedding(token_ids_ttnn, model.tok_embeddings, layout=ttnn.TILE_LAYOUT)
    ttnn.deallocate(token_ids_ttnn)
    cos, sin = model.rope.get_rot_mats(torch.full((1, 1), T, dtype=torch.long))
    logits = model.forward_decode(x, cos, sin)
    ttnn.synchronize_device(device)
    warmup_token = ttnn.to_torch(logits).squeeze().argmax().item()
    for layer in model.layers:
        if layer.is_full_attention:
            layer.attention.update_cache_after_trace(T)

    model.capture_decode_trace(device)

    generated = [next_token, warmup_token]
    decode_times = []
    current_token = warmup_token
    current_pos = T + 1

    for i in range(max_generated_tokens):
        next_input = torch.tensor([[current_token]], dtype=torch.long)

        t_step = time.time()
        logits = model.decode_traced(next_input, current_pos=current_pos)
        ttnn.synchronize_device(device)
        decode_times.append(time.time() - t_step)

        logits_torch = ttnn.to_torch(logits).squeeze()
        assert not torch.isnan(logits_torch).any(), f"NaN in traced decode at step {i}"
        current_token = logits_torch.argmax().item()
        generated.append(current_token)
        current_pos += 1

        if current_token == tokenizer.eos_token_id:
            break

    avg_decode = sum(decode_times) / len(decode_times) if decode_times else float("inf")
    return generated, {"ttft": ttft, "avg_decode_s": avg_decode, "decode_steps": len(decode_times)}


def _log_results(perf, prompt_len, num_generated, text):
    ttft = perf["ttft"]
    avg_ms = perf["avg_decode_s"] * 1000
    tok_s = 1000.0 / avg_ms if avg_ms > 0 else 0

    logger.info("=" * 70)
    logger.info(f"  Prefill {prompt_len} tokens:  TTFT = {ttft:.3f}s ({prompt_len / ttft:.0f} tok/s)")
    logger.info(f"  Decode:  {avg_ms:.1f}ms/token  ({tok_s:.1f} tok/s)")
    logger.info(f"  Generated {num_generated} tokens in {perf['decode_steps']} steps")
    logger.info(f"  Text: {text[:200]}")
    logger.info("=" * 70)


def _assert_results(perf, prompt_len, num_generated):
    assert num_generated >= 1, "Should generate at least 1 token"

    targets = PERF_TARGETS.get(prompt_len)
    if targets is None:
        return

    tok_s = 1.0 / perf["avg_decode_s"] if perf["avg_decode_s"] > 0 else 0
    min_tok_s = targets["min_decode_tok_s"]
    assert (
        tok_s >= min_tok_s
    ), f"Decode throughput {tok_s:.1f} tok/s below target {min_tok_s} tok/s at seqlen={prompt_len}"

    max_ttft = targets["max_ttft_s"]
    assert perf["ttft"] < max_ttft, f"TTFT {perf['ttft']:.1f}s exceeds target {max_ttft}s at seqlen={prompt_len}"
