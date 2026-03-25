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

import hashlib
import json
import time
from pathlib import Path

import pytest
import requests
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen3_5_9b.tt.qwen35_model import Qwen35Model

CHECKPOINT_DIR = "/local/ttuser/atupe/Qwen9b"

DEVICE_PARAMS = [{"l1_small_size": 24576, "num_command_queues": 2}]

SAMPLE_PROMPTS_DIR = "models/demos/blackhole/qwen3_5_9b/demo/sample_prompts"

PERF_TARGETS = {
    128: {"min_decode_tok_s": 5.0, "max_ttft_s": 15.0},
    2048: {"min_decode_tok_s": 4.0, "max_ttft_s": 20.0},
    4096: {"min_decode_tok_s": 2.0, "max_ttft_s": 40.0},
    8192: {"min_decode_tok_s": 1.0, "max_ttft_s": 80.0},
}

# Frankenstein prompt config: seqlen → json_index in eval_frankenstein_long.json.
# Each entry clips Frankenstein text to enough characters to exceed the target token count
# (English text ≈ 4.3 chars/token). Full Frankenstein is ~450K chars ≈ 104K tokens.
_FRANKENSTEIN_CONFIGS = {
    8192: 0,  # 70k chars ≈ 16k tokens (covers 8k with margin)
    16384: 1,  # 140k chars ≈ 32k tokens
    32768: 1,  # 140k chars ≈ 32k tokens
    65536: 2,  # 300k chars ≈ 70k tokens
    131072: 3,  # 500k chars ≈ 104k tokens (full text, Frankenstein caps at ~104k tokens)
}


def _load_and_cache_context(context_url, max_length=None):
    """Download text from URL, cache locally, clip to max_length."""
    cache_dir = Path(SAMPLE_PROMPTS_DIR) / ".context_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / hashlib.md5(context_url.encode()).hexdigest()

    if cache_file.exists():
        context_text = cache_file.read_text()
        logger.info(f"Loaded context from cache: {context_url}")
    else:
        response = requests.get(context_url)
        response.raise_for_status()
        context_text = response.text
        cache_file.write_text(context_text)
        logger.info(f"Downloaded and cached context: {context_url}")

    if max_length:
        context_text = context_text[:max_length]
    return context_text


def _get_prompt(seqlen, tokenizer):
    """Load or generate a prompt of approximately seqlen tokens."""
    if seqlen <= 128:
        prompt = "<|im_start|>user\n" "What is the capital of France?<|im_end|>\n" "<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt")
        return inputs["input_ids"]

    # For long sequences (16k+), use Frankenstein from Project Gutenberg.
    # Feed the raw text and let the model continue it — tests long-context processing.
    if seqlen in _FRANKENSTEIN_CONFIGS:
        idx = _FRANKENSTEIN_CONFIGS[seqlen]
        path = f"{SAMPLE_PROMPTS_DIR}/eval_frankenstein_long.json"
        with open(path) as f:
            data = json.load(f)
        entry = data[idx]
        context = _load_and_cache_context(entry["context"], entry.get("max_length"))
        # Wrap in chat template so the model enters instruction-following mode.
        # Truncate context to leave room for the template + instruction.
        instruction = entry["prompt"]
        prefix = "<|im_start|>user\n"
        # Seed <think> to start the reasoning chain. At very long contexts (100K+),
        # the DeltaNet recurrent state's finite capacity [B,32,128,128] dilutes the
        # suffix signal, making argmax split ~33% <|im_end|> vs ~15% <think>.
        # Explicit <think> ensures the model enters reasoning mode.
        suffix = f"\n\nBased on the above text: {instruction}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        wrapper_ids = tokenizer(prefix + suffix, add_special_tokens=False, return_tensors="pt")["input_ids"]
        max_context_tokens = seqlen - wrapper_ids.shape[1]
        context_ids = tokenizer(context, add_special_tokens=False, return_tensors="pt")["input_ids"][
            :, :max_context_tokens
        ]
        prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"]
        suffix_ids = tokenizer(suffix, add_special_tokens=False, return_tensors="pt")["input_ids"]
        import torch as _torch

        return _torch.cat([prefix_ids, context_ids, suffix_ids], dim=1)

    # For medium sequences (1k-8k), use static prompt files
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
        (128, 2048, 50, True),
        (4096, 8192, 100, True),
        (8192, 16384, 50, True),
        (65536, 131072, 100, True),
        (131072, 262144, 100, True),
    ],
    ids=["prefill_128", "prefill_4k", "prefill_8k", "prefill_64k", "prefill_128k"],
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
    """Prefill + traced decode loop. Returns (generated_tokens, perf_dict).

    Trace captures embedding + full forward + norm + LM head.
    Before each replay, inputs are updated via fast host-to-device DMA.
    """
    T = token_ids.shape[1]
    model.enable_trace(batch_size=1, use_paged_cache=False)

    t0 = time.time()
    logits = model._prefill_for_trace(token_ids)
    ttnn.synchronize_device(device)
    ttft = time.time() - t0

    logits_torch = ttnn.to_torch(logits).squeeze()
    assert not torch.isnan(logits_torch).any(), "NaN in prefill logits"
    next_token = logits_torch.argmax().item()

    # Warmup decode — uses old forward_decode to produce warmup token + populate cache at pos T
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

    # Profile one eager decode step to see per-layer breakdown
    logger.info("=" * 70)
    logger.info("PROFILING EAGER DECODE (per-layer breakdown):")
    profile_input = torch.tensor([[warmup_token]], dtype=torch.long)
    model.decode(profile_input, current_pos=T + 1, profile=True)
    ttnn.synchronize_device(device)
    logger.info("=" * 70)

    # Capture trace (embedding inside trace, internal warmup compiles new path)
    model.capture_decode_trace(device)

    generated = [next_token, warmup_token]
    decode_times = []
    current_token = warmup_token
    current_pos = T + 1

    for i in range(max_generated_tokens):
        next_input = torch.tensor([[current_token]], dtype=torch.long)

        t_step = time.time()
        logits = model.decode_traced(next_input, current_pos=current_pos, profile=(i < 5))
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
