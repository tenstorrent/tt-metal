# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Llasa-3B Text-to-Speech Demo on Tenstorrent Hardware

This demo runs the Llasa-3B TTS model using the existing tt_transformers infrastructure.
Llasa-3B is architecturally identical to LLaMA-3.2-3B but with an extended vocabulary
that includes 65,536 XCodec2 speech tokens.

Usage:
    # Set environment variable for the model
    export HF_MODEL=HKUSTAudio/Llasa-3B

    # Run zero-shot TTS
    pytest models/demos/llasa3b/demo/llasa_demo.py -k "test_llasa_tts" -s

    # Run prompted TTS (voice cloning)
    pytest models/demos/llasa3b/demo/llasa_demo.py -k "test_llasa_tts_prompted" -s
"""

import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llasa3b.tt.llasa_utils import (
    XCODEC2_TOKENS_PER_SECOND,
    decode_speech_to_audio,
    encode_prompt_audio,
    extract_speech_ids,
    format_llasa_chat,
)
from models.tt_transformers.tt.common import PagedAttentionConfig, create_tt_model, sample_host
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import DecodersPrecision

# ============================================================================
# Model Setup
# ============================================================================


def prepare_llasa_generator(mesh_device, optimizations, batch_size=1, max_seq_len=2048):
    """Create the TTNN model and generator for Llasa-3B inference.

    This follows the same pattern as prepare_generator_args in simple_text_demo.py
    but simplified for single-user TTS (batch=1, no data parallel).

    Args:
        mesh_device: TTNN mesh device.
        optimizations: Decoder precision/optimization settings.
        batch_size: Number of concurrent users (default: 1).
        max_seq_len: Maximum sequence length (default: 2048).

    Returns:
        Tuple of (model_args, generator, tt_kv_cache, paged_attention_config).
    """
    hf_model = os.environ.get("HF_MODEL", "HKUSTAudio/Llasa-3B")
    os.environ["HF_MODEL"] = hf_model

    paged_attention_config = PagedAttentionConfig(block_size=32, max_num_blocks=1024)

    model_args, model, tt_kv_cache, state_dict = create_tt_model(
        mesh_device,
        instruct=False,  # Llasa is not an instruct model
        max_batch_size=batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        paged_attention_config=paged_attention_config,
        dtype=ttnn.bfloat8_b,
    )

    tokenizer = model_args.tokenizer
    generator = Generator([model], [model_args], mesh_device, tokenizer=tokenizer)

    logger.info(f"Model loaded: {model_args.model_name}, vocab_size={model_args.vocab_size}")
    return model_args, generator, tt_kv_cache, paged_attention_config


def create_page_table(batch_size, paged_attention_config):
    """Create a page table for paged attention.

    Args:
        batch_size: Number of users.
        paged_attention_config: Paged attention configuration.

    Returns:
        Page table tensor [batch_size, max_num_blocks // batch_size].
    """
    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    return reverse_permutation.reshape(batch_size, paged_attention_config.max_num_blocks // batch_size)


# ============================================================================
# Generation Pipeline
# ============================================================================


def run_llasa_tts(
    generator,
    model_args,
    tt_kv_cache,
    paged_attention_config,
    input_text,
    max_generated_tokens=500,
    prompt_text=None,
    prompt_speech_tokens=None,
):
    """Run the full Llasa TTS pipeline: tokenize → prefill → decode → extract speech tokens.

    Supports both zero-shot TTS (text only) and prompted TTS (voice cloning).

    Args:
        generator: TTNN Generator instance.
        model_args: Model arguments with tokenizer.
        tt_kv_cache: KV cache tensors.
        paged_attention_config: Paged attention config.
        input_text: Text to convert to speech.
        max_generated_tokens: Maximum speech tokens to generate.
        prompt_text: Optional transcript of prompt audio (for voice cloning).
        prompt_speech_tokens: Optional list of speech token strings from encode_prompt_audio().

    Returns:
        Tuple of (speech_ids, metrics_dict).
    """
    tokenizer = model_args.tokenizer
    batch_size = 1
    is_prompted = prompt_text is not None and prompt_speech_tokens is not None

    # ── Tokenize ──────────────────────────────────────────────
    if is_prompted:
        logger.info(f"Tokenizing input (prompted TTS with {len(prompt_speech_tokens)} prompt tokens)...")
    else:
        logger.info("Tokenizing input (zero-shot TTS)...")

    input_ids = format_llasa_chat(
        input_text, tokenizer, prompt_text=prompt_text, prompt_speech_tokens=prompt_speech_tokens
    )
    prompt_len = input_ids.shape[1]
    logger.info(f"Input tokens: {prompt_len}")

    # Check token budget
    if prompt_len >= 2048:
        logger.error(f"Input ({prompt_len} tokens) exceeds max_length=2048. " f"Use a shorter prompt audio or text.")
        return [], {"error": "input_too_long"}

    token_budget = 2048 - prompt_len
    effective_max_tokens = min(max_generated_tokens, token_budget)
    logger.info(f"Token budget: {token_budget} available, generating up to {effective_max_tokens} tokens")

    speech_end_id = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
    logger.info(f"Stop token ID (<|SPEECH_GENERATION_END|>): {speech_end_id}")

    input_tokens_pt = input_ids.view(batch_size, -1)
    page_table = create_page_table(batch_size, paged_attention_config)
    kv_cache_list = [tt_kv_cache]

    # ── Prefill ───────────────────────────────────────────────
    logger.info("Running prefill warmup...")
    _ = generator.prefill_forward_text(
        input_tokens_pt,
        page_table=page_table,
        kv_cache=kv_cache_list,
        prompt_lens=[prompt_len],
    )

    logger.info("Running prefill...")
    t_prefill_start = time.time()
    logits = generator.prefill_forward_text(
        input_tokens_pt,
        page_table=page_table,
        kv_cache=kv_cache_list,
        prompt_lens=[prompt_len],
        warmup_prefill=False,
    )
    prefill_time = time.time() - t_prefill_start
    logger.info(f"Prefill completed in {prefill_time:.3f}s ({prompt_len / prefill_time:.1f} tok/s)")

    # Sample first token
    _, out_tok = sample_host(logits, temperature=0.5, top_p=1.0, on_host=True)
    generated_tokens = [out_tok[0].item()]
    current_pos = torch.tensor([prompt_len])

    # ── Autoregressive Decode ─────────────────────────────────
    logger.info(f"Starting decode (max {effective_max_tokens} tokens)...")
    t_decode_start = time.time()
    iteration = 0
    compile_time = 0

    while iteration < effective_max_tokens:
        t_iter_start = time.time()

        logits, _ = generator.decode_forward_text(
            out_tok,
            current_pos,
            enable_trace=(iteration > 0),  # Skip trace on first iteration (compile)
            page_table=page_table,
            kv_cache=kv_cache_list,
            reset_batch=(iteration == 0),
        )

        _, out_tok = sample_host(logits, temperature=0.7, top_p=1.0, on_host=True)
        iter_time = time.time() - t_iter_start

        if iteration == 0:
            compile_time = iter_time
            logger.info(f"Decode compile: {compile_time:.3f}s")

        token_id = out_tok[0].item()
        generated_tokens.append(token_id)
        current_pos += 1
        iteration += 1

        # Check for stop token
        if token_id == speech_end_id:
            logger.info(f"Stop token generated at iteration {iteration}")
            break

        # Log progress every 50 tokens
        if iteration % 50 == 0:
            tok_s = 1.0 / iter_time if iter_time > 0 else 0
            logger.info(f"  Iteration {iteration}: {tok_s:.1f} tok/s")

    total_decode_time = time.time() - t_decode_start
    decode_time_no_compile = total_decode_time - compile_time

    # ── Log Metrics ───────────────────────────────────────────
    logger.info("=== Decode Summary ===")
    logger.info(f"  Generated tokens: {len(generated_tokens)}")
    logger.info(f"  Total decode time: {total_decode_time:.3f}s (including {compile_time:.3f}s compile)")

    decode_tok_s = 0
    if iteration > 1 and decode_time_no_compile > 0:
        decode_tok_s = (iteration - 1) / decode_time_no_compile
        logger.info(f"  Decode throughput: {decode_tok_s:.1f} tok/s/user")

        audio_duration = len(generated_tokens) / XCODEC2_TOKENS_PER_SECOND
        rtf = total_decode_time / audio_duration if audio_duration > 0 else float("inf")
        logger.info(f"  Estimated audio duration: {audio_duration:.2f}s")
        logger.info(f"  Real-Time Factor (RTF): {rtf:.3f}")

    # ── Extract Speech Tokens ─────────────────────────────────
    # For prompted TTS, the generated tokens include continuation from the prompt
    # speech tokens. We extract all speech IDs from the generated portion.
    speech_token_strs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    speech_ids = extract_speech_ids(speech_token_strs)
    logger.info(f"Extracted {len(speech_ids)} speech IDs from {len(generated_tokens)} generated tokens")

    # For prompted TTS, we typically want only the generated audio (the new text).
    # If you want to include the prompt in the output, uncomment the following:
    # if is_prompted and prompt_speech_tokens:
    #     from models.demos.llasa3b.tt.llasa_utils import extract_speech_ids as _extract
    #     prompt_speech_ids = _extract(prompt_speech_tokens)
    #     speech_ids = prompt_speech_ids + speech_ids
    #     logger.info(f"Total speech IDs (prompt + generated): {len(speech_ids)}")

    metrics = {
        "prompt_len": prompt_len,
        "generated_tokens": len(generated_tokens),
        "prefill_time": prefill_time,
        "prefill_tok_s": prompt_len / prefill_time,
        "decode_time": total_decode_time,
        "compile_time": compile_time,
        "decode_tok_s": decode_tok_s,
        "mode": "prompted" if is_prompted else "zero_shot",
    }

    return speech_ids, metrics


# ============================================================================
# Test Configuration & Pytest Entry Point
# ============================================================================


def load_input_prompts(prompts_file=None):
    """Load input prompts from JSON file.

    Args:
        prompts_file: Path to JSON file with list of {lang, text} entries.
            Defaults to the bundled input_data.json.

    Returns:
        List of prompt dicts.
    """
    if prompts_file is None:
        prompts_file = Path(__file__).parent / "input_data.json"
    with open(prompts_file, "r") as f:
        return json.load(f)


SAMPLE_INPUTS = load_input_prompts()

# Shared pytest device/optimization parameters
DEVICE_PARAMS = [{"fabric_config": True, "trace_region_size": 50000000, "num_command_queues": 1}]
MESH_SHAPE = [
    {
        "N150": (1, 1),
        "N300": (1, 2),
        "T3K": (1, 8),
    }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
]
OPTIMIZATIONS = [
    lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name),
]


# ── Zero-Shot TTS ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "index, input_text",
    [(i, entry["text"]) for i, entry in enumerate(SAMPLE_INPUTS)],
    ids=[f"{entry['lang']}-{i}" for i, entry in enumerate(SAMPLE_INPUTS)],
)
@pytest.mark.parametrize("max_generated_tokens", [500])
@pytest.mark.parametrize("optimizations", OPTIMIZATIONS, ids=["performance"])
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", MESH_SHAPE, indirect=True)
def test_llasa_zero_shot(
    index,
    input_text,
    max_generated_tokens,
    optimizations,
    mesh_device,
):
    """
    End-to-end Llasa-3B zero-shot TTS demo on Tenstorrent hardware.

    Pipeline: text → TTNN inference → speech tokens → XCodec2 → WAV
    """
    logger.info("=== Llasa-3B TTS Demo (Zero-Shot) ===")
    logger.info(f"Input: '{input_text}'")

    # 1. Create model and generator
    model_args, generator, tt_kv_cache, paged_attention_config = prepare_llasa_generator(mesh_device, optimizations)

    # 2. Run the TTS pipeline
    speech_ids, metrics = run_llasa_tts(
        generator, model_args, tt_kv_cache, paged_attention_config, input_text, max_generated_tokens
    )

    # 3. Decode speech tokens to audio
    if speech_ids:
        output_dir = "models/demos/llasa3b/demo/output"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"llasa_output_{index}.wav"
        output_path = os.path.join(output_dir, output_filename)
        result = decode_speech_to_audio(speech_ids, output_path=output_path)

        if result:
            logger.info(f"Audio saved to {result}")
    else:
        logger.warning("No speech tokens found in generated output!")

    logger.info("=== Demo Complete ===")


# ── Prompted TTS (Voice Cloning) ──────────────────────────────

# Official Anna.wav prompt from HuggingFace
ANNA_PROMPT_TEXT = (
    "A chance to leave him alone, but... No. She just wanted to see him again. "
    "Anna, you don't know how it feels to lose a sister. "
    "Anna, I'm sorry, but your father asked me not to tell you anything."
)
ANNA_TARGET_TEXT = (
    "Dealing with family secrets is never easy. Yet, sometimes, omission is a form of protection, "
    "intending to safeguard some from the harsh truths. One day, I hope you understand the reasons "
    "behind my actions. Until then, Anna, please, bear with me."
)


def download_anna_wav():
    """Download the official Anna.wav prompt from HuggingFace if not already cached."""
    cache_dir = Path("models/demos/llasa3b/demo/prompts")
    wav_path = cache_dir / "Anna.wav"
    if wav_path.exists():
        logger.info(f"Using cached prompt: {wav_path}")
        return str(wav_path)

    logger.info("Downloading Anna.wav from HuggingFace...")
    from huggingface_hub import hf_hub_download

    hf_hub_download("HKUSTAudio/Llasa-3B", "Anna.wav", local_dir=str(cache_dir))
    logger.info(f"Downloaded to {wav_path}")
    return str(wav_path)


@pytest.mark.parametrize("max_generated_tokens", [1000])
@pytest.mark.parametrize("optimizations", OPTIMIZATIONS, ids=["performance"])
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", MESH_SHAPE, indirect=True)
def test_llasa_voice_cloning(
    max_generated_tokens,
    optimizations,
    mesh_device,
):
    """
    End-to-end Llasa-3B prompted TTS (voice cloning) demo on Tenstorrent hardware.

    Pipeline: prompt WAV → XCodec2 encode → TTNN inference → speech tokens → XCodec2 decode → WAV
    Uses the official Anna.wav prompt from the Llasa-3B HuggingFace repo.
    """
    logger.info("=== Llasa-3B TTS Demo (Prompted / Voice Cloning) ===")
    logger.info(f"Prompt text: '{ANNA_PROMPT_TEXT[:80]}...'")
    logger.info(f"Target text: '{ANNA_TARGET_TEXT[:80]}...'")

    # 1. Download and encode prompt audio (CPU)
    wav_path = download_anna_wav()
    prompt_speech_tokens = encode_prompt_audio(wav_path)
    if prompt_speech_tokens is None:
        pytest.skip("XCodec2 encoder not available — skipping prompted TTS test")

    logger.info(f"Prompt encoded: {len(prompt_speech_tokens)} speech tokens")

    # 2. Create model and generator
    model_args, generator, tt_kv_cache, paged_attention_config = prepare_llasa_generator(mesh_device, optimizations)

    # 3. Run the TTS pipeline with prompt
    speech_ids, metrics = run_llasa_tts(
        generator,
        model_args,
        tt_kv_cache,
        paged_attention_config,
        ANNA_TARGET_TEXT,
        max_generated_tokens,
        prompt_text=ANNA_PROMPT_TEXT,
        prompt_speech_tokens=prompt_speech_tokens,
    )

    # 4. Decode speech tokens to audio
    if speech_ids:
        output_dir = "models/demos/llasa3b/demo/output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "llasa_output_prompted.wav")
        result = decode_speech_to_audio(speech_ids, output_path=output_path)
        if result:
            logger.info(f"Voice-cloned audio saved to {result}")
    else:
        logger.warning("No speech tokens found in generated output!")

    logger.info("=== Demo Complete ===")
