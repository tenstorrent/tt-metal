# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.llasa3b.tt.llasa_utils import XCODEC2_TOKENS_PER_SECOND, extract_speech_ids, format_llasa_chat
from models.tt_transformers.tt.common import PagedAttentionConfig, create_tt_model, sample_host
from models.tt_transformers.tt.generator import Generator


def prepare_llasa_generator(mesh_device, optimizations, batch_size=1, max_seq_len=2048):
    """Create the TTNN model and generator for Llasa-3B inference.

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

    def setup_llasa_args(args):
        # Isolated Llasa-3B performance optimizations
        args.lm_head_math_fidelity = ttnn.MathFidelity.LoFi
        args.skip_lm_head_all_reduce = True
        args.skip_lm_head_all_gather = True

    model_args, model, tt_kv_cache, state_dict = create_tt_model(
        mesh_device,
        instruct=False,  # Llasa is not an instruct model
        max_batch_size=batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        paged_attention_config=paged_attention_config,
        dtype=ttnn.bfloat8_b,
        setup_args_cb=setup_llasa_args,
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


def run_llasa_tts(
    generator,
    model_args,
    tt_kv_cache,
    paged_attention_config,
    input_text,
    max_generated_tokens=1000,
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
    iteration = 1
    compile_time = 0

    while iteration < effective_max_tokens:
        t_iter_start = time.time()

        logits, _ = generator.decode_forward_text(
            out_tok,
            current_pos,
            enable_trace=(iteration > 1),  # Skip trace on first decode iteration (compile)
            page_table=page_table,
            kv_cache=kv_cache_list,
            reset_batch=(iteration == 1),
        )

        _, out_tok = sample_host(logits, temperature=0.7, top_p=1.0, on_host=True)
        iter_time = time.time() - t_iter_start

        if iteration == 1:
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
