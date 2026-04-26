# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Phi-1 Inference Demo on Wormhole N150/N300

End-to-end demo for running microsoft/phi-1 on Tenstorrent Wormhole devices.
Supports both prefill and decode phases with performance profiling via Tracy signposts.
"""

import os
import sys
import torch
import ttnn
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM

import models.demos.wormhole.phi_1.tt.phi_model as phi_model
from models.utility_functions import (
    disable_compilation_reports,
    enable_persistent_kernel_cache,
    profiler,
    use_hf_model,
)
from models.demos.wormhole.phi_1.config import PhiConfig
from models.perf.perf_utils import prep_perf_report


def preprocess_inputs(input_prompts, tokenizer, device, batch_size, seq_len):
    tokenizer.pad_token = tokenizer.eos_token
    tokenized = tokenizer(
        input_prompts,
        padding="max_length",
        max_length=seq_len,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = tokenized.input_ids

    if len(input_ids) < batch_size:
        pad_size = batch_size - len(input_ids)
        padding = torch.full((pad_size, seq_len), tokenizer.pad_token_id)
        input_ids = torch.cat((input_ids, padding), dim=0)

    input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    return input_ids


def run_phi1_inference(
    device,
    model_version: str = "microsoft/phi-1",
    batch_size: int = 1,
    seq_len: int = 1024,
    num_tokens_to_generate: int = 25,
    temperature: float = 0.9,
    top_k: int = 1,
    demo_mode: str = "decode",  # "prefill" or "decode"
    model_config: Optional[dict] = None,
    enable_tracy: bool = False,
):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_version)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize model configuration
    config = PhiConfig()
    config.update_model_config(model_config or {})

    # Load TT model
    tt_model = phi_model.TtPhi(
        device=device,
        state_dict={},
        base_url="",
        max_position_embeddings=config.max_position_embeddings,
        config=config,
        rotary_dim=config.rotary_dim,
        use_xformer_rotary=False,
    )

    # Warmup
    profiler.start("compile")
    for _ in range(3):
        dummy_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        tt_input_ids = ttnn.from_torch(dummy_ids, dtype=ttnn.uint32, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
        _ = tt_model(tt_input_ids)
    profiler.end("compile")
    compile_time = profiler.get("compile")
    logger.info(f"Model compilation time: {compile_time:.2f} s")

    # Prefill phase
    if demo_mode == "prefill":
        prompt = "def hello_world():\n    print('Hello, World!')\n\n# A function to add two numbers"
        input_prompts = [prompt] * batch_size
        input_ids = preprocess_inputs(input_prompts, tokenizer, device, batch_size, seq_len)

        if enable_tracy:
            signpost(message="START_PREFILL")

        profiler.start("prefill")
        tt_logits = tt_model(input_ids)
        tt_logits_torch = ttnn.to_torch(tt_logits).squeeze(0)
        profiler.end("prefill")
        prefill_time = profiler.get("prefill")
        logger.info(f"Prefill time: {prefill_time:.4f} s")

        if enable_tracy:
            signpost(message="END_PREFILL")

        # Decode first token
        next_token = sample_token(tt_logits_torch[-1, :], temperature, top_k)
        generated_tokens = [next_token]

        # Decode phase
        for i in range(num_tokens_to_generate - 1):
            if enable_tracy and i == 0:
                signpost(message="START_DECODE")

            profiler.start(f"decode_{i}")
            # Reshape input to (batch_size, 1)
            tt_new_token = ttnn.from_torch(
                torch.tensor([[next_token]], dtype=torch.int32),
                dtype=ttnn.uint32,
                device=device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            tt_logits = tt_model(tt_new_token)
            tt_logits_torch = ttnn.to_torch(tt_logits).squeeze(0).squeeze(0)
            next_token = sample_token(tt_logits_torch, temperature, top_k)
            generated_tokens.append(next_token)
            profiler.end(f"decode_{i}")

        if enable_tracy:
            signpost(message="END_DECODE")

        avg_decode_time = sum(profiler.get(f"decode_{i}") for i in range(num_tokens_to_generate - 1)) / (
            num_tokens_to_generate - 1
        )
        tokens_per_second = 1.0 / avg_decode_time
        logger.info(f"Average decode time: {avg_decode_time:.4f} s")
        logger.info(f"Tokens per second per user: {tokens_per_second:.2f}")

        # Decode only
    elif demo_mode == "decode":
        # Start with a single token
        next_token = tokenizer.bos_token_id or tokenizer.eos_token_id
        generated_tokens = []

        if enable_tracy:
            signpost(message="START_DECODE")

        for i in range(num_tokens_to_generate):
            profiler.start(f"decode_{i}")
            tt_new_token = ttnn.from_torch(
                torch.tensor([[next_token]], dtype=torch.int32),
                dtype=ttnn.uint32,
                device=device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            tt_logits = tt_model(tt_new_token)
            tt_logits_torch = ttnn.to_torch(tt_logits).squeeze(0).squeeze(0)
            next_token = sample_token(tt_logits_torch, temperature, top_k)
            generated_tokens.append(next_token)
            profiler.end(f"decode_{i}")

        if enable_tracy:
            signpost(message="END_DECODE")

        avg_decode_time = sum(profiler.get(f"decode_{i}") for i in range(num_tokens_to_generate)) / num_tokens_to_generate
        tokens_per_second = 1.0 / avg_decode_time
        logger.info(f"Average decode time: {avg_decode_time:.4f} s")
        logger.info(f"Tokens per second per user: {tokens_per_second:.2f}")

    # Decode to text
    generated_text = tokenizer.decode(generated_tokens)
    logger.info(f"Generated text: {generated_text}")

    # Final metrics
    if demo_mode == "prefill":
        return {
            "compile_time": compile_time,
            "prefill_time": prefill_time,
            "avg_decode_time": avg_decode_time,
            "tokens_per_second_per_user": tokens_per_second,
        }
    else:
        return {
            "compile_time": compile_time,
            "avg_decode_time": avg_decode_time,
            "tokens_per_second_per_user": tokens_per_second,
        }


def sample_token(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    if temperature == 0:
        return int(torch.argmax(logits))
    if top_k > 1:
        values, indices = torch.topk(logits, top_k)
        probs = torch.softmax(values / temperature, dim=-1)
        chosen_idx = torch.multinomial(probs, num_samples=1)
        return int(indices[chosen_idx])
    probs = torch.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1))


def run_demo():
    parser = argparse.ArgumentParser(description="Run Phi-1 inference on Wormhole device")
    parser.add_argument(
        "--model_version",
        type=str,
        default="microsoft/phi-1",
        help="Model version to load from HuggingFace",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for prefill phase",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=1024,
        help="Sequence length for prefill",
    )
    parser.add_argument(
        "--num_tokens_to_generate",
        type=int,
        default=25,
        help="Number of tokens to generate during decode",
    )
    parser.add_argument(
        "--demo_mode",
        type=str,
        default="decode",
        choices=["prefill", "decode"],
        help="Demo mode: prefill (prefill + decode) or decode (autoregressive only)",
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="Device ID to use (for multi-device setups)",
    )
    parser.add_argument("--enable_tracy", action="store_true", help="Enable Tracy signposts for profiling")
    parser.add_argument("--compile_only", action="store_true", help="Only compile the model, do not run inference")

    args = parser.parse_args()

    # Setup device
    device = ttnn.open_device(device_id=args.device_id)
    ttnn.enable_program_cache(device)
    ttnn.enable_async(device)

    # Set configs
    if args.batch_size == 1:
        model_config = phi_model.get_model_config(batch_size=1, seq_len=args.seq_len)
    else:
        raise NotImplementedError("Batch size > 1 not yet supported for Phi-1")

    try:
        if args.compile_only:
            logger.info("Compiling model only...")
            _ = run_phi1_inference(
                device=device,
                model_version=args.model_version,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                num_tokens_to_generate=1,
                demo_mode=args.demo_mode,
                model_config=model_config,
                enable_tracy=args.enable_tracy,
            )
            logger.info("Model compiled successfully.")
        else:
            logger.info("Starting Phi-1 inference demo...")
            metrics = run_phi1_inference(
                device=device,
                model_version=args.model_version,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                num_tokens_to_generate=args.num_tokens_to_generate,
                demo_mode=args.demo_mode,
                model_config=model_config,
                enable_tracy=args.enable_tracy,
            )

            # Report performance
            first_token_time = metrics.get("prefill_time", 0.0)
            report = prep_perf_report(
                model_name="Phi-1",
                batch_size=args.batch_size,
                device="Wormhole N150/N300",
                metrics=metrics,
                expected_perf_metrics={
                    "tokens_per_second_per_user": 14.0,  # Based on prior work
                },
                comments="",
            )
            report.display()

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    run_demo()