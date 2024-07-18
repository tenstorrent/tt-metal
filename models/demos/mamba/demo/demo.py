# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time
from typing import List, Optional
from loguru import logger
import ttnn
import pytest
import torch
from tqdm import tqdm

from transformers import AutoTokenizer

from models.demos.mamba.reference.decode_model import MambaPretrainedModelName
from models.demos.mamba.reference.args import ModelMode
from models.demos.mamba.tt import model_config
from models.demos.mamba.tt.preprocessing import split_sequence_length


def get_cpu_reference_model(version: MambaPretrainedModelName, batch_size: int):
    from models.demos.mamba.reference.decode_model import MambaDecode

    return MambaDecode.from_pretrained(version, batch_size=batch_size)


def get_tt_metal_model(
    version: MambaPretrainedModelName,
    device: ttnn.Device,
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
    mode: ModelMode = ModelMode.DECODE,
    seq_len: int = 1,
):
    from models.demos.mamba.tt.full_model import MambaTT
    from models.demos.mamba.tt import model_config

    reference_model = get_cpu_reference_model(version, batch_size=batch_size)
    config = model_config.create_model_config(batch_size, reference_model.args.d_model, mode=mode, seq_len=seq_len)
    model = MambaTT(reference_model, device, config, tt_cache_path=cache_dir)

    return model


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", padding=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def display_tokens(tokens: List[str]):
    print("\n" * 1000)
    for text in tokens:
        eos = text.find("<|endoftext|>")
        if eos != -1:
            text = text[:eos] + "<|endoftext|>"
        print(f"{text}\n")
        print("-" * 150)  # Print a separator line for readability
        print(f"\n")


def apply_repetition_penalty_(logits, sequences, penalty=1.2):
    """
    Applies a repetition penalty to the logits.

    Args:
        logits (torch.Tensor): The logits from the model, shape [batch_size, vocab_size].
        sequences (torch.Tensor): The generated sequences so far, shape [batch_size, sequence_length].
        penalty (float): The penalty factor for repeated tokens (values > 1 discourage repetition).
    """
    batch_size, vocab_size = logits.shape
    for i in range(batch_size):
        for j in range(sequences.shape[1]):
            token_id = sequences[i, j]
            if token_id < vocab_size:  # Protect against potential out-of-bounds errors
                logits[i, token_id] /= penalty  # Decrease the probability of already generated tokens
    return logits


def run_mamba_demo(
    prompts: List[str],
    device: ttnn.Device,
    model_version: MambaPretrainedModelName = "state-spaces/mamba-2.8b-slimpj",
    batch_size: int = 32,
    generated_sequence_length: int = 50,
    cache_dir: Optional[str] = None,
    display: bool = True,
):
    if len(prompts) == 1:
        prompts = prompts * batch_size  # Duplicate the prompt to fill the batch

    assert batch_size == len(prompts), "32 prompts are required"

    logger.info(f"Running Mamba demo (weights='{model_version}') with batch={batch_size}")
    logger.info(f"Using tensor cache at '{cache_dir}'")

    model = get_tt_metal_model(model_version, device, cache_dir, batch_size)

    model.eval()

    tokenizer = get_tokenizer()

    sequences: torch.Tensor = tokenizer(prompts, return_tensors="pt", padding=True).input_ids

    # Prefill
    prefill_iterations = sequences.shape[1] - 1
    for idx in tqdm(range(prefill_iterations), desc="Prefilling the prompt(s)..."):
        logits = model(sequences[:, idx].unsqueeze(1))

    # Decode
    total_iterations = generated_sequence_length + prefill_iterations

    logger.info("Starting decoding...")
    for idx in range(prefill_iterations, total_iterations):
        with torch.no_grad():
            start = time.time()
            logits = model(sequences[:, idx].unsqueeze(1))
            end = time.time()

        logits = apply_repetition_penalty_(logits.squeeze(1), sequences, penalty=1.2)  # Adjust penalty as needed
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1)

        sequences = torch.cat([sequences, next_token.unsqueeze(-1)], dim=1)
        decoded = tokenizer.batch_decode(sequences, skip_special_tokens=False)

        throughput = batch_size / (end - start)

        if display:
            display_tokens(decoded)
            print(f"Current total throughput: {throughput:.2f} tok/s")
            print(f"Current throughput per user: {(throughput/batch_size):.2f} tok/s/u")


def run_mamba_prefill_decode_demo(
    prompts: List[str],
    device: ttnn.Device,
    model_version: MambaPretrainedModelName = "state-spaces/mamba-2.8b-slimpj",
    batch_size: int = 32,
    generated_sequence_length: int = 50,
    cache_dir: Optional[str] = None,
    display: bool = True,
):
    if len(prompts) == 1:
        prompts = prompts * batch_size  # Duplicate the prompt to fill the batch

    assert batch_size == len(prompts), "32 prompts are required"

    tokenizer = get_tokenizer()
    sequences: torch.Tensor = tokenizer(prompts, return_tensors="pt", padding=True).input_ids
    prefill_length = sequences.shape[1] - 1
    logger.info(f"Prefilling the prompt(s) with {prefill_length} tokens...")

    logger.info(f"Running Mamba demo (weights='{model_version}') with batch={batch_size}")
    logger.info(f"Using tensor cache at '{cache_dir}'")
    model = get_tt_metal_model(
        model_version, device, cache_dir, batch_size=1, mode=ModelMode.PREFILL, seq_len=prefill_length
    )
    model.eval()

    # Prefill
    model.to_prefill()
    prefill_chunk_size = 32
    num_users = sequences.shape[0]

    prefill_tokens = sequences[:, :-1]  # Omit the last token in the sequence (B, L - 1)

    prefill_tokens = ttnn.from_torch(
        prefill_tokens.view(1, 1, prefill_tokens.shape[0], prefill_tokens.shape[1]),
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.uint32,
    )
    for user_idx in tqdm(range(num_users), desc="Prefilling the prompt(s)..."):
        with torch.no_grad():
            for chunk in split_sequence_length(prefill_tokens, batch=user_idx, chunk_size=prefill_chunk_size):
                chunk = ttnn.reshape(chunk, [1, chunk.shape[3]])  # Mamba expects (1, L) in prefill mode
                model._forward(chunk)
        model.configs["current_user"] += 1

    # Decode
    decode_model_config = model_config.create_model_config(
        batch_size, model.args.d_model, mode=ModelMode.DECODE, seq_len=1
    )
    model.to_decode(decode_model_config)

    total_length = prefill_length + generated_sequence_length
    logger.info("Starting decoding...")
    for token_idx in range(prefill_length, total_length):
        with torch.no_grad():
            start = time.time()
            logits = model(sequences[:, token_idx].unsqueeze(1))
            end = time.time()

        logits = apply_repetition_penalty_(logits.squeeze(1), sequences, penalty=1.2)  # Adjust penalty as needed
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1)

        sequences = torch.cat([sequences, next_token.unsqueeze(-1)], dim=1)
        decoded = tokenizer.batch_decode(sequences, skip_special_tokens=False)

        throughput = batch_size / (end - start)

        if display:
            display_tokens(decoded)
            print(f"Current total throughput: {throughput:.2f} tok/s")
            print(f"Current throughput per user: {(throughput/batch_size):.2f} tok/s/u")


@pytest.mark.parametrize(
    "model_version, max_gen_len",
    (
        (
            "state-spaces/mamba-2.8b-slimpj",
            100,
        ),
    ),
)
def test_demo(user_input, device, use_program_cache, get_tt_cache_path, model_version, max_gen_len):
    return run_mamba_demo(
        prompts=user_input,
        device=device,
        cache_dir=get_tt_cache_path(model_version),
        generated_sequence_length=max_gen_len,
    )
