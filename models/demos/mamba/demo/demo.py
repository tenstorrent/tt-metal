# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time
import argparse
from typing import List, Optional
from loguru import logger
import ttnn
import pytest
import torch
import json
from tqdm import tqdm

from transformers import AutoTokenizer

from models.demos.mamba.reference.decode_model import MambaPretrainedModelName


def get_cpu_reference_model(version: MambaPretrainedModelName, batch_size: int):
    from models.demos.mamba.reference.decode_model import MambaDecode

    return MambaDecode.from_pretrained(version, batch_size=batch_size)


def get_tt_metal_model(
    version: MambaPretrainedModelName, device: ttnn.Device, cache_dir: Optional[str] = None, batch_size: int = 32
):
    from models.demos.mamba.tt.full_model import MambaTT
    from models.demos.mamba.tt import model_config

    reference_model = get_cpu_reference_model(version, batch_size=batch_size)
    if cache_dir:
        cache_path = model_config.get_weights_cache_path(version, cache_dir)
    else:
        cache_path = None

    config = model_config.create_model_config(batch_size, reference_model.args.d_model)
    model = MambaTT(reference_model, device, config, tt_cache_path=cache_path)

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

    model = get_tt_metal_model(model_version, device, cache_dir, batch_size)

    model.eval()

    tokenizer = get_tokenizer()

    sequences: torch.Tensor = tokenizer(prompts, return_tensors="pt", padding=True).input_ids

    # prefill
    prefill_iterations = sequences.shape[1] - 1
    for idx in tqdm(range(prefill_iterations), desc="Prefilling the prompt(s)..."):
        logits = model(sequences[:, idx].unsqueeze(1))

    # Decode Starts
    start = time.time()
    token_counts = 0
    total_iterations = generated_sequence_length + prefill_iterations

    print("Starting decoding...")
    for idx in range(prefill_iterations, total_iterations):
        with torch.no_grad():
            logits = model(sequences[:, idx].unsqueeze(1))
        logits = apply_repetition_penalty_(logits.squeeze(1), sequences, penalty=1.2)  # Adjust penalty as needed
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1)
        sequences = torch.cat([sequences, next_token.unsqueeze(-1)], dim=1)

        decoded = tokenizer.batch_decode(sequences, skip_special_tokens=False)

        token_counts += sequences.shape[0]
        throughput = token_counts / (time.time() - start)

        if display:
            display_tokens(decoded)
            print(f"Current total throughput: {throughput:.2f} tok/s")
            print(f"Current throughput per user: {(throughput/batch_size):.2f} tok/s/u")


@pytest.mark.parametrize(
    "max_gen_len",
    ([100]),
)
def test_demo(user_input, device, use_program_cache, max_gen_len):
    return run_mamba_demo(prompts=user_input, device=device, generated_sequence_length=max_gen_len)
