# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time
import argparse
from typing import List
from loguru import logger

import torch

from transformers import AutoTokenizer

from models.experimental.mamba.reference.decode_model import MambaPretrainedModelName


def get_cpu_reference_model(version: MambaPretrainedModelName, batch_size: int):
    from models.experimental.mamba.reference.decode_model import MambaDecode

    return MambaDecode.from_pretrained(version, batch_size=batch_size)


def get_tt_metal_model(version: MambaPretrainedModelName, use_cache: bool, batch_size: int):
    import tt_lib
    from models.experimental.mamba.tt.full_model import MambaTT

    device = tt_lib.device.CreateDevice(0)
    device.disable_and_clear_program_cache()
    tt_lib.device.SetDefaultDevice(device)
    device = tt_lib.device.GetDefaultDevice()
    reference_model = get_cpu_reference_model(version, batch_size=batch_size)
    if use_cache:
        cache_path = f"/tmp/{version}"
    else:
        cache_path = ""
    model = MambaTT(reference_model, device, tt_cache_path=cache_path)
    return model, device


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", padding=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def display_tokens(tokens: List[str]):
    print("\n" * 1000)
    for text in tokens:
        print(f"{text}\n")
        print("-" * 80)  # Print a separator line for readability
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


def run_demo(
    prompts: List[str],
    model_type: str,
    model_version: MambaPretrainedModelName = "state-spaces/mamba-2.8b-slimpj",
    generated_sequence_length: int = 50,
    display: bool = True,
    use_cache: bool = True,
):
    batch_size = len(prompts)

    logger.info(f"Running Mamba demo (weights='{model_version}') on '{model_type.upper()}' with batch={batch_size}")

    if model_type == "cpu":
        model = get_cpu_reference_model(model_version, batch_size)
    elif model_type == "wh":
        model, _ = get_tt_metal_model(model_version, use_cache, batch_size)
    else:
        raise RuntimeError("Invalid model type was encountered")

    tokenizer = get_tokenizer()

    sequences: torch.Tensor = tokenizer(prompts, return_tensors="pt", padding=True).input_ids

    all_decoded_sequences = []
    count = 0
    start = time.time()
    for idx in range(generated_sequence_length + sequences.shape[1]):
        logits = model(sequences[:, idx].unsqueeze(1))
        count += sequences.shape[0]
        if idx >= sequences.shape[1] - 1:
            logits = apply_repetition_penalty_(logits.squeeze(1), sequences, penalty=1.2)  # Adjust penalty as needed
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1)
            sequences = torch.cat([sequences, next_token.unsqueeze(-1)], dim=1)

            decoded = tokenizer.batch_decode(sequences, skip_special_tokens=False)
            all_decoded_sequences.append(decoded)

            if display:
                display_tokens(decoded)
                _ = count / (time.time() - start)
                # print(f"Current throughput: {throughput:.2f} tok/s")
        else:
            logger.info(f"Decoding the prompt(s)... ({idx + 1}/{sequences.shape[1]})")

    return all_decoded_sequences


def main():
    parser = argparse.ArgumentParser(description="Run inference benchmarks on set of supported models")
    parser.add_argument("prompts", nargs="+")
    parser.add_argument("--model", choices=["cpu", "wh"], default="wh", help="The model under test")
    args = parser.parse_args()
    run_demo(args.prompts, args.model)


if __name__ == "__main__":
    main()
