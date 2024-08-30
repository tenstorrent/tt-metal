# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time
import textwrap
from typing import List, Optional, Callable
from loguru import logger
import ttnn
import pytest
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from dataclasses import dataclass

from models.demos.wormhole.mamba.reference.decode_model import MambaPretrainedModelName
from models.demos.wormhole.mamba.reference.args import ModelMode
from models.demos.wormhole.mamba.tt import model_config
from models.demos.wormhole.mamba.tt.preprocessing import (
    split_sequence_length,
    select_chunk_size,
    split_input_into_prefill_and_decode_segments,
)

from models.perf.benchmarking_utils import BenchmarkProfiler
from models.demos.utils.llm_demo_utils import create_benchmark_data, verify_perf


class TokenDisplay:
    def __init__(self):
        self.sequences: List[str] = []

    def add_token(self, tokens: List[str]):
        for i, token in enumerate(tokens):
            if i >= len(self.sequences):
                self.sequences.append(token)
            else:
                self.sequences[i] += token
                text = self.sequences[i]
                eos = text.find("<|endoftext|>")
                if eos != -1:
                    text = text[:eos] + "<|endoftext|>"

    def display_sequences(self):
        print("\033[H\033[J", end="")  # Clear the screen
        for i, sequence in enumerate(self.sequences):
            wrapped_sequence = "\n".join(textwrap.wrap(sequence, width=90))
            print(f"User {i + 1}:\n{wrapped_sequence}\n")


def get_cpu_reference_model(version: MambaPretrainedModelName, batch_size: int):
    from models.demos.wormhole.mamba.reference.decode_model import MambaDecode

    return MambaDecode.from_pretrained(version, batch_size=batch_size)


def get_tt_metal_model(
    version: MambaPretrainedModelName,
    device: ttnn.Device,
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
    mode: ModelMode = ModelMode.DECODE,
    seq_len: int = 1,
    num_layers: int = 64,
):
    from models.demos.wormhole.mamba.tt.mamba_model import MambaTT
    from models.demos.wormhole.mamba.tt import model_config

    reference_model = get_cpu_reference_model(version, batch_size=batch_size)
    config = model_config.create_model_config(batch_size, reference_model.args.d_model, mode=mode, seq_len=seq_len)
    model = MambaTT(reference_model, device, config, tt_cache_path=cache_dir, num_layers=num_layers)

    return model


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", padding=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


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


@dataclass
class PrefillRunStatisitics:
    total_time: float
    mean_throughput: float
    mean_throughput_per_user: float


def run_mamba_prefill(device, model, sequences, prefill_chunk_size):
    prefill_tokens = sequences
    num_users, prefill_length = prefill_tokens.shape

    prefill_tokens = ttnn.from_torch(
        prefill_tokens.view(1, 1, prefill_tokens.shape[0], prefill_tokens.shape[1]),
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.uint32,
    )
    total_prefill_time = 0
    for user_idx in tqdm(range(num_users), desc="Prefilling the prompt(s)..."):
        prefill_start = time.time()
        with torch.no_grad():
            for chunk in split_sequence_length(prefill_tokens, batch=user_idx, chunk_size=prefill_chunk_size):
                chunk = ttnn.reshape(chunk, [1, chunk.shape[3]])  # Mamba expects (1, L) in prefill mode
                model._forward(chunk)
        model.configs["current_user"] += 1
        prefill_end = time.time()
        prefill_time_per_user = prefill_end - prefill_start
        total_prefill_time += prefill_time_per_user

    total_prefill_tokens = prefill_length * num_users
    mean_throughput = total_prefill_tokens / total_prefill_time
    mean_throughput_per_user = mean_throughput / num_users

    return PrefillRunStatisitics(
        total_time=total_prefill_time,
        mean_throughput=mean_throughput,
        mean_throughput_per_user=mean_throughput_per_user,
    )


@dataclass
class DecodeRunStatisitics:
    total_time: float
    mean_throughput: float
    mean_throughput_per_user: float


def run_mamba_decode(
    model, input_tokens, batch_size, generated_sequence_length, callback: Callable[[torch.Tensor, float], None]
):
    num_input_tokens = input_tokens.shape[1]
    assert num_input_tokens >= 1, "Expected at least one input token to run Mamba decode"

    num_prefill_by_decode_tokens = num_input_tokens - 1
    total_length = num_input_tokens + generated_sequence_length - 1

    with torch.no_grad():
        for token_idx in range(0, num_prefill_by_decode_tokens):
            start = time.time()
            model(input_tokens[:, token_idx].unsqueeze(1))
            end = time.time()

            next_token = input_tokens[:, token_idx + 1]  # Force next token instead of using predicted one
            callback(next_token, end - start)

        total_decode_time = 0
        total_decode_tokens = 0
        next_token = input_tokens[:, -1]
        for token_idx in range(num_prefill_by_decode_tokens, total_length):
            start = time.time()
            logits = model(next_token.unsqueeze(1))
            end = time.time()

            logits = apply_repetition_penalty_(logits.squeeze(1), input_tokens, penalty=1.4)  # Adjust penalty as needed
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1)
            callback(next_token, end - start)

            total_decode_time += end - start
            total_decode_tokens += 1

    assert (
        generated_sequence_length == total_decode_tokens
    ), f"Expected to generate {generated_sequence_length} tokens (generated {total_decode_tokens} tokens)"

    mean_decode_time = total_decode_time / total_decode_tokens
    return DecodeRunStatisitics(
        total_time=total_decode_time,
        mean_throughput=(batch_size / mean_decode_time),
        mean_throughput_per_user=(1 / mean_decode_time),
    )


def split_into_tokens_into_prefill_and_decode_slices(tokenized_prompts):
    prefill_chunk_size = select_chunk_size(tokenized_prompts.shape[1], model_config.MAMBA_MAX_SEQUENCE_LEN)
    if prefill_chunk_size == 0:
        prefill_chunk_size = 32  # If there is no valid chunk size use the smalles possible value
    prefill_tokens, decode_tokens = split_input_into_prefill_and_decode_segments(tokenized_prompts, prefill_chunk_size)

    if prefill_tokens is None:
        num_prefill_tokens = 0
    else:
        num_prefill_tokens = prefill_tokens.shape[1]
    num_decode_tokens = decode_tokens.shape[1]
    return prefill_chunk_size, num_prefill_tokens, num_decode_tokens, prefill_tokens, decode_tokens


@dataclass
class DemoResult:
    generated_text: List[str]


def run_mamba_demo(
    prompts: List[str],
    device: ttnn.Device,
    model_version: MambaPretrainedModelName = "state-spaces/mamba-2.8b-slimpj",
    batch_size: int = 32,
    generated_sequence_length: int = 50,
    cache_dir: Optional[str] = None,
    display: bool = True,
    prefill_chunk_size: int = 32,
):
    profiler = BenchmarkProfiler()
    profiler.start("run")

    logger.info(f"Running Mamba demo (weights='{model_version}') with batch={batch_size}")
    logger.info(f"Using tensor cache at '{cache_dir}'")

    profiler.start("tokenizing_inputs")

    assert (
        len(prompts) == 1 or len(prompts) == batch_size
    ), f"Expected number of prompts equal to batch (was {batch_size}) or 1"
    prompts = prompts * batch_size if len(prompts) == 1 else prompts

    tokenizer = get_tokenizer()
    tokenized_prompts: torch.Tensor = tokenizer(prompts, return_tensors="pt", padding=True).input_ids
    logger.info(f"Input prompts tokenized (shape={list(tokenized_prompts.shape)})")

    (
        prefill_chunk_size,
        num_prefill_tokens,
        num_decode_tokens,
        prefill_tokens,
        decode_tokens,
    ) = split_into_tokens_into_prefill_and_decode_slices(tokenized_prompts)
    logger.info(f"Selected prefill chunk size of {prefill_chunk_size} tokens")
    logger.info(f"Will use prefill mode for {num_prefill_tokens} tokens and decode mode for {num_decode_tokens} tokens")

    profiler.end("tokenizing_inputs")

    logger.info(f"Initalizing Mamba model in prefill mode")
    profiler.start("loading_model")
    model = get_tt_metal_model(
        model_version,
        device,
        cache_dir,
        batch_size=1,
        mode=ModelMode.PREFILL,
        seq_len=prefill_chunk_size,
        num_layers=64,
    )
    profiler.end("loading_model")
    logger.info(f"Done initializing model in {profiler.get_duration('loading_model'):.2f} s")

    model.eval()

    profiler.start("compile_prefill")
    if num_prefill_tokens > 0:
        logger.info("Compiling prefill graph")
        prefill_model_config = model_config.create_model_config(
            1, model.args.d_model, mode=ModelMode.PREFILL, seq_len=prefill_chunk_size
        )
        model.to_prefill(prefill_model_config)

        prefill_tokens = tokenized_prompts[:, :-1]  # Omit the last token in the sequence (B, L - 1)
        run_mamba_prefill(device, model, prefill_tokens[:1, :], prefill_chunk_size)
    profiler.end("compile_prefill")

    logger.info("Compiling decode graph")
    decode_model_config = model_config.create_model_config(
        batch_size, model.args.d_model, mode=ModelMode.DECODE, seq_len=1
    )
    model.to_decode(decode_model_config)

    profiler.start("compile_decode")
    run_mamba_decode(
        model,
        tokenized_prompts[:, :1],
        batch_size,
        1,
        callback=lambda _, t: logger.info(f"Decode compilation took {t:.2f} seconds"),
    )
    profiler.end("compile_decode")

    model.reset()

    token_display = TokenDisplay()
    token_display.add_token(prompts)

    profiler.start("inference_prefill_decode")
    profiler.start("inference_prefill")
    if num_prefill_tokens > 0:
        assert prefill_tokens is not None, "Expected prefill tokens"
        prefill_model_config = model_config.create_model_config(
            1, model.args.d_model, mode=ModelMode.PREFILL, seq_len=prefill_chunk_size
        )
        model.to_prefill(prefill_model_config)
        logger.info(f"Running prefill with {prefill_tokens.shape[-1]} tokens")
        prefill_stats = run_mamba_prefill(device, model, prefill_tokens, prefill_chunk_size)
    else:
        prefill_stats = PrefillRunStatisitics(float("inf"), float("inf"), float("inf"))
    profiler.end("inference_prefill")

    # Decode
    decode_model_config = model_config.create_model_config(
        batch_size, model.args.d_model, mode=ModelMode.DECODE, seq_len=1
    )
    model.to_decode(decode_model_config)
    profiler.start("inference_decode")
    logger.info(f"Running decode with {decode_tokens.shape[-1]} tokens")

    def callback(token: torch.Tensor, inference_time: float) -> None:
        decoded_token: list[str] = tokenizer.batch_decode(token, skip_special_tokens=False)
        token_display.add_token(decoded_token)
        if display:
            token_display.display_sequences()
        throughput = batch_size / inference_time
        print(f"Current total decode throughput: {throughput:.2f} tok/s")
        print(f"Current decode throughput per user: {(throughput/batch_size):.2f} tok/s/u")

    decode_stats = run_mamba_decode(
        model,
        decode_tokens,
        batch_size,
        generated_sequence_length,
        callback=callback,
    )
    profiler.end("inference_decode")
    profiler.end("inference_prefill_decode")
    profiler.end("run")

    logger.info(f"Total demo duration: {(profiler.get_duration('run')):.2f} s")

    prefill_time_to_token_per_user = prefill_stats.mean_throughput_per_user
    decode_time_to_token_per_user = decode_stats.mean_throughput_per_user

    measurements = {
        "total_demo_time": profiler.get_duration("run"),
        "compile_prefill": profiler.get_duration("compile_prefill"),
        "compile_decode": profiler.get_duration("compile_decode"),
        "inference_prefill": prefill_stats.total_time,
        "inference_decode": decode_stats.total_time,
        "prefill_t/s": prefill_stats.mean_throughput,
        "prefill_time_to_token": prefill_stats.total_time,
        "decode_t/s": decode_stats.mean_throughput,
        "decode_t/s/u": decode_stats.mean_throughput_per_user,
        "prefill_decode_t/s/u": 1 / (prefill_time_to_token_per_user + decode_time_to_token_per_user),  # t/s/u
        "token_verification": 1,  # This is checked by the caller - but we could also do a match here
    }

    logger.info(f"Prefill total time: {prefill_stats.total_time:.1f} s")
    logger.info(
        f"Prefill throughput: {prefill_stats.mean_throughput:.1f} t/s, {prefill_stats.mean_throughput_per_user:.2f} t/s/u"
    )
    logger.info(f"Decode total time: {decode_stats.total_time:.1f} s")
    logger.info(
        f"Decode throughput: {decode_stats.mean_throughput:.1f} t/s, {decode_stats.mean_throughput_per_user:.2f} t/s/u"
    )
    logger.info(f"Time to first token: {(1e3 * measurements['prefill_decode_t/s/u']):.2f} ms")

    chunk_size_to_prefill_targets_tok_per_s = {32: 135.0, 128: 270.0}  # perf is different for different chunk sizes
    targets = {
        "prefill_t/s": chunk_size_to_prefill_targets_tok_per_s[prefill_chunk_size],
        "decode_t/s": 235.0,
        "decode_t/s/u": 7.3,
    }
    warmup_iterations = {"inference_prefill": 0, "inference_decode": 0}
    benchmark_data = create_benchmark_data(profiler, measurements, warmup_iterations, targets)
    benchmark_data.prep_csvs(
        profiler,
        run_type=f"demo_perf",
        ml_model_name=model_version,
        ml_model_type="llm",
        num_layers=64,
        batch_size=batch_size,
        config_params={"prefill_chunk_size": prefill_chunk_size},
        precision=f"decode[{decode_model_config['dtype']}]",
        input_sequence_length=tokenized_prompts.shape[1],
        output_sequence_length=tokenized_prompts.shape[1] + generated_sequence_length,
    )

    verify_perf(measurements, targets)

    return DemoResult(generated_text=token_display.sequences)


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "model_version, max_gen_len",
    (
        (
            "state-spaces/mamba-2.8b-slimpj",
            50,
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
