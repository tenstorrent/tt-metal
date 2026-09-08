# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Demo workload helpers for the TTTv2 Llama-3.1-8B path."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from loguru import logger

EncodePrompt = Callable[[str, bool], list[int]]
DecodePrompt = Callable[[list[int]], str]


def load_input_prompts(path: str | Path, batch_size: int, *, fallback_prompt: str = "What is the meaning of life?"):
    path = Path(path)
    if not path.exists():
        return [fallback_prompt] * batch_size

    with open(path) as f:
        data = json.load(f)

    prompts = (
        [entry["prompt"] for entry in data] if isinstance(data, list) else data.get("prompts", [data.get("prompt", "")])
    )
    while len(prompts) < batch_size:
        prompts = prompts * 2
    return prompts[:batch_size]


def tokenize_prompts_to_batch(
    prompts: Sequence[str],
    *,
    encode_fn: EncodePrompt,
    decode_fn: DecodePrompt | None,
    instruct: bool,
    max_seq_len: int,
    max_context_len: int,
    reserve_decode_tokens: int,
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_prefill_len = max_seq_len
    assert (
        max_prefill_len <= max_context_len
    ), f"max_prefill_len {max_prefill_len} cannot exceed max_context_len {max_context_len}"

    max_prefill_len -= reserve_decode_tokens
    assert (
        max_prefill_len > 0
    ), f"max_prefill_len ({max_prefill_len + reserve_decode_tokens}) must be greater than max_generated_tokens ({reserve_decode_tokens})"

    encoded_prompts = [encode_fn(prompt, instruct) for prompt in prompts]
    logger.info("Encoded prompt lengths:" + ", ".join(str(len(prompt)) for prompt in encoded_prompts))

    prompt_lens = [len(prompt) for prompt in encoded_prompts]
    min_prompt_len = min(prompt_lens)
    max_prompt_len = max(prompt_lens)

    if min_prompt_len > max_prefill_len:
        logger.info(f"Left-clipping prompts to {max_prefill_len}")
        if instruct:
            if decode_fn is None:
                raise ValueError("decode_fn is required to preserve instruct prompt clipping semantics")
            raw_prompts = [encode_fn(prompt, False) for prompt in prompts]
            overhead = [len(encoded) - len(raw) for encoded, raw in zip(encoded_prompts, raw_prompts)]

            shortened = []
            for raw_prompt, prompt_overhead in zip(raw_prompts, overhead):
                raw_budget = max_prefill_len - prompt_overhead
                if raw_budget <= 0:
                    raise ValueError(
                        f"max_prefill_len {max_prefill_len} leaves no room after chat template overhead {prompt_overhead}"
                    )
                shortened.append(decode_fn(raw_prompt[-raw_budget:]))

            encoded_prompts = [encode_fn(prompt, instruct) for prompt in shortened]
            assert all(
                len(encoded) == max_prefill_len for encoded in encoded_prompts
            ), f"Clipped prompts are not of the correct length, expected {max_prefill_len} but got {[len(e) for e in encoded_prompts]}"
        else:
            encoded_prompts = [encoded[-max_prefill_len:] for encoded in encoded_prompts]

        prompt_lens = [len(prompt) for prompt in encoded_prompts]
        min_prompt_len = min(prompt_lens)
        max_prompt_len = max(prompt_lens)

    assert max_prompt_len <= max_seq_len, f"Max prompt length {max_prompt_len} exceeds model max seq len {max_seq_len}"
    assert min_prompt_len > 0, "Minimum prompt length must be greater than 0"
    assert min_prompt_len <= max_prompt_len, f"Minimum prompt length {min_prompt_len} exceeds max len {max_prompt_len}"

    logger.info(f"# of users: {len(encoded_prompts)}")
    input_tokens = torch.full((len(encoded_prompts), max_prompt_len), pad_id, dtype=torch.int32)
    for idx, encoded in enumerate(encoded_prompts):
        input_tokens[idx, : len(encoded)] = torch.tensor(encoded, dtype=torch.int32)
    return input_tokens, torch.tensor(prompt_lens, dtype=torch.long)


def preprocess_llama3_8b_chat_prompts(
    prompts: Sequence[str],
    llm,
    *,
    reserve_decode_tokens: int = 128,
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    return tokenize_prompts_to_batch(
        prompts,
        encode_fn=lambda prompt, instruct: llm.encode_prompt(prompt, instruct=instruct),
        decode_fn=llm.tokenizer.decode,
        instruct=llm.instruct,
        max_seq_len=llm.max_seq_len,
        max_context_len=llm.max_context_len,
        reserve_decode_tokens=reserve_decode_tokens,
        pad_id=pad_id,
    )


def evaluate_seeded_cross_cardinality_consistency(
    outputs_by_cardinality: dict[int, dict[str, list[int]]],
    sequential_controls: dict[str, list[int]],
    *,
    request_ids: tuple[str, ...],
    expected_token_count: int,
    expected_cardinalities: tuple[int, ...] = (1, 2, 4, 32),
) -> tuple[str, tuple[dict[str, object], ...]]:
    """Validate a complete experiment and return its exact-token disposition.

    A complete token mismatch is a scientifically useful negative result rather
    than a malformed execution.  Missing, reordered, empty, or truncated output
    still fails closed and therefore cannot be recorded as a rejection verdict.
    """

    if tuple(outputs_by_cardinality) != expected_cardinalities:
        raise AssertionError(
            f"seeded cross-cardinality experiment expected {expected_cardinalities}, "
            f"got {tuple(outputs_by_cardinality)}"
        )
    if tuple(sequential_controls) != request_ids:
        raise AssertionError(
            "sequential controls must contain every fixed request in order: "
            f"expected {request_ids}, got {tuple(sequential_controls)}"
        )

    if expected_token_count <= 0:
        raise AssertionError("seeded cross-cardinality experiment requires a positive expected token count")
    bad_controls = {
        request_id: len(token_ids)
        for request_id, token_ids in sequential_controls.items()
        if len(token_ids) != expected_token_count
    }
    if bad_controls:
        raise AssertionError(
            f"sequential controls must each return {expected_token_count} generated tokens: {bad_controls}"
        )

    mismatches = []
    for cardinality, outputs in outputs_by_cardinality.items():
        expected_request_ids = request_ids[:cardinality]
        if tuple(outputs) != expected_request_ids:
            raise AssertionError(
                f"cardinality {cardinality} must contain the fixed request prefix "
                f"{expected_request_ids}, got {tuple(outputs)}"
            )
        for request_id, token_ids in outputs.items():
            control_token_ids = sequential_controls[request_id]
            if len(token_ids) != expected_token_count:
                raise AssertionError(
                    f"request {request_id!r} returned {len(token_ids)} tokens at cardinality {cardinality}; "
                    f"expected {expected_token_count}"
                )
            if token_ids != control_token_ids:
                mismatch_index = next(
                    (
                        index
                        for index, (actual, control) in enumerate(zip(token_ids, control_token_ids, strict=False))
                        if actual != control
                    ),
                    min(len(token_ids), len(control_token_ids)),
                )
                mismatches.append(
                    {
                        "cardinality": cardinality,
                        "request_id": request_id,
                        "first_token_difference": mismatch_index,
                        "control_token_count": len(control_token_ids),
                        "batched_token_count": len(token_ids),
                    }
                )

    verdict = "INVARIANT" if not mismatches else "BATCHED_PREFILL_REJECTED"
    return verdict, tuple(mismatches)
