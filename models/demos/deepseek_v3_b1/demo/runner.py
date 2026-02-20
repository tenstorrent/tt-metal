# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


class ModelLike(Protocol):
    def start(self) -> None:
        ...

    def prefill(self, prompt_tokens: list[Any]) -> Any:
        ...

    def decode_step(self, input_tensor: Any) -> Any:
        ...

    def stop(self) -> None:
        ...


@dataclass(frozen=True)
class GenerationResult:
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    generated_text: str


def tokenize_prompt(tokenizer: Any, prompt: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        token_ids = tokenizer.encode(prompt, add_special_tokens=True)
    else:
        encoded = tokenizer(prompt, add_special_tokens=True, return_attention_mask=False)
        token_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if len(token_ids) > 0 and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def ensure_non_empty_prompt_tokens(prompt_token_ids: list[int], bos_token_id: int | None) -> list[int]:
    if len(prompt_token_ids) > 0:
        return prompt_token_ids
    if bos_token_id is None:
        raise ValueError("Prompt tokenization produced zero tokens and tokenizer has no bos_token_id fallback.")
    return [int(bos_token_id)]


def run_generation(
    *,
    model: ModelLike,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    make_input_tensor: Callable[[int], Any],
    extract_token_id: Callable[[Any], int],
    write_text: Callable[[str], None] | None = None,
    on_before_decode: Callable[[], None] | None = None,
) -> GenerationResult:
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")

    write_text = write_text or (lambda _: None)
    on_before_decode = on_before_decode or (lambda: None)
    prompt_token_ids = ensure_non_empty_prompt_tokens(
        tokenize_prompt(tokenizer, prompt),
        getattr(tokenizer, "bos_token_id", None),
    )
    prefill_inputs = [make_input_tensor(token_id) for token_id in prompt_token_ids]

    started = False
    try:
        model.start()
        started = True

        last_prefill_output = model.prefill(prefill_inputs)
        next_token_id = extract_token_id(last_prefill_output)
        on_before_decode()

        generated_token_ids: list[int] = []
        generated_chunks: list[str] = []
        for _ in range(max_new_tokens):
            decode_input = make_input_tensor(next_token_id)
            decode_output = model.decode_step(decode_input)
            next_token_id = extract_token_id(decode_output)

            generated_token_ids.append(next_token_id)
            chunk = tokenizer.decode([next_token_id], skip_special_tokens=False)
            generated_chunks.append(chunk)
            write_text(chunk)

        return GenerationResult(
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=generated_token_ids,
            generated_text="".join(generated_chunks),
        )
    finally:
        if started:
            model.stop()
