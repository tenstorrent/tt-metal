# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Generator contract for the model-readiness check (and vLLM integration).

Every model that the readiness check can drive must expose a generator that
satisfies the `Generator` Protocol below. The same generator is the
delegation target for `tt/generator_vllm.py` for vLLM serving.

# Two API levels, one file

The contract has two layers:

1. **Low level** (`prefill_forward`, `decode_forward`) — used by
   `generator_vllm.py`. The caller (vLLM) owns the KV cache and page table
   and threads them through each call.

2. **High level** (`generate`) — used by demo scripts, the readiness check,
   and any HF-style host driver. The generator owns the KV cache and page
   table internally; the caller only sees token IDs in and token IDs out.

A typical implementation builds the low level first, then implements
`generate()` by allocating its own KV cache + page table and looping on top.

# Discovery

The readiness runner imports a generator by **convention**:

    <model_dir>/tt/generator.py  must define a module-level function

        def build_generator(model_dir: str | Path, mesh_device, **kwargs) -> Generator: ...

`model_dir` is the path the user passes on the CLI (e.g.
`models/autoports/llama31_8b_readiness_shim`). `**kwargs` is the escape valve for
per-model knobs (override_num_layers, max_seq_len, dtype, …).

# Teacher forcing

`generate()` takes an optional `next_input(step, predicted_token) -> token`
callback. After each decode step, the generator calls it and feeds the
returned token to the next step *instead of* its own prediction. The
readiness check just passes `TokenAccuracy.collect_predicted_tokens` as
this callback.

If `next_input is None`, the generator feeds its own prediction back
(HF-style autoregressive generation).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Protocol, runtime_checkable

import torch

NextInputFn = Callable[[int, int], int]
"""(step_index, model_predicted_token_id) -> token_id_to_feed_next"""


@runtime_checkable
class Generator(Protocol):
    """
    Required surface for a tt-metal generator usable by both vLLM and the
    readiness check.

    Implementations must expose the attributes and methods below. Extra
    methods (e.g. for multimodal preprocessing or model-specific warmup)
    are fine; the contract only mandates the minimum.
    """

    # --- Attributes ------------------------------------------------------

    tokenizer: Any
    """HuggingFace-compatible tokenizer. Must support .encode / .decode and
    chat templating if the model uses it. The readiness check uses it for
    debug decoding only; it does not assume a specific class."""

    # --- Low-level: caller-managed KV cache + page table ----------------

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_table: torch.Tensor,
        kv_cache: Any,
        prompt_lens: List[int],
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Run prefill on `tokens` and update `kv_cache` in place.

        Args:
            tokens:        [batch, prompt_len_padded] int token ids.
            page_table:    [batch, max_blocks] virtual-to-physical block map.
            kv_cache:      List-of-lists of TT tensors (layer × {K, V}), allocated by caller.
            prompt_lens:   [batch] real (unpadded) prompt length per user.

        Returns:
            Logits at the last prompt position, shape [batch, 1, vocab].
            May alternatively return sampled tokens [batch, 1] if the model
            samples on device — the readiness runner does not use this
            return value directly, but `generator_vllm.py` does.
        """

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        *,
        page_table: torch.Tensor,
        kv_cache: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Run one decode step on `tokens` and update `kv_cache` in place.

        Args:
            tokens:     [batch, 1] int token ids to decode.
            start_pos:  [batch] current position per user.
            page_table: [batch, max_blocks].
            kv_cache:   Same handles passed to `prefill_forward`.

        Returns:
            Logits at the current position, shape [batch, vocab].
            May alternatively return sampled tokens [batch] if sampling on
            device — same caveat as `prefill_forward`.
        """

    # --- High-level: generator-managed KV cache + page table ------------

    def generate(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int,
        *,
        next_input: Optional[NextInputFn] = None,
        **kwargs: Any,
    ) -> List[int]:
        """
        HF-style host driver: prefill on `prompt_token_ids`, then run
        greedy/argmax decode for up to `max_new_tokens` steps. Manages KV
        cache and page table internally.

        Args:
            prompt_token_ids: 1D list of prompt token ids (single user).
            max_new_tokens:   Decode-step budget. Generation may stop early
                              on EOS — see Returns.
            next_input:       Optional callback. If provided, after each
                              decode step the generator calls
                              `next_input(step, predicted_token)` and feeds
                              the returned token into the next step instead
                              of its own prediction. Use this for teacher
                              forcing.
            **kwargs:         Per-model extras (e.g. `stop_on_eos: bool`).
                              Implementations should ignore unknown kwargs.

        Returns:
            List of the model's predicted token ids, one per decode step
            attempted. Length is `<= max_new_tokens` (less if EOS hit, when
            `stop_on_eos=True`).

            The list contains the model's *own* predictions even when
            teacher forcing overrode the next input. The readiness check
            relies on this to score accuracy.
        """

    # --- Lifecycle ------------------------------------------------------

    def reset(self) -> None:
        """
        Wipe per-prompt state (KV cache, decode position counters, traces
        that depend on prompt length). Called between prompts by the
        readiness runner. After `reset()`, the generator must be ready to
        accept a new `generate()` or `prefill_forward()` call.

        Device weights, allocated KV cache buffers, and compiled traces
        should NOT be freed here — only their contents.
        """


# --- Abstract base (recommended parent for implementations) -------------


class GeneratorBase(ABC):
    """
    Abstract base class documenting the readiness-check Generator contract.

    Inheriting from `GeneratorBase` is **optional** — the runner only needs
    the `Generator` Protocol to be satisfied structurally. But concrete
    implementations are encouraged to subclass this ABC because:

    - The docstrings on each abstract method spell out the expected shapes,
      ownership rules, and side effects in detail. Subclassers see those
      in their IDE.
    - Missing methods cause a clean `TypeError` at construction rather than
      a late `AttributeError` deep inside the runner.
    - The skill (`.agents/skills/decoder-to-productized`) emits generators
      that subclass this base so the expected interface is explicit.

    A typical implementation:

        class LlamaGenerator(GeneratorBase):
            def __init__(self, mesh_device, model_args, ...):
                self._inner = tt_transformers.Generator(...)
                self._kv_cache = ...
                self._page_table = ...
                self.tokenizer = model_args.tokenizer

            def prefill_forward(self, tokens, *, page_table, kv_cache, prompt_lens, **kw):
                return self._inner.prefill_forward_text(...)

            def decode_forward(self, tokens, start_pos, *, page_table, kv_cache, **kw):
                return self._inner.decode_forward(...)

            def generate(self, prompt_token_ids, max_new_tokens, *, next_input=None, **kw):
                ...  # see method docstring

            def reset(self):
                ...  # zero the KV cache and clear cached state
    """

    #: HuggingFace-compatible tokenizer. Subclasses must set this in __init__.
    #: The readiness runner only uses .decode for debug output, so any
    #: tokenizer exposing .decode(list[int]) -> str is acceptable.
    tokenizer: Any = None

    @abstractmethod
    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_table: torch.Tensor,
        kv_cache: Any,
        prompt_lens: List[int],
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Run prefill and update `kv_cache` in place.

        This is the **low-level** API. vLLM (via `tt/generator_vllm.py`)
        calls this directly; the readiness runner does not. Implementations
        must:

        - Accept `tokens` of shape ``[batch, prompt_len_padded]``. The
          caller is responsible for padding to whatever length the model
          requires (typically a power of two ≥ 128).
        - Accept `page_table` of shape ``[batch, max_blocks]`` mapping
          virtual to physical KV blocks.
        - Accept `kv_cache` as the list-of-lists of TT tensors allocated
          by the caller (one ``[K, V]`` pair per decoder layer).
        - Accept `prompt_lens` as the **real** (unpadded) prompt length
          per user.
        - Return logits at the final prompt position with shape
          ``[batch, 1, vocab]``. If the implementation samples on device,
          it may instead return sampled tokens ``[batch, 1]``; this is
          allowed because `generator_vllm.py` consumes both forms, but the
          high-level `generate()` is responsible for normalising.

        Implementations must update `kv_cache` in place so the subsequent
        `decode_forward` calls can read prior keys/values.
        """

    @abstractmethod
    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        *,
        page_table: torch.Tensor,
        kv_cache: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Run one decode step and update `kv_cache` in place.

        - `tokens` is ``[batch, 1]`` — one input token per user.
        - `start_pos` is ``[batch]`` — current decode position per user.
        - `page_table` and `kv_cache` are the same handles passed to
          `prefill_forward`.
        - Return logits ``[batch, vocab]`` (or sampled tokens ``[batch]``
          on the on-device-sampling path).

        Like `prefill_forward`, this method must be safe to call
        repeatedly; advancing `start_pos` is the caller's responsibility.
        """

    @abstractmethod
    def generate(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int,
        *,
        next_input: Optional[NextInputFn] = None,
        **kwargs: Any,
    ) -> List[int]:
        """
        High-level HF-style host driver. Manages KV cache + page table
        internally so the caller only sees token IDs.

        Workflow:

        1. Run prefill on ``prompt_token_ids`` (a single user; the
           implementation may pad to the model's batch / sequence
           constraints).
        2. Take the argmax of the prefill logits at the last prompt
           position to produce the first prediction ``p0``.
        3. Call ``next_input(0, p0)`` if provided; otherwise the next
           input is ``p0`` itself.
        4. Loop ``max_new_tokens - 1`` more times: call `decode_forward`
           with the chosen input token, argmax to get ``p_i``, call
           ``next_input(i, p_i)`` to decide the next input.
        5. Append every ``p_i`` (the model's *own* prediction, regardless
           of whether teacher forcing overrode the next input) to the
           returned list and return when the loop ends or when EOS is hit
           (if the implementation chooses to stop early).

        The returned list is the basis the readiness check uses to score
        accuracy — implementations must return predictions, not the
        possibly-forced next inputs.

        Sampling must be greedy/argmax for readiness compatibility. The
        ``**kwargs`` slot is reserved for per-model extras (e.g.
        ``stop_on_eos`` toggles); implementations should ignore unknown
        kwargs.
        """

    @abstractmethod
    def reset(self) -> None:
        """
        Wipe per-prompt state between prompts.

        Must zero the KV cache contents and clear any cached page-table
        / decode-position state. Must **not** free device buffers,
        compiled traces, or reload weights — those should survive across
        calls so multiple prompts can be processed cheaply.

        After `reset()` returns, the next `generate()` (or
        `prefill_forward`) call must behave as if the generator had just
        been constructed.
        """


# --- Factory convention -------------------------------------------------

# `<model_dir>/tt/generator.py` MUST define:
#
#     def build_generator(model_dir: str | Path, mesh_device, **kwargs) -> Generator: ...
#
# The readiness runner imports this function dynamically and calls it.
# `model_dir` is the user-supplied path (absolute or repo-relative); the
# function may read its own config / weights from inside it.

BUILD_GENERATOR_FUNCTION_NAME = "build_generator"
"""Name of the required factory function in <model_dir>/tt/generator.py."""

GENERATOR_MODULE_RELPATH = "tt/generator.py"
"""Path of the generator file relative to <model_dir>."""


BuildGeneratorFn = Callable[..., Generator]
"""Signature: build_generator(model_dir: str | Path, mesh_device, **kwargs) -> Generator"""


def _ensure_protocol(_: Generator) -> None:
    """Type-check sentinel; not invoked at runtime."""


__all__ = [
    "BUILD_GENERATOR_FUNCTION_NAME",
    "BuildGeneratorFn",
    "GENERATOR_MODULE_RELPATH",
    "Generator",
    "GeneratorBase",
    "NextInputFn",
]
