# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Stub generator for memory-profiling GRPO training.

Exposes :class:`StubLlamaGRPOCompleter`, a :class:`LlamaGRPOCompleter`
subclass whose ``generate`` / ``generate_str`` skip autoregressive decode
entirely and return a fixed canned completion for every prompt x replica.
``compute_nlog_probs`` and the underlying ttml model are inherited unchanged,
so the training-side forward / backward pass is exercised exactly as it would
be with the real completer; only the generation phase is short-circuited.

Use this when memory-profiling GRPO training to isolate optimizer + forward +
backward allocations from the generation KV cache and decode activations.

Typical wiring inside a memprof driver::

    from utils.llama_completer import LlamaCompletionCtx
    from utils.llama_completer_stub import StubLlamaGRPOCompleter

    completer = StubLlamaGRPOCompleter(
        ctx=LlamaCompletionCtx(
            max_tokens_to_complete=grpo_config.max_completion_length,
            temperature=grpo_config.temperature,
            completions_per_prompt=grpo_config.num_generations,
        ),
        transformer_config=transformer_config,
        device_config=device_config,
        model_source=MODEL_ID,
    )
"""

from __future__ import annotations

from typing import List, Optional

from .llama_completer import LlamaGRPOCompleter


CANNED_COMPLETION_TEXT: str = (
    "Mr. Benson had a 5% discount for each of the 12 - 10 = <<12-10=2>>2 tickets.\n"
    "So, those two tickets had a $40 x 5/100 = $<<40*5/100=2>>2 discount each.\n"
    "Hence, each ticket cost $40 - $2 = $<<40-2=38>>38 each.\n"
    "Thus, two discounted tickets amount to $38 x 2 = $<<38*2=76>>76.\n"
    "And the other ten tickets amount to $40 x 10 = $<<40*10=400>>400.\n"
    "Hence, Mr. Benson paid a total of $400 + $76 = $<<400+76=476>>476.\n"
    "#### 476"
)


class StubLlamaGRPOCompleter(LlamaGRPOCompleter):
    """``LlamaGRPOCompleter`` whose ``generate`` returns a canned completion.

    The canned text is tokenised lazily on first call with the completer's
    own tokenizer (so it matches whatever model was loaded).
    ``completions_per_prompt`` from :class:`LlamaCompletionCtx` still governs
    the output count (``len(prompts) * completions_per_prompt``), so
    downstream batching code in ``GRPOTrainer`` is identical to the real path.

    The canned completion is truncated to fit
    ``min(ctx.max_tokens_to_complete, max_sequence_length - max_prompt_len)``,
    mirroring the cap the real ``generate`` applies before its decode loop.
    """

    def __init__(self, *args, canned_text: str = CANNED_COMPLETION_TEXT, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._canned_text: str = canned_text
        self._canned_tokens: Optional[List[int]] = None

    def _get_canned_tokens(self) -> List[int]:
        if self._canned_tokens is None:
            self._canned_tokens = list(self.tokenizer.encode(self._canned_text, add_special_tokens=False))
        return self._canned_tokens

    def _truncated_canned_tokens(self, prompts: List[List[int]]) -> List[int]:
        canned = self._get_canned_tokens()
        cap = self._ctx.max_tokens_to_complete
        if prompts:
            max_prompt = max(len(p) for p in prompts)
            cap = min(cap, self.transformer_config.max_sequence_length - max_prompt)
        # compute_nlog_probs requires len(prompt + completion) >= 2; with
        # well-formed prompts of length >= 2 a single canned token is enough.
        cap = max(cap, 1)
        return list(canned[:cap]) if cap < len(canned) else list(canned)

    def generate(self, prompts: List[List[int]]) -> List[List[int]]:
        tokens = self._truncated_canned_tokens(prompts)
        out_count = len(prompts) * self._ctx.completions_per_prompt
        return [list(tokens) for _ in range(out_count)]

    def generate_str(self, prompt_strs: List[str]) -> List[str]:
        out_count = len(prompt_strs) * self._ctx.completions_per_prompt
        return [self._canned_text for _ in range(out_count)]
