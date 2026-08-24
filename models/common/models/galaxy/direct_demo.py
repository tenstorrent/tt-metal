# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Model-neutral direct demo over a Galaxy `(8, 4)` 2D tensor model.

Milestone B's "direct demo" is the product path before the model-owned executor
exists: load a checkpoint, prefill prompts, decode, detokenize. Everything here
works from the handle contract both reconstructions expose — ``model``,
``tokenizer``, ``generation_config``, ``encode_prompt`` — so neither model
package needs its own copy of the loop.

**Unqualified.** Nothing here has run on a Galaxy mesh.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from models.common.models.galaxy.direct_runner import GalaxyDirectRunner, GalaxySamplingPolicy

#: Short, deterministic prompts. Batch-32 demos repeat them to fill the mesh.
DEFAULT_DEMO_PROMPTS: tuple[str, ...] = (
    "Explain what a tensor is to a software engineer in two sentences.",
    "Name three prime numbers greater than one hundred.",
    "Write a one-line summary of the water cycle.",
    "What is the capital of Portugal, and why is it on a river?",
    "Give one reason distributed training needs collective communication.",
    "Describe the difference between latency and throughput.",
    "List two advantages of paged attention for serving.",
    "In one sentence, what does a rotary position embedding do?",
)


@dataclass(frozen=True)
class GalaxyDirectDemoResult:
    """One slot's demo output."""

    slot: int
    prompt: str
    text: str
    tokens: tuple[int, ...]
    finished: bool


def fill_prompt_slots(prompts: Sequence[str], slots: int) -> tuple[str, ...]:
    """Repeat ``prompts`` up to ``slots`` entries, preserving their order."""

    if not prompts:
        raise ValueError("at least one prompt is required")
    return tuple(prompts[index % len(prompts)] for index in range(slots))


def run_direct_demo(
    handle: Any,
    *,
    prompts: Sequence[str],
    max_new_tokens: int = 32,
    policy: GalaxySamplingPolicy = GalaxySamplingPolicy(),
    batched_prefill: bool = False,
    instruct: bool | None = None,
) -> list[GalaxyDirectDemoResult]:
    """Generate a continuation for each prompt and return the decoded text."""

    tokenizer = handle.tokenizer
    encoded = [list(handle.encode_prompt(prompt, instruct=instruct)) for prompt in prompts]
    stop_token_ids = tuple(getattr(handle.generation_config, "stop_token_ids", ()) or ())
    with GalaxyDirectRunner(handle.model, stop_token_ids=stop_token_ids) as runner:
        generations = runner.generate(
            encoded,
            max_new_tokens=max_new_tokens,
            policy=policy,
            batched_prefill=batched_prefill,
        )
    results: list[GalaxyDirectDemoResult] = []
    for prompt, generation in zip(prompts, generations):
        tokens = tuple(generation.generated_tokens)
        results.append(
            GalaxyDirectDemoResult(
                slot=generation.slot,
                prompt=prompt,
                text=tokenizer.decode(list(tokens), skip_special_tokens=True),
                tokens=tokens,
                finished=generation.finished,
            )
        )
    return results


__all__ = [
    "DEFAULT_DEMO_PROMPTS",
    "GalaxyDirectDemoResult",
    "fill_prompt_slots",
    "run_direct_demo",
]
