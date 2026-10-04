# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Rollout samplers for ``GRPOTrainer``, selected by ``GRPOConfig.rollout_source``."""

from __future__ import annotations

from typing import Any

from .base import RolloutBatch, RolloutSampler

ROLLOUT_SOURCES = ("ttml",)


def build_rollout_sampler(
    source: str,
    *,
    transformer_config: Any,
    device_config: Any,
    model_source: str,
    max_completion_length: int,
    temperature: float,
    completions_per_prompt: int,
) -> RolloutSampler:
    """Build the :class:`RolloutSampler` named by ``source``.

    ``"ttml"`` builds a :class:`TTMLRolloutSampler`, which opens the device and
    loads the ttml model and tokenizer for ``transformer_config.model_type``
    from ``model_source``.
    """
    if source == "ttml":
        # Imported lazily so ``import ttml.trainers`` doesn't pull in transformers / huggingface_hub.
        from .ttml_rollout_sampler import TTMLRolloutSampler

        return TTMLRolloutSampler(
            model_kind=transformer_config.model_type,
            transformer_config=transformer_config,
            device_config=device_config,
            model_source=model_source,
            max_completion_length=max_completion_length,
            temperature=temperature,
            completions_per_prompt=completions_per_prompt,
        )
    raise ValueError(f"unknown rollout_source {source!r}; expected one of {list(ROLLOUT_SOURCES)}")


__all__ = ["ROLLOUT_SOURCES", "RolloutBatch", "RolloutSampler", "build_rollout_sampler"]
