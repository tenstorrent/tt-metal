# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic, bounded-memory synthetic Gemma4 decoder weights.

The stats file is collected from canonical real checkpoint tensors.  A caller
can either request meta tensors to validate the complete real-shape loading
contract without allocating checkpoint-sized storage, or stream reproducible
BF16 values in fixed-size chunks for numerical tests.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path

import torch

STATS_PATH = Path(__file__).parents[1] / "doc" / "functional_decoder" / "real_weight_stats.json"


def load_weight_stats() -> dict:
    return json.loads(STATS_PATH.read_text())


def canonical_layer_state_dict(layer_idx: int, *, device: str = "meta") -> dict[str, torch.Tensor]:
    """Return every canonical layer key at its real target shape.

    ``device="meta"`` is the normal CI path. It exercises all transpose,
    concatenate, split, reshape, and key accesses in ``from_state_dict`` with
    no large allocation. Numerical consumers should use
    :func:`iter_synthetic_weight_chunks`.
    """
    specs = load_weight_stats()["layers"][str(layer_idx)]["weights"]
    prefix = f"model.language_model.layers.{layer_idx}."
    return {
        prefix + name: torch.empty(spec["shape"], dtype=torch.bfloat16, device=device) for name, spec in specs.items()
    }


def synthetic_layer_state_dict(layer_idx: int) -> dict[str, torch.Tensor]:
    """Materialize a deterministic numerical state at every real target shape."""
    specs = load_weight_stats()["layers"][str(layer_idx)]["weights"]
    prefix = f"model.language_model.layers.{layer_idx}."
    return {prefix + name: synthetic_weight(spec) for name, spec in specs.items()}


def iter_synthetic_weight_chunks(spec: dict, *, chunk_elements: int | None = None) -> Iterator[torch.Tensor]:
    """Yield a complete deterministic flattened tensor without a full allocation."""
    if spec["dtype"] != "bf16":
        raise ValueError(f"unsupported synthetic checkpoint dtype: {spec['dtype']}")
    if math.prod(spec["shape"]) != spec["numel"]:
        raise ValueError("weight stats shape and numel disagree")
    chunk_elements = chunk_elements or load_weight_stats()["generator"]["chunk_elements"]
    generator = torch.Generator(device="cpu").manual_seed(spec["seed"])
    remaining = spec["numel"]
    while remaining:
        count = min(remaining, chunk_elements)
        if spec["std"] == 0.0:
            chunk = torch.full((count,), spec["mean"], dtype=torch.bfloat16)
        else:
            chunk = torch.normal(
                mean=spec["mean"],
                std=spec["std"],
                size=(count,),
                generator=generator,
            ).to(torch.bfloat16)
        yield chunk
        remaining -= count


def synthetic_weight(spec: dict) -> torch.Tensor:
    """Materialize one real-shape synthetic tensor, bounded to tensor + one chunk."""
    flattened = torch.empty(spec["numel"], dtype=torch.bfloat16)
    offset = 0
    for chunk in iter_synthetic_weight_chunks(spec):
        flattened[offset : offset + chunk.numel()].copy_(chunk)
        offset += chunk.numel()
    return flattened.reshape(spec["shape"])
