# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared setup of the CPU model tests: the pinned Transformers ``Qwen4Exp`` oracle, the QSA test rotary tables and
the MRoPE prompt builder."""

import importlib
import os
import sys
from pathlib import Path

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.mrope import (
    IMAGE_TOKEN_ID,
    VISION_END_TOKEN_ID,
    VISION_START_TOKEN_ID,
    Qwen38ImageGrid,
)

TRANSFORMERS_SRC = os.environ.get("QWEN38_TRANSFORMERS_SRC")
requires_transformers = pytest.mark.skipif(
    not TRANSFORMERS_SRC, reason="set QWEN38_TRANSFORMERS_SRC to pinned Transformers source"
)


def qwen4_exp_transformers():
    """The pinned Transformers ``(configuration_qwen4_exp, modeling_qwen4_exp)`` modules."""

    source = str(Path(TRANSFORMERS_SRC))
    if source not in sys.path:
        sys.path.insert(0, source)
    package = "transformers.models.qwen4_exp"
    return (
        importlib.import_module(f"{package}.configuration_qwen4_exp"),
        importlib.import_module(f"{package}.modeling_qwen4_exp"),
    )


def qsa_position_embeddings(sequence: int) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(sequence, dtype=torch.float32).view(1, sequence, 1)
    angles = positions * torch.linspace(0.001, 0.037, 32).view(1, 1, 32)
    return (
        torch.cat((angles.cos(), angles.cos()), dim=-1).to(torch.bfloat16),
        torch.cat((angles.sin(), angles.sin()), dim=-1).to(torch.bfloat16),
    )


def qsa_causal_mask(sequence: int) -> torch.Tensor:
    visible = torch.ones(sequence, sequence, dtype=torch.bool).tril()
    minimum = torch.tensor(torch.finfo(torch.float32).min)
    return torch.where(visible, torch.tensor(0.0), minimum).view(1, 1, sequence, sequence)


def mrope_prompt(*parts) -> list[int]:
    """The ids of ``parts`` in order: a ``Qwen38ImageGrid`` is its pad run between the vision markers, any other part a
    sequence of token ids."""

    ids = []
    for part in parts:
        if isinstance(part, Qwen38ImageGrid):
            ids += [VISION_START_TOKEN_ID] + [IMAGE_TOKEN_ID] * part.merged_tokens + [VISION_END_TOKEN_ID]
        else:
            ids += list(part)
    return ids
