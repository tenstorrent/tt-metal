# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Shared Chronos configs used by the PyTorch reference and TTNN stubs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal


@dataclass
class ChronosModelConfig:
    """Bring-up config aligned with the dummy Chronos T5 tokenizer settings."""

    tokenizer_class: str = "MeanScaleUniformBins"
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=lambda: {"low_limit": -15.0, "high_limit": 15.0})
    n_tokens: int = 32
    n_special_tokens: int = 2
    pad_token_id: int = 0
    eos_token_id: int = 1
    use_eos_token: bool = True
    model_type: Literal["causal", "seq2seq"] = "seq2seq"
    context_length: int = 512
    prediction_length: int = 64
    num_samples: int = 20
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    pretrained_model_name_or_path: str = "amazon/chronos-t5-tiny"
