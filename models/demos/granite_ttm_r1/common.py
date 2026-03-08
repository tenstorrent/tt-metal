# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoConfig, AutoModel

DEFAULT_MODEL_NAME = "ibm-granite/granite-timeseries-ttm-r1"
DEFAULT_CONTEXT_LENGTH = 512
DEFAULT_FORECAST_LENGTH = 96
DEFAULT_NUM_CHANNELS = 1


@dataclass
class GraniteTTMExample:
    history: torch.Tensor
    future_values: torch.Tensor | None = None
    observed_mask: torch.Tensor | None = None


def load_granite_ttm_config(model_name: str = DEFAULT_MODEL_NAME, cache_dir: str | None = None):
    return AutoConfig.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)


def load_granite_ttm_reference_model(
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    cache_dir: str | None = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    local_files_only: bool | None = None,
):
    if local_files_only is None:
        local_files_only = os.getenv("CI") == "true"

    model = AutoModel.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    return model.eval()


def get_forward_parameter_names(model: torch.nn.Module) -> list[str]:
    return list(inspect.signature(model.forward).parameters.keys())


def infer_num_channels(config: Any, default: int = DEFAULT_NUM_CHANNELS) -> int:
    for key in ("num_input_channels", "input_size", "num_channels", "feature_size"):
        value = getattr(config, key, None)
        if isinstance(value, int) and value > 0:
            return value
    return default


def create_synthetic_example(
    *,
    batch_size: int = 1,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    forecast_length: int = DEFAULT_FORECAST_LENGTH,
    num_channels: int = DEFAULT_NUM_CHANNELS,
    dtype: torch.dtype = torch.float32,
) -> GraniteTTMExample:
    history = torch.randn(batch_size, context_length, num_channels, dtype=dtype)
    future_values = torch.randn(batch_size, forecast_length, num_channels, dtype=dtype)
    observed_mask = torch.ones_like(history, dtype=dtype)
    return GraniteTTMExample(history=history, future_values=future_values, observed_mask=observed_mask)
