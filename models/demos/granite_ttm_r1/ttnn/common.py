# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

import ttnn

from models.demos.granite_ttm_r1.reference.model import extract_prediction_tensor
from models.demos.granite_ttm_r1.reference.preprocess import build_reference_inputs


def to_ttnn_tensor(
    tensor: torch.Tensor,
    *,
    device=None,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    return ttnn.from_torch(tensor, device=device, dtype=dtype, layout=layout, memory_config=memory_config)


def to_torch_tensor(tensor: ttnn.Tensor | torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, torch.Tensor):
        return tensor
    return ttnn.to_torch(tensor)


def preprocess_inputs(history: torch.Tensor, observed_mask: torch.Tensor | None = None, *, device=None):
    ttnn_history = to_ttnn_tensor(history, device=device, dtype=ttnn.bfloat16)
    ttnn_mask = None
    if observed_mask is not None:
        ttnn_mask = to_ttnn_tensor(observed_mask, device=device, dtype=ttnn.bfloat16)
    return ttnn_history, ttnn_mask


def custom_preprocessor(torch_model, name):
    parameters = {}
    if isinstance(torch_model, torch.nn.Linear):
        parameters["weight"] = preprocess_linear_weight(torch_model.weight, dtype=ttnn.bfloat16)
        parameters["bias"] = (
            preprocess_linear_bias(torch_model.bias, dtype=ttnn.bfloat16) if torch_model.bias is not None else None
        )
    elif isinstance(torch_model, torch.nn.LayerNorm):
        parameters["weight"] = ttnn.from_torch(torch_model.weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        parameters["bias"] = ttnn.from_torch(torch_model.bias, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    return parameters


def resolve_module(root_module: torch.nn.Module, candidates: list[str]) -> torch.nn.Module | None:
    named_modules = dict(root_module.named_modules())
    for candidate in candidates:
        if candidate in named_modules:
            return named_modules[candidate]
        current = root_module
        found = True
        for attr in candidate.split("."):
            if attr.isdigit():
                index = int(attr)
                if not hasattr(current, "__getitem__"):
                    found = False
                    break
                current = current[index]
            elif hasattr(current, attr):
                current = getattr(current, attr)
            else:
                found = False
                break
        if found and isinstance(current, torch.nn.Module):
            return current
    return None


class TorchModuleFallback:
    def __init__(self, module: torch.nn.Module):
        self.module = module.eval()

    def __call__(self, *inputs, device=None, output_selector: str | None = None, **kwargs):
        torch_inputs = [to_torch_tensor(value) for value in inputs]
        torch_kwargs = {
            key: to_torch_tensor(value) if isinstance(value, ttnn.Tensor) else value for key, value in kwargs.items()
        }
        outputs = self.module(*torch_inputs, **torch_kwargs)
        if output_selector is not None:
            outputs = getattr(outputs, output_selector)
        elif not isinstance(outputs, torch.Tensor):
            outputs = extract_prediction_tensor(outputs)
        return to_ttnn_tensor(outputs, device=device, dtype=ttnn.bfloat16)


def run_model_as_torch_fallback(
    reference_model: torch.nn.Module,
    history: ttnn.Tensor | torch.Tensor,
    *,
    observed_mask: ttnn.Tensor | torch.Tensor | None = None,
    future_values: ttnn.Tensor | torch.Tensor | None = None,
    device=None,
    extra_inputs: dict[str, Any] | None = None,
):
    torch_history = to_torch_tensor(history)
    torch_mask = to_torch_tensor(observed_mask) if observed_mask is not None else None
    torch_future = to_torch_tensor(future_values) if future_values is not None else None
    inputs = build_reference_inputs(
        reference_model,
        torch_history,
        future_values=torch_future,
        observed_mask=torch_mask,
        extra_inputs=extra_inputs,
    )
    outputs = reference_model(**inputs)
    prediction = extract_prediction_tensor(outputs)
    return to_ttnn_tensor(prediction, device=device, dtype=ttnn.bfloat16)
