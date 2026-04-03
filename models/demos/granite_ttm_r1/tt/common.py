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


def preprocess_parameters(hf_model: torch.nn.Module, device) -> Any:
    """Pre-process all model weights for TTNN.

    Returns a nested attribute-dict compatible with ttnn.model_preprocessing's
    output format, ready for use in TtnnGraniteTTMModel.
    """
    from ttnn.model_preprocessing import preprocess_model_parameters

    return preprocess_model_parameters(
        initialize_model=lambda: hf_model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )


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
    # torch.nn.LayerNorm is handled by the default preprocessor via
    # preprocess_layernorm_parameter, which reshapes weight/bias to (1, d_model)
    # in TILE_LAYOUT — satisfying the TTNN layer_norm gamma shape requirement.
    return parameters


import os as _os

_LINEAR_COMPUTE_CONFIG = None
_LINEAR_COMPUTE_CONFIG_LOFI = None

# Set TTNN_LOFI=1 in the environment to switch all linear layers to LoFi
# math fidelity.  Used for the E3 experiment in Stage 4.
_USE_LOFI = _os.getenv("TTNN_LOFI", "0") == "1"


def get_linear_compute_config():
    """Return a WormholeComputeKernelConfig tuned for TTM-R1 linear layers.

    HiFi2 math fidelity (default) keeps bfloat16 accumulation accurate enough
    for PCC ≥ 0.99 while enabling packer L1 accumulation for lower memory
    traffic.  math_approx_mode=True allows faster transcendentals.

    Set env var ``TTNN_LOFI=1`` to switch to LoFi (4-bit mantissa) which is
    5–10% faster but must be verified to preserve PCC ≥ 0.99.
    """
    global _LINEAR_COMPUTE_CONFIG, _LINEAR_COMPUTE_CONFIG_LOFI
    if _USE_LOFI:
        if _LINEAR_COMPUTE_CONFIG_LOFI is None:
            _LINEAR_COMPUTE_CONFIG_LOFI = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=True,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
        return _LINEAR_COMPUTE_CONFIG_LOFI
    if _LINEAR_COMPUTE_CONFIG is None:
        _LINEAR_COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
    return _LINEAR_COMPUTE_CONFIG


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
        # Detect the dtype of the module's parameters so we can cast inputs.
        _params = [p for p in module.parameters()]
        self._module_dtype: torch.dtype | None = _params[0].dtype if _params else None

    def __call__(self, *inputs, device=None, output_selector: str | None = None, **kwargs):
        torch_inputs = [to_torch_tensor(value) for value in inputs]
        # Cast inputs to match the module's parameter dtype to avoid mixed-dtype errors.
        if self._module_dtype is not None:
            torch_inputs = [t.to(self._module_dtype) if isinstance(t, torch.Tensor) else t for t in torch_inputs]
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
