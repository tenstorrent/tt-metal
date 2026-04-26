# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import OrderedDict
from contextlib import ExitStack
from typing import Any

import torch

from models.demos.granite_ttm_r1.common import load_granite_ttm_reference_model
from models.demos.granite_ttm_r1.reference.preprocess import build_reference_inputs


def extract_prediction_tensor(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        return outputs
    if isinstance(outputs, (tuple, list)):
        for item in outputs:
            if isinstance(item, torch.Tensor):
                return item
    for key in ("prediction_outputs", "predictions", "forecast", "logits", "last_hidden_state"):
        value = getattr(outputs, key, None)
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(outputs, dict) and isinstance(outputs.get(key), torch.Tensor):
            return outputs[key]
    raise TypeError(f"Unable to extract prediction tensor from output type {type(outputs)!r}")


class GraniteTTMReferenceModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model.eval()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(load_granite_ttm_reference_model(*args, **kwargs))

    def forward(self, history: torch.Tensor, *, future_values: torch.Tensor | None = None, **extra_inputs):
        inputs = build_reference_inputs(self.model, history, future_values=future_values, extra_inputs=extra_inputs)
        return self.model(**inputs)

    def predict(self, history: torch.Tensor, *, future_values: torch.Tensor | None = None, **extra_inputs) -> torch.Tensor:
        outputs = self.forward(history, future_values=future_values, **extra_inputs)
        return extract_prediction_tensor(outputs)

    def module_tree(self) -> str:
        return str(self.model)

    def capture_intermediates(
        self,
        history: torch.Tensor,
        *,
        module_names: list[str],
        future_values: torch.Tensor | None = None,
        **extra_inputs,
    ) -> tuple[Any, OrderedDict[str, torch.Tensor]]:
        captures: OrderedDict[str, torch.Tensor] = OrderedDict()

        def register_hook(name: str, module: torch.nn.Module):
            def hook(_module, _args, output):
                if isinstance(output, torch.Tensor):
                    captures[name] = output.detach().cpu()
                elif isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
                    captures[name] = output[0].detach().cpu()

            return module.register_forward_hook(hook)

        with ExitStack() as stack:
            for name, module in self.model.named_modules():
                if name in module_names:
                    stack.enter_context(register_hook(name, module))
            outputs = self.forward(history, future_values=future_values, **extra_inputs)

        return outputs, captures
