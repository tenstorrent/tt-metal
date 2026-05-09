# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNNModule base class for hybrid TTNN-accelerated modules.

Adapted from tt-symbiote's core/module.py. Provides:
- from_torch() classmethod for HuggingFace module replacement
- Weight lifecycle: preprocess_weights -> move_weights_to_device -> deallocate_weights
- Automatic child module traversal for weight management
"""

from __future__ import annotations

import functools
from typing import Any

import torch


def set_module_name_recursively(module: "TTNNModule", prefix: str = "") -> None:
    for name, child in module.__dict__.items():
        if isinstance(child, TTNNModule):
            child._unique_name = f"{prefix}.{name}"
            child.override_children_module_names()
        elif isinstance(child, torch.nn.Module):
            pass
        elif isinstance(child, dict):
            for k, v in child.items():
                if isinstance(v, TTNNModule):
                    v._unique_name = f"{prefix}.{name}[{k}]"
                    v.override_children_module_names()
        elif isinstance(child, (list, tuple)):
            for i, v in enumerate(child):
                if isinstance(v, TTNNModule):
                    v._unique_name = f"{prefix}.{name}[{i}]"
                    v.override_children_module_names()


class TTNNModule:
    """Base class for TTNN-accelerated modules with HuggingFace integration.

    Subclasses override:
    - forward() for the compute path
    - preprocess_weights_impl() for one-time weight conversion
    - move_weights_to_device_impl() to push weights to TT device
    """

    def __init__(self):
        self._device = None
        self._preprocessed_weight = False
        self._weights_on_device = False
        self._fallback_torch_layer = None
        self._unique_name = None
        self._model_config: dict[str, Any] = {}

    def set_model_config(self, model_config: dict | None) -> None:
        self._model_config = model_config if model_config is not None else {}

    def __call__(self, *args, **kwargs):
        if not self._preprocessed_weight:
            self.preprocess_weights()
        if not self._weights_on_device and self._device is not None:
            self.move_weights_to_device()
        return self.forward(*args, **kwargs)

    # --- Weight lifecycle ---

    def preprocess_weights(self) -> None:
        if self._preprocessed_weight:
            return
        self._preprocessed_weight = True
        self.preprocess_weights_impl()

    def move_weights_to_device(self) -> None:
        assert (
            self._preprocessed_weight
        ), f"Weights must be preprocessed for {self.module_name} before moving to device."
        assert self.device is not None, f"Device must be set for {self.module_name} before moving weights."
        if self._weights_on_device:
            return
        self._weights_on_device = True
        self.move_weights_to_device_impl()

    def deallocate_weights(self) -> None:
        self.deallocate_weights_impl()
        self._weights_on_device = False

    def preprocess_weights_impl(self) -> None:
        for child in self.__dict__.values():
            if isinstance(child, TTNNModule):
                child.preprocess_weights()

    def move_weights_to_device_impl(self) -> None:
        for child in self.__dict__.values():
            if isinstance(child, TTNNModule):
                child.move_weights_to_device()

    def deallocate_weights_impl(self) -> None:
        for child in self.__dict__.values():
            if isinstance(child, TTNNModule):
                child.deallocate_weights()

    # --- Device management ---

    def to_device(self, device: Any) -> "TTNNModule":
        self._device = device
        return self

    def set_device_recursive(self, device: Any) -> None:
        self._device = device
        for child in self.__dict__.values():
            if isinstance(child, TTNNModule):
                child.set_device_recursive(device)
            elif isinstance(child, dict):
                for v in child.values():
                    if isinstance(v, TTNNModule):
                        v.set_device_recursive(device)
            elif isinstance(child, (list, tuple)):
                for v in child:
                    if isinstance(v, TTNNModule):
                        v.set_device_recursive(device)

    @property
    def model_config(self) -> dict:
        return self._model_config

    @property
    def device(self) -> Any:
        return self._device

    @classmethod
    def from_torch(cls, torch_layer, *args, **kwargs) -> "TTNNModule":
        new_layer = cls(*args, **kwargs)
        new_layer._fallback_torch_layer = torch_layer
        return new_layer

    @property
    def torch_layer(self):
        return self._fallback_torch_layer

    @property
    def module_name(self) -> str:
        if self._unique_name is None:
            self._unique_name = f"{self.__class__.__name__}_{id(self)}"
        return self._unique_name

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward method must be implemented by subclasses.")

    def override_children_module_names(self) -> None:
        set_module_name_recursively(self, self.module_name)

    def named_modules(self, memo=None, prefix="", remove_duplicate=True):
        if memo is None:
            memo = set()
        if remove_duplicate:
            if self in memo:
                return
            memo.add(self)
        yield prefix, self
        for name, child in self.__dict__.items():
            if isinstance(child, (torch.nn.Module, TTNNModule)):
                child_prefix = prefix + ("." if prefix else "") + name
                yield from child.named_modules(memo, child_prefix, remove_duplicate)

    def named_children(self):
        for name, child in self.__dict__.items():
            if name in ("_fallback_torch_layer", "torch_layer"):
                continue
            if isinstance(child, (torch.nn.Module, TTNNModule)):
                yield name, child
            elif isinstance(child, dict):
                for k, v in child.items():
                    if isinstance(v, (torch.nn.Module, TTNNModule)):
                        yield f"{name}[{k}]", v
            elif isinstance(child, (list, tuple)):
                for i, v in enumerate(child):
                    if isinstance(v, (torch.nn.Module, TTNNModule)):
                        yield f"{name}[{i}]", v

    def __repr__(self) -> str:
        child_lines = []
        for key, value in self.__dict__.items():
            if isinstance(value, (torch.nn.Module, TTNNModule)) and key != "_fallback_torch_layer":
                mod_str = repr(value).replace("\n", "\n  ")
                child_lines.append(f"({key}): {mod_str}")
        main_str = f"{self.__class__.__name__}(module_name={self.module_name}"
        if child_lines:
            main_str += "\n  " + "\n  ".join(child_lines) + "\n"
        main_str += ")"
        return main_str


def deallocate_weights_after(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        self.deallocate_weights()
        return result

    return wrapper
