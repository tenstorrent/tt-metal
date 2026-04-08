# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Base TTNNModule class for TTNN-accelerated neural network modules."""

import functools

import torch

from models.experimental.tt_symbiote.core.run_config import get_tensor_run_implementation

TENSOR_RUN_IMPLEMENTATION = get_tensor_run_implementation()


class TTNNModule:
    """Base class for TTNN-accelerated modules with automatic fallback to PyTorch."""

    def __init__(self):
        self._device = None  # Device can be set later
        self._preprocessed_weight = False
        self._weights_on_device = False
        self._fallback_torch_layer = None
        self._unique_name = None
        self._model_config = {}

    def set_model_config(self, model_config):
        """Set model configuration dictionary."""
        self._model_config = model_config if model_config is not None else {}

    def __call__(self, *args, **kwds):
        return TENSOR_RUN_IMPLEMENTATION.module_run(self, *args, **kwds)

    def preprocess_weights(self):
        """Preprocess weights (called once before first use)."""
        if not self._preprocessed_weight:
            self._preprocessed_weight = True
        else:
            return
        self.preprocess_weights_impl()

    def move_weights_to_device(self):
        """Move preprocessed weights to device."""
        assert (
            self._preprocessed_weight
        ), f"Weights must be preprocessed for {self.module_name} before moving to device."
        assert self.device is not None, f"Device must be set for {self.module_name} before moving weights to device."
        if not self._weights_on_device:
            self._weights_on_device = True
        else:
            return
        self.move_weights_to_device_impl()

    def deallocate_weights(self):
        """Deallocate weights from device."""
        self.deallocate_weights_impl()
        self._weights_on_device = False

    def preprocess_weights_impl(self):
        """Override to implement weight preprocessing."""

        for child in self.__dict__.values():
            if isinstance(child, TTNNModule):
                child.preprocess_weights()
        return self

    def move_weights_to_device_impl(self):
        """Override to implement weight movement to device."""

        for child in self.__dict__.values():
            if isinstance(child, TTNNModule):
                child.move_weights_to_device()
        return self

    def deallocate_weights_impl(self):
        """Override to implement weight deallocation."""
        for child in self.__dict__.values():
            if isinstance(child, TTNNModule):
                child.deallocate_weights()
        return self

    def to_device(self, device):
        """Set the device for this module."""
        self._device = device
        return self

    @property
    def model_config(self):
        """Get model configuration."""
        return self._model_config

    @property
    def device(self):
        """Get current device."""
        return self._device

    @classmethod
    def from_torch(cls, torch_layer, *args, **kwargs):
        """Create TTNN module from PyTorch module."""
        new_layer = cls(*args, **kwargs)
        new_layer._fallback_torch_layer = torch_layer
        return new_layer

    def train(self, mode: bool = True):
        """Set training mode (TTNN modules don't have training/eval modes)."""
        if self.torch_layer is not None:
            self.torch_layer.train(mode)
        return self

    @property
    def module_name(self):
        """Get unique module name."""
        if self._unique_name is None:
            self._unique_name = f"{self.__class__.__name__}_{id(self)}"
        return self._unique_name

    def __repr__(self):
        # recursive representation
        # similar to pytorch nn.Module __repr__
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

    @property
    def torch_layer(self):
        """Get fallback PyTorch layer."""
        return self._fallback_torch_layer

    def forward(self, *args, **kwargs):
        """Forward pass - must be implemented by subclasses."""
        raise NotImplementedError("Forward method must be implemented by subclasses.")

    def named_modules(self, memo=None, prefix="", remove_duplicate=True):
        """Iterator over all modules in the network, yielding both the name of the module as well as the module itself."""
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


def deallocate_weights_after(func):
    """Decorator to deallocate weights after forward pass."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        self.deallocate_weights()
        return result

    return wrapper
