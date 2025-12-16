"""Base TTNNModule class for TTNN-accelerated neural network modules."""

import functools

import torch
from torch.utils._pytree import tree_map

import ttnn


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
        from models.tt_symbiote.core.tensor import TorchTTNNTensor

        print(f"{self.__class__.__name__}: {self.module_name} on device {self.device}")

        def wrap(e):
            result = TorchTTNNTensor(e) if isinstance(e, torch.Tensor) and not isinstance(e, TorchTTNNTensor) else e
            if not isinstance(e, TorchTTNNTensor) and isinstance(e, ttnn.Tensor):
                result = TorchTTNNTensor(e)
            return result

        def to_ttnn_wrap(e):
            if isinstance(e, TorchTTNNTensor):
                e = e.to_ttnn
            return e

        def set_device_wrap(e):
            if isinstance(e, ttnn.Tensor) and self.device is not None and e.device() != self.device:
                e = ttnn.to_device(e, self.device)
            return e

        result = None
        if self.device is not None:
            func_args = tree_map(wrap, args)
            func_kwargs = tree_map(wrap, kwds)
            func_args = tree_map(to_ttnn_wrap, func_args)
            func_kwargs = tree_map(to_ttnn_wrap, func_kwargs)
            func_args = tree_map(set_device_wrap, func_args)
            func_kwargs = tree_map(set_device_wrap, func_kwargs)
            self.preprocess_weights()
            self.move_weights_to_device()
            try:
                result = self.forward(*func_args, **func_kwargs)
                result = tree_map(wrap, result)
            except Exception as e:
                print(f"Error {e} in {self.__class__.__name__} forward, falling back to torch")
                assert (
                    self.torch_layer is not None
                ), f"torch_layer must be set for fallback, {self} does not have torch_layer set."
                result = self.torch_layer(*args, **kwds)
        else:
            print("Device not set, falling back to torch")
            assert (
                self.torch_layer is not None
            ), f"torch_layer must be set for fallback, {self} does not have torch_layer set."
            result = self.torch_layer(*args, **kwds)
        return result

    def preprocess_weights(self):
        """Preprocess weights (called once before first use)."""
        if not self._preprocessed_weight:
            self._preprocessed_weight = True
        else:
            return
        self.preprocess_weights_impl()

    def move_weights_to_device(self):
        """Move preprocessed weights to device."""
        assert self._preprocessed_weight, "Weights must be preprocessed before moving to device."
        assert self.device is not None, "Device must be set before moving weights to device."
        if not self._weights_on_device:
            self._weights_on_device = True
        else:
            return
        self.move_weights_to_device_impl()

    def move_weights_to_host(self):
        """Move weights back to host."""
        self.move_weights_to_host_impl()
        self._weights_on_device = False

    def deallocate_weights(self):
        """Deallocate weights from device."""
        self.deallocate_weights_impl()
        self._preprocessed_weight = False
        self._weights_on_device = False

    def preprocess_weights_impl(self):
        """Override to implement weight preprocessing."""

    def move_weights_to_host_impl(self):
        """Override to implement weight movement to host."""

    def move_weights_to_device_impl(self):
        """Override to implement weight movement to device."""

    def deallocate_weights_impl(self):
        """Override to implement weight deallocation."""

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
        return f"{self.__class__.__name__}(module_name={self.module_name})"

    @property
    def torch_layer(self):
        """Get fallback PyTorch layer."""
        return self._fallback_torch_layer

    def forward(self, *args, **kwargs):
        """Forward pass - must be implemented by subclasses."""
        raise NotImplementedError("Forward method must be implemented by subclasses.")


def deallocate_weights_after(func):
    """Decorator to deallocate weights after forward pass."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        self.deallocate_weights()
        return result

    return wrapper
