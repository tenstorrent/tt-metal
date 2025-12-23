"""TorchTTNNTensor: A PyTorch tensor subclass with TTNN backend support."""

import torch

from models.tt_symbiote.core.run_config import get_tensor_run_implementation

TENSOR_RUN_IMPLEMENTATION = get_tensor_run_implementation()


class TorchTTNNTensor(torch.Tensor):
    """PyTorch tensor wrapper that can dispatch operations to TTNN backend."""

    elem: torch.Tensor

    __slots__ = ["elem"]

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        return TENSOR_RUN_IMPLEMENTATION.new_instance(cls, elem, *args, **kwargs)

    def __repr__(self):
        return TENSOR_RUN_IMPLEMENTATION.repr(self)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        return TENSOR_RUN_IMPLEMENTATION.torch_dispatch(cls, func, types, args, kwargs)

    @property
    def shape(self):
        return self.elem.shape if self.elem is not None else tuple(int(i) for i in self.ttnn_tensor.shape)

    def __mul__(self, other):
        return TENSOR_RUN_IMPLEMENTATION.torch_dispatch(
            TorchTTNNTensor, torch.ops.aten.mul.Tensor, None, (self, other), {}
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return TENSOR_RUN_IMPLEMENTATION.torch_dispatch(
            TorchTTNNTensor, torch.ops.aten.sub.Tensor, None, (self, other), {}
        )

    def __rsub__(self, other):
        return TENSOR_RUN_IMPLEMENTATION.torch_dispatch(
            TorchTTNNTensor, torch.ops.aten.sub.Tensor, None, (other, self), {}
        )

    def __add__(self, other):
        return TENSOR_RUN_IMPLEMENTATION.torch_dispatch(
            TorchTTNNTensor, torch.ops.aten.add.Tensor, None, (self, other), {}
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __abs__(self):
        raise RuntimeError("Absolute value is not yet implemented for TTNN tensors.")

    def __matmul__(self, other):
        return TENSOR_RUN_IMPLEMENTATION.torch_dispatch(
            TorchTTNNTensor, torch.ops.aten.mm.default, None, (self, other), {}
        )

    def __rmatmul__(self, other):
        return TENSOR_RUN_IMPLEMENTATION.torch_dispatch(
            TorchTTNNTensor, torch.ops.aten.mm.default, None, (other, self), {}
        )

    def bool(self):
        if self.ttnn_tensor is not None:
            return TorchTTNNTensor(self.ttnn_tensor, dtype=torch.bool)
        assert self.elem is not None, "Both ttnn_tensor and elem are None. This should not happen."
        return TorchTTNNTensor(self.elem.bool())

    @property
    def to_ttnn(self):
        return TENSOR_RUN_IMPLEMENTATION.to_ttnn(self)

    @property
    def to_torch(self):
        return TENSOR_RUN_IMPLEMENTATION.to_torch(self)

    def tolist(self):
        return self.to_torch.tolist()
