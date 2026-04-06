# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""TorchTTNNTensor: A PyTorch tensor subclass with TTNN backend support."""

from typing import Optional

import torch

from models.experimental.tt_symbiote.core.run_config import DistributedTensorConfig, get_tensor_run_implementation


class TorchTTNNTensor(torch.Tensor):
    """PyTorch tensor wrapper that can dispatch operations to TTNN backend."""

    elem: torch.Tensor

    __slots__ = ["elem"]

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        return get_tensor_run_implementation().new_instance(cls, elem, *args, **kwargs)

    def __repr__(self):
        return get_tensor_run_implementation().repr(self)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        return get_tensor_run_implementation().torch_dispatch(cls, func, types, args, kwargs)

    @property
    def shape(self):
        return self.elem.shape if self.elem is not None else tuple(int(i) for i in self.ttnn_tensor.shape)

    def __mul__(self, other):
        return get_tensor_run_implementation().torch_dispatch(
            TorchTTNNTensor, torch.ops.aten.mul.Tensor, None, (self, other), {}
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return get_tensor_run_implementation().torch_dispatch(
            TorchTTNNTensor, torch.ops.aten.sub.Tensor, None, (self, other), {}
        )

    def __rsub__(self, other):
        return get_tensor_run_implementation().torch_dispatch(
            TorchTTNNTensor, torch.ops.aten.sub.Tensor, None, (other, self), {}
        )

    def __add__(self, other):
        return get_tensor_run_implementation().torch_dispatch(
            TorchTTNNTensor, torch.ops.aten.add.Tensor, None, (self, other), {}
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __abs__(self):
        raise RuntimeError("Absolute value is not yet implemented for TTNN tensors.")

    def __matmul__(self, other):
        return get_tensor_run_implementation().torch_dispatch(
            TorchTTNNTensor, torch.ops.aten.mm.default, None, (self, other), {}
        )

    def __rmatmul__(self, other):
        return get_tensor_run_implementation().torch_dispatch(
            TorchTTNNTensor, torch.ops.aten.mm.default, None, (other, self), {}
        )

    def bool(self):
        if self.ttnn_tensor is not None:
            return TorchTTNNTensor(self.ttnn_tensor, dtype=torch.bool)
        assert self.elem is not None, "Both ttnn_tensor and elem are None. This should not happen."
        return TorchTTNNTensor(self.elem.bool())

    @property
    def to_ttnn(self):
        return get_tensor_run_implementation().to_ttnn(self)

    @property
    def to_torch(self):
        return get_tensor_run_implementation().to_torch(self)

    def tolist(self):
        return self.to_torch.tolist()

    def numpy(self):
        return self.to_torch.numpy()

    def clone(self, **kwargs):
        return TorchTTNNTensor(
            self.ttnn_tensor.clone() if self.ttnn_tensor is not None else self.elem.clone(**kwargs), dtype=self.dtype
        )

    @property
    def ttnn_distributed_config(self) -> Optional[DistributedTensorConfig]:
        if "distributed_config" in self.__dict__:
            return self.__dict__["distributed_config"]
        return None
