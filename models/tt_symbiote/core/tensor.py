"""TorchTTNNTensor: A PyTorch tensor subclass with TTNN backend support."""

import contextlib
from typing import Iterator

import torch
from torch.utils._pytree import tree_map

import ttnn


@contextlib.contextmanager
def no_dispatch() -> Iterator[None]:
    """Context manager to disable torch dispatch."""
    guard = torch._C._DisableTorchDispatch()  # type: ignore[attr-defined]
    try:
        yield
    finally:
        del guard


def get_empty_torch_tensor_from_ttnn(tt_tensor) -> torch.Tensor:
    """Convert TTNN tensor shape/dtype to empty torch tensor on meta device."""
    ttnn_shape = [int(i) for i in tt_tensor.shape]
    ttnn_dtype = tt_tensor.dtype

    def ttnn_dtype_to_torch_dtype(ttnn_dtype):
        if "float" in str(ttnn_dtype).lower():
            return torch.float32
        else:
            return torch.int64  # Default fallback, extend as needed

    torch_dtype = ttnn_dtype_to_torch_dtype(ttnn_dtype)
    empty_torch_tensor = torch.empty(ttnn_shape, dtype=torch_dtype, device="meta")
    return empty_torch_tensor


class TorchTTNNTensor(torch.Tensor):
    """PyTorch tensor wrapper that can dispatch operations to TTNN backend."""

    elem: torch.Tensor

    __slots__ = ["elem"]

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        delete_elem = False
        ttnn_tensor = None
        if isinstance(elem, ttnn.Tensor):
            ttnn_tensor = elem
            elem = get_empty_torch_tensor_from_ttnn(ttnn_tensor)
            delete_elem = True
        elif isinstance(elem, torch.Tensor) and not isinstance(elem, TorchTTNNTensor):
            if elem.device.type == "meta":
                print("Warning: wrapping meta tensor. This will fail if conversion to TTNN tensor is attempted.")
        output_shape = elem.size()
        strides = elem.stride()
        output_dtype = elem.dtype
        requires_grad = elem.requires_grad
        assert not isinstance(
            elem, TorchTTNNTensor
        ), "Wrapping a TorchTTNNTensor inside another TorchTTNNTensor. This is not allowed."
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            output_shape,
            strides=strides,
            storage_offset=0 if elem.device.type == "meta" else elem.storage_offset(),
            dtype=output_dtype,
            layout=elem.layout,
            device="cpu",
            requires_grad=requires_grad,
        )
        # ...the real tensor is held as an element on the tensor.
        r.ttnn_tensor = ttnn_tensor  # Initialize ttnn_tensor
        r.elem = elem if not delete_elem else None
        assert isinstance(r.elem, torch.Tensor) or isinstance(
            ttnn_tensor, ttnn.Tensor
        ), f"elem must be a torch.Tensor (or None when ttnn.Tensor is defined), but got {type(r.elem)}"
        return r

    def __repr__(self):
        return (
            f"TTNNTensor({self.ttnn_tensor.__repr__()})"
            if self.ttnn_tensor is not None
            else f"TorchTensor({self.elem.__repr__()})"
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        """Dispatch torch operations to TTNN when possible."""
        from models.tt_symbiote.core.dispatcher import can_dispatch_to_ttnn, dispatch_to_ttnn

        def unwrap(e):
            res = e
            if isinstance(e, TorchTTNNTensor) and e.ttnn_tensor is not None:
                if func.name() != "aten::_scaled_dot_product_flash_attention_for_cpu":
                    print(
                        f"Found Operation {func.name()} that if written in ttnn would be more efficient. "
                        "Please map this function to an appropriate ttnn function."
                    )
                res = e.to_torch
            elif isinstance(e, TorchTTNNTensor):
                res = e.to_torch
            elif isinstance(e, torch.Tensor):
                res = e
            return res

        def wrap(e):
            return TorchTTNNTensor(e) if isinstance(e, torch.Tensor) else e

        # no_dispatch is only needed if you use enable_python_mode.
        # It prevents infinite recursion.
        if can_dispatch_to_ttnn(func.name(), args, kwargs):
            return dispatch_to_ttnn(func.name(), args, kwargs)
        with no_dispatch():
            func_args = tree_map(unwrap, args)
            func_kwargs = tree_map(unwrap, kwargs)
            func_res = func(*func_args, **func_kwargs)
            rs = tree_map(wrap, func_res)
        return rs

    @property
    def shape(self):
        return self.elem.shape if self.elem is not None else tuple(int(i) for i in self.ttnn_tensor.shape)

    def __mul__(self, other):
        from models.tt_symbiote.core.dispatcher import can_dispatch_to_ttnn, dispatch_to_ttnn

        if can_dispatch_to_ttnn("aten::mul.Tensor", (self, other), {}):
            return dispatch_to_ttnn("aten::mul.Tensor", (self, other), {})
        return self.elem * other

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        from models.tt_symbiote.core.dispatcher import can_dispatch_to_ttnn, dispatch_to_ttnn

        if can_dispatch_to_ttnn("aten::sub.Tensor", (self, other), {}):
            return dispatch_to_ttnn("aten::sub.Tensor", (self, other), {})
        return self.to_torch - other

    def __rsub__(self, other):
        from models.tt_symbiote.core.dispatcher import can_dispatch_to_ttnn, dispatch_to_ttnn

        if can_dispatch_to_ttnn("aten::sub.Tensor", (other, self), {}):
            return dispatch_to_ttnn("aten::sub.Tensor", (other, self), {})
        return other - self.elem.to_torch

    def __add__(self, other):
        from models.tt_symbiote.core.dispatcher import can_dispatch_to_ttnn, dispatch_to_ttnn

        if can_dispatch_to_ttnn("aten::add.Tensor", (self, other), {}):
            return dispatch_to_ttnn("aten::add.Tensor", (self, other), {})
        return self.elem + other

    def __radd__(self, other):
        return self.__add__(other)

    def __abs__(self):
        raise RuntimeError("Absolute value is not yet implemented for TTNN tensors.")

    def __matmul__(self, other):
        raise RuntimeError("Matrix multiplication is not yet implemented for TTNN tensors.")

    def __rmatmul__(self, other):
        raise RuntimeError("Matrix multiplication is not yet implemented for TTNN tensors.")

    @property
    def to_ttnn(self):
        """Convert to TTNN tensor, creating if necessary."""
        if self.ttnn_tensor is not None:
            return self.ttnn_tensor
        assert self.elem is not None, "Both ttnn_tensor and elem are None. This should not happen."
        # convert elem to ttnn tensor here
        if self.elem.device.type == "meta":
            raise RuntimeError(
                "Cannot convert META tensor to TTNN tensor. Please ensure the tensor is on a real device before conversion."
            )
        self.ttnn_tensor = ttnn.from_torch(self.elem.cpu())
        return self.ttnn_tensor

    @property
    def to_torch(self):
        """Convert to PyTorch tensor."""
        if self.ttnn_tensor is not None and self.elem is None:
            return ttnn.to_torch(self.ttnn_tensor)
        assert self.elem is not None, "Both ttnn_tensor and elem are None. This should not happen."
        if self.elem.device.type == "meta" and self.ttnn_tensor is not None:
            return ttnn.to_torch(self.ttnn_tensor)
        return self.elem

    def tolist(self):
        return self.to_torch.tolist()
