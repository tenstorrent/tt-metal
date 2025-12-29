import contextlib
import os
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch.utils._pytree import tree_map

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.tt_symbiote.core.utils import (
    TORCH_TO_TTNN,
    compare_fn_outputs,
    torch_dtype_to_ttnn_dtype,
    ttnn_dtype_to_torch_dtype,
)


@dataclass
class DistributedTensorConfig:
    """Configuration for distributed tensor operations."""

    mesh_mapper: Any


@contextlib.contextmanager
def no_dispatch() -> Iterator[None]:
    """Context manager to disable torch dispatch."""
    guard = torch._C._DisableTorchDispatch()  # type: ignore[attr-defined]
    try:
        yield
    finally:
        del guard


def get_empty_torch_tensor_from_ttnn(tt_tensor, dtype=None) -> torch.Tensor:
    """Convert TTNN tensor shape/dtype to empty torch tensor on meta device."""
    ttnn_shape = [int(i) for i in tt_tensor.shape]
    ttnn_dtype = tt_tensor.dtype

    torch_dtype = ttnn_dtype_to_torch_dtype(ttnn_dtype) if dtype is None else dtype
    empty_torch_tensor = torch.empty(ttnn_shape, dtype=torch_dtype, device="meta")
    return empty_torch_tensor


def unwrap_to_torch(func):
    def _unwrap_to_torch(e):
        from models.tt_symbiote.core.tensor import TorchTTNNTensor

        res = e
        if isinstance(e, TorchTTNNTensor) and e.ttnn_tensor is not None:
            res = e.to_torch
        elif isinstance(e, TorchTTNNTensor):
            res = e.to_torch
        elif isinstance(e, torch.Tensor):
            res = e
        elif isinstance(e, ttnn.Tensor):
            res = TorchTTNNTensor(e).to_torch
        return res

    return _unwrap_to_torch


def copy_to_torch(func):
    def _unwrap_to_torch(e):
        from models.tt_symbiote.core.tensor import TorchTTNNTensor

        res = e
        if isinstance(e, TorchTTNNTensor):
            res = TorchTTNNTensor(e.to_torch.clone())
        elif isinstance(e, torch.Tensor):
            res = e.clone()
        return res

    return _unwrap_to_torch


def copy_to_ttnn(func):
    def _remove_ttnn_tensor(e):
        from models.tt_symbiote.core.tensor import TorchTTNNTensor

        res = e
        if isinstance(e, TorchTTNNTensor) and e.elem is not None and e.ttnn_tensor is not None:
            res = TorchTTNNTensor(ttnn.from_torch(e.elem.clone()))
            res.ttnn_tensor = ttnn.to_device(res.to_ttnn, e.ttnn_tensor.device())
            res.ttnn_tensor = ttnn.to_layout(
                res.ttnn_tensor, e.ttnn_tensor.layout, memory_config=e.ttnn_tensor.memory_config()
            )
        return res

    return _remove_ttnn_tensor


def wrap_from_torch(e):
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

    return TorchTTNNTensor(e) if isinstance(e, torch.Tensor) else e


def dispatch_to_ttnn_wrapper(func, ttnn_args, ttnn_kwargs):
    from models.tt_symbiote.core.dispatcher import dispatch_to_ttnn

    if get_tensor_run_implementation().verbose:
        print(f"Dispatching {func.name()} to TTNN backend.")
    res = dispatch_to_ttnn(func.name(), ttnn_args, ttnn_kwargs)
    if get_tensor_run_implementation().verbose:
        print(f"Finished {func.name()} on TTNN backend.")
    return res


def dispatch_to_torch_wrapper(func, torch_args, torch_kwargs):
    from models.tt_symbiote.core.torch_dispatcher import can_dispatch_to_torch, dispatch_to_torch

    # no_dispatch is only needed if you use enable_python_mode.
    # It prevents infinite recursion.
    with no_dispatch():
        func_args = tree_map(unwrap_to_torch(func), torch_args)
        func_kwargs = tree_map(unwrap_to_torch(func), torch_kwargs)
        if can_dispatch_to_torch(func.name(), func_args, func_kwargs):
            func_res = dispatch_to_torch(func.name(), func_args, func_kwargs)
        else:
            func_res = func(*func_args, **func_kwargs)
        rs = tree_map(wrap_from_torch, func_res)
    return rs


def wrap_to_torch_ttnn_tensor(e):
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

    result = TorchTTNNTensor(e) if isinstance(e, torch.Tensor) and not isinstance(e, TorchTTNNTensor) else e
    if not isinstance(e, TorchTTNNTensor) and isinstance(e, ttnn.Tensor):
        result = TorchTTNNTensor(e)
    return result


def to_ttnn_wrap(e):
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

    if isinstance(e, TorchTTNNTensor):
        e = e.to_ttnn
    return e


def set_device_wrap(device):
    def _set_device_wrap(e):
        if isinstance(e, ttnn.Tensor) and device is not None and e.device() != device:
            e = ttnn.to_device(e, device)
        return e

    return _set_device_wrap


def create_new_ttnn_tensors_using_torch_output(torch_output, ttnn_output, assign_ttnn_to_torch=False):
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

    if isinstance(torch_output, TorchTTNNTensor) and isinstance(ttnn_output, TorchTTNNTensor):
        assert len(torch_output.shape) == len(ttnn_output.shape) and all(
            [s1 == s2 for s1, s2 in zip(torch_output.shape, ttnn_output.shape)]
        ), "Mismatched output shapes between TTNN and Torch."
        assert torch_output.elem is not None, "torch_output.elem is None, cannot assign to ttnn_output."
        torch_output.ttnn_tensor = ttnn_output.to_ttnn
        if not assign_ttnn_to_torch:
            torch_output.elem = None
    elif isinstance(torch_output, (list, tuple)) and isinstance(ttnn_output, (list, tuple)):
        assert len(torch_output) == len(ttnn_output), "Mismatched output lengths between TTNN and Torch."
        for t_item, n_item in zip(torch_output, ttnn_output):
            if isinstance(t_item, TorchTTNNTensor) and isinstance(n_item, TorchTTNNTensor):
                assert len(t_item.shape) == len(n_item.shape) and all(
                    [s1 == s2 for s1, s2 in zip(t_item.shape, n_item.shape)]
                ), "Mismatched output shapes between TTNN and Torch."
                assert t_item.elem is not None, "t_item.elem is None, cannot assign to n_item."
                t_item.ttnn_tensor = n_item.to_ttnn
                if not assign_ttnn_to_torch:
                    t_item.elem = None
    else:
        print("Warning: Mismatched output types between TTNN and Torch in create_new_ttnn_tensors_using_torch_output.")
    return torch_output


def compose_transforms(*transforms):
    """Compose multiple transformation functions into a single pass."""

    def _composed(e):
        result = e
        for transform in transforms:
            result = transform(result)
        return result

    return _composed


class NormalRun:
    verbose = False

    def __new__(cls, *args, **kwargs):
        raise TypeError("This class cannot be instantiated")

    @staticmethod
    def new_instance(cls, elem, *args, **kwargs):
        from models.tt_symbiote.core.tensor import TorchTTNNTensor

        delete_elem = False
        ttnn_tensor = None
        if isinstance(elem, ttnn.Tensor):
            ttnn_tensor = elem
            elem = get_empty_torch_tensor_from_ttnn(ttnn_tensor, dtype=kwargs.get("dtype"))
            delete_elem = True
        elif isinstance(elem, torch.Tensor) and not isinstance(elem, TorchTTNNTensor):
            if kwargs.get("dtype") is not None:
                elem = elem.to(dtype=kwargs.get("dtype"))
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

    @staticmethod
    def repr(self):
        return (
            f"TTNNTensor({self.ttnn_tensor.__repr__()})"
            if self.ttnn_tensor is not None
            else f"TorchTensor({self.elem.__repr__()})"
        )

    @staticmethod
    def torch_dispatch(cls, func, types, args=(), kwargs=None):
        """Dispatch torch operations to TTNN when possible."""
        from models.tt_symbiote.core.dispatcher import can_dispatch_to_ttnn

        if can_dispatch_to_ttnn(func.name(), args, kwargs):
            rs = dispatch_to_ttnn_wrapper(func, args, kwargs)
        else:
            rs = dispatch_to_torch_wrapper(func, args, kwargs)
        return rs

    @staticmethod
    def to_torch(self):
        """Convert to PyTorch tensor."""

        def _to_torch(self):
            is_mesh_device = self.ttnn_tensor.device().__class__.__name__ == "MeshDevice"
            is_mesh_device = is_mesh_device and self.ttnn_tensor.device().get_num_devices() != 1
            if is_mesh_device:
                result = to_torch_auto_compose(self.ttnn_tensor, device=self.ttnn_tensor.device())
            else:
                result = ttnn.to_torch(self.ttnn_tensor).to(self.device, self.dtype)
            return result

        result = self.elem
        if self.ttnn_tensor is not None and self.elem is None:
            result = _to_torch(self)
        assert result is not None, "Both ttnn_tensor and elem are None. This should not happen."
        if result.device.type == "meta" and self.ttnn_tensor is not None:
            result = _to_torch(self)
        self.elem = result if self.elem is None else self.elem
        return self.elem

    @staticmethod
    def to_ttnn(self):
        """Convert to TTNN tensor, creating if necessary."""
        if self.ttnn_tensor is not None:
            return self.ttnn_tensor
        assert self.elem is not None, "Both ttnn_tensor and elem are None. This should not happen."
        # convert elem to ttnn tensor here
        is_mesh_device = self.device.__class__.__name__ == "MeshDevice"
        if self.ttnn_distributed_config is None and is_mesh_device:
            self.__dict__["distributed_config"] = DistributedTensorConfig(
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device)
            )
        if self.elem.device.type == "meta":
            raise RuntimeError(
                "Cannot convert META tensor to TTNN tensor. Please ensure the tensor is on a real device before conversion."
            )
        if self.elem.dtype not in TORCH_TO_TTNN:
            raise RuntimeError(f"Unsupported dtype {self.elem.dtype} for conversion to TTNN tensor.")
        self.ttnn_tensor = ttnn.from_torch(
            self.elem.cpu(),
            dtype=torch_dtype_to_ttnn_dtype(self.elem.dtype),
            mesh_mapper=self.ttnn_distributed_config.mesh_mapper if self.ttnn_distributed_config else None,
            layout=ttnn.TILE_LAYOUT if self.dtype == torch.bool else None,
        )
        return self.ttnn_tensor

    @staticmethod
    def module_run(self, *args, **kwds):
        print(f"{self.__class__.__name__}: {self.module_name} on device {self.device}")
        assert self.device is not None, "Device must be set for TTNN module execution."
        transform = compose_transforms(wrap_to_torch_ttnn_tensor, to_ttnn_wrap, set_device_wrap(self.device))
        func_args = tree_map(transform, args)
        func_kwargs = tree_map(transform, kwds)
        self.preprocess_weights()
        self.move_weights_to_device()
        result = self.forward(*func_args, **func_kwargs)
        result = tree_map(wrap_to_torch_ttnn_tensor, result)
        return result


class NormalRunWithFallback(NormalRun):
    @staticmethod
    def torch_dispatch(cls, func, types, args=(), kwargs=None):
        """Dispatch torch operations to TTNN when possible."""
        from models.tt_symbiote.core.dispatcher import can_dispatch_to_ttnn

        try:
            if can_dispatch_to_ttnn(func.name(), args, kwargs):
                rs = dispatch_to_ttnn_wrapper(func, args, kwargs)
            else:
                rs = dispatch_to_torch_wrapper(func, args, kwargs)
        except Exception as e:
            print(f"Error {e} in dispatching {func.name()}, falling back to torch")
            rs = dispatch_to_torch_wrapper(func, args, kwargs)
        return rs

    @staticmethod
    def module_run(self, *args, **kwds):
        print(f"{self.__class__.__name__}: {self.module_name} on device {self.device}")
        result = None
        if self.device is not None:
            transform = compose_transforms(wrap_to_torch_ttnn_tensor, to_ttnn_wrap, set_device_wrap(self.device))
            func_args = tree_map(transform, args)
            func_kwargs = tree_map(transform, kwds)
            self.preprocess_weights()
            self.move_weights_to_device()
            try:
                result = self.forward(*func_args, **func_kwargs)
                result = tree_map(wrap_to_torch_ttnn_tensor, result)
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


class SELRun(NormalRun):
    @staticmethod
    def torch_dispatch(cls, func, types, args=(), kwargs=None):
        """Dispatch torch operations to TTNN when possible."""

        from models.tt_symbiote.core.dispatcher import can_dispatch_to_ttnn

        copied_torch_tensors_args = tree_map(copy_to_torch(func), args)
        copied_torch_tensors_kwargs = tree_map(copy_to_torch(func), kwargs)
        result = dispatch_to_torch_wrapper(func, copied_torch_tensors_args, copied_torch_tensors_kwargs)
        if can_dispatch_to_ttnn(func.name(), args, kwargs):
            ttnn_output = dispatch_to_ttnn_wrapper(func, args, kwargs)
            # Compare inputs
            compare_fn_outputs(copied_torch_tensors_args, args, func.name())
            # Compare outputs
            compare_fn_outputs(result, ttnn_output, func.name())
            result = create_new_ttnn_tensors_using_torch_output(result, ttnn_output)
        return result

    @staticmethod
    def module_run(self, *args, **kwds):
        pass

        print(f"{self.__class__.__name__}: {self.module_name} on device {self.device}")
        copied_torch_tensors_args = tree_map(copy_to_torch(self.__class__.__name__), args)
        copied_torch_tensors_kwargs = tree_map(copy_to_torch(self.__class__.__name__), kwds)
        func_args = tree_map(wrap_to_torch_ttnn_tensor, copied_torch_tensors_args)
        func_kwargs = tree_map(wrap_to_torch_ttnn_tensor, copied_torch_tensors_kwargs)
        torch_output = tree_map(wrap_to_torch_ttnn_tensor, self.torch_layer(*func_args, **func_kwargs))
        result = torch_output
        if self.device is not None:
            transform = compose_transforms(to_ttnn_wrap, set_device_wrap(self.device))
            func_args = tree_map(transform, func_args)
            func_kwargs = tree_map(transform, func_kwargs)
            self.preprocess_weights()
            self.move_weights_to_device()
            ttnn_output = tree_map(wrap_to_torch_ttnn_tensor, self.forward(*func_args, **func_kwargs))
            # Compare inputs
            compare_fn_outputs(copied_torch_tensors_args, func_args, self.__class__.__name__)
            # Compare outputs
            compare_fn_outputs(torch_output, ttnn_output, self.__class__.__name__)
            result = create_new_ttnn_tensors_using_torch_output(torch_output, ttnn_output)
        return result


class DPLRun(NormalRun):
    @staticmethod
    def torch_dispatch(cls, func, types, args=(), kwargs=None):
        """Dispatch torch operations to TTNN when possible."""
        from models.tt_symbiote.core.dispatcher import can_dispatch_to_ttnn

        copied_torch_tensors_args = tree_map(copy_to_torch(func), args)
        copied_torch_tensors_kwargs = tree_map(copy_to_torch(func), kwargs)
        result = dispatch_to_torch_wrapper(func, copied_torch_tensors_args, copied_torch_tensors_kwargs)
        if can_dispatch_to_ttnn(func.name(), args, kwargs):
            ttnn_output = dispatch_to_ttnn_wrapper(func, args, kwargs)
            # Compare inputs
            compare_fn_outputs(copied_torch_tensors_args, args, func.name())
            # Compare outputs
            compare_fn_outputs(result, ttnn_output, func.name())
            result = create_new_ttnn_tensors_using_torch_output(result, ttnn_output, assign_ttnn_to_torch=True)
        return result

    @staticmethod
    def module_run(self, *args, **kwds):
        assert (
            self.torch_layer is not None
        ), f"torch_layer must be set for DPLRun, {self} does not have torch_layer set."

        print(f"{self.__class__.__name__}: {self.module_name} on device {self.device}")
        copied_torch_tensors_args = tree_map(copy_to_torch(self.__class__.__name__), args)
        copied_torch_tensors_kwargs = tree_map(copy_to_torch(self.__class__.__name__), kwds)
        func_args = tree_map(wrap_to_torch_ttnn_tensor, copied_torch_tensors_args)
        func_kwargs = tree_map(wrap_to_torch_ttnn_tensor, copied_torch_tensors_kwargs)
        torch_output = tree_map(wrap_to_torch_ttnn_tensor, self.torch_layer(*func_args, **func_kwargs))
        result = torch_output
        if self.device is not None:
            transform = compose_transforms(wrap_to_torch_ttnn_tensor, to_ttnn_wrap, set_device_wrap(self.device))
            func_args = tree_map(transform, func_args)
            func_kwargs = tree_map(transform, func_kwargs)
            self.preprocess_weights()
            self.move_weights_to_device()
            ttnn_output = tree_map(wrap_to_torch_ttnn_tensor, self.forward(*func_args, **func_kwargs))
            # Compare inputs
            compare_fn_outputs(copied_torch_tensors_args, func_args, self.__class__.__name__)
            # Compare outputs
            compare_fn_outputs(torch_output, ttnn_output, self.__class__.__name__)
            result = create_new_ttnn_tensors_using_torch_output(torch_output, ttnn_output, assign_ttnn_to_torch=True)
        return result


class DPLRunNoErrorProp(NormalRun):
    @staticmethod
    def torch_dispatch(cls, func, types, args=(), kwargs=None):
        """Dispatch torch operations to TTNN when possible."""
        from models.tt_symbiote.core.dispatcher import can_dispatch_to_ttnn

        copied_torch_tensors_args = tree_map(copy_to_torch(func), args)
        copied_torch_tensors_kwargs = tree_map(copy_to_torch(func), kwargs)
        result = dispatch_to_torch_wrapper(func, copied_torch_tensors_args, copied_torch_tensors_kwargs)
        ttnn_no_error_prop_args = tree_map(copy_to_ttnn(func), args)
        ttnn_no_error_prop_kwargs = tree_map(copy_to_ttnn(func), kwargs)
        if can_dispatch_to_ttnn(func.name(), ttnn_no_error_prop_args, ttnn_no_error_prop_kwargs):
            ttnn_output = dispatch_to_ttnn_wrapper(func, ttnn_no_error_prop_args, ttnn_no_error_prop_kwargs)
            # Compare inputs
            compare_fn_outputs(copied_torch_tensors_args, ttnn_no_error_prop_args, func.name())
            # Compare outputs
            compare_fn_outputs(result, ttnn_output, func.name())
            result = create_new_ttnn_tensors_using_torch_output(result, ttnn_output, assign_ttnn_to_torch=True)
            print(f"DPLNoErrorPropRun: Done Executing {func.name()}")
        return result

    @staticmethod
    def module_run(self, *args, **kwds):
        assert (
            self.torch_layer is not None
        ), f"torch_layer must be set for DPLRun, {self} does not have torch_layer set."

        copied_torch_tensors_args = tree_map(copy_to_torch(self.__class__.__name__), args)
        copied_torch_tensors_kwargs = tree_map(copy_to_torch(self.__class__.__name__), kwds)
        func_args = tree_map(wrap_to_torch_ttnn_tensor, copied_torch_tensors_args)
        func_kwargs = tree_map(wrap_to_torch_ttnn_tensor, copied_torch_tensors_kwargs)
        torch_output = tree_map(wrap_to_torch_ttnn_tensor, self.torch_layer(*func_args, **func_kwargs))
        result = torch_output
        if self.device is not None:
            ttnn_no_error_prop_args = tree_map(copy_to_ttnn(self.__class__.__name__), args)
            ttnn_no_error_prop_kwargs = tree_map(copy_to_ttnn(self.__class__.__name__), kwds)
            transform = compose_transforms(wrap_to_torch_ttnn_tensor, to_ttnn_wrap, set_device_wrap(self.device))

            func_args = tree_map(transform, ttnn_no_error_prop_args)
            func_kwargs = tree_map(transform, ttnn_no_error_prop_kwargs)
            self.preprocess_weights()
            self.move_weights_to_device()
            ttnn_output = tree_map(wrap_to_torch_ttnn_tensor, self.forward(*func_args, **func_kwargs))
            # Compare inputs
            compare_fn_outputs(copied_torch_tensors_args, func_args, self.__class__.__name__)
            # Compare outputs
            compare_fn_outputs(torch_output, ttnn_output, self.__class__.__name__)
            result = create_new_ttnn_tensors_using_torch_output(torch_output, ttnn_output, assign_ttnn_to_torch=True)
            print(
                f"DPLNoErrorPropRun: Done Executing {self.__class__.__name__} from {self.module_name} on device {self.device}"
            )
        return result


class CPU(NormalRun):
    @staticmethod
    def torch_dispatch(cls, func, types, args=(), kwargs=None):
        """Dispatch torch operations to CPU."""
        print(f"Executing {func.name()} on CPU")
        rs = dispatch_to_torch_wrapper(func, args, kwargs)
        return rs

    @staticmethod
    def module_run(self, *args, **kwds):
        print(f"{self.__class__.__name__}: {self.module_name} on CPU")
        func_args = tree_map(wrap_to_torch_ttnn_tensor, args)
        func_kwargs = tree_map(wrap_to_torch_ttnn_tensor, kwds)
        result = tree_map(wrap_to_torch_ttnn_tensor, self.torch_layer(*func_args, **func_kwargs))
        return result


# Add at module level
_RUN_MODE_REGISTRY = {
    "NORMAL": NormalRun,
    "NORMAL_WITH_FALLBACK": NormalRunWithFallback,
    "SEL": SELRun,
    "DPL": DPLRun,
    "DPL_NO_ERROR_PROP": DPLRunNoErrorProp,
    "CPU": CPU,
}

_current_run_mode = None  # Default


def set_run_mode(mode: str) -> None:
    """Set the global run mode. Must be called before any tensor operations."""
    global _current_run_mode
    assert (
        _current_run_mode is None or _current_run_mode == mode
    ), "Run mode has already been set and cannot be changed."
    if mode not in _RUN_MODE_REGISTRY:
        raise ValueError(f"Invalid run mode '{mode}'. Valid modes: {list(_RUN_MODE_REGISTRY.keys())}")
    _current_run_mode = mode


def add_run_mode(mode: str, implementation: Any) -> None:
    """Add a new run mode to the registry."""
    global _RUN_MODE_REGISTRY
    if mode in _RUN_MODE_REGISTRY:
        raise ValueError(f"Run mode '{mode}' already exists.")
    _RUN_MODE_REGISTRY[mode] = implementation


def get_tensor_run_implementation():
    # Environment variable takes precedence for backward compatibility
    global _current_run_mode
    global _RUN_MODE_REGISTRY
    env_mode = os.environ.get("TT_SYMBIOTE_RUN_MODE", _current_run_mode)
    if env_mode is None and _current_run_mode is None:
        _current_run_mode = "NORMAL"
    if env_mode != _current_run_mode and _current_run_mode is not None and env_mode is not None:
        print(
            f"Warning: Run mode from environment variable '{env_mode}' overrides the previously set run mode '{_current_run_mode}'."
        )
    if env_mode is not None:
        if env_mode not in _RUN_MODE_REGISTRY:
            raise ValueError(
                f"Invalid run mode '{env_mode}' from environment variable. Valid modes: {list(_RUN_MODE_REGISTRY.keys())}"
            )
        return _RUN_MODE_REGISTRY[env_mode]
    return _RUN_MODE_REGISTRY[_current_run_mode]
