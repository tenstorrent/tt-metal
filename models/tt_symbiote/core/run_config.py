import contextlib
import os
from typing import Iterator

import torch
from torch.utils._pytree import tree_map

import ttnn
from models.tt_symbiote.core.utils import TORCH_TO_TTNN, torch_dtype_to_ttnn_dtype, ttnn_dtype_to_torch_dtype


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


def compare_fn_outputs(torch_output, ttnn_output, func_name):
    from models.tt_symbiote.core.tensor import TorchTTNNTensor

    torch_output_tensors = []
    ttnn_output_tensors = []
    if isinstance(torch_output, TorchTTNNTensor):
        torch_output_tensors.append(torch_output.to_torch)
    elif isinstance(torch_output, (list, tuple)):
        for item in torch_output:
            if isinstance(item, TorchTTNNTensor):
                torch_output_tensors.append(item.to_torch)
    if isinstance(ttnn_output, TorchTTNNTensor):
        ttnn_output.elem = None
        ttnn_output_tensors.append(ttnn_output.to_torch)
        assert isinstance(torch_output, TorchTTNNTensor), "Mismatched output types between TTNN and Torch."
    elif isinstance(ttnn_output, (list, tuple)):
        assert isinstance(torch_output, (list, tuple)), "Mismatched output types between TTNN and Torch."
        assert len(ttnn_output) == len(torch_output), "Mismatched output lengths between TTNN and Torch."
        for index, item in enumerate(ttnn_output):
            if isinstance(item, TorchTTNNTensor):
                assert isinstance(
                    torch_output[index], TorchTTNNTensor
                ), "Mismatched output types between TTNN and Torch."
                item.elem = None
                ttnn_output_tensors.append(item.to_torch)

    passed = True
    for t_tensor, n_tensor in zip(torch_output_tensors, ttnn_output_tensors):
        # calculate PCC between t_tensor and n_tensor
        t_tensor = t_tensor.to(torch.float32)
        n_tensor = n_tensor.to(torch.float32)
        assert t_tensor.shape == n_tensor.shape, "Mismatched output shapes between TTNN and Torch."
        pcc = torch.corrcoef(torch.stack([t_tensor.flatten(), n_tensor.flatten()]))[0, 1]
        diff = torch.abs(t_tensor - n_tensor)
        if pcc < 0.999 or (torch.median(diff) > torch.mean(diff) and torch.max(diff).item() > 1):
            passed = False
            print(
                f"Warning: High discrepancy detected in operation {func_name}. "
                f"PCC: {pcc.item()}, Max Abs Diff: {torch.max(diff).item()}, Median Abs Diff: {torch.median(diff).item()}, Mean Abs Diff: {torch.mean(diff).item()}"
            )
        if torch.logical_xor((n_tensor == 0).all(), (t_tensor == 0).all()):
            passed = False
            print(f"Warning: One of the outputs is all zeros while the other is not in operation {func_name}.")

        if func_name == "aten::topk":
            break
    if not passed:
        print(f"Operation {func_name} PCC < 0.99.")


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
        result = self.elem
        if self.ttnn_tensor is not None and self.elem is None:
            result = ttnn.to_torch(self.ttnn_tensor).to(self.device, self.dtype)
        assert result is not None, "Both ttnn_tensor and elem are None. This should not happen."
        if result.device.type == "meta" and self.ttnn_tensor is not None:
            result = ttnn.to_torch(self.ttnn_tensor).to(self.device, self.dtype)
        self.elem = result if self.elem is None else self.elem
        return self.elem

    @staticmethod
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
        if self.elem.dtype not in TORCH_TO_TTNN:
            raise RuntimeError(f"Unsupported dtype {self.elem.dtype} for conversion to TTNN tensor.")
        self.ttnn_tensor = ttnn.from_torch(self.elem.cpu(), dtype=torch_dtype_to_ttnn_dtype(self.elem.dtype))
        return self.ttnn_tensor

    @staticmethod
    def module_run(self, *args, **kwds):
        print(f"{self.__class__.__name__}: {self.module_name} on device {self.device}")
        assert self.device is not None, "Device must be set for TTNN module execution."
        func_args = tree_map(wrap_to_torch_ttnn_tensor, args)
        func_kwargs = tree_map(wrap_to_torch_ttnn_tensor, kwds)
        func_args = tree_map(to_ttnn_wrap, func_args)
        func_kwargs = tree_map(to_ttnn_wrap, func_kwargs)
        func_args = tree_map(set_device_wrap(self.device), func_args)
        func_kwargs = tree_map(set_device_wrap(self.device), func_kwargs)
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
            func_args = tree_map(wrap_to_torch_ttnn_tensor, args)
            func_kwargs = tree_map(wrap_to_torch_ttnn_tensor, kwds)
            func_args = tree_map(to_ttnn_wrap, func_args)
            func_kwargs = tree_map(to_ttnn_wrap, func_kwargs)
            func_args = tree_map(set_device_wrap(self.device), func_args)
            func_kwargs = tree_map(set_device_wrap(self.device), func_kwargs)
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
    def to_torch(self):
        """Convert to PyTorch tensor."""
        result = self.elem
        if self.ttnn_tensor is not None and self.elem is None:
            result = ttnn.to_torch(self.ttnn_tensor).to(self.device, self.dtype)
        assert result is not None, "Both ttnn_tensor and elem are None. This should not happen."
        if result.device.type == "meta" and self.ttnn_tensor is not None:
            result = ttnn.to_torch(self.ttnn_tensor).to(self.device, self.dtype)
        self.elem = result if self.elem is None else self.elem
        return self.elem

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
            func_args = tree_map(to_ttnn_wrap, func_args)
            func_kwargs = tree_map(to_ttnn_wrap, func_kwargs)
            func_args = tree_map(set_device_wrap(self.device), func_args)
            func_kwargs = tree_map(set_device_wrap(self.device), func_kwargs)
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
            func_args = tree_map(wrap_to_torch_ttnn_tensor, args)
            func_kwargs = tree_map(wrap_to_torch_ttnn_tensor, kwds)
            func_args = tree_map(to_ttnn_wrap, func_args)
            func_kwargs = tree_map(to_ttnn_wrap, func_kwargs)
            func_args = tree_map(set_device_wrap(self.device), func_args)
            func_kwargs = tree_map(set_device_wrap(self.device), func_kwargs)
            self.preprocess_weights()
            self.move_weights_to_device()
            ttnn_output = tree_map(wrap_to_torch_ttnn_tensor, self.forward(*func_args, **func_kwargs))
            # Compare inputs
            compare_fn_outputs(copied_torch_tensors_args, func_args, self.__class__.__name__)
            # Compare outputs
            compare_fn_outputs(torch_output, ttnn_output, self.__class__.__name__)
            result = create_new_ttnn_tensors_using_torch_output(torch_output, ttnn_output, assign_ttnn_to_torch=True)
        return result

    @staticmethod
    def to_torch(self):
        """Convert to PyTorch tensor."""
        result = self.elem
        if self.ttnn_tensor is not None and self.elem is None:
            result = ttnn.to_torch(self.ttnn_tensor).to(self.device, self.dtype)
        assert result is not None, "Both ttnn_tensor and elem are None. This should not happen."
        if result.device.type == "meta" and self.ttnn_tensor is not None:
            result = ttnn.to_torch(self.ttnn_tensor).to(self.device, self.dtype)
        self.elem = result if self.elem is None else self.elem
        return self.elem


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
            func_args = tree_map(wrap_to_torch_ttnn_tensor, ttnn_no_error_prop_args)
            func_kwargs = tree_map(wrap_to_torch_ttnn_tensor, ttnn_no_error_prop_kwargs)
            func_args = tree_map(to_ttnn_wrap, func_args)
            func_kwargs = tree_map(to_ttnn_wrap, func_kwargs)
            func_args = tree_map(set_device_wrap(self.device), func_args)
            func_kwargs = tree_map(set_device_wrap(self.device), func_kwargs)
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


def get_tensor_run_implementation():
    if os.environ.get("TT_SYMBIOTE_RUN_MODE") == "NORMAL":
        return NormalRun
    elif os.environ.get("TT_SYMBIOTE_RUN_MODE") == "NORMAL_WITH_FALLBACK":
        return NormalRunWithFallback
    elif os.environ.get("TT_SYMBIOTE_RUN_MODE") == "SEL":
        return SELRun
    elif os.environ.get("TT_SYMBIOTE_RUN_MODE") == "DPL":
        return DPLRun
    elif os.environ.get("TT_SYMBIOTE_RUN_MODE") == "DPL_NO_ERROR_PROP":
        return DPLRunNoErrorProp
    elif os.environ.get("TT_SYMBIOTE_RUN_MODE") == "CPU":
        return CPU
    return NormalRun
