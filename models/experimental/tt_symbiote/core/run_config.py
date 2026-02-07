# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import torch
from torch.utils._pytree import tree_flatten, tree_map
from tracy import signpost

import ttnn
from models.experimental.tt_symbiote.core.utils import (
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
    return torch.empty(ttnn_shape, dtype=torch_dtype, device="meta")


def unwrap_to_torch(func):
    def _unwrap_to_torch(e):
        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

        if isinstance(e, TorchTTNNTensor):
            return e.to_torch
        if isinstance(e, torch.Tensor):
            return e
        if isinstance(e, ttnn.Tensor):
            return TorchTTNNTensor(e).to_torch
        return e

    return _unwrap_to_torch


def copy_to_torch(func):
    def _unwrap_to_torch(e):
        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

        if isinstance(e, TorchTTNNTensor):
            return TorchTTNNTensor(e.to_torch.clone())
        if isinstance(e, torch.Tensor):
            return e.clone()
        return e

    return _unwrap_to_torch


def copy_to_ttnn(func):
    def _remove_ttnn_tensor(e):
        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

        if isinstance(e, TorchTTNNTensor) and e.elem is not None and e.ttnn_tensor is not None:
            res = TorchTTNNTensor(ttnn.from_torch(e.elem.clone()))
            res.ttnn_tensor = ttnn.to_layout(res.to_ttnn, e.ttnn_tensor.layout)
            if e.ttnn_tensor.is_allocated() and e.ttnn_tensor.device() is not None:
                res.ttnn_tensor = ttnn.to_device(res.to_ttnn, e.ttnn_tensor.device())
            return res
        return e

    return _remove_ttnn_tensor


def wrap_from_torch(e):
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    return TorchTTNNTensor(e) if isinstance(e, torch.Tensor) else e


class DispatchManager:
    timings: Dict[str, Any] = {}
    _modules_in_progress: List[str] = []
    current_module_name: Optional[str] = None

    @staticmethod
    def set_current_module_name(module_name: Optional[str]) -> None:
        if module_name is None:
            if DispatchManager._modules_in_progress:
                DispatchManager._modules_in_progress.pop()
            DispatchManager.current_module_name = (
                DispatchManager._modules_in_progress[-1] if DispatchManager._modules_in_progress else None
            )
        else:
            DispatchManager._modules_in_progress.append(module_name)
            DispatchManager.current_module_name = module_name

    @staticmethod
    def dispatch_to_ttnn_wrapper(func, ttnn_args, ttnn_kwargs):
        from models.experimental.tt_symbiote.core.dispatcher import dispatch_to_ttnn

        begin = time.time()
        res = dispatch_to_ttnn(func.name(), ttnn_args, ttnn_kwargs)
        end = time.time()
        DispatchManager.record_timing(
            "TTNN",
            (
                DispatchManager.current_module_name + f".{func.name()}"
                if DispatchManager.current_module_name
                else func.name()
            ),
            func.name(),
            {},
            end - begin,
        )
        return res

    @staticmethod
    def dispatch_to_torch_wrapper(func, torch_args, torch_kwargs):
        from models.experimental.tt_symbiote.core.torch_dispatcher import can_dispatch_to_torch, dispatch_to_torch

        with no_dispatch():
            func_args, func_kwargs = tree_map(unwrap_to_torch(func), torch_args), tree_map(
                unwrap_to_torch(func), torch_kwargs
            )
            # Avoid BFloat16 != float errors when one operand is from TTNN (bfloat16) and another is module param (float32)
            target_dtype = _dtype_from_torch_args(func_args, func_kwargs)
            if target_dtype is not None:

                def _cast_to_dtype(e):
                    if isinstance(e, torch.Tensor) and e.is_floating_point() and e.dtype != target_dtype:
                        return e.to(target_dtype)
                    return e

                func_args = tree_map(_cast_to_dtype, func_args)
                func_kwargs = tree_map(_cast_to_dtype, func_kwargs)
            begin = time.time()
            func_res = (
                dispatch_to_torch(func.name(), func_args, func_kwargs)
                if can_dispatch_to_torch(func.name(), func_args, func_kwargs)
                else func(*func_args, **func_kwargs)
            )
            end = time.time()
            DispatchManager.record_timing(
                "Torch",
                (
                    DispatchManager.current_module_name + f".{func.name()}"
                    if DispatchManager.current_module_name
                    else func.name()
                ),
                func.name(),
                {},
                end - begin,
            )
            return tree_map(wrap_from_torch, func_res)

    @staticmethod
    def record_timing(backend: str, module_name: str, func_name: str, attrs: dict, duration: float) -> None:
        if "TimingEntries" not in DispatchManager.timings:
            DispatchManager.timings["TimingEntries"] = []
        DispatchManager.timings["TimingEntries"].append(
            {
                "attrs": attrs,
                "module_name": module_name,
                "func_name": func_name,
                "duration": duration,
                "backend": backend,
            }
        )

    @staticmethod
    def save_stats_to_file(file_name: str):
        import pandas as pd

        df = pd.DataFrame(DispatchManager.timings.get("TimingEntries", []))
        df.to_csv(file_name, index=True)


def wrap_to_torch_ttnn_tensor(e):
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    if isinstance(e, torch.Tensor) and not isinstance(e, TorchTTNNTensor):
        return TorchTTNNTensor(e)
    if not isinstance(e, TorchTTNNTensor) and isinstance(e, ttnn.Tensor):
        return TorchTTNNTensor(e)
    return e


def to_ttnn_wrap(e):
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    return e.to_ttnn if isinstance(e, TorchTTNNTensor) else e


def set_device_wrap(device):
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    def _set_device_wrap(e):
        if isinstance(e, ttnn.Tensor) and device and e.device() != device:
            return ttnn.to_device(e, device)
        if isinstance(e, TorchTTNNTensor) and e.ttnn_tensor and e.ttnn_tensor.device() != device:
            e.ttnn_tensor = ttnn.to_device(e.ttnn_tensor, device)
        return e

    return _set_device_wrap


def create_new_ttnn_tensors_using_torch_output(torch_output, ttnn_output, assign_ttnn_to_torch=False):
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    if isinstance(torch_output, (list, tuple)) and isinstance(ttnn_output, TorchTTNNTensor):
        if len(torch_output) > 0 and isinstance(torch_output[0], TorchTTNNTensor):
            torch_output[0].ttnn_tensor = ttnn_output.to_ttnn
            if not assign_ttnn_to_torch:
                torch_output[0].elem = None
            return torch_output

    if isinstance(torch_output, TorchTTNNTensor) and isinstance(ttnn_output, TorchTTNNTensor):
        torch_output.ttnn_tensor = ttnn_output.to_ttnn
        if not assign_ttnn_to_torch:
            torch_output.elem = None

    elif isinstance(torch_output, (list, tuple)) and isinstance(ttnn_output, (list, tuple)):
        for t_item, n_item in zip(torch_output, ttnn_output):
            if isinstance(t_item, TorchTTNNTensor) and isinstance(n_item, TorchTTNNTensor):
                t_item.ttnn_tensor = n_item.to_ttnn
                if not assign_ttnn_to_torch:
                    t_item.elem = None

    return torch_output


def compose_transforms(*transforms):
    def _composed(e):
        res = e
        for t in transforms:
            res = t(res)
        return res

    return _composed


def _dtype_from_torch_args(args, kwds):
    """Get the first floating-point tensor dtype from flattened args/kwds (TorchTTNNTensor or torch.Tensor)."""
    from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

    flat, _ = tree_flatten((args, kwds))
    for a in flat:
        if isinstance(a, (TorchTTNNTensor, torch.Tensor)) and a.is_floating_point():
            return a.dtype
    return None


def _cast_module_to_dtype(module, dtype):
    """Cast module's floating parameters and buffers to dtype in-place (e.g. for torch fallback dtype match)."""
    if dtype is None:
        return
    for p in module.parameters():
        if p.is_floating_point():
            p.data = p.data.to(dtype)
    for b in module.buffers():
        if b.is_floating_point():
            b.data = b.data.to(dtype)


class NormalRun:
    verbose = False
    signpost_mode = None

    def __new__(cls, *args, **kwargs):
        raise TypeError("Cannot instantiate")

    @staticmethod
    def new_instance(cls, elem, *args, **kwargs):
        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

        ttnn_tensor = elem if isinstance(elem, ttnn.Tensor) else None
        if ttnn_tensor:
            elem = get_empty_torch_tensor_from_ttnn(ttnn_tensor, dtype=kwargs.get("dtype"))
        elif isinstance(elem, torch.Tensor) and not isinstance(elem, TorchTTNNTensor) and kwargs.get("dtype"):
            elem = elem.to(dtype=kwargs.get("dtype"))

        # ERROR GUARD: If a tuple reached here, SigLIP math will crash. Force unwrap.
        if isinstance(elem, (list, tuple)):
            elem = elem[0]

        r = torch.Tensor._make_wrapper_subclass(
            cls,
            elem.size(),
            strides=elem.stride(),
            storage_offset=0 if elem.device.type == "meta" else elem.storage_offset(),
            dtype=elem.dtype,
            layout=elem.layout,
            device="cpu",
            requires_grad=elem.requires_grad,
        )
        r.ttnn_tensor, r.elem = ttnn_tensor, (elem if not ttnn_tensor else None)
        return r

    @staticmethod
    def torch_dispatch(cls, func, types, args=(), kwargs=None):
        from models.experimental.tt_symbiote.core.dispatcher import can_dispatch_to_ttnn

        return (
            DispatchManager.dispatch_to_ttnn_wrapper(func, args, kwargs)
            if can_dispatch_to_ttnn(func.name(), args, kwargs)
            else DispatchManager.dispatch_to_torch_wrapper(func, args, kwargs)
        )

    @staticmethod
    def to_torch(self):
        if self.elem is not None and self.elem.device.type != "meta" and self.ttnn_tensor is None:
            return self.elem
        res = ttnn.to_torch(self.ttnn_tensor).to(self.device, self.dtype)
        self.elem = res if self.elem is None else self.elem
        return self.elem

    @staticmethod
    def to_ttnn(self):
        if self.ttnn_tensor is not None:
            return self.ttnn_tensor
        self.ttnn_tensor = ttnn.from_torch(
            self.elem.cpu(),
            dtype=torch_dtype_to_ttnn_dtype(self.elem.dtype),
            layout=ttnn.TILE_LAYOUT if self.dtype == torch.bool else None,
        )
        return self.ttnn_tensor

    @staticmethod
    def module_run(self, *args, **kwds):
        transform = compose_transforms(wrap_to_torch_ttnn_tensor, to_ttnn_wrap, set_device_wrap(self.device))
        func_args, func_kwargs = tree_map(transform, args), tree_map(
            transform, {k: v for k, v in kwds.items() if "past_key_value" not in k}
        )
        func_kwargs.update({k: v for k, v in kwds.items() if "past_key_value" in k})
        self.preprocess_weights()
        self.move_weights_to_device()
        DispatchManager.set_current_module_name(self.module_name)
        if NormalRun.signpost_mode:
            signpost(f"{self.module_name}", f"{self.__class__.__name__}")
        result = tree_map(wrap_to_torch_ttnn_tensor, self.forward(*func_args, **func_kwargs))
        DispatchManager.set_current_module_name(None)
        return result


class DPLRun(NormalRun):
    @staticmethod
    def module_run(self, *args, **kwds):
        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

        torch_args = tree_map(copy_to_torch(self.__class__.__name__), args)
        target_dtype = _dtype_from_torch_args(torch_args, kwds)
        if target_dtype is not None:
            _cast_module_to_dtype(self.torch_layer, target_dtype)
        torch_output = tree_map(wrap_to_torch_ttnn_tensor, self.torch_layer(*torch_args, **kwds))
        self.preprocess_weights()
        self.move_weights_to_device()
        ttnn_output = tree_map(
            wrap_to_torch_ttnn_tensor,
            self.forward(
                *tree_map(
                    compose_transforms(wrap_to_torch_ttnn_tensor, to_ttnn_wrap, set_device_wrap(self.device)), args
                ),
                **kwds,
            ),
        )

        # Safe Audit: Avoid tree_map crash on Tuple vs Tensor
        t_aud = torch_output[0] if isinstance(torch_output, (list, tuple)) else torch_output
        n_aud = ttnn_output[0] if isinstance(ttnn_output, (list, tuple)) else ttnn_output
        if isinstance(t_aud, TorchTTNNTensor) and isinstance(n_aud, TorchTTNNTensor):
            compare_fn_outputs(t_aud, n_aud, self.module_name)

        return create_new_ttnn_tensors_using_torch_output(torch_output, ttnn_output, assign_ttnn_to_torch=True)


class DPLRunNoErrorProp(NormalRun):
    @staticmethod
    def module_run(self, *args, **kwds):
        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

        target_dtype = _dtype_from_torch_args(args, kwds)
        if target_dtype is not None:
            _cast_module_to_dtype(self.torch_layer, target_dtype)
        torch_output = tree_map(wrap_to_torch_ttnn_tensor, self.torch_layer(*args, **kwds))
        self.preprocess_weights()
        self.move_weights_to_device()
        ttnn_args = tree_map(copy_to_ttnn(self.__class__.__name__), args)
        ttnn_output = tree_map(
            wrap_to_torch_ttnn_tensor,
            self.forward(
                *tree_map(
                    compose_transforms(wrap_to_torch_ttnn_tensor, to_ttnn_wrap, set_device_wrap(self.device)), ttnn_args
                ),
                **kwds,
            ),
        )

        t_aud = torch_output[0] if isinstance(torch_output, (list, tuple)) else torch_output
        n_aud = ttnn_output[0] if isinstance(ttnn_output, (list, tuple)) else ttnn_output
        if isinstance(t_aud, TorchTTNNTensor) and isinstance(n_aud, TorchTTNNTensor):
            compare_fn_outputs(t_aud, n_aud, self.module_name)

        return create_new_ttnn_tensors_using_torch_output(torch_output, ttnn_output, assign_ttnn_to_torch=True)


class SELRun(NormalRun):
    @staticmethod
    def module_run(self, *args, **kwds):
        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

        torch_args = tree_map(copy_to_torch(self.__class__.__name__), args)
        target_dtype = _dtype_from_torch_args(torch_args, kwds)
        if target_dtype is not None:
            _cast_module_to_dtype(self.torch_layer, target_dtype)
        torch_output = tree_map(wrap_to_torch_ttnn_tensor, self.torch_layer(*torch_args, **kwds))
        self.preprocess_weights()
        self.move_weights_to_device()
        ttnn_output = tree_map(
            wrap_to_torch_ttnn_tensor,
            self.forward(
                *tree_map(
                    compose_transforms(wrap_to_torch_ttnn_tensor, to_ttnn_wrap, set_device_wrap(self.device)), args
                ),
                **kwds,
            ),
        )

        t_aud = torch_output[0] if isinstance(torch_output, (list, tuple)) else torch_output
        n_aud = ttnn_output[0] if isinstance(ttnn_output, (list, tuple)) else ttnn_output
        if isinstance(t_aud, TorchTTNNTensor) and isinstance(n_aud, TorchTTNNTensor):
            compare_fn_outputs(t_aud, n_aud, self.module_name)

        return create_new_ttnn_tensors_using_torch_output(torch_output, ttnn_output)


class LightweightRun(NormalRun):
    pass


class NormalRunWithFallback(NormalRun):
    pass


class CPU(NormalRun):
    pass


_RUN_MODE_REGISTRY = {
    "LIGHTWEIGHT": LightweightRun,
    "NORMAL": NormalRun,
    "NORMAL_WITH_FALLBACK": NormalRunWithFallback,
    "SEL": SELRun,
    "DPL": DPLRun,
    "DPL_NO_ERROR_PROP": DPLRunNoErrorProp,
    "CPU": CPU,
}

_current_run_mode = None


def set_run_mode(mode: str) -> None:
    global _current_run_mode
    _current_run_mode = mode


def get_tensor_run_implementation():
    global _current_run_mode
    env_mode = os.environ.get("TT_SYMBIOTE_RUN_MODE", _current_run_mode or "NORMAL")
    signpost_mode = os.environ.get("TT_SYMBIOTE_SIGNPOST_MODE", None)
    res = _RUN_MODE_REGISTRY[env_mode]
    res.signpost_mode = signpost_mode
    return res
