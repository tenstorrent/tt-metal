# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Type

import torch
from torch.utils._pytree import tree_flatten, tree_map
from tracy import signpost

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.experimental.tt_symbiote.core.utils import (
    compare_fn_outputs,
    torch_dtype_to_ttnn_dtype,
    ttnn_dtype_to_torch_dtype,
)
from models.tt_transformers.tt.ccl import TT_CCL


@dataclass
class CCLManagerConfig:
    """Configuration for CCLManager."""

    mesh_device: Any
    num_links: Optional[int] = None
    topology: Optional[Any] = None

    def __post_init__(self):
        if self.num_links is None:
            self.num_links = 1
        if self.topology is None:
            self.topology = ttnn.Topology.Linear


@dataclass
class CCLManagerConfig:
    """Configuration for CCLManager."""

    mesh_device: Any
    num_links: Optional[int] = None
    topology: Optional[Any] = None

    def __post_init__(self):
        if self.num_links is None:
            self.num_links = 1
        if self.topology is None:
            try:
                self.topology = ttnn.Topology.Linear
            except Exception:
                self.topology = None


@dataclass
class DistributedTensorConfig:
    """Configuration for distributed tensor operations."""

    mesh_mapper: Any
    mesh_composer: Optional[Any] = None
    logical_shape_fn: Optional[Any] = None

    def get_logical_shape(self, sharded_shape):
        if self.logical_shape_fn is not None:
            return self.logical_shape_fn(sharded_shape)
        return sharded_shape


def logical_shape_for_batch_channel_sharding(mesh_shape):
    def _logical_shape(shape):
        shape = list(shape)
        logical_shape = [shape[0] * mesh_shape[0]] + shape[1:-1] + [shape[-1] * mesh_shape[1]]
        return tuple(logical_shape)

    return _logical_shape


@dataclass
class DistributedConfig:
    """Configuration for distributed operations."""

    mesh_device: Any
    tensor_config: Optional[DistributedTensorConfig] = None
    ccl_manager: Optional[Any] = None

    def __post_init__(self):
        num_devices = getattr(self.mesh_device, "get_num_devices", lambda: 1)()
        if num_devices > 1:
            try:
                if (
                    self.tensor_config is None
                    and hasattr(ttnn, "ShardTensor2dMesh")
                    and hasattr(ttnn, "ConcatMesh2dToTensor")
                ):
                    mesh_shape = getattr(self.mesh_device, "shape", (1, 1))
                    self.tensor_config = DistributedTensorConfig(
                        mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape, (0, -1)),
                        mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, mesh_shape, (0, -1)),
                        logical_shape_fn=logical_shape_for_batch_channel_sharding(mesh_shape),
                    )
            except Exception:
                pass
            try:
                if self.ccl_manager is None:
                    self.ccl_manager = TT_CCL(self.mesh_device)
            except Exception:
                pass

    def get_tensor_config_for_tensor(self, module_name, tensor):
        if tensor is not None:
            shape = getattr(self.mesh_device, "shape", (1, 1))
            shp = tuple(shape) if shape is not None else (1, 1)
            if len(tensor.shape) < 2 or (
                len(shp) >= 2 and (tensor.shape[-1] % shp[-1] != 0 or tensor.shape[0] % shp[0] != 0)
            ):
                return None
        return self.tensor_config


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
            res = TorchTTNNTensor(e.to_torch.clone())
            res.ttnn_tensor = None
        elif isinstance(e, ttnn.Tensor):
            res = TorchTTNNTensor(e).to_torch
            res.ttnn_tensor = None
        elif isinstance(e, torch.Tensor):
            res = e.clone()
        return res

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

        _record = os.environ.get("TT_SYMBIOTE_DISPATCH_TIMING", "0") == "1"
        if _record:
            begin = time.time()
        res = dispatch_to_ttnn(func.name(), ttnn_args, ttnn_kwargs)
        if _record:
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

        # Capture logical shape for im2col before unwrap (to_torch can make .shape physical/padded).
        im2col_logical_shape = None
        if func.name().startswith("aten::im2col") and len(torch_args) >= 5:
            from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

            if isinstance(torch_args[0], TorchTTNNTensor) and torch_args[0].ttnn_tensor is not None:
                im2col_logical_shape = tuple(int(i) for i in torch_args[0].ttnn_tensor.shape)
        with no_dispatch():
            func_args, func_kwargs = tree_map(unwrap_to_torch(func), torch_args), tree_map(
                unwrap_to_torch(func), torch_kwargs
            )
            # For im2col CPU fallback: input synced from device may have padded storage; copy to logical shape
            # so downstream view(1, C, 196, 4) is valid. Prefer shape from TTNN tensor (captured above).
            if func.name().startswith("aten::im2col") and len(torch_args) >= 5:
                from math import isqrt

                from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

                if isinstance(torch_args[0], TorchTTNNTensor) and isinstance(func_args[0], torch.Tensor):
                    t = func_args[0]
                    shp = (
                        im2col_logical_shape
                        if (im2col_logical_shape is not None and len(im2col_logical_shape) == 4)
                        else torch_args[0].shape
                    )
                    if len(shp) == 4:
                        N, C = int(shp[0]), int(shp[1])
                        expected_in_numel = N * C * int(shp[2]) * int(shp[3])
                        func_args = list(func_args)
                        # Vision fallback: known padded buffer 2965872 with logical (1,1152,28,28)=903168.
                        if t.numel() == 2965872 and N == 1 and C == 1152:
                            func_args[0] = t.flatten()[:903168].clone().reshape(1, 1152, 28, 28)
                        elif t.numel() > expected_in_numel:
                            # Padded buffer: .shape is logical; copy first expected_in_numel elements.
                            func_args[0] = (
                                t.flatten()[:expected_in_numel].clone().reshape(N, C, int(shp[2]), int(shp[3]))
                            )
                        elif t.numel() < expected_in_numel:
                            # Buffer smaller than .shape (e.g. .shape physical): use all elements, infer H*W.
                            spatial = t.numel() // (N * C)
                            if spatial > 0:
                                H = isqrt(spatial)
                                W = spatial // H
                                if H * W == spatial and H > 0 and W > 0:
                                    func_args[0] = t.flatten().clone().reshape(N, C, H, W)
                                else:
                                    func_args[0] = t.contiguous().clone()
                            else:
                                func_args[0] = t.contiguous().clone()
                        else:
                            func_args[0] = (
                                t[: int(shp[0]), : int(shp[1]), : int(shp[2]), : int(shp[3])].contiguous().clone()
                            )
                        func_args = tuple(func_args)
            # Avoid BFloat16 != float errors when one operand is from TTNN (bfloat16) and another is module param (float32)
            target_dtype = _dtype_from_torch_args(func_args, func_kwargs)
            if target_dtype is not None:

                def _cast_to_dtype(e):
                    if isinstance(e, torch.Tensor) and e.is_floating_point() and e.dtype != target_dtype:
                        return e.to(target_dtype)
                    return e

                func_args = tree_map(_cast_to_dtype, func_args)
                func_kwargs = tree_map(_cast_to_dtype, func_kwargs)
            _record = os.environ.get("TT_SYMBIOTE_DISPATCH_TIMING", "0") == "1"
            if _record:
                begin = time.time()
            func_res = (
                dispatch_to_torch(func.name(), func_args, func_kwargs)
                if can_dispatch_to_torch(func.name(), func_args, func_kwargs)
                else func(*func_args, **func_kwargs)
            )
            if _record:
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


def post_process_ttnn_module_output(self, result):
    result = tree_map(wrap_to_torch_ttnn_tensor, result)
    if self.device_state is not None:
        result = self.set_output_tensors_config(result)
    return result


def get_default_distributed_tensor_config(mesh_device=None, torch_tensor=None, module_name=None):
    from models.experimental.tt_symbiote.utils.device_management import DeviceInit

    state = None
    if mesh_device is not None:
        assert (
            DeviceInit.DEVICE_TO_STATE_DICT.get(mesh_device) is not None
        ), f"Device {mesh_device} not found in DeviceInit.DEVICE_TO_STATE_DICT, cannot set distributed config for mesh device."
        state = DeviceInit.DEVICE_TO_STATE_DICT[mesh_device]
    elif DeviceInit.DEVICE_TO_STATE_DICT is not None and len(DeviceInit.DEVICE_TO_STATE_DICT) == 1:
        state = next(iter(DeviceInit.DEVICE_TO_STATE_DICT.values()))
    if state is None:
        return None
    if torch_tensor is not None:
        return state.get_tensor_config_for_tensor(module_name, torch_tensor)
    return state.tensor_config


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
        # ...the real tensor is held as an element on the tensor.
        r.ttnn_tensor = ttnn_tensor  # Initialize ttnn_tensor
        r.elem = elem if not ttnn_tensor else None
        distributed_tensor_config = get_default_distributed_tensor_config(torch_tensor=elem)
        r.set_distributed_tensor_config(distributed_tensor_config)
        assert isinstance(r.elem, torch.Tensor) or isinstance(
            ttnn_tensor, ttnn.Tensor
        ), f"elem must be a torch.Tensor (or None when ttnn.Tensor is defined), but got {type(r.elem)}"
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
        begin = time.time()
        if self.elem is not None and self.elem.device.type != "meta" and self.ttnn_tensor is None:
            return self.elem

        def _to_torch(self):
            is_mesh_device = self.ttnn_distributed_tensor_config is not None
            if is_mesh_device:
                result = ttnn.to_torch(
                    self.ttnn_tensor, mesh_composer=self.ttnn_distributed_tensor_config.mesh_composer
                ).to(self.device, self.dtype)
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
        end = time.time()
        DispatchManager.record_timing(
            "TTNN",
            (
                ""
                if DispatchManager.current_module_name is None
                else DispatchManager.current_module_name + ".ttnn_to_torch"
            ),
            "ttnn_to_torch",
            {},
            end - begin,
        )
        return self.elem

    @staticmethod
    def to_ttnn(self):
        if self.ttnn_tensor is not None:
            return self.ttnn_tensor
        assert self.elem is not None, "Both ttnn_tensor and elem are None. This should not happen."
        if self.elem.device.type == "meta":
            raise RuntimeError(
                "Cannot convert META tensor to TTNN tensor. Please ensure the tensor is on a real device before conversion."
            )
        if self.elem.dtype not in TORCH_TO_TTNN:
            raise RuntimeError(f"Unsupported dtype {self.elem.dtype} for conversion to TTNN tensor.")
        self.ttnn_tensor = ttnn.from_torch(
            self.elem.cpu(),
            dtype=torch_dtype_to_ttnn_dtype(self.elem.dtype),
            mesh_mapper=self.ttnn_distributed_tensor_config.mesh_mapper
            if self.ttnn_distributed_tensor_config
            else None,
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
        begin = time.time()
        result = post_process_ttnn_module_output(self, self.forward(*func_args, **func_kwargs))
        end = time.time()
        DispatchManager.record_timing("TTNN", self.module_name, self.__class__.__name__ + "_forward", {}, end - begin)
        DispatchManager.set_current_module_name(None)
        return result


class LightweightRun(NormalRun):
    @staticmethod
    def to_torch(self):
        """Convert to PyTorch tensor."""

        def _to_torch(self):
            is_mesh_device = self.ttnn_distributed_tensor_config is not None
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
        if self.elem.device.type == "meta":
            raise RuntimeError(
                "Cannot convert META tensor to TTNN tensor. Please ensure the tensor is on a real device before conversion."
            )
        if self.elem.dtype not in TORCH_TO_TTNN:
            raise RuntimeError(f"Unsupported dtype {self.elem.dtype} for conversion to TTNN tensor.")
        self.ttnn_tensor = ttnn.from_torch(
            self.elem.cpu(),
            dtype=torch_dtype_to_ttnn_dtype(self.elem.dtype),
            mesh_mapper=self.ttnn_distributed_tensor_config.mesh_mapper
            if self.ttnn_distributed_tensor_config
            else None,
            layout=ttnn.TILE_LAYOUT if self.dtype == torch.bool else None,
        )
        return self.ttnn_tensor

    @staticmethod
    def module_run(self, *args, **kwds):
        print(f"{self.__class__.__name__}: {self.module_name} on device {self.device}")
        assert self.device is not None, "Device must be set for TTNN module execution."
        transform = compose_transforms(wrap_to_torch_ttnn_tensor, to_ttnn_wrap, set_device_wrap(self.device))
        func_args = tree_map(transform, args)
        # TODO: fix kwds not being passed correctly
        other_kwargs = {k: v for k, v in kwds.items() if "past_key_value" not in k}
        func_kwargs = tree_map(transform, other_kwargs)
        func_kwargs.update({k: v for k, v in kwds.items() if "past_key_value" in k})
        self.preprocess_weights()
        self.move_weights_to_device()
        result = post_process_ttnn_module_output(self, self.forward(*func_args, **func_kwargs))
        return result


class NormalRunWithFallback(NormalRun):
    @staticmethod
    def torch_dispatch(cls, func, types, args=(), kwargs=None):
        """Dispatch torch operations to TTNN when possible."""
        from models.experimental.tt_symbiote.core.dispatcher import can_dispatch_to_ttnn

        try:
            if can_dispatch_to_ttnn(func.name(), args, kwargs):
                rs = DispatchManager.dispatch_to_ttnn_wrapper(func, args, kwargs)
            else:
                rs = DispatchManager.dispatch_to_torch_wrapper(func, args, kwargs)
        except Exception as e:
            print(f"Error {e} in dispatching {func.name()}, falling back to torch")
            rs = DispatchManager.dispatch_to_torch_wrapper(func, args, kwargs)
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
                result = post_process_ttnn_module_output(self, self.forward(*func_args, **func_kwargs))
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

        from models.experimental.tt_symbiote.core.dispatcher import can_dispatch_to_ttnn

        copied_torch_tensors_args = tree_map(copy_to_torch(func), args)
        copied_torch_tensors_kwargs = tree_map(copy_to_torch(func), kwargs)
        result = DispatchManager.dispatch_to_torch_wrapper(func, copied_torch_tensors_args, copied_torch_tensors_kwargs)
        if can_dispatch_to_ttnn(func.name(), args, kwargs):
            ttnn_output = DispatchManager.dispatch_to_ttnn_wrapper(func, args, kwargs)
            # Compare inputs
            compare_fn_outputs(copied_torch_tensors_args, args, func.name())
            # Compare outputs
            compare_fn_outputs(result, ttnn_output, func.name())
            result = create_new_ttnn_tensors_using_torch_output(result, ttnn_output)
        return result

    @staticmethod
    def module_run(self, *args, **kwds):
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
            ttnn_output = post_process_ttnn_module_output(self, self.forward(*func_args, **func_kwargs))
            # Compare inputs
            compare_fn_outputs(copied_torch_tensors_args, func_args, self.__class__.__name__)
            # Compare outputs
            compare_fn_outputs(torch_output, ttnn_output, self.__class__.__name__)
            result = create_new_ttnn_tensors_using_torch_output(torch_output, ttnn_output)
        return result


class DPLRun(NormalRun):
    @staticmethod
    def module_run(self, *args, **kwds):
        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

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
            ttnn_output = post_process_ttnn_module_output(self, self.forward(*func_args, **func_kwargs))
            # Compare inputs
            compare_fn_outputs(
                tree_map(wrap_to_torch_ttnn_tensor, copied_torch_tensors_args),
                tree_map(wrap_to_torch_ttnn_tensor, func_args),
                self.__class__.__name__,
            )
            # Compare outputs
            compare_fn_outputs(torch_output, ttnn_output, self.__class__.__name__)
            result = create_new_ttnn_tensors_using_torch_output(torch_output, ttnn_output, assign_ttnn_to_torch=True)
        return result


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

        return create_new_ttnn_tensors_using_torch_output(torch_output, ttnn_output, assign_ttnn_to_torch=True)


class CPU(NormalRun):
    pass


# --- Trace Infrastructure ---


def _compute_tensor_signature(tensor) -> Tuple:
    """Compute hashable signature from tensor properties."""
    if isinstance(tensor, ttnn.Tensor):
        return (tuple(tensor.shape), tensor.dtype, tensor.layout)
    if hasattr(tensor, "ttnn_tensor") and tensor.ttnn_tensor is not None:
        t = tensor.ttnn_tensor
        return (tuple(t.shape), t.dtype, t.layout)
    if isinstance(tensor, torch.Tensor):
        return (tuple(tensor.shape), tensor.dtype)
    return ()


def _compute_args_signature(args) -> Tuple:
    """Compute signature for all tensor args."""
    sigs = []
    for arg in args:
        sig = _compute_tensor_signature(arg)
        if sig:
            sigs.append(sig)
    return tuple(sigs)


@dataclass(slots=True)
class TraceEntry:
    """Single trace cache entry."""

    trace_id: int
    trace_inputs: List[Any]
    trace_output: Any
    device: Any


# Registry of trace-enabled classes
_TRACE_ENABLED_CLASSES: Set[Type] = set()
_TRACE_RUNNING = False


def trace_enabled(cls: Type) -> Type:
    """
    Decorator to mark a TTNNModule subclass as trace-enabled.

    Usage:
        @trace_enabled
        class MyModule(TTNNModule):
            ...
    """
    _TRACE_ENABLED_CLASSES.add(cls)
    return cls


def is_trace_enabled(module) -> bool:
    """Check if module's class is trace-enabled."""
    return isinstance(module, tuple(_TRACE_ENABLED_CLASSES))


class TracedRun(NormalRun):
    """
    Traced execution mode with automatic caching.
    Only traces modules decorated with @trace_enabled.
    """

    _device: Any = None
    _cq_id: int = 0
    _input_memory_config: Any = None
    _trace_cache: Dict[Tuple, TraceEntry] = {}

    @classmethod
    def configure(
        cls,
        device=None,
        cq_id: int = 0,
        input_memory_config=None,
    ) -> None:
        """Configure traced run mode."""
        cls._device = device
        cls._cq_id = cq_id
        cls._input_memory_config = input_memory_config or ttnn.DRAM_MEMORY_CONFIG
        cls._trace_cache = {}

    @classmethod
    def cache_size(cls) -> int:
        return len(cls._trace_cache)

    @classmethod
    def cached_keys(cls) -> List[Tuple]:
        return list(cls._trace_cache.keys())

    @classmethod
    def release_all(cls) -> None:
        """Release all cached traces."""
        for entry in cls._trace_cache.values():
            ttnn.release_trace(entry.device, entry.trace_id)
        cls._trace_cache.clear()

    @classmethod
    def release(cls, module_name: str) -> int:
        """Release all traces for a specific module. Returns count released."""
        to_remove = [k for k in cls._trace_cache if k[0] == module_name]
        for key in to_remove:
            entry = cls._trace_cache.pop(key)
            ttnn.release_trace(entry.device, entry.trace_id)
        return len(to_remove)

    @staticmethod
    def _make_cache_key(module_name: str, args) -> Tuple:
        """Create cache key from module name and input signatures."""
        return (module_name, _compute_args_signature(args))

    @staticmethod
    def _copy_inputs_to_trace_buffer(new_args, trace_inputs) -> None:
        """Copy new inputs to trace input buffers."""
        trace_idx = 0
        for arg in new_args:
            if trace_idx >= len(trace_inputs):
                break
            trace_input = trace_inputs[trace_idx]
            if trace_input is None:
                trace_idx += 1
                continue
            if isinstance(arg, ttnn.Tensor):
                ttnn.copy(arg, trace_input)
                trace_idx += 1
            elif hasattr(arg, "ttnn_tensor") and arg.ttnn_tensor is not None:
                ttnn.copy(arg.ttnn_tensor, trace_input)
                trace_idx += 1

    @staticmethod
    def _capture_trace(module, func_args, func_kwargs, cache_key) -> TraceEntry:
        """Capture trace for module."""
        from loguru import logger

        device = module.device
        cq_id = TracedRun._cq_id
        mem_config = TracedRun._input_memory_config or ttnn.DRAM_MEMORY_CONFIG

        logger.debug(f"Capturing trace for {module.module_name}")

        # Warm-up
        _ = module.forward(*func_args, **func_kwargs)
        ttnn.synchronize_device(device)

        # Allocate persistent input buffers
        trace_inputs = []
        trace_func_args = []

        for arg in func_args:
            if isinstance(arg, ttnn.Tensor):
                host_tensor = arg.cpu() if arg.storage_type() != ttnn.StorageType.HOST else arg
                trace_input = ttnn.to_device(host_tensor, device, memory_config=mem_config)
                trace_inputs.append(trace_input)
                trace_func_args.append(trace_input)
            elif hasattr(arg, "ttnn_tensor") and arg.ttnn_tensor is not None:
                from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

                t = arg.ttnn_tensor
                host_tensor = t.cpu() if t.storage_type() != ttnn.StorageType.HOST else t
                trace_input = ttnn.to_device(host_tensor, device, memory_config=mem_config)
                trace_inputs.append(trace_input)
                new_arg = TorchTTNNTensor(trace_input)
                trace_func_args.append(new_arg)
            else:
                trace_inputs.append(None)
                trace_func_args.append(arg)

        # Capture
        trace_id = ttnn.begin_trace_capture(device, cq_id=cq_id)
        trace_output = module.forward(*trace_func_args, **func_kwargs)
        ttnn.end_trace_capture(device, trace_id, cq_id=cq_id)

        entry = TraceEntry(
            trace_id=trace_id,
            trace_inputs=trace_inputs,
            trace_output=trace_output,
            device=device,
        )
        TracedRun._trace_cache[cache_key] = entry
        logger.debug(f"Trace cached id={trace_id}, total={TracedRun.cache_size()}")
        return entry

    @staticmethod
    def module_run(self, *args, **kwds):
        assert self.device is not None, "Device must be set for TTNN module execution."

        # Transform inputs
        transform = compose_transforms(wrap_to_torch_ttnn_tensor, to_ttnn_wrap, set_device_wrap(self.device))
        func_args = tree_map(transform, args)
        other_kwargs = {k: v for k, v in kwds.items() if "past_key_value" not in k}
        func_kwargs = tree_map(transform, other_kwargs)
        func_kwargs.update({k: v for k, v in kwds.items() if "past_key_value" in k})

        begin = time.time()
        self.preprocess_weights()
        end = time.time()
        DispatchManager.set_current_module_name(self.module_name)
        DispatchManager.record_timing(
            "TTNN", self.module_name, self.__class__.__name__ + "_preprocess_weights", {}, end - begin
        )
        begin = time.time()
        self.move_weights_to_device()
        end = time.time()
        DispatchManager.record_timing(
            "TTNN", self.module_name, self.__class__.__name__ + "_move_weights_to_device", {}, end - begin
        )
        if NormalRun.signpost_mode is not None:
            signpost(f"{self.module_name}", f"{self.__class__.__name__}")

        begin = time.time()
        # Check if this module is trace-enabled
        global _TRACE_RUNNING
        if not is_trace_enabled(self) or _TRACE_RUNNING:
            if _TRACE_RUNNING:
                print(
                    f"{self.__class__.__name__}: {self.module_name} on device {self.device} [Not Trace-Enabled, Already Running Trace Elsewhere, Running Normally]"
                )
            else:
                print(
                    f"{self.__class__.__name__}: {self.module_name} on device {self.device} [Not Trace-Enabled, Running Normally]"
                )
            # Fall back to normal execution
            result = self.forward(*func_args, **func_kwargs)
            end = time.time()
            DispatchManager.record_timing(
                "TTNN", self.module_name, self.__class__.__name__ + "_forward", {}, end - begin
            )
            DispatchManager.set_current_module_name(None)
            return post_process_ttnn_module_output(self, result)

        # Traced execution path
        cache_key = TracedRun._make_cache_key(self.module_name, func_args)

        if cache_key in TracedRun._trace_cache:
            # Execute cached trace
            print(f"{self.__class__.__name__}: {self.module_name} on device {self.device} [TRACED]")
            entry = TracedRun._trace_cache[cache_key]
            TracedRun._copy_inputs_to_trace_buffer(func_args, entry.trace_inputs)
            ttnn.execute_trace(entry.device, entry.trace_id, cq_id=TracedRun._cq_id, blocking=False)
            result = entry.trace_output
        else:
            _TRACE_RUNNING = True
            print(
                f"{self.__class__.__name__}: {self.module_name} on device {self.device} [First Run - Capturing Trace]"
            )
            # Capture new trace
            entry = TracedRun._capture_trace(self, func_args, func_kwargs, cache_key)
            result = entry.trace_output
            _TRACE_RUNNING = False
        end = time.time()
        DispatchManager.record_timing("TTNN", self.module_name, self.__class__.__name__ + "_forward", {}, end - begin)
        DispatchManager.set_current_module_name(None)
        return post_process_ttnn_module_output(self, result)


def disable_trace(fn):
    """Decorator to disable trace capture during fn execution."""

    def new_fn(*args, **kwargs):
        global _TRACE_RUNNING
        was_tracing = _TRACE_RUNNING
        _TRACE_RUNNING = True
        try:
            return fn(*args, **kwargs)
        finally:
            _TRACE_RUNNING = was_tracing

    return new_fn


# Add at module level
_RUN_MODE_REGISTRY = {
    "LIGHTWEIGHT": LightweightRun,
    "NORMAL": NormalRun,
    "NORMAL_WITH_FALLBACK": NormalRunWithFallback,
    "SEL": SELRun,
    "DPL": DPLRun,
    "DPL_NO_ERROR_PROP": DPLRunNoErrorProp,
    "CPU": CPU,
    "TRACED": TracedRun,
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
