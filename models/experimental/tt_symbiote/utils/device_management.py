# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device placement, tensor unwrapping, and torch-dispatch arg preparation for TTNN/Symbiote."""
import operator
import time
from functools import reduce

import torch
import ttnn
from torch import nn

from models.experimental.tt_symbiote.core.run_config import DispatchManager, DistributedConfig


class DeviceInit:
    DEVICE_TO_STATE_DICT = {}

    @classmethod
    def init_state(cls, device):
        """Initialize or return cached device state."""
        if device not in cls.DEVICE_TO_STATE_DICT:
            res = cls.init_state_impl(device)
            if res is not None:
                assert isinstance(res, DistributedConfig), f"Expected DistributedConfig, got {type(res)}"
            cls.DEVICE_TO_STATE_DICT[device] = res
        return cls.DEVICE_TO_STATE_DICT[device]

    @classmethod
    def init_state_impl(cls, device) -> DistributedConfig:
        """Override for custom device state; default returns DistributedConfig(device)."""
        return DistributedConfig(device)


def _initialize_module_on_device(module: "TTNNModule", device, device_init=DeviceInit):
    """Place TTNN module on device and set distributed state if multi-device."""
    module.to_device(device)
    if device.get_num_devices() > 1:
        module.set_device_state(device_init.init_state(device))


def set_device(obj, device, device_init=DeviceInit, **kwargs):
    """Set device for all TTNN modules in a model, recursively."""
    from models.experimental.tt_symbiote.core.module import TTNNModule

    module_names = {}
    if isinstance(obj, nn.Module):
        module_names = {module: name for name, module in obj.named_modules()}

    def _set_device_recursive(current_obj, module_name=None):
        if isinstance(current_obj, nn.Module):
            name = module_names.get(current_obj, module_name or "")
            if kwargs.get("register_forward_hook", True):

                def timed_call(original_call, module_name, module_class):
                    def new_call(*args, **kwargs):
                        begin = time.time()
                        DispatchManager.set_current_module_name(module_name)
                        result = original_call(*args, **kwargs)
                        DispatchManager.set_current_module_name(None)
                        end = time.time()
                        DispatchManager.record_timing("TorchModules", module_name, module_class, {}, end - begin)
                        return result

                    return new_call

                if hasattr(current_obj, "forward"):
                    if not hasattr(current_obj.forward, "_is_timed"):
                        current_obj.forward = timed_call(current_obj.forward, name, current_obj.__class__.__name__)
                        current_obj.forward._is_timed = True
                elif hasattr(current_obj, "__call__"):
                    if not hasattr(current_obj.__call__, "_is_timed"):
                        current_obj.__call__ = timed_call(current_obj.__call__, name, current_obj.__class__.__name__)
                        current_obj.__call__._is_timed = True

            for child_name, module in current_obj._modules.items():
                if module is None:
                    continue
                if isinstance(module, TTNNModule):
                    _initialize_module_on_device(module, device, device_init)
                _set_device_recursive(module)

            for attr_name in dir(current_obj):
                if attr_name.startswith("_"):
                    continue
                try:
                    value = getattr(current_obj, attr_name)
                except Exception as e:
                    continue
                if isinstance(value, TTNNModule):
                    _initialize_module_on_device(value, device, device_init)
                    _set_device_recursive(value)
                if isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, TTNNModule):
                            _initialize_module_on_device(v, device, device_init)
                        _set_device_recursive(v)
                if isinstance(value, (list, tuple)):
                    for v in value:
                        if isinstance(v, TTNNModule):
                            _initialize_module_on_device(v, device, device_init)
                        _set_device_recursive(v)
        elif isinstance(current_obj, TTNNModule):
            _initialize_module_on_device(current_obj, device, device_init)
            for attr_name in dir(current_obj):
                if attr_name.startswith("_"):
                    continue
                try:
                    value = getattr(current_obj, attr_name)
                except Exception as e:
                    continue
                if isinstance(value, (nn.Module, TTNNModule)):
                    if isinstance(value, TTNNModule):
                        _initialize_module_on_device(value, device, device_init)
                    _set_device_recursive(value)
                if isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, TTNNModule):
                            _initialize_module_on_device(v, device, device_init)
                        _set_device_recursive(v)
                if isinstance(value, (list, tuple)):
                    for v in value:
                        if isinstance(v, TTNNModule):
                            _initialize_module_on_device(v, device, device_init)
                        _set_device_recursive(v)

    _set_device_recursive(obj)


def unwrap_ttnn(tensor):
    """Extract underlying tensor from TTNN/Symbiote wrappers."""
    if tensor is None:
        return None
    curr = tensor
    while hasattr(curr, "ttnn_tensor") or hasattr(curr, "value") or hasattr(curr, "tensor"):
        curr = getattr(curr, "ttnn_tensor", getattr(curr, "value", getattr(curr, "tensor", curr)))
    return curr


def assimilate_to_device(tensor, device):
    """Unwrap, convert complex→real if needed, and move tensor to device."""
    if tensor is None:
        return None

    from models.experimental.tt_symbiote.core.utils import ensure_tile_layout

    curr = unwrap_ttnn(tensor)

    if isinstance(curr, ttnn.Tensor) and curr.storage_type() == ttnn.StorageType.DEVICE:
        return ensure_tile_layout(curr)

    import torch

    torch_t = curr if isinstance(curr, torch.Tensor) else ttnn.to_torch(curr)
    if torch.is_complex(torch_t):
        torch_t = torch_t.real

    return ttnn.from_torch(torch_t.to(torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)


def prepare_args_for_torch_dispatch(func_name, func_args, func_kwargs):
    """Preprocess dispatch args (e.g. trim padded view buffers); returns (args, kwargs)."""
    if func_name != "aten::view" or len(func_args) < 2:
        return func_args, func_kwargs

    t = func_args[0]
    shape = func_args[1]
    if not isinstance(t, torch.Tensor) or not isinstance(shape, (list, tuple)) or len(shape) == 0:
        return func_args, func_kwargs

    target_numel = reduce(operator.mul, shape, 1)
    if t.numel() <= target_numel or target_numel <= 0:
        return func_args, func_kwargs

    func_args = list(func_args)
    func_args[0] = t.flatten()[:target_numel].clone()
    return tuple(func_args), func_kwargs


def handle_view(func, args, kwargs):
    """Trim view args if padded, then reshape."""
    args, kwargs = prepare_args_for_torch_dispatch("aten::view", args, kwargs)
    return args[0].reshape(args[1])
