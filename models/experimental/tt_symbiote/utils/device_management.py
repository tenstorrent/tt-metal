# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device management utilities for TTNN modules."""
import time

from torch import nn

from models.experimental.tt_symbiote.core.run_config import DispatchManager, DistributedConfig


class DeviceInit:
    DEVICE_TO_STATE_DICT = {}

    @classmethod
    def init_state(cls, device):
        """Initialize device state if not already initialized."""
        if device not in cls.DEVICE_TO_STATE_DICT:
            res = cls.init_state_impl(device)
            if res is not None:
                assert isinstance(res, DistributedConfig), f"Expected DistributedConfig, got {type(res)}"
            cls.DEVICE_TO_STATE_DICT[device] = res
        return cls.DEVICE_TO_STATE_DICT[device]

    @classmethod
    def init_state_impl(cls, device) -> DistributedConfig:
        """Implementation-specific device state initialization."""
        # Placeholder for actual device state initialization logic
        return DistributedConfig(device)


def _initialize_module_on_device(module: "TTNNModule", device, device_init=DeviceInit):
    """Initialize a TTNN module on the specified device."""
    module.to_device(device)
    if device.get_num_devices() > 1:
        module.set_device_state(device_init.init_state(device))


def set_device(obj, device, device_init=DeviceInit, **kwargs):
    """Recursively set device for all TTNN modules in a model."""
    from models.experimental.tt_symbiote.core.module import TTNNModule

    # Build module name mapping before recursion
    module_names = {}
    if isinstance(obj, nn.Module):
        module_names = {module: name for name, module in obj.named_modules()}

    def _set_device_recursive(current_obj, module_name=None):
        if isinstance(current_obj, nn.Module):
            # Get the name for this module from the mapping
            name = module_names.get(current_obj, module_name or "")

            # Register forward hook for this module
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


# --- Patches (run at import end): pad fallback + linear dtype alignment ---


def _install_pad_fallback():
    """Force aten::pad to torch fallback so DPL comparison does not hit TTNN tile-rounding shape mismatch."""
    from models.experimental.tt_symbiote.core.dispatchers import default_dispatcher

    _orig = default_dispatcher.can_dispatch_to_ttnn

    def _can_dispatch_to_ttnn_with_pad_fallback(func_name, args=None, kwargs=None):
        if func_name == "aten::pad":
            return False
        return _orig(func_name, args, kwargs)

    default_dispatcher.can_dispatch_to_ttnn = _can_dispatch_to_ttnn_with_pad_fallback


_original_dispatch_to_torch_wrapper = None


def _align_linear_dtypes_for_fallback(func_name, func_args, func_kwargs):
    """Cast weight/bias to input dtype for aten::linear so torch fallback works when params are fp32 (e.g. DPL)."""
    import torch

    if func_name != "aten::linear" or not func_args or not isinstance(func_args[0], torch.Tensor):
        return func_args, func_kwargs
    inp = func_args[0]
    target_dtype = inp.dtype
    args_list = list(func_args)
    if len(args_list) >= 2 and isinstance(args_list[1], torch.Tensor) and args_list[1].dtype != target_dtype:
        args_list[1] = args_list[1].to(target_dtype)
    if len(args_list) >= 3 and args_list[2] is not None and isinstance(args_list[2], torch.Tensor):
        if args_list[2].dtype != target_dtype:
            args_list[2] = args_list[2].to(target_dtype)
    kwargs_dict = dict(func_kwargs) if func_kwargs else {}
    if "bias" in kwargs_dict and kwargs_dict["bias"] is not None and isinstance(kwargs_dict["bias"], torch.Tensor):
        if kwargs_dict["bias"].dtype != target_dtype:
            kwargs_dict["bias"] = kwargs_dict["bias"].to(target_dtype)
    return tuple(args_list), kwargs_dict


def _patched_dispatch_to_torch_wrapper(func, torch_args, torch_kwargs):
    """Wrapper that aligns linear dtypes before calling the original dispatch_to_torch_wrapper."""
    import torch
    from torch.utils._pytree import tree_map

    from models.experimental.tt_symbiote.core.run_config import unwrap_to_torch, wrap_from_torch

    if func.name() != "aten::linear":
        return _original_dispatch_to_torch_wrapper(func, torch_args, torch_kwargs)
    unwrap = unwrap_to_torch(func)
    func_args = tree_map(unwrap, torch_args)
    func_kwargs = tree_map(unwrap, torch_kwargs)
    func_args, func_kwargs = _align_linear_dtypes_for_fallback(func.name(), func_args, func_kwargs)
    args_list = list(torch_args)
    if len(args_list) >= 2 and isinstance(func_args[1], torch.Tensor):
        args_list[1] = wrap_from_torch(func_args[1])
    if len(args_list) >= 3 and args_list[2] is not None and isinstance(func_args[2], torch.Tensor):
        args_list[2] = wrap_from_torch(func_args[2])
    new_torch_kwargs = dict(torch_kwargs) if torch_kwargs else {}
    if (
        "bias" in new_torch_kwargs
        and new_torch_kwargs["bias"] is not None
        and isinstance(func_kwargs.get("bias"), torch.Tensor)
    ):
        new_torch_kwargs["bias"] = wrap_from_torch(func_kwargs["bias"])
    return _original_dispatch_to_torch_wrapper(func, tuple(args_list), new_torch_kwargs)


def _install_linear_dtype_patch():
    """Patch DispatchManager.dispatch_to_torch_wrapper to align linear dtypes (keeps run_config.py unchanged)."""
    global _original_dispatch_to_torch_wrapper
    if _original_dispatch_to_torch_wrapper is None:
        _original_dispatch_to_torch_wrapper = DispatchManager.dispatch_to_torch_wrapper
        DispatchManager.dispatch_to_torch_wrapper = staticmethod(_patched_dispatch_to_torch_wrapper)


_install_pad_fallback()
_install_linear_dtype_patch()
