# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device management utilities for TTNN modules."""
import time

from torch import nn

from models.experimental.tt_symbiote.core.module import TTNNModule
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


def _initialize_module_on_device(module: TTNNModule, device, device_init=DeviceInit):
    """Initialize a TTNN module on the specified device."""
    module.to_device(device)
    if device.get_num_devices() > 1:
        module.set_device_state(device_init.init_state(device))


def set_device(obj, device, device_init=DeviceInit, **kwargs):
    """Recursively set device for all TTNN modules in a model."""
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
