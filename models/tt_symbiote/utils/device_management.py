"""Device management utilities for TTNN modules."""

from torch import nn


def set_device(obj, device):
    """Recursively set device for all TTNN modules in a model."""
    from models.tt_symbiote.core.module import TTNNModule

    if isinstance(obj, nn.Module):
        for name, module in obj._modules.items():
            if module is None:
                continue
            if isinstance(module, TTNNModule):
                module.to_device(device)
            set_device(module, device)
        for attr_name in dir(obj):
            if attr_name.startswith("_"):
                continue
            try:
                value = getattr(obj, attr_name)
            except:
                continue
            if isinstance(value, TTNNModule):
                value.to_device(device)
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, TTNNModule):
                        v.to_device(device)
                    set_device(v, device)
            if isinstance(value, (list, tuple)):
                for idx, v in enumerate(value):
                    if isinstance(v, TTNNModule):
                        v.to_device(device)
                    set_device(v, device)
    elif isinstance(obj, TTNNModule):
        obj.to_device(device)
        for attr_name in dir(obj):
            if attr_name.startswith("_"):
                continue
            try:
                value = getattr(obj, attr_name)
            except:
                continue
            if isinstance(value, TTNNModule):
                value.to_device(device)
                set_device(module, device)
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, TTNNModule):
                        v.to_device(device)
                    set_device(v, device)
            if isinstance(value, (list, tuple)):
                for idx, v in enumerate(value):
                    if isinstance(v, TTNNModule):
                        v.to_device(device)
                    set_device(v, device)
