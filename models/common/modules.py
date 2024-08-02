# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
from enum import Enum
from typing import Any
import ttnn


class Mode(Enum):
    DECODE = 0
    PREFILL = 1


class WeightSetting:
    def __init__(self, state_dict_key, dtype, conversion_fn=None, mapper=None):
        self.state_dict_key = state_dict_key
        self.dtype = dtype
        self.conversion_fn = conversion_fn
        self.mapper = mapper


class LightweightModule:
    """Torch modules add a surprising amount of host overhead for attribute
    access and method calls. This class is a lightweight alternative that
    just wraps a forward function for now."""

    def __init__(self, device):
        self.device = device
        self.is_device_mesh = device.__class__.__name__ == "DeviceMesh"
        self.weight_settings = {}

    def load_weights(self, state_dict, weight_cache_path=None):
        for name, ws in self.weight_settings.items():
            torch_tensor = state_dict[ws.state_dict_key]
            torch_tensor = ws.conversion_fn(torch_tensor) if ws.conversion_fn else torch_tensor
            cache_file_name = weight_cache_path / ws.state_dict_key if weight_cache_path else None
            self.__dict__[name] = ttnn.as_tensor(
                torch_tensor,
                device=self.device,
                dtype=ws.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_file_name,
                mesh_mapper=ws.mapper,
            )

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)


class MultiModeModule(LightweightModule):
    """Calls an appropriate forward function based on the mode:
       Mode.DECODE : decode_forward
       Mode.PREFILL: prefill_forward
    Uses a property for mode() so you can replace the getter and setter to
    e.g. pick up the mode from a model_config.
    """

    def __init__(self, device, mode=Mode.DECODE):
        super().__init__(device)
        self._mode = mode

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value

    def forward(self, *args, **kwargs):
        if self.mode == Mode.DECODE:
            return self.decode_forward(*args, **kwargs)
        elif self.mode == Mode.PREFILL:
            return self.prefill_forward(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def decode_forward(self, *args, **kwargs):
        # Placeholder method for decode_forward
        pass

    def prefill_forward(self, *args, **kwargs):
        # Placeholder method for prefill_forward
        pass
