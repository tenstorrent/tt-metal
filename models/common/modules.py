# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
from enum import Enum
from typing import Any


class Mode(Enum):
    DECODE = 0
    PREFILL = 1


class LightweightModule:
    """Torch modules add a surprising amount of host overhead for attribute
    access and method calls. This class is a lightweight alternative that
    just wraps a forward function for now."""

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)


class MultiModeModule(LightweightModule):
    """Calls an appropriate forward function based on the mode:
       Mode.DECODE : decode_forward
       Mode.PREFILL: prefill_forward
    Uses a property for mode() so you can replace the getter and setter to
    e.g. pick up the mode from a model_config.
    """

    def __init__(self, mode=Mode.DECODE):
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
