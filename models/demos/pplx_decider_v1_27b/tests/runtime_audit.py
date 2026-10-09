# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Count host conversions, host round trips and torch ops issued inside a block of code.

Used by the runtime fallback audit (tests/pcc/test_runtime_audit.py) and by the perf harness
to show that a measured prefill pass stays on device after setup.

What is counted while the context is active:
- every call of ``ttnn.from_torch``, ``ttnn.to_torch``, ``ttnn.from_device``, ``ttnn.to_device``,
  ``ttnn.copy_host_to_device_tensor(_partial)``, ``ttnn.copy_device_to_host_tensor``,
  ``ttnn.allocate_tensor_on_host``, ``ttnn.Tensor.cpu`` and ``ttnn.Tensor.to_torch``
  (patched on the ``ttnn`` module and on ``ttnn.operations.core``, where ttnn's own Python
  helpers look these names up);
- every torch function call (``TorchFunctionMode``) and every aten op on a torch tensor
  (``TorchDispatchMode``), wherever it is issued from.
"""

from __future__ import annotations

import sys
from collections import Counter
from contextlib import contextmanager

from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

import ttnn

HOST_FUNCTIONS = (
    "from_torch",
    "to_torch",
    "from_device",
    "to_device",
    "copy_host_to_device_tensor",
    "copy_host_to_device_tensor_partial",
    "copy_device_to_host_tensor",
    "allocate_tensor_on_host",
)
TENSOR_METHODS = ("cpu", "to_torch")


class _TorchFunctionCounter(TorchFunctionMode):
    def __init__(self, counts: Counter):
        super().__init__()
        self.counts = counts

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.counts[f"torch_function:{getattr(func, '__name__', func)}"] += 1
        return func(*args, **(kwargs or {}))


class _TorchDispatchCounter(TorchDispatchMode):
    def __init__(self, counts: Counter):
        super().__init__()
        self.counts = counts

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.counts[f"aten:{func}"] += 1
        return func(*args, **(kwargs or {}))


@contextmanager
def count_host_calls():
    """Yields a ``Counter`` of host-side calls made inside the block (empty == fully on device)."""
    counts: Counter = Counter()
    core = sys.modules.get("ttnn.operations.core")
    patched = []

    def wrap(owner, name, label):
        original = getattr(owner, name, None)
        if original is None:
            return

        def counted(*args, **kwargs):
            counts[label] += 1
            return original(*args, **kwargs)

        setattr(owner, name, counted)
        patched.append((owner, name, original))

    for name in HOST_FUNCTIONS:
        wrap(ttnn, name, f"ttnn.{name}")
        if core is not None and getattr(core, name, None) is not None:
            wrap(core, name, f"ttnn.{name}")
    for name in TENSOR_METHODS:
        wrap(ttnn.Tensor, name, f"ttnn.Tensor.{name}")
    try:
        with _TorchFunctionCounter(counts), _TorchDispatchCounter(counts):
            yield counts
    finally:
        for owner, name, original in reversed(patched):
            setattr(owner, name, original)
