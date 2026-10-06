# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fail a measured layer pass on torch execution or a host tensor boundary."""

from contextlib import ExitStack, contextmanager
from unittest.mock import patch

from torch.utils._python_dispatch import TorchDispatchMode


class RejectTorch(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        raise AssertionError(f"Host torch operation in decoder pass: {func}")


@contextmanager
def device_only():
    with ExitStack() as stack:
        stack.enter_context(RejectTorch())
        for name in ("from_torch", "to_torch", "as_tensor", "to_device", "zeros", "ones", "full", "arange"):
            stack.enter_context(patch(f"ttnn.{name}", side_effect=AssertionError(f"Host boundary ttnn.{name}")))
        yield
