# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reject tensor computation and boundary conversions in a measured device pass."""

from contextlib import contextmanager
from unittest.mock import patch

from torch.utils._python_dispatch import TorchDispatchMode

import ttnn


class NoTorch(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        raise AssertionError(f"Host tensor operation inside device forward: {func}")


@contextmanager
def device_only():
    def forbidden(*args, **kwargs):
        raise AssertionError("Host/device conversion inside forward")

    with (
        NoTorch(),
        patch.object(ttnn, "from_torch", forbidden),
        patch.object(ttnn, "to_torch", forbidden),
        patch.object(ttnn, "copy_host_to_device_tensor", forbidden),
    ):
        yield
