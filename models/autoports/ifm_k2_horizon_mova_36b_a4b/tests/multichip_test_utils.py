# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-transfer prohibition used around all measured decoder forward calls."""

from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import ttnn


@contextmanager
def device_only():
    def forbidden(*args, **kwargs):
        raise AssertionError("Host transfer inside decoder forward")

    with ExitStack() as stack:
        for name in ("from_torch", "to_torch", "as_tensor", "to_device", "copy_host_to_device_tensor"):
            stack.enter_context(patch.object(ttnn, name, forbidden))
        yield
