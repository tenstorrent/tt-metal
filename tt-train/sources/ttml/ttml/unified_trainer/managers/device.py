# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import numpy as np
import ttnn
import ttml


class DeviceManager:
    """Opens a single chip as ttml's device mesh; ``close`` releases it."""

    def __init__(self) -> None:
        self._closed = False
        ttml.open_device_mesh((1, 1))
        try:
            self.device = ttml.autograd.AutoContext.get_instance().get_device()
            self._composer = ttml.core.distributed.concat_mesh_to_tensor_composer(self.device, 0)
        except BaseException:
            # The trainer has no handle on the device yet, so it can't close it.
            self.close()
            raise

    def to_numpy(self, tensor: Any) -> np.ndarray:
        """Read a device tensor back to the host as float32."""
        return tensor.to_numpy(ttnn.DataType.FLOAT32, composer=self._composer)

    def close(self) -> None:
        """Release the device so the next run, or another process, can open it. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        self.device = None
        self._composer = None
        ttml.autograd.AutoContext.get_instance().reset_graph()
        ttml.close_device_mesh()
