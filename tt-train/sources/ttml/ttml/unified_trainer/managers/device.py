# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import numpy as np
import ttnn
import ttml

from ..configs import DeviceConfig


class DeviceManager:
    """Opens the mesh ``config`` names as ttml's device mesh, and moves data to and from it; ``close`` releases it."""

    def __init__(self, config: DeviceConfig) -> None:
        self.config = config
        self._closed = False
        ttml.open_device_mesh(config.mesh)
        try:
            self.device = ttml.autograd.AutoContext.get_instance().get_device()
            # How a host array is placed on the mesh, and a tensor read back from it. No mapper: the mesh is a
            # single chip, which takes the array whole.
            self._mapper = None
            self._composer = ttml.core.distributed.concat_mesh_to_tensor_composer(self.device, 0)
        except BaseException:
            # The trainer has no handle on the device yet, so it can't close it.
            self.close()
            raise

    def to_device(self, values: np.ndarray, layout: Any, dtype: Any) -> Any:
        """Place a host array on the mesh as a ttml tensor of ``layout`` and ``dtype``."""
        return ttml.autograd.Tensor.from_numpy(np.ascontiguousarray(values), layout, dtype, self._mapper)

    def to_numpy(self, tensor: Any) -> np.ndarray:
        """Read a device tensor back to the host as float32."""
        return tensor.to_numpy(ttnn.DataType.FLOAT32, composer=self._composer)

    def close(self) -> None:
        """Release the device so the next run, or another process, can open it. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        self.device = None
        self._mapper = None
        self._composer = None
        ttml.autograd.AutoContext.get_instance().reset_graph()
        ttml.close_device_mesh()
