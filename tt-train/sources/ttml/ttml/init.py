# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Parameter initialization functions for ttml modules.

Each function returns an initializer with signature::

    init_fn(shape, layout=None, mesh_device=None) -> Tensor

- Single device: ``ttnn.zeros`` / ``ones`` on AutoContext device (full logical shape).
- Mesh: always a host buffer whose shape equals the parameter's **global** logical
  shape (``TensorMetadata.shape``), then ``from_numpy`` + ``TensorToMesh`` — same as
  ``ttnn::distributed::create_distributed_tensor`` (buffer size = global shape).
  Do not pre-shard the NumPy array here; the mapper splits. Replicate when no layout.
"""

from __future__ import annotations

import numpy as np
import ml_dtypes
import ttnn
import ttml


def _get_device():
    return ttml.autograd.AutoContext.get_instance().get_device()


def _mesh_ndim(mesh_device):
    mesh_shape = mesh_device.shape
    return mesh_shape.dims()


def _to_tensor(arr, layout=None, mesh_device=None):
    """Numpy array → ttml Tensor on mesh.

    With ``layout`` (from TpPlan): shard/replicate per layout. With ``layout is None``
    but ``mesh_device`` set: **replicate** full tensor on the mesh so non-TP params
    (embedding, norm gamma, etc.) match distributed ops.
    """
    from ttml.distributed.layout import Layout, set_layout

    mapper = None
    stamp_layout = None
    if mesh_device is not None:
        rank = len(arr.shape)
        if layout is not None:
            mapper = layout.build_mapper(mesh_device, tensor_rank=rank)
            stamp_layout = layout
        else:
            mapper = ttml.core.distributed.replicate_tensor_to_mesh_mapper(mesh_device)
            stamp_layout = Layout(ndim=_mesh_ndim(mesh_device))
    tensor = ttml.autograd.Tensor.from_numpy(arr, ttnn.Layout.TILE, mapper=mapper)
    if stamp_layout is not None:
        set_layout(tensor, stamp_layout)
    return tensor


def _stamp(tensor, layout):
    if layout is not None:
        from ttml.distributed.layout import set_layout

        set_layout(tensor, layout)
    return tensor


def uniform(low: float = 0.0, high: float = 1.0):
    """Uniform distribution over [low, high)."""

    def init_fn(shape, layout=None, mesh_device=None):
        arr = np.random.uniform(low, high, shape).astype(ml_dtypes.bfloat16)
        return _to_tensor(arr, layout, mesh_device)

    return init_fn


def normal(mean: float = 0.0, std: float = 1.0):
    """Normal (Gaussian) distribution."""

    def init_fn(shape, layout=None, mesh_device=None):
        arr = np.random.normal(mean, std, shape).astype(ml_dtypes.bfloat16)
        return _to_tensor(arr, layout, mesh_device)

    return init_fn


def zeros():
    """All zeros."""

    def init_fn(shape, layout=None, mesh_device=None):
        if mesh_device is not None:
            arr = np.zeros(shape, dtype=ml_dtypes.bfloat16)
            return _to_tensor(arr, layout, mesh_device)
        device = _get_device()
        t = ttnn.zeros(
            shape, device=device, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE
        )
        return _stamp(ttml.autograd.create_tensor(t), layout)

    return init_fn


def ones():
    """All ones."""

    def init_fn(shape, layout=None, mesh_device=None):
        if mesh_device is not None:
            arr = np.ones(shape, dtype=ml_dtypes.bfloat16)
            return _to_tensor(arr, layout, mesh_device)
        device = _get_device()
        t = ttnn.ones(
            shape, device=device, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE
        )
        return _stamp(ttml.autograd.create_tensor(t), layout)

    return init_fn


__all__ = [
    "normal",
    "ones",
    "uniform",
    "zeros",
]
