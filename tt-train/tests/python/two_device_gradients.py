# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Two devices' bf16 gradients and their exact mean, for the ``reduce_dtype`` tests of the DDP sync and of FSDP.

Each element scatters around a per-element mean with a spread as large as the mean, so the two devices' values often
cancel partly, which is where a bf16 sum loses bits.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np
import ttnn
import ttml


def to_bf16(values: np.ndarray) -> np.ndarray:
    """Round float32 to the nearest bf16 (ties to even), returned as float32."""
    return np.asarray(values, dtype=np.float32).astype(ml_dtypes.bfloat16).astype(np.float32)


def make_gradients(rows: int, cols: int, seed: int) -> np.ndarray:
    """Two devices' bf16 gradients, stacked on dim 0, scattered around a per-element mean by as much as the mean."""
    rng = np.random.default_rng(seed)
    mean = rng.choice([-1.0, 1.0], size=(1, 1, rows, cols)) * 10.0 ** rng.uniform(-3, 0, size=(1, 1, rows, cols))
    return to_bf16((mean + np.abs(mean) * rng.standard_normal((2, 1, rows, cols))).astype(np.float32))


def exact_mean(values: np.ndarray) -> np.ndarray:
    """bf16 of the exact mean of the two devices' values: what a perfect reduction returns."""
    return to_bf16((values.astype(np.float64).sum(axis=0, keepdims=True) / 2.0).astype(np.float32))


def device():
    return ttml.autograd.AutoContext.get_instance().get_device()


def per_device_tensor(values: np.ndarray):
    """A bf16 tensor on a ``[1, 2]`` mesh whose copy on device i is ``values[i]``."""
    mapper = ttnn.create_mesh_mapper(
        device(), ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementShard(0)])
    )
    return ttml.autograd.Tensor.from_numpy(values, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)


def per_device(tensor) -> np.ndarray:
    """``tensor`` read back with device i's copy at index i."""
    composer = ttnn.create_mesh_composer(device(), ttnn.MeshComposerConfig([1, 0]))
    return ttml.autograd.create_tensor(tensor).to_numpy(ttnn.DataType.FLOAT32, composer=composer)
