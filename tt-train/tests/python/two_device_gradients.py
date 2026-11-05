# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Devices' bf16 gradients and their exact mean, for the ``reduce_dtype`` tests of the DDP sync and of FSDP.

Each element scatters around a per-element mean with a spread as large as the mean, so the devices' values often
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


def make_gradients(rows: int, cols: int, seed: int, devices: int = 2) -> np.ndarray:
    """``devices`` bf16 gradients, stacked on dim 0, scattered around a per-element mean by as much as the mean."""
    rng = np.random.default_rng(seed)
    mean = rng.choice([-1.0, 1.0], size=(1, 1, rows, cols)) * 10.0 ** rng.uniform(-3, 0, size=(1, 1, rows, cols))
    return to_bf16((mean + np.abs(mean) * rng.standard_normal((devices, 1, rows, cols))).astype(np.float32))


def exact_mean(values: np.ndarray) -> np.ndarray:
    """bf16 of the exact mean of the devices' values (stacked on dim 0): what a perfect reduction returns."""
    return to_bf16(values.astype(np.float64).mean(axis=0, keepdims=True).astype(np.float32))


def device():
    return ttml.autograd.AutoContext.get_instance().get_device()


def one_parameter(gradient):
    """A ``NamedParameters`` holding one data-parallel parameter: a replicated value, and ``gradient`` as its gradient."""
    value = np.zeros(tuple(gradient.shape), dtype=np.float32)
    parameter = ttml.autograd.Tensor.from_numpy(
        value, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, ttnn.replicate_tensor_to_mesh_mapper(device())
    )
    parameter.set_grad(gradient)
    parameters = ttml.NamedParameters()
    parameters["parameter"] = parameter
    return parameters, parameter


def per_device_tensor(values: np.ndarray):
    """A bf16 tensor on a ``[1, N]`` mesh whose copy on device i is ``values[i]``."""
    mapper = ttnn.create_mesh_mapper(
        device(), ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementShard(0)])
    )
    return ttml.autograd.Tensor.from_numpy(values, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)


def per_device(tensor) -> np.ndarray:
    """``tensor`` read back with device i's copy at index i."""
    composer = ttnn.create_mesh_composer(device(), ttnn.MeshComposerConfig([1, 0]))
    return ttml.autograd.create_tensor(tensor).to_numpy(ttnn.DataType.FLOAT32, composer=composer)


def device_operations(call) -> list:
    """The device operations ``call`` runs, in order, recorded with ttnn's graph capture."""
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        call()
    finally:
        captured = ttnn.graph.end_graph_capture()
    return [name for name in ttnn.graph.extract_calltrace(captured) if name.endswith("DeviceOperation")]


def divides_before_gathering(operations: list) -> bool:
    """Whether a sum divides by the device count before an all-gather, as the fp32 reduce-scatter route does so that its
    all-gather carries bf16; ttnn's all_reduce divides after its sum is complete (as the C++ route test checks)."""
    if "BinaryNgDeviceOperation" not in operations or "AllGatherAsyncDeviceOperation" not in operations:
        return False
    return operations.index("BinaryNgDeviceOperation") < operations.index("AllGatherAsyncDeviceOperation")
