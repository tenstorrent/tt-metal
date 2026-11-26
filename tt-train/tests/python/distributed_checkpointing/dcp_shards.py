# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""This process's shards of model and optimizer tensors as numpy arrays, for the round-trip tests to compare.

A tensor's shards come from the host buffer of the tensor brought to the host, one per mesh coordinate asked for (by
default every coordinate of the device ``AutoContext`` holds, all of which belong to a single process; a rank of a
multi-rank world passes the coordinates it holds). That works on one device, on a mesh, and on a mesh spread over
several ranks, where the per-device tensors ``ttnn.get_device_tensors`` hands out still carry the mesh distribution and
``ttnn.to_torch`` refuses them.
"""

import itertools

import numpy as np
import ttnn

import ttml

NATIVE = ttml.autograd.PreferredPrecision.NATIVE
CARRIER = {ttnn.DataType.BFLOAT16: np.uint16, ttnn.DataType.FLOAT32: np.float32}


def every_coord() -> list[tuple[int, ...]]:
    """Every coordinate of the device ``AutoContext`` holds, in row-major order."""
    device = ttml.autograd.AutoContext.get_instance().get_device()
    return list(itertools.product(*(range(int(n)) for n in device.shape)))


def shard_arrays(value, coords) -> dict[tuple[int, ...], np.ndarray]:
    """``{coord: float32 array}`` of the shards of ``value`` (a ttnn tensor) at ``coords``, in logical row-major shape."""
    host = ttnn.from_device(value) if ttnn.is_tensor_storage_on_device(value) else value
    host = host.to(ttnn.ROW_MAJOR_LAYOUT)
    buffer = host.host_buffer()
    carrier = CARRIER[value.dtype]
    sizes = tuple(int(n) for n in value.shape)
    itemsize = np.dtype(carrier).itemsize
    rows = int(np.prod(sizes[:-1]))
    out = {}
    for coord in coords:
        shard = buffer.get_shard(ttnn.MeshCoordinate(list(coord)))
        if shard is None:  # a coordinate this process holds always has one: no silent gaps in a comparison
            raise AssertionError(f"no shard came back for coordinate {coord}")
        raw = np.from_dlpack(shard).reshape(-1).view(np.uint8)
        if raw.nbytes != rows * sizes[-1] * itemsize:  # the last dim may be padded to an alignment
            raw = np.ascontiguousarray(raw.reshape(rows, raw.nbytes // rows)[:, : sizes[-1] * itemsize]).reshape(-1)
        array = raw.view(carrier).reshape(sizes)
        if carrier is np.uint16:  # bf16 bits -> float32
            array = (array.astype(np.uint32) << 16).view(np.float32)
        out[tuple(coord)] = np.ascontiguousarray(array.astype(np.float32))
    return out


def tensor_shards(tensor, coords=None) -> list[np.ndarray]:
    """An autograd tensor's shards at ``coords`` (``every_coord()`` by default), in that order."""
    coords = every_coord() if coords is None else list(coords)
    arrays = shard_arrays(tensor.get_value(NATIVE), coords)
    return [arrays[tuple(c)] for c in coords]


def model_shards(params, coords=None) -> dict[str, list[np.ndarray]]:
    """``{name: tensor_shards}`` of a ``{name: autograd tensor}`` mapping, such as ``model.parameters()``."""
    return {name: tensor_shards(tensor, coords) for name, tensor in params.items()}


def optimizer_shards(optimizer, coords=None) -> tuple[dict[str, list[np.ndarray]], dict]:
    """``(shards, scalars)`` of an optimizer's state dict: every state tensor's ``tensor_shards`` keyed by its state
    path and parameter name, and every scalar (steps, learning rates, ...) by its path."""
    shards, scalars = {}, {}

    def walk(node, path):
        if isinstance(node, ttml.NamedParameters):
            for name, tensor in node.items():
                shards["/".join(path) + ":" + name] = tensor_shards(tensor, coords)
        elif isinstance(node, dict):
            for key, child in node.items():
                walk(child, path + (str(key),))
        else:
            scalars["/".join(path)] = node

    walk(optimizer.get_state_dict(), ())
    return shards, scalars


def assert_same_shards(actual: dict, expected: dict, what: str = "shards") -> None:
    """The same keys, at least one, and under each the same number of shards with exactly the same values."""
    assert expected, f"{what}: nothing to compare"
    assert actual.keys() == expected.keys(), f"{what}: {sorted(set(actual) ^ set(expected))[:4]}"
    for key in expected:
        assert len(actual[key]) == len(expected[key]), f"{what}: {key} has {len(actual[key])} shards"
        for device, (a, e) in enumerate(zip(actual[key], expected[key])):
            assert np.array_equal(a, e), f"{what}: {key} differs on device {device}"


def assert_some_differ(actual: dict, expected: dict) -> None:
    """A precondition: the fresh state is not already the saved one."""
    assert any(
        not np.array_equal(a, e) for key in expected for a, e in zip(actual[key], expected[key])
    ), "precondition: fresh state differs"
