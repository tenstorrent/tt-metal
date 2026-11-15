# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""This process's shards of a mesh tensor as numpy arrays, by mesh coordinate.

On a mesh spread over several ranks the per-device tensors ``ttnn.get_device_tensors`` hands out still carry the
mesh distribution, so ``ttnn.to_torch`` refuses them. The host buffer of the tensor brought to the host holds one
shard per coordinate this process owns, which is what the tests and the driver compare, on one rank or many.
"""

import numpy as np
import ttnn

import ttml

CARRIER = {ttnn.DataType.BFLOAT16: np.uint16, ttnn.DataType.FLOAT32: np.float32}


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


def param_shards(params, coords) -> dict[tuple[str, tuple[int, ...]], np.ndarray]:
    """Every parameter's shards at ``coords``, keyed by (name, coordinate)."""
    out = {}
    for name, tensor in params.items():
        for coord, array in shard_arrays(tensor.get_value(ttml.autograd.PreferredPrecision.NATIVE), coords).items():
            out[(name, coord)] = array
    return out


def optimizer_state(optimizer, coords) -> tuple[dict[tuple[str, tuple[int, ...]], np.ndarray], dict]:
    """``(shards, scalars)`` of an optimizer's state: every state tensor's shards at ``coords`` keyed by (state path,
    coordinate), and every scalar (steps, learning rates, ...) by its path."""
    shards, scalars = {}, {}

    def walk(node, path):
        if isinstance(node, ttml.NamedParameters):
            for name, tensor in node.items():
                value = tensor.get_value(ttml.autograd.PreferredPrecision.NATIVE)
                for coord, array in shard_arrays(value, coords).items():
                    shards[("/".join(path) + ":" + name, coord)] = array
        elif isinstance(node, dict):
            for key, child in node.items():
                walk(child, path + (str(key),))
        else:
            scalars["/".join(path)] = node

    walk(optimizer.get_state_dict(), ())
    return shards, scalars


def assert_same(actual: dict, expected: dict, what: str) -> None:
    """Exactly the same arrays under exactly the same keys, and at least one of them."""
    assert expected, f"{what}: nothing to compare"
    assert actual.keys() == expected.keys(), f"{what}: {sorted(set(actual) ^ set(expected))[:4]}"
    for key, array in expected.items():
        assert np.array_equal(actual[key], array), f"{what}: {key} differs from the saved shard"
