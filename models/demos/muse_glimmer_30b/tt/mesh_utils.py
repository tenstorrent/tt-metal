# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Mesh helpers: system mesh selection from env, TP submeshes and DP replica submeshes."""

import os

import pytest

import ttnn

MESH_DEVICE_SHAPES = {
    "P300x2": (1, 4),
    "P150x4": (1, 4),
    "QB2": (1, 4),
    "P150x8": (1, 8),
    "BHGLX": (8, 4),
    "GLX": (8, 4),
}


def system_mesh_shape_from_env():
    """Mesh shape from MESH_DEVICE (known name or "RxC"); None if unset/unknown (root fixture opens the full system mesh)."""
    name = os.environ.get("MESH_DEVICE")
    if not name:
        return None
    if name in MESH_DEVICE_SHAPES:
        return MESH_DEVICE_SHAPES[name]
    if "x" in name.lower():
        try:
            r, c = name.lower().split("x")
            return (int(r), int(c))
        except ValueError:
            pass
    return None


def _shape(mesh_device):
    s = tuple(mesh_device.shape)
    return (int(s[0]), int(s[1]))


def tp_submesh(mesh_device, tp: int):
    """Return a (1, tp) mesh carved out of mesh_device."""
    shape = _shape(mesh_device)
    n = shape[0] * shape[1]
    assert tp <= n, f"tp={tp} exceeds {n} devices"
    if shape == (1, tp):
        return mesh_device
    if shape[0] == 1:
        return mesh_device.create_submesh(ttnn.MeshShape(1, tp))
    if n == tp:  # e.g. QB2 opened as 2x2
        mesh_device.reshape(ttnn.MeshShape(1, tp))
        return mesh_device
    if tp <= shape[1]:
        return mesh_device.create_submesh(ttnn.MeshShape(1, tp))
    if tp == 8 and shape[1] == 4:
        mesh_device.reshape(ttnn.MeshShape(4, 8))
        return mesh_device.create_submesh(ttnn.MeshShape(1, 8))
    raise ValueError(f"Cannot build a (1,{tp}) submesh from mesh {shape}")


def dp_submeshes(mesh_device, dp: int, tp: int) -> list:
    """dp replica meshes, each (1, tp)."""
    if dp == 1:
        return [tp_submesh(mesh_device, tp)]
    from models.tt_transformers.tt.generator import create_submeshes

    subs = create_submeshes(mesh_device, dp)
    for sm in subs:
        assert _shape(sm) == (1, tp), f"replica submesh must be (1,{tp}), got {_shape(sm)} (dp={dp})"
    return subs


def device_params(trace_region_size=None) -> dict:
    params = {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        "num_command_queues": 1,
        "l1_small_size": 24576,
        "trace_region_size": 256_000_000 if trace_region_size is None else trace_region_size,
    }
    return params


def mesh_test_params(trace_region_size=None):
    """Decorator: indirect mesh_device (only if MESH_DEVICE set) + device_params, plus plain `tp` in [4, 8]."""

    def deco(func):
        shape = system_mesh_shape_from_env()
        func = pytest.mark.parametrize("tp", [4, 8], ids=["tp4", "tp8"])(func)
        func = pytest.mark.parametrize("device_params", [device_params(trace_region_size)], indirect=True)(func)
        if shape is not None:
            func = pytest.mark.parametrize("mesh_device", [shape], indirect=True)(func)
        return func

    return deco


def skip_and_submesh(mesh_device, tp: int):
    """pytest-skip if the system has fewer than tp devices, else return the (1, tp) submesh."""
    if tp > mesh_device.get_num_devices():
        pytest.skip(f"tp={tp} needs more devices than the {mesh_device.get_num_devices()} available")
    return tp_submesh(mesh_device, tp)
