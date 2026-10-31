# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""A tensor on a different device than q is rejected by the forward and backward chunk_gated_delta_rule ops.

The kernels are handed raw buffer addresses, so a buffer that lives on another device would be read as if it
were local; `check_same_device` in each validator turns that into an error.  This needs two devices, so it
uses the shared `tp_mesh` fixture (a [1, 2] mesh, which skips the module when it cannot be opened) and puts
q on one 1x1 submesh and the tensor under test on the other.

The shapes are the smallest the ops accept (B=1, T=64, HK=HV=1, K=V=32, chunk 32); every tensor is valid
apart from where it lives, so the same-device check is the only thing that can fail.
"""

from __future__ import annotations

import gc

import pytest
import torch

import ttnn
import ttml

pytestmark = pytest.mark.requires_device

CHUNK = 32
T = 64
D = 32


@pytest.fixture(scope="module")
def submeshes(tp_mesh):
    """Two 1x1 submeshes of the [1, 2] mesh, one per device."""
    mesh = ttml.autograd.AutoContext.get_instance().get_device()
    first = mesh.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0))
    second = mesh.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 1))
    yield first, second
    # A submesh keeps its parent mesh alive, and a mesh that stays open leaves the fabric initialised, so the
    # next module that opens a single device times out on its ethernet cores.  Drop every reference (the
    # tensors of failed tests live in their tracebacks) and close the submeshes before the fixture tears the
    # mesh down.
    gc.collect()
    for submesh in (first, second):
        ttnn.close_mesh_device(submesh)
    del first, second
    gc.collect()


def _tensor(device, *shape, dtype=ttnn.bfloat16):
    return ttnn.from_torch(torch.zeros(*shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _fw_inputs(device_of):
    """(args, kwargs) for the forward; `device_of(name)` says which device each tensor goes on."""
    f32 = ttnn.float32
    args = [
        _tensor(device_of("q"), 1, 1, T, D),
        _tensor(device_of("k"), 1, 1, T, D),
        _tensor(device_of("v"), 1, 1, T, D),
        _tensor(device_of("g"), 1, 1, T, 1, dtype=f32),
        _tensor(device_of("beta"), 1, 1, T, 1, dtype=f32),
    ]
    kwargs = {"chunk_size": CHUNK, "initial_state": _tensor(device_of("initial_state"), 1, 1, D, D, dtype=f32)}
    return args, kwargs


def _bw_inputs(device_of):
    """(args, kwargs) for the backward with every optional input present."""
    f32 = ttnn.float32
    args = [
        _tensor(device_of("q"), 1, 1, T, D),
        _tensor(device_of("k"), 1, 1, T, D),
        _tensor(device_of("v"), 1, 1, T, D),
        _tensor(device_of("g"), 1, 1, T, 1),
        _tensor(device_of("beta"), 1, 1, T, 1),
        _tensor(device_of("d_out"), 1, 1, T, D),
    ]
    kwargs = {
        "chunk_size": CHUNK,
        "initial_state": _tensor(device_of("initial_state"), 1, 1, D, D),
        "d_final_state": _tensor(device_of("d_final_state"), 1, 1, D, D),
        "h": _tensor(device_of("h"), 1, T // CHUNK, D, D, dtype=f32),
        "v_new": _tensor(device_of("v_new"), 1, 1, T, D, dtype=f32),
        "a": _tensor(device_of("a"), 1, 1, T, CHUNK, dtype=f32),
    }
    return args, kwargs


FW_NAMES = ["k", "v", "g", "beta", "initial_state"]
BW_NAMES = ["k", "v", "g", "beta", "d_out", "initial_state", "d_final_state", "h", "v_new", "a"]


def _fw(args, kwargs):
    return ttml.ops.metal.chunk_gated_delta_rule_fw(*args, D, **kwargs)


def _bw(args, kwargs):
    return ttml.ops.metal.chunk_gated_delta_rule_bw(*args, D, **kwargs)


def test_fw_control_all_on_one_device(submeshes):
    """The same inputs, all on one device, run: the rejections below are about placement and nothing else."""
    first, _ = submeshes
    args, kwargs = _fw_inputs(lambda name: first)
    outputs = _fw(args, kwargs)
    assert outputs[0].shape[-1] == D


def test_bw_control_all_on_one_device(submeshes):
    first, _ = submeshes
    args, kwargs = _bw_inputs(lambda name: first)
    outputs = _bw(args, kwargs)
    assert outputs[0] is not None


@pytest.mark.parametrize("name", FW_NAMES)
def test_fw_rejects_a_tensor_on_another_device(name, submeshes, expect_error):
    first, second = submeshes
    args, kwargs = _fw_inputs(lambda n: second if n == name else first)
    with expect_error(RuntimeError, f"{name} must be on the same device as"):
        _fw(args, kwargs)


@pytest.mark.parametrize("name", BW_NAMES)
def test_bw_rejects_a_tensor_on_another_device(name, submeshes, expect_error):
    first, second = submeshes
    args, kwargs = _bw_inputs(lambda n: second if n == name else first)
    with expect_error(RuntimeError, f"{name} must be on the same device as q"):
        _bw(args, kwargs)
