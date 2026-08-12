# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the MiniMax-H3 bringup tests."""

import pytest
import torch

import ttnn

from ....utils.test import ring_params_8k_req_exact_devices, ring_params_req_exact_devices

# The two mesh shapes MiniMax-H3 is tuned for, as one list so a new shape is added in one place
# rather than at thirteen call sites. Use with
# `@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_PARAMS, indirect=[...])`.
#
# Both rows carry `require_exact_physical_num_devices`, so exactly one of them runs on any given
# cluster and the other skips: 4x8 skips on the 128-chip quad and 4x32 skips on a single Galaxy.
# That is the mechanism, not a workaround -- it is how Wan2.2 keeps one parametrize list covering
# everything from a 2x2 Quiet Box to the quad.
#
# `l1_small_size` is raised from the 16 KB default because the pipeline's conv3d and SDPA programs
# want the headroom. The fabric must be FABRIC_1D_RING: the CCLManager runs ring collectives, and on
# a plain FABRIC_1D (line) fabric a ring collective cannot resolve a forwarding direction and fails
# as `TT_FATAL fabric.cpp:174 forwarding_direction.has_value()`, which reads like a CCL bug.
#
# 4x32 additionally takes the 8 KB fabric router payload (`ring_params_8k`), matching Wan's 4x32
# pipeline and perf rows -- at 32 devices on the ring axis the default payload is the wrong tradeoff.
_TUNED_MESH_SHAPES = ((4, 8), (4, 32))
_RING_PARAMS_BY_SHAPE = {
    (4, 8): ring_params_req_exact_devices,
    # The quad traces its denoise step (`trace_denoise` in the pipeline's `_PRESETS_BH`), and a trace
    # needs somewhere to live. Same 150 MB Wan reserves for its 4x32 perf row; the region is only
    # reserved, so shapes that never capture a trace pay nothing but address space.
    (4, 32): {**ring_params_8k_req_exact_devices, "trace_region_size": 150000000},
}


def mesh_params(*, l1_small_size: int = 65536, **device_params) -> list:
    """`(mesh_device, device_params)` params for every tuned MiniMax-H3 mesh shape.

    `l1_small_size` differs per test -- the t2va pipeline and perf paths want 64 KB, ref2va runs at
    16 KB -- so it is an argument rather than baked in, and any other `device_params` key (a
    `trace_region_size`, say) passes straight through.
    """
    return [
        pytest.param(
            shape,
            {**_RING_PARAMS_BY_SHAPE[shape], "l1_small_size": l1_small_size, **device_params},
            id=f"{shape[0]}x{shape[1]}",
        )
        for shape in _TUNED_MESH_SHAPES
    ]


MESH_PARAMS = mesh_params()


def dit_mesh_params() -> list:
    """`(mesh_device, sp_axis, tp_axis, num_links, device_params, topology, is_fsdp)` per tuned shape.

    The transformer-level tests take the axes explicitly rather than reading a pipeline preset, so
    the axis assignment is repeated here -- and it is the same for both shapes, which is the point:
    TP stays on axis 0 at factor 4 and SP absorbs the rest, so 4x8 -> 4x32 moves only `sp_factor`,
    which every test body already derives from `tuple(mesh_device.shape)`.

    `device_params` travels inside the tuple rather than as its own parametrize because the fabric
    router payload differs per shape; crossing them independently would pair a 4x8 mesh with the
    4x32 router config.
    """
    return [
        pytest.param(
            shape,
            1,
            0,
            2,
            _RING_PARAMS_BY_SHAPE[shape],
            ttnn.Topology.Ring,
            False,
            id=f"{shape[0]}x{shape[1]}sp1tp0nl2_ring_is_fsdp0",
        )
        for shape in _TUNED_MESH_SHAPES
    ]


def randomize_norm_weights(module: torch.nn.Module, *, scale: float = 0.5) -> torch.nn.Module:
    """Give every `nn.RMSNorm` in `module` a non-trivial affine weight, in place.

    `nn.RMSNorm` initialises `weight` to all ones, so a reference model built with random weights
    (rather than loaded from the checkpoint) has an *identity* affine in every norm. That makes the
    norm weights invisible to a PCC comparison: a port that loaded the wrong norm weight, swapped two
    of them, or never loaded them at all would still match the reference exactly.

    MiniMax-H3 is full of RMSNorms -- `norm1`, `norm2`, the per-head `norm_q`/`norm_k`, the refiner's
    `final_norm` -- so this blind spot covers most of the model's non-matmul parameters. Measured on
    the token refiner at real dims, randomizing the norms moves "norm weights never loaded" from PCC
    1.000000 (undetectable) to 0.887, and "norm1/norm2 swapped" from 1.000000 to 0.986.

    Call this on the torch reference *before* taking its `state_dict`, so the TT module under test
    loads the same non-trivial values.
    """
    for submodule in module.modules():
        if isinstance(submodule, torch.nn.RMSNorm) and submodule.weight is not None:
            submodule.weight.data = 1.0 + scale * torch.randn_like(submodule.weight.data)
    return module
