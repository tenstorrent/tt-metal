# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise automatic native/additive selection and execution across the fidelity cutoffs."""

import pytest
import torch

import ttnn
from ttnn.operations.examples.reduce_block import program_descriptor_with_inline_kernels as example


@pytest.mark.parametrize(
    "fidelity,cutoffs",
    [
        (ttnn.MathFidelity.LoFi, (64, 32, 16)),
        (ttnn.MathFidelity.HiFi2, (30, 16, 8)),
        (ttnn.MathFidelity.HiFi3, (16, 8, 8)),
        (ttnn.MathFidelity.HiFi4, (10, 4, 4)),
    ],
)
@pytest.mark.parametrize("dim,axis", [("row", 0), ("col", 1), ("scalar", 2)])
@pytest.mark.parametrize("offset", [-1, 0])
def test_reduce_fidelity_crossover(device, fidelity, cutoffs, dim, axis, offset):
    if device.arch() != ttnn.device.Arch.BLACKHOLE:
        pytest.skip("Cutoffs were measured on Blackhole")
    tiles = cutoffs[axis] + offset
    ht, wt = (tiles, 1) if dim == "col" else (1, tiles)
    shape = (ht * 32, wt * 32)
    # Small exactly representable values also let LoFi exercise the native path accurately.
    values = (torch.arange(shape[0] * shape[1]).reshape(shape) % 7 - 3).to(torch.bfloat16)
    source = ttnn.from_torch(
        values,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=example.create_sharded_memory_config(shape),
    )
    planner = ttnn.reduce_planner
    hardware = planner.ReduceHardwareConfig(device.arch(), True, False, math_fidelity=fidelity)
    assert hardware.math_fidelity == fidelity
    plan = planner.make_reduce_plan(
        block=planner.ReduceBlockSpec(*shape, ttnn.bfloat16, ttnn.float32),
        reduce_math=planner.ReduceMath.AVG,
        reduce_dim=getattr(planner.ReduceDimension, {"row": "ROW", "col": "COLUMN", "scalar": "SCALAR"}[dim]),
        scalar=None,
        fp32_mode=planner.ReduceFp32Mode.FAST,
        hardware=hardware,
        input_policy=planner.ReduceInputPolicy.BULK_WAIT_BULK_POP,
    )
    assert plan.algorithm == (
        planner.ReduceAlgorithm.REDUCE_TILE if offset < 0 else planner.ReduceAlgorithm.ACCUMULATE_VIA_ADD
    )
    result = ttnn.to_torch(example.run_op(source, variant="automatic", dim=dim, Ht=ht, Wt=wt, math_fidelity=fidelity))
    if dim == "row":
        actual, expected = result[:, 0], values.float().mean(-1)
    elif dim == "col":
        actual, expected = result[0, :], values.float().mean(-2)
    else:
        actual, expected = result[0, 0], values.float().mean()
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.002)
