# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.experimental.ops.descriptors.fusion import Parallel, Sequential, clear_build_cache
from tests.ttnn.unit_tests.operations.fused.parallel_sequential.test_parallel_sequential_global_cb import (
    build_identity_op,
)


@pytest.mark.parametrize(
    "persistent,sequential",
    [
        pytest.param(
            False,
            False,
            marks=pytest.mark.merge_gate,
        ),
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_fusion_cache_preserves_bound_outputs(device, persistent, sequential):
    clear_build_cache()

    def tensor(value):
        return ttnn.from_torch(
            torch.full((32, 32), value, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
        )

    def core(x):
        return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x, 0), ttnn.CoreCoord(x, 0))})

    retained = []
    container = None
    try:
        for iteration in range(3):
            outputs = [tensor(0), tensor(0)]
            if container is None or not persistent:
                inputs = [tensor(1), tensor(2)]
                first = build_identity_op(inputs[0], outputs[0], core(0), 1)
                second = build_identity_op(
                    outputs[0] if sequential else inputs[1], outputs[1], core(0 if sequential else 1), 1
                )
                container = (Sequential if sequential else Parallel)(first, second)
                retained.append((inputs, first, second, container))
            else:
                # Rebind the caller's output without rebuilding the descriptor.
                if not sequential:
                    first.output_tensors[0] = outputs[0]
                second.output_tensors[0] = outputs[1]
            returned = container.run()
            targets = [outputs[1]] if sequential else outputs
            expected = [1] if sequential else [1, 2]
            assert len(returned) == len(targets)
            for actual, target, value in zip(returned, targets, expected):
                assert torch.equal(ttnn.to_torch(target), torch.full((32, 32), value, dtype=torch.bfloat16))
                assert actual.buffer_address() == target.buffer_address()
            retained.append((outputs, returned))

        if persistent and not sequential:
            # Missing output slots still allocate ephemeral outputs; bound slots stay bound.
            first.output_tensors = []
            second.output_tensors[0] = tensor(0)
            returned = container.run()
            assert torch.equal(ttnn.to_torch(returned[0]), torch.ones((32, 32), dtype=torch.bfloat16))
            assert returned[1].buffer_address() == second.output_tensors[0].buffer_address()
    finally:
        clear_build_cache()
