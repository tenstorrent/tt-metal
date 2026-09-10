# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import ttnn


@pytest.mark.parametrize(
    "repeats,output_l1,eligible",
    [
        ([1, 1, 2, 1], False, True),
        ([1, 1, 2, 1], True, False),
        ([0, 1, 2, 1], False, False),
        ([1, 1, 1, 1], False, False),
    ],
)
def test_repeat_support_query_matches_forced_gate(device, expect_error, repeats, output_l1, eligible):
    api = ttnn._ttnn.operations.data_movement
    value = torch.randn([1, 1, 32, 32], dtype=torch.bfloat16)
    tensor = ttnn.from_torch(value, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    memory = ttnn.L1_MEMORY_CONFIG if output_l1 else ttnn.DRAM_MEMORY_CONFIG
    decision = api.repeat_codegen_support(tensor, repeats, memory_config=memory)
    assert decision.eligible == eligible
    # Repeat currently has no performance demotions; refusal is not demotion.
    assert not decision.demoted
    if eligible:
        output = api.repeat_force_codegen(tensor, repeats, memory_config=memory)
        assert torch.equal(ttnn.to_torch(output), value.repeat(repeats))
    else:
        with expect_error(RuntimeError, "repeat_force_codegen invoked"):
            api.repeat_force_codegen(tensor, repeats, memory_config=memory)


def test_repeat_support_query_validates_preallocated_output(device, expect_error):
    api = ttnn._ttnn.operations.data_movement
    tensor = ttnn.from_torch(torch.zeros([1, 1, 32, 32], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    # Wrong shape for doubling H: the query must not silently approve the call.
    for operation in (api.repeat_codegen_support, ttnn.repeat):
        with expect_error(RuntimeError, "repeat optional output shape mismatch"):
            operation(tensor, [1, 1, 2, 1], optional_output_tensor=tensor)
