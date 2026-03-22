# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.common.utility_functions import skip_for_blackhole

mem_configs = [
    ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
]


@pytest.mark.parametrize(
    "input_shapes",
    [
        [[1, 1, 32, 32]],
    ],
)
@pytest.mark.parametrize(
    "dst_mem_config",
    mem_configs,
)
class TestSignbit:
    def test_run_signbit_op(
        self,
        input_shapes,
        dst_mem_config,
        device,
    ):
        shape = input_shapes[0]
        torch_input = (torch.rand(shape) * 200 - 100).to(torch.bfloat16)

        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tt_output = ttnn.signbit(tt_input, memory_config=dst_mem_config)
        torch_output = ttnn.to_torch(tt_output)

        assert torch.equal(torch.signbit(torch_input), torch_output.to(torch.bool))

    def test_run_signbit_negative_zero(
        self,
        input_shapes,
        dst_mem_config,
        device,
    ):
        shape = input_shapes[0]
        torch_input = torch.full(shape, -0.0, dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tt_output = ttnn.signbit(tt_input, memory_config=dst_mem_config)
        torch_output = ttnn.to_torch(tt_output)

        assert torch.equal(torch.signbit(torch_input), torch_output.to(torch.bool))
