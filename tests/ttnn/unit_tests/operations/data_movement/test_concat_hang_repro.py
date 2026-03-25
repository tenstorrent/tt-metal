# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import ttnn


def _make_repeat_lowering_concat_inputs(input_tensor, repeat_count):
    return [input_tensor] * repeat_count


def test_repeat_lowering_concat_exact_ir_repro(device):
    """
    Manual repro that mirrors the provided IR exactly:
    1xbf16 DRAM row-major -> to_layout(tile) -> concat 100 tiled inputs on dim 0.

    ROOT CAUSE IDENTIFIED (2026-03-25):
    - Hangs with ETH dispatch at 48+ inputs, passes with WORKER dispatch
    - N300 clusters default to ETH dispatch via CreateDevice()/pytest fixtures
    - Threshold: 47 inputs passes, 48+ hangs with ETH dispatch
    - WORKER dispatch works at all tested input counts (up to 100+)
    """

    input_shape = (1,)
    repeat_count = 100  # Hangs at 48+ with ETH dispatch on N300

    torch_input_tensor = torch.zeros(input_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.concat([torch_input_tensor] * repeat_count, dim=0)

    input_tensor = None
    tiled_tensor = None
    output_tensor = None

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tiled_tensor = ttnn.to_layout(
        input_tensor,
        ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(input_tensor)
    input_tensor = None

    print(f"running concat")

    output_tensor = ttnn.concat(
        _make_repeat_lowering_concat_inputs(tiled_tensor, repeat_count),
        dim=0,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(tiled_tensor)
    tiled_tensor = None
    print("output_tensor", output_tensor)
    output_host = ttnn.to_torch(output_tensor)
    print("output_host", output_host)
    print("torch_output_tensor", torch_output_tensor)
    assert torch.equal(output_host, torch_output_tensor)
