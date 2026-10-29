# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fixed TTNN workload for the NVFP4 agent evaluator; run with safe pytest --profile."""

import json
import math
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole

pytestmark = pytest.mark.skipif(not is_blackhole(), reason="NVFP4 performance workload requires Blackhole")


def test_nvfp4_l1_performance(device, tmp_path):
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(17)
    height, width = 2048, 4096
    codes = torch.randint(0, 16, (height, width), generator=generator, dtype=torch.int32)
    scales = torch.randint(0, 127, (height, width // 16), generator=generator, dtype=torch.int32)
    packed, blocks = ttnn.experimental.nvfp4.pack(codes, scales, experimental_fast=True)
    data = ttnn.from_torch(
        packed, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    scale_data = ttnn.from_torch(
        blocks, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    grid = device.compute_with_storage_grid_size()
    packed_tiles = packed.numel() // 1024
    cores = min(grid.x * grid.y, packed_tiles)
    calls = 8
    device.enable_program_cache()
    try:
        for _ in range(calls):
            output = ttnn.experimental.dequantize_nvfp4(
                data,
                scale_data,
                global_scale=0.8125,
                experimental_fast=True,
                fpu_tensor_scale=False,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.synchronize_device(device)
            assert tuple(output.shape) == (height, width), "Performance workload output shape changed"
            assert output.dtype == ttnn.bfloat16 and output.layout == ttnn.TILE_LAYOUT, "Output format changed"
            ttnn.deallocate(output)
    finally:
        ttnn.deallocate(data)
        ttnn.deallocate(scale_data)
    metadata = {
        "shape": [height, width],
        "elements": height * width,
        "calls": calls,
        "warmup_calls": 2,
        "cores": cores,
        "packed_tiles": packed_tiles,
        "output_tiles_per_packed_tile": 4,
        "maximum_output_tiles_per_core": math.ceil(packed_tiles / cores) * 4,
        "memory": "L1",
        "global_scale": 0.8125,
        "fp32_dest_acc_en": False,
        "fpu_tensor_scale": False,
    }
    output_path = Path(os.environ.get("NVFP4_BENCH_REPORT", tmp_path / "workload.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2) + "\n")
