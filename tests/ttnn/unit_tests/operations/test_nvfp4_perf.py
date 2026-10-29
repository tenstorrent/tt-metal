# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""L1-capacity-sized NVFP4 workload; run with safe pytest --profile."""

import json
import math
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole

pytestmark = pytest.mark.skipif(not is_blackhole(), reason="NVFP4 performance workload requires Blackhole")


def _l1_workload(device):
    grid = device.compute_with_storage_grid_size()
    cores = grid.x * grid.y
    view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    available = min(view.total_bytes_free_per_bank, view.largest_contiguous_bytes_free_per_bank)
    # Per packed tile: 2048 B data, 512 B scales, and four 2048 B outputs.
    # The current SFPU-scale kernel uses 25,664 B of CBs; leave 64 KiB for
    # those buffers, alignment, and runtime headroom beneath tensor allocations.
    reserve = 64 * 1024
    bytes_per_packed_tile = 2048 + 512 + 4 * 2048
    capacity = (available - reserve) * view.num_banks // (cores * bytes_per_packed_tile)
    assert capacity > 0, "Insufficient free L1 for a balanced NVFP4 workload"
    requested = os.environ.get("NVFP4_BENCH_PACKED_TILES_PER_CORE")
    if os.environ.get("NVFP4_BENCH_LEGACY", "0") == "1":
        height, width = 2048, 4096
        mode = "legacy"
    else:
        per_core = int(requested) if requested is not None else (capacity // 16 * 16 or capacity)
        assert 0 < per_core <= capacity, f"Requested {per_core} packed tiles/core exceeds L1 budget {capacity}"
        packed_tiles = cores * per_core
        packed_columns = math.gcd(packed_tiles, 32)
        height, width = packed_tiles // packed_columns * 32, packed_columns * 128
        mode = "balanced_fixed" if requested is not None else "capacity_fill"
    packed_tiles = height * width // 4096
    tensor_bytes_per_bank = (
        math.ceil(packed_tiles / view.num_banks) * (2048 + 512) + math.ceil(4 * packed_tiles / view.num_banks) * 2048
    )
    assert tensor_bytes_per_bank + reserve <= available, "NVFP4 workload exceeds the reserved L1 budget"
    return dict(
        shape=[height, width],
        workload_mode=mode,
        packed_tiles_per_core_min=packed_tiles // cores,
        packed_tiles_per_core_max=math.ceil(packed_tiles / cores),
        l1_num_banks=view.num_banks,
        l1_total_bytes_per_bank=view.total_bytes_per_bank,
        l1_initial_free_bytes_per_bank=view.total_bytes_free_per_bank,
        l1_initial_largest_free_bytes_per_bank=view.largest_contiguous_bytes_free_per_bank,
        l1_reserved_headroom_bytes=reserve,
        l1_expected_tensor_bytes_per_bank=tensor_bytes_per_bank,
    )


def _check_output_samples(output, codes, scales):
    # Sample across the full matrix, including its endpoints, after timing.
    # Download is outside the converter and allocates no additional L1 tensor.
    actual = ttnn.to_torch(output)
    flat = torch.linspace(0, codes.numel() - 1, 512, dtype=torch.int64)
    rows, columns = flat // codes.shape[1], flat % codes.shape[1]
    selected = codes[rows, columns]
    magnitude = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)[selected & 7]
    values = magnitude * torch.where((selected & 8) != 0, -1.0, 1.0)
    block = scales[rows, columns // 16].long()
    exponent, mantissa = block >> 3, block & 7
    decoded = torch.where(
        exponent == 0, mantissa.double() * 2**-9, torch.ldexp(1 + mantissa.double() / 8, exponent - 7)
    )
    reference = (values * decoded * 0.8125).bfloat16().float()
    torch.testing.assert_close(actual[rows, columns].float(), reference, rtol=1 / 128, atol=0)


def test_nvfp4_l1_performance(device, tmp_path):
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(17)
    workload = _l1_workload(device)
    height, width = workload["shape"]
    codes = torch.randint(0, 16, (height, width), generator=generator, dtype=torch.int32)
    scales = torch.randint(0, 127, (height, width // 16), generator=generator, dtype=torch.int32)
    packed, blocks = ttnn.experimental.nvfp4.pack(codes, scales, experimental_fast=True)
    grid = device.compute_with_storage_grid_size()
    packed_tiles = packed.numel() // 1024
    cores = min(grid.x * grid.y, packed_tiles)
    calls = 8
    device.enable_program_cache()
    peak = None
    data = scale_data = output = None
    try:
        data = ttnn.from_torch(
            packed, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        scale_data = ttnn.from_torch(
            blocks,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        for call in range(calls):
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
            if call == 0:
                view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
                peak = dict(
                    l1_peak_tensor_bytes_per_bank=view.total_bytes_allocated_per_bank,
                    l1_free_bytes_per_bank_with_output=view.total_bytes_free_per_bank,
                    l1_largest_free_bytes_per_bank_with_output=view.largest_contiguous_bytes_free_per_bank,
                )
            if call == calls - 1:
                _check_output_samples(output, codes, scales)
            ttnn.deallocate(output)
            output = None
    finally:
        if output is not None:
            ttnn.deallocate(output)
        if data is not None:
            ttnn.deallocate(data)
        if scale_data is not None:
            ttnn.deallocate(scale_data)
    metadata = {
        **workload,
        **peak,
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
