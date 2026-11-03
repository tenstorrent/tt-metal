# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""L1-capacity-sized MXFP8 workload in the primary configuration; run with safe pytest --profile."""

import json
import math
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.unit_tests.operations.mxfp8_reference import (
    CONTAINERS,
    OUTPUT_DTYPES,
    PRIMARY,
    decode_e4m3,
    decode_e8m0,
)

pytestmark = pytest.mark.skipif(not is_blackhole(), reason="MXFP8 performance workload requires Blackhole")

# The program factory's circular buffers use about 33 KiB/core for UINT16 input;
# leave 64 KiB beneath tensor allocations for them, alignment, and runtime headroom.
RESERVE_BYTES = 64 * 1024


def _l1_workload(device, planes, output_bytes):
    grid = device.compute_with_storage_grid_size()
    cores = grid.x * grid.y
    view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    available = min(view.total_bytes_free_per_bank, view.largest_contiguous_bytes_free_per_bank)
    data_bytes = 1024 * planes
    scale_bytes = 32 * planes
    output_tile_bytes = 1024 * output_bytes
    bytes_per_packed_tile = data_bytes + scale_bytes + planes * output_tile_bytes
    capacity = (available - RESERVE_BYTES) * view.num_banks // (cores * bytes_per_packed_tile)
    assert capacity > 0, "Insufficient free L1 for a balanced MXFP8 workload"
    requested = os.environ.get("MXFP8_BENCH_PACKED_TILES_PER_CORE")
    per_core = int(requested) if requested is not None else (capacity // 16 * 16 or capacity)
    assert 0 < per_core <= capacity, f"Requested {per_core} packed tiles/core exceeds L1 budget {capacity}"
    packed_tiles = cores * per_core
    packed_columns = math.gcd(packed_tiles, 32)
    height, width = packed_tiles // packed_columns * 32, packed_columns * 32 * planes
    tensor_bytes_per_bank = (
        math.ceil(packed_tiles / view.num_banks) * (data_bytes + scale_bytes)
        + math.ceil(planes * packed_tiles / view.num_banks) * output_tile_bytes
    )
    assert tensor_bytes_per_bank + RESERVE_BYTES <= available, "MXFP8 workload exceeds the reserved L1 budget"
    return dict(
        shape=[height, width],
        workload_mode="balanced_fixed" if requested is not None else "capacity_fill",
        packed_tiles_per_core=per_core,
        l1_num_banks=view.num_banks,
        l1_total_bytes_per_bank=view.total_bytes_per_bank,
        l1_initial_free_bytes_per_bank=view.total_bytes_free_per_bank,
        l1_initial_largest_free_bytes_per_bank=view.largest_contiguous_bytes_free_per_bank,
        l1_reserved_headroom_bytes=RESERVE_BYTES,
        l1_expected_tensor_bytes_per_bank=tensor_bytes_per_bank,
    )


def _check_output_samples(output, codes, scales):
    # Sample across the full matrix, including its endpoints, after timing.
    actual = ttnn.to_torch(output).float()
    flat = torch.linspace(0, codes.numel() - 1, 512, dtype=torch.int64)
    rows, columns = flat // codes.shape[1], flat % codes.shape[1]
    expected = decode_e4m3(codes[rows, columns]) * decode_e8m0(scales[rows, columns // 32])
    torch.testing.assert_close(actual[rows, columns].double(), expected, rtol=0, atol=0)


def test_mxfp8_l1_performance(device, tmp_path):
    torch.set_num_threads(4)
    word_dtype, planes = CONTAINERS[PRIMARY["container"]]
    output_dtype = OUTPUT_DTYPES[PRIMARY["dtype"]]
    output_bytes = 2 if output_dtype == ttnn.bfloat16 else 4
    workload = _l1_workload(device, planes, output_bytes)
    height, width = workload["shape"]
    generator = torch.Generator().manual_seed(17)
    codes = torch.randint(0, 256, (height, width), generator=generator, dtype=torch.int32)
    codes = torch.where((codes & 0x7F) == 0x7F, codes - 1, codes)
    # Scales within +-20 of 1.0 keep every sampled product a finite normal value.
    scales = torch.randint(107, 148, (height, width // 32), generator=generator, dtype=torch.int32)
    packed, blocks = ttnn.experimental.mxfp8.pack(codes, scales, dtype=word_dtype)
    grid = device.compute_with_storage_grid_size()
    packed_tiles = packed.numel() // 1024
    cores = min(grid.x * grid.y, packed_tiles)
    calls = 8
    device.enable_program_cache()
    peak = None
    data = scale_data = output = None
    try:
        data = ttnn.from_torch(
            packed, dtype=word_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        scale_data = ttnn.from_torch(
            blocks, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        for call in range(calls):
            output = ttnn.experimental.dequantize_mxfp8(
                data,
                scale_data,
                dtype=output_dtype,
                fp32_dest_acc_en=PRIMARY["fp32_dest_acc_en"],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.synchronize_device(device)
            assert tuple(output.shape) == (height, width), "Performance workload output shape changed"
            assert output.dtype == output_dtype and output.layout == ttnn.TILE_LAYOUT, "Output format changed"
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
        for tensor in (output, data, scale_data):
            if tensor is not None:
                ttnn.deallocate(tensor)
    metadata = {
        **workload,
        **peak,
        "elements": height * width,
        "calls": calls,
        "warmup_calls": 2,
        "cores": cores,
        "packed_tiles": packed_tiles,
        "output_tiles_per_packed_tile": planes,
        "maximum_output_tiles_per_core": math.ceil(packed_tiles / cores) * planes,
        "memory": "L1",
        **{f"primary_{key}": value for key, value in PRIMARY.items()},
    }
    output_path = Path(os.environ.get("MXFP8_BENCH_REPORT", tmp_path / "workload.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2) + "\n")
