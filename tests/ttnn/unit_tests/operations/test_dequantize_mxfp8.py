# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Blackhole MXFP8 (E4M3 + E8M0) decoder, including the packing contract."""

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.unit_tests.operations.mxfp8_reference import (
    CONFIGS,
    CONTAINERS,
    assert_exact,
    reference,
    run_decode,
)

requires_blackhole = pytest.mark.skipif(not is_blackhole(), reason="MXFP8 decoder requires Blackhole")


def _random_inputs(height, width, seed):
    generator = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 256, (height, width), generator=generator, dtype=torch.int32)
    codes = torch.where((codes & 0x7F) == 0x7F, codes - 1, codes)
    # Mostly moderate scales, with the extremes present.
    scales = torch.randint(100, 155, (height, width // 32), generator=generator, dtype=torch.int32)
    scales[0, 0], scales[-1, -1] = 0, 254
    return codes, scales


# ---------------------------------------------------------------- host-only contract


@pytest.mark.parametrize("container", list(CONTAINERS))
def test_mxfp8_pack_contract(container):
    word_dtype, planes = CONTAINERS[container]
    height, width = 64, 128 * planes
    codes = torch.arange(height * width).reshape(height, width) * 7 % 254
    codes = (codes + (codes >= 0x7F)).to(torch.int32)  # Skip the NaN codes 0x7F and 0xFF.
    scales = (torch.arange(height * width // 32).reshape(height, -1) % 255).to(torch.int32)
    packed, ordered = ttnn.experimental.mxfp8.pack(codes, scales, dtype=word_dtype)
    assert packed.dtype == torch.int32 and ordered.dtype == torch.int32
    assert tuple(packed.shape) == (height, width // planes)
    tile_rows, packed_cols = height // 32, width // (32 * planes)
    assert tuple(ordered.shape) == (tile_rows * packed_cols, 32 * planes)
    for tile_row in range(tile_rows):
        for packed_col in range(packed_cols):
            words = packed[tile_row * 32 : (tile_row + 1) * 32, packed_col * 32 : (packed_col + 1) * 32]
            for plane in range(planes):
                output_col = packed_col * planes + plane
                block = codes[tile_row * 32 : (tile_row + 1) * 32, output_col * 32 : (output_col + 1) * 32]
                torch.testing.assert_close((words >> (8 * plane)) & 0xFF, block, rtol=0, atol=0)
                row_scales = ordered[tile_row * packed_cols + packed_col, plane * 32 : (plane + 1) * 32]
                torch.testing.assert_close(
                    row_scales, scales[tile_row * 32 : (tile_row + 1) * 32, output_col], rtol=0, atol=0
                )
    if planes == 1:
        torch.testing.assert_close(packed, codes, rtol=0, atol=0)


def test_mxfp8_pack_validation(expect_error):
    codes = torch.zeros((32, 64), dtype=torch.int32)
    scales = torch.full((32, 2), 127, dtype=torch.int32)
    for nan in (0x7F, 0xFF):
        bad = codes.clone()
        bad[3, 5] = nan
        with expect_error(ValueError, "NaN"):
            ttnn.experimental.mxfp8.pack(bad, scales)
    with expect_error(ValueError, "E8M0"):
        ttnn.experimental.mxfp8.pack(codes, torch.full((32, 2), 255, dtype=torch.int32))
    with expect_error(ValueError, "shape"):
        ttnn.experimental.mxfp8.pack(codes, torch.zeros((32, 4), dtype=torch.int32))
    with expect_error(ValueError, "multiple"):
        ttnn.experimental.mxfp8.pack(torch.zeros((32, 32), dtype=torch.int32), torch.zeros((32, 1), dtype=torch.int32))
    with expect_error(ValueError, "container"):
        ttnn.experimental.mxfp8.pack(codes, scales, dtype=ttnn.uint32)


def test_mxfp8_reference_overflow_and_underflow():
    codes = torch.tensor([[0x7E] + [0x01] + [0x08] + [0] * 29], dtype=torch.int32)
    exact = reference(codes, torch.tensor([[254]], dtype=torch.int32)).flatten()
    assert exact[0] == 448 * 2.0**127 and exact[0] >= 2.0**128
    tiny = reference(codes, torch.tensor([[0]], dtype=torch.int32)).flatten()
    assert tiny[1] == 2.0**-136 and tiny[2] == 2.0**-133


# ---------------------------------------------------------------- device


@requires_blackhole
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("container,fp32_dest_acc_en,dtype", CONFIGS)
def test_mxfp8_layouts(device, container, input_layout, output_layout, fp32_dest_acc_en, dtype):
    # Several tile rows/columns catch tile-ID and face-order bugs in both layouts.
    codes, scales = _random_inputs(96, 256, seed=11)
    actual, _ = run_decode(
        device,
        codes,
        scales,
        container=container,
        fp32_dest_acc_en=fp32_dest_acc_en,
        dtype=dtype,
        input_layout=input_layout,
        output_layout=output_layout,
    )
    assert_exact(actual, codes, scales, dtype)


@requires_blackhole
@pytest.mark.parametrize("container", list(CONTAINERS))
@pytest.mark.parametrize("shape", [(2, 64, 128), (2, 3, 32, 128)])
def test_mxfp8_batched_rank(device, container, shape):
    height = 1
    for dim in shape[:-1]:
        height *= dim
    codes, scales = _random_inputs(height, shape[-1], seed=23)
    actual, _ = run_decode(device, codes, scales, container=container, shape=shape)
    assert_exact(actual.reshape(height, shape[-1]), codes, scales, "bfloat16")


@requires_blackhole
@pytest.mark.parametrize("container,fp32_dest_acc_en,dtype", CONFIGS)
def test_mxfp8_cache_and_l1(device, container, fp32_dest_acc_en, dtype):
    device.enable_program_cache()
    keep_alive = []
    for iteration in range(3):
        codes, scales = _random_inputs(32, 128, seed=113 + iteration)
        actual, tensors = run_decode(
            device,
            codes,
            scales,
            container=container,
            fp32_dest_acc_en=fp32_dest_acc_en,
            dtype=dtype,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        keep_alive.append(tensors)
        assert_exact(actual, codes, scales, dtype)
        if iteration == 0:
            entries = device.num_program_cache_entries()
            assert entries > 0, "Program cache was not populated"
        else:
            assert device.num_program_cache_entries() == entries, "Cache hit unexpectedly created a new program"


@requires_blackhole
@pytest.mark.parametrize("container", list(CONTAINERS))
@pytest.mark.parametrize("memory_config", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
def test_mxfp8_streaming(device, container, memory_config):
    # More packed tiles than worker cores exercises reuse/wrap of every FIFO.
    codes, scales = _random_inputs(1024, 1024, seed=143)
    actual, _ = run_decode(device, codes, scales, container=container, memory_config=memory_config)
    assert_exact(actual, codes, scales, "bfloat16")


@requires_blackhole
@pytest.mark.parametrize("container", list(CONTAINERS))
@pytest.mark.parametrize("memory_config", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
def test_mxfp8_many_tiles_per_core(device, container, memory_config):
    # Sixteen packed tiles on every core lets the compute UNPACK thread run ahead
    # of MATH and the reader refill consumed CB pages; catches kernels that read a
    # popped scale page. The streaming case alone (~4 tiles/core) can miss it.
    _, planes = CONTAINERS[container]
    grid = device.compute_with_storage_grid_size()
    packed_tiles = grid.x * grid.y * 16
    codes, scales = _random_inputs(packed_tiles // 8 * 32, 8 * 32 * planes, seed=211)
    actual, _ = run_decode(device, codes, scales, container=container, memory_config=memory_config)
    assert_exact(actual, codes, scales, "bfloat16")


@requires_blackhole
@pytest.mark.parametrize("device_params", [{"trace_region_size": 262144}], indirect=True)
def test_mxfp8_batched_trace(device):
    device.enable_program_cache()
    codes, scales = _random_inputs(64, 256, seed=5)
    packed, blocks = ttnn.experimental.mxfp8.pack(codes, scales)
    data = ttnn.from_torch(packed.reshape(2, 32, 128), dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device)
    block_data = ttnn.from_torch(blocks, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn.experimental.dequantize_mxfp8(data, block_data)
    ttnn.synchronize_device(device)
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    output = ttnn.experimental.dequantize_mxfp8(data, block_data)
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        for seed in [6, 7]:
            updated_codes, updated_scales = _random_inputs(64, 256, seed=seed)
            updated_packed, updated_blocks = ttnn.experimental.mxfp8.pack(updated_codes, updated_scales)
            host_data = ttnn.from_torch(updated_packed.reshape(2, 32, 128), dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT)
            host_blocks = ttnn.from_torch(updated_blocks, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy_host_to_device_tensor(host_data, data)
            ttnn.copy_host_to_device_tensor(host_blocks, block_data)
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            assert_exact(ttnn.to_torch(output).float().reshape(64, 256), updated_codes, updated_scales, "bfloat16")
    finally:
        ttnn.release_trace(device, trace)


@requires_blackhole
def test_mxfp8_matmul_consumer(device):
    codes, scales = _random_inputs(32, 128, seed=74)
    scales.fill_(127)
    actual, (_, _, output) = run_decode(device, codes, scales)
    identity = ttnn.from_torch(
        torch.eye(128, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    consumed = ttnn.matmul(
        output,
        identity,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False
        ),
    )
    torch.testing.assert_close(ttnn.to_torch(consumed).float(), actual, rtol=0, atol=0)


@requires_blackhole
def test_mxfp8_validation(device, expect_error):
    codes, scales = _random_inputs(32, 64, seed=1)
    packed, blocks = ttnn.experimental.mxfp8.pack(codes, scales)
    data = ttnn.from_torch(packed, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device)
    block_data = ttnn.from_torch(blocks, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "FLOAT32 output requires fp32_dest_acc_en"):
        ttnn.experimental.dequantize_mxfp8(data, block_data, dtype=ttnn.float32)
    byte_packed, byte_scales = ttnn.experimental.mxfp8.pack(codes, scales, dtype=ttnn.uint8)
    byte_data = ttnn.from_torch(byte_packed, dtype=ttnn.uint8, layout=ttnn.TILE_LAYOUT, device=device)
    byte_blocks = ttnn.from_torch(byte_scales, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "UINT8 packed data requires fp32_dest_acc_en"):
        ttnn.experimental.dequantize_mxfp8(byte_data, byte_blocks, fp32_dest_acc_en=False)
    with expect_error(RuntimeError, "output must be BFLOAT16 or FLOAT32"):
        ttnn.experimental.dequantize_mxfp8(data, block_data, dtype=ttnn.uint16)
    wrong_shape = ttnn.from_torch(
        torch.zeros((1, 32), dtype=torch.int32), dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    with expect_error(RuntimeError, "scales must have shape"):
        ttnn.experimental.dequantize_mxfp8(data, wrong_shape)
    wrong_dtype = ttnn.from_torch(blocks, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "row-major UINT8 E8M0"):
        ttnn.experimental.dequantize_mxfp8(data, wrong_dtype)
    float_data = ttnn.from_torch(packed.float(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "UINT8 or UINT16"):
        ttnn.experimental.dequantize_mxfp8(float_data, block_data)
