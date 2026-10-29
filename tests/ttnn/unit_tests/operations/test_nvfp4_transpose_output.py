# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Full-matrix transposed NVFP4 output and a native RHS-transpose consumer."""

import math

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole

pytestmark = pytest.mark.skipif(not is_blackhole(), reason="NVFP4 requires Blackhole")


def prepare(device, shape, input_layout, memory_config=ttnn.DRAM_MEMORY_CONFIG, seed=312):
    torch.manual_seed(seed)
    batch = math.prod(shape[:-2])
    height, width = shape[-2:]
    codes = torch.randint(0, 16, (batch * height, width), dtype=torch.int32)
    scales = (torch.arange(batch * height * width // 16).reshape(batch * height, width // 16) % 127).to(torch.uint8)
    packed, blocks = ttnn.experimental.nvfp4.pack(codes, scales, experimental_fast=True)
    packed = packed.reshape(*shape[:-2], height, width // 4)
    data = ttnn.from_torch(
        packed,
        dtype=ttnn.uint16,
        layout=input_layout,
        device=device,
        memory_config=memory_config,
    )
    scale_data = ttnn.from_torch(
        blocks,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    return data, scale_data


@pytest.mark.parametrize("shape", [(96, 256), (2, 96, 256), (2, 3, 96, 256)])
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("fpu_tensor_scale", [False, True])
def test_nvfp4_transpose_output(device, shape, input_layout, fpu_tensor_scale):
    # K=96 intentionally cannot be packed as the width of W.T by the old helper.
    data, scales = prepare(device, shape, input_layout)
    kwargs = dict(global_scale=0.8125, experimental_fast=True, fpu_tensor_scale=fpu_tensor_scale)
    normal = ttnn.experimental.dequantize_nvfp4(data, scales, **kwargs)
    transposed = ttnn.experimental.dequantize_nvfp4(data, scales, transpose_output=True, **kwargs)
    expected = ttnn.to_torch(normal).transpose(-1, -2).contiguous()
    torch.testing.assert_close(ttnn.to_torch(transposed), expected, rtol=0, atol=0)
    assert tuple(transposed.shape) == (*shape[:-2], shape[-1], shape[-2])
    assert transposed.layout == ttnn.TILE_LAYOUT
    assert transposed.dtype == ttnn.bfloat16


def test_nvfp4_transpose_output_cache(device):
    device.enable_program_cache()
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    entry_counts = {}
    keep_alive = []
    for iteration in range(3):
        data, scales = prepare(device, (2, 96, 256), ttnn.TILE_LAYOUT, seed=83 + iteration)
        outputs = {}
        for transposed in [False, True]:
            outputs[transposed] = ttnn.experimental.dequantize_nvfp4(
                data,
                scales,
                global_scale=0.8125,
                experimental_fast=True,
                fpu_tensor_scale=False,
                transpose_output=transposed,
            )
            entries = device.num_program_cache_entries()
            if iteration == 0:
                entry_counts[transposed] = entries
            else:
                assert entries == entry_counts[True], "Cache hit unexpectedly created another program"
        if iteration == 0:
            assert entry_counts[True] > entry_counts[False], "Transpose flag must distinguish cached programs"
        torch.testing.assert_close(
            ttnn.to_torch(outputs[True]),
            ttnn.to_torch(outputs[False]).transpose(-1, -2).contiguous(),
            rtol=0,
            atol=0,
        )
        keep_alive.append((data, scales, outputs))


@pytest.mark.parametrize("memory_config", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("matmul_fp32_dest", [False, True])
def test_nvfp4_transpose_output_matmul(device, memory_config, matmul_fp32_dest):
    data, scales = prepare(device, (96, 256), ttnn.TILE_LAYOUT, memory_config)
    common = dict(
        global_scale=0.8125,
        experimental_fast=True,
        fpu_tensor_scale=False,
        memory_config=memory_config,
    )
    normal = ttnn.experimental.dequantize_nvfp4(data, scales, **common)
    transposed = ttnn.experimental.dequantize_nvfp4(data, scales, transpose_output=True, **common)
    torch.manual_seed(51)
    x = torch.randn((64, 96), dtype=torch.bfloat16)
    input_x = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=2,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=matmul_fp32_dest,
    )
    kwargs = dict(
        program_config=config,
        compute_kernel_config=kernel_config,
        dtype=ttnn.bfloat16,
        memory_config=memory_config,
    )
    expected = ttnn.matmul(input_x, normal, **kwargs)
    actual = ttnn.matmul(input_x, transposed, transpose_b=True, **kwargs)
    torch.testing.assert_close(ttnn.to_torch(actual), ttnn.to_torch(expected), rtol=0, atol=0)


def test_nvfp4_transpose_output_validation(device, expect_error):
    data, scales = prepare(device, (96, 256), ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "transpose_output requires"):
        ttnn.experimental.dequantize_nvfp4(
            data,
            scales,
            experimental_fast=True,
            transpose_output=True,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
        )
    codes = torch.zeros((96, 256), dtype=torch.int32)
    scale_codes = torch.zeros((96, 16), dtype=torch.uint8)
    precise_data, precise_scales = ttnn.experimental.nvfp4.pack(codes, scale_codes)
    precise_data = ttnn.from_torch(precise_data, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device)
    precise_scales = ttnn.from_torch(precise_scales, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "transpose_output requires"):
        ttnn.experimental.dequantize_nvfp4(precise_data, precise_scales, transpose_output=True)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 262144}], indirect=True)
@pytest.mark.parametrize("fpu_tensor_scale", [False, True])
def test_nvfp4_transpose_output_trace(device, fpu_tensor_scale):
    device.enable_program_cache()
    data, scales = prepare(device, (2, 96, 256), ttnn.TILE_LAYOUT)
    kwargs = dict(global_scale=0.8125, experimental_fast=True, fpu_tensor_scale=fpu_tensor_scale)
    normal = ttnn.experimental.dequantize_nvfp4(data, scales, **kwargs)
    warmup = ttnn.experimental.dequantize_nvfp4(data, scales, transpose_output=True, **kwargs)
    ttnn.synchronize_device(device)
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    transposed = ttnn.experimental.dequantize_nvfp4(data, scales, transpose_output=True, **kwargs)
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        expected = ttnn.to_torch(normal).transpose(-1, -2).contiguous()
        for _ in range(3):
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            torch.testing.assert_close(ttnn.to_torch(transposed), expected, rtol=0, atol=0)
    finally:
        ttnn.release_trace(device, trace)
    assert tuple(warmup.shape) == (2, 256, 96)


@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_nvfp4_every_packed_uint16_word(device, input_layout):
    torch.set_num_threads(4)
    # Invert the offline fast packing so every possible UINT16 word reaches
    # the decoder unchanged, exercising all combinations of adjacent nibbles.
    words = torch.arange(65536, dtype=torch.int32).reshape(256, 256)
    tiles = words.reshape(8, 32, 8, 32).permute(0, 2, 1, 3)
    codes = torch.stack([(tiles >> shift) & 15 for shift in [0, 4, 8, 12]], dim=2).transpose(-2, -1)
    codes = codes.permute(0, 3, 1, 2, 4).reshape(256, 1024)
    scales = torch.full((256, 64), 0x38, dtype=torch.uint8)  # E4M3 code for 1.0.
    packed, blocks = ttnn.experimental.nvfp4.pack(codes, scales, experimental_fast=True)
    torch.testing.assert_close(packed, words, rtol=0, atol=0)
    data = ttnn.from_torch(packed, dtype=ttnn.uint16, layout=input_layout, device=device)
    scale_data = ttnn.from_torch(blocks, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    magnitudes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])[codes & 7]
    expected = torch.where((codes & 8) != 0, -magnitudes, magnitudes).bfloat16()
    for transposed in [False, True]:
        actual = ttnn.experimental.dequantize_nvfp4(
            data,
            scale_data,
            global_scale=1.0,
            experimental_fast=True,
            fpu_tensor_scale=False,
            transpose_output=transposed,
        )
        reference = expected.T.contiguous() if transposed else expected
        torch.testing.assert_close(ttnn.to_torch(actual), reference, rtol=0, atol=0)
