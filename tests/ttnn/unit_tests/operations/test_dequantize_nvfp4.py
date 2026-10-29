# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Blackhole NVFP4 decoder, including the tile-plane packing contract."""

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.unit_tests.operations.test_nvfp4_accuracy import _accuracy_gate_failures, _metrics, _reference

pytestmark = pytest.mark.skipif(not is_blackhole(), reason="NVFP4 SFPU decoder requires Blackhole")


def reference(codes, scales, global_scale, e2m1_scales=False):
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    values = lut[codes.long() & 7]
    values = torch.where((codes & 8) != 0, -values, values)
    if e2m1_scales:
        decoded_scales = lut[scales.long()]
    else:
        # Independent IEEE E4M3FN decoder, including exponent-zero subnormals.
        exponent = (scales.long() >> 3) & 15
        mantissa = scales.long() & 7
        decoded_scales = torch.where(
            exponent == 0,
            mantissa.float() * 2**-9,
            torch.ldexp(1 + mantissa.float() / 8, exponent.int() - 7),
        )
    return values * decoded_scales.repeat_interleave(16, dim=-1) * global_scale


def run_decode(
    device,
    codes,
    scales,
    word_dtype,
    input_layout,
    output_layout,
    fp32_dest,
    output_dtype,
    e2m1=False,
    global_scale=1.0,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    packed, ordered_scales = ttnn.experimental.nvfp4.pack(codes, scales, dtype=word_dtype)
    data = ttnn.from_torch(packed, dtype=word_dtype, layout=input_layout, device=device, memory_config=memory_config)
    blocks = ttnn.from_torch(
        ordered_scales.to(torch.int32),
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    output = ttnn.experimental.dequantize_nvfp4(
        data,
        blocks,
        global_scale=global_scale,
        scale_format=ttnn.experimental.Nvfp4ScaleFormat.E2M1 if e2m1 else ttnn.experimental.Nvfp4ScaleFormat.E4M3,
        dtype=output_dtype,
        output_layout=output_layout,
        fp32_dest_acc_en=fp32_dest,
        memory_config=memory_config,
    )
    result = ttnn.to_torch(output).float()
    expected = reference(codes, scales, global_scale, e2m1)
    if output_dtype == ttnn.bfloat16:
        expected = expected.bfloat16().float()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert tuple(output.shape) == tuple(codes.shape)
    assert output.layout == output_layout
    assert output.dtype == output_dtype
    return data, blocks, output


@pytest.mark.parametrize("word_dtype", [ttnn.uint8, ttnn.uint16])
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize(
    "fp32_dest,output_dtype", [(False, ttnn.bfloat16), (True, ttnn.bfloat16), (True, ttnn.float32)]
)
def test_nvfp4_all_encodings(device, word_dtype, input_layout, output_layout, fp32_dest, output_dtype):
    if word_dtype == ttnn.uint8 and not fp32_dest:
        pytest.skip("UINT8 integer unpack requires FP32 DEST; UINT16 covers BF16 DEST")
    # Every FP4 sign/magnitude combination, every valid E4M3FN scale, and distinct
    # adjacent rows/faces/planes. Multiple packed tile columns catch tile-ID bugs.
    width = 256 if word_dtype == ttnn.uint8 else 512
    indices = torch.arange(64 * width).reshape(64, width)
    codes = ((indices * 7 + indices // width * 3 + indices // 32) % 16).to(torch.int32)
    scales = (torch.arange(64 * (width // 16)).reshape(64, -1) % 127).to(torch.uint8)
    run_decode(
        device,
        codes,
        scales,
        word_dtype,
        input_layout,
        output_layout,
        fp32_dest,
        output_dtype,
        global_scale=0.7314453125,
    )


@pytest.mark.parametrize("word_dtype", [ttnn.uint8, ttnn.uint16])
@pytest.mark.parametrize("e2m1", [False, True])
@pytest.mark.parametrize("fp32_dest", [False, True])
def test_nvfp4_cache_and_l1(device, word_dtype, e2m1, fp32_dest):
    if word_dtype == ttnn.uint8 and not fp32_dest:
        pytest.skip("UINT8 integer unpack requires FP32 DEST; UINT16 covers BF16 DEST")
    width = 64 if word_dtype == ttnn.uint8 else 128
    device.enable_program_cache()
    keep_alive = []
    for iteration in range(3):
        torch.manual_seed(113 + iteration)
        codes = torch.randint(0, 16, (32, width), dtype=torch.int32)
        scales = torch.randint(0, 8 if e2m1 else 127, (32, width // 16), dtype=torch.uint8)
        keep_alive.append(
            run_decode(
                device,
                codes,
                scales,
                word_dtype,
                ttnn.TILE_LAYOUT,
                ttnn.TILE_LAYOUT,
                fp32_dest,
                ttnn.bfloat16,
                e2m1,
                global_scale=0.91357421875,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        )
        if iteration == 0:
            entries = device.num_program_cache_entries()
            assert entries > 0, "Program cache was not populated"
        else:
            assert device.num_program_cache_entries() == entries, "Cache hit unexpectedly created a new program"


@pytest.mark.parametrize("global_scale", [0.0, 2**-40, 2**40])
def test_nvfp4_global_scale(device, global_scale):
    codes = (torch.arange(32 * 64).reshape(32, 64) % 16).to(torch.int32)
    scales = torch.full((32, 4), 126, dtype=torch.uint8)
    run_decode(
        device,
        codes,
        scales,
        ttnn.uint8,
        ttnn.TILE_LAYOUT,
        ttnn.TILE_LAYOUT,
        True,
        ttnn.float32,
        global_scale=global_scale,
    )


def test_nvfp4_validation(device, expect_error):
    data = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.int32), dtype=ttnn.uint8, layout=ttnn.TILE_LAYOUT, device=device
    )
    scales = ttnn.from_torch(torch.zeros((1, 128), dtype=torch.int32), dtype=ttnn.uint8, device=device)
    with expect_error(RuntimeError, "UINT8 unpack requires"):
        ttnn.experimental.dequantize_nvfp4(data, scales, dtype=ttnn.float32)
    with expect_error(RuntimeError, "global_scale"):
        ttnn.experimental.dequantize_nvfp4(data, scales, global_scale=float("nan"), fp32_dest_acc_en=True)
    wrong_scales = ttnn.from_torch(torch.zeros((1, 64), dtype=torch.int32), dtype=ttnn.uint8, device=device)
    with expect_error(RuntimeError, "scales must have shape"):
        ttnn.experimental.dequantize_nvfp4(data, wrong_scales)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 262144}], indirect=True)
@pytest.mark.parametrize("experimental_fast,fpu_tensor_scale", [(False, True), (True, True), (True, False)])
def test_nvfp4_batched_trace(device, experimental_fast, fpu_tensor_scale):
    device.enable_program_cache()
    codes = (torch.arange(64 * 128).reshape(64, 128) % 16).to(torch.int32)
    scales = (torch.arange(64 * 8).reshape(64, 8) % 127).to(torch.uint8)
    packed, blocks = ttnn.experimental.nvfp4.pack(codes, scales, experimental_fast=experimental_fast)
    data = ttnn.from_torch(packed.reshape(2, 32, 32), dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device)
    block_dtype = ttnn.bfloat16 if experimental_fast else ttnn.uint8
    block_data = ttnn.from_torch(blocks, dtype=block_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    warmup = ttnn.experimental.dequantize_nvfp4(
        data, block_data, global_scale=0.8125, experimental_fast=experimental_fast, fpu_tensor_scale=fpu_tensor_scale
    )
    ttnn.synchronize_device(device)
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    output = ttnn.experimental.dequantize_nvfp4(
        data, block_data, global_scale=0.8125, experimental_fast=experimental_fast, fpu_tensor_scale=fpu_tensor_scale
    )
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        for multiplier in [1, 3]:
            updated_codes = (codes * multiplier) % 16
            updated_scales = ((scales.int() + multiplier) % 127).to(torch.uint8)
            updated_packed, updated_blocks = ttnn.experimental.nvfp4.pack(
                updated_codes, updated_scales, experimental_fast=experimental_fast
            )
            host_data = ttnn.from_torch(updated_packed.reshape(2, 32, 32), dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT)
            host_blocks = ttnn.from_torch(updated_blocks, dtype=block_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy_host_to_device_tensor(host_data, data)
            ttnn.copy_host_to_device_tensor(host_blocks, block_data)
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            expected = reference(updated_codes, updated_scales, 0.8125).bfloat16().reshape(2, 32, 128)
            torch.testing.assert_close(
                ttnn.to_torch(output).float(), expected.float(), rtol=1 / 128 if experimental_fast else 0, atol=0
            )
    finally:
        ttnn.release_trace(device, trace)


@pytest.mark.parametrize("fp32_dest", [False, True])
@pytest.mark.parametrize("global_scale", [1.00390625, 1.01171875])
def test_nvfp4_bf16_rounding_ties(device, fp32_dest, global_scale):
    codes = (torch.arange(32 * 128).reshape(32, 128) % 16).to(torch.int32)
    scales = torch.full((32, 8), 0x38, dtype=torch.uint8)
    run_decode(
        device,
        codes,
        scales,
        ttnn.uint16,
        ttnn.TILE_LAYOUT,
        ttnn.TILE_LAYOUT,
        fp32_dest,
        ttnn.bfloat16,
        global_scale=global_scale,
    )


@pytest.mark.parametrize("fp32_dest", [False, True])
def test_nvfp4_all_uint16_words(device, fp32_dest):
    # Include opaque words with a zero low byte and arbitrary high bits. Moving
    # these through the floating-point path could flush or change their bits.
    words = torch.arange(65536, dtype=torch.int32).reshape(256, 256)
    tiles = words.reshape(8, 32, 8, 32).permute(0, 2, 1, 3)
    codes = torch.stack([(tiles >> shift) & 15 for shift in [0, 4, 8, 12]], dim=2)
    codes = codes.permute(0, 3, 1, 2, 4).reshape(256, 1024)
    scales = torch.full((256, 64), 0x38, dtype=torch.uint8)
    packed, _ = ttnn.experimental.nvfp4.pack(codes, scales)
    torch.testing.assert_close(packed, words, rtol=0, atol=0)
    run_decode(
        device,
        codes,
        scales,
        ttnn.uint16,
        ttnn.TILE_LAYOUT,
        ttnn.TILE_LAYOUT,
        fp32_dest,
        ttnn.float32 if fp32_dest else ttnn.bfloat16,
    )


def run_fast_decode(
    device,
    codes,
    scales,
    input_layout=ttnn.TILE_LAYOUT,
    output_layout=ttnn.TILE_LAYOUT,
    e2m1=False,
    global_scale=1.0,
    fpu_tensor_scale=True,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    scale_format = ttnn.experimental.Nvfp4ScaleFormat.E2M1 if e2m1 else ttnn.experimental.Nvfp4ScaleFormat.E4M3
    packed, blocks = ttnn.experimental.nvfp4.pack(codes, scales, experimental_fast=True, scale_format=scale_format)
    data = ttnn.from_torch(packed, dtype=ttnn.uint16, layout=input_layout, device=device, memory_config=memory_config)
    scale_data = ttnn.from_torch(
        blocks, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=memory_config
    )
    output = ttnn.experimental.dequantize_nvfp4(
        data,
        scale_data,
        global_scale=global_scale,
        scale_format=scale_format,
        output_layout=output_layout,
        experimental_fast=True,
        fpu_tensor_scale=fpu_tensor_scale,
        memory_config=memory_config,
    )
    actual = ttnn.to_torch(output).float()
    expected = reference(codes, scales, 1.0, e2m1).bfloat16().float() * float(torch.tensor(global_scale).bfloat16())
    if fpu_tensor_scale:
        # BF16 DEST FPU multiplication rounds during its fidelity phases.
        torch.testing.assert_close(actual, expected, rtol=1 / 128, atol=0)
    else:
        # Accept any SFPU sequence meeting the independent numerical contract.
        reference_fp64 = (
            _reference(codes, scales, global_scale, flush_subnormals=True)
            if not e2m1
            else reference(codes, scales, 1.0, True).double() * float(torch.tensor(global_scale, dtype=torch.float32))
        )
        failures = _accuracy_gate_failures(_metrics(actual, reference_fp64))
        assert not failures, "SFPU reconstruction accuracy failed: " + "; ".join(failures)
    return data, scale_data, output


@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("e2m1", [False, True])
@pytest.mark.parametrize("global_scale", [1.0, 0.7314453125, 1.00390625, 1.0234375])
@pytest.mark.parametrize("fpu_tensor_scale", [False, True])
def test_nvfp4_fast(device, input_layout, output_layout, e2m1, global_scale, fpu_tensor_scale):
    torch.manual_seed(412)
    codes = torch.randint(0, 16, (64, 512), dtype=torch.int32)
    scales = (torch.arange(64 * 32).reshape(64, 32) % (8 if e2m1 else 127)).to(torch.uint8)
    run_fast_decode(device, codes, scales, input_layout, output_layout, e2m1, global_scale, fpu_tensor_scale)


@pytest.mark.parametrize("fpu_tensor_scale", [False, True])
def test_nvfp4_fast_cache_and_matmul(device, fpu_tensor_scale):
    device.enable_program_cache()
    keep_alive = []
    for iteration in range(3):
        torch.manual_seed(74 + iteration)
        codes = torch.randint(0, 16, (32, 128), dtype=torch.int32)
        scales = torch.randint(0, 127, (32, 8), dtype=torch.int32)
        inputs = run_fast_decode(
            device,
            codes,
            scales,
            global_scale=0.7314453125,
            fpu_tensor_scale=fpu_tensor_scale,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        keep_alive.append(inputs)
        if iteration == 0:
            entries = device.num_program_cache_entries()
            assert entries > 0, "Program cache was not populated"
        else:
            assert device.num_program_cache_entries() == entries, "Cache hit unexpectedly created a new program"
    output = keep_alive[-1][-1]
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
    torch.testing.assert_close(ttnn.to_torch(consumed), ttnn.to_torch(output), rtol=0, atol=0)


def test_nvfp4_fast_validation(device, expect_error):
    packed, blocks = ttnn.experimental.nvfp4.pack(
        torch.zeros((32, 128), dtype=torch.int32), torch.zeros((32, 8), dtype=torch.int32), experimental_fast=True
    )
    data = ttnn.from_torch(packed, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device)
    block_data = ttnn.from_torch(blocks, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "experimental_fast requires"):
        ttnn.experimental.dequantize_nvfp4(data, block_data, experimental_fast=True, fp32_dest_acc_en=True)
    with expect_error(RuntimeError, "remain finite"):
        ttnn.experimental.dequantize_nvfp4(data, block_data, experimental_fast=True, global_scale=3.4e38)
    with expect_error(ValueError, "requires ttnn.uint16"):
        ttnn.experimental.nvfp4.pack(
            torch.zeros((32, 64), dtype=torch.int32),
            torch.zeros((32, 4), dtype=torch.int32),
            dtype=ttnn.uint8,
            experimental_fast=True,
        )


@pytest.mark.parametrize("fpu_tensor_scale", [False, True])
@pytest.mark.parametrize("memory_config", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
def test_nvfp4_fast_streaming(device, fpu_tensor_scale, memory_config):
    # More packed tiles than worker cores exercises reuse/wrap of both input
    # FIFOs and repeated format changes within a single kernel invocation.
    torch.manual_seed(143)
    codes = torch.randint(0, 16, (1024, 1024), dtype=torch.int32)
    scales = torch.randint(0, 127, (1024, 64), dtype=torch.int32)
    run_fast_decode(
        device, codes, scales, global_scale=0.7314453125, fpu_tensor_scale=fpu_tensor_scale, memory_config=memory_config
    )
