# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.utils_for_testing import assert_numeric_metrics

pytestmark = pytest.mark.use_module_device


@dataclass
class AllCloseThresholds:
    rtol: float
    atol: float


# Poison value to ensure Welford's algorithm ignores padded elements (#31982)
PAD_VALUE = -42


def assert_output_accuracy(torch_output, ttnn_output, use_welford=False):
    """Layer_norm output accuracy check with dtype-/path-conditional bounds.

    The bf16 path and the legacy (non-Welford) fp32 path keep the wider tolerance
    calibrated for bf16's 7-bit mantissa quantization.

    The fp32 + Welford path tightens to bounds derived from a per-output-element
    error model. With unpack_to_dest_mode=UnpackToDestFp32 applied to the Welford
    input CB and to the mean/M2 spill CBs, the dominant per-element error budget is:
    - mean/var estimate accuracy: O(ε_tf32) relative (no per-block compounding)
    - per-op SrcA/SrcB Tf32 truncation in the post-Welford eltwise (sub, rsqrt,
      mul, mul gamma, add bias): each contributes ε_tf32 * |operand|

    For inputs U[0,1) (mean(x) ≈ 1, var(x) ≈ 1/6, |normalized| ≤ 2.45,
    |y| ≤ 3.5), the per-element absolute error sums to ~1e-2 worst case. Typical
    error from random-sign cancellation is ~5e-3, giving relative Frobenius
    ≈ typical_err / typical_|y| ≈ 5e-3 / 1.5 ≈ 3e-3. PCC tracks 1 - O((ε/std)²)
    which is well below 1e-5 of mismatch -- pcc_threshold of 0.99999 leaves ample
    margin.

    Bounds below carry ~1.5x safety margin over the analytical worst case.
    """
    dtype = ttnn_output.dtype
    if dtype == torch.float32 and use_welford:
        rtol = 5e-3
        atol = 1.5e-2
        pcc_threshold = 0.99999
        frobenius_threshold = 5e-3
    elif dtype == torch.bfloat16:
        rtol = 1e-2
        atol = 5e-2
        pcc_threshold = 0.9999
        frobenius_threshold = 0.015
    else:
        rtol = 1e-2
        atol = 5e-2
        pcc_threshold = 0.9999
        frobenius_threshold = 0.0105
    assert_numeric_metrics(
        torch_output,
        ttnn_output,
        rtol=rtol,
        atol=atol,
        pcc_threshold=pcc_threshold,
        frobenius_threshold=frobenius_threshold,
        check_frobenius=True,
        check_pcc=True,
    )


def create_recip_tensor(device, w, use_welford):
    """Helper to create reciprocal tensor for non-sharded welford tests."""
    if not use_welford:
        return None
    grid = device.compute_with_storage_grid_size()
    core_range_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    return ttnn.create_layer_norm_reciprocals(device, core_range_set, w)


@pytest.mark.parametrize("h", [32, 42])
@pytest.mark.parametrize("w", [24, 64])
@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_layer_norm(device, h, w, use_welford, dtype):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=dtype)
    torch_output_tensor = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w])

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, PAD_VALUE)
    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    recip_tensor = create_recip_tensor(device, w, use_welford)
    output_tensor = ttnn.layer_norm(input_tensor, program_config=program_config, recip_tensor=recip_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_output_accuracy(torch_output_tensor, output_tensor, use_welford=use_welford)


@pytest.mark.parametrize("h", [32, 42])
@pytest.mark.parametrize("w", [24, 64])
@pytest.mark.parametrize("use_welford", [True, False])
def test_layer_norm_with_weight_and_bias(device, h, w, use_welford):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    torch_input_tensor = torch.rand((h, w), dtype=dtype)
    torch_weight = torch.rand((w,), dtype=dtype)
    torch_bias = torch.rand((w,), dtype=dtype)

    torch_output_tensor = torch.nn.functional.layer_norm(
        torch_input_tensor, normalized_shape=[w], weight=torch_weight, bias=torch_bias
    )

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, PAD_VALUE)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    recip_tensor = create_recip_tensor(device, w, use_welford)
    output_tensor = ttnn.layer_norm(
        input_tensor, weight=weight, bias=bias, program_config=program_config, recip_tensor=recip_tensor
    )
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_output_accuracy(torch_output_tensor, output_tensor, use_welford=use_welford)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [32])
@pytest.mark.parametrize("use_welford", [True, False])
def test_layer_norm_with_weight_and_bias_row_major(device, h, w, use_welford):
    torch.manual_seed(0)
    dtype = torch.bfloat16

    torch_input_tensor = torch.rand((h, w), dtype=dtype)
    torch_weight = torch.rand((w,), dtype=dtype)
    torch_bias = torch.rand((w,), dtype=dtype)

    torch_output_tensor = torch.nn.functional.layer_norm(
        torch_input_tensor, normalized_shape=[w], weight=torch_weight, bias=torch_bias
    )

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    recip_tensor = create_recip_tensor(device, w, use_welford)
    output_tensor = ttnn.layer_norm(
        input_tensor, weight=weight, bias=bias, program_config=program_config, recip_tensor=recip_tensor
    )
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_output_accuracy(torch_output_tensor, output_tensor, use_welford=use_welford)


@pytest.mark.parametrize("h", [24, 32, 2048])
@pytest.mark.parametrize("w", [42, 64, 127, 519, 4096])
@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_layer_norm_with_weight_bias_and_residual_input(device, h, w, use_welford, dtype):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=dtype)
    torch_residual_input_tensor = torch.rand((h, w), dtype=dtype)
    torch_weight = torch.rand((w,), dtype=dtype)
    torch_bias = torch.rand((w,), dtype=dtype)
    torch_output_tensor = torch.nn.functional.layer_norm(
        torch_input_tensor + torch_residual_input_tensor, normalized_shape=[w], weight=torch_weight, bias=torch_bias
    )

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, PAD_VALUE)
    residual_input_tensor = ttnn.from_torch(torch_residual_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    residual_input_tensor = ttnn.fill_implicit_tile_padding(residual_input_tensor, PAD_VALUE)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    recip_tensor = create_recip_tensor(device, w, use_welford)
    output_tensor = ttnn.layer_norm(
        input_tensor,
        residual_input_tensor=residual_input_tensor,
        weight=weight,
        bias=bias,
        program_config=program_config,
        recip_tensor=recip_tensor,
    )
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_output_accuracy(torch_output_tensor, output_tensor, use_welford=use_welford)


@pytest.mark.parametrize("h", [2, 42])
@pytest.mark.parametrize("w", [24, 512])
def test_layer_norm_with_tile_layout(device, h, w):
    torch.manual_seed(0)
    dtype = torch.bfloat16

    torch_input_tensor = torch.rand((1, h, w), dtype=dtype)
    torch_weight = torch.ones(w, dtype=dtype)
    torch_bias = torch.zeros(w, dtype=dtype)
    torch_output_tensor = torch.nn.functional.layer_norm(
        torch_input_tensor,
        (w,),
        torch_weight,
        torch_bias,
    )

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)
    input_tensor = ttnn.to_device(input_tensor, device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, PAD_VALUE)

    weight = ttnn.from_torch(torch_weight)
    weight = ttnn.to_layout(weight, ttnn.TILE_LAYOUT)
    weight = ttnn.to_device(weight, device)

    bias = ttnn.from_torch(torch_bias)
    bias = ttnn.to_layout(bias, ttnn.TILE_LAYOUT)
    bias = ttnn.to_device(bias, device)

    output_tensor = ttnn.layer_norm(
        input_tensor,
        weight=weight,
        bias=bias,
    )

    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    # test_layer_norm_with_tile_layout exercises the default (non-Welford) bf16 path.
    assert_output_accuracy(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("h", [24, 1024, 2080])
@pytest.mark.parametrize("w", [42, 3200, 4128])
@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_large_layer_norm(device, h, w, use_welford, dtype):
    torch.manual_seed(15)

    torch_input_tensor = torch.rand((h, w), dtype=dtype)
    torch_output_tensor = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w])

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, PAD_VALUE)
    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    recip_tensor = create_recip_tensor(device, w, use_welford)
    output_tensor = ttnn.layer_norm(input_tensor, program_config=program_config, recip_tensor=recip_tensor)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_output_accuracy(torch_output_tensor, output_tensor, use_welford=use_welford)


@pytest.mark.parametrize("h", [24, 2048])
@pytest.mark.parametrize("w", [42, 4096])
@pytest.mark.parametrize("use_welford", [True, False])
def test_large_layer_norm_with_weight_and_bias(device, h, w, use_welford):
    torch.manual_seed(0)
    dtype = torch.bfloat16

    torch_input_tensor = torch.rand((h, w), dtype=dtype)
    torch_weight = torch.rand((w,), dtype=dtype)
    torch_bias = torch.rand((w,), dtype=dtype)

    torch_output_tensor = torch.nn.functional.layer_norm(
        torch_input_tensor, normalized_shape=[w], weight=torch_weight, bias=torch_bias
    )

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, PAD_VALUE)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    recip_tensor = create_recip_tensor(device, w, use_welford)
    output_tensor = ttnn.layer_norm(
        input_tensor, weight=weight, bias=bias, program_config=program_config, recip_tensor=recip_tensor
    )
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_output_accuracy(torch_output_tensor, output_tensor, use_welford=use_welford)


@pytest.mark.parametrize("h", [24, 2048])
@pytest.mark.parametrize("w", [42, 4096])
@pytest.mark.parametrize("use_welford", [True, False])
def test_large_layer_norm_with_weight(device, h, w, use_welford):
    torch.manual_seed(0)
    dtype = torch.bfloat16

    torch_input_tensor = torch.rand((h, w), dtype=dtype)
    torch_weight = torch.rand((w,), dtype=dtype)

    torch_output_tensor = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w], weight=torch_weight)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, PAD_VALUE)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    recip_tensor = create_recip_tensor(device, w, use_welford)
    output_tensor = ttnn.layer_norm(
        input_tensor, weight=weight, program_config=program_config, recip_tensor=recip_tensor
    )
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_output_accuracy(torch_output_tensor, output_tensor, use_welford=use_welford)


@pytest.mark.parametrize("h", [24, 2048])
@pytest.mark.parametrize("w", [42, 4096])
@pytest.mark.parametrize("use_welford", [True, False])
def test_large_layer_norm_with_bias(device, h, w, use_welford):
    torch.manual_seed(0)
    dtype = torch.bfloat16

    torch_input_tensor = torch.rand((h, w), dtype=dtype)
    torch_bias = torch.rand((w,), dtype=dtype)

    torch_output_tensor = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w], bias=torch_bias)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, PAD_VALUE)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    recip_tensor = create_recip_tensor(device, w, use_welford)
    output_tensor = ttnn.layer_norm(input_tensor, bias=bias, program_config=program_config, recip_tensor=recip_tensor)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_output_accuracy(torch_output_tensor, output_tensor, use_welford=use_welford)


@pytest.mark.parametrize("h, w", [(24, 42), (2048, 2048)])
@pytest.mark.parametrize("legacy_reduction", [True, False])
@pytest.mark.parametrize("legacy_rsqrt", [True, False])
def test_large_layer_norm_with_legacy_reduction_and_rsqrt(device, h, w, legacy_reduction, legacy_rsqrt):
    torch.manual_seed(0)
    dtype = torch.bfloat16

    torch_input_tensor = torch.rand((h, w), dtype=dtype)
    torch_bias = torch.rand((w,), dtype=dtype)

    torch_output_tensor = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w], bias=torch_bias)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, PAD_VALUE)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.LayerNormDefaultProgramConfig(
        legacy_reduction=legacy_reduction, legacy_rsqrt=legacy_rsqrt, use_welford=False
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    output_tensor = ttnn.layer_norm(
        input_tensor, bias=bias, compute_kernel_config=compute_kernel_config, program_config=program_config
    )
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    # Non-fp32 accumulation is inaccurate
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.2,
        atol=0.2,
        frobenius_threshold=0.15,
    )


@pytest.mark.parametrize(
    "h, w",
    [
        (32, 2592),
        (32, 3232),
        (1024, 2880),
        # Unaligned shapes to test padding (Issue #31982)
        (19, 2865),
        (19, 4083),
        (1001, 2865),
        (1001, 4083),
    ],
)
@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_large_layer_norm_with_weight_bias_and_residual_input(device, h, w, use_welford, dtype):
    torch.manual_seed(3333)

    torch_input_tensor = torch.rand((h, w), dtype=dtype)
    torch_residual_input_tensor = torch.rand((h, w), dtype=dtype)
    torch_weight = torch.rand((w,), dtype=dtype)
    torch_bias = torch.rand((w,), dtype=dtype)
    torch_output_tensor = torch.nn.functional.layer_norm(
        torch_input_tensor + torch_residual_input_tensor, normalized_shape=[w], weight=torch_weight, bias=torch_bias
    )

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, PAD_VALUE)
    residual_input_tensor = ttnn.from_torch(torch_residual_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    residual_input_tensor = ttnn.fill_implicit_tile_padding(residual_input_tensor, PAD_VALUE)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    recip_tensor = create_recip_tensor(device, w, use_welford)
    output_tensor = ttnn.layer_norm(
        input_tensor,
        residual_input_tensor=residual_input_tensor,
        weight=weight,
        bias=bias,
        program_config=program_config,
        recip_tensor=recip_tensor,
    )
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_output_accuracy(torch_output_tensor, output_tensor, use_welford=use_welford)


@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_l1_interleaved(device, use_welford, dtype):
    torch.manual_seed(0)

    h, w = 32, 64
    torch_input_tensor = torch.rand((h, w), dtype=dtype)
    torch_output_tensor = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w])

    # Create L1 interleaved memory config
    l1_interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.L1,
    )

    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=l1_interleaved_mem_config
    )
    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    recip_tensor = create_recip_tensor(device, w, use_welford)
    output_tensor = ttnn.layer_norm(input_tensor, program_config=program_config, recip_tensor=recip_tensor)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_output_accuracy(torch_output_tensor, output_tensor, use_welford=use_welford)


@run_for_blackhole("The near-capacity allocation is calibrated for Blackhole L1")
def test_l1_interleaved_near_capacity(device):
    torch.manual_seed(20260731)
    device.enable_program_cache()

    h, w = 32, 2048
    torch_input = torch.rand((h, w), dtype=torch.float32)
    torch_residual = torch.rand((h, w), dtype=torch.float32)
    torch_weight = torch.rand((w,), dtype=torch.float32)
    torch_bias = torch.rand((w,), dtype=torch.float32)
    torch_output = torch.nn.functional.layer_norm(
        torch_input + torch_residual,
        normalized_shape=[w],
        weight=torch_weight,
        bias=torch_bias,
    )

    def to_interleaved_l1(tensor):
        return ttnn.from_torch(
            tensor,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def run_layer_norm():
        inputs = (
            to_interleaved_l1(torch_input),
            to_interleaved_l1(torch_residual),
            to_interleaved_l1(torch_weight),
            to_interleaved_l1(torch_bias),
            create_recip_tensor(device, w, True),
        )
        output = ttnn.layer_norm(
            inputs[0],
            residual_input_tensor=inputs[1],
            weight=inputs[2],
            bias=inputs[3],
            program_config=ttnn.LayerNormDefaultProgramConfig(use_welford=True),
            recip_tensor=inputs[4],
        )
        return output, inputs

    warm_output, warm_inputs = run_layer_norm()
    ttnn.synchronize_device(device)
    warm_output.deallocate(force=True)
    for tensor in warm_inputs:
        tensor.deallocate(force=True)

    # Warm the empty-L1 program first, then require a distinct large-tensor
    # program after the allocator span contracts. Occupying 650 KiB/core leaves
    # room for the block-streamed path, but not full-row residual replay.
    grid = device.compute_with_storage_grid_size()
    pressure_tiles = (650 * 1024 * grid.x * grid.y + 2047) // 2048
    l1_pressure = ttnn.allocate_tensor_on_device(
        ttnn.Shape((1, 1, 32, pressure_tiles * 32)),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.L1_MEMORY_CONFIG,
    )
    output, _ = run_layer_norm()

    assert_output_accuracy(torch_output, ttnn.to_torch(output), use_welford=True)
    assert l1_pressure.is_allocated()


@run_for_blackhole("Blackhole selects the tile backend for parameter-free FP32 LayerNorm")
def test_layer_norm_tile_backend_does_not_require_reciprocal(device):
    torch.manual_seed(20260824)
    torch_input = torch.rand((32, 4096), dtype=torch.float32)
    reference = torch.nn.functional.layer_norm(torch_input, normalized_shape=[4096])
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.layer_norm(
        input_tensor,
        program_config=ttnn.LayerNormDefaultProgramConfig(use_welford=True),
    )

    assert_output_accuracy(reference, ttnn.to_torch(output), use_welford=True)


@run_for_blackhole("Blackhole uses two-pass statistics for wide fused BFP8 LayerNorm")
def test_layer_norm_bfp8_residual_affine_two_pass(device):
    torch.manual_seed(20260824)
    shape = (128, 2880)
    torch_input = torch.rand(shape, dtype=torch.float32)
    torch_residual = torch.rand(shape, dtype=torch.float32)
    torch_weight = torch.rand((shape[-1],), dtype=torch.float32)
    torch_bias = torch.rand((shape[-1],), dtype=torch.float32)
    reference = torch.nn.functional.layer_norm(
        torch_input + torch_residual,
        normalized_shape=[shape[-1]],
        weight=torch_weight,
        bias=torch_bias,
    )

    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    residual_tensor = ttnn.from_torch(torch_residual, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    reciprocal = create_recip_tensor(device, shape[-1], use_welford=True)

    output = ttnn.layer_norm(
        input_tensor,
        residual_input_tensor=residual_tensor,
        weight=weight,
        bias=bias,
        program_config=ttnn.LayerNormDefaultProgramConfig(use_welford=True),
        recip_tensor=reciprocal,
    )

    assert_numeric_metrics(
        reference,
        ttnn.to_torch(output),
        pcc_threshold=0.9999,
        rtol=0.01,
        atol=0.07,
        frobenius_threshold=0.015,
    )


@run_for_blackhole("Blackhole replays retained residual rows and multicasts affine parameters")
def test_layer_norm_fp32_residual_affine_replay_program_cache(device):
    torch.manual_seed(20260824)
    h, w = 1024, 2880
    torch_input = torch.rand((h, w), dtype=torch.float32)
    torch_residual = torch.rand((h, w), dtype=torch.float32)
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    residual_tensor = ttnn.from_torch(torch_residual, layout=ttnn.TILE_LAYOUT, device=device)
    reciprocal = create_recip_tensor(device, w, use_welford=True)
    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=True)
    device.enable_program_cache()

    first_weight = ttnn.from_torch(torch.rand((w,), dtype=torch.float32), layout=ttnn.TILE_LAYOUT, device=device)
    first_bias = ttnn.from_torch(torch.rand((w,), dtype=torch.float32), layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.layer_norm(
        input_tensor,
        residual_input_tensor=residual_tensor,
        weight=first_weight,
        bias=first_bias,
        program_config=program_config,
        recip_tensor=reciprocal,
    )

    torch_weight = torch.zeros((w,), dtype=torch.float32)
    torch_bias = torch.full((w,), 2.0, dtype=torch.float32)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.layer_norm(
        input_tensor,
        residual_input_tensor=residual_tensor,
        weight=weight,
        bias=bias,
        program_config=program_config,
        recip_tensor=reciprocal,
    )

    reference = torch.nn.functional.layer_norm(
        torch_input + torch_residual,
        normalized_shape=[w],
        weight=torch_weight,
        bias=torch_bias,
    )
    assert_output_accuracy(reference, ttnn.to_torch(output), use_welford=True)


@pytest.mark.parametrize("dim_a", [24, 2048, 3072, 4096])
@pytest.mark.parametrize("dim_b", [32, 2048, 3072, 4096])
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
def test_layer_norm_across_dtypes(*, device: ttnn.Device, dim_a: int, dim_b: int, dtype: ttnn.DataType) -> None:
    torch.manual_seed(0)

    epsilon = 1e-5
    input_shape = [1, 1, dim_a, dim_b]

    torch_input = torch.randn(input_shape)
    torch_output = torch.nn.functional.layer_norm(torch_input, (input_shape[-1],), eps=epsilon)

    tt_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
    tt_input = ttnn.fill_implicit_tile_padding(tt_input, PAD_VALUE)
    tt_output = ttnn.layer_norm(tt_input, epsilon=epsilon)

    tt_output_torch = ttnn.to_torch(tt_output)

    if dtype == ttnn.bfloat16:
        assert_output_accuracy(torch_output, tt_output_torch)
    elif dtype == ttnn.bfloat8_b:
        assert_numeric_metrics(
            torch_output,
            tt_output_torch,
            pcc_threshold=0.9999,
            rtol=0.01,
            atol=0.07,
            frobenius_threshold=0.015,
        )


@pytest.mark.parametrize("h", [32, 2999, 32 * 64 + 18])
@pytest.mark.parametrize("w", [31, 487, 3821])
@pytest.mark.parametrize("use_welford", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_layer_norm_with_padding(device, h, w, use_welford, dtype):
    """
    Test layer norm on a tensor that is padded with zeros
    in the width dimension.
    Compare against analytic layer norm calculation: (x - mean) / sqrt(var + eps)
    """

    torch.manual_seed(191919)

    # Fill a random number of columns with ones
    non_zero_columns = torch.randint(1, w + 1, (1,)).item()
    torch_input_tensor = torch.zeros((h, w), dtype=dtype)
    torch_input_tensor[:, :non_zero_columns] = torch.ones((h, non_zero_columns), dtype=dtype)

    # Convert to TTNN tensor
    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.Layout.TILE,
        device=device,
    )
    tt_input_tensor = ttnn.fill_implicit_tile_padding(tt_input_tensor, PAD_VALUE)

    # Run layer norm
    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    recip_tensor = create_recip_tensor(device, w, use_welford)
    output_ttnn = ttnn.layer_norm(
        tt_input_tensor,
        program_config=program_config,
        recip_tensor=recip_tensor,
    )
    output_ttnn = ttnn.to_torch(output_ttnn)

    # Compute golden layer normoutput
    golden = ttnn.get_golden_function(ttnn.layer_norm)
    golden_output = golden(torch_input_tensor, weight=None, bias=None, eps=1e-5)

    assert_output_accuracy(golden_output, output_ttnn)


def test_layer_norm_inputs_requires_input_tensor(expect_error):
    """``LayerNormInputs()`` without an input tensor must raise."""

    with expect_error(TypeError, "."):
        ttnn.LayerNormInputs()


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [42, 127, 519])
@pytest.mark.parametrize("op_name", ["layer_norm", "rms_norm"])
def test_norm_row_major_weight_partial_tile_padding(device, h, w, op_name):
    """Row-major weight/bias with a TILE input whose logical W is not tile-aligned must
    not let implicit-padding values pollute per-row mean/variance.

    The interleaved row-major-gamma reader generates only a single full-tile reduce
    scaler. The reduce LLK then sums across all 32 columns of the last tile, including
    the implicit padded region past the logical width. If those padded bytes are
    non-zero, sum(x) and sum(x^2) absorb them and the resulting normalization is wrong.
    A correct kernel must either zero the input padding before the reduce or generate
    a masked partial-last-tile scaler so the padded columns contribute zero.

    Welford is excluded from this case: layer_norm-Welford handles the partial last
    tile via last_tile_rows in the compute kernel, and rms_norm forbids Welford.
    """
    torch.manual_seed(0)
    dtype = torch.bfloat16
    tile_w = 32
    padded_w = (w + tile_w - 1) // tile_w * tile_w
    wt = padded_w // tile_w

    torch_input_tensor = torch.rand((h, w), dtype=dtype)
    torch_weight = torch.rand((w,), dtype=dtype)
    torch_bias = torch.rand((w,), dtype=dtype)

    # Row-major weight/bias must have padded last dim == tile_width and physical volume
    # == padded_W. Pad the logical W values with zeros and reshape to (Wt, 32) so the
    # Wt-element tile pages match the padded input width.
    weight_rm = torch.cat([torch_weight, torch.zeros(padded_w - w, dtype=dtype)]).reshape(wt, tile_w)
    bias_rm = torch.cat([torch_bias, torch.zeros(padded_w - w, dtype=dtype)]).reshape(wt, tile_w)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, PAD_VALUE)
    weight = ttnn.from_torch(weight_rm, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    bias = ttnn.from_torch(bias_rm, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=False)

    if op_name == "layer_norm":
        torch_output_tensor = torch.nn.functional.layer_norm(
            torch_input_tensor, normalized_shape=[w], weight=torch_weight, bias=torch_bias
        )
        output_tensor = ttnn.layer_norm(input_tensor, weight=weight, bias=bias, program_config=program_config)
    else:
        x = torch_input_tensor.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-12)
        torch_output_tensor = (x * rms * torch_weight.float() + torch_bias.float()).to(dtype)
        output_tensor = ttnn.rms_norm(input_tensor, weight=weight, bias=bias, program_config=program_config)

    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_output_accuracy(torch_output_tensor, output_tensor)
