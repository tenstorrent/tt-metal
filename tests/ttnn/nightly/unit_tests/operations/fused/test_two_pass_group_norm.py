# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Full two-pass regression matrices; sanity retains a small representative sample."""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_numeric_metrics
from models.common.utility_functions import run_for_wormhole_b0_or_blackhole
from tests.ttnn.unit_tests.operations.fused.test_group_norm import DEVICE_PARAMS_L1_SMALL_SIZE


@pytest.fixture
def enabled_program_cache(device):
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize("base, amplitude", [(1.0, 0.01), (10.0, 0.05)])
@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
def test_group_norm_stable_stats_translation_stability(device, base, amplitude):
    """Preserve low BF16 group variance in the presence of a shared input offset."""
    torch.manual_seed(7)
    N, C, H, W, num_groups = 1, 1280, 32, 32, 32
    grid = ttnn.CoreGrid(y=8, x=8)

    torch_input = (base + amplitude * torch.randn((N, C, H, W))).to(torch.bfloat16)
    reference = torch.nn.functional.group_norm(torch_input.float(), num_groups, eps=1e-12)
    reference = reference.permute(0, 2, 3, 1).reshape(N, 1, H * W, C)
    nhwc = torch_input.permute(0, 2, 3, 1).reshape(N, 1, H * W, C)
    memory_config = ttnn.create_sharded_memory_config(
        shape=nhwc.shape,
        core_grid=grid,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_tensor = ttnn.from_torch(
        nhwc,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=memory_config,
        device=device,
    )
    input_mask = ttnn.create_group_norm_input_mask(C, num_groups, grid.x, ttnn.bfloat8_b)
    input_mask = ttnn.to_device(input_mask, device)

    output = ttnn.group_norm(
        input_tensor,
        num_groups=num_groups,
        epsilon=1e-12,
        input_mask=input_mask,
        memory_config=memory_config,
        core_grid=grid,
        inplace=False,
        use_welford=True,
    )
    output = ttnn.to_torch(ttnn.from_device(output)).float()

    assert torch.isfinite(output).all()
    rmse = torch.mean((output - reference) ** 2).sqrt().item()
    pcc = torch.corrcoef(torch.stack((output.flatten(), reference.flatten())))[0, 1].item()
    assert rmse < 0.06
    assert pcc > 0.9995


@pytest.mark.parametrize("has_affine", [False, True], ids=["plain", "affine"])
@pytest.mark.parametrize("num_groups", [1, 16])
@pytest.mark.parametrize(
    "constant",
    [None, 1e38, -1e38, 1e-37, -1e-37],
    ids=["offset", "large_positive", "large_negative", "small_positive", "small_negative"],
)
def test_group_norm_sharded_fp32_large_offset(device, has_affine, num_groups, constant):
    """Sharded FP32 normalization must retain low-order input variation."""
    torch.manual_seed(7)
    N, C, H, W = 1, 256, 1, 256
    grid = ttnn.CoreGrid(y=1, x=1)
    x = 1_000_000.0 + 128.0 * (torch.rand((N, C, H, W), dtype=torch.float32) - 0.5)
    if constant is not None:
        x.fill_(constant)
    weight = torch.linspace(0.75, 1.25, C, dtype=torch.float32) if has_affine else None
    bias = torch.linspace(-0.25, 0.25, C, dtype=torch.float32) if has_affine else None
    if constant is not None and bias is not None:
        # Keep beta exact through the TF32 affine stage so equality tests the
        # statistics result, not parameter truncation in that later stage.
        bias = bias.to(torch.bfloat16).to(torch.float32)
    if constant is None:
        reference = (
            torch.nn.functional.group_norm(x, num_groups, weight=weight, bias=bias)
            .permute(0, 2, 3, 1)
            .reshape(N, 1, H * W, C)
        )
    else:
        # The analytical result avoids overflow in the CPU reference too.
        reference = torch.zeros((N, 1, H * W, C), dtype=torch.float32)
        if bias is not None:
            reference += bias.view(1, 1, 1, C)

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    input_tensor = ttnn.from_torch(
        x.permute(0, 2, 3, 1).reshape(N, 1, H * W, C),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_mask = ttnn.to_device(ttnn.create_group_norm_input_mask(C, num_groups, grid.y, ttnn.bfloat8_b), device)
    if has_affine:
        gamma = ttnn.from_torch(
            ttnn.create_group_norm_weight_bias_rm(weight, C, grid.y),
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        beta = ttnn.from_torch(
            ttnn.create_group_norm_weight_bias_rm(bias, C, grid.y),
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
        gamma = beta = None
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    shard_spec = ttnn.ShardSpec(shard_grid, (H * W, C), ttnn.ShardOrientation.COL_MAJOR)
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )
    input_tensor = ttnn.to_memory_config(input_tensor, memory_config)
    output = ttnn.group_norm(
        input_tensor,
        num_groups=num_groups,
        input_mask=input_mask,
        weight=gamma,
        bias=beta,
        memory_config=memory_config,
        core_grid=grid,
        dtype=ttnn.float32,
        compute_kernel_config=compute_kernel_config,
        use_welford=True,
        output_layout=ttnn.TILE_LAYOUT,
        inplace=False,
    )
    actual = ttnn.to_torch(ttnn.from_device(ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG))).float()

    error = actual - reference
    assert torch.isfinite(actual).all()
    if constant is not None:
        # Exact equality also detects small means flushed by premature scaling.
        assert torch.equal(actual, reference)
    else:
        assert_numeric_metrics(reference, actual, rtol=0, atol=0.02, frobenius_threshold=0.02)
        assert error.abs().mean() < 0.004


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("grid_size, spatial, num_groups", [(1, 128, 16), (8, 1024, 32)], ids=["single_core", "8x8"])
@pytest.mark.parametrize(
    "dtype,layout,mask_dtype",
    [
        pytest.param(ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.bfloat8_b, id="bf16_tile"),
        pytest.param(ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat8_b, id="bf16_row_major"),
        pytest.param(ttnn.float32, ttnn.TILE_LAYOUT, ttnn.bfloat8_b, id="fp32_tile"),
        pytest.param(ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat8_b, id="fp32_row_major"),
        pytest.param(ttnn.float32, ttnn.TILE_LAYOUT, ttnn.bfloat16, id="fp32_tile_bf16_mask"),
        pytest.param(ttnn.float32, ttnn.TILE_LAYOUT, ttnn.float32, id="fp32_tile_fp32_mask"),
    ],
)
@pytest.mark.parametrize(
    "has_weight, has_bias",
    [(False, False), (True, False), (False, True), (True, True)],
    ids=["no_affine", "weight_only", "bias_only", "full_affine"],
)
def test_group_norm_sharded_optional_affine_program_cache(
    device, enabled_program_cache, grid_size, spatial, num_groups, dtype, layout, mask_dtype, has_weight, has_bias
):
    """Mask/affine format changes must preserve every output tile on cached calls too."""
    available_grid = device.compute_with_storage_grid_size()
    if min(available_grid.x, available_grid.y) < grid_size:
        pytest.skip(f"Requires a {grid_size}x{grid_size} compute grid")
    channels = 256
    grid = ttnn.CoreGrid(y=grid_size, x=grid_size)
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    weight = torch.linspace(0.75, 1.25, channels).to(torch_dtype) if has_weight else None
    bias = torch.linspace(-0.25, 0.25, channels).to(torch_dtype) if has_bias else None

    def make_parameter(value):
        if value is None:
            return None
        return ttnn.from_torch(
            ttnn.create_group_norm_weight_bias_rm(value, channels, grid.y),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    gamma, beta = make_parameter(weight), make_parameter(bias)
    input_mask = ttnn.to_device(ttnn.create_group_norm_input_mask(channels, num_groups, grid.y, mask_dtype), device)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size - 1, grid_size - 1))})
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED if grid_size == 1 else ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(shard_grid, (spatial // grid_size, channels // grid_size), ttnn.ShardOrientation.COL_MAJOR),
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    previous_input = None
    cache_entries = None
    for repeat in range(3):
        torch.manual_seed(41 + repeat)
        # Quantise before forming the FP64 reference, matching values sent to the device.
        source = torch.rand((1, channels, 1, spatial), dtype=torch.float32).to(torch_dtype)
        reference = (
            torch.nn.functional.group_norm(
                source.double(),
                num_groups,
                weight.double() if weight is not None else None,
                bias.double() if bias is not None else None,
            )
            .permute(0, 2, 3, 1)
            .reshape(1, 1, spatial, channels)
        )
        input_tensor = ttnn.from_torch(
            source.permute(0, 2, 3, 1).reshape(1, 1, spatial, channels),
            dtype=dtype,
            layout=layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        input_tensor = ttnn.to_memory_config(input_tensor, memory_config)
        if previous_input is not None:
            assert input_tensor.buffer_address() != previous_input.buffer_address()
        output = ttnn.group_norm(
            input_tensor,
            num_groups=num_groups,
            input_mask=input_mask,
            weight=gamma,
            bias=beta,
            memory_config=memory_config,
            core_grid=grid,
            dtype=dtype,
            compute_kernel_config=compute_config,
            use_welford=True,
            output_layout=layout,
            inplace=False,
        )
        actual = ttnn.to_torch(ttnn.from_device(ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG))).double()
        assert torch.isfinite(actual).all()
        assert_numeric_metrics(
            reference,
            actual.reshape(reference.shape),
            pcc_threshold=0.999,
            rtol=0.008 if dtype == ttnn.float32 else 0.01,
            atol=0.02 if dtype == ttnn.float32 else 0.06,
            frobenius_threshold=0.004 if dtype == ttnn.float32 else 0.015,
        )
        if cache_entries is None:
            cache_entries = device.num_program_cache_entries()
        else:
            assert device.num_program_cache_entries() == cache_entries
        # Keep the old allocation live until the next input is allocated, exercising address updates.
        previous_input = input_tensor
        del output
