# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Full two-pass regression matrices; sanity retains a small representative sample."""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_numeric_metrics
from models.common.utility_functions import run_for_blackhole, run_for_wormhole_b0_or_blackhole
from tests.ttnn.unit_tests.operations.fused.test_layer_norm import assert_output_accuracy, create_recip_tensor

pytestmark = pytest.mark.use_module_device


@pytest.fixture
def enabled_program_cache(device):
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    try:
        yield
    finally:
        device.disable_and_clear_program_cache()


@pytest.mark.parametrize("width", [128, 4096, 16384])
@pytest.mark.parametrize("outlier_column", [0, -1], ids=["outlier_anchor", "representative_anchor"])
def test_layer_norm_welford_unrepresentative_anchor(device, width, outlier_column):
    """An isolated outlier must not make the first-value shift lose row statistics."""
    values = torch.full((32, width), 1.1015625, dtype=torch.float32)
    values[:, outlier_column] = 0
    epsilon = 1e-5
    reference = torch.nn.functional.layer_norm(values.double(), [width], eps=epsilon)
    input_tensor = ttnn.from_torch(values, layout=ttnn.TILE_LAYOUT, device=device)
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    output = ttnn.layer_norm(
        input_tensor,
        epsilon=epsilon,
        program_config=ttnn.LayerNormDefaultProgramConfig(use_welford=True),
        recip_tensor=create_recip_tensor(device, width, use_welford=True),
        compute_kernel_config=compute_config,
    )
    actual = ttnn.to_torch(output).double()
    assert torch.isfinite(actual).all()
    assert_numeric_metrics(reference, actual, rtol=5e-4, atol=1e-4, frobenius_threshold=5e-4)
    # PCC alone cannot detect a common error in the subtracted mean.
    assert actual.mean(dim=-1).abs().max() < 1e-4


@pytest.mark.parametrize("width", [256, 16384])
@pytest.mark.parametrize("has_residual", [False, True], ids=["plain", "residual"])
def test_layer_norm_welford_large_offset(device, width, has_residual):
    """The final subtraction must preserve variations below the large row mean."""
    torch.manual_seed(19)
    rows = 64
    base = 5000.0 if has_residual else 10000.0
    torch_input = (base + 64.0 * torch.randn((rows, width))).to(torch.bfloat16)
    torch_residual = (base + 64.0 * torch.randn((rows, width))).to(torch.bfloat16) if has_residual else None
    reference_input = torch_input.to(torch.float64)
    if torch_residual is not None:
        reference_input += torch_residual.to(torch.float64)
    reference = torch.nn.functional.layer_norm(reference_input, [width])

    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    residual_tensor = (
        ttnn.from_torch(torch_residual, layout=ttnn.TILE_LAYOUT, device=device) if torch_residual is not None else None
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    output = ttnn.layer_norm(
        input_tensor,
        residual_input_tensor=residual_tensor,
        program_config=ttnn.LayerNormDefaultProgramConfig(use_welford=True),
        recip_tensor=create_recip_tensor(device, width, use_welford=True),
        compute_kernel_config=compute_kernel_config,
    )
    actual = ttnn.to_torch(output).to(torch.float64)

    assert torch.isfinite(actual).all()
    assert_numeric_metrics(reference, actual, rtol=0, atol=0.025, frobenius_threshold=0.025)
    assert actual.mean(dim=-1).abs().max() < 0.004


@pytest.mark.parametrize(
    "rows,width",
    [
        (32, 8192),
        (512, 8192),
        (1024, 8192),
        (32, 16384),
        pytest.param(None, 487, id="repeated_rows_partial_width"),
        pytest.param(None, 2880, id="repeated_rows_aligned_width"),
    ],
)
def test_layer_norm_welford_fp32_residual_large_offset(device, rows, width):
    """Fused FP32 pre-add must preserve variation below a large shared offset."""
    torch.manual_seed(29)
    base = 1_000_000.0
    scale = 64.0
    if rows is None:
        # Tiled FP32 residuals select the large kernel. Force several NCHt
        # iterations per core and vary statistics to expose stale DST/DFB state.
        grid = device.compute_with_storage_grid_size()
        rows = 32 * (2 * grid.x * grid.y + 1)
        tile_row = (torch.arange(rows, dtype=torch.float32) // 32).unsqueeze(1)
        base = base + 128.0 * tile_row
        scale = scale + 8.0 * (tile_row % 7)
    torch_input = base + scale * torch.randn((rows, width), dtype=torch.float32)
    torch_residual = base + scale * torch.randn((rows, width), dtype=torch.float32)
    reference = torch.nn.functional.layer_norm(
        torch_input.to(torch.float64) + torch_residual.to(torch.float64),
        [width],
    )

    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    residual_tensor = ttnn.from_torch(torch_residual, layout=ttnn.TILE_LAYOUT, device=device)
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    output = ttnn.layer_norm(
        input_tensor,
        residual_input_tensor=residual_tensor,
        program_config=ttnn.LayerNormDefaultProgramConfig(use_welford=True),
        recip_tensor=create_recip_tensor(device, width, use_welford=True),
        compute_kernel_config=compute_kernel_config,
    )
    actual = ttnn.to_torch(output).to(torch.float64)

    error = actual - reference
    assert torch.isfinite(actual).all()
    assert_numeric_metrics(reference, actual, rtol=0, atol=0.025, frobenius_threshold=0.025)
    assert error.abs().mean() < 0.004


@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize("width", [128, 2880])
@pytest.mark.parametrize("anchor", [float("inf"), -float("inf"), float("nan")], ids=["inf", "neg_inf", "nan"])
def test_layer_norm_fp32_residual_nonfinite_anchor_is_row_local(device, enabled_program_cache, width, anchor):
    """Unused statistics lanes must not broadcast another row's non-finite anchor."""
    torch.manual_seed(71)
    clean_input = torch.randn((64, width), dtype=torch.float32)
    residual = torch.randn_like(clean_input)
    residual_tensor = ttnn.from_torch(residual, layout=ttnn.TILE_LAYOUT, device=device)
    config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
    )
    reciprocals = create_recip_tensor(device, width, use_welford=True)
    for poison in (True, False):
        x = clean_input.clone()
        if poison:
            # These anchors used to occupy padding that transposes into other rows.
            x[16, 0] = anchor
            x[62, 0] = anchor
        reference = torch.nn.functional.layer_norm(x.double() + residual.double(), [width])
        input_tensor = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
        output = ttnn.layer_norm(
            input_tensor,
            residual_input_tensor=residual_tensor,
            program_config=ttnn.LayerNormDefaultProgramConfig(use_welford=True),
            recip_tensor=reciprocals,
            compute_kernel_config=config,
        )
        actual = ttnn.to_torch(output).double()
        finite_rows = torch.isfinite(reference).all(dim=-1)
        assert torch.isfinite(actual[finite_rows]).all()
        torch.testing.assert_close(actual[finite_rows], reference[finite_rows], rtol=5e-3, atol=1.5e-2)


@pytest.mark.parametrize(
    "rows,width,has_residual,has_gamma,has_beta",
    [
        pytest.param(32, 64, False, False, False, id="compact_plain"),
        pytest.param(32, 64, False, True, False, id="compact_gamma"),
        pytest.param(32, 64, False, False, True, id="compact_beta"),
        pytest.param(64, 2880, False, True, True, id="affine_multi_row_tile"),
        pytest.param(32, 2880, True, False, False, id="residual_plain"),
        pytest.param(32, 2880, True, True, False, id="residual_gamma"),
        pytest.param(32, 2880, True, False, True, id="residual_beta"),
        pytest.param(32, 2880, True, True, True, id="residual_affine"),
    ],
)
def test_layer_norm_welford_fp32_finalizer_large_offset(device, rows, width, has_residual, has_gamma, has_beta):
    """All FP32 finalizer variants must retain variation below a shared offset."""
    torch.manual_seed(37)
    base = 1_000_000.0
    torch_input = base + 64.0 * torch.randn((rows, width), dtype=torch.float32)
    torch_residual = base + 64.0 * torch.randn((rows, width), dtype=torch.float32) if has_residual else None
    torch_weight = torch.linspace(0.75, 1.25, width, dtype=torch.float32) if has_gamma else None
    torch_bias = torch.linspace(-0.25, 0.25, width, dtype=torch.float32) if has_beta else None

    reference_input = torch_input.to(torch.float64)
    if torch_residual is not None:
        reference_input += torch_residual.to(torch.float64)
    reference = torch.nn.functional.layer_norm(
        reference_input,
        [width],
        weight=torch_weight.to(torch.float64) if torch_weight is not None else None,
        bias=torch_bias.to(torch.float64) if torch_bias is not None else None,
    )

    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    residual_tensor = (
        ttnn.from_torch(torch_residual, layout=ttnn.TILE_LAYOUT, device=device) if torch_residual is not None else None
    )
    weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device) if torch_weight is not None else None
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device) if torch_bias is not None else None
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    output = ttnn.layer_norm(
        input_tensor,
        residual_input_tensor=residual_tensor,
        weight=weight,
        bias=bias,
        program_config=ttnn.LayerNormDefaultProgramConfig(use_welford=True),
        recip_tensor=create_recip_tensor(device, width, use_welford=True),
        compute_kernel_config=compute_kernel_config,
    )
    actual = ttnn.to_torch(output).to(torch.float64)

    error = actual - reference
    assert torch.isfinite(actual).all()
    assert_numeric_metrics(reference, actual, rtol=0, atol=0.025, frobenius_threshold=0.025)
    assert error.abs().mean() < 0.004


@pytest.mark.parametrize(
    "has_gamma,has_beta",
    [
        pytest.param(True, False, id="gamma"),
        pytest.param(False, True, id="beta"),
        pytest.param(True, True, id="gamma_beta"),
    ],
)
@pytest.mark.parametrize("repeat_rows_per_core", [False, True], ids=["single_row", "repeated_rows"])
def test_layer_norm_fp32_residual_with_row_major_affine(device, has_gamma, has_beta, repeat_rows_per_core):
    """Every row-major affine variant must select a matching full-precision finaliser."""
    torch.manual_seed(11)
    grid = device.compute_with_storage_grid_size()
    tile_rows = 2 * grid.x * grid.y + 1 if repeat_rows_per_core else 1
    h, w = 32 * tile_rows, 32
    base = 1_000_000.0
    # More tile rows than cores forces DST/DFB reuse. Vary both mean and
    # variance across tile rows to expose stale statistics on later iterations.
    row_id = (torch.arange(h, dtype=torch.float32) // 32).unsqueeze(1)
    row_base = base + 128.0 * row_id
    row_scale = 64.0 + 8.0 * (row_id % 7)
    torch_input = row_base + row_scale * torch.randn((h, w), dtype=torch.float32)
    torch_residual = row_base + row_scale * torch.randn((h, w), dtype=torch.float32)
    torch_weight = torch.randn((w,), dtype=torch.float32) if has_gamma else None
    torch_bias = torch.randn((w,), dtype=torch.float32) if has_beta else None
    reference = torch.nn.functional.layer_norm(
        torch_input.to(torch.float64) + torch_residual.to(torch.float64),
        [w],
        torch_weight.to(torch.float64) if torch_weight is not None else None,
        torch_bias.to(torch.float64) if torch_bias is not None else None,
    )

    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    residual_tensor = ttnn.from_torch(torch_residual, layout=ttnn.TILE_LAYOUT, device=device)
    weight = (
        ttnn.from_torch(torch_weight.reshape(-1, 32), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        if torch_weight is not None
        else None
    )
    bias = (
        ttnn.from_torch(torch_bias.reshape(-1, 32), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        if torch_bias is not None
        else None
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    output = ttnn.layer_norm(
        input_tensor,
        residual_input_tensor=residual_tensor,
        weight=weight,
        bias=bias,
        program_config=ttnn.LayerNormDefaultProgramConfig(use_welford=True),
        compute_kernel_config=compute_kernel_config,
    )

    actual = ttnn.to_torch(output).to(torch.float64)
    error = actual - reference
    assert torch.isfinite(actual).all()
    assert_numeric_metrics(reference, actual, rtol=0, atol=0.025, frobenius_threshold=0.025)
    assert error.abs().mean() < 0.004


@run_for_blackhole("The near-capacity allocation is calibrated for Blackhole L1")
def test_layer_norm_compact_fp32_omits_centred_buffer(device, enabled_program_cache):
    torch.manual_seed(20260907)
    shape = (64, 4096)
    inputs = [1_000_000.0 + 128.0 * torch.rand(shape) for _ in range(2)]
    device_inputs = [ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device) for x in inputs]
    references = [torch.nn.functional.layer_norm(x.double(), [shape[-1]]).float() for x in inputs]
    config = ttnn.LayerNormDefaultProgramConfig(use_welford=True)

    # No reciprocal tensor: falling back to the large streaming kernel is an error.
    warm_output = ttnn.layer_norm(device_inputs[0], program_config=config)
    assert_output_accuracy(references[0], ttnn.to_torch(warm_output), use_welford=True)
    warm_output.deallocate(force=True)

    # The compact footprint fits only if the unused full-row XMM buffer is absent.
    # 650 KiB per bank leaves this Blackhole configuration between the compact
    # footprints with and without XMM; this is a footprint boundary, not random pressure.
    grid = device.compute_with_storage_grid_size()
    pressure_bytes_per_bank = 650 * 1024
    bf16_tile_bytes = 32 * 32 * 2
    pressure_tiles = (pressure_bytes_per_bank * grid.x * grid.y + bf16_tile_bytes - 1) // bf16_tile_bytes
    l1_pressure = ttnn.allocate_tensor_on_device(
        ttnn.Shape((1, 1, 32, pressure_tiles * 32)),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.L1_MEMORY_CONFIG,
    )
    cache_entries = None
    for input_tensor, reference in zip(device_inputs, references):
        output = ttnn.layer_norm(input_tensor, program_config=config)
        assert_output_accuracy(reference, ttnn.to_torch(output), use_welford=True)
        if cache_entries is not None:
            assert device.num_program_cache_entries() == cache_entries
        cache_entries = device.num_program_cache_entries()
    assert l1_pressure.is_allocated()


@run_for_blackhole("The near-capacity allocation is calibrated for Blackhole L1")
def test_l1_interleaved_near_capacity(device, enabled_program_cache):
    torch.manual_seed(20260731)

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
    pressure_bytes_per_bank = 650 * 1024
    bf16_tile_bytes = 32 * 32 * 2
    pressure_tiles = (pressure_bytes_per_bank * grid.x * grid.y + bf16_tile_bytes - 1) // bf16_tile_bytes
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


@pytest.mark.parametrize("shape", [(1, 1, 64, 256), (1, 1, 37, 288)])
@pytest.mark.parametrize("gamma_dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("gamma_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_layer_norm_bfp8_compensated_subtraction_tile_stride(device, shape, gamma_dtype, gamma_layout):
    # Each BFP8 tile has its own exponent section. Advancing the unpacker's W
    # counter instead of the CB page address corrupts tiles 1-3 of every block.
    torch.manual_seed(0)
    width = shape[-1]
    torch_input = torch.rand(shape, dtype=torch.float32)
    torch_weight = torch.rand((width,), dtype=torch.float32)
    torch_bias = torch.rand((width,), dtype=torch.float32)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    affine_shape = (1, 1, width // 32, 32) if gamma_layout == ttnn.ROW_MAJOR_LAYOUT else (1, 1, 1, width)
    weight = ttnn.from_torch(torch_weight.reshape(affine_shape), dtype=gamma_dtype, layout=gamma_layout, device=device)
    bias = ttnn.from_torch(torch_bias.reshape(affine_shape), dtype=gamma_dtype, layout=gamma_layout, device=device)
    reference = torch.nn.functional.layer_norm(
        ttnn.to_torch(input_tensor).float(),
        (width,),
        weight=ttnn.to_torch(weight).float().reshape(width),
        bias=ttnn.to_torch(bias).float().reshape(width),
        eps=1e-12,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    output = ttnn.layer_norm(
        input_tensor,
        epsilon=1e-12,
        weight=weight,
        bias=bias,
        program_config=ttnn.LayerNormDefaultProgramConfig(use_welford=True),
        compute_kernel_config=compute_config,
        recip_tensor=create_recip_tensor(device, width, use_welford=True),
    )
    actual = ttnn.to_torch(output).float()
    assert torch.isfinite(actual).all()
    # Check each tile separately so correct first tiles cannot hide subsequent
    # tiles with misaddressed mantissas or shared exponents.
    for tile_start in range(0, width, 32):
        assert_numeric_metrics(
            reference[..., tile_start : tile_start + 32],
            actual[..., tile_start : tile_start + 32],
            rtol=0.02,
            atol=0.05,
            frobenius_threshold=0.05,
        )


@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize("multicast", [False, True], ids=["replay", "affine_multicast"])
@pytest.mark.parametrize("width", [487, 2880, 3217])
@pytest.mark.parametrize("offset", [0.0, 1_000_000.0])
def test_layer_norm_fp32_residual_affine_replay_program_cache(device, enabled_program_cache, multicast, width, offset):
    torch.manual_seed(20260824)

    def check_output(reference, output):
        if offset:
            # Match the large-offset FP32 residual regressions; the ordinary
            # accuracy helper's tighter absolute bound assumes U[0, 1) inputs.
            assert_numeric_metrics(
                reference, output, rtol=0, atol=0.025, frobenius_threshold=0.005, pcc_threshold=0.99999
            )
        else:
            assert_output_accuracy(reference, output, use_welford=True)

    # Sixteen tile rows stay below the 20-core multicast crossover.
    h, w = 512, width
    if multicast:
        grid = device.compute_with_storage_grid_size()
        # Reach the 20-core crossover with a complete rectangle in row-wise allocation order.
        core_rows = (20 + grid.x - 1) // grid.x
        if core_rows > grid.y:
            pytest.skip("Affine multicast requires at least 20 compute cores")
        h = 32 * grid.x * core_rows
    scale = 128.0 if offset else 1.0
    torch_input = offset + scale * torch.rand((h, w), dtype=torch.float32)
    torch_residual = offset + scale * torch.rand((h, w), dtype=torch.float32)
    reference_input = torch_input.double() + torch_residual.double()
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    residual_tensor = ttnn.from_torch(torch_residual, layout=ttnn.TILE_LAYOUT, device=device)
    reciprocal = create_recip_tensor(device, w, use_welford=True)
    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=True)
    first_torch_weight = torch.rand((w,), dtype=torch.float32)
    first_torch_bias = torch.rand((w,), dtype=torch.float32)
    first_weight = ttnn.from_torch(first_torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
    first_bias = ttnn.from_torch(first_torch_bias, layout=ttnn.TILE_LAYOUT, device=device)
    first_output = ttnn.layer_norm(
        input_tensor,
        residual_input_tensor=residual_tensor,
        weight=first_weight,
        bias=first_bias,
        program_config=program_config,
        recip_tensor=reciprocal,
    )
    first_reference = torch.nn.functional.layer_norm(
        reference_input, [w], weight=first_torch_weight.double(), bias=first_torch_bias.double()
    )
    check_output(first_reference, ttnn.to_torch(first_output))
    cache_entries = device.num_program_cache_entries()

    torch_weight = torch.rand((w,), dtype=torch.float32) + 0.5
    torch_bias = torch.rand((w,), dtype=torch.float32) + 2.0
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
        reference_input,
        normalized_shape=[w],
        weight=torch_weight.double(),
        bias=torch_bias.double(),
    )
    check_output(reference, ttnn.to_torch(output))
    assert device.num_program_cache_entries() == cache_entries
