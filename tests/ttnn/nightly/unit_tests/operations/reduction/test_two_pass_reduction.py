# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Full two-pass regression matrices; sanity retains a small representative sample."""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_numeric_metrics
from models.common.utility_functions import is_blackhole
from tests.ttnn.unit_tests.operations.reduce.test_reduction import enabled_program_cache


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("correction", [False, True])
@pytest.mark.parametrize(
    "shape,dim",
    [
        ((1, 1, 32, 32), (-2, -1)),
        ((16, 1, 64, 64), (-2, -1)),
        ((2, 1, 65, 128), (-2, -1)),
        ((2, 3, 33, 128), (1, 2, 3)),
        ((1, 1, 32, 10528), (-2, -1)),
        ((1, 4, 33, 32), (1, 2, 3)),
        ((1, 1, 33, 96), (-2, -1)),
        ((1, 1, 65, 63), (-2, -1)),
        ((2, 3, 33, 33), (1, 2, 3)),
        ((1, 1, 33, 129), (-2, -1)),
        ((1, 1, 480, 128), (-2, -1)),
        ((1, 1, 481, 128), (-2, -1)),
        ((1, 4, 512, 32), (1, 2, 3)),
        ((1, 32, 481, 65), (-2, -1)),
        ((1, 1, 512, 64), (-2, -1)),
        ((1, 32, 736, 96), (-2, -1)),
    ],
    ids=[
        "single_tile",
        "two_columns_multiple_outputs",
        "partial_height",
        "batch_merge",
        "uneven_tree",
        "four_batched_columns",
        "three_columns",
        "narrow_partial_width",
        "narrow_batch_partial_width",
        "partial_width_tail",
        "hw_below_replay_boundary",
        "hw_replay_partial_height",
        "hw_replay_batch_columns",
        "hw_replay_partial_width",
        "hw_replay_narrow_single_core",
        "hw_replay_narrow_multi_core",
    ],
)
def test_std_var_hw_compact_lane_combine(device, enabled_program_cache, dtype, correction, shape, dim):
    # Exercise the shared Wormhole/Blackhole SFPU combine and its scalar fallbacks. Distinct lane means
    # and nonzero lane variances exercise both terms and their relative scaling.
    torch.manual_seed(123)
    values = torch.randn(shape) + (torch.arange(shape[-1]) % 32).float() + 1024
    tt_input = ttnn.from_torch(values, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    quantised_input = ttnn.to_torch(tt_input).to(torch.float64)
    rtol = 2e-4 if dtype == ttnn.float32 else 0.01 if dtype == ttnn.bfloat16 else 0.03
    for torch_op, ttnn_op in ((torch.var, ttnn.var), (torch.std, ttnn.std)):
        expected = torch_op(quantised_input, dim=dim, keepdim=True, correction=int(correction))
        for _ in range(3):
            actual = ttnn.to_torch(ttnn_op(tt_input, dim=dim, keepdim=True, correction=correction)).to(torch.float64)
            assert torch.isfinite(actual).all()
            # Scalar or nearly constant variance outputs make correlation uninformative.
            assert_numeric_metrics(expected, actual, rtol=rtol, atol=1e-7, frobenius_threshold=rtol, check_pcc=False)


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("tail_columns", [1, 15, 16, 31])
@pytest.mark.parametrize("batches", [1, 3, 32, 33])
@pytest.mark.parametrize("height", [33, 512], ids=["partial_height", "replay"])
def test_std_var_hw_compact_partial_width(device, enabled_program_cache, dtype, tail_columns, batches, height):
    # Each batch contributes four compact leaves and a scalar tail. The tails
    # cross leaf boundaries, sometimes filling the final leaf exactly, while
    # the two outputs exercise repeated publication through the same buffers.
    width = 128 + tail_columns
    torch.manual_seed(31)
    values = torch.randn((2, batches, height, width)) * 2 + 1024
    values += torch.arange(width).float() * 0.125
    values += torch.arange(batches).reshape(1, batches, 1, 1).float() * 16
    tt_input = ttnn.from_torch(values, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_input = ttnn.fill_implicit_tile_padding(tt_input, -65536.0)
    reference_input = ttnn.to_torch(tt_input).double()
    dim = (1, 2, 3)
    tolerance = 2e-4 if dtype == ttnn.float32 else 0.02 if dtype == ttnn.bfloat16 else 0.04
    for correction in (False, True):
        for torch_op, ttnn_op in ((torch.var, ttnn.var), (torch.std, ttnn.std)):
            expected = torch_op(reference_input, dim=dim, keepdim=True, correction=int(correction))
            for _ in range(2):
                output = ttnn_op(tt_input, dim=dim, keepdim=True, correction=correction)
                padded = output.cpu().to_torch_with_padded_shape()
                actual = padded[..., :1, :1].double()
                assert torch.isfinite(padded).all()
                assert_numeric_metrics(
                    expected, actual, rtol=tolerance, atol=1e-6, frobenius_threshold=tolerance, check_pcc=False
                )
                padding = padded.clone()
                padding[..., 0, 0] = 0
                assert torch.count_nonzero(padding) == 0


def test_std_w_streaming_output_padding_is_finite(device):
    torch.manual_seed(0)
    torch_input = -torch.rand((1, 1, 32, 96), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.std(input_tensor, dim=-1, correction=False)
    padded_output = output_tensor.cpu().to_torch_with_padded_shape()

    assert torch.isfinite(padded_output).all()


@pytest.mark.parametrize("ttnn_op", [ttnn.var, ttnn.std], ids=["var", "std"])
@pytest.mark.parametrize(
    "torch_dtype,ttnn_dtype",
    [
        (torch.bfloat16, ttnn.bfloat16),
        (torch.float32, ttnn.float32),
        (torch.bfloat16, ttnn.bfloat8_b),
    ],
    ids=["bf16", "fp32", "bfp8"],
)
@pytest.mark.parametrize("width", [96, 128], ids=["three_tiles", "four_tiles"])
@pytest.mark.parametrize("height", [32, 64, 160], ids=["one_tile", "retained_pair", "streaming"])
def test_std_var_hw_output_padding_is_zero(device, torch_dtype, ttnn_dtype, ttnn_op, width, height):
    torch.manual_seed(0)
    # More outputs than either supported Gen1 compute grid ensures that the
    # one-entry combined buffer and partial/output packer formats are reused.
    # Ht >= 2 also leaves retained input in unused rows of the internal mean tile.
    torch_input = -torch.rand((1, 256, height, width), dtype=torch_dtype)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    quantized_input = ttnn.to_torch(input_tensor).double()
    torch_op = torch.var if ttnn_op == ttnn.var else torch.std
    expected = torch_op(quantized_input, dim=(-2, -1), keepdim=True, correction=0)

    output_tensor = ttnn_op(input_tensor, dim=(-2, -1), keepdim=True, correction=False)
    padded_output = output_tensor.cpu().to_torch_with_padded_shape()
    padding = padded_output.clone()
    padding[..., 0, 0] = 0

    assert torch.isfinite(padded_output).all()
    assert torch.count_nonzero(padding) == 0
    assert_numeric_metrics(
        expected, padded_output[..., :1, :1].double(), rtol=0.02, atol=1e-5, frobenius_threshold=0.02, check_pcc=False
    )


@pytest.mark.parametrize("ttnn_op,torch_op", [(ttnn.var, torch.var), (ttnn.std, torch.std)], ids=["var", "std"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b], ids=["bf16", "fp32", "bfp8"])
@pytest.mark.parametrize("dim", [-1, -2], ids=["W", "H"])
@pytest.mark.parametrize("width", [33, 96, 128])
@pytest.mark.parametrize("scalar", [1.0, 2.5])
def test_std_var_w_h_output_padding_is_zero(device, ttnn_op, torch_op, dtype, dim, width, scalar):
    torch.manual_seed(17)
    # Repeated outputs exercise DST reuse, including retained-input and streaming paths.
    source = -torch.rand((1, 256, 33, width))
    input_tensor = ttnn.from_torch(source, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    quantised = ttnn.to_torch(ttnn.from_device(input_tensor)).double()
    reference = torch_op(quantised * scalar, dim=dim, correction=0, keepdim=True)
    output = ttnn_op(input_tensor, dim=dim, correction=False, keepdim=True, scalar=scalar)
    padded = output.cpu().to_torch_with_padded_shape()
    logical_h, logical_w = reference.shape[-2:]
    actual = padded[..., :logical_h, :logical_w].double()
    assert_numeric_metrics(
        reference,
        actual,
        rtol=1e-4 if dtype == ttnn.float32 else 0.03,
        atol=2e-6 if dtype == ttnn.float32 else 0.004,
        frobenius_threshold=1e-4 if dtype == ttnn.float32 else 0.03,
        check_pcc=False,
    )
    assert torch.isfinite(padded).all()
    padded[..., :logical_h, :logical_w] = 0
    assert torch.count_nonzero(padded) == 0


@pytest.mark.parametrize(
    "shape,dim",
    [
        ((32, 8192), -1),
        ((32, 16384), -1),
        ((4096, 32), -2),
        ((4097, 32), -2),
        ((12289, 32), -2),
        ((4096, 64), (-2, -1)),
        ((4097, 64), (-2, -1)),
        ((12289, 64), (-2, -1)),
        ((4097, 128), (-2, -1)),
        ((12289, 128), (-2, -1)),
    ],
    ids=[
        "W_dynamic_l1_replay",
        "W_streaming_fallback",
        "H_l1_replay",
        "H_dynamic_l1_replay",
        "H_streaming_fallback",
        "HW_l1_replay",
        "HW_dynamic_l1_replay",
        "HW_streaming_fallback",
        "HW_compact_l1_replay",
        "HW_compact_streaming_fallback",
    ],
)
def test_var_fp32_large_reduction_translation_stability(device, shape, dim):
    """Keep a long FP32 reduction stable when the variance is tiny relative to its mean."""
    torch.manual_seed(123)
    torch_input = (torch.randn(shape, dtype=torch.float32) + 1e6).contiguous()
    torch_ref = torch.var(torch_input.to(torch.float64), dim=dim, keepdim=True, correction=True)

    tt_in = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.from_device(ttnn.var(tt_in, dim=dim, keepdim=True, correction=True)))

    assert_numeric_metrics(torch_ref, actual, rtol=1e-3, atol=1e-3, frobenius_threshold=2e-3, check_pcc=False)


def test_std_var_fp32_w_l1_replay_respects_occupied_l1(device, enabled_program_cache):
    torch.manual_seed(20260731)
    # This FP32 row requires 512 KiB of replay storage.
    torch_input = (torch.randn((1, 1, 32, 4096), dtype=torch.float32) + 1e4).contiguous()
    warm_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    for ttnn_op in (ttnn.var, ttnn.std):
        warm_output = ttnn_op(warm_input, dim=-1, keepdim=True, correction=True)
        ttnn.synchronize_device(device)
        warm_output.deallocate(force=True)
    warm_input.deallocate(force=True)

    # Reuse the same operation keys after allocator state changes. Without the
    # occupied-L1 cache discriminator, this resurrects the replay programs.
    # Leave half a replay row free in each bank: enough for streaming and the
    # interleaved input, but too little for full-row replay on either architecture.
    memory_view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    replay_bytes = torch_input.numel() * torch_input.element_size()
    remaining_bytes_per_bank = replay_bytes // 2
    bf16_tile_bytes = 32 * 32 * 2
    free_bytes_per_bank = memory_view.largest_contiguous_bytes_free_per_bank
    if free_bytes_per_bank <= replay_bytes:
        pytest.skip("Initial L1 span is too small to exercise the replay-to-streaming transition")
    pressure_tiles_per_bank = (free_bytes_per_bank - remaining_bytes_per_bank) // bf16_tile_bytes
    pressure_tiles = pressure_tiles_per_bank * memory_view.num_banks
    l1_pressure = ttnn.allocate_tensor_on_device(
        ttnn.Shape((1, 1, 32, pressure_tiles * 32)),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.L1_MEMORY_CONFIG,
    )

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    free_after_pressure = ttnn.get_memory_view(device, ttnn.BufferType.L1).largest_contiguous_bytes_free_per_bank
    assert 0 < free_after_pressure < replay_bytes
    warm_cache_entries = device.num_program_cache_entries()

    for torch_op, ttnn_op in ((torch.var, ttnn.var), (torch.std, ttnn.std)):
        reference = torch_op(torch_input.to(torch.float64), dim=-1, keepdim=True, correction=1)
        actual = ttnn.to_torch(ttnn.from_device(ttnn_op(tt_input, dim=-1, keepdim=True, correction=True)))

        assert torch.isfinite(actual).all()
        assert_numeric_metrics(reference, actual, rtol=1e-3, atol=1e-3, frobenius_threshold=2e-3, check_pcc=False)

    assert device.num_program_cache_entries() > warm_cache_entries
    assert l1_pressure.is_allocated()


@pytest.mark.skipif(not is_blackhole(), reason="The near-capacity reservation is calibrated for Blackhole L1")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 1024 * 1024}], indirect=True)
def test_std_var_fp32_w_l1_replay_preserves_l1_small(device, enabled_program_cache):
    torch.manual_seed(20260910)
    torch_input = torch.randn((1, 1, 32, 8192), dtype=torch.float32)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Warm the cache while L1_SMALL is empty. Its reserved region must still be
    # excluded from the CB budget: this row would otherwise use 1 MiB of replay.
    for ttnn_op in (ttnn.var, ttnn.std):
        warm_output = ttnn_op(tt_input, dim=-1, keepdim=True, correction=True)
        ttnn.synchronize_device(device)
        warm_output.deallocate(force=True)

    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    sentinel_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1_SMALL,
        ttnn.ShardSpec(core_grid, (32, 16384), ttnn.ShardOrientation.ROW_MAJOR),
    )
    sentinel_reference = torch.full((1, 1, 32, 16384), 3.25, dtype=torch.bfloat16)
    sentinel = ttnn.from_torch(
        sentinel_reference,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sentinel_memory_config,
    )

    for _ in range(2):
        for torch_op, ttnn_op in ((torch.var, ttnn.var), (torch.std, ttnn.std)):
            reference = torch_op(torch_input.to(torch.float64), dim=-1, keepdim=True, correction=1)
            actual = ttnn.to_torch(ttnn_op(tt_input, dim=-1, keepdim=True, correction=True))
            assert_numeric_metrics(reference, actual, rtol=1e-3, atol=1e-3, frobenius_threshold=2e-3, check_pcc=False)
            torch.testing.assert_close(ttnn.to_torch(sentinel), sentinel_reference, rtol=0, atol=0)


@pytest.mark.parametrize("correction", [False, True])
# 10529 = 32 * 329 + 1: partial tail leaf, 8 carry levels, and 3 cross-level finalize_tree
# merges, which neither 16385 (512 leaves) nor 131072 (4096 leaves) exercised, since both had a
# single-bit leaf count. Detects a re-widened centered-moment block by ~43x the 1% tolerance.
@pytest.mark.parametrize("width", [10529], ids=["partial_leaf_uneven_tree"])
@pytest.mark.parametrize("torch_dtype,ttnn_dtype", [(torch.bfloat16, ttnn.bfloat16), (torch.float32, ttnn.float32)])
@pytest.mark.parametrize("dim", [-1, -2, (-2, -1)], ids=["W", "H", "HW"])
def test_std_var_wide_low_variance(device, torch_dtype, ttnn_dtype, width, correction, dim):
    # The HW writer combines one equal-count partial per column. For sufficiently
    # wide inputs, directly subtracting the first and second moments of the partial
    # means can round to a negative M2 even though the input is non-constant.
    torch_input = torch.full((1, 1, 32, width), 1.1015625, dtype=torch_dtype)
    torch_input[:, :, :, 0] = 0.0
    # The first sample is an unrepresentative anchor; nearly all samples are 1.1015625.
    # Transpose for H so that the long reduction sees the same sample sequence.
    if dim == -2:
        torch_input = torch_input.transpose(-2, -1).contiguous()

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    for torch_op, ttnn_op in ((torch.var, ttnn.var), (torch.std, ttnn.std)):
        reference = torch_op(torch_input.to(torch.float64), dim=dim, keepdim=True, correction=int(correction))
        output = ttnn_op(tt_input, dim=dim, keepdim=True, correction=correction)
        actual = ttnn.to_torch(ttnn.from_device(output)).to(torch.float64)

        assert torch.isfinite(actual).all()
        assert_numeric_metrics(reference, actual, rtol=0.01, atol=1e-15, frobenius_threshold=0.01)
