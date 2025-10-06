# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
import math

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import is_blackhole, comp_pcc


def skip_welford_blackhole(use_welford):
    return pytest.mark.skipif(
        use_welford and is_blackhole(), reason="Welford's algorithm is not supported on Blackhole"
    )


import torch

import torch


def assert_tensor_allclose(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6):
    """
    Robust allclose assertion for PyTorch tensors.
    Supports float32, float64, and bfloat16.

    Uses combined ULP + abs/rel tolerance.
    Near zero, abs tolerance dominates (so ULP doesn't blow up).
    """

    if a.shape != b.shape:
        raise AssertionError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if a.dtype != b.dtype:
        raise AssertionError(f"Dtype mismatch: {a.dtype} vs {b.dtype}")
    if a.dtype not in (torch.float32, torch.float64, torch.bfloat16):
        raise TypeError(f"Only float32/float64/bfloat16 supported, got {a.dtype}")

    if not torch.equal(torch.isnan(a), torch.isnan(b)):
        raise AssertionError("NaN mismatch between tensors")
    if not torch.equal(torch.isinf(a), torch.isinf(b)):
        raise AssertionError("Inf mismatch between tensors")

    # --- dtype-specific defaults ---
    if a.dtype == torch.float64:
        max_ulp = 2
        base_eps = 1e-12
        bits = 64
    elif a.dtype == torch.float32:
        max_ulp = 4
        base_eps = 1e-6
        bits = 32
    elif a.dtype == torch.bfloat16:
        max_ulp = 16
        base_eps = 1e-3
        bits = 16

    eff_eps = max(eps, base_eps)
    atol = eff_eps
    rtol = eff_eps

    # --- ULP distance ---
    if a.dtype == torch.float32:
        a_int, b_int = a.view(torch.int32), b.view(torch.int32)
    elif a.dtype == torch.float64:
        a_int, b_int = a.view(torch.int64), b.view(torch.int64)
    elif a.dtype == torch.bfloat16:
        a_int, b_int = a.view(torch.int16).to(torch.int32), b.view(torch.int16).to(torch.int32)

    def to_ordered(x, bits):
        return torch.where(x < 0, ~(x) + 1 << (bits - 1), x)

    ulp_diff = (to_ordered(a_int, bits) - to_ordered(b_int, bits)).abs()

    # --- absolute/relative check ---
    diff = (a - b).abs().to(torch.float32)
    tol = atol + rtol * b.abs().to(torch.float32)

    # --- final decision ---
    ok_mask = (ulp_diff <= max_ulp) | (diff <= tol)
    if not torch.all(ok_mask):
        # Get multi-dimensional indices of failing elements
        bad_idx = (~ok_mask).nonzero(as_tuple=True)

        failing_a = a[bad_idx]
        failing_b = b[bad_idx]
        failing_diffs = diff[bad_idx]
        failing_tols = tol[bad_idx]
        failing_ulps = ulp_diff[bad_idx]

        max_diff = failing_diffs.max().item()
        max_tol = failing_tols.max().item()
        max_ulp_found = failing_ulps.max().item()

        # Print a small sample for debug
        n_show = min(10, failing_a.numel())
        print("---- Failing elements (showing up to 10) ----")
        for i in range(n_show):
            idx = tuple(dim[i].item() for dim in bad_idx)
            print(
                f"Index {idx}: a={failing_a[i].item()}, b={failing_b[i].item()}, "
                f"abs diff={failing_diffs[i].item()}, tol={failing_tols[i].item()}, "
                f"ULP diff={failing_ulps[i].item()}"
            )

        raise AssertionError(
            f"{failing_a.numel()} elements failed. "
            f"Max abs diff={max_diff} (tol={max_tol}), "
            f"Max ULP={max_ulp_found} (limit={max_ulp}), dtype={a.dtype}"
        )


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("use_welford", [True, False])
@skip_welford_blackhole("'use_welford'")
def test_layer_norm(device, h, w, use_welford):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w])

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    output_tensor = ttnn.layer_norm(input_tensor, program_config=program_config)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("use_welford", [True, False])
@skip_welford_blackhole("'use_welford'")
def test_layer_norm_with_weight_and_bias(device, h, w, use_welford):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)
    torch_bias = torch.rand((w,), dtype=torch.bfloat16)

    torch_output_tensor = torch.nn.functional.layer_norm(
        torch_input_tensor, normalized_shape=[w], weight=torch_weight, bias=torch_bias
    )

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    output_tensor = ttnn.layer_norm(input_tensor, weight=weight, bias=bias, program_config=program_config)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("use_welford", [True, False])
@skip_welford_blackhole("'use_welford'")
def test_layer_norm_with_weight_bias_and_residual_input(device, h, w, use_welford):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_residual_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)
    torch_bias = torch.rand((w,), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.layer_norm(
        torch_input_tensor + torch_residual_input_tensor, normalized_shape=[w], weight=torch_weight, bias=torch_bias
    )

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    residual_input_tensor = ttnn.from_torch(torch_residual_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    output_tensor = ttnn.layer_norm(
        input_tensor,
        residual_input_tensor=residual_input_tensor,
        weight=weight,
        bias=bias,
        program_config=program_config,
    )
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9997)


@pytest.mark.parametrize("h", [2])
@pytest.mark.parametrize("w", [512])
def test_layer_norm_with_tile_layout(device, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((1, h, w), dtype=torch.bfloat16)
    torch_weight = torch.ones(w, dtype=torch.bfloat16)
    torch_bias = torch.zeros(w, dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.layer_norm(
        torch_input_tensor,
        (w,),
        torch_weight,
        torch_bias,
    )

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)
    input_tensor = ttnn.to_device(input_tensor, device)

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

    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)


@pytest.mark.parametrize("h", [1024, 2080])
@pytest.mark.parametrize("w", [3200, 4128])
@pytest.mark.parametrize("use_welford", [True, False])
@skip_welford_blackhole("'use_welford'")
def test_large_layer_norm(device, h, w, use_welford):
    if h == 2080:
        pytest.skip("Bug, see https://github.com/tenstorrent/tt-metal/issues/27126")

    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w))
    torch_output_tensor = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w])

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    output_tensor = ttnn.layer_norm(input_tensor, program_config=program_config)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9998)


@pytest.mark.parametrize("h", [2048])
@pytest.mark.parametrize("w", [4096])
@pytest.mark.parametrize("use_welford", [True, False])
@skip_welford_blackhole("'use_welford'")
def test_large_layer_norm_with_weight_and_bias(device, h, w, use_welford):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)
    torch_bias = torch.rand((w,), dtype=torch.bfloat16)

    torch_output_tensor = torch.nn.functional.layer_norm(
        torch_input_tensor, normalized_shape=[w], weight=torch_weight, bias=torch_bias
    )

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    output_tensor = ttnn.layer_norm(input_tensor, weight=weight, bias=bias, program_config=program_config)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.97)


@pytest.mark.parametrize("h", [2048])
@pytest.mark.parametrize("w", [4096])
@pytest.mark.parametrize("use_welford", [True, False])
@skip_welford_blackhole("'use_welford'")
def test_large_layer_norm_with_weight(device, h, w, use_welford):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)

    torch_output_tensor = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w], weight=torch_weight)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    output_tensor = ttnn.layer_norm(input_tensor, weight=weight, program_config=program_config)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.97)


@pytest.mark.parametrize("h", [2048])
@pytest.mark.parametrize("w", [4096])
@pytest.mark.parametrize("use_welford", [True, False])
@skip_welford_blackhole("'use_welford'")
def test_large_layer_norm_with_bias(device, h, w, use_welford):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_bias = torch.rand((w,), dtype=torch.bfloat16)

    torch_output_tensor = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w], bias=torch_bias)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    output_tensor = ttnn.layer_norm(input_tensor, bias=bias, program_config=program_config)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.97)


@pytest.mark.parametrize("h, w", [(2048, 2048)])
@pytest.mark.parametrize("legacy_reduction", [True, False])
@pytest.mark.parametrize("legacy_rsqrt", [True, False])
def test_large_layer_norm_with_legacy_reduction_and_rsqrt(device, h, w, legacy_reduction, legacy_rsqrt):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_bias = torch.rand((w,), dtype=torch.bfloat16)

    torch_output_tensor = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[w], bias=torch_bias)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
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
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.97)


@pytest.mark.parametrize("h", [32, 1024])
@pytest.mark.parametrize("w", [2880, 4096])
@pytest.mark.parametrize("use_welford", [True, False])
@skip_welford_blackhole("'use_welford'")
def test_large_layer_norm_with_weight_bias_and_residual_input(device, h, w, use_welford):
    if not use_welford:
        pytest.skip("Low PCC, see https://github.com/tenstorrent/tt-metal/issues/27291")

    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_residual_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)
    torch_bias = torch.rand((w,), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.layer_norm(
        torch_input_tensor + torch_residual_input_tensor, normalized_shape=[w], weight=torch_weight, bias=torch_bias
    )

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    residual_input_tensor = ttnn.from_torch(torch_residual_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=use_welford)
    output_tensor = ttnn.layer_norm(
        input_tensor,
        residual_input_tensor=residual_input_tensor,
        weight=weight,
        bias=bias,
        program_config=program_config,
    )
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9997)


@pytest.mark.parametrize("use_welford", [True])
@pytest.mark.parametrize("two_stage", [False])
@pytest.mark.parametrize("tensor_type", ["ascending"])  # , "repeating", "random"])
@pytest.mark.parametrize("large", [True])  # , False])
@pytest.mark.parametrize("dtype", [torch.bfloat16])  # , torch.float32])
def test_layer_norm_sharded(device, use_welford, two_stage, tensor_type, large, dtype):
    torch.manual_seed(0)
    tile_height = 32
    tile_width = 32

    # Test parameters
    if two_stage:
        # Two-stage
        if large:
            tensor_height = 32 * 8
            tensor_width = 32 * 16
            block_wt = 2
        else:
            tensor_height = 32 * 4
            tensor_width = 32 * 8
            block_wt = 1
        shard_grid_rows = 2
        shard_grid_cols = 4
        block_ht = tensor_height // tile_height
        subblock_w = 1
        shard_height = tensor_height
        shard_width = tensor_width // (shard_grid_cols * shard_grid_rows)
        mem_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    else:
        # Block-sharded
        if large:
            tensor_height = 32 * 8
            tensor_width = 32 * 8
        else:
            tensor_height = 32 * 2
            tensor_width = 32 * 2
        shard_grid_rows = 2
        shard_grid_cols = 2
        block_wt = tensor_width // tile_width // shard_grid_cols
        block_ht = tensor_height // tile_height // shard_grid_rows
        subblock_w = 1
        shard_height = tensor_height // shard_grid_rows
        shard_width = tensor_width // shard_grid_cols
        mem_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    if tensor_type == "repeating":
        torch_input_tensor = torch.arange(tensor_width).repeat(tensor_height, 1).to(dtype)
    elif tensor_type == "ascending":
        torch_input_tensor = torch.arange(tensor_height * tensor_width).reshape(tensor_height, tensor_width).to(dtype)
    elif tensor_type == "random":
        torch_input_tensor = torch.rand((tensor_height, tensor_width), dtype=dtype)

    # torch_input_tensor = torch.rand((tensor_height, tensor_width), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.layer_norm(torch_input_tensor, normalized_shape=[tensor_width])

    # Tensor dimensions in tiles (padded)
    Mt = (tensor_height + tile_height - 1) // tile_height
    Kt = (tensor_width + tile_width - 1) // tile_width

    # Block dimensions in elements
    block_h = block_ht * tile_height
    block_w = block_wt * tile_width

    # Check mcast_1d condition
    mcast_1d = tensor_height == block_h

    # All-to-all worker calculations
    num_blocks = shard_grid_cols
    num_rows_per_all_to_all_worker = (block_ht + num_blocks - 1) // num_blocks
    num_cores_all_to_all = (block_ht + num_rows_per_all_to_all_worker - 1) // num_rows_per_all_to_all_worker

    # print(f"Tensor dimensions: {tensor_height}x{tensor_width}")
    # print(f"Shard grid: {shard_grid_rows}x{shard_grid_cols}")
    # print(f"Block dimensions: {block_ht}x{block_wt} tiles = {block_h}x{block_w} elements")
    # print(f"Mt={Mt}, Kt={Kt}")
    # print(f"mcast_1d={mcast_1d}")
    # print(f"num_cores_all_to_all={num_cores_all_to_all}")

    # Create shard spec
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(shard_grid_cols - 1, shard_grid_rows - 1),
                )
            }
        ),
        [shard_height, shard_width],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    print(f"Shard spec: {shard_spec}")

    # Create memory config with sharding
    memory_config = ttnn.MemoryConfig(memory_layout=mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=shard_spec)

    # Convert to TTNN tensor
    input_ttnn = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=memory_config,
    )

    print(f"Input tensor: {input_ttnn}")

    # Create output memory config (same sharding as input)
    output_memory_config = ttnn.MemoryConfig(
        memory_layout=mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=shard_spec
    )

    # Run layernorm
    output_ttnn = ttnn.layer_norm(
        input_ttnn,
        memory_config=output_memory_config,
        program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            subblock_w=subblock_w,
            block_h=block_ht,
            block_w=block_wt,
            use_welford=use_welford,
            inplace=False,
        ),
    )
    output_ttnn = ttnn.to_layout(output_ttnn, ttnn.ROW_MAJOR_LAYOUT)
    output_ttnn = ttnn.from_device(output_ttnn)
    output_ttnn = ttnn.to_torch(output_ttnn)

    # Print some stuff
    # Compute means and vars in float32, like tt does internally
    torch_input_tensor = torch_input_tensor.to(torch.float32)
    if two_stage:
        # # Print the means and vars of each (shard_grid cols * shard_grid rows) width shard across the width dim
        for i in range(shard_grid_cols * shard_grid_rows):
            if i != 1:
                continue
            print(f"Shard {i} means: {torch_input_tensor[:, i * shard_width:(i + 1) * shard_width].mean(dim=1)}")
            print(
                f"Shard {i} vars: {torch_input_tensor[:, i * shard_width:(i + 1) * shard_width].var(dim=1, correction=0)}"
            )

        # # Print the mean of the first shard_grid_cols shards across the width dim
        # print(
        #     f"First {shard_grid_cols} shards means: {torch_input_tensor[:, :shard_grid_cols * shard_width].mean(dim=1)}"
        # )
        # # Print the mean of the last shard_grid_cols shards across the width dim
        # print(
        #     f"Last {shard_grid_cols} shards means: {torch_input_tensor[:, -shard_grid_cols * shard_width:].mean(dim=1)}"
        # )
        # # Print means of first tile_height rows across the first half of the tensor width
        # print(
        #     f"First {tile_height} rows means (first half): {torch_input_tensor[:tile_height, :tensor_width//2].mean(dim=1)}"
        # )
        # # Print vars of first tile_height rows across the first half of the tensor width
        # print(
        #     f"First {tile_height} rows vars (first half): {torch_input_tensor[:tile_height, :tensor_width//2].var(dim=1, correction=0)}"
        # )
        # # Print means of first tile_height rows across the second half of the tensor width
        # print(
        #     f"First {tile_height} rows means (second half): {torch_input_tensor[:tile_height, tensor_width//2:].mean(dim=1)}"
        # )
        # # Print vars of first tile_height rows across the second half of the tensor width
        # print(
        #     f"First {tile_height} rows vars (second half): {torch_input_tensor[:tile_height, tensor_width//2:].var(dim=1, correction=0)}"
        # )
        # Print means of first tile_height rows across the entire tensor width
        print(f"First {tile_height} rows means (entire width): {torch_input_tensor[:tile_height, :].mean(dim=1)}")
        # # Print vars of first tile_height rows across the entire tensor width
        # print(
        #     f"First {tile_height} rows vars (entire width): {torch_input_tensor[:tile_height, :].var(dim=1, correction=0)}"
        # )
        # # Print means of second tile_height rows across the entire tensor width
        # print(
        #     f"Second {tile_height} rows means (entire width): {torch_input_tensor[tile_height:, :].mean(dim=1)}"
        # )
        # # Print vars of second tile_height rows across the entire tensor width
        # print(
        #     f"Second {tile_height} rows vars (entire width): {torch_input_tensor[tile_height:, :].var(dim=1, correction=0)}"
        # )
    else:
        # Print the partial means and varsof each (X,Y) shard of the input tensor
        for i in range(shard_grid_rows):
            for j in range(shard_grid_cols):
                print(
                    f"Shard {j},{i} means: {torch_input_tensor[i * shard_height:(i + 1) * shard_height, j * shard_width:(j + 1) * shard_width].mean(dim=1)}"
                )
                print(
                    f"Shard {j},{i} vars: {torch_input_tensor[i * shard_height:(i + 1) * shard_height, j * shard_width:(j + 1) * shard_width].var(dim=1, correction=0)}"
                )

        # Print the mean of the first tile_height rows across the entire tensor width
        print(f"First {tile_height} rows means (entire width): {torch_input_tensor[:tile_height, :].mean(dim=1)}")
        # Print the vars of the first tile_height rows across the entire tensor width
        print(
            f"First {tile_height} rows vars (entire width): {torch_input_tensor[:tile_height, :].var(dim=1, correction=0)}"
        )

    # xmm = torch_input_tensor - torch_input_tensor.mean(dim=1).unsqueeze(1)
    # inv_sqrt_var = 1 / torch.sqrt((torch_input_tensor + 1e-12).var(dim=1, correction=0)).unsqueeze(1)
    # means = torch_input_tensor.mean(dim=1)
    # print(f"Input tensor: {torch_input_tensor}")
    # print(f"Means: {means}")
    # print(f"Variances: {torch_input_tensor.var(dim=1, correction=0)}")
    # print(f"1/sqrt(var+eps): {inv_sqrt_var}")
    # print(f"x - mean: {xmm}")
    # print(f"xmm * 1/sqrt(var+eps): {xmm * inv_sqrt_var}")

    print(f"Torch output tensor: {torch_output_tensor}")
    print(f"TTNN output tensor: {output_ttnn}")

    # Print ULP differences
    assert_tensor_allclose(torch_output_tensor, output_ttnn, 1e-6)

    # print(f"Torch output tensor: {torch_output_tensor}")
    # print(f"TTNN output tensor: {output_ttnn}")
    print(f"PCC: {comp_pcc(torch_output_tensor, output_ttnn)[1]}")

    # torch.set_printoptions(profile="full")  # Adjust threshold as needed
    # with open("/localdev/rmiller/scratch/torch_input.txt", "w") as f:
    #     f.write(str(torch_input_tensor))
    # with open("/localdev/rmiller/scratch/torch_output.txt", "w") as f:
    #     # Write the string representation of the tensor to the file
    #     f.write(str(torch_output_tensor[:, :10]))
    # with open("/localdev/rmiller/scratch/means.txt", "w") as f:
    #     f.write(str(means))
    # with open(f"/localdev/rmiller/scratch/ttnn_output_{use_welford}.txt", "w") as f:
    #     # Write the string representation of the tensor to the file
    #     f.write(str(output_ttnn[:, :10]))
    # with open("/localdev/rmiller/scratch/xmm.txt", "w") as f:
    #     f.write(str(xmm[:, 0]))
    # assert_with_pcc(torch_output_tensor, output_ttnn, 0.9998)
