# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

pytestmark = pytest.mark.use_module_device

import math
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_numeric_metrics
from models.common.utility_functions import torch_random, comp_pcc, is_blackhole


def _create_compute_kernel_config(fp32_dest_acc_en, math_fidelity=ttnn.MathFidelity.HiFi4):
    compute_config_class = (
        ttnn.types.BlackholeComputeKernelConfig if is_blackhole() else ttnn.WormholeComputeKernelConfig
    )
    return compute_config_class(
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=False,
    )


def _flatten_to_rows(tensor):
    if tensor.ndim == 0:
        return tensor.reshape(1, 1)
    if tensor.ndim == 1:
        return tensor.reshape(1, tensor.shape[0])
    return tensor.reshape(-1, tensor.shape[-1])


def _compute_max_relative_error(actual, golden):
    abs_diff = torch.abs(actual - golden)
    denom = torch.abs(golden)
    finite_mask = torch.isfinite(abs_diff) & torch.isfinite(denom)
    nonzero_mask = finite_mask & (denom != 0)

    rel_error = torch.zeros_like(abs_diff, dtype=torch.float32)
    rel_error[nonzero_mask] = (abs_diff[nonzero_mask] / denom[nonzero_mask]).to(torch.float32)

    zero_mask = finite_mask & (denom == 0)
    rel_error[zero_mask] = torch.where(abs_diff[zero_mask] == 0, 0.0, math.inf)
    return torch.max(rel_error).item()


def _get_max_relative_error_details(actual, golden):
    abs_diff = torch.abs(actual - golden)
    denom = torch.abs(golden)
    finite_mask = torch.isfinite(abs_diff) & torch.isfinite(denom)
    nonzero_mask = finite_mask & (denom != 0)

    rel_error = torch.zeros_like(abs_diff, dtype=torch.float32)
    rel_error[nonzero_mask] = (abs_diff[nonzero_mask] / denom[nonzero_mask]).to(torch.float32)

    zero_mask = finite_mask & (denom == 0)
    rel_error[zero_mask] = torch.where(abs_diff[zero_mask] == 0, 0.0, math.inf)

    flat_index = torch.argmax(rel_error.reshape(-1)).item()
    unraveled_index = tuple(int(i) for i in torch.unravel_index(torch.tensor(flat_index), rel_error.shape))

    return {
        "index": unraveled_index,
        "relative_error": rel_error[unraveled_index].item(),
        "golden": golden[unraveled_index].item(),
        "output": actual[unraveled_index].item(),
        "abs_error": abs_diff[unraveled_index].item(),
    }


def _run_mean_precision_diagnostics(
    device,
    input_dtype,
    torch_input_dtype,
    fp32_dest_acc_en,
    input_shape,
    dim,
    label,
    math_fidelity=ttnn.MathFidelity.HiFi4,
):
    torch.manual_seed(0)

    base_input = torch_random(input_shape, -100, 100, dtype=torch.float32)
    torch_input_tensor = base_input.to(torch_input_dtype)
    torch_golden = torch.mean(torch_input_tensor.to(torch.float32), dim=dim, keepdim=False)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=input_dtype)
    ttnn.fill_implicit_tile_padding(input_tensor, 42)

    output_tensor = ttnn.mean(
        input_tensor,
        dim=dim,
        keepdim=False,
        compute_kernel_config=_create_compute_kernel_config(fp32_dest_acc_en, math_fidelity),
    )
    torch_output = ttnn.to_torch(output_tensor).to(torch.float32)

    assert list(torch_output.shape) == list(torch_golden.shape)
    assert torch.all(torch.isfinite(torch_output))

    _, pcc = comp_pcc(torch_golden, torch_output, pcc=0.0)
    max_abs_error = torch.max(torch.abs(torch_output - torch_golden)).item()
    max_relative_error = _compute_max_relative_error(torch_output, torch_golden)
    max_relative_error_details = _get_max_relative_error_details(torch_output, torch_golden)

    output_rows = _flatten_to_rows(torch_output)
    golden_rows = _flatten_to_rows(torch_golden)
    first_row_output = output_rows[0].tolist()
    first_row_golden = golden_rows[0].tolist()

    print(
        f"{label} "
        f"input_dtype={input_dtype} "
        f"fp32_dest_acc_en={fp32_dest_acc_en} "
        f"math_fidelity={math_fidelity} "
        f"pcc={pcc:.9f} "
        f"max_abs_error={max_abs_error:.9e} "
        f"max_relative_error={max_relative_error:.9e}"
    )
    print(f"first_row_output={first_row_output}")
    print(f"first_row_golden={first_row_golden}")
    print(
        "max_relative_error_details="
        f"index={max_relative_error_details['index']} "
        f"golden={max_relative_error_details['golden']} "
        f"output={max_relative_error_details['output']} "
        f"abs_error={max_relative_error_details['abs_error']} "
        f"relative_error={max_relative_error_details['relative_error']}"
    )


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("h", [32, 64, 41, 37])
@pytest.mark.parametrize("w", [32, 64, 31, 63])
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("keepdim", [True, False])
def test_mean(device, batch_size, h, w, dim, keepdim):
    torch.manual_seed(0)

    torch_input_tensor = torch_random((batch_size, h, w), -1, 1, dtype=torch.bfloat16)
    torch_output_tensor = torch.mean(torch_input_tensor, dim=dim, keepdim=keepdim, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.fill_implicit_tile_padding(input_tensor, 42)  # garbage padding to test that mean removes it

    output_tensor = ttnn.mean(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_torch(output_tensor)

    # test for equivalance
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.118,
        atol=0.002,
        frobenius_threshold=0.005,
        check_ulp=False if dim == -2 else True,
    )


@pytest.mark.parametrize("shape", [(2, 3, 4, 5), (7, 17, 41, 31)])
@pytest.mark.parametrize("dim", [0, 1, 2, 3, [0, 1], [2, 3], [0, 1, 2]])
@pytest.mark.parametrize("keepdim", [True, False])
def test_mean_scaling(device, shape, dim, keepdim):
    """Use assert_allclose with ones() to test that mean's scaling factor is
    computed correctly.
    """
    torch_input_tensor = torch.ones(shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.mean(torch_input_tensor, dim=dim, keepdim=keepdim, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.fill_implicit_tile_padding(input_tensor, 42)  # garbage padding to test that mean removes it

    output_tensor = ttnn.mean(input_tensor, dim=dim, keepdim=keepdim)
    output_tensor = ttnn.to_torch(output_tensor)

    # test for equivalance
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.004,
        atol=0.004,
        frobenius_threshold=0.004,
        check_ulp=True,
    )


@pytest.mark.parametrize("shape", [(2, 3, 4, 5), (7, 17, 41, 31)])
@pytest.mark.parametrize("dim", [0, 1, 2, 3, [0, 1], [2, 3], [0, 1, 2]])
@pytest.mark.parametrize("scalar", [2.0])
def test_mean_scaling_factor(device, shape, dim, scalar):
    torch_input_tensor = torch.ones(shape, dtype=torch.bfloat16)
    torch_output_tensor = torch.mean(torch_input_tensor, dim=dim, dtype=torch.bfloat16)
    torch_output_tensor = torch_output_tensor * scalar

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.fill_implicit_tile_padding(input_tensor, 42)  # garbage padding to test that mean removes it

    output_tensor = ttnn.mean(input_tensor, dim=dim, scalar=scalar)
    output_tensor = ttnn.to_torch(output_tensor)

    # test for equivalance
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.9999,
        rtol=0.004,
        atol=0.008,
        frobenius_threshold=0.004,
        check_ulp=True,
    )


@pytest.mark.parametrize("mem_config", [None, ttnn.DRAM_MEMORY_CONFIG, "block"])
@pytest.mark.parametrize("keepdim", [True, False])
def test_mean_shard(device, mem_config, keepdim):
    if mem_config is None and not keepdim:
        pytest.skip("Skipping because reshape does not work in this scenario. Issue #35145")
    torch_input_tensor = torch.randn(1, 1024, 160, dtype=torch.bfloat16)
    block_sharded_config = ttnn.create_sharded_memory_config(
        shape=(1, 1024, 160),
        core_grid=ttnn.CoreGrid(x=5, y=8),
        strategy=ttnn.ShardStrategy.BLOCK,
        use_height_and_width_as_shard_shape=False,
    )
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=block_sharded_config,
    )

    memory_config = block_sharded_config if mem_config == "block" else mem_config
    output_tensor = ttnn.mean(
        input_tensor,
        dim=-1,
        keepdim=keepdim,
        memory_config=memory_config,
    )
    tt_output_torch = ttnn.to_torch(output_tensor)
    torch_output = torch.mean(torch_input_tensor, -1, keepdim)
    # test for equivalance
    assert_numeric_metrics(
        torch_output,
        tt_output_torch,
        pcc_threshold=0.999,
        rtol=0.610,
        atol=0.002,
        frobenius_threshold=0.005,
    )


@pytest.mark.parametrize(
    "input_dtype, torch_input_dtype",
    [
        pytest.param(ttnn.bfloat16, torch.bfloat16, id="input_bf16"),
        pytest.param(ttnn.float32, torch.float32, id="input_fp32"),
    ],
)
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True], ids=["fp32acc_off", "fp32acc_on"])
def test_mean_reduce_row_precision_diagnostics(device, input_dtype, torch_input_dtype, fp32_dest_acc_en):
    _run_mean_precision_diagnostics(
        device=device,
        input_dtype=input_dtype,
        torch_input_dtype=torch_input_dtype,
        fp32_dest_acc_en=fp32_dest_acc_en,
        input_shape=(1, 4, 32, 64),
        dim=-1,
        label="reduce_row_avg_diagnostics",
        math_fidelity=ttnn.MathFidelity.HiFi4,
    )


@pytest.mark.parametrize(
    "input_dtype, torch_input_dtype",
    [
        pytest.param(ttnn.bfloat16, torch.bfloat16, id="input_bf16"),
        pytest.param(ttnn.float32, torch.float32, id="input_fp32"),
    ],
)
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True], ids=["fp32acc_off", "fp32acc_on"])
def test_mean_reduce_row_precision_diagnostics_hifi2(device, input_dtype, torch_input_dtype, fp32_dest_acc_en):
    _run_mean_precision_diagnostics(
        device=device,
        input_dtype=input_dtype,
        torch_input_dtype=torch_input_dtype,
        fp32_dest_acc_en=fp32_dest_acc_en,
        input_shape=(1, 4, 32, 64),
        dim=-1,
        label="reduce_row_avg_diagnostics_hifi2",
        math_fidelity=ttnn.MathFidelity.HiFi2,
    )


@pytest.mark.parametrize(
    "input_dtype, torch_input_dtype",
    [
        pytest.param(ttnn.bfloat16, torch.bfloat16, id="input_bf16"),
        pytest.param(ttnn.float32, torch.float32, id="input_fp32"),
    ],
)
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True], ids=["fp32acc_off", "fp32acc_on"])
def test_mean_reduce_row_precision_diagnostics_lofi(device, input_dtype, torch_input_dtype, fp32_dest_acc_en):
    _run_mean_precision_diagnostics(
        device=device,
        input_dtype=input_dtype,
        torch_input_dtype=torch_input_dtype,
        fp32_dest_acc_en=fp32_dest_acc_en,
        input_shape=(1, 4, 32, 64),
        dim=-1,
        label="reduce_row_avg_diagnostics_lofi",
        math_fidelity=ttnn.MathFidelity.LoFi,
    )


@pytest.mark.parametrize(
    "input_dtype, torch_input_dtype",
    [
        pytest.param(ttnn.bfloat16, torch.bfloat16, id="input_bf16"),
        pytest.param(ttnn.float32, torch.float32, id="input_fp32"),
    ],
)
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True], ids=["fp32acc_off", "fp32acc_on"])
def test_mean_reduce_row_precision_diagnostics_hifi3(device, input_dtype, torch_input_dtype, fp32_dest_acc_en):
    _run_mean_precision_diagnostics(
        device=device,
        input_dtype=input_dtype,
        torch_input_dtype=torch_input_dtype,
        fp32_dest_acc_en=fp32_dest_acc_en,
        input_shape=(1, 4, 32, 64),
        dim=-1,
        label="reduce_row_avg_diagnostics_hifi3",
        math_fidelity=ttnn.MathFidelity.HiFi3,
    )


@pytest.mark.parametrize(
    "input_dtype, torch_input_dtype",
    [
        pytest.param(ttnn.bfloat16, torch.bfloat16, id="input_bf16"),
        pytest.param(ttnn.float32, torch.float32, id="input_fp32"),
    ],
)
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True], ids=["fp32acc_off", "fp32acc_on"])
@pytest.mark.parametrize(
    "math_fidelity",
    [
        pytest.param(ttnn.MathFidelity.HiFi4, id="hifi4"),
        pytest.param(ttnn.MathFidelity.HiFi3, id="hifi3"),
        pytest.param(ttnn.MathFidelity.HiFi2, id="hifi2"),
        pytest.param(ttnn.MathFidelity.LoFi, id="lofi"),
    ],
)
@pytest.mark.parametrize(
    "input_shape",
    [
        pytest.param((1, 4, 128, 512), id="128x512"),
        pytest.param((1, 8, 256, 256), id="256x256"),
    ],
)
def test_mean_reduce_row_perf(device, input_dtype, torch_input_dtype, fp32_dest_acc_en, math_fidelity, input_shape):
    torch.manual_seed(0)
    torch_input_tensor = torch_random(input_shape, -100, 100, dtype=torch_input_dtype)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=input_dtype)

    output_tensor = ttnn.mean(
        input_tensor,
        dim=-1,
        keepdim=False,
        compute_kernel_config=_create_compute_kernel_config(fp32_dest_acc_en, math_fidelity),
    )
    output_tensor = ttnn.to_torch(output_tensor)


@pytest.mark.parametrize(
    "input_dtype, torch_input_dtype",
    [
        pytest.param(ttnn.bfloat16, torch.bfloat16, id="input_bf16"),
        pytest.param(ttnn.float32, torch.float32, id="input_fp32"),
    ],
)
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True], ids=["fp32acc_off", "fp32acc_on"])
@pytest.mark.parametrize(
    "math_fidelity",
    [
        pytest.param(ttnn.MathFidelity.HiFi4, id="hifi4"),
        pytest.param(ttnn.MathFidelity.HiFi3, id="hifi3"),
        pytest.param(ttnn.MathFidelity.HiFi2, id="hifi2"),
        pytest.param(ttnn.MathFidelity.LoFi, id="lofi"),
    ],
)
@pytest.mark.parametrize(
    "input_shape, num_cores",
    [
        pytest.param((1, 4, 128, 512), 16, id="128x512"),
        pytest.param((1, 8, 256, 256), 32, id="256x256"),
    ],
)
def test_mean_reduce_row_perf_sharded(
    device, input_dtype, torch_input_dtype, fp32_dest_acc_en, math_fidelity, input_shape, num_cores
):
    torch.manual_seed(0)
    torch_input_tensor = torch_random(input_shape, -100, 100, dtype=torch_input_dtype)

    N, C, H, _ = input_shape
    total_rows = N * C * H
    assert total_rows % num_cores == 0, f"total_rows {total_rows} not divisible by num_cores {num_cores}"

    # Height shard: full width on each core, rows distributed across cores
    grid_x = min(num_cores, 8)
    grid_y = num_cores // grid_x
    shard_config = ttnn.create_sharded_memory_config(
        shape=input_shape,
        core_grid=ttnn.CoreGrid(x=grid_x, y=grid_y),
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=False,
    )

    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=input_dtype, memory_config=shard_config
    )

    output_tensor = ttnn.mean(
        input_tensor,
        dim=-1,
        keepdim=False,
        compute_kernel_config=_create_compute_kernel_config(fp32_dest_acc_en, math_fidelity),
    )
    output_tensor = ttnn.to_torch(output_tensor)
