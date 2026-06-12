# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range_batch_norm,
    compare_results_batch_norm,
)
from itertools import product
from models.common.utility_functions import comp_pcc
from tests.ttnn.utils_for_testing import assert_numeric_metrics

TEST_PADDING_VALUE = -42
pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([5, 8, 32, 32]),
        torch.Size([7, 3, 23, 23]),
        torch.Size([3, 5, 64, 120]),
        torch.Size([1, 128, 14, 14]),
        torch.Size([1, 8, 24, 42]),
    ],
)
@pytest.mark.parametrize(
    "check_mean, check_var",
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
)
@pytest.mark.parametrize("weight", [True, False])
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("eps", [1.0, 1e-05])
@pytest.mark.parametrize("momentum", [0.0, 0.1])
@pytest.mark.parametrize("testing_dtype", ["float32", "bfloat16"])
def test_batch_norm_tests(
    input_shapes, check_mean, check_var, weight, bias, eps, device, momentum, training, testing_dtype
):
    in_data, input_tensor = data_gen_with_range_batch_norm(
        input_shapes, 5, 10, device, is_input=True, testing_dtype=testing_dtype
    )
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)
    mean_data, mean_tensor = (
        data_gen_with_range_batch_norm(input_shapes, 4, 10, device, testing_dtype=testing_dtype)
        if (check_mean)
        else (None, None)
    )
    var_data, var_tensor = (
        data_gen_with_range_batch_norm(input_shapes, 4, 20, device, testing_dtype=testing_dtype)
        if (check_var)
        else (None, None)
    )
    weight_data, weight_tensor = (
        data_gen_with_range_batch_norm(input_shapes, 4, 10, device, testing_dtype=testing_dtype)
        if weight
        else (None, None)
    )
    bias_data, bias_tensor = (
        data_gen_with_range_batch_norm(input_shapes, 4, 10, device, testing_dtype=testing_dtype)
        if bias
        else (None, None)
    )

    if (not training) and ((not check_mean) or (not check_var)):
        pytest.xfail("running_mean and running_var must be defined in evaluation mode")

    tt_output_tensor_on_device = ttnn.batch_norm(
        input_tensor,
        running_mean=mean_tensor,
        running_var=var_tensor,
        training=training,
        eps=eps,
        weight=weight_tensor,
        bias=bias_tensor,
        momentum=momentum,
    )
    tt_output = ttnn.to_torch(tt_output_tensor_on_device)
    tt_updated_mean = None
    tt_updated_var = None
    if training:
        if check_mean:
            tt_updated_mean = ttnn.to_torch(mean_tensor)
        if check_var:
            tt_updated_var = ttnn.to_torch(var_tensor)

    # PyTorch 2.11+ requires running_mean and running_var to both be None or both not-None.
    # When only one stat is requested (e.g. check_mean=True, check_var=False), supply a neutral
    # dummy for the missing stat in the reference call. Assertions still only cover the stats
    # that were actually generated.
    channels = input_shapes[1]
    ref_dtype = in_data.dtype
    ref_mean_data = (
        mean_data
        if mean_data is not None
        else (torch.zeros(channels, dtype=ref_dtype) if var_data is not None else None)
    )
    ref_var_data = (
        var_data if var_data is not None else (torch.ones(channels, dtype=ref_dtype) if mean_data is not None else None)
    )

    torch_result = torch.nn.functional.batch_norm(
        input=in_data,
        running_mean=ref_mean_data,
        running_var=ref_var_data,
        weight=weight_data,
        bias=bias_data,
        training=training,
        eps=eps,
        momentum=momentum,
    )
    assert_numeric_metrics(torch_result, tt_output, pcc_threshold=0.99, rtol=0.1, atol=4.0, frobenius_threshold=0.15)
    if training:
        if check_mean:
            assert_numeric_metrics(
                ref_mean_data.view(1, channels, 1, 1),
                tt_updated_mean,
                rtol=0.1,
                atol=4.0,
                frobenius_threshold=0.15,
                check_pcc=False,
            )
        if check_var:
            assert_numeric_metrics(
                ref_var_data.view(1, channels, 1, 1),
                tt_updated_var,
                rtol=0.1,
                atol=4.0,
                frobenius_threshold=0.15,
                check_pcc=False,
            )


@pytest.mark.parametrize("eps", [1.0, 1e-05])
@pytest.mark.parametrize("channel_size", [1, 4])
@pytest.mark.parametrize("weight", [True, False])
@pytest.mark.parametrize("bias", [True, False])
def test_BN_fp32_full_value(device, channel_size, eps, weight, bias):
    input_tensor_torch = torch.full(torch.Size([3, channel_size, 64, 120]), 1, dtype=torch.float32)
    batch_mean_torch = torch.full(torch.Size([channel_size]), 0.00030171126, dtype=torch.float32)
    batch_var_torch = torch.full(torch.Size([channel_size]), 0.1262342343, dtype=torch.float32)
    weight_torch = torch.full(torch.Size([channel_size]), 0.246943565369, dtype=torch.float32) if weight else None
    bias_torch = torch.full(torch.Size([channel_size]), 0.59, dtype=torch.float32) if bias else None

    result_torch = torch.nn.functional.batch_norm(
        input=input_tensor_torch,
        running_mean=batch_mean_torch,
        running_var=batch_var_torch,
        weight=weight_torch,
        bias=bias_torch,
        eps=eps,
    )

    batch_mean_torch = batch_mean_torch.view(1, channel_size, 1, 1)
    batch_var_torch = batch_var_torch.view(1, channel_size, 1, 1)
    weight_torch = weight_torch.view(1, channel_size, 1, 1) if weight else None
    bias_torch = bias_torch.view(1, channel_size, 1, 1) if bias else None

    input_tensor_tt = ttnn.from_torch(input_tensor_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_tt = ttnn.fill_implicit_tile_padding(input_tensor_tt, TEST_PADDING_VALUE)
    batch_mean_tt = ttnn.from_torch(batch_mean_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    batch_var_tt = ttnn.from_torch(batch_var_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    weight_tt = (
        ttnn.from_torch(weight_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device) if weight else None
    )
    bias_tt = ttnn.from_torch(bias_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device) if bias else None

    result_tt = ttnn.batch_norm(
        input_tensor_tt, running_mean=batch_mean_tt, running_var=batch_var_tt, eps=eps, weight=weight_tt, bias=bias_tt
    )
    tt_out = ttnn.to_torch(result_tt)

    assert_numeric_metrics(result_torch, tt_out, pcc_threshold=0.99, rtol=1e-5, atol=1e-10, frobenius_threshold=0.01)


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([5, 8, 32, 32]),
        torch.Size([7, 3, 23, 23]),
        torch.Size([3, 5, 64, 120]),
        torch.Size([1, 8, 24, 42]),
    ],
)
@pytest.mark.parametrize(
    "check_mean, check_var",
    [
        (False, False),  # xfail case
        (True, False),  # xfail case
        (False, True),  # xfail case
        (True, True),
    ],
)
@pytest.mark.parametrize("weight", [True, False])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("eps", [0.0, 1e-05])
def test_batch_norm_fp32(
    input_shapes, check_mean, check_var, weight, bias, eps, device, training=False, testing_dtype="float32"
):
    in_data, input_tensor = data_gen_with_range_batch_norm(
        input_shapes, 5, 10, device, is_input=True, testing_dtype=testing_dtype
    )
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)
    mean_data, mean_tensor = (
        data_gen_with_range_batch_norm(input_shapes, 4, 10, device, testing_dtype=testing_dtype)
        if (check_mean)
        else (None, None)
    )
    var_data, var_tensor = (
        data_gen_with_range_batch_norm(input_shapes, 4, 20, device, testing_dtype=testing_dtype)
        if (check_var)
        else (None, None)
    )
    weight_data, weight_tensor = (
        data_gen_with_range_batch_norm(input_shapes, 4, 10, device, testing_dtype=testing_dtype)
        if weight
        else (None, None)
    )
    bias_data, bias_tensor = (
        data_gen_with_range_batch_norm(input_shapes, 4, 10, device, testing_dtype=testing_dtype)
        if bias
        else (None, None)
    )

    if (not training) and ((not check_mean) or (not check_var)):
        pytest.xfail("running_mean and running_var must be defined in evaluation mode")

    tt_output_tensor_on_device = ttnn.batch_norm(
        input_tensor,
        running_mean=mean_tensor,
        running_var=var_tensor,
        training=training,
        eps=eps,
        weight=weight_tensor,
        bias=bias_tensor,
    )
    tt_output = ttnn.to_torch(tt_output_tensor_on_device)
    # PyTorch 2.11+ requires eps > 0; substitute a negligible positive value for the reference call.
    ref_eps = eps if eps > 0 else 1e-12
    torch_result = torch.nn.functional.batch_norm(
        input=in_data,
        running_mean=mean_data,
        running_var=var_data,
        weight=weight_data,
        bias=bias_data,
        training=training,
        eps=ref_eps,
    )
    assert_numeric_metrics(torch_result, tt_output, pcc_threshold=0.99, rtol=1e-3, atol=1e-6, frobenius_threshold=0.05)


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([5, 8, 32, 32]),
        torch.Size([7, 3, 23, 23]),
        torch.Size([3, 5, 64, 120]),
        torch.Size([1, 8, 24, 42]),
    ],
)
@pytest.mark.parametrize(
    "training, check_mean, check_var",
    [
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (True, False, False),
        (False, False, False),  # xfail case
        (False, True, False),  # xfail case
        (False, False, True),  # xfail case
        (False, True, True),
    ],
)
@pytest.mark.parametrize("weight", [True, False])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("eps", [1.0, 2.34])
@pytest.mark.parametrize("momentum", [0.0, 0.5])
def test_batch_norm(input_shapes, training, check_mean, check_var, weight, bias, eps, momentum, device):
    in_data, input_tensor = data_gen_with_range_batch_norm(input_shapes, 5, 10, device, is_input=True)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)
    mean_data, mean_tensor = (
        data_gen_with_range_batch_norm(input_shapes, 4, 10, device) if (check_mean) else (None, None)
    )
    var_data, var_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 20, device) if (check_var) else (None, None)
    weight_data, weight_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 10, device) if weight else (None, None)
    bias_data, bias_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 10, device) if bias else (None, None)

    if (not training) and ((not check_mean) or (not check_var)):
        pytest.xfail("running_mean and running_var must be defined in evaluation mode")

    tt_output_tensor_on_device = ttnn.batch_norm(
        input_tensor,
        running_mean=mean_tensor,
        running_var=var_tensor,
        training=training,
        eps=eps,
        momentum=momentum,
        weight=weight_tensor,
        bias=bias_tensor,
    )
    tt_output = ttnn.to_torch(tt_output_tensor_on_device)
    tt_updated_mean = None
    tt_updated_var = None
    if training:
        if check_mean:
            tt_updated_mean = ttnn.to_torch(mean_tensor)
        if check_var:
            tt_updated_var = ttnn.to_torch(var_tensor)

    # PyTorch 2.11+ requires running_mean and running_var to both be None or both not-None.
    # When only one stat is requested (e.g. check_mean=True, check_var=False), supply a neutral
    # dummy for the missing stat in the reference call. Assertions still only cover the stats
    # that were actually generated.
    channels = input_shapes[1]
    ref_dtype = in_data.dtype
    ref_mean_data = (
        mean_data
        if mean_data is not None
        else (torch.zeros(channels, dtype=ref_dtype) if var_data is not None else None)
    )
    ref_var_data = (
        var_data if var_data is not None else (torch.ones(channels, dtype=ref_dtype) if mean_data is not None else None)
    )

    torch_result = torch.nn.functional.batch_norm(
        input=in_data,
        running_mean=ref_mean_data,
        running_var=ref_var_data,
        weight=weight_data,
        bias=bias_data,
        training=training,
        eps=eps,
        momentum=momentum,
    )
    assert_numeric_metrics(torch_result, tt_output, pcc_threshold=0.99, rtol=0.1, atol=4.0, frobenius_threshold=0.15)
    if training:
        if check_mean:
            assert_numeric_metrics(
                ref_mean_data.view(1, channels, 1, 1),
                tt_updated_mean,
                rtol=0.1,
                atol=4.0,
                frobenius_threshold=0.15,
                check_pcc=False,
            )
        if check_var:
            assert_numeric_metrics(
                ref_var_data.view(1, channels, 1, 1),
                tt_updated_var,
                rtol=0.1,
                atol=4.0,
                frobenius_threshold=0.15,
                check_pcc=False,
            )


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([3, 2, 32, 32]),
        torch.Size([1, 16, 32, 64]),
        torch.Size([4, 2, 64, 32]),
        torch.Size([1, 128, 14, 14]),
        torch.Size([2, 16, 64, 120]),
        torch.Size([1, 8, 24, 42]),
    ],
)
@pytest.mark.parametrize("mem_layout", [ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.TensorMemoryLayout.HEIGHT_SHARDED])
@pytest.mark.parametrize("prealloc_out_mem_config", [None, ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_batch_norm_program_cache_and_default(input_shapes, mem_layout, prealloc_out_mem_config, device):
    N, H, W, C = input_shapes
    in_data, input_tensor = data_gen_with_range_batch_norm(input_shapes, 5, 10, device, is_input=True)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)
    mean_data, mean_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 10, device)
    var_data, var_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 20, device)
    output_tensor = None
    if prealloc_out_mem_config is not None:
        output_tensor = ttnn.from_torch(
            torch.zeros(input_shapes, dtype=in_data.dtype),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=prealloc_out_mem_config,
        )

    grid_size = ttnn.CoreGrid(y=1, x=8)
    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_shape = N * H * W // grid_size.x, C // grid_size.y
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(mem_layout, ttnn.types.BufferType.L1, shard_spec)

    if mem_layout is not ttnn.TensorMemoryLayout.INTERLEAVED:
        pytest.xfail("Input tensors to batch norm must be interleaved")

    tt_output_tensor_on_device = ttnn.batch_norm(
        input_tensor,
        running_mean=mean_tensor,
        running_var=var_tensor,
        memory_config=sharded_mem_config,
        output=output_tensor,
    )
    tt_output = ttnn.to_torch(tt_output_tensor_on_device)
    torch_result = torch.nn.functional.batch_norm(input=in_data, running_mean=mean_data, running_var=var_data)
    assert_numeric_metrics(torch_result, tt_output, pcc_threshold=0.99, rtol=0.1, atol=4.0, frobenius_threshold=0.15)


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([3, 2, 32, 32]),
        torch.Size([1, 8, 24, 42]),
    ],
)
def test_batch_norm_qid_Default(input_shapes, device):
    N, H, W, C = input_shapes
    in_data, input_tensor = data_gen_with_range_batch_norm(input_shapes, 5, 10, device, is_input=True)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)
    mean_data, mean_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 10, device)
    var_data, var_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 20, device)

    tt_output_tensor_on_device = ttnn.batch_norm(
        input_tensor, running_mean=mean_tensor, running_var=var_tensor, queue_id=0
    )
    tt_output = ttnn.to_torch(tt_output_tensor_on_device)
    torch_result = torch.nn.functional.batch_norm(input=in_data, running_mean=mean_data, running_var=var_data)
    assert_numeric_metrics(torch_result, tt_output, pcc_threshold=0.99, rtol=0.1, atol=4.0, frobenius_threshold=0.15)


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([3, 2, 32, 32]),
        torch.Size([1, 8, 24, 42]),
    ],
)
def test_batch_norm_qid(input_shapes, device):
    N, H, W, C = input_shapes
    in_data, input_tensor = data_gen_with_range_batch_norm(input_shapes, 2, 10, device, is_input=True)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)
    mean_data, mean_tensor = data_gen_with_range_batch_norm(input_shapes, 2, 10, device)
    var_data, var_tensor = data_gen_with_range_batch_norm(input_shapes, 2, 20, device)

    tt_output_tensor_on_device = ttnn.batch_norm(input_tensor, running_mean=mean_tensor, running_var=var_tensor)
    tt_output = ttnn.to_torch(tt_output_tensor_on_device)
    torch_result = torch.nn.functional.batch_norm(input=in_data, running_mean=mean_data, running_var=var_data)
    assert_numeric_metrics(torch_result, tt_output, pcc_threshold=0.99, rtol=0.1, atol=4.0, frobenius_threshold=0.15)


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([2, 3, 120, 120]),
        torch.Size([1, 8, 24, 42]),
    ],
)
def test_batch_norm_output_Default(input_shapes, device):
    N, H, W, C = input_shapes
    _, tt_output_tensor = data_gen_with_range_batch_norm(input_shapes, 5, 10, device, is_input=True)
    in_data, input_tensor = data_gen_with_range_batch_norm(input_shapes, 5, 10, device, is_input=True)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)
    mean_data, mean_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 10, device)
    var_data, var_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 20, device)

    ttnn.batch_norm(input_tensor, running_mean=mean_tensor, running_var=var_tensor, queue_id=0, output=tt_output_tensor)
    tt_output = ttnn.to_torch(tt_output_tensor)
    torch_result = torch.nn.functional.batch_norm(input=in_data, running_mean=mean_data, running_var=var_data)
    assert_numeric_metrics(torch_result, tt_output, pcc_threshold=0.99, rtol=0.1, atol=4.0, frobenius_threshold=0.15)


@pytest.mark.parametrize(
    "input_shapes",
    [
        # Training mode PCC ordering is unreliable. Keep `input_shapes[1] >= 14` to avoid this test failure.
        torch.Size([3, 17, 47, 32]),
        torch.Size([1, 8, 24, 42]),
    ],
)
@pytest.mark.parametrize(
    "training, weight, bias",
    [
        (True, True, True),
        (True, False, False),
        (False, True, True),
        (False, False, False),
    ],
)
@pytest.mark.parametrize(
    "input_dtype, param_dtype", [("bfloat16", "bfloat16"), ("bfloat16", "float32"), ("float32", "float32")]
)
def test_batch_norm_compute_config(input_shapes, training, weight, bias, input_dtype, param_dtype, device):
    if input_dtype == "float32" and os.environ.get("TT_METAL_SIMULATOR"):
        pytest.skip(
            "Skipping float32 batch_norm compute_config on ttsim - fp16a untested functionality (ttsim-private issue #324)"
        )
    N, H, W, C = input_shapes
    torch.manual_seed(0)

    # Generate the inputs
    torch_input_tensor, tt_input_tensor = data_gen_with_range_batch_norm(
        input_shapes, 5, 10, device, is_input=True, testing_dtype=input_dtype
    )
    tt_input_tensor = ttnn.fill_implicit_tile_padding(tt_input_tensor, TEST_PADDING_VALUE)
    torch_mean_tensor, tt_mean_tensor = data_gen_with_range_batch_norm(
        input_shapes, 4, 10, device, testing_dtype=param_dtype
    )
    torch_var_tensor, tt_var_tensor = data_gen_with_range_batch_norm(
        input_shapes, 4, 20, device, testing_dtype=param_dtype
    )
    torch_weight_tensor, tt_weight_tensor = (
        data_gen_with_range_batch_norm(input_shapes, 4, 10, device, testing_dtype=param_dtype)
        if weight
        else (None, None)
    )
    torch_bias_tensor, tt_bias_tensor = (
        data_gen_with_range_batch_norm(input_shapes, 4, 10, device, testing_dtype=param_dtype) if bias else (None, None)
    )

    # Helper function to execute batch_norm for a given compute config
    # and return torch and tt tensors to compare
    def do_batch_norm_for_config(compute_config):
        tt_mean = ttnn.clone(tt_mean_tensor)
        tt_var = ttnn.clone(tt_var_tensor)

        tt_output_tensor = ttnn.batch_norm(
            input=tt_input_tensor,
            running_mean=tt_mean,
            running_var=tt_var,
            weight=tt_weight_tensor,
            bias=tt_bias_tensor,
            training=training,
            compute_kernel_config=compute_config,
        )

        torch_mean_ref = torch_mean_tensor.clone()
        torch_var_ref = torch_var_tensor.clone()
        torch_output_ref = torch.nn.functional.batch_norm(
            input=torch_input_tensor,
            running_mean=torch_mean_ref,
            running_var=torch_var_ref,
            weight=torch_weight_tensor,
            bias=torch_bias_tensor,
            training=training,
        )

        if training:
            channels = input_shapes[1]
            tt_tensors = [ttnn.to_torch(tt_output_tensor), ttnn.to_torch(tt_mean), ttnn.to_torch(tt_var)]
            torch_tensors = [
                torch_output_ref.to(tt_tensors[0].dtype),
                torch_mean_ref.view(1, channels, 1, 1).to(tt_tensors[1].dtype),
                torch_var_ref.view(1, channels, 1, 1).to(tt_tensors[2].dtype),
            ]
        else:
            tt_tensors = [ttnn.to_torch(tt_output_tensor)]
            torch_tensors = [torch_output_ref.to(tt_tensors[0].dtype)]

        return torch_tensors, tt_tensors

    def compute_pccs_for_tensors(torch_tensors, tt_tensors):
        pccs = []
        for torch_tensor, tt_tensor in zip(torch_tensors, tt_tensors):
            _, pcc = comp_pcc(torch_tensor, tt_tensor)
            pccs.append(pcc)
        return pccs

    # Execute low-accuracy groupnorm
    config_low = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
    )
    torch_tensors_low, tt_tensors_low = do_batch_norm_for_config(config_low)
    pccs_low = compute_pccs_for_tensors(torch_tensors_low, tt_tensors_low)

    # Execute high-accuracy groupnorm
    config_high = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )
    torch_tensors_high, tt_tensors_high = do_batch_norm_for_config(config_high)
    pccs_high = compute_pccs_for_tensors(torch_tensors_high, tt_tensors_high)

    print(f"pccs_low={pccs_low}, pccs_high={pccs_high}")
    assert all(high > low for high, low in zip(pccs_high, pccs_low)), (
        f"High-accuracy config should have higher PCC than low-accuracy config: "
        f"pccs_high={pccs_high}, pccs_low={pccs_low}"
    )


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([3, 5, 64, 120]),
        torch.Size([1, 8, 24, 42]),
    ],
)
@pytest.mark.parametrize("use_output_tensor", [False, True])
@pytest.mark.parametrize(
    "training, check_mean, check_var",
    [
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (True, False, False),
        # running_mean and running_var must be defined in evaluation mode:
        # (False, False, False),
        # (False, True, False),
        # (False, False, True),
        (False, True, True),
    ],
)
@pytest.mark.parametrize("weight", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("eps", [1e-05])
@pytest.mark.parametrize("momentum", [0.1])
@pytest.mark.parametrize("testing_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("testing_dtype2", ["bfloat16", "float32"])
def test_batch_norm_mixed_precision(
    input_shapes,
    use_output_tensor,
    training,
    check_mean,
    check_var,
    weight,
    bias,
    eps,
    device,
    momentum,
    testing_dtype,
    testing_dtype2,
):
    in_data, input_tensor = data_gen_with_range_batch_norm(
        input_shapes, 5, 10, device, is_input=True, testing_dtype=testing_dtype
    )
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)
    mean_data, mean_tensor = (
        data_gen_with_range_batch_norm(input_shapes, 4, 10, device, testing_dtype=testing_dtype2)
        if (check_mean)
        else (None, None)
    )
    var_data, var_tensor = (
        data_gen_with_range_batch_norm(input_shapes, 4, 20, device, testing_dtype=testing_dtype2)
        if (check_var)
        else (None, None)
    )
    weight_data, weight_tensor = (
        data_gen_with_range_batch_norm(input_shapes, 4, 10, device, testing_dtype=testing_dtype2)
        if weight
        else (None, None)
    )
    bias_data, bias_tensor = (
        data_gen_with_range_batch_norm(input_shapes, 4, 10, device, testing_dtype=testing_dtype2)
        if bias
        else (None, None)
    )

    if use_output_tensor:
        _, tt_output_tensor_on_device = data_gen_with_range_batch_norm(
            input_shapes, 0, 1, device, is_input=True, testing_dtype=testing_dtype
        )
        ttnn.batch_norm(
            input_tensor,
            running_mean=mean_tensor,
            running_var=var_tensor,
            training=training,
            eps=eps,
            weight=weight_tensor,
            bias=bias_tensor,
            momentum=momentum,
            output=tt_output_tensor_on_device,
        )
    else:
        tt_output_tensor_on_device = ttnn.batch_norm(
            input_tensor,
            running_mean=mean_tensor,
            running_var=var_tensor,
            training=training,
            eps=eps,
            weight=weight_tensor,
            bias=bias_tensor,
            momentum=momentum,
        )
    tt_output = ttnn.to_torch(tt_output_tensor_on_device)

    tt_updated_mean = None
    tt_updated_var = None
    if training:
        if check_mean:
            tt_updated_mean = ttnn.to_torch(mean_tensor)
        if check_var:
            tt_updated_var = ttnn.to_torch(var_tensor)

    # PyTorch batch_norm does not support float32 input, bfloat16 output; use float32 reference then cast back
    in_ref = in_data.float()
    mean_ref = mean_data.float() if mean_data is not None else None
    var_ref = var_data.float() if var_data is not None else None
    weight_ref = weight_data.float() if weight_data is not None else None
    bias_ref = bias_data.float() if bias_data is not None else None
    # PyTorch 2.11+ requires running_mean and running_var to both be None or both not-None.
    # Supply a neutral dummy for the missing stat; assertions still only check the stats actually
    # generated.
    channels_ref = input_shapes[1]
    if mean_ref is None and var_ref is not None:
        mean_ref_call = torch.zeros(channels_ref, dtype=torch.float32)
    else:
        mean_ref_call = mean_ref
    if var_ref is None and mean_ref is not None:
        var_ref_call = torch.ones(channels_ref, dtype=torch.float32)
    else:
        var_ref_call = var_ref
    torch_result = torch.nn.functional.batch_norm(
        input=in_ref,
        running_mean=mean_ref_call,
        running_var=var_ref_call,
        weight=weight_ref,
        bias=bias_ref,
        training=training,
        eps=eps,
        momentum=momentum,
    )
    torch_result = torch_result.to(tt_output.dtype)
    comp_BN_Output = compare_results_batch_norm([tt_output], [torch_result])
    if training:
        channels = input_shapes[1]
        if check_mean:
            mean_compare = mean_ref.view(1, channels, 1, 1).to(mean_data.dtype)
            comp_BN_running_mean = compare_results_batch_norm(
                [tt_updated_mean], [mean_compare], stats=True
            )  # Check Updated running mean
        else:
            comp_BN_running_mean = tt_updated_mean is None
        if check_var:
            var_compare = var_ref.view(1, channels, 1, 1).to(var_data.dtype)
            comp_BN_running_var = compare_results_batch_norm(
                [tt_updated_var], [var_compare], stats=True
            )  # Check Updated running var
        else:
            comp_BN_running_var = tt_updated_var is None
        comp_BN_Output = comp_BN_Output and comp_BN_running_mean and comp_BN_running_var
    assert comp_BN_Output


def test_batch_norm_aliased_running_and_affine_tensors(device):
    """Covers #41127: batch_norm must normalize before updating the running stats.

    A traced graph can pass the same tensor as both running_mean and bias (and as both running_var
    and weight). Since the running stats are updated in place, doing that update first overwrote
    weight and bias before batch_norm used them.
    """
    input_shape = torch.Size([1, 152, 24, 32])
    channels = input_shape[1]
    eps = 9.99999996e-13
    momentum = 1.0

    # Config matches the traced graph: math_fidelity=hifi4, fp32_dest_acc_en=True.
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
    )

    in_data, input_tensor = data_gen_with_range_batch_norm(input_shape, 5, 10, device, is_input=True)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

    weight_data = torch.zeros(channels, dtype=torch.bfloat16)
    bias_data = torch.ones(channels, dtype=torch.bfloat16)
    weight_tensor = ttnn.from_torch(
        weight_data.view(1, channels, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    bias_tensor = ttnn.from_torch(
        bias_data.view(1, channels, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    # Same tensor in two roles: bias_tensor as bias and running_mean, weight_tensor as weight and running_var.
    tt_output_tensor = ttnn.batch_norm(
        input_tensor,
        weight=weight_tensor,
        bias=bias_tensor,
        running_mean=bias_tensor,
        running_var=weight_tensor,
        training=True,
        eps=eps,
        momentum=momentum,
        compute_kernel_config=compute_config,
    )
    tt_output = ttnn.to_torch(tt_output_tensor)

    # PyTorch would also overwrite these buffers if we passed the same tensor twice, so the reference
    # uses separate running stat tensors that start with the same values as bias and weight.
    running_mean_ref = torch.ones(channels, dtype=torch.bfloat16)
    running_var_ref = torch.zeros(channels, dtype=torch.bfloat16)
    torch_output = torch.nn.functional.batch_norm(
        input=in_data,
        running_mean=running_mean_ref,
        running_var=running_var_ref,
        weight=weight_data,
        bias=bias_data,
        training=True,
        eps=eps,
        momentum=momentum,
    )
    # weight is all zeros, so the normalized term is zeroed out and the output collapses to the
    # bias (all ones) on both backends - the comparison is therefore exact.
    assert_numeric_metrics(torch_output, tt_output, pcc_threshold=1.0, rtol=0.0, atol=0.0, frobenius_threshold=0.0)


def test_batch_norm_program_cache_shape_collision(device):
    """Covers #46018: program cache collision in BatchNormOperation.

    BatchNormOperation::compute_program_hash keys only on operation attributes and tensor
    dtype/memory_config, not on tensor shape/layout/padded shape. Two batch_norm calls that
    differ only in shape therefore share a program cache entry, and with the program cache
    enabled the second call silently reuses the first call's program.

    This runs a sequence of distinct shapes (all sharing dtype/memory_config/eps, so they only
    differ in shape) with a freshly-cleared program cache and asserts each distinct shape adds
    its own cache entry. On buggy main the shapes collide and the cache stays at a single entry,
    so this test FAILS. Once the hash includes tensor shape/layout the cache grows by one per
    distinct shape and the test PASSES. Outputs are also validated to be numerically correct.
    """
    device.enable_program_cache()
    device.clear_program_cache()

    # Distinct shapes, all bfloat16 / TILE / DRAM-interleaved / same eps -> identical buggy hash.
    # NOTE: deliberately do NOT call fill_implicit_tile_padding here - that op adds its own
    # shape-dependent program cache entries which would mask the batch_norm collision we test.
    # With plain batch_norm, ttnn.batch_norm contributes exactly one program per distinct shape.
    shapes = [
        torch.Size([1, 8, 32, 32]),
        torch.Size([2, 16, 64, 120]),
        torch.Size([4, 32, 128, 128]),
        torch.Size([1, 128, 14, 14]),
        torch.Size([3, 2, 64, 96]),
    ]
    eps = 1e-05

    try:
        for idx, input_shapes in enumerate(shapes, start=1):
            in_data, input_tensor = data_gen_with_range_batch_norm(input_shapes, 5, 10, device, is_input=True)
            mean_data, mean_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 10, device)
            var_data, var_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 20, device)

            tt_output_tensor_on_device = ttnn.batch_norm(
                input_tensor,
                running_mean=mean_tensor,
                running_var=var_tensor,
                eps=eps,
            )
            tt_output = ttnn.to_torch(tt_output_tensor_on_device)
            torch_result = torch.nn.functional.batch_norm(
                input=in_data, running_mean=mean_data, running_var=var_data, eps=eps
            )
            assert_numeric_metrics(
                torch_result,
                tt_output,
                pcc_threshold=0.99,
                rtol=0.1,
                atol=4.0,
                frobenius_threshold=0.15,
            )

            # Each distinct shape must create a distinct program cache entry. If the hash ignores
            # shape (the #46018 bug) the entry count stays at 1 and this assertion fails.
            entries = device.num_program_cache_entries()
            assert entries == idx, (
                f"Program cache collision (#46018): after {idx} distinct shapes the cache holds "
                f"{entries} entries, expected {idx}. Shape {tuple(input_shapes)} reused a program "
                f"built for a different shape."
            )
    finally:
        device.disable_and_clear_program_cache()


def test_batch_norm_training_avoids_variance_cancellation(device):
    """Regression test for #45968: avoid bf16 E[x^2] - E[x]^2 cancellation in training mode."""

    input_shape = torch.Size([2, 64, 32, 32])
    channels = input_shape[1]
    eps = 1e-5

    torch.manual_seed(0)
    channel_offsets = torch.where(torch.arange(channels, dtype=torch.float32) % 2 == 0, 5.0, -5.0).view(
        1, channels, 1, 1
    )
    in_data = (channel_offsets + 0.1 * torch.randn(input_shape, dtype=torch.float32)).to(torch.bfloat16)

    def to_tt(tensor):
        return ttnn.from_torch(tensor.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    input_tensor = to_tt(in_data)
    input_tensor = ttnn.fill_implicit_tile_padding(input_tensor, TEST_PADDING_VALUE)

    running_mean_tensor = to_tt(torch.zeros(1, channels, 1, 1))
    running_var_tensor = to_tt(torch.ones(1, channels, 1, 1))
    weight_tensor = to_tt(torch.ones(1, channels, 1, 1))
    bias_tensor = to_tt(torch.zeros(1, channels, 1, 1))

    tt_output_tensor = ttnn.batch_norm(
        input_tensor,
        running_mean=running_mean_tensor,
        running_var=running_var_tensor,
        weight=weight_tensor,
        bias=bias_tensor,
        training=True,
        eps=eps,
    )
    tt_output = ttnn.to_torch(tt_output_tensor).float()
    tt_running_var = ttnn.to_torch(running_var_tensor)

    torch_output = torch.nn.functional.batch_norm(
        input=in_data.float(),
        running_mean=None,
        running_var=None,
        weight=torch.ones(channels),
        bias=torch.zeros(channels),
        training=True,
        eps=eps,
    )

    assert torch.isfinite(tt_output).all()
    assert torch.isfinite(tt_running_var).all()
    assert_numeric_metrics(torch_output, tt_output, pcc_threshold=0.99, rtol=0.1, atol=4.0, frobenius_threshold=0.15)
