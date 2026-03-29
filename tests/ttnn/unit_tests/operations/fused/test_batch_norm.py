# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range_batch_norm,
    compare_results_batch_norm,
)
from itertools import product
from models.common.utility_functions import comp_pcc

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([5, 8, 32, 32]),
        torch.Size([7, 3, 23, 23]),
        torch.Size([3, 5, 64, 120]),
        torch.Size([1, 128, 14, 14]),
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

    torch_result = torch.nn.functional.batch_norm(
        input=in_data,
        running_mean=mean_data,
        running_var=var_data,
        weight=weight_data,
        bias=bias_data,
        training=training,
        eps=eps,
        momentum=momentum,
    )
    comp_BN_Output = compare_results_batch_norm([tt_output], [torch_result])
    if training:
        channels = input_shapes[1]
        if check_mean:
            comp_BN_running_mean = compare_results_batch_norm(
                [tt_updated_mean], [mean_data.view(1, channels, 1, 1)], stats=True
            )  # Check Updated running mean
        else:
            if tt_updated_mean is None:
                comp_BN_running_mean = True
            else:
                comp_BN_running_mean = False
        if check_var:
            comp_BN_running_var = compare_results_batch_norm(
                [tt_updated_var], [var_data.view(1, channels, 1, 1)], stats=True
            )  # Check Updated running var
        else:
            if tt_updated_var is None:
                comp_BN_running_var = True
            else:
                comp_BN_running_var = False
        comp_BN_Output = comp_BN_Output and comp_BN_running_mean and comp_BN_running_var
    assert comp_BN_Output


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

    status_1 = torch.allclose(result_torch, tt_out, atol=1e-10, rtol=1e-5)
    status_2 = compare_results_batch_norm([result_torch], [tt_out])
    assert status_2 and status_1


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([5, 8, 32, 32]),
        torch.Size([7, 3, 23, 23]),
        torch.Size([3, 5, 64, 120]),
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
    torch_result = torch.nn.functional.batch_norm(
        input=in_data,
        running_mean=mean_data,
        running_var=var_data,
        weight=weight_data,
        bias=bias_data,
        training=training,
        eps=eps,
    )
    comp_BN_Output = compare_results_batch_norm([tt_output], [torch_result]) and torch.allclose(
        torch_result, tt_output, atol=1e-6, rtol=1e-3
    )
    assert comp_BN_Output


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([5, 8, 32, 32]),
        torch.Size([7, 3, 23, 23]),
        torch.Size([3, 5, 64, 120]),
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

    torch_result = torch.nn.functional.batch_norm(
        input=in_data,
        running_mean=mean_data,
        running_var=var_data,
        weight=weight_data,
        bias=bias_data,
        training=training,
        eps=eps,
        momentum=momentum,
    )
    comp_BN_Output = compare_results_batch_norm([tt_output], [torch_result])  # Check BN Result
    if training:
        channels = input_shapes[1]
        if check_mean:
            comp_BN_running_mean = compare_results_batch_norm(
                [tt_updated_mean], [mean_data.view(1, channels, 1, 1)], stats=True
            )  # Check Updated running mean
        else:
            if tt_updated_mean is None:
                comp_BN_running_mean = True
            else:
                comp_BN_running_mean = False
        if check_var:
            comp_BN_running_var = compare_results_batch_norm(
                [tt_updated_var], [var_data.view(1, channels, 1, 1)], stats=True
            )  # Check Updated running var
        else:
            if tt_updated_var is None:
                comp_BN_running_var = True
            else:
                comp_BN_running_var = False
        comp_BN_Output = comp_BN_Output and comp_BN_running_mean and comp_BN_running_var

    assert comp_BN_Output


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([3, 2, 32, 32]),
    ],
)
@pytest.mark.parametrize("mem_layout", [ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.TensorMemoryLayout.HEIGHT_SHARDED])
def test_batch_norm_program_cache_and_default(input_shapes, mem_layout, device):
    N, H, W, C = input_shapes
    in_data, input_tensor = data_gen_with_range_batch_norm(input_shapes, 5, 10, device, is_input=True)
    mean_data, mean_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 10, device)
    var_data, var_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 20, device)

    grid_size = ttnn.CoreGrid(y=1, x=8)
    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_shape = N * H * W // grid_size.x, C // grid_size.y
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(mem_layout, ttnn.types.BufferType.L1, shard_spec)

    if mem_layout is not ttnn.TensorMemoryLayout.INTERLEAVED:
        pytest.xfail("Input tensors to batch norm must be interleaved")

    tt_output_tensor_on_device = ttnn.batch_norm(
        input_tensor, running_mean=mean_tensor, running_var=var_tensor, memory_config=sharded_mem_config
    )
    tt_output = ttnn.to_torch(tt_output_tensor_on_device)
    torch_result = torch.nn.functional.batch_norm(input=in_data, running_mean=mean_data, running_var=var_data)
    comp_BN_Output = compare_results_batch_norm([tt_output], [torch_result])
    assert comp_BN_Output


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([3, 2, 32, 32]),
    ],
)
def test_batch_norm_qid_Default(input_shapes, device):
    N, H, W, C = input_shapes
    in_data, input_tensor = data_gen_with_range_batch_norm(input_shapes, 5, 10, device, is_input=True)
    mean_data, mean_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 10, device)
    var_data, var_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 20, device)

    tt_output_tensor_on_device = ttnn.batch_norm(
        input_tensor, running_mean=mean_tensor, running_var=var_tensor, queue_id=0
    )
    tt_output = ttnn.to_torch(tt_output_tensor_on_device)
    torch_result = torch.nn.functional.batch_norm(input=in_data, running_mean=mean_data, running_var=var_data)
    comp_BN_Output = compare_results_batch_norm([tt_output], [torch_result])
    assert comp_BN_Output


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([3, 2, 32, 32]),
    ],
)
def test_batch_norm_qid(input_shapes, device):
    N, H, W, C = input_shapes
    in_data, input_tensor = data_gen_with_range_batch_norm(input_shapes, 2, 10, device, is_input=True)
    mean_data, mean_tensor = data_gen_with_range_batch_norm(input_shapes, 2, 10, device)
    var_data, var_tensor = data_gen_with_range_batch_norm(input_shapes, 2, 20, device)

    tt_output_tensor_on_device = ttnn.batch_norm(input_tensor, running_mean=mean_tensor, running_var=var_tensor)
    tt_output = ttnn.to_torch(tt_output_tensor_on_device)
    torch_result = torch.nn.functional.batch_norm(input=in_data, running_mean=mean_data, running_var=var_data)
    comp_BN_Output = compare_results_batch_norm([tt_output], [torch_result])
    assert comp_BN_Output


@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([2, 3, 120, 120]),
    ],
)
def test_batch_norm_output_Default(input_shapes, device):
    N, H, W, C = input_shapes
    _, tt_output_tensor = data_gen_with_range_batch_norm(input_shapes, 5, 10, device, is_input=True)
    in_data, input_tensor = data_gen_with_range_batch_norm(input_shapes, 5, 10, device, is_input=True)
    mean_data, mean_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 10, device)
    var_data, var_tensor = data_gen_with_range_batch_norm(input_shapes, 4, 20, device)

    ttnn.batch_norm(input_tensor, running_mean=mean_tensor, running_var=var_tensor, queue_id=0, output=tt_output_tensor)
    tt_output = ttnn.to_torch(tt_output_tensor)
    torch_result = torch.nn.functional.batch_norm(input=in_data, running_mean=mean_data, running_var=var_data)
    comp_BN_Output = compare_results_batch_norm([tt_output], [torch_result])
    assert comp_BN_Output


@pytest.mark.parametrize(
    "input_shapes",
    [
        # Training mode PCC ordering is unreliable. Keep `input_shapes[1] >= 14` to avoid this test failure.
        torch.Size([3, 17, 47, 32]),
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
    N, H, W, C = input_shapes
    torch.manual_seed(0)

    # Generate the inputs
    torch_input_tensor, tt_input_tensor = data_gen_with_range_batch_norm(
        input_shapes, 5, 10, device, is_input=True, testing_dtype=input_dtype
    )
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
    torch_result = torch.nn.functional.batch_norm(
        input=in_ref,
        running_mean=mean_ref,
        running_var=var_ref,
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
