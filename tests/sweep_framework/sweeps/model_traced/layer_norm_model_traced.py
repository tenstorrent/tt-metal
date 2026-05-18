# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import torch

import ttnn
from models.common.utility_functions import torch_random
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    create_mesh_device,
    create_tensor_on_mesh,
    get_mesh_composer,
    get_model_traced_mesh_shape,
    mesh_tensor_to_torch,
    reconcile_golden_to_actual,
)
from tests.sweep_framework.sweep_utils.op_kwargs_utils import (
    build_op_kwargs,
    extract_named_tensor_kwargs,
    parse_dict_value,
)
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time

# Override the default timeout in seconds for hang detection.
TIMEOUT = 300

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("layer_norm")

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
    },
}

# Only add model_traced suite if it has valid configurations
if model_traced_params:
    parameters["model_traced"] = model_traced_params


def mesh_device_fixture():
    mesh_shape = get_model_traced_mesh_shape()
    device = create_mesh_device(mesh_shape)
    device_name = ttnn.get_arch_name()
    yield (device, device_name)
    ttnn.close_mesh_device(device)


def run(
    input_a_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,  # Accept traced_source, traced_machine_info, etc.
) -> list:
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, output_memory_config=output_memory_config)

    # layer_norm needs memory_config paired with program_config for the kernel
    # to compute correct block_w.  build_op_kwargs filters memory_config by
    # default, so we add it back when a program_config is present.
    if "program_config" in op_kwargs and output_memory_config is not None:
        op_kwargs["memory_config"] = output_memory_config

    # Restore memory_config from traced kwargs when master recorded it
    absent_keys = set(kwargs.get("__absent_keys__") or [])
    traced_memory_config = kwargs.get("memory_config")
    if (
        "memory_config" not in absent_keys
        and traced_memory_config is not None
        and traced_memory_config != "__ABSENT__"
        and "memory_config" not in op_kwargs
    ):
        parsed_mc = parse_dict_value("memory_config", traced_memory_config)
        if parsed_mc is not None:
            op_kwargs["memory_config"] = parsed_mc

    # Handle tuple input_a_shape for sample suite
    shape = tuple(input_a_shape) if isinstance(input_a_shape, (tuple, list)) else input_a_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)

    # Extract named tensor kwargs from traced config
    weight_kwargs = extract_named_tensor_kwargs(kwargs, "weight")
    bias_kwargs = extract_named_tensor_kwargs(kwargs, "bias")
    residual_kwargs = extract_named_tensor_kwargs(kwargs, "residual_input_tensor")

    weight_shape = None
    bias_shape = None
    torch_weight = None
    torch_bias = None
    torch_residual = None

    if weight_kwargs is not None and weight_kwargs.get("shape") is not None:
        weight_shape = tuple(weight_kwargs["shape"])
        torch_weight = torch.randn(weight_shape, dtype=torch.float32)

    if bias_kwargs is not None and bias_kwargs.get("shape") is not None:
        bias_shape = tuple(bias_kwargs["shape"])
        torch_bias = torch.randn(bias_shape, dtype=torch.float32)

    if residual_kwargs is not None and residual_kwargs.get("shape") is not None:
        torch_residual = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
        )(tuple(residual_kwargs["shape"]))

    # Layer norm on last dimension — PyTorch expects weight/bias to be 1D
    normalized_shape = shape[-1:]
    golden_weight = torch_weight.squeeze() if torch_weight is not None else None
    golden_bias = torch_bias.squeeze() if torch_bias is not None else None
    # Matches test_layer_norm_with_weight_bias_and_residual_input: residual is added to input before layer_norm.
    golden_input = torch_input_tensor_a + torch_residual if torch_residual is not None else torch_input_tensor_a
    torch_output_tensor = torch.nn.functional.layer_norm(
        golden_input,
        normalized_shape,
        weight=golden_weight,
        bias=golden_bias,
    )

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    # Create input tensor with mesh support
    if not is_host:
        if is_mesh_device and input_a_tensor_placement:
            input_tensor_a = create_tensor_on_mesh(
                torch_input_tensor_a,
                device,
                input_a_dtype,
                input_a_layout,
                input_a_memory_config,
                input_a_tensor_placement,
            )
        else:
            input_tensor_a = ttnn.from_torch(
                torch_input_tensor_a,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=input_a_memory_config,
            )
    else:
        input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=input_a_dtype, layout=input_a_layout)

    # Create weight tensor on device if traced config had it
    if torch_weight is not None:
        w_dtype = weight_kwargs.get("dtype") or input_a_dtype
        if isinstance(w_dtype, dict):
            w_dtype = parse_dict_value("weight_dtype", w_dtype) or input_a_dtype
        w_layout = weight_kwargs.get("layout") or ttnn.TILE_LAYOUT
        if isinstance(w_layout, dict):
            w_layout = parse_dict_value("weight_layout", w_layout) or ttnn.TILE_LAYOUT
        w_mem = weight_kwargs.get("memory_config") or ttnn.DRAM_MEMORY_CONFIG
        if isinstance(w_mem, dict):
            w_mem = parse_dict_value("weight_memory_config", w_mem) or ttnn.DRAM_MEMORY_CONFIG
        w_placement = weight_kwargs.get("tensor_placement")

        if not is_host:
            if is_mesh_device and w_placement:
                tt_weight = create_tensor_on_mesh(torch_weight, device, w_dtype, w_layout, w_mem, w_placement)
            else:
                tt_weight = ttnn.from_torch(
                    torch_weight, dtype=w_dtype, layout=w_layout, device=device, memory_config=w_mem
                )
        else:
            tt_weight = ttnn.from_torch(torch_weight, dtype=w_dtype, layout=w_layout)
        op_kwargs["weight"] = tt_weight

    # Create bias tensor on device if traced config had it
    if torch_bias is not None:
        b_dtype = bias_kwargs.get("dtype") or input_a_dtype
        if isinstance(b_dtype, dict):
            b_dtype = parse_dict_value("bias_dtype", b_dtype) or input_a_dtype
        b_layout = bias_kwargs.get("layout") or ttnn.TILE_LAYOUT
        if isinstance(b_layout, dict):
            b_layout = parse_dict_value("bias_layout", b_layout) or ttnn.TILE_LAYOUT
        b_mem = bias_kwargs.get("memory_config") or ttnn.DRAM_MEMORY_CONFIG
        if isinstance(b_mem, dict):
            b_mem = parse_dict_value("bias_memory_config", b_mem) or ttnn.DRAM_MEMORY_CONFIG
        b_placement = bias_kwargs.get("tensor_placement")

        if not is_host:
            if is_mesh_device and b_placement:
                tt_bias = create_tensor_on_mesh(torch_bias, device, b_dtype, b_layout, b_mem, b_placement)
            else:
                tt_bias = ttnn.from_torch(
                    torch_bias, dtype=b_dtype, layout=b_layout, device=device, memory_config=b_mem
                )
        else:
            tt_bias = ttnn.from_torch(torch_bias, dtype=b_dtype, layout=b_layout)
        op_kwargs["bias"] = tt_bias

    # Create residual_input_tensor on device if traced config had it
    if torch_residual is not None:
        r_dtype = residual_kwargs.get("dtype") or input_a_dtype
        if isinstance(r_dtype, dict):
            r_dtype = parse_dict_value("residual_input_tensor_dtype", r_dtype) or input_a_dtype
        r_layout = residual_kwargs.get("layout") or ttnn.TILE_LAYOUT
        if isinstance(r_layout, dict):
            r_layout = parse_dict_value("residual_input_tensor_layout", r_layout) or ttnn.TILE_LAYOUT
        r_mem = residual_kwargs.get("memory_config") or input_a_memory_config
        if isinstance(r_mem, dict):
            r_mem = parse_dict_value("residual_input_tensor_memory_config", r_mem) or input_a_memory_config
        r_placement = residual_kwargs.get("tensor_placement")

        if not is_host:
            if is_mesh_device and r_placement:
                tt_residual = create_tensor_on_mesh(torch_residual, device, r_dtype, r_layout, r_mem, r_placement)
            else:
                tt_residual = ttnn.from_torch(
                    torch_residual, dtype=r_dtype, layout=r_layout, device=device, memory_config=r_mem
                )
        else:
            tt_residual = ttnn.from_torch(torch_residual, dtype=r_dtype, layout=r_layout)
        op_kwargs["residual_input_tensor"] = tt_residual

    # recip_tensor is a Welford-algorithm LUT — build via the ttnn factory so
    # numerical output matches torch's golden (random data would corrupt it).
    recip_kwargs = extract_named_tensor_kwargs(kwargs, "recip_tensor")
    if recip_kwargs is not None and recip_kwargs.get("shape") is not None and not is_host:
        recip_width = tuple(recip_kwargs["shape"])[-1]
        grid = device.compute_with_storage_grid_size()
        recip_core_range_set = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
        )
        op_kwargs["recip_tensor"] = ttnn.create_layer_norm_reciprocals(device, recip_core_range_set, recip_width)

    start_time = start_measuring_time()
    output_tensor = ttnn.layer_norm(input_tensor_a, **op_kwargs)
    mesh_composer = get_mesh_composer(device, input_a_tensor_placement) if is_mesh_device else None
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None, mesh_composer=mesh_composer)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC
    if is_mesh_device:
        torch_output_tensor = reconcile_golden_to_actual(torch_output_tensor, output_tensor, input_a_tensor_placement)
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
