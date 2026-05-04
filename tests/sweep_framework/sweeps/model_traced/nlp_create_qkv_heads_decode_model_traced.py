# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random
from functools import partial
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
    reconcile_golden_to_actual,
)

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs, parse_dict_value

TIMEOUT = 300

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("experimental::nlp_create_qkv_heads_decode")

parameters = {
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 1536)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "num_heads": [16],
        "num_kv_heads": [4],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
    },
}

if model_traced_params:
    parameters["model_traced"] = model_traced_params


def mesh_device_fixture():
    mesh_shape = get_mesh_shape()
    if mesh_shape:
        try:
            device = create_mesh_device(mesh_shape)
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_mesh_device(device)
        except Exception as e:
            print(f"Failed to create mesh device {mesh_shape}: {e}, falling back to single device")
            device = ttnn.open_device(device_id=0, l1_small_size=79104, dispatch_core_config=ttnn.DispatchCoreConfig())
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_device(device)
    else:
        device = ttnn.open_device(device_id=0, l1_small_size=79104, dispatch_core_config=ttnn.DispatchCoreConfig())
        device_name = ttnn.get_arch_name()
        yield (device, device_name)
        ttnn.close_device(device)
        del device


def _qkvd_input_shard_axis_and_factor(placement_dict):
    if not isinstance(placement_dict, dict):
        return None, 1
    plac_raw = placement_dict.get("placement")
    dist_raw = placement_dict.get("distribution_shape")
    if plac_raw is None or dist_raw is None:
        return None, 1
    if isinstance(plac_raw, (list, tuple)):
        plac_items = [str(x).strip().strip("'") for x in plac_raw]
    else:
        s_inner = str(plac_raw).strip()
        if s_inner.startswith("[") and s_inner.endswith("]"):
            s_inner = s_inner[1:-1]
        plac_items = [x.strip().strip("'") for x in s_inner.split(",") if x.strip()]
    if isinstance(dist_raw, (list, tuple)):
        dist_items = [int(x) for x in dist_raw]
    else:
        d_inner = str(dist_raw).strip()
        if d_inner.startswith("[") and d_inner.endswith("]"):
            d_inner = d_inner[1:-1]
        dist_items = [int(x.strip()) for x in d_inner.split(",") if x.strip()]
    axis = None
    factor = 1
    for entry, n in zip(plac_items, dist_items):
        if entry.startswith("PlacementShard("):
            try:
                d = int(entry[len("PlacementShard(") : -1])
            except ValueError:
                continue
            axis = d
            factor *= n
    return axis, factor


def _qkvd_per_chip_q(per_chip_input, num_heads, num_kv_heads):
    seq_len = per_chip_input.shape[1]
    batch = per_chip_input.shape[2]
    hidden_dim = per_chip_input.shape[3]
    head_dim = hidden_dim // (num_heads + 2 * num_kv_heads)
    q = per_chip_input[:, :, :batch, : head_dim * num_heads].view(seq_len, batch, num_heads, head_dim)
    return q


def run(
    input_a_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    num_heads,
    num_kv_heads,
    output_memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, output_memory_config=output_memory_config)

    # Re-inject memory_config from kwargs (build_op_kwargs strips it by default)
    mc_raw = kwargs.get("memory_config")
    if mc_raw is not None and "memory_config" not in op_kwargs:
        parsed_mc = parse_dict_value("memory_config", mc_raw) if isinstance(mc_raw, dict) else mc_raw
        if parsed_mc is not None:
            op_kwargs["memory_config"] = parsed_mc
    elif output_memory_config is not None and "memory_config" not in op_kwargs:
        op_kwargs["memory_config"] = output_memory_config

    if isinstance(input_a_shape, (tuple, list)):
        shape = tuple(input_a_shape)
    else:
        shape = input_a_shape

    # Try to infer num_heads and num_kv_heads from shape if missing
    if num_heads is None or num_kv_heads is None:
        if len(shape) == 4:
            hidden_dim = shape[3]
            head_dim_guess = 64
            total_heads = hidden_dim // head_dim_guess
            if num_heads is None and num_kv_heads is None:
                num_heads = total_heads // 2
                num_kv_heads = num_heads // 2
            elif num_heads is None:
                num_heads = total_heads - 2 * num_kv_heads
            elif num_kv_heads is None:
                num_kv_heads = (total_heads - num_heads) // 2
        else:
            if num_heads is None:
                num_heads = 16
            if num_kv_heads is None:
                num_kv_heads = num_heads // 2

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-1, high=1, dtype=torch.float32), input_a_dtype
    )(shape)

    # Sharded-aware reference: when input is sharded along hidden_dim, the
    # kernel runs per-chip with per_chip head_dim = global / mesh_factor and
    # the mesh assembler concats Q outputs along the input shard axis.
    # Trace-validation mode: every chip receives the FULL per-chip input via
    # replicate_with_topology. The op runs per-chip and the gathered output is
    # the per-chip Q tiled along the shard axis — handled by
    # reconcile_golden_to_actual below.
    if len(shape) == 4:
        seq_len = shape[1]
        batch = shape[2]
        hidden_dim = shape[3]
        head_dim = hidden_dim // (num_heads + 2 * num_kv_heads)
        q_heads_torch = torch_input_tensor_a[:, :, :batch, : head_dim * num_heads].view(
            seq_len, batch, num_heads, head_dim
        )
        torch_output_tensor = q_heads_torch
    else:
        torch_output_tensor = torch_input_tensor_a.clone()

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

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

    start_time = start_measuring_time()
    output_result = ttnn.experimental.nlp_create_qkv_heads_decode(
        input_tensor_a, num_heads=num_heads, num_kv_heads=num_kv_heads, **op_kwargs
    )
    # nlp_create_qkv_heads_decode returns a tuple of tensors (q_heads, k_heads, v_heads)
    if isinstance(output_result, tuple):
        output_tensor = mesh_tensor_to_torch(output_result[0], device if is_mesh_device else None)
    else:
        output_tensor = mesh_tensor_to_torch(output_result, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    # Check with PCC - using proper torch reference from unit test
    if is_mesh_device:
        torch_output_tensor = reconcile_golden_to_actual(torch_output_tensor, output_tensor, input_a_tensor_placement)
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.9999)
    return [pcc, e2e_perf]
