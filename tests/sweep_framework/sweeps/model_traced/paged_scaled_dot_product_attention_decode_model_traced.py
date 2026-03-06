# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
)

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs

TIMEOUT = 300

# NOTE:
# -----
# For most ops, the model_traced suite uses real traced configurations from
# production models plus a PyTorch/TTNN golden.  For paged SDPA decode the
# correctness oracle is substantially more complex (see
# tests/tt_eager/python_api_testing/unit_testing/misc/test_scaled_dot_product_attention_decode.py),
# and we do not yet have a lightweight reference that matches all traced cases.
# Until such a golden is implemented, we deliberately *do not* enable the
# model_traced suite for this op to avoid claiming coverage we do not have.
#
# The sample suite below still exercises the operation shape/layout path; the
# traced configurations will be wired in a follow-up once a proper golden exists.
loader = MasterConfigLoader()
_model_traced_params = None  # reserved for future enablement

parameters = {
    "model_traced_sample": {
        "input_a_shape": [(1, 8, 32, 64)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
    },
}

# Intentionally do not attach a "model_traced" suite yet.


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
            device = ttnn.open_device(device_id=0, dispatch_core_config=ttnn.DispatchCoreConfig())
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_device(device)
    else:
        device = ttnn.open_device(device_id=0, dispatch_core_config=ttnn.DispatchCoreConfig())
        device_name = ttnn.get_arch_name()
        yield (device, device_name)
        ttnn.close_device(device)
        del device


def run(
    input_a_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    input_b_dtype=None,
    input_b_layout=None,
    input_b_memory_config=None,
    input_c_dtype=None,
    input_c_layout=None,
    input_c_memory_config=None,
    input_d_dtype=None,
    input_d_layout=None,
    input_d_memory_config=None,
    input_e_dtype=None,
    input_e_layout=None,
    input_e_memory_config=None,
    output_memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    input_b_tensor_placement = kwargs.get("input_b_tensor_placement", None)
    input_c_tensor_placement = kwargs.get("input_c_tensor_placement", None)
    input_d_tensor_placement = kwargs.get("input_d_tensor_placement", None)
    input_e_tensor_placement = kwargs.get("input_e_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, output_memory_config=output_memory_config)

    # Handle dict input_a_shape from traced configurations (multi-input)
    if isinstance(input_a_shape, dict):
        # Traced configuration with multiple inputs
        shape_a = input_a_shape.get("input_a", input_a_shape.get("self"))
        shape_b = input_a_shape.get("input_b", input_a_shape.get("other"))
        shape_c = input_a_shape.get("input_c")
        shape_d = input_a_shape.get("input_d")
        shape_e = input_a_shape.get("input_e")
    else:
        # Fallback for sample configurations
        if isinstance(input_a_shape, (tuple, list)):
            shape = tuple(input_a_shape)
        else:
            shape = input_a_shape
        shape_a = shape_b = shape_c = shape_d = shape_e = shape

    # Use provided dtypes - fail if not provided (no fallbacks)
    dtype_a = input_a_dtype
    if input_b_dtype is None:
        raise ValueError("input_b_dtype is None - required parameter missing")
    if input_c_dtype is None:
        raise ValueError("input_c_dtype is None - required parameter missing")
    if input_d_dtype is None:
        raise ValueError("input_d_dtype is None - required parameter missing")
    if input_e_dtype is None:
        raise ValueError("input_e_dtype is None - required parameter missing")
    dtype_b = input_b_dtype
    dtype_c = input_c_dtype
    dtype_d = input_d_dtype
    dtype_e = input_e_dtype

    # Use provided layouts - fail if not provided (no fallbacks)
    layout_a = input_a_layout
    if input_b_layout is None:
        raise ValueError("input_b_layout is None - required parameter missing")
    if input_c_layout is None:
        raise ValueError("input_c_layout is None - required parameter missing")
    if input_d_layout is None:
        raise ValueError("input_d_layout is None - required parameter missing")
    if input_e_layout is None:
        raise ValueError("input_e_layout is None - required parameter missing")
    layout_b = input_b_layout
    layout_c = input_c_layout

    # Use provided memory configs - fail if not provided (no fallbacks)
    mem_config_a = input_a_memory_config
    if input_b_memory_config is None:
        raise ValueError("input_b_memory_config is None - required parameter missing")
    if input_c_memory_config is None:
        raise ValueError("input_c_memory_config is None - required parameter missing")
    if input_d_memory_config is None:
        raise ValueError("input_d_memory_config is None - required parameter missing")
    if input_e_memory_config is None:
        raise ValueError("input_e_memory_config is None - required parameter missing")
    if output_memory_config is None:
        raise ValueError("output_memory_config is None - required parameter missing")
    mem_config_b = input_b_memory_config
    mem_config_c = input_c_memory_config
    mem_config_d = input_d_memory_config
    mem_config_e = input_e_memory_config
    output_mem_config = output_memory_config

    # Create input tensors
    torch_input_a = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), dtype_a)(shape_a)
    torch_input_b = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), dtype_b)(shape_b)
    torch_input_c = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), dtype_c)(shape_c)
    torch_input_d = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), dtype_d)(shape_d)
    torch_input_e = gen_func_with_cast_tt(partial(torch_random, low=-1, high=1, dtype=torch.float32), dtype_e)(shape_e)

    # TODO: Compute a true PyTorch attention golden using traced K/V/page table inputs.
    torch_output_tensor = torch_input_a.clone()

    # Check if storage_type is HOST
    is_host = storage_type and "HOST" in str(storage_type)

    # Convert to TTNN tensors
    tensor_a = ttnn.from_torch(
        torch_input_a,
        dtype=dtype_a,
        layout=layout_a,
        device=device,
        memory_config=mem_config_a,
    )

    tensor_b = ttnn.from_torch(
        torch_input_b,
        dtype=dtype_b,
        layout=layout_b,
        device=device,
        memory_config=mem_config_b,
    )

    tensor_c = ttnn.from_torch(
        torch_input_c,
        dtype=dtype_c,
        layout=layout_c,
        device=device,
        memory_config=mem_config_c,
    )

    tensor_d = ttnn.from_torch(
        torch_input_d,
        dtype=dtype_d,
        layout=ttnn.ROW_MAJOR_LAYOUT,  # page_table_tensor must be ROW_MAJOR
        device=device,
        memory_config=mem_config_d,
    )

    tensor_e = ttnn.from_torch(
        torch_input_e,
        dtype=dtype_e,
        layout=ttnn.ROW_MAJOR_LAYOUT,  # cur_pos_tensor must be ROW_MAJOR
        device=device,
        memory_config=mem_config_e,
    )

    start_time = start_measuring_time()
    # paged_scaled_dot_product_attention_decode signature:
    # (input_tensor_q, input_tensor_k, input_tensor_v, page_table_tensor, *, is_causal=True, attn_mask=None, cur_pos_tensor=None, ...)
    # So tensor_a=Q, tensor_b=K, tensor_c=V, tensor_d=page_table, tensor_e=cur_pos
    output_tensor = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        tensor_a,  # Q
        tensor_b,  # K
        tensor_c,  # V
        tensor_d,  # page_table (required positional)
        is_causal=True,
        cur_pos_tensor=tensor_e,  # cur_pos (optional keyword), **op_kwargs,
    )
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.99)
    return [pcc, e2e_perf]
