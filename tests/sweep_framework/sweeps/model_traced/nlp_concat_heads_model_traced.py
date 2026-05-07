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
    get_model_traced_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
    get_mesh_composer,
    reconcile_golden_to_actual,
)

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs


# Override the default timeout in seconds for hang detection.
TIMEOUT = 300

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("experimental::nlp_concat_heads")

parameters = {
    "model_traced_sample": {
        "input_a_shape": [(1, 12, 32, 64)],  # Batch, heads, seq, head_dim
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
    },
}

if model_traced_params:
    parameters["model_traced"] = model_traced_params


def mesh_device_fixture():
    mesh_shape = get_model_traced_mesh_shape()
    device = create_mesh_device(mesh_shape)
    device_name = ttnn.get_arch_name()
    yield (device, device_name)
    ttnn.close_mesh_device(device)


def _nch_input_shard_axis_and_factor(placement_dict):
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


def _nch_per_chip_concat_heads(per_chip_input):
    if per_chip_input.ndim != 4:
        return per_chip_input.clone()
    b, h, s, d = per_chip_input.shape
    return per_chip_input.permute(0, 2, 1, 3).contiguous().view(b, s, h * d).unsqueeze(1)


def run(
    input_a_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
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

    # memory_config is a legitimate op kwarg for nlp_concat_heads (controls output layout).
    # build_op_kwargs strips it by default, so re-add from traced kwargs if present.
    traced_memory_config = kwargs.get("memory_config")
    if traced_memory_config is not None and traced_memory_config != "__ABSENT__":
        from tests.sweep_framework.sweep_utils.op_kwargs_utils import parse_dict_value

        parsed_mc = parse_dict_value("memory_config", traced_memory_config)
        if parsed_mc is not None:
            op_kwargs["memory_config"] = parsed_mc

    if isinstance(input_a_shape, (tuple, list)):
        shape = tuple(input_a_shape)
    else:
        shape = input_a_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-1, high=1, dtype=torch.float32), input_a_dtype
    )(shape)

    # nlp_concat_heads: [B, H, S, D] -> [B, 1, S, H*D]. With a sharded input
    # the kernel runs this per-chip and the mesh assembler concats along the
    # input shard axis; concat-of-per-chip differs element-wise from the
    # global op, so we mirror the kernel.
    # Trace-validation mode: every chip receives the FULL per-chip input via
    # replicate_with_topology and runs nlp_concat_heads on it. The gathered
    # output is the per-chip result tiled along the shard axis — handled by
    # reconcile_golden_to_actual below.
    if len(shape) == 4:
        torch_output_tensor = _nch_per_chip_concat_heads(torch_input_tensor_a)
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
            # Create on DRAM first, then try to move to the traced sharded config.
            # This avoids "shard width must match physical width" errors with HEIGHT_SHARDED configs.
            actual_memory_config = input_a_memory_config
            if hasattr(input_a_memory_config, "is_sharded") and input_a_memory_config.is_sharded():
                actual_memory_config = ttnn.DRAM_MEMORY_CONFIG

            input_tensor_a = ttnn.from_torch(
                torch_input_tensor_a,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=actual_memory_config,
            )

            if actual_memory_config is not input_a_memory_config:
                try:
                    input_tensor_a = ttnn.to_memory_config(input_tensor_a, input_a_memory_config)
                except Exception:
                    pass  # stay on DRAM if sharded config is incompatible
    else:
        input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=input_a_dtype, layout=input_a_layout)

    start_time = start_measuring_time()
    # memory_config kwarg controls the output layout (the op auto-derives shard spec).
    output_tensor = ttnn.experimental.nlp_concat_heads(input_tensor_a, **op_kwargs)

    mesh_composer = get_mesh_composer(device, input_a_tensor_placement) if is_mesh_device else None
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None, mesh_composer=mesh_composer)
    e2e_perf = stop_measuring_time(start_time)

    if is_mesh_device:
        torch_output_tensor = reconcile_golden_to_actual(torch_output_tensor, output_tensor, input_a_tensor_placement)
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.99)
    return [pcc, e2e_perf]
