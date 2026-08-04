# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import ast
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from models.common.utility_functions import torch_random
from functools import partial
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader, dict_to_memory_config
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_model_traced_mesh_shape,
    create_mesh_device,
    create_tensor_on_mesh,
    mesh_tensor_to_torch,
    get_mesh_composer,
    reconcile_golden_to_actual,
)


TIMEOUT = 300

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("rms_norm_post_all_gather")

parameters = {
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_b_dtype": [ttnn.bfloat16],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
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


def run(
    input_a_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    input_b_dtype=None,
    input_b_memory_config=None,
    output_memory_config=None,
    memory_config=None,
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    if input_a_tensor_placement is None:
        input_a_tensor_placement = kwargs.get("input_tensor_a_tensor_placement") or kwargs.get(
            "input_tensor_tensor_placement"
        )
    is_mesh_device = hasattr(device, "get_num_devices")

    if isinstance(input_a_shape, dict) and "self" in input_a_shape:
        shape = input_a_shape["self"] if isinstance(input_a_shape["self"], tuple) else tuple(input_a_shape["self"])
        # "other" is the stats tensor shape (not weight), use it to determine n_devices
        stats_shape_from_trace = input_a_shape.get("other")
        if stats_shape_from_trace is not None:
            stats_shape_from_trace = (
                tuple(stats_shape_from_trace) if isinstance(stats_shape_from_trace, list) else stats_shape_from_trace
            )
    elif isinstance(input_a_shape, (tuple, list)):
        shape = tuple(input_a_shape) if isinstance(input_a_shape, list) else input_a_shape
        stats_shape_from_trace = None
    else:
        shape = (1, 1, 32, 32)
        stats_shape_from_trace = None

    op_kwargs = build_op_kwargs(kwargs, output_memory_config=output_memory_config)

    eps = op_kwargs.get("epsilon", 1e-5)
    hidden_dim = shape[-1]

    # rms_norm_post_all_gather typically expects BFLOAT16/BFLOAT8_B, but some
    # master configs are traced with FLOAT32 input. Try the master dtype first;
    # if the kernel rejects it later we'll fall back to BFLOAT16 below.
    _kernel_compat_dtypes = (ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32)
    if input_a_dtype not in _kernel_compat_dtypes:
        input_a_dtype = ttnn.bfloat16

    torch_input = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype)(
        shape
    )

    # Weight shape: [1, 1, hidden_dim//32, 32] in ROW_MAJOR_LAYOUT, matching
    # the kernel's required gamma format. Master's traced arg1 metadata can
    # show e.g. [1,1,8,256] TILE (the model's in-flight state), but the
    # ttnn.rms_norm_post_all_gather kernel rejects that combo (gamma last-dim
    # must equal input last-dim). Stick with the kernel-compatible default.
    weight_sticks = max(hidden_dim // 32, 1)
    weight_4d_shape = (1, 1, weight_sticks, 32)
    weight_size = weight_sticks * 32
    torch_gamma_1d = torch.randn(weight_size, dtype=torch.float32)
    torch_gamma_4d = torch_gamma_1d.reshape(weight_4d_shape)

    # Golden: output = x * gamma / sqrt(mean(x^2) + eps)
    torch_output = (
        torch_input * torch_gamma_1d[:hidden_dim] / torch.sqrt(torch.mean(torch_input**2, dim=-1, keepdim=True) + eps)
    )

    if is_mesh_device:
        input_tensor = create_tensor_on_mesh(
            torch_input, device, input_a_dtype, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, input_a_tensor_placement
        )
        weight_tensor = create_tensor_on_mesh(
            torch_gamma_4d,
            device,
            input_b_dtype,
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.DRAM_MEMORY_CONFIG,
            kwargs.get("weight_tensor_placement", input_a_tensor_placement),
        )
    else:
        input_tensor = ttnn.from_torch(
            torch_input,
            dtype=input_a_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        weight_tensor = ttnn.from_torch(
            torch_gamma_4d,
            dtype=input_b_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # Land input_tensor on master's exact memory_config (often L1-sharded).
    # The kernel expects that exact layout when a sharded program_config is
    # in use; if conversion fails we leave the tensor in DRAM-interleaved
    # rather than crashing the run.
    # Parse dict form first — validation-vector runs deliver memory_config as
    # a serialized dict that ttnn.to_memory_config can't consume directly.
    if isinstance(input_a_memory_config, dict):
        input_a_memory_config = dict_to_memory_config(input_a_memory_config)
    if input_a_memory_config is not None and input_a_memory_config != ttnn.DRAM_MEMORY_CONFIG:
        try:
            input_tensor = ttnn.to_memory_config(input_tensor, input_a_memory_config)
        except Exception:
            # Best-effort: leave tensor in DRAM-interleaved if the kernel
            # rejects the master shard layout (e.g. shard_spec incompatible
            # with current dispatch grid). PCC may degrade for that cid but
            # we'd rather log it than crash the whole sweep.
            pass

    # Determine n_devices from the traced stats shape
    # Stats shape is (batch..., 32 * n_devices), each device contributes a tile-width (32) of stats
    # V2 vectors expose the stats shape via input_b_shape (the second positional
    # arg of ttnn.rms_norm_post_all_gather is the stats tensor). Prefer that
    # when present so the trace records master's stats width.
    _stats_shape_kw = kwargs.get("input_b_shape")
    if _stats_shape_kw is not None:
        if isinstance(_stats_shape_kw, str):
            try:
                _stats_shape_kw = ast.literal_eval(_stats_shape_kw)
            except Exception:
                _stats_shape_kw = None
        if isinstance(_stats_shape_kw, (list, tuple)) and len(_stats_shape_kw) >= 1:
            stats_shape_from_trace = tuple(_stats_shape_kw)
    if stats_shape_from_trace and len(stats_shape_from_trace) >= 1:
        n_devices = max(stats_shape_from_trace[-1] // 32, 1)
    else:
        n_devices = 1

    # Construct stats tensor matching the gathered format that rms_norm_post_all_gather expects.
    # The full sum(x^2) needs to be split across n_devices slots as if each device
    # computed partial stats on 1/n_devices of the hidden_dim.
    sum_x2 = torch_input.pow(2).sum(dim=-1, keepdim=True)
    stats_width = 32 * n_devices
    stats_tensor_shape = list(shape[:-1]) + [stats_width]
    torch_stats = torch.zeros(stats_tensor_shape, dtype=torch.float32)
    per_device_sum = sum_x2 / n_devices
    for i in range(n_devices):
        torch_stats[..., i * 32 : i * 32 + 1] = per_device_sum

    if is_mesh_device:
        # The stats (arg1) tensor has its OWN traced placement (e.g.
        # Shard(2)+Shard(3)) distinct from input_a's (Replicate+Shard(3)); using
        # input_a's placement here is an arg1.tensor_placement diff vs master.
        _stats_placement = kwargs.get("input_b_tensor_placement")
        if not isinstance(_stats_placement, dict):
            _stats_placement = input_a_tensor_placement
        stats_tensor = create_tensor_on_mesh(
            torch_stats, device, ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, _stats_placement
        )
    else:
        stats_tensor = ttnn.from_torch(
            torch_stats,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # Land stats_tensor on master's input_b_memory_config when present. The
    # sharded program_config rejects DRAM-interleaved stats; master often
    # records stats as L1-WIDTH-SHARDED. Best-effort, fall back silently.
    if isinstance(input_b_memory_config, dict):
        input_b_memory_config = dict_to_memory_config(input_b_memory_config)
    if input_b_memory_config is not None and input_b_memory_config != ttnn.DRAM_MEMORY_CONFIG:
        try:
            stats_tensor = ttnn.to_memory_config(stats_tensor, input_b_memory_config)
        except Exception:
            # Best-effort: a sharded stats layout rejected by the kernel
            # leaves the stats in DRAM-interleaved; the kernel's program_config
            # may then assert is_sharded() but that diff surfaces as a clear
            # failure rather than silent corruption.
            pass

    start_time = start_measuring_time()
    output_tensor = ttnn.rms_norm_post_all_gather(input_tensor, stats_tensor, weight=weight_tensor, **op_kwargs)
    mesh_composer = get_mesh_composer(device, input_a_tensor_placement) if is_mesh_device else None
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None, mesh_composer=mesh_composer)
    e2e_perf = stop_measuring_time(start_time)

    if is_mesh_device:
        torch_output = reconcile_golden_to_actual(torch_output, output_tensor, input_a_tensor_placement)

    return [check_with_pcc(torch_output, output_tensor, 0.999), e2e_perf]
