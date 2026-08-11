# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
)

from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs

# Override the default timeout in seconds for hang detection.
TIMEOUT = 300

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("split")

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 64)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
        "split_size": [32],  # Default split size for sample
        "dim": [3],  # Default dimension for sample
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
    arg1=None,  # split_size — positional in the traced JSON
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, exclude={"dim"}, output_memory_config=output_memory_config)

    # split_size is POSITIONAL in ttnn.split, so the tracer records it as arg1 -- there is no
    # "split_size" key in a traced config (its arguments are exactly arg0/arg1/dim), and
    # build_op_kwargs strips argN by design. Reading it from op_kwargs therefore always missed
    # and fell back to 32, so every traced config ran at 32 instead of its recorded size
    # (128..16384) -- both the golden and the op, so they agreed and the run looked healthy
    # while testing a configuration no model ever traced. Prefer the positional value, matching
    # how concat_model_traced reads its own positional dim.
    split_size = arg1 if arg1 is not None else op_kwargs.get("split_size", 32)
    dim = kwargs.get("dim", 3)

    # Handle tuple input_a_shape for sample suite
    shape = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)

    torch_output_tensors = torch.split(torch_input_tensor_a, split_size, dim=dim)

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
    # Pop split_size from op_kwargs since ttnn.split takes it as a positional argument
    op_kwargs.pop("split_size", None)
    try:
        output_tensors = ttnn.split(input_tensor_a, split_size, dim=dim, **op_kwargs)
    except Exception as e:
        # Splitting a very wide tensor (e.g. a 128256/32064 vocab projection) with
        # a traced sharded/L1 output config overflows L1 — the op's static CBs
        # clash with the output buffers. Retry with a DRAM-interleaved output so
        # the op sizes its footprint to DRAM (the result is layout-independent).
        _m = str(e).lower()
        if not any(s in _m for s in ("clash", "circular buffer", "out of memory", "l1 buffer")):
            raise
        _kw = {k: v for k, v in op_kwargs.items() if k != "memory_config"}
        _kw["memory_config"] = ttnn.DRAM_MEMORY_CONFIG
        output_tensors = ttnn.split(input_tensor_a, split_size, dim=dim, **_kw)
    output_tensors = [mesh_tensor_to_torch(t, device if is_mesh_device else None) for t in output_tensors]
    e2e_perf = stop_measuring_time(start_time)

    # Compare EVERY chunk, and the chunk count. Checking only output_tensors[0] is what let the
    # split_size bug above go unnoticed: with the wrong size, chunk 0 of the op still lined up
    # with chunk 0 of the golden often enough to report PCC 1.0, so half the configs "passed"
    # while splitting at 32 instead of their traced size. A split is only correct if the whole
    # partition matches -- the number of pieces and each piece.
    if len(output_tensors) != len(torch_output_tensors):
        pcc = (
            False,
            f"split produced {len(output_tensors)} chunks, expected {len(torch_output_tensors)} "
            f"(split_size={split_size}, dim={dim})",
        )
    else:
        pcc = (True, "")
        for i, (golden, actual) in enumerate(zip(torch_output_tensors, output_tensors)):
            ok, msg = check_with_pcc(golden, actual, 0.999)
            if not ok:
                pcc = (False, f"chunk {i}/{len(output_tensors)}: {msg}")
                break

    return [pcc, e2e_perf]
