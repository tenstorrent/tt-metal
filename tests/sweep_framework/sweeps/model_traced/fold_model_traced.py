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

# Import V2 master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs

# Override the default timeout in seconds for hang detection.
TIMEOUT = 300

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
# Default: Run exact traced configs from real models with all parameter values in vectors
model_traced_params = loader.get_suite_parameters("fold")

# Parameters provided to the test vector generator are defined here.
parameters = {
    # Quick sample test with basic configurations for fast validation
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.ROW_MAJOR_LAYOUT],  # fold requires ROW_MAJOR
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "storage_type": ["StorageType::DEVICE"],
        "stride_h": [1],
        "stride_w": [1],
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
    stride_h=None,
    stride_w=None,
    arg1=None,
    arg2=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,  # Accept placements, traced_source, traced_machine_info, etc.
) -> list:
    torch.manual_seed(0)

    # V2 JSON provides stride_h/stride_w as positional args: arg1, arg2
    if stride_h is None:
        stride_h = arg1 if arg1 is not None else 1
    if stride_w is None:
        stride_w = arg2 if arg2 is not None else 1

    # Extract kwargs
    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)

    # Check if device is a mesh device (from fixture)
    is_mesh_device = hasattr(device, "get_num_devices")
    output_memory_config = kwargs.get("output_memory_config", None)
    op_kwargs = build_op_kwargs(kwargs, output_memory_config=output_memory_config)

    # V2 format provides input_a_shape
    shape = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
    )(shape)

    # ttnn.fold treats the input as NHWC and requires H (dim 1) divisible by
    # stride_h and W (dim 2) by stride_w ("Input height must be divisible by
    # stride_h"). A trace can record the shape in NCHW order (N, C, H, W) — e.g.
    # (16, 3, 224, 224) with stride 2 has dim 1 = 3 (indivisible) but is really a
    # (16, 224, 224, 3) NHWC image that folds to (16, 112, 112, 12). When the
    # as-NHWC dims don't divide the strides but the NCHW->NHWC permute does,
    # permute the input so both the device op and the golden see valid NHWC.
    _permuted = (
        len(shape) == 4
        and (shape[1] % stride_h != 0 or shape[2] % stride_w != 0)
        and shape[2] % stride_h == 0
        and shape[3] % stride_w == 0
    )
    if _permuted:
        torch_input_tensor_a = torch_input_tensor_a.permute(0, 2, 3, 1).contiguous()
        shape = tuple(torch_input_tensor_a.shape)
        # The traced shard memory configs were laid out for the NCHW shape and no
        # longer match the permuted NHWC tensor (they would mis-shard -> L1 OOM),
        # so feed the permuted input from DRAM interleaved and let fold choose the
        # output layout (drop the traced sharded output/override configs).
        input_a_memory_config = ttnn.DRAM_MEMORY_CONFIG
        op_kwargs.pop("memory_config", None)
        op_kwargs.pop("override_memory_config", None)
        op_kwargs.pop("output_memory_config", None)
        # grid_size / use_transpose_as_fold were chosen for the original sharded
        # NCHW layout; on the re-laid-out NHWC DRAM input they force a huge
        # sharded output (L1 OOM). Drop them so fold sizes itself for this input.
        op_kwargs.pop("grid_size", None)
        op_kwargs.pop("use_transpose_as_fold", None)

    # The device applies the traced `padding` to the NHWC input before folding;
    # mirror it in the golden. padding = [H_top, H_bot, W_left, W_right, C_front,
    # C_back] (e.g. [3,3,3,3,0,1] pads 224->230 H/W and 3->4 C, so a stride-2 fold
    # yields 115x115x16). torch.nn.functional.pad takes dims last-first.
    _golden_input = torch_input_tensor_a
    _pad = op_kwargs.get("padding")
    if isinstance(_pad, (list, tuple)) and len(_pad) == 6 and any(int(p) for p in _pad):
        import torch.nn.functional as _F

        ht, hb, wl, wr, cf, cb = (int(p) for p in _pad)
        _golden_input = _F.pad(torch_input_tensor_a, (cf, cb, wl, wr, ht, hb))

    # fold_torch golden function (NHWC format)
    def fold_torch(input_tensor, stride_h, stride_w):
        N, H, W, C = input_tensor.shape
        reshaped = input_tensor.reshape(N, H // stride_h, stride_h, W // stride_w, stride_w, C)
        transposed = reshaped.permute(0, 1, 3, 2, 4, 5)
        return transposed.reshape(N, H // stride_h, W // stride_w, C * stride_h * stride_w)

    torch_output_tensor = fold_torch(_golden_input, stride_h, stride_w)

    # ttnn.fold outputs in TTNN format [1, 1, N*H'*W', C'] — reshape golden to match
    if torch_output_tensor.ndim == 4:
        N_out, H_out, W_out, C_out = torch_output_tensor.shape
        torch_output_tensor = torch_output_tensor.reshape(1, 1, N_out * H_out * W_out, C_out)

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    if not is_host:
        if is_mesh_device and input_a_tensor_placement:
            # Use mesh with placement
            input_tensor_a = create_tensor_on_mesh(
                torch_input_tensor_a,
                device,
                input_a_dtype,
                input_a_layout,
                input_a_memory_config,
                input_a_tensor_placement,
            )
        else:
            # Regular single-device tensor
            # Check if memory config is sharded - fold may not support all sharded configs
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
    else:
        # Host storage
        input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=input_a_dtype, layout=input_a_layout)

    start_time = start_measuring_time()
    # fold expects stride_h and stride_w as positional args
    output_tensor = ttnn.fold(input_tensor_a, stride_h, stride_w, **op_kwargs)
    if _permuted and is_mesh_device:
        # The re-laid-out NHWC input was replicated across the mesh (a Shard
        # placement is materialized replicated on this device), so each chip holds
        # the full fold result — read one device instead of concatenating (which
        # would multiply the batch by the mesh factor). The DRAM fold returns a 4D
        # [N, H', W', C']; flatten to the [1, 1, N*H'*W', C'] form the golden uses.
        output_tensor = mesh_tensor_to_torch(output_tensor, device, force_single_device=True)
        if output_tensor.ndim == 4:
            _n, _h, _w, _c = output_tensor.shape
            output_tensor = output_tensor.reshape(1, 1, _n * _h * _w, _c)
    else:
        mesh_composer = get_mesh_composer(device, input_a_tensor_placement) if is_mesh_device else None
        output_tensor = mesh_tensor_to_torch(
            output_tensor, device if is_mesh_device else None, mesh_composer=mesh_composer
        )
        if is_mesh_device:
            torch_output_tensor = reconcile_golden_to_actual(
                torch_output_tensor, output_tensor, input_a_tensor_placement
            )
    e2e_perf = stop_measuring_time(start_time)

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
