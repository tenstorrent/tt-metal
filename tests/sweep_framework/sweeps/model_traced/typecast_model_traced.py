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
)

from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader, parse_dtype
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs, extract_positional_args

TIMEOUT = 300

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("typecast")

parameters = {
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_dtype": [ttnn.float32],
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


def run(
    input_a_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_dtype=None,
    output_memory_config=None,
    memory_config=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, exclude={"arg1", "dtype"}, output_memory_config=output_memory_config)

    pos_args = extract_positional_args(kwargs)
    output_dtype = output_dtype or kwargs.get("dtype", pos_args.get(1, ttnn.float32))
    if isinstance(output_dtype, dict):
        output_dtype = parse_dtype(output_dtype.get("repr", ""))
    elif isinstance(output_dtype, str):
        output_dtype = parse_dtype(output_dtype)
    if output_dtype is None:
        output_dtype = ttnn.float32
    if output_memory_config is None and memory_config is not None:
        output_memory_config = memory_config

    shape = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape

    if input_a_dtype == ttnn.uint16:
        # ttnn.from_torch expects torch.int16 for uint16 (see gen_func_with_cast_tt)
        torch_input_tensor_a = torch.randint(0, 65536, shape, dtype=torch.int32).clamp(0, 65535).to(torch.int16)
    elif input_a_dtype == ttnn.uint32:
        torch_input_tensor_a = torch.randint(0, 2**32, shape, dtype=torch.int64)
    else:
        torch_input_tensor_a = gen_func_with_cast_tt(
            partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
        )(shape)

    if output_dtype == ttnn.float32:
        torch_output_tensor = torch_input_tensor_a.to(torch.float32)
    elif output_dtype == ttnn.bfloat16:
        torch_output_tensor = torch_input_tensor_a.to(torch.bfloat16).to(torch.float32)
    elif output_dtype == ttnn.bfloat8_b:
        torch_output_tensor = torch_input_tensor_a.to(torch.float32)
    elif output_dtype == ttnn.uint16:
        torch_output_tensor = torch_input_tensor_a.clamp(0, 65535).to(torch.int32)
    elif output_dtype == ttnn.uint32:
        if input_a_dtype == ttnn.uint32:
            torch_output_tensor = torch_input_tensor_a.clamp(0, 2**32 - 1)
        else:
            torch_output_tensor = torch_input_tensor_a.clamp(0, 2**32 - 1).to(torch.int64)
    elif output_dtype == ttnn.int32:
        # For uint16→int32, interpret as unsigned: mask with 0xFFFF to get unsigned value
        if input_a_dtype == ttnn.uint16:
            torch_output_tensor = torch_input_tensor_a.to(torch.int32) & 0xFFFF
        else:
            torch_output_tensor = torch_input_tensor_a.to(torch.int32)
    else:
        torch_output_tensor = torch_input_tensor_a.to(torch.float32)

    is_host = storage_type and "HOST" in str(storage_type)

    if not is_host:
        if is_mesh_device:
            # Typecast is element-wise: replicate to all devices and compare
            # device-0 output against the original reference tensor.
            # Using create_tensor_on_mesh with ShardTensor2dMesh repeats/shards
            # the input, causing a mismatch when extracting device 0 only.
            input_tensor_a = ttnn.from_torch(
                torch_input_tensor_a,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=input_a_memory_config,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
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

    # === DEBUG: dump input tensor from device before typecast ===
    _debug = input_a_dtype == ttnn.uint16 and output_dtype == ttnn.int32
    if _debug:
        print(f"\n{'='*80}")
        print(f"[DEBUG typecast] UINT16→INT32 detected")
        print(f"[DEBUG typecast] is_mesh_device={is_mesh_device}, shape={shape}")
        print(f"[DEBUG typecast] input_a_dtype={input_a_dtype}, output_dtype={output_dtype}")
        print(f"[DEBUG typecast] placement={input_a_tensor_placement}")
        if is_mesh_device:
            print(f"[DEBUG typecast] num_devices={device.get_num_devices()}")
            # Read back input from device 0 BEFORE typecast
            dev_tensors_in = ttnn.get_device_tensors(input_tensor_a)
            input_back_d0 = ttnn.to_torch(dev_tensors_in[0])
            print(f"[DEBUG typecast] input_back_d0 shape={input_back_d0.shape}, dtype={input_back_d0.dtype}")
            print(f"[DEBUG typecast] input_back_d0 first 16 values: {input_back_d0.flatten()[:16].tolist()}")
            print(f"[DEBUG typecast] input_back_d0 min={input_back_d0.min().item()}, max={input_back_d0.max().item()}")
            # Also check golden
            print(f"[DEBUG typecast] torch_input (host) first 16: {torch_input_tensor_a.flatten()[:16].tolist()}")
            print(f"[DEBUG typecast] torch_input (host) dtype={torch_input_tensor_a.dtype}")
            # Check if device 0 input matches host input
            host_flat = torch_input_tensor_a.flatten()[:16].to(torch.int32)
            dev_flat = input_back_d0.flatten()[:16].to(torch.int32)
            print(f"[DEBUG typecast] host_as_int32={host_flat.tolist()}")
            print(f"[DEBUG typecast] dev0_as_int32={dev_flat.tolist()}")
            match = torch.equal(host_flat, dev_flat)
            print(f"[DEBUG typecast] input host==dev0: {match}")
        else:
            print(f"[DEBUG typecast] single device mode")

    start_time = start_measuring_time()
    output_tensor = ttnn.typecast(input_tensor_a, output_dtype, **op_kwargs)
    # Use device-0 extraction (no mesh composer) to get per-device output that
    # matches the per-device reference tensor.  Typecast is element-wise so each
    # device's output independently matches the reference.
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    if output_dtype == ttnn.uint32 or input_a_dtype == ttnn.uint32:
        if torch_output_tensor.dtype != torch.int64:
            torch_output_tensor_f32 = torch_output_tensor.to(torch.int64).to(torch.float32)
        else:
            torch_output_tensor_f32 = torch_output_tensor.to(torch.float32)
        if output_tensor.dtype != torch.int64:
            output_tensor_f32 = output_tensor.to(torch.int64).to(torch.float32)
        else:
            output_tensor_f32 = output_tensor.to(torch.float32)
    else:
        torch_output_tensor_f32 = torch_output_tensor.to(torch.float32)
        output_tensor_f32 = output_tensor.to(torch.float32)

    # === DEBUG: dump output comparison details ===
    if _debug:
        print(f"[DEBUG typecast] output_tensor shape={output_tensor.shape}, dtype={output_tensor.dtype}")
        print(f"[DEBUG typecast] output first 16: {output_tensor.flatten()[:16].tolist()}")
        print(f"[DEBUG typecast] golden first 16: {torch_output_tensor.flatten()[:16].tolist()}")
        print(f"[DEBUG typecast] output_f32 first 16: {output_tensor_f32.flatten()[:16].tolist()}")
        print(f"[DEBUG typecast] golden_f32 first 16: {torch_output_tensor_f32.flatten()[:16].tolist()}")
        # Element-wise diff
        n_total = output_tensor_f32.numel()
        n_mismatch = (output_tensor_f32 != torch_output_tensor_f32).sum().item()
        print(f"[DEBUG typecast] total elements={n_total}, mismatches={n_mismatch} ({100*n_mismatch/max(n_total,1):.1f}%)")
        # Show some mismatched indices
        diff_mask = (output_tensor_f32 != torch_output_tensor_f32).flatten()
        diff_indices = torch.where(diff_mask)[0][:10]
        for idx in diff_indices:
            g = torch_output_tensor_f32.flatten()[idx].item()
            o = output_tensor_f32.flatten()[idx].item()
            print(f"[DEBUG typecast]   idx={idx.item()}: golden={g}, output={o}, diff={o-g}")
        print(f"{'='*80}\n")

    # bfloat8_b and bfloat4_b are block floating-point formats with significant
    # quantisation loss, especially for wide value ranges.  Use a relaxed PCC
    # threshold when either the input or output dtype is one of these formats.
    lossy_dtypes = {ttnn.bfloat8_b, ttnn.bfloat4_b}
    if input_a_dtype in lossy_dtypes or output_dtype in lossy_dtypes:
        pcc_threshold = 0.79
    else:
        pcc_threshold = 0.999

    pcc = check_with_pcc(torch_output_tensor_f32, output_tensor_f32, pcc_threshold)
    return [pcc, e2e_perf]
