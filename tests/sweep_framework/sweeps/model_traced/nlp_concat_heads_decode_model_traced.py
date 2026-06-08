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
    replicate_with_topology,
    mesh_tensor_to_torch,
    get_mesh_composer,
)

# Import master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs

TIMEOUT = 300

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("experimental::nlp_concat_heads_decode")

parameters = {
    "model_traced_sample": {
        "input_a_shape": [(1, 12, 32, 64)],
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


def _nchd_input_shard_axis_and_factor(placement_dict):
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


def run(
    input_a_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config=None,
    num_heads=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, output_memory_config=output_memory_config)

    if isinstance(input_a_shape, (tuple, list)):
        shape = tuple(input_a_shape)
    else:
        shape = input_a_shape

    # num_heads is required - try to get from op_kwargs first, then kwargs, then infer.
    # op_kwargs is the authoritative source because it mirrors the master trace kwargs.
    if num_heads is None:
        num_heads = op_kwargs.pop("num_heads", None)
    if num_heads is None:
        num_heads = kwargs.get("num_heads", None)
    if num_heads is None:
        # Fallback: infer from input shape (may be tile-padded — less reliable)
        if len(shape) == 4 and shape[1] == 1:
            num_heads = shape[2]
        else:
            num_heads = 16

    # The V2 vector may contain a tile-padded shape (e.g. dim-2 = 32) while the
    # master trace records the original logical shape (e.g. dim-2 = 8 = num_heads).
    # Override dim-2 with the true num_heads so the tracer records the correct shape.
    if len(shape) == 4 and num_heads is not None and shape[2] != num_heads:
        shape = (shape[0], shape[1], num_heads, shape[3])

    # When input is sharded on a non-head axis (typically dim -1 / hidden),
    # V2 may have expanded that dim to the GLOBAL value, in which case we
    # need to shrink to per-chip before replicate_with_topology so the
    # per-chip tensor width matches shard_spec. But traced V2 vectors
    # already carry the per-chip shape (matches master original_shape) —
    # detect that case by comparing shape[-1] to the shard_spec width and
    # skip the shrink to avoid driving width below the shard_spec.
    _nchd_axis, _nchd_factor = _nchd_input_shard_axis_and_factor(input_a_tensor_placement)
    n_in = len(shape)
    _nchd_axis_norm = (_nchd_axis if _nchd_axis >= 0 else _nchd_axis + n_in) if _nchd_axis is not None else None

    def _shard_spec_width(mc):
        try:
            ss = getattr(mc, "shard_spec", None)
            if ss is not None and getattr(ss, "shape", None) is not None:
                return int(ss.shape[1])
        except Exception:
            # mc may be a serialized dict (no .shard_spec attr) or a memory
            # config without a shard spec. Fall through to the dict path /
            # None default below.
            pass
        if isinstance(mc, dict):
            ss = mc.get("data", {}).get("shard_spec")
            if isinstance(ss, dict) and isinstance(ss.get("shape"), list) and len(ss["shape"]) >= 2:
                return int(ss["shape"][1])
        return None

    _ss_w = _shard_spec_width(input_a_memory_config)
    _already_per_chip = (
        _ss_w is not None and _nchd_axis_norm is not None and _nchd_axis_norm == n_in - 1 and shape[-1] == _ss_w
    )
    if (
        _nchd_factor > 1
        and _nchd_axis_norm is not None
        and _nchd_axis_norm != 2  # dim 2 is the heads axis; do not shrink it
        and shape[_nchd_axis_norm] % _nchd_factor == 0
        and not _already_per_chip
    ):
        shape = tuple((s_ // _nchd_factor) if i == _nchd_axis_norm else s_ for i, s_ in enumerate(shape))

    torch_input_tensor_a = gen_func_with_cast_tt(
        partial(torch_random, low=-1, high=1, dtype=torch.float32), input_a_dtype
    )(shape)

    # Proper torch reference from test_nlp_concat_heads_decode.py (line 95)
    # Per-chip: Input (1, batch, padded_heads, head_dim) -> Output (1, 1, batch, head_dim*num_heads)
    # mesh_tensor_to_torch then concats per-chip outputs on the input shard axis,
    # so tile the per-chip golden by _nchd_factor on that axis to match.
    if len(shape) == 4:
        _, batch, padded_heads, head_dim = shape
        per_chip_out = torch_input_tensor_a[:, :, :num_heads, :].reshape(1, 1, batch, head_dim * num_heads)
        if _nchd_factor > 1 and _nchd_axis_norm is not None:
            # Output dim-3 is head_dim*num_heads. mesh_tensor_to_torch concats
            # along the recorded input shard axis. For Shard(-1) on the input
            # hidden, the corresponding output axis is also dim -1.
            out_axis = _nchd_axis_norm if _nchd_axis_norm < per_chip_out.ndim else per_chip_out.ndim - 1
            torch_output_tensor = torch.cat([per_chip_out] * _nchd_factor, dim=out_axis)
        else:
            torch_output_tensor = per_chip_out
    else:
        torch_output_tensor = torch_input_tensor_a.clone()

    # Check if storage_type is HOST - if so, don't pass device to from_torch
    is_host = storage_type and "HOST" in str(storage_type)

    if not is_host:
        if is_mesh_device and input_a_tensor_placement:
            # The input tensor shape is (1, batch, num_heads, head_dim) per-device.
            # The master model creates it per-device (replicated) so logical_shape()
            # returns the per-device shape with dim[2]=num_heads.
            # create_tensor_on_mesh would expand+shard, causing logical_shape() to
            # return the global shape (dim[2]=num_heads*mesh_factor), mismatching
            # the master trace.  Replicate to preserve the per-device shape.
            input_tensor_a = replicate_with_topology(
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
    output_tensor = ttnn.experimental.nlp_concat_heads_decode(input_tensor_a, num_heads=num_heads, **op_kwargs)
    mesh_composer = get_mesh_composer(device, input_a_tensor_placement) if is_mesh_device else None
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None, mesh_composer=mesh_composer)
    e2e_perf = stop_measuring_time(start_time)

    # Unpad the output - TTNN output may be padded to tile size (32)
    # We need to extract only the actual batch size
    if len(shape) == 4:
        _, batch, _, _ = shape
        output_tensor = output_tensor[:, :, :batch, :]

    # Check with PCC - using standard threshold
    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.99)

    return [pcc, e2e_perf]
