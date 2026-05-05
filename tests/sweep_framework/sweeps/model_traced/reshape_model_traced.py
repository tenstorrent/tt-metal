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
    reconcile_golden_to_actual,
)

from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs, extract_positional_args

TIMEOUT = 300

loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("reshape")

parameters = {
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 32)],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "target_shape": [(1, 32, 1, 32)],
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


def _input_shard_axis_and_factor(placement_dict):
    """Return (axis, factor) for the input's shard dim. axis may be negative.

    Handles placement entries from the trace which may be either a list of
    strings (post-parse) or a string repr like
    "['PlacementReplicate', 'PlacementShard(-1)']". Returns (None, 1) when
    the input is fully replicated.
    """
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
            # Combine multiple shard axes by multiplying factors. Prefer the
            # last (innermost) shard axis as the "axis" anchor.
            axis = d
            factor *= n
    return axis, factor


def _scale_per_chip_to_global(shape, axis, factor):
    """Scale shape's `axis` dim by factor; fall back to last positive dim."""
    if factor == 1 or shape is None:
        return shape
    out = list(shape)
    n = len(out)
    if n == 0:
        return shape
    target = None
    if axis is not None:
        a = axis if axis >= 0 else axis + n
        if 0 <= a < n and out[a] > 0:
            target = a
    if target is None:
        for i in range(n - 1, -1, -1):
            if out[i] > 0:
                target = i
                break
    if target is None:
        return shape
    out[target] = out[target] * factor
    return tuple(out) if isinstance(shape, tuple) else out


def run(
    input_a_shape,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config=None,
    target_shape=None,
    shape=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    input_a_tensor_placement = kwargs.get("input_a_tensor_placement", None)
    is_mesh_device = hasattr(device, "get_num_devices")
    op_kwargs = build_op_kwargs(kwargs, exclude={"arg1", "arg2"}, output_memory_config=output_memory_config)

    # v2 tracer puts target shape in arg1 or shape; arg2 may hold a
    # secondary shape (e.g. padded output shape) used by some internal paths.
    # Filter out "__ABSENT__" sentinel values from the V2 loader.
    def _clean_absent(val):
        """Return None if val is the __ABSENT__ sentinel."""
        return None if val == "__ABSENT__" else val

    pos_args = extract_positional_args(kwargs)
    tgt_shape = _clean_absent(target_shape) or _clean_absent(shape) or _clean_absent(pos_args.get(1, None))
    if tgt_shape is None:
        tgt_shape = (1, 32, 1, 32)  # fallback for sample

    if isinstance(tgt_shape, list):
        tgt_shape = tuple(tgt_shape)
    elif isinstance(tgt_shape, dict) and "value" in tgt_shape:
        import re

        m = re.search(r"\[([0-9, ]+)\]", str(tgt_shape["value"]))
        if m:
            tgt_shape = tuple(int(x) for x in m.group(1).split(","))
    elif isinstance(tgt_shape, dict):
        # Unrecognized dict format -- try to extract shape from any string repr
        import re

        m = re.search(r"\[([0-9, ]+)\]", str(tgt_shape))
        if m:
            tgt_shape = tuple(int(x) for x in m.group(1).split(","))

    # arg2 may be a padded output shape; extract if present.
    # Detect Shape-object form so we can mirror it back when calling the op.
    _arg2_raw = _clean_absent(pos_args.get(2, None))
    arg2_was_shape_obj = isinstance(_arg2_raw, dict) and _arg2_raw.get("type") == "Shape"
    arg2 = _arg2_raw
    if arg2 is not None and isinstance(arg2, dict) and "value" in arg2:
        import re

        m = re.search(r"\[([0-9, ]+)\]", str(arg2["value"]))
        if m:
            arg2 = tuple(int(x) for x in m.group(1).split(","))
        else:
            arg2 = None  # Could not parse dict-style arg2
    elif arg2 is not None and isinstance(arg2, dict):
        import re

        m = re.search(r"\[([0-9, ]+)\]", str(arg2))
        if m:
            arg2 = tuple(int(x) for x in m.group(1).split(","))
        else:
            arg2 = None  # Could not parse dict-style arg2
    if isinstance(arg2, list):
        arg2 = tuple(arg2)
    # Final guard: arg2 must be a tuple of ints if present
    if arg2 is not None and not isinstance(arg2, tuple):
        arg2 = None

    in_shape = tuple(input_a_shape) if isinstance(input_a_shape, (list, tuple)) else input_a_shape

    # Ensure shape is at least 2D for TILE_LAYOUT compatibility
    if len(in_shape) == 1 and input_a_layout == ttnn.TILE_LAYOUT:
        in_shape = (1, in_shape[0])

    torch_input = gen_func_with_cast_tt(partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype)(
        in_shape
    )

    import math

    # Trace-validation mode: every chip receives the FULL per-chip input via
    # replicate_with_topology. ttnn.reshape runs per-chip with the per-chip
    # target shape, and the gathered output is the per-chip reshape tiled
    # along the shard axis — reconcile_golden_to_actual handles that below.
    def _per_chip_reshape(per_chip_input):
        per_chip_numel = per_chip_input.numel()
        per_chip_tgt_numel = math.prod(tgt_shape)
        per_chip_has_padded = (
            per_chip_tgt_numel != per_chip_numel and arg2 is not None and math.prod(arg2) == per_chip_numel
        )
        if per_chip_has_padded:
            out = torch.reshape(per_chip_input, arg2)
            slices = tuple(slice(0, sz) for sz in tgt_shape)
            return out[slices]
        return torch.reshape(per_chip_input, tgt_shape)

    try:
        torch_output = _per_chip_reshape(torch_input)
    except RuntimeError:
        torch_output = torch_input  # placeholder; trace still captured even if PCC fails

    is_host = storage_type and "HOST" in str(storage_type)

    if not is_host:
        if is_mesh_device and input_a_tensor_placement:
            input_tensor = create_tensor_on_mesh(
                torch_input,
                device,
                input_a_dtype,
                input_a_layout,
                input_a_memory_config,
                input_a_tensor_placement,
            )
        elif is_mesh_device:
            # No placement info on a mesh device (e.g. model_traced_sample suite).
            # Replicate the tensor to all devices to avoid undefined behaviour.
            input_tensor = ttnn.from_torch(
                torch_input,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=input_a_memory_config,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
        else:
            input_tensor = ttnn.from_torch(
                torch_input,
                dtype=input_a_dtype,
                layout=input_a_layout,
                device=device,
                memory_config=input_a_memory_config,
            )
    else:
        input_tensor = ttnn.from_torch(torch_input, dtype=input_a_dtype, layout=input_a_layout)

    start_time = start_measuring_time()
    # If master traced arg2 as a Shape object, wrap it back so the tracer
    # captures the same {"type": "Shape"} structure rather than a plain list.
    arg2_to_pass = arg2
    if arg2_was_shape_obj and isinstance(arg2, tuple):
        try:
            arg2_to_pass = ttnn.Shape(list(arg2))
        except Exception:
            arg2_to_pass = arg2
    if arg2 is not None:
        try:
            output_tensor = ttnn.reshape(input_tensor, tgt_shape, arg2_to_pass, **op_kwargs)
        except (TypeError, RuntimeError):
            output_tensor = ttnn.reshape(input_tensor, tgt_shape, **op_kwargs)
    else:
        output_tensor = ttnn.reshape(input_tensor, tgt_shape, **op_kwargs)
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None)
    e2e_perf = stop_measuring_time(start_time)

    if is_mesh_device:
        torch_output = reconcile_golden_to_actual(torch_output, output_tensor, input_a_tensor_placement)
    pcc = check_with_pcc(torch_output, output_tensor, 0.999)
    return [pcc, e2e_perf]
