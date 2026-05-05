# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Sweep for ``ttnn.full`` traced from real model invocations (V2 format).

The Flux 1 dev pipeline calls ``ttnn.full`` to materialize per-step ``timestep``
and ``sigma_difference`` scalar tensors inside the denoising loop. Both
``device=mesh_device`` (warmup / non-traced path) and ``device=None`` (traced
region path) are exercised, so the master records 4 distinct configurations.
"""

import re

import torch
import ttnn

from tests.ttnn.utils_for_testing import start_measuring_time, stop_measuring_time
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_model_traced_mesh_shape,
    create_mesh_device,
)

# Override the default timeout in seconds for hang detection.
TIMEOUT = 300

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("full")

parameters = {
    "model_traced_sample": {
        "arg0": [(1, 1)],
        "fill_value_value": [0.5],
        "dtype": [ttnn.float32],
        "layout": [ttnn.TILE_LAYOUT],
        "device_was_traced": [True],
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


def _parse_shape(arg0):
    """Return a python list of ints from arg0.

    arg0 is either a list/tuple of ints or a serialized ``Shape`` dict like
    ``{'type': 'Shape', 'value': 'Shape([1, 1024, 64])'}``.
    """
    if isinstance(arg0, (list, tuple)):
        return [int(x) for x in arg0]
    if isinstance(arg0, dict):
        value = str(arg0.get("value", ""))
        nums = re.findall(r"-?\d+", value)
        if nums:
            return [int(x) for x in nums]
    if isinstance(arg0, str):
        nums = re.findall(r"-?\d+", arg0)
        if nums:
            return [int(x) for x in nums]
    raise ValueError(f"Cannot parse shape from arg0={arg0!r}")


def _parse_dtype(value):
    if isinstance(value, ttnn.DataType):
        return value
    s = str(value)
    if "BFLOAT16" in s:
        return ttnn.bfloat16
    if "FLOAT32" in s:
        return ttnn.float32
    if "FLOAT16" in s:
        return ttnn.float16
    if "INT32" in s:
        return ttnn.int32
    if "UINT32" in s:
        return ttnn.uint32
    if "UINT16" in s:
        return ttnn.uint16
    return ttnn.bfloat16


def _parse_layout(value):
    if isinstance(value, ttnn.Layout):
        return value
    s = str(value)
    if "ROW_MAJOR" in s:
        return ttnn.ROW_MAJOR_LAYOUT
    return ttnn.TILE_LAYOUT


def _torch_dtype_from_dict(value):
    if isinstance(value, dict):
        s = str(value.get("dtype", "torch.float32"))
        if "float32" in s:
            return torch.float32
        if "bfloat16" in s:
            return torch.bfloat16
        if "float16" in s:
            return torch.float16
        if "int32" in s or "int64" in s:
            return torch.int32
    return torch.float32


_ABSENT = object()  # sentinel: traced_device_kwarg key was not present in the vector


def _master_device_state(traced_device_kwarg):
    """Decide how to forward ``device`` to ``ttnn.full`` to match the master.

    Returns one of:
        ``"omit"``   — master never passed ``device``; do not include it.
        ``"none"``   — master passed ``device=None`` explicitly; pass None.
        ``"mesh"``   — master passed ``device=mesh_device``; forward the fixture.

    Pass ``_ABSENT`` (the module sentinel) when the key was missing from the
    vector, so ``None`` unambiguously means the master traced ``device=None``.
    """
    if traced_device_kwarg is _ABSENT:
        return "omit"
    if isinstance(traced_device_kwarg, dict):
        return "mesh"
    if traced_device_kwarg is None:
        return "none"
    return "omit"


def _make_arg0(arg0):
    """Return the value to pass as the first positional argument to ttnn.full.

    The tracer records the *exact* runtime type of arg0 in the config_hash
    (``ttnn.Shape`` vs a Python list). To round-trip, mirror whatever the
    master used: vector arg0 == ``{'type': 'Shape', ...}`` -> wrap in
    ``ttnn.Shape``; list/tuple -> pass as a list.
    """
    if isinstance(arg0, dict) and arg0.get("type") == "Shape":
        nums = re.findall(r"-?\d+", str(arg0.get("value", "")))
        return ttnn.Shape([int(n) for n in nums])
    if isinstance(arg0, (list, tuple)):
        return [int(x) for x in arg0]
    if isinstance(arg0, str):
        nums = re.findall(r"-?\d+", arg0)
        if nums:
            return [int(n) for n in nums]
    raise ValueError(f"Cannot convert arg0={arg0!r} to ttnn.full first argument")


def run(
    arg0=None,
    fill_value=None,
    dtype=None,
    layout=None,
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    # Sample-suite path: explicit fill_value and a flag to drive whether to
    # forward device= to ttnn.full. model_traced uses the traced kwarg below.
    sample_fill_value = kwargs.get("fill_value_value")
    sample_device_was_traced = kwargs.get("device_was_traced")

    shape_arg = _make_arg0(arg0)
    logical_shape = _parse_shape(arg0)
    dtype_v = _parse_dtype(dtype)
    layout_v = _parse_layout(layout)

    # fill_value_value is the path used by the sample suite. For the model_traced
    # suite the master records fill_value as a 0-dim torch.Tensor (the scheduler
    # passed `t` which is a torch scalar tensor); reproduce that.
    if sample_fill_value is not None:
        fill_value_t = torch.tensor(float(sample_fill_value), dtype=torch.float32)
    else:
        torch_dtype = _torch_dtype_from_dict(fill_value)
        fill_value_t = torch.tensor(0.5, dtype=torch_dtype)

    # Decide how to forward ``device`` to ttnn.full so the sweep trace
    # config_hash matches the master. Sample suite uses a boolean flag; the
    # model_traced suite reads the traced device value (mesh dict vs explicit
    # None vs absent key).
    if sample_device_was_traced is not None:
        device_state = "mesh" if sample_device_was_traced else "omit"
    else:
        device_state = _master_device_state(kwargs.get("traced_device_kwarg", _ABSENT))

    full_kwargs = dict(fill_value=fill_value_t, dtype=dtype_v, layout=layout_v)
    if device_state == "mesh":
        full_kwargs["device"] = device
    elif device_state == "none":
        # Master traced device=None explicitly. Tracer hashes "device": null
        # which differs from the kwarg being absent — pass it through.
        full_kwargs["device"] = None

    start_time = start_measuring_time()
    output = ttnn.full(shape_arg, **full_kwargs)
    e2e_perf = stop_measuring_time(start_time)

    # Read back and verify every element equals the fill value. The output
    # may be HOST (when device was None) or on-mesh; handle both.
    try:
        device_tensors = ttnn.get_device_tensors(output)
        torch_output = ttnn.to_torch(device_tensors[0]).float()
    except Exception:
        torch_output = ttnn.to_torch(output).float()

    # Trim to requested logical shape (ignore tile padding the op may add).
    expected_shape = tuple(logical_shape)
    if torch_output.ndim < len(expected_shape):
        torch_output = torch_output.reshape(expected_shape)
    else:
        slicing = tuple(slice(0, s) for s in expected_shape)
        torch_output = torch_output[slicing]

    fill_scalar = float(fill_value_t.item())
    diff = (torch_output - fill_scalar).abs()
    max_diff = float(diff.max().item()) if diff.numel() > 0 else 0.0

    # bfloat16 has ~1e-2 absolute error tolerance for arbitrary fills; float32
    # should match exactly.
    tol = 1e-5 if dtype_v == ttnn.float32 else 1e-2
    pcc_pass = max_diff <= tol
    msg = f"max abs diff vs fill={max_diff:.6g} (tol={tol})"
    return [(bool(pcc_pass), msg), e2e_perf]
