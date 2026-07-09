# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import re

import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_model_traced_mesh_shape,
    create_mesh_device,
    mesh_tensor_to_torch,
    get_mesh_composer,
)

# Import V2 master config loader for traced model configurations
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader

# Override the default timeout in seconds for hang detection.
TIMEOUT = 300

# Load traced configurations from real model tests (V2 format). ttnn.full has no
# dedicated parameter extractor, so the loader returns the generic packed form:
# the run() args (arg0=shape, fill_value, dtype, layout) arrive as kwargs.
loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("full")

# Parameters provided to the test vector generator are defined here.
parameters = {
    "model_traced_sample": {
        "input_a_shape": [(1, 1, 32, 32)],
        "fill_value": [1.0],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "storage_type": ["StorageType::DEVICE"],
    },
}

if model_traced_params:
    parameters["model_traced"] = model_traced_params


# ttnn.full records its `device` arg in the trace (None for host creation, or a
# MeshDevice of a specific shape). The harness strips the vector's `device` key
# before run() and injects the fixture device, so run() can't see the traced
# device directly — recover it per-config from the master by config_hash and
# reproduce it (host = no device arg; otherwise open the matching mesh shape).
def _build_hash_to_device_shape():
    # The per-config device arg lives only in the validation master
    # (tests/sweep_framework/ttnn_operations_master.json); the runtime
    # auto-discovered master has no ttnn.full. Read it directly (path relative
    # to this file), honoring TTNN_MASTER_JSON_PATH when set.
    import os
    import json as _json

    mapping = {}
    candidates = []
    env_path = os.environ.get("TTNN_MASTER_JSON_PATH")
    if env_path:
        candidates.append(env_path)
    candidates.append(os.path.join(os.path.dirname(__file__), "..", "..", "ttnn_operations_master.json"))
    for path in candidates:
        try:
            if not os.path.isfile(path):
                continue
            with open(path) as _f:
                ops = _json.load(_f).get("operations", {})
            cfgs = ops.get("ttnn.full", {}).get("configurations", [])
            if not cfgs:
                continue
            for c in cfgs:
                dev = c.get("arguments", {}).get("device")
                ch = c.get("config_hash")
                if not ch:
                    continue
                if dev is None:
                    mapping[ch] = None  # host creation (no device arg)
                else:
                    rep = dev.get("repr", "") if isinstance(dev, dict) else str(dev)
                    m = re.search(r"MeshDevice\((\d+)x(\d+)\s+grid", rep)
                    mapping[ch] = (int(m.group(1)), int(m.group(2))) if m else "fixture"
            if mapping:
                break
        except Exception:
            continue
    return mapping


_HASH_TO_DEVICE_SHAPE = _build_hash_to_device_shape()

_CUR_DEVICE = None
_CUR_SHAPE = "__uninit__"


def _close_vector_device():
    global _CUR_DEVICE, _CUR_SHAPE
    if _CUR_DEVICE is not None:
        try:
            ttnn.close_mesh_device(_CUR_DEVICE)
        except Exception:
            # best-effort teardown; a close failure must not mask the test result
            pass
    _CUR_DEVICE = None
    _CUR_SHAPE = "__uninit__"


def _ensure_vector_device(shape):
    """Open (and cache) a mesh device of the given (rows, cols) shape, or None
    for host creation."""
    global _CUR_DEVICE, _CUR_SHAPE
    if shape is None:
        return None
    if _CUR_DEVICE is None or shape != _CUR_SHAPE:
        _close_vector_device()
        _CUR_DEVICE = create_mesh_device(shape)
        _CUR_SHAPE = shape
    return _CUR_DEVICE


def mesh_device_fixture():
    # Device opened per-vector in run() (see _ensure_vector_device) so each
    # config's traced device shape (or host) can be reproduced.
    yield (None, "wormhole_b0")
    _close_vector_device()


def _parse_shape(value):
    """Parse the traced arg0 into a shape tuple.

    Accepts {"type": "Shape", "value": "Shape([1, 1, 1024, 64])"}, a list/tuple,
    or the structured-sample input_a_shape.
    """
    if isinstance(value, (tuple, list)):
        return tuple(value)
    if isinstance(value, dict):
        s = str(value.get("value", ""))
        m = re.search(r"\[([\d,\s]+)\]", s)
        if m:
            return tuple(int(x) for x in m.group(1).split(",") if x.strip())
    return None


def _parse_dtype(value, default=ttnn.bfloat16):
    if value is None or value == "__ABSENT__":
        return default
    if isinstance(value, ttnn.DataType):
        return value
    s = value.get("repr", "") if isinstance(value, dict) else str(value)
    from tests.sweep_framework.master_config_loader_v2 import parse_dtype as _pd

    return _pd(s) or default


def _parse_layout(value, default=ttnn.TILE_LAYOUT):
    if value is None or value == "__ABSENT__":
        return default
    if isinstance(value, ttnn.Layout):
        return value
    s = value.get("repr", "") if isinstance(value, dict) else str(value)
    from tests.sweep_framework.master_config_loader_v2 import parse_layout as _pl

    return _pl(s) or default


def _parse_fill_value(value, default=1.0):
    # The master records fill_value as a 0-dim torch.Tensor
    # ({"type": "torch.Tensor", "shape": [], "dtype": "torch.float32"}) without
    # the concrete scalar; default the scalar but keep the torch.Tensor wrapper
    # so the recorded fill_value arg matches the master (passing a bare float
    # records it as a float -> a value diff).
    if isinstance(value, (int, float)):
        return float(value)
    return default


def run(
    # Structured-sample params (model_traced_sample suite)
    input_a_shape=None,
    input_a_dtype=None,
    input_a_layout=None,
    fill_value=None,
    storage_type="StorageType::DEVICE",
    # Generic packed params (model_traced suite): arg0/dtype/layout arrive here
    arg0=None,
    dtype=None,
    layout=None,
    *,
    device,
    **kwargs,
) -> list:
    torch.manual_seed(0)

    # Reproduce this config's traced `device` arg (stripped by the harness):
    # None -> host creation; a (rows, cols) -> open that mesh; "fixture"/unknown
    # -> the model-traced mesh (MESH_DEVICE_SHAPE).
    config_hash = kwargs.get("config_hash")
    traced_dev = _HASH_TO_DEVICE_SHAPE.get(config_hash, "fixture")
    if traced_dev is None:
        device = None
    elif isinstance(traced_dev, tuple):
        device = _ensure_vector_device(traced_dev)
    else:
        device = _ensure_vector_device(get_model_traced_mesh_shape())

    is_mesh_device = device is not None and hasattr(device, "get_num_devices")

    # Resolve shape: structured sample provides input_a_shape; the traced
    # generic form provides arg0 (a Shape).
    shape = _parse_shape(input_a_shape if input_a_shape is not None else arg0)
    if shape is None:
        return [(False, "Could not resolve shape for ttnn.full"), 0.0]

    full_dtype = _parse_dtype(dtype if dtype is not None else input_a_dtype)
    full_layout = _parse_layout(layout if layout is not None else input_a_layout)
    fv = _parse_fill_value(fill_value)
    # Pass fill_value as a 0-dim float32 torch.Tensor so the recorded arg matches
    # the master (which traced a torch.Tensor scalar, not a Python float).
    fv_arg = torch.tensor(fv, dtype=torch.float32)

    # Golden: a tensor of `shape` filled with `fv`.
    torch_output_tensor = torch.full(shape, fv, dtype=torch.float32)

    # Host creation when the traced device arg was None (device resolved above);
    # otherwise create on the reproduced mesh device.
    start_time = start_measuring_time()
    if device is None:
        output_tensor = ttnn.full(shape, fill_value=fv_arg, dtype=full_dtype, layout=full_layout)
    else:
        output_tensor = ttnn.full(shape, fill_value=fv_arg, dtype=full_dtype, layout=full_layout, device=device)
    mesh_composer = get_mesh_composer(device, None) if is_mesh_device else None
    output_tensor = mesh_tensor_to_torch(output_tensor, device if is_mesh_device else None, mesh_composer=mesh_composer)
    e2e_perf = stop_measuring_time(start_time)

    pcc = check_with_pcc(torch_output_tensor, output_tensor, 0.999)

    return [pcc, e2e_perf]
