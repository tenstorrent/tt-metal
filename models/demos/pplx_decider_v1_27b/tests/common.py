# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest


def load_example_state(example, ckpt_dir):
    if "state_file" in example:
        return (Path(ckpt_dir) / example["state_file"]).read_text()
    return example["state"]


def ids_sha256(input_ids):
    return hashlib.sha256(json.dumps(input_ids).encode()).hexdigest()


def parametrize_mesh_traced():
    """Like qwen36 parametrize_mesh_tp, with a trace region for capture_prefill_trace."""
    # Local imports keep CPU-only tests free of the device stack.
    import ttnn
    from models.demos.blackhole.qwen36.tests.test_factory import _resolve_mesh_shape
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE
    from models.demos.pplx_decider_v1_27b.tt.model import TRACE_REGION_SIZE

    shape = _resolve_mesh_shape()
    device_params = {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        "l1_small_size": GDN_CONV1D_L1_SMALL_SIZE,
        "trace_region_size": TRACE_REGION_SIZE,
    }

    def decorator(fn):
        fn = pytest.mark.parametrize("device_params", [device_params], indirect=True)(fn)
        return pytest.mark.parametrize(
            "mesh_device", [pytest.param(shape, id=f"{shape[0]}x{shape[1]}")], indirect=True
        )(fn)

    return decorator
