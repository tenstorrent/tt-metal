# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest

# The 24 traced shapes use 262 MiB together (measured, 7-26 MiB each); plus 50% margin, in 64 MiB steps.
TRACE_REGION_SIZE = 448 * 1024 * 1024


def load_example_state(example, ckpt_dir):
    if "state_file" in example:
        return (Path(ckpt_dir) / example["state_file"]).read_text()
    return example["state"]


def ids_sha256(input_ids):
    return hashlib.sha256(json.dumps(input_ids).encode()).hexdigest()


def parametrize_mesh():
    """One Blackhole chip (1x1 mesh), no fabric, a trace region for the short-shape traces."""
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE

    device_params = {"l1_small_size": GDN_CONV1D_L1_SMALL_SIZE, "trace_region_size": TRACE_REGION_SIZE}

    def decorator(fn):
        fn = pytest.mark.parametrize("device_params", [device_params], indirect=True)(fn)
        return pytest.mark.parametrize("mesh_device", [pytest.param((1, 1), id="1x1")], indirect=True)(fn)

    return decorator
