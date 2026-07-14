# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the single-device component PCC unit tests.

These tests validate the Qwen3.5-9B single-device forward path against torch
references. The component files all open the device via the repo-root ``device``
fixture parametrized with ``DEVICE_PARAMS`` (same pattern as test_model.py /
test_prefill.py), so device acquisition is uniform across the whole suite and goes
through the managed ``ttnn.CreateDevice`` path.

``setup`` (the real checkpoint loaded + remapped) is shared here so the per-component
files don't each duplicate the loader. It's function-scoped — it depends on the
function-scoped root ``device`` — so the checkpoint is reloaded per test (only
test_layer.py has >1 test, so this costs a couple of extra loads vs. module scope).
"""

import glob
import os

import pytest

from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
from models.demos.blackhole.qwen36.tt.weight_mapping import remap_qwen36_state_dict

# Single-device component tests run against the 9B checkpoint (27B needs a TP mesh).
os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.5-9B")

# Standard single-device config (matches the prefill / generator tests). Component files apply it
# via ``pytestmark = [..., pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)]``.
DEVICE_PARAMS = [{"l1_small_size": 24576, "num_command_queues": 2}]


@pytest.fixture
def setup(device):
    """Load the real checkpoint for the current single device: returns (args, remapped_sd, raw_sd)."""
    from safetensors import safe_open

    args = Qwen36ModelArgs(mesh_device=device)
    raw = {}
    for path in sorted(glob.glob(f"{args.CKPT_DIR}/model.safetensors-*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                raw[key] = f.get_tensor(key)
    sd = remap_qwen36_state_dict(raw)
    return args, sd, raw
