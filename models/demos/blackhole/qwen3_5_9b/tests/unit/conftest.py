# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the single-device component PCC unit tests.

These tests validate the Qwen3.5-9B single-device forward path against torch
references. ``setup`` (the real checkpoint loaded + remapped once per module) is
shared here so the split per-component files don't each duplicate the loader.

Component files open the device via the repo-root ``device`` fixture using the
``@pytest.mark.use_module_device`` marker — that path opens once per module through
``ttnn.CreateDevice`` with the standard ``get_updated_device_params`` defaults
(dispatch-core config, L1 sizing), which a bare ``ttnn.open_device(device_id=0)``
skips (that bare call fails TLB allocation on Blackhole). ``setup`` here depends on
the module-scoped device so the checkpoint is loaded only once per module.

We deliberately do NOT define a ``device`` fixture here: ``test_model.py`` needs the
function-scoped root ``device`` (parametrized via ``device_params`` for tracing), and
a conftest-level ``device`` would shadow it.
"""

import glob
import os

import pytest

from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs
from models.demos.blackhole.qwen3_5_9b.tt.weight_mapping import remap_qwen35_state_dict

# Single-device component tests run against the 9B checkpoint (27B needs a TP mesh).
os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.5-9B")

# Standard single-device config for this model (matches the prefill / generator tests);
# component files apply it via ``pytestmark = [..., pytest.mark.use_module_device(...)]``.
MODULE_DEVICE_PARAMS = {"l1_small_size": 24576, "num_command_queues": 2}


@pytest.fixture(scope="module")
def setup(_device_module_impl):
    """Load the real checkpoint once per module: returns (args, remapped_sd, raw_sd).

    Depends on the module-scoped ``_device_module_impl`` (the device the
    ``use_module_device`` marker opens) so the load and the device share module scope.
    """
    from safetensors import safe_open

    device = _device_module_impl
    args = Qwen35ModelArgs(mesh_device=device)
    raw = {}
    for path in sorted(glob.glob(f"{args.CKPT_DIR}/model.safetensors-*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                raw[key] = f.get_tensor(key)
    sd = remap_qwen35_state_dict(raw)
    return args, sd, raw
