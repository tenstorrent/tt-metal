# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the Qwen3-TTS full-model PCC gates.

These live here rather than in ``qwen3_tts_full_model_pcc_common.py`` because a
fixture imported into two test modules is registered TWICE — pytest keys the
session cache on the definition it finds in each importing module's namespace, so
both ``test_qwen3_tts_prefill_pcc.py`` and ``test_qwen3_tts_decode_pcc.py`` would
open their own mesh device. The second open then fails outright:

    SetFabricConfig(FABRIC_1D) is not allowed while devices are still open

A conftest gives one definition for the whole directory, so the two files share a
single device and a single 40 s model load.

The names are prefixed ``pcc_`` on purpose: a conftest applies to every test in
this directory, and ``test_qwen3_tts_pcc.py`` already has its own ``device``
fixture that must not be shadowed.
"""

import pytest

import ttnn
from models.demos.qwen3_tts.tests.qwen3_tts_profile_demo_common import close_profile_device, hf_id, open_profile_device


@pytest.fixture(scope="session")
def pcc_device():
    dev, mesh_shape = open_profile_device()
    yield dev
    close_profile_device(dev, mesh_shape)


@pytest.fixture(scope="session")
def pcc_model(pcc_device):
    from models.demos.qwen3_tts.tt.model_config import talker_config_for_hf_id
    from models.demos.qwen3_tts.tt.qwen3_tts import Qwen3TTS
    from models.demos.qwen3_tts.tt.server import load_weights

    main_weights, _ = load_weights(hf_id())
    model = Qwen3TTS(device=pcc_device, state_dict=main_weights, talker_config=talker_config_for_hf_id(hf_id()))
    ttnn.synchronize_device(pcc_device)
    return model, main_weights
