# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Layer-0 decode PCC vs HF ``SeamlessM4Tv2DecoderLayer`` (``models/tt_transformers/tests/test_decoder.py`` pattern).

Run: ``pytest …/test_decode.py -v``
"""

from __future__ import annotations

import pytest

from models.experimental.seamless_m4t_v2_large.tests.pcc.decoder_pcc_common import (
    load_hf_model_for_layer_pcc,
    run_decode_layer_pcc,
)
from models.experimental.seamless_m4t_v2_large.tests.pcc.pcc_test_common import (
    legacy_device_params,
    legacy_mesh_device_param,
    weights_dir_or_skip,
)
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import mesh_default_device


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [legacy_mesh_device_param()], indirect=True)
@pytest.mark.parametrize("device_params", [legacy_device_params()], indirect=True)
def test_decode_pcc(mesh_device, device_params, reset_seeds):
    """10 decode steps: random hidden per step, KV cache positions 0–9, PCC ≥ 0.99."""
    del device_params, reset_seeds
    with mesh_default_device(mesh_device):
        run_decode_layer_pcc(mesh_device, load_hf_model_for_layer_pcc(weights_dir_or_skip()))
