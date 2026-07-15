# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Layer-0 prefill PCC vs HF ``SeamlessM4Tv2DecoderLayer`` (``models/tt_transformers/tests/test_decoder_prefill.py`` pattern).

Run: ``pytest …/test_prefill.py -v``
"""

from __future__ import annotations

import pytest

from models.experimental.seamless_m4t_v2_large.tests.pcc.decoder_pcc_common import (
    PREFILL_LAYER_SEQ_LENGTHS,
    load_hf_model_for_layer_pcc,
    run_prefill_layer_pcc,
)
from models.experimental.seamless_m4t_v2_large.tests.pcc.pcc_test_common import (
    legacy_device_params,
    legacy_mesh_device_param,
    weights_dir_or_skip,
)
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import mesh_default_device


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("max_seq_len", PREFILL_LAYER_SEQ_LENGTHS)
@pytest.mark.parametrize("mesh_device", [legacy_mesh_device_param()], indirect=True)
@pytest.mark.parametrize("device_params", [legacy_device_params()], indirect=True)
def test_prefill_pcc(mesh_device, device_params, reset_seeds, max_seq_len):
    """One prefill forward per seq length: random hidden → TT vs HF layer 0, PCC ≥ 0.99."""
    del device_params, reset_seeds
    with mesh_default_device(mesh_device):
        run_prefill_layer_pcc(
            mesh_device,
            load_hf_model_for_layer_pcc(weights_dir_or_skip()),
            seq_len=max_seq_len,
        )
