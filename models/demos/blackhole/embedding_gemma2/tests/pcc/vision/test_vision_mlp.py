# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Vision MLP (gated gelu-tanh 768 -> 3072 -> 768) vs HF, fed the HF `pre_feedforward_layernorm` output."""

import pytest

from models.demos.blackhole.embedding_gemma2.tests.common import PCC_LAYER, log_result, pcc
from models.demos.blackhole.embedding_gemma2.tests.pcc.vision.vision_common import (
    LAYER_CASES,
    download_rows,
    upload_rows,
)
from models.demos.blackhole.embedding_gemma2.tt.vision.config import patch_bucket_for


@pytest.mark.parametrize("layer", (0, 7, 15))
@pytest.mark.parametrize("case", LAYER_CASES)
def test_vision_mlp(case, layer, eg2_device, tt_vision_model, vision_reference_for):
    pv, pid, ref = vision_reference_for(case)
    N = pid.shape[1]
    S = patch_bucket_for(N)
    y = download_rows(tt_vision_model.layers[layer].mlp(upload_rows(eg2_device, ref.mlp_in[layer], S)), N)
    valid = ref.valid
    value = pcc(ref.mlp_out[layer][valid], y[valid])
    assert log_result("vision_mlp_pcc", value, PCC_LAYER, case=case, layer=layer), f"PCC {value:.6f}"
