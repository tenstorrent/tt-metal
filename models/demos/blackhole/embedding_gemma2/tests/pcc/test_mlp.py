# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MLP vs HF (gated gelu-tanh 512 -> 2048 -> 512), fed the HF `pre_feedforward_layernorm` output."""

import pytest

from models.demos.blackhole.embedding_gemma2.tests.common import (
    CASE_IDS,
    CASES,
    PCC_LAYER,
    download,
    log_result,
    pcc,
    upload_activation,
    valid_rows,
)
from models.demos.blackhole.embedding_gemma2.tt.config import bucket_for


@pytest.mark.parametrize("layer_idx", [0, 5], ids=["layer0_sliding", "layer5_full"])
@pytest.mark.parametrize("lengths", CASES, ids=CASE_IDS)
def test_mlp(layer_idx, lengths, eg2_device, tt_text_model, reference_for):
    ids, mask, ref = reference_for(lengths)
    B, L = ids.shape
    m = tt_text_model
    S = bucket_for(L, m.config.max_seq_len)
    x = upload_activation(eg2_device, ref.mlp_in[layer_idx], S)
    y = m.layers[layer_idx].mlp(x)
    value = pcc(valid_rows(ref.mlp_out[layer_idx], mask), valid_rows(download(y, B, L), mask))
    assert log_result(
        "mlp_pcc", value, PCC_LAYER, layer=layer_idx, case=list(lengths)
    ), f"mlp layer {layer_idx} PCC {value:.6f} < {PCC_LAYER}"
