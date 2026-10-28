# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One EncoderLayer of each kind vs HF (sandwich norms + PLE block + layer_scalar).

The layer gets the HF hidden state entering it and the TT per-layer input computed from the TT
embedding (so the PLE path is exercised end to end); output is compared with the HF layer output.
"""

import pytest

from models.demos.embedding_gemma2.tests.common import (
    CASE_IDS,
    CASES,
    PCC_LAYER,
    download,
    log_result,
    pcc,
    upload_activation,
    upload_inputs,
    valid_rows,
)


@pytest.mark.parametrize("layer_idx", [0, 5], ids=["layer0_sliding", "layer5_full"])
@pytest.mark.parametrize("lengths", CASES, ids=CASE_IDS)
def test_encoder_layer(layer_idx, lengths, eg2_device, tt_text_model, reference_for):
    ids, mask, ref = reference_for(lengths)
    B, L = ids.shape
    m = tt_text_model
    ids_d, keep_d, S = upload_inputs(eg2_device, ids, mask, m.config.max_seq_len)
    m.prepare(S)
    masks = m.masks(keep_d)
    proj = m.per_layer_input(m.embedding(ids_d))
    pli = m.per_layer_input.layer_input(proj, layer_idx)

    x = upload_activation(eg2_device, ref.layer_in[layer_idx], S)
    y = m.layers[layer_idx](x, pli, masks[m.config.layer_types[layer_idx]])
    value = pcc(valid_rows(ref.layer_out[layer_idx], mask), valid_rows(download(y, B, L), mask))
    assert log_result(
        "encoder_layer_pcc",
        value,
        PCC_LAYER,
        layer=layer_idx,
        kind=m.config.layer_types[layer_idx],
        case=list(lengths),
    ), f"layer {layer_idx} PCC {value:.6f} < {PCC_LAYER}"
