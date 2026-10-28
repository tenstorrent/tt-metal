# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TextEmbedding + PerLayerInput vs HF (`embed_tokens * sqrt(512)` and `ple(inputs_embeds)`)."""

import pytest

from models.demos.embedding_gemma2.tests.common import (
    CASE_IDS,
    CASES,
    PCC_LAYER,
    download,
    log_result,
    pcc,
    upload_inputs,
    valid_rows,
)


@pytest.mark.parametrize("lengths", CASES, ids=CASE_IDS)
def test_embedding_and_per_layer_input(lengths, eg2_device, tt_text_model, reference_for):
    ids, mask, ref = reference_for(lengths)
    B, L = ids.shape
    m = tt_text_model
    ids_d, keep_d, S = upload_inputs(eg2_device, ids, mask, m.config.max_seq_len)

    emb = m.embedding(ids_d)
    p_emb = pcc(valid_rows(ref.inputs_embeds, mask), valid_rows(download(emb, B, L), mask))
    ok = log_result("embedding_pcc", p_emb, PCC_LAYER, case=list(lengths))

    proj = m.per_layer_input(emb)
    worst = 1.0
    for i in range(m.config.num_layers):
        pli = m.per_layer_input.layer_input(proj, i)
        worst = min(worst, pcc(valid_rows(ref.per_layer_inputs[:, :, i], mask), valid_rows(download(pli, B, L), mask)))
    ok &= log_result("per_layer_input_min_pcc", worst, PCC_LAYER, case=list(lengths))
    assert ok, f"embedding {p_emb:.6f}, per-layer-input min {worst:.6f} (bar {PCC_LAYER})"
