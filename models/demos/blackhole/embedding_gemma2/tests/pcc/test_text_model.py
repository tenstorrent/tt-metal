# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full 24-layer text stack vs HF.

test_every_layer   teacher-forced: each TT layer gets the HF input of that layer; PCC per layer.
test_text_model    free-running TT forward (ids -> last hidden [B,L,768] -> pooled + normalized):
                   last-hidden PCC over valid tokens and per-prompt embedding cosine (bar 0.99).
"""

import pytest

from models.demos.blackhole.embedding_gemma2.tests.common import (
    CASES,
    COSINE_EMBEDDING,
    PCC_LAYER,
    case_id,
    cosine_rows,
    download,
    log_result,
    pcc,
    upload_activation,
    upload_inputs,
    valid_rows,
)

# Hard floor for the free-running last hidden state, asserted together with the per-prompt
# embedding cosine (bar 0.99). It is the binding metric of the E6a datatype sweep: measured min
# 0.99035 ([513]) with the default precision (selected_precision_config.json), 0.99692 with the
# E5a policy (baseline_precision_config.json); see doc/datatype_sweep/README.md.
PCC_LAST_HIDDEN = 0.99

PER_LAYER_CASES = ((1025,), (8192,), (1025, 513, 31))
BATCH8 = (512, 480, 400, 300, 257, 129, 64, 7)
LONGEST = (32768,)
E2E_CASES = CASES + (BATCH8, LONGEST)


@pytest.mark.parametrize("lengths", PER_LAYER_CASES, ids=[case_id(c) for c in PER_LAYER_CASES])
def test_every_layer(lengths, eg2_device, tt_text_model, reference_for):
    ids, mask, ref = reference_for(lengths)
    B, L = ids.shape
    m = tt_text_model
    ids_d, keep_d, S = upload_inputs(eg2_device, ids, mask, m.config.max_seq_len)
    m.prepare(S)
    masks = m.masks(keep_d)
    proj = m.per_layer_input(m.embedding(ids_d))
    failures = []
    for i, layer in enumerate(m.layers):
        pli = m.per_layer_input.layer_input(proj, i)
        x = upload_activation(eg2_device, ref.layer_in[i], S)
        y = layer(x, pli, masks[m.config.layer_types[i]])
        value = pcc(valid_rows(ref.layer_out[i], mask), valid_rows(download(y, B, L), mask))
        if not log_result("layer_pcc", value, PCC_LAYER, layer=i, kind=m.config.layer_types[i], case=list(lengths)):
            failures.append((i, value))
    assert not failures, f"layers below {PCC_LAYER}: {failures}"


@pytest.mark.parametrize("lengths", E2E_CASES, ids=[case_id(c) for c in E2E_CASES])
def test_text_model(lengths, eg2_device, tt_text_model, reference_for):
    ids, mask, ref = reference_for(lengths)
    B, L = ids.shape
    m = tt_text_model
    ids_d, keep_d, S = upload_inputs(eg2_device, ids, mask, m.config.max_seq_len)
    m.prepare(S)
    last, emb = m(ids_d, keep_d)

    last_t = download(last, B, L)
    p_last = pcc(valid_rows(ref.last_hidden, mask), valid_rows(last_t, mask))
    ok_last = log_result("last_hidden_pcc", p_last, PCC_LAST_HIDDEN, case=list(lengths), bucket=S)

    import ttnn

    emb_t = ttnn.to_torch(emb).float().reshape(B, -1)
    cos = cosine_rows(ref.embedding, emb_t)
    ok_cos = True
    for b in range(B):
        ok_cos &= log_result(
            "embedding_cosine", float(cos[b]), COSINE_EMBEDDING, case=list(lengths), row=b, length=int(mask[b].sum())
        )
    assert ok_last and ok_cos, f"last-hidden PCC {p_last:.6f}, cosines {cos.tolist()}"
