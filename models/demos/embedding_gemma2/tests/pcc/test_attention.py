# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Attention vs HF for both layer kinds, fed the HF `input_layernorm` output of that layer.

layer 0: sliding (head_dim 256, 2 KV heads, |q-k| <= 512 band + padding mask)
layer 5: full    (head_dim 512, 1 KV head, padding mask)
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

LAYERS = {"sliding": 0, "full": 5}


@pytest.mark.parametrize("kind", list(LAYERS))
@pytest.mark.parametrize("lengths", CASES, ids=CASE_IDS)
def test_attention(kind, lengths, eg2_device, tt_text_model, reference_for):
    layer_idx = LAYERS[kind]
    ids, mask, ref = reference_for(lengths)
    B, L = ids.shape
    m = tt_text_model
    assert m.config.layer_types[layer_idx].startswith(kind)
    ids_d, keep_d, S = upload_inputs(eg2_device, ids, mask, m.config.max_seq_len)
    m.prepare(S)
    masks = m.masks(keep_d)

    x = upload_activation(eg2_device, ref.attn_in[layer_idx], S)
    y = m.layers[layer_idx].self_attn(x, masks[m.config.layer_types[layer_idx]])
    value = pcc(valid_rows(ref.attn_out[layer_idx], mask), valid_rows(download(y, B, L), mask))
    assert log_result(
        "attention_pcc", value, PCC_LAYER, kind=kind, layer=layer_idx, case=list(lengths)
    ), f"attention {kind} PCC {value:.6f} < {PCC_LAYER}"
