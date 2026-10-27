# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Conformer layers vs HF (teacher forced, real audio).

test_conformer_layer0   layer 0 on every case (incl. crops around the 12-token chunk)
test_conformer_stack    all 12 layers, each fed the HF input of that layer, on the layer cases
"""

import pytest

from models.demos.blackhole.embedding_gemma2.tests.common import PCC_LAYER, log_result, pcc
from models.demos.blackhole.embedding_gemma2.tests.pcc.audio.audio_common import (
    ALL_CASES,
    LAYER_CASES,
    device_inputs,
    download_rows,
    upload_rows,
    valid,
)


@pytest.mark.parametrize("case", ALL_CASES + ("crop13", "crop25"))
def test_conformer_layer0(case, eg2_device, tt_audio_encoder, audio_reference_for):
    feats, mask, ref = audio_reference_for(case)
    host, x = device_inputs(tt_audio_encoder, feats, mask)
    N = ref.token_mask.shape[1]
    L = tt_audio_encoder.model.layers[0]
    y = download_rows(L(upload_rows(eg2_device, ref.layer_in[0], host.seq_len), x.attn_mask), N)
    value = pcc(*valid(ref.layer_out[0], y, ref.token_mask))
    assert log_result("audio_layer_pcc", value, PCC_LAYER, case=case, layer=0)


@pytest.mark.parametrize("case", LAYER_CASES)
def test_conformer_stack(case, eg2_device, tt_audio_encoder, audio_reference_for):
    feats, mask, ref = audio_reference_for(case)
    host, x = device_inputs(tt_audio_encoder, feats, mask)
    N = ref.token_mask.shape[1]
    log_result("audio_hf_max_abs_activation", ref.max_abs_activation, 0.0, case=case)
    ok = True
    for i, L in enumerate(tt_audio_encoder.model.layers):
        y = download_rows(L(upload_rows(eg2_device, ref.layer_in[i], host.seq_len), x.attn_mask), N)
        ok &= log_result(
            "audio_layer_pcc", pcc(*valid(ref.layer_out[i], y, ref.token_mask)), PCC_LAYER, case=case, layer=i
        )
    assert ok
