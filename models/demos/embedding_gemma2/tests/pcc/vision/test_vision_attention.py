# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Vision attention (q/k norm, 2D axial RoPE, v_norm, padding mask, scaling 1.0) vs HF.

Each TT attention gets the HF `input_layernorm` output of that layer; layers 0, 7 and 15.
"""

import pytest

from models.demos.embedding_gemma2.tests.common import PCC_LAYER, log_result, pcc
from models.demos.embedding_gemma2.tests.pcc.vision.vision_common import LAYER_CASES, download_rows, upload_rows
from models.demos.embedding_gemma2.tt.vision.vision_model import key_padding_mask


@pytest.mark.parametrize("layer", (0, 7, 15))
@pytest.mark.parametrize("case", LAYER_CASES)
def test_vision_attention(case, layer, eg2_device, tt_vision_encoder, vision_reference_for):
    pv, pid, ref = vision_reference_for(case)
    N = pid.shape[1]
    enc = tt_vision_encoder
    host = enc.prepare_inputs(pv, pid)
    x = enc.to_device(host)
    mask = key_padding_mask(x.keep)
    attn = enc.model.layers[layer].self_attn
    y = download_rows(attn(upload_rows(eg2_device, ref.attn_in[layer], host.seq_len), mask, x.cos, x.sin), N)
    valid = ref.valid
    value = pcc(ref.attn_out[layer][valid], y[valid])
    assert log_result("vision_attention_pcc", value, PCC_LAYER, case=case, layer=layer), f"PCC {value:.6f}"
