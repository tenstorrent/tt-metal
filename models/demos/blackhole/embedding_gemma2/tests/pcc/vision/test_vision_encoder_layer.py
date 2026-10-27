# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Vision encoder layer and the full 16-layer stack vs HF.

test_one_layer          layer 0 on every case (teacher-forced: HF layer input).
test_every_layer        all 16 layers teacher-forced on LAYER_CASES; PCC per layer.
test_encoder_stack      free-running patch embedder + 16 layers on every case; encoder-output PCC
                        over valid patches (regression floor; the acceptance bar is the soft-token
                        PCC in test_vision_model.py, see doc/functional_vision/README.md).
"""

import pytest

from models.demos.blackhole.embedding_gemma2.tests.common import PCC_LAYER, log_result, pcc
from models.demos.blackhole.embedding_gemma2.tests.pcc.vision.vision_common import (
    ALL_CASES,
    LAYER_CASES,
    download_rows,
    upload_rows,
)
from models.demos.blackhole.embedding_gemma2.tt.vision.vision_model import key_padding_mask

# Floor for the free-running encoder output (measured 0.9974-0.9998; HF's own bf16 run scores
# 0.9972-0.9996 against the fp32 reference on the same images).
PCC_ENCODER_OUT = 0.99


def _layer_pcc(eg2_device, enc, case, ref, pv, pid, layers):
    N = pid.shape[1]
    host = enc.prepare_inputs(pv, pid)
    x = enc.to_device(host)
    mask = key_padding_mask(x.keep)
    out = {}
    for i in layers:
        y = enc.model.layers[i](upload_rows(eg2_device, ref.layer_in[i], host.seq_len), mask, x.cos, x.sin)
        out[i] = pcc(ref.layer_out[i][ref.valid], download_rows(y, N)[ref.valid])
    return out


@pytest.mark.parametrize("case", ALL_CASES)
def test_one_layer(case, eg2_device, tt_vision_encoder, vision_reference_for):
    pv, pid, ref = vision_reference_for(case)
    value = _layer_pcc(eg2_device, tt_vision_encoder, case, ref, pv, pid, (0,))[0]
    assert log_result("vision_layer_pcc", value, PCC_LAYER, case=case, layer=0), f"PCC {value:.6f}"


@pytest.mark.parametrize("case", LAYER_CASES)
def test_every_layer(case, eg2_device, tt_vision_encoder, vision_reference_for):
    pv, pid, ref = vision_reference_for(case)
    values = _layer_pcc(eg2_device, tt_vision_encoder, case, ref, pv, pid, range(16))
    failures = [
        (i, v) for i, v in values.items() if not log_result("vision_layer_pcc", v, PCC_LAYER, case=case, layer=i)
    ]
    assert not failures, f"layers below {PCC_LAYER}: {failures}"


@pytest.mark.parametrize("case", ALL_CASES)
def test_encoder_stack(case, tt_vision_encoder, vision_reference_for):
    pv, pid, ref = vision_reference_for(case)
    enc = tt_vision_encoder
    x = enc.to_device(enc.prepare_inputs(pv, pid))
    h = download_rows(enc.model.forward_hidden(x), pid.shape[1])
    value = pcc(ref.encoder_out[ref.valid], h[ref.valid])
    assert log_result("vision_encoder_out_pcc", value, PCC_ENCODER_OUT, case=case), f"PCC {value:.6f}"
