# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.blackhole.pp_ocrv6.common import (
    DET_L1_SMALL_SIZE,
    REC_HEIGHT,
    REC_WIDTH,
    ctc_greedy_decode,
    load_rec_charset,
    load_torch_rec,
    rec_normalize_bgr,
    synthetic_gate_image,
)
from models.demos.blackhole.pp_ocrv6.tt.ttnn_rec import TtPPOCRv6SmallRec
from tests.ttnn.utils_for_testing import check_with_pcc

TEXTS = ["TRLU 482913", "MSCU 7731045", "53 FT 0921", "XTRA 51207", "UNIT 4417", "JBHU 238811", "GATE 3", "HUB 99"]


def _crops(batch):
    crops = []
    for text in TEXTS[:batch]:
        img = synthetic_gate_image(text, size=(1600, 720))
        dark = (img.max(-1) < 60).nonzero()
        y0, x0 = dark[0].min(), dark[1].min()
        y1, x1 = dark[0].max(), dark[1].max()
        crops.append(rec_normalize_bgr(img[y0 - 12 : y1 + 12, x0 - 12 : x1 + 12]))
    return torch.cat(crops), TEXTS[:batch]


@pytest.mark.parametrize("device_params", [{"l1_small_size": DET_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("source", ["text_crop", "random"])
def test_rec(device, batch, source):
    torch.manual_seed(0)
    torch_model = load_torch_rec()
    if source == "random":
        x, texts = torch.rand(batch, 3, REC_HEIGHT, REC_WIDTH) * 2 - 1, None
    else:
        x, texts = _crops(batch)
    with torch.no_grad():
        ref = torch_model(x)

    tt_model = TtPPOCRv6SmallRec(device, torch_model)
    tt_x = ttnn.from_torch(x.permute(0, 2, 3, 1), dtype=ttnn.bfloat16)
    out = ttnn.to_torch(tt_model(tt_x, batch, REC_HEIGHT, REC_WIDTH)).float().reshape(ref.shape)

    passed, pcc = check_with_pcc(ref, out, pcc=0.99)
    step_match = (out.argmax(-1) == ref.argmax(-1)).float().mean().item()
    print(f"[{source} b{batch}] PCC {pcc}  per-step argmax match {step_match:.2%}")
    if texts:
        charset = load_rec_charset()
        for i, expected in enumerate(texts):
            ref_text, _ = ctc_greedy_decode(ref[i], charset)
            tt_text, conf = ctc_greedy_decode(out[i], charset)
            print(f"  expected {expected!r}  torch {ref_text!r}  ttnn {tt_text!r} ({conf:.3f})")
            assert tt_text == ref_text
    assert passed, pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": DET_L1_SMALL_SIZE}], indirect=True)
def test_rec_mixed_batches(device):
    """One model instance serving several batch sizes, as the pipeline does."""
    torch_model = load_torch_rec()
    tt_model = TtPPOCRv6SmallRec(device, torch_model)
    x8, _ = _crops(8)
    for batch in (8, 1, 8):
        x = x8[:batch]
        with torch.no_grad():
            ref = torch_model(x)
        tt_x = ttnn.from_torch(x.permute(0, 2, 3, 1), dtype=ttnn.bfloat16)
        out = ttnn.to_torch(tt_model(tt_x, batch, REC_HEIGHT, REC_WIDTH)).float().reshape(ref.shape)
        passed, pcc = check_with_pcc(ref, out, pcc=0.99)
        print(f"[mixed b{batch}] PCC {pcc}")
        assert passed, pcc
