# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Patch embedder (input projection + learned 2D position table) vs HF on real images / video.

Compared over valid patches (padding rows hold `input_proj(-1)` in both HF and TT and are masked
downstream); the padding rows' position embedding must be exactly zero, checked via the HF rows.
"""

import pytest

from models.demos.embedding_gemma2.tests.common import PCC_LAYER, log_result, pcc
from models.demos.embedding_gemma2.tests.pcc.vision.vision_common import ALL_CASES, download_rows


@pytest.mark.parametrize("case", ALL_CASES)
def test_patch_embedder(case, tt_vision_encoder, vision_reference_for):
    pv, pid, ref = vision_reference_for(case)
    N = pid.shape[1]
    enc = tt_vision_encoder
    host = enc.prepare_inputs(pv, pid)
    x = enc.to_device(host)
    y = download_rows(enc.model.patch_embedder(x.pixel_values, x.pos_x, x.pos_y), N)
    valid = ref.valid
    assert (~valid).any(), "case has no padding patches"
    value = pcc(ref.patch_embeds[valid], y[valid])
    pad_value = pcc(ref.patch_embeds[~valid], y[~valid])
    log_result("vision_patch_embed_pad_rows_pcc", pad_value, PCC_LAYER, case=case)
    assert log_result(
        "vision_patch_embed_pcc", value, PCC_LAYER, case=case, n_valid=host.num_valid
    ), f"patch embedder PCC {value:.6f}"
    assert pad_value >= PCC_LAYER, f"padding rows (zero position embedding) PCC {pad_value:.6f}"
