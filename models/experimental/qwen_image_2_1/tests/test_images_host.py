# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only: condition-image preprocessing and edit prompt template vs the diffusers editing goldens."""
import os

import pytest
import torch
from PIL import Image

from models.experimental.qwen_image_2_1.common import images as im
from models.experimental.qwen_image_2_1.common.config import DROP_IDX, GOLDENS_DIR

EDIT = os.path.join(GOLDENS_DIR, "edit")
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(EDIT, "text_encoder_edit.pt")), reason="edit goldens missing"
)


def test_condition_preprocess_matches_reference():
    t = torch.load(os.path.join(EDIT, "text_encoder_edit.pt"), weights_only=False)
    v = torch.load(os.path.join(EDIT, "vae_encode.pt"), weights_only=False)
    import json

    meta = json.load(open(os.path.join(EDIT, "meta.json")))
    img = Image.open(meta["image"][0])
    white, vae_t, (w, h) = im.prepare_condition_image(img, 1024)
    assert (w, h) == tuple(meta["cond_size"])
    assert vae_t.shape == tuple(v["vae_in"].shape)
    d = (vae_t.float() - v["vae_in"].float()).abs()
    assert d.max() < 0.02, float(d.max())  # bf16 golden vs fp32 host
    ref_white = t["cond_rgb_white"]
    got = torch.from_numpy(__import__("numpy").asarray(white))
    assert got.shape == ref_white.shape and (got.int() - ref_white.int()).abs().max() <= 1
    assert im.edit_prompt_text(t["prompt"], 1) == t["template_text"]
    mask = im.image_pad_mask_from_ids(t["input_ids"], DROP_IDX)
    assert torch.equal(mask, t["image_pad_mask"][0].bool())
