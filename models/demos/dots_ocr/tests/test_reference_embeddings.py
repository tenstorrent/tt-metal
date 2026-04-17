# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import pytest
import torch

from models.demos.dots_ocr.reference.embeddings import build_inputs_embeds, image_token_mask


def test_image_token_mask():
    image_token_id = 151665
    input_ids = torch.tensor([[1, image_token_id, 3, image_token_id, 5]])
    m = image_token_mask(input_ids, image_token_id)
    assert m.dtype == torch.bool
    assert m.sum().item() == 2


class _StubDotsLike(torch.nn.Module):
    """Minimal surface for `build_inputs_embeds` text-only and fusion paths."""

    def __init__(self, vocab: int = 320, hidden: int = 16, image_token_id: int = 99):
        super().__init__()
        self.config = type("C", (), {"image_token_id": image_token_id})()
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.hidden = hidden

    def get_input_embeddings(self):
        return self.embed

    def prepare_inputs_embeds(self, input_ids, pixel_values, grid_thw, img_mask):
        # Return deterministic fused tensor: text embeds + fake vision scatter
        te = self.embed(input_ids)
        # Simulate vision rows matching number of True in img_mask
        n = int(img_mask.sum().item())
        ve = torch.ones(n, self.hidden, device=input_ids.device, dtype=te.dtype) * 0.5
        out = te.clone()
        b, s, d = out.shape
        flat = out.view(-1, d)
        flat[img_mask.view(-1)] = ve
        return out


def test_build_inputs_embeds_stub_fusion():
    torch.manual_seed(0)
    m = _StubDotsLike()
    input_ids = torch.tensor([[1, 99, 3, 99]])
    pv = torch.randn(1, 3, 4, 4)
    grid = torch.tensor([[1, 2, 2]])
    out = build_inputs_embeds(m, input_ids, pv, grid)
    assert out.shape == (1, 4, m.hidden)
    # Vision positions overwritten
    assert not torch.allclose(out[0, 1], m.embed(torch.tensor([99]))[0, 0])


def test_build_inputs_embeds_stub_text_only():
    m = _StubDotsLike()
    input_ids = torch.tensor([[1, 2, 3]])
    out = build_inputs_embeds(m, input_ids, None, None)
    assert torch.allclose(out, m.get_input_embeddings()(input_ids))


@pytest.mark.skipif(
    os.environ.get("RUN_DOTS_HF_TESTS") != "1", reason="Set RUN_DOTS_HF_TESTS=1 and HF access for HF parity test"
)
def test_build_inputs_embeds_matches_hf_prepare():
    """Parity: our helper vs direct `prepare_inputs_embeds` on real Dots weights."""
    import numpy as np
    from PIL import Image

    from models.demos.dots_ocr.reference.hf_utils import HFLoadSpec, load_processor_and_model

    spec = HFLoadSpec(model_id=os.environ.get("HF_MODEL", "rednote-hilab/dots.mocr"))
    processor, model = load_processor_and_model(spec)
    model.eval()
    image_token_id = int(model.config.image_token_id)
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    inputs = processor(images=img, text="<|image_pad|>", return_tensors="pt")
    input_ids = inputs["input_ids"]
    pv = inputs["pixel_values"]
    grid = inputs["image_grid_thw"]
    img_mask = input_ids == image_token_id
    with torch.no_grad():
        ref = model.prepare_inputs_embeds(input_ids, pv, grid, img_mask)
        ours = build_inputs_embeds(model, input_ids, pv, grid)
    assert ref.shape == ours.shape
    assert torch.allclose(ref, ours, atol=1e-5, rtol=1e-4)
