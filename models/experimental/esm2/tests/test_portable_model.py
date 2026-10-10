# SPDX-License-Identifier: MIT
"""Portable CPU tests: config/shape logic, reference twin invariants (no device)."""

from __future__ import annotations

import pytest
import torch

from models.experimental.esm2.tt.esm2.config import Esm2TTConfig
from models.experimental.esm2.tt.esm2.reference_layers import Esm2Model


@pytest.fixture(scope="module")
def small_cfg():
    return Esm2TTConfig(
        num_hidden_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        intermediate_size=256,
        vocab_size=33,
        layer_norm_eps=1e-5,
        pad_token_id=1,
        mask_token_id=32,
        max_position_embeddings=128,
    )


@pytest.fixture(scope="module")
def model(small_cfg):
    torch.manual_seed(42)
    m = Esm2Model(small_cfg)
    # init embedding table randomly so input-dependence is meaningful
    with torch.no_grad():
        m.embeddings.weight.normal_(0, 0.02)
    m.eval()
    return m


def _make_ids(small_cfg, batch=2, length=32):
    ids = torch.randint(3, small_cfg.vocab_size - 1, (batch, length))
    ids[:, 0] = 0  # <cls>
    ids[:, -1] = 2  # <eos>
    mask = torch.ones(batch, length, dtype=torch.long)
    return ids, mask


class TestReferenceTwin:
    def test_determinism(self, model, small_cfg):
        ids, mask = _make_ids(small_cfg)
        with torch.no_grad():
            out1 = model(ids, mask)
            out2 = model(ids, mask)
        for a, b in zip(out1, out2):
            assert torch.equal(a, b), "FP32 twin must be bit-deterministic"

    def test_output_shapes(self, model, small_cfg):
        ids, mask = _make_ids(small_cfg)
        with torch.no_grad():
            logits, hidden = model(ids, mask)
        assert logits.shape == (ids.shape[0], ids.shape[1], small_cfg.vocab_size)
        assert hidden.shape == (ids.shape[0], ids.shape[1], small_cfg.hidden_size)

    def test_trailing_pad_invariance(self, model, small_cfg):
        ids, mask = _make_ids(small_cfg, batch=1, length=32)
        ids_padded = torch.cat([ids, torch.full((1, 16), small_cfg.pad_token_id)], dim=1)
        mask_padded = torch.cat([mask, torch.zeros(1, 16, dtype=torch.long)], dim=1)
        with torch.no_grad():
            logits_plain, _ = model(ids, mask)
            logits_padded, _ = model(ids_padded, mask_padded)
        # real positions must be bitwise identical (pad contributions are exact zeros)
        assert torch.equal(
            logits_plain, logits_padded[:, :32]
        ), "Appending pad rows must not change real-position outputs"

    def test_mask_sensitivity(self, model, small_cfg):
        ids, mask = _make_ids(small_cfg, batch=1)
        ids_masked = ids.clone()
        ids_masked[0, 5] = small_cfg.mask_token_id
        with torch.no_grad():
            logits_plain, _ = model(ids, mask)
            logits_masked, _ = model(ids_masked, mask)
        delta = (logits_masked - logits_plain).abs().max().item()
        assert delta > 0.01, f"Masking a residue must change that position's logits (delta={delta:.6f})"


class TestConfig:
    def test_from_json(self, checkpoint):
        cfg = Esm2TTConfig.from_json_file(str(checkpoint / "config.json"))
        assert cfg.num_hidden_layers == 33
        assert cfg.hidden_size == 1280
        assert cfg.num_attention_heads == 20
        assert cfg.intermediate_size == 5120
        assert cfg.vocab_size == 33
        assert cfg.position_embedding_type == "rotary"
