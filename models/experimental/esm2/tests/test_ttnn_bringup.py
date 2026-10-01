# SPDX-License-Identifier: MIT
"""On-device bring-up tests with random weights (no checkpoint required)."""

from __future__ import annotations

import pytest
import torch

from models.experimental.esm2.tt.esm2.config import Esm2TTConfig
from models.experimental.esm2.tt.esm2.reference_layers import Esm2Model
from models.experimental.esm2.tt.esm2.ttnn_backend import TtnnEsm2


def _make_cfg():
    """Small config with head_dim=32 (TT tile multiple)."""
    return Esm2TTConfig(
        num_hidden_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        intermediate_size=256,
        vocab_size=33,
        layer_norm_eps=1e-5,
        pad_token_id=1,
        mask_token_id=32,
        max_position_embeddings=256,
    )


def _make_weights(cfg):
    torch.manual_seed(42)
    H, V, I = cfg.hidden_size, cfg.vocab_size, cfg.intermediate_size
    w = {"embeddings.word_embeddings.weight": torch.randn(V, H) * 0.02}
    for i in range(cfg.num_hidden_layers):
        p = f"layers.{i}."
        for nm, shape in [
            ("attn.q.weight", (H, H)),
            ("attn.q.bias", (H,)),
            ("attn.k.weight", (H, H)),
            ("attn.k.bias", (H,)),
            ("attn.v.weight", (H, H)),
            ("attn.v.bias", (H,)),
            ("attn_out.weight", (H, H)),
            ("attn_out.bias", (H,)),
            ("ffn1.weight", (I, H)),
            ("ffn1.bias", (I,)),
            ("ffn2.weight", (H, I)),
            ("ffn2.bias", (H,)),
        ]:
            w[p + nm] = torch.randn(*shape) * 0.02 if "weight" in nm else torch.zeros(shape)
        for nm in ["ln_attn", "ln_ffn"]:
            w[p + nm + ".weight"] = torch.ones(H)
            w[p + nm + ".bias"] = torch.zeros(H)
    w["final_ln.weight"] = torch.ones(H)
    w["final_ln.bias"] = torch.zeros(H)
    w["lm.dense.weight"] = torch.randn(H, H) * 0.02
    w["lm.dense.bias"] = torch.zeros(H)
    w["lm.ln.weight"] = torch.ones(H)
    w["lm.ln.bias"] = torch.zeros(H)
    w["lm.bias"] = torch.zeros(V)
    return w


def _nrmse(ref, got):
    ref, got = ref.float(), got.float()
    return ((got - ref).norm() / ref.norm()).item()


NRMSE_GATE = 0.10


class TestFullModel:
    def test_hidden_and_logits(self, device):
        cfg = _make_cfg()
        w = _make_weights(cfg)
        ref = Esm2Model(cfg, w)
        ref.eval()

        ids = torch.randint(3, 30, (1, 32))
        ids[:, 0], ids[:, -1] = 0, 2
        mask = torch.ones(1, 32, dtype=torch.long)
        with torch.no_grad():
            logits_ref, hidden_ref = ref(ids, mask)

        backend = TtnnEsm2(cfg, w, device=device, precision="fp32")
        backend.build()
        out = backend.forward(ids, mask)

        hidden_got = torch.from_numpy(out["hidden"])
        logits_got = torch.from_numpy(out["logits"])
        nh = _nrmse(hidden_ref, hidden_got)
        nl = _nrmse(logits_ref, logits_got)
        print(f"\n[bringup] hidden NRMSE={nh:.6f} logits NRMSE={nl:.6f}")
        assert nh < NRMSE_GATE, f"hidden NRMSE {nh:.6f} > {NRMSE_GATE}"
        assert nl < NRMSE_GATE, f"logits NRMSE {nl:.6f} > {NRMSE_GATE}"

    def test_batch_consistency(self, device):
        """Same sequence alone vs in a batch of 2 must give identical hidden rows."""
        cfg = _make_cfg()
        w = _make_weights(cfg)
        ids1 = torch.randint(3, 30, (1, 32))
        ids1[:, 0], ids1[:, -1] = 0, 2
        mask1 = torch.ones(1, 32, dtype=torch.long)
        ids2 = torch.randint(3, 30, (1, 32))
        ids2[:, 0], ids2[:, -1] = 0, 2
        mask2 = torch.ones(1, 32, dtype=torch.long)

        backend = TtnnEsm2(cfg, w, device=device, precision="fp32")
        backend.build()

        out_single = backend.forward(ids1, mask1)
        batch_ids = torch.cat([ids1, ids2])
        batch_mask = torch.cat([mask1, mask2])
        out_batch = backend.forward(batch_ids, batch_mask)

        h_single = torch.from_numpy(out_single["hidden"])
        h_batch_row0 = torch.from_numpy(out_batch["hidden"])[0:1]
        diff = (h_single - h_batch_row0).abs().max().item()
        print(f"\n[batch] row-0 max diff = {diff:.8f}")
        assert diff < 1e-4, f"batch row-0 max diff {diff} > 1e-4"
