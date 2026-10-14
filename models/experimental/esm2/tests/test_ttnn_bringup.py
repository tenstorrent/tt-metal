# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: MIT
"""On-device bring-up tests with random weights (no checkpoint required)."""

from __future__ import annotations

import numpy as np
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


class TestCachesAndTrace:
    """Codex-review regressions: shape-keyed device caches, bounded retention,
    and trace first-call correctness (a newly captured trace must execute
    before its outputs are read)."""

    def _backend(self, device):
        cfg = _make_cfg()
        w = _make_weights(cfg)
        b = TtnnEsm2(cfg, w, device=device, precision="fp32")
        b.build()
        return b, cfg

    def _ids(self, batch, length, seed):
        g = torch.Generator().manual_seed(seed)
        ids = torch.randint(3, 30, (batch, length), generator=g)
        ids[:, 0], ids[:, -1] = 0, 2
        return ids

    def test_mask_cache_keys_include_shape(self, device):
        """[1,64] and [2,32] all-ones masks flatten to identical bytes but
        must resolve to distinct, correctly-shaped device tensors."""
        b, cfg = self._backend(device)
        H = cfg.num_attention_heads
        m64 = b.mask_tensor(torch.ones(1, 64, dtype=torch.long))
        m32 = b.mask_tensor(torch.ones(2, 32, dtype=torch.long))
        assert m64 is not m32
        assert [int(d) for d in m64.shape] == [1, H, 64, 64]
        assert [int(d) for d in m32.shape] == [2, H, 32, 32]

    def test_rotary_cache_keys_include_shape(self, device):
        """Equal-element-count ids shapes must not collide in the rotary cache."""
        b, cfg = self._backend(device)
        H, D = cfg.num_attention_heads, cfg.head_dim
        t1 = b.rotary_tensors(torch.zeros(1, 64, dtype=torch.long))
        t2 = b.rotary_tensors(torch.zeros(2, 32, dtype=torch.long))
        assert t1 is not t2
        assert [int(d) for d in t1[0][0].shape] == [1, H, 64, D]  # q_cos
        assert [int(d) for d in t2[0][0].shape] == [2, H, 32, D]

    def test_rotary_cache_reuses_same_position_pattern(self, device):
        """Tables depend only on positions: different ids with the same pad
        structure (here: no pads) reuse one device copy."""
        b, cfg = self._backend(device)
        ids_a = self._ids(1, 32, seed=101)
        ids_b = self._ids(1, 32, seed=202)
        assert not torch.equal(ids_a, ids_b)
        assert b.rotary_tensors(ids_a) is b.rotary_tensors(ids_b)

    def test_per_input_caches_are_bounded(self, device):
        """Rotary/mask caches must not grow without bound across shapes."""
        b, cfg = self._backend(device)
        b._MASK_CACHE_MAX = 2
        b._ROTARY_CACHE_MAX = 2
        for length in (32, 64, 96):
            b.mask_tensor(torch.ones(1, length, dtype=torch.long))
            b.rotary_tensors(self._ids(1, length, seed=length))
        assert len(b._mask_cache) <= 2
        assert len(b._rotary_cache) <= 2

    def test_trace_first_call_matches_eager_and_replay(self, device):
        """The FIRST traced call (capture) must already return executed
        outputs, bit-identical to eager; replay must reproduce them."""
        b, cfg = self._backend(device)
        ids = self._ids(1, 32, seed=11)
        mask = torch.ones(1, 32, dtype=torch.long)
        eager = b.forward(ids, mask)
        b.set_trace(True)
        first = b.forward(ids, mask)  # capture + first execute
        assert b.trace_status == "captured", b.trace_status
        assert b.last_trace_mode == "trace_capture"
        replay = b.forward(ids, mask)
        assert b.trace_status == "replayed", b.trace_status
        assert b.last_trace_mode == "trace_replay"
        for out in (first, replay):
            assert np.array_equal(out["hidden"], eager["hidden"])
            assert np.array_equal(out["logits"], eager["logits"])

    def test_trace_survives_cache_eviction(self, device):
        """Evicting the captured shape's rotary/mask cache entries must not
        invalidate the trace: entries hold their own tensor references."""
        b, cfg = self._backend(device)
        ids = self._ids(1, 32, seed=13)
        mask = torch.ones(1, 32, dtype=torch.long)
        b.set_trace(True)
        first = b.forward(ids, mask)  # capture + execute
        b._MASK_CACHE_MAX = 1
        b._ROTARY_CACHE_MAX = 1
        for length in (64, 96):  # new shapes evict the captured shape's entries
            b.forward(self._ids(1, length, seed=length), torch.ones(1, length, dtype=torch.long))
        assert len(b._mask_cache) <= 1 and len(b._rotary_cache) <= 1
        replay = b.forward(ids, mask)  # trace hit with its own pinned tensors
        assert b.trace_status == "replayed", b.trace_status
        assert np.array_equal(replay["hidden"], first["hidden"])
        assert np.array_equal(replay["logits"], first["logits"])
