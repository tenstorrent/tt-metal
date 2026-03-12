# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PCC tests for Qwen3.5 GatedDeltaNet (reference vs reference, no device needed)."""

import json
from pathlib import Path

import pytest
import torch

from models.demos.qwen3_5.reference.gated_delta_net import GatedDeltaNet
from models.demos.qwen3_5.reference.model import FullAttention, Qwen3_5Config, Qwen3_5TextTransformer

CONFIG_PATH = Path("models/tt_transformers/model_params/Qwen3.5-27B/config.json")


@pytest.fixture(scope="module")
def config():
    with open(CONFIG_PATH) as f:
        return Qwen3_5Config(json.load(f))


# -------------------------------------------------------------------------
# GatedDeltaNet prefill vs decode consistency
# -------------------------------------------------------------------------


class TestGatedDeltaNetRef:
    """Test the reference GatedDeltaNet produces consistent results."""

    def test_decode_single_step(self, config):
        """Decode step should produce deterministic output from zero state."""
        model = GatedDeltaNet(config)
        model.eval()
        x = torch.randn(1, 1, config.hidden_size)
        with torch.no_grad():
            out, cv, rv = model(x)
        assert out.shape == (1, 1, config.hidden_size)
        assert cv is not None
        assert rv is not None

    def test_prefill_decode_consistency(self, config):
        """Running T steps in prefill then 0 more should equal T sequential decode steps."""
        torch.manual_seed(42)
        model = GatedDeltaNet(config)
        model.eval()
        T = 4
        x = torch.randn(1, T, config.hidden_size)

        # Prefill: process all T tokens at once
        with torch.no_grad():
            out_prefill, cv_pre, rv_pre = model(x)

        # Sequential decode: process one token at a time from scratch
        model2 = GatedDeltaNet(config)
        model2.load_state_dict(model.state_dict())
        model2.eval()
        cv, rv = None, None
        outs = []
        with torch.no_grad():
            for t in range(T):
                o, cv, rv = model2(x[:, t : t + 1, :], conv_state=cv, recurrent_state=rv)
                outs.append(o)
        out_decode = torch.cat(outs, dim=1)

        # PCC should be > 0.99
        from models.common.utility_functions import comp_pcc

        pcc, _ = comp_pcc(out_prefill.float(), out_decode.float())
        assert pcc > 0.99, f"Prefill vs decode PCC={pcc:.4f} < 0.99"

    def test_pcc_output_range(self, config):
        """Output should not be NaN or explode."""
        torch.manual_seed(0)
        model = GatedDeltaNet(config)
        model.eval()
        x = torch.randn(2, 8, config.hidden_size) * 0.1
        with torch.no_grad():
            out, _, _ = model(x)
        assert not torch.isnan(out).any(), "NaN in output"
        assert out.abs().max() < 1e6, "Output explodes"


# -------------------------------------------------------------------------
# FullAttention reference
# -------------------------------------------------------------------------


class TestFullAttentionRef:
    """Test the reference FullAttention."""

    def test_decode_single_step(self, config):
        """Single decode step should produce correct shape."""
        model = FullAttention(config)
        model.eval()
        x = torch.randn(1, 1, config.hidden_size)
        pos = torch.arange(0, 1).unsqueeze(0)
        rope_theta = config.rope_parameters.get("rope_theta", 10_000_000)
        rope_dim = int(config.head_dim * config.rope_parameters.get("partial_rotary_factor", 1.0))
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
        freqs = torch.einsum("bi,j->bij", pos.float(), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos(), emb.sin()
        with torch.no_grad():
            out, kv = model(x, cos, sin)
        assert out.shape == (1, 1, config.hidden_size)
        assert len(kv) == 2

    def test_pcc_prefill_vs_incremental(self, config):
        """Incremental decode with KV cache should match single prefill pass."""
        torch.manual_seed(7)
        model = FullAttention(config)
        model.eval()
        T = 6
        x = torch.randn(1, T, config.hidden_size) * 0.1
        rope_theta = config.rope_parameters.get("rope_theta", 10_000_000)
        rope_dim = int(config.head_dim * config.rope_parameters.get("partial_rotary_factor", 1.0))
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))

        def get_cos_sin(start, length):
            pos = torch.arange(start, start + length).unsqueeze(0).float()
            freqs = torch.einsum("bi,j->bij", pos, inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            return emb.cos(), emb.sin()

        # Prefill
        cos, sin = get_cos_sin(0, T)
        with torch.no_grad():
            out_prefill, _ = model(x, cos, sin)

        # Incremental (auto-regressive)
        model2 = FullAttention(config)
        model2.load_state_dict(model.state_dict())
        model2.eval()
        outs, kv = [], None
        with torch.no_grad():
            for t in range(T):
                c, s = get_cos_sin(t, 1)
                o, kv = model2(x[:, t : t + 1], c, s, kv_cache=kv)
                outs.append(o)
        out_incremental = torch.cat(outs, dim=1)

        from models.common.utility_functions import comp_pcc

        pcc, _ = comp_pcc(out_prefill.float(), out_incremental.float())
        assert pcc > 0.99, f"Prefill vs incremental PCC={pcc:.4f} < 0.99"


# -------------------------------------------------------------------------
# Small model smoke test (CPU only, no device)
# -------------------------------------------------------------------------


class TestSmallModelRef:
    """Smoke test using a tiny 4-layer Qwen3.5 reference model."""

    @pytest.fixture
    def tiny_config(self, config):
        """Create a 4-layer config for fast testing."""
        import json

        with open(CONFIG_PATH) as f:
            d = json.load(f)
        tc = d.get("text_config", d)
        tc["num_hidden_layers"] = 4
        tc["layer_types"] = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
        return Qwen3_5Config(d)

    def test_forward_prefill(self, tiny_config):
        """Forward pass through all 4 layers should complete without error."""
        torch.manual_seed(0)
        model = Qwen3_5TextTransformer(tiny_config)
        model.eval()
        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 8))
        with torch.no_grad():
            logits = model(input_ids)
        assert logits.shape == (1, 8, tiny_config.vocab_size)
        assert not torch.isnan(logits).any()

    def test_forward_decode(self, tiny_config):
        """Single-token decode step should complete without error."""
        torch.manual_seed(0)
        model = Qwen3_5TextTransformer(tiny_config)
        model.eval()
        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 1))
        with torch.no_grad():
            logits, kvs, cvs, rvs = model(input_ids, return_caches=True)
        assert logits.shape == (1, 1, tiny_config.vocab_size)
        # Second decode step with caches
        input_ids2 = torch.randint(0, tiny_config.vocab_size, (1, 1))
        pos2 = torch.tensor([[1]])
        with torch.no_grad():
            logits2, _, _, _ = model(
                input_ids2, position_ids=pos2, kv_caches=kvs, conv_states=cvs, recurrent_states=rvs, return_caches=True
            )
        assert logits2.shape == (1, 1, tiny_config.vocab_size)
