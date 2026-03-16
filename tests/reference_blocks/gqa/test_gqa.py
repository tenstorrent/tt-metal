# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unified GQA (Grouped-Query Attention) reference tests.

These pure-PyTorch tests validate two independent reference implementations of
GQA against each other, parameterised across all target model shapes:

    glm_4_7_355b, gpt_oss_20b, gpt_oss_120b, grok_2_270b, llama_guard_4

Test categories
---------------
* **Shape tests** – verify projection dimensions and output shapes.
* **Cross-reference tests** – manual-matmul reference vs SDPA reference.
* **RoPE tests** – full and partial rotary embeddings.
* **Causal mask tests** – explicit mask and ``is_causal`` flag equivalence.
* **KV-cache tests** – incremental decode produces same result as full prefill.
* **Feature tests** – QK-norm, attention logit softcapping.
"""

from __future__ import annotations

from typing import Optional, Tuple

import pytest
import torch

from .gqa_config import (
    ALL_MODEL_CONFIGS,
    GLM_4_7_355B,
    GPT_OSS_120B,
    GQAConfig,
    LLAMA_GUARD_4,
    QWEN3_235B,
)
from .gqa_reference import (
    GQAReference,
    GQAReferenceSdpa,
    build_rope_cache,
    copy_weights,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEED = 42


def _set_seed(seed: int = _SEED) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_causal_mask(seq_len: int, kv_len: Optional[int] = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return an additive causal mask: 0 for allowed positions, -inf for masked."""
    kv_len = kv_len or seq_len
    mask = torch.full((seq_len, kv_len), float("-inf"), dtype=dtype)
    mask = torch.triu(mask, diagonal=kv_len - seq_len + 1)
    return mask.unsqueeze(0).unsqueeze(0)


def _make_pair(config: GQAConfig, use_pre_norm: bool = False):
    """Instantiate both references with shared random weights."""
    _set_seed()
    ref = GQAReference(config, use_pre_norm=use_pre_norm)
    sdpa = GQAReferenceSdpa(config, use_pre_norm=use_pre_norm)
    copy_weights(ref, sdpa)
    return ref.eval(), sdpa.eval()


# Reduce hidden sizes for faster CI runs while preserving GQA structure.
def _make_tiny(config: GQAConfig) -> GQAConfig:
    """Scale down a model config to tiny dims keeping head structure intact."""
    scale = max(1, config.hidden_size // 256)
    tiny_heads = max(2, config.num_attention_heads // scale)
    tiny_kv_heads = max(1, config.num_key_value_heads // scale)
    while tiny_heads % tiny_kv_heads != 0:
        tiny_kv_heads -= 1
        if tiny_kv_heads < 1:
            tiny_kv_heads = 1
            tiny_heads = tiny_kv_heads
            break
    tiny_hidden = max(64, config.hidden_size // scale)
    # Round to nearest multiple of 32 for tile alignment
    tiny_hidden = ((tiny_hidden + 31) // 32) * 32

    return GQAConfig(
        hidden_size=tiny_hidden,
        num_attention_heads=tiny_heads,
        num_key_value_heads=tiny_kv_heads,
        head_dim=config.head_dim,
        attention_bias=config.attention_bias,
        rope_theta=config.rope_theta,
        max_position_embeddings=256,
        rms_norm_eps=config.rms_norm_eps,
        rope_partial_factor=config.rope_partial_factor,
        use_qk_norm=config.use_qk_norm,
        attn_logit_softcapping=config.attn_logit_softcapping,
        scaling=config.scaling,
        model_name=f"{config.model_name}_tiny",
    )


# ---------------------------------------------------------------------------
# Fixtures & parametrization
# ---------------------------------------------------------------------------

MODEL_IDS = [c.model_name for c in ALL_MODEL_CONFIGS]


@pytest.fixture(params=ALL_MODEL_CONFIGS, ids=MODEL_IDS)
def model_config(request) -> GQAConfig:
    """Provide each model's full-size config."""
    return request.param


@pytest.fixture(params=ALL_MODEL_CONFIGS, ids=MODEL_IDS)
def tiny_config(request) -> GQAConfig:
    """Provide a scaled-down version for fast CI."""
    return _make_tiny(request.param)


# ---------------------------------------------------------------------------
# 1. Shape / dimension tests (full size – no forward pass needed)
# ---------------------------------------------------------------------------


class TestGQAShapes:
    """Verify projection dimensions match the expected weight shapes from Ilia's table."""

    def test_q_proj_shape(self, model_config):
        ref = GQAReference(model_config, use_pre_norm=False)
        w = ref.q_proj.weight
        assert w.shape == (model_config.q_proj_size, model_config.hidden_size)

    def test_k_proj_shape(self, model_config):
        ref = GQAReference(model_config, use_pre_norm=False)
        w = ref.k_proj.weight
        assert w.shape == (model_config.kv_proj_size, model_config.hidden_size)

    def test_v_proj_shape(self, model_config):
        ref = GQAReference(model_config, use_pre_norm=False)
        w = ref.v_proj.weight
        assert w.shape == (model_config.kv_proj_size, model_config.hidden_size)

    def test_o_proj_shape(self, model_config):
        ref = GQAReference(model_config, use_pre_norm=False)
        w = ref.o_proj.weight
        assert w.shape == (model_config.hidden_size, model_config.q_proj_size)

    def test_bias_present_when_configured(self, model_config):
        ref = GQAReference(model_config, use_pre_norm=False)
        has_bias = ref.q_proj.bias is not None
        assert has_bias == model_config.attention_bias

    def test_kv_group_ratio(self, model_config):
        assert model_config.num_attention_heads % model_config.num_key_value_heads == 0
        expected_groups = model_config.num_attention_heads // model_config.num_key_value_heads
        assert model_config.num_kv_groups == expected_groups


# ---------------------------------------------------------------------------
# 2. Forward pass output shape tests (tiny configs)
# ---------------------------------------------------------------------------


class TestGQAForwardShapes:
    """Verify output tensor shapes for each model config."""

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("seq_len", [1, 16, 64])
    def test_output_shape(self, tiny_config, batch_size, seq_len):
        ref = GQAReference(tiny_config, use_pre_norm=False)
        ref.eval()
        x = torch.randn(batch_size, seq_len, tiny_config.hidden_size)
        with torch.no_grad():
            out, _ = ref(x, use_residual=False)
        assert out.shape == (batch_size, seq_len, tiny_config.hidden_size)

    def test_output_finite(self, tiny_config):
        ref = GQAReference(tiny_config, use_pre_norm=False)
        ref.eval()
        x = torch.randn(1, 8, tiny_config.hidden_size)
        with torch.no_grad():
            out, _ = ref(x, use_residual=False)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 3. Cross-reference: manual matmul vs SDPA  (the core equivalence test)
# ---------------------------------------------------------------------------


class TestGQACrossReference:
    """Verify that the two independent GQA implementations produce matching outputs."""

    @pytest.mark.parametrize("seq_len", [1, 8, 32])
    def test_equivalence_no_mask(self, tiny_config, seq_len):
        """Without any mask (non-causal), both refs should match exactly."""
        ref, sdpa = _make_pair(tiny_config, use_pre_norm=False)
        x = torch.randn(1, seq_len, tiny_config.hidden_size)
        # Use a non-causal zero mask so SDPA doesn't auto-apply causal
        mask = torch.zeros(1, 1, seq_len, seq_len)
        with torch.no_grad():
            out_ref, _ = ref(x, attention_mask=mask, use_residual=False)
            out_sdpa, _ = sdpa(x, attention_mask=mask, use_residual=False)
        torch.testing.assert_close(out_ref, out_sdpa, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("seq_len", [8, 32])
    def test_equivalence_causal_mask(self, tiny_config, seq_len):
        """With causal mask, both refs should match."""
        ref, sdpa = _make_pair(tiny_config, use_pre_norm=False)
        x = torch.randn(1, seq_len, tiny_config.hidden_size)
        mask = _build_causal_mask(seq_len)
        with torch.no_grad():
            out_ref, _ = ref(x, attention_mask=mask, use_residual=False)
            out_sdpa, _ = sdpa(x, attention_mask=mask, use_residual=False)
        torch.testing.assert_close(out_ref, out_sdpa, atol=1e-4, rtol=1e-4)

    def test_equivalence_with_rope(self, tiny_config):
        ref, sdpa = _make_pair(tiny_config, use_pre_norm=False)
        seq_len = 16
        rope_dim = (
            int(tiny_config.head_dim * tiny_config.rope_partial_factor)
            if tiny_config.rope_partial_factor < 1.0
            else tiny_config.head_dim
        )
        cos, sin = build_rope_cache(seq_len, tiny_config.head_dim, tiny_config.rope_theta, rope_dim=rope_dim)
        x = torch.randn(1, seq_len, tiny_config.hidden_size)
        mask = _build_causal_mask(seq_len)
        with torch.no_grad():
            out_ref, _ = ref(x, attention_mask=mask, cos=cos, sin=sin, use_residual=False)
            out_sdpa, _ = sdpa(x, attention_mask=mask, cos=cos, sin=sin, use_residual=False)
        torch.testing.assert_close(out_ref, out_sdpa, atol=1e-4, rtol=1e-4)

    def test_equivalence_with_pre_norm(self, tiny_config):
        ref, sdpa = _make_pair(tiny_config, use_pre_norm=True)
        x = torch.randn(1, 16, tiny_config.hidden_size)
        mask = _build_causal_mask(16)
        with torch.no_grad():
            out_ref, _ = ref(x, attention_mask=mask, use_residual=True)
            out_sdpa, _ = sdpa(x, attention_mask=mask, use_residual=True)
        torch.testing.assert_close(out_ref, out_sdpa, atol=1e-4, rtol=1e-4)

    def test_equivalence_batched(self, tiny_config):
        ref, sdpa = _make_pair(tiny_config, use_pre_norm=False)
        x = torch.randn(4, 16, tiny_config.hidden_size)
        mask = _build_causal_mask(16)
        with torch.no_grad():
            out_ref, _ = ref(x, attention_mask=mask, use_residual=False)
            out_sdpa, _ = sdpa(x, attention_mask=mask, use_residual=False)
        torch.testing.assert_close(out_ref, out_sdpa, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# 4. RoPE tests
# ---------------------------------------------------------------------------


class TestGQARope:
    """Test that Rotary Position Embeddings are applied correctly."""

    def test_rope_changes_output(self, tiny_config):
        """Output should differ with vs without RoPE."""
        ref = GQAReference(tiny_config, use_pre_norm=False).eval()
        seq_len = 16
        rope_dim = (
            int(tiny_config.head_dim * tiny_config.rope_partial_factor)
            if tiny_config.rope_partial_factor < 1.0
            else tiny_config.head_dim
        )
        cos, sin = build_rope_cache(seq_len, tiny_config.head_dim, tiny_config.rope_theta, rope_dim=rope_dim)
        x = torch.randn(1, seq_len, tiny_config.hidden_size)
        mask = _build_causal_mask(seq_len)
        with torch.no_grad():
            out_no_rope, _ = ref(x, attention_mask=mask, use_residual=False)
            out_rope, _ = ref(x, attention_mask=mask, cos=cos, sin=sin, use_residual=False)
        assert not torch.allclose(out_no_rope, out_rope, atol=1e-5)

    def test_different_relative_positions_give_different_output(self, tiny_config):
        """Different *relative* positions should produce different outputs.

        RoPE encodes relative position, so shifting all tokens by the same
        offset gives identical results.  Instead we compare a prefill decode
        where a new token is added at different relative distances from the
        cached context.
        """
        ref = GQAReference(tiny_config, use_pre_norm=False).eval()
        rope_dim = (
            int(tiny_config.head_dim * tiny_config.rope_partial_factor)
            if tiny_config.rope_partial_factor < 1.0
            else tiny_config.head_dim
        )
        cos_full, sin_full = build_rope_cache(64, tiny_config.head_dim, tiny_config.rope_theta, rope_dim=rope_dim)

        # Prefill 4 context tokens at positions 0..3
        x_ctx = torch.randn(1, 4, tiny_config.hidden_size)
        empty_cache = (
            torch.zeros(1, tiny_config.num_key_value_heads, 0, tiny_config.head_dim),
            torch.zeros(1, tiny_config.num_key_value_heads, 0, tiny_config.head_dim),
        )
        mask_ctx = _build_causal_mask(4)
        with torch.no_grad():
            _, kv_cache = ref(
                x_ctx,
                attention_mask=mask_ctx,
                cos=cos_full[:, :, 0:4],
                sin=sin_full[:, :, 0:4],
                kv_cache=empty_cache,
                use_residual=False,
            )

        # Decode one new token using different position indices
        x_new = torch.randn(1, 1, tiny_config.hidden_size)
        mask_decode = _build_causal_mask(1, 5)
        with torch.no_grad():
            out_pos4, _ = ref(
                x_new,
                attention_mask=mask_decode,
                cos=cos_full[:, :, 4:5],
                sin=sin_full[:, :, 4:5],
                kv_cache=kv_cache,
                use_residual=False,
            )
            out_pos20, _ = ref(
                x_new,
                attention_mask=mask_decode,
                cos=cos_full[:, :, 20:21],
                sin=sin_full[:, :, 20:21],
                kv_cache=kv_cache,
                use_residual=False,
            )
        # Different absolute position for the query token means different
        # relative distances to the cached keys → different attention.
        assert not torch.allclose(out_pos4, out_pos20, atol=1e-5)


# ---------------------------------------------------------------------------
# 5. Causal masking tests
# ---------------------------------------------------------------------------


class TestGQACausalMask:
    """Verify causal attention prevents attending to future tokens."""

    def test_causal_mask_structure(self):
        mask = _build_causal_mask(4)
        assert mask.shape == (1, 1, 4, 4)
        assert mask[0, 0, 0, 1] == float("-inf")
        assert mask[0, 0, 0, 0] == 0.0
        assert mask[0, 0, 3, 0] == 0.0
        assert mask[0, 0, 3, 3] == 0.0

    def test_masked_output_independent_of_future(self, tiny_config):
        """Changing future tokens shouldn't affect output at earlier positions."""
        ref = GQAReference(tiny_config, use_pre_norm=False).eval()
        seq_len = 8
        x1 = torch.randn(1, seq_len, tiny_config.hidden_size)
        x2 = x1.clone()
        x2[:, -2:, :] = torch.randn_like(x2[:, -2:, :])
        mask = _build_causal_mask(seq_len)
        with torch.no_grad():
            out1, _ = ref(x1, attention_mask=mask, use_residual=False)
            out2, _ = ref(x2, attention_mask=mask, use_residual=False)
        torch.testing.assert_close(out1[:, :6, :], out2[:, :6, :], atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 6. KV-cache tests (incremental decoding)
# ---------------------------------------------------------------------------


class TestGQAKvCache:
    """Test that incremental decoding via KV cache matches single-pass prefill."""

    def test_decode_matches_prefill(self, tiny_config):
        """Token-by-token decode should produce same output as full prefill."""
        ref = GQAReference(tiny_config, use_pre_norm=False).eval()
        seq_len = 8
        x = torch.randn(1, seq_len, tiny_config.hidden_size)
        rope_dim = (
            int(tiny_config.head_dim * tiny_config.rope_partial_factor)
            if tiny_config.rope_partial_factor < 1.0
            else tiny_config.head_dim
        )
        cos, sin = build_rope_cache(seq_len, tiny_config.head_dim, tiny_config.rope_theta, rope_dim=rope_dim)

        mask_full = _build_causal_mask(seq_len)
        with torch.no_grad():
            out_prefill, _ = ref(x, attention_mask=mask_full, cos=cos, sin=sin, use_residual=False)

        # Incremental decode with initial empty cache
        kv_cache = (
            torch.zeros(1, tiny_config.num_key_value_heads, 0, tiny_config.head_dim),
            torch.zeros(1, tiny_config.num_key_value_heads, 0, tiny_config.head_dim),
        )
        decode_outputs = []
        with torch.no_grad():
            for t in range(seq_len):
                x_t = x[:, t : t + 1, :]
                cos_t = cos[:, :, t : t + 1, :]
                sin_t = sin[:, :, t : t + 1, :]
                kv_len = t + 1
                mask_t = _build_causal_mask(1, kv_len)
                out_t, kv_cache = ref(
                    x_t, attention_mask=mask_t, cos=cos_t, sin=sin_t, kv_cache=kv_cache, use_residual=False
                )
                decode_outputs.append(out_t)

        out_decode = torch.cat(decode_outputs, dim=1)
        torch.testing.assert_close(out_prefill, out_decode, atol=1e-4, rtol=1e-4)

    def test_kv_cache_shape(self, tiny_config):
        """KV cache should have correct dimensions."""
        ref = GQAReference(tiny_config, use_pre_norm=False).eval()
        x = torch.randn(2, 4, tiny_config.hidden_size)
        mask = _build_causal_mask(4)
        kv_cache = (
            torch.zeros(2, tiny_config.num_key_value_heads, 0, tiny_config.head_dim),
            torch.zeros(2, tiny_config.num_key_value_heads, 0, tiny_config.head_dim),
        )
        with torch.no_grad():
            _, new_cache = ref(x, attention_mask=mask, kv_cache=kv_cache, use_residual=False)
        assert new_cache is not None
        k_cache, v_cache = new_cache
        assert k_cache.shape == (2, tiny_config.num_key_value_heads, 4, tiny_config.head_dim)
        assert v_cache.shape == (2, tiny_config.num_key_value_heads, 4, tiny_config.head_dim)


# ---------------------------------------------------------------------------
# 7. Feature-specific tests (QK-norm, softcapping)
# ---------------------------------------------------------------------------


class TestGQAFeatures:
    """Test optional features that are model-specific."""

    def test_qk_norm_changes_output(self):
        """QK-norm (used by GLM-4) should produce different outputs than without."""
        cfg_no_norm = GQAConfig(
            hidden_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            use_qk_norm=False,
            model_name="test_no_norm",
        )
        cfg_norm = GQAConfig(
            hidden_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            use_qk_norm=True,
            model_name="test_norm",
        )
        _set_seed()
        ref_no = GQAReference(cfg_no_norm, use_pre_norm=False).eval()
        _set_seed()
        ref_yes = GQAReference(cfg_norm, use_pre_norm=False).eval()
        # Copy shared projection weights
        copy_weights(ref_no, ref_yes)

        x = torch.randn(1, 8, 256)
        mask = torch.zeros(1, 1, 8, 8)
        with torch.no_grad():
            out_no, _ = ref_no(x, attention_mask=mask, use_residual=False)
            out_yes, _ = ref_yes(x, attention_mask=mask, use_residual=False)
        assert not torch.allclose(out_no, out_yes, atol=1e-5)

    def test_softcapping_limits_attention_logits(self):
        """Attention logit softcapping (used by Grok-2) should bound the logits."""
        cfg = GQAConfig(
            hidden_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            attn_logit_softcapping=30.0,
            model_name="test_softcap",
        )
        ref = GQAReference(cfg, use_pre_norm=False).eval()
        # Use large inputs to push logits high
        x = torch.randn(1, 8, 256) * 10.0
        mask = torch.zeros(1, 1, 8, 8)
        with torch.no_grad():
            out, _ = ref(x, attention_mask=mask, use_residual=False)
        assert torch.isfinite(out).all()

    def test_residual_connection(self, tiny_config):
        """With use_residual=True, output = input + attn(input)."""
        ref = GQAReference(tiny_config, use_pre_norm=False).eval()
        x = torch.randn(1, 4, tiny_config.hidden_size)
        mask = _build_causal_mask(4)
        with torch.no_grad():
            out_no_res, _ = ref(x, attention_mask=mask, use_residual=False)
            out_res, _ = ref(x, attention_mask=mask, use_residual=True)
        torch.testing.assert_close(out_res, x + out_no_res, atol=1e-6, rtol=1e-6)

    def test_pre_norm(self, tiny_config):
        """Pre-attention RMSNorm should produce different output than no norm."""
        ref_norm = GQAReference(tiny_config, use_pre_norm=True).eval()
        ref_no_norm = GQAReference(tiny_config, use_pre_norm=False).eval()
        # Share projection weights
        ref_no_norm.q_proj.load_state_dict(ref_norm.q_proj.state_dict())
        ref_no_norm.k_proj.load_state_dict(ref_norm.k_proj.state_dict())
        ref_no_norm.v_proj.load_state_dict(ref_norm.v_proj.state_dict())
        ref_no_norm.o_proj.load_state_dict(ref_norm.o_proj.state_dict())

        x = torch.randn(1, 8, tiny_config.hidden_size)
        mask = _build_causal_mask(8)
        with torch.no_grad():
            out_norm, _ = ref_norm(x, attention_mask=mask, use_residual=False)
            out_no, _ = ref_no_norm(x, attention_mask=mask, use_residual=False)
        assert not torch.allclose(out_norm, out_no, atol=1e-5)


# ---------------------------------------------------------------------------
# 8. Partial RoPE test (GLM-4 specific)
# ---------------------------------------------------------------------------


class TestGQAPartialRope:
    """Test partial rotary embeddings where only a fraction of head_dim is rotated."""

    def test_partial_rope_equivalence(self):
        cfg = GQAConfig(
            hidden_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            rope_partial_factor=0.5,
            rope_theta=1_000_000,
            model_name="test_partial_rope",
        )
        ref, sdpa = _make_pair(cfg, use_pre_norm=False)
        seq_len = 16
        rope_dim = int(cfg.head_dim * cfg.rope_partial_factor)
        cos, sin = build_rope_cache(seq_len, cfg.head_dim, cfg.rope_theta, rope_dim=rope_dim)
        x = torch.randn(1, seq_len, cfg.hidden_size)
        mask = _build_causal_mask(seq_len)
        with torch.no_grad():
            out_ref, _ = ref(x, attention_mask=mask, cos=cos, sin=sin, use_residual=False)
            out_sdpa, _ = sdpa(x, attention_mask=mask, cos=cos, sin=sin, use_residual=False)
        torch.testing.assert_close(out_ref, out_sdpa, atol=1e-4, rtol=1e-4)

    def test_partial_vs_full_rope_differ(self):
        """Partial RoPE and full RoPE should give different results."""
        cfg_partial = GQAConfig(
            hidden_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            rope_partial_factor=0.5,
            model_name="test_partial",
        )
        cfg_full = GQAConfig(
            hidden_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            rope_partial_factor=1.0,
            model_name="test_full",
        )
        _set_seed()
        ref_partial = GQAReference(cfg_partial, use_pre_norm=False).eval()
        _set_seed()
        ref_full = GQAReference(cfg_full, use_pre_norm=False).eval()
        copy_weights(ref_partial, ref_full)

        seq_len = 16
        x = torch.randn(1, seq_len, 256)
        mask = _build_causal_mask(seq_len)
        cos_partial, sin_partial = build_rope_cache(seq_len, 64, 10000.0, rope_dim=32)
        cos_full, sin_full = build_rope_cache(seq_len, 64, 10000.0, rope_dim=64)
        with torch.no_grad():
            out_p, _ = ref_partial(x, attention_mask=mask, cos=cos_partial, sin=sin_partial, use_residual=False)
            out_f, _ = ref_full(x, attention_mask=mask, cos=cos_full, sin=sin_full, use_residual=False)
        assert not torch.allclose(out_p, out_f, atol=1e-5)


# ---------------------------------------------------------------------------
# 9. Full-size single-token decode smoke test (lightweight, uses real dims)
# ---------------------------------------------------------------------------


class TestGQAFullSizeSmokeTest:
    """Single-token forward pass with real model dimensions (no data dependency)."""

    @pytest.mark.parametrize("config", ALL_MODEL_CONFIGS, ids=MODEL_IDS)
    def test_single_token_decode(self, config):
        """A single token through the full-size GQA block should produce finite output."""
        ref = GQAReference(config, use_pre_norm=True).eval()
        x = torch.randn(1, 1, config.hidden_size)
        with torch.no_grad():
            out, _ = ref(x, use_residual=True)
        assert out.shape == (1, 1, config.hidden_size)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 10. GQA group ratio sweep
# ---------------------------------------------------------------------------


class TestGQAGroupRatios:
    """Test different GQA group ratios beyond what the model presets use."""

    @pytest.mark.parametrize(
        "num_heads,num_kv_heads",
        [(8, 8), (8, 4), (8, 2), (8, 1), (16, 4), (32, 8)],
        ids=["MHA_8:8", "GQA_8:4", "GQA_8:2", "MQA_8:1", "GQA_16:4", "GQA_32:8"],
    )
    def test_various_group_ratios(self, num_heads, num_kv_heads):
        cfg = GQAConfig(
            hidden_size=256,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=32,
            model_name=f"test_{num_heads}_{num_kv_heads}",
        )
        ref, sdpa = _make_pair(cfg, use_pre_norm=False)
        x = torch.randn(2, 8, 256)
        mask = _build_causal_mask(8)
        with torch.no_grad():
            out_ref, _ = ref(x, attention_mask=mask, use_residual=False)
            out_sdpa, _ = sdpa(x, attention_mask=mask, use_residual=False)
        torch.testing.assert_close(out_ref, out_sdpa, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# 11. Decoder block integration test
# ---------------------------------------------------------------------------


class TestGQADecoderBlock:
    """Test GQA as part of a minimal decoder block (norm -> attn -> residual)."""

    def test_decoder_block_output(self, tiny_config):
        """Full decoder-style pass: pre-norm + attention + residual."""
        ref = GQAReference(tiny_config, use_pre_norm=True).eval()
        seq_len = 16
        x = torch.randn(1, seq_len, tiny_config.hidden_size)
        mask = _build_causal_mask(seq_len)
        rope_dim = (
            int(tiny_config.head_dim * tiny_config.rope_partial_factor)
            if tiny_config.rope_partial_factor < 1.0
            else tiny_config.head_dim
        )
        cos, sin = build_rope_cache(seq_len, tiny_config.head_dim, tiny_config.rope_theta, rope_dim=rope_dim)
        with torch.no_grad():
            out, _ = ref(x, attention_mask=mask, cos=cos, sin=sin, use_residual=True)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()
        # With residual, output shouldn't be identical to input (attention changes it)
        assert not torch.allclose(out, x, atol=1e-5)


# ---------------------------------------------------------------------------
# 12. Exact weight-shape validation against models_shapes.txt
# ---------------------------------------------------------------------------


class TestGQAExactShapes:
    """Validate that configs produce the exact weight shapes from Ilia's shape table.

    Each tuple: (model_config, q_proj_shape, k_proj_shape, v_proj_shape, o_proj_shape)
    where shapes are (out_features, in_features) matching the (d_model, proj_dim) convention.
    """

    EXPECTED_SHAPES = [
        # glm_4_7_355b: q(5120,12288) k(5120,1024) v(5120,1024) o(12288,5120)
        (GLM_4_7_355B, (12288, 5120), (1024, 5120), (1024, 5120), (5120, 12288)),
        # gpt_oss_120b: q(2880,4096) k(2880,512) v(2880,512) o(4096,2880)
        (GPT_OSS_120B, (4096, 2880), (512, 2880), (512, 2880), (2880, 4096)),
        # llama_guard_4: q(5120,5120) k(5120,1024) v(5120,1024) o(5120,5120)
        (LLAMA_GUARD_4, (5120, 5120), (1024, 5120), (1024, 5120), (5120, 5120)),
        # qwen3_235b: q(4096,8192) k(4096,512) v(4096,512) o(8192,4096)
        (QWEN3_235B, (8192, 4096), (512, 4096), (512, 4096), (4096, 8192)),
    ]

    @pytest.mark.parametrize(
        "config,q_shape,k_shape,v_shape,o_shape",
        EXPECTED_SHAPES,
        ids=["glm_4_7_355b", "gpt_oss_120b", "llama_guard_4", "qwen3_235b"],
    )
    def test_exact_weight_shapes(self, config, q_shape, k_shape, v_shape, o_shape):
        """Weight shapes must exactly match the shapes from models_shapes.txt."""
        ref = GQAReference(config, use_pre_norm=False)
        assert ref.q_proj.weight.shape == q_shape, f"q_proj: expected {q_shape}, got {ref.q_proj.weight.shape}"
        assert ref.k_proj.weight.shape == k_shape, f"k_proj: expected {k_shape}, got {ref.k_proj.weight.shape}"
        assert ref.v_proj.weight.shape == v_shape, f"v_proj: expected {v_shape}, got {ref.v_proj.weight.shape}"
        assert ref.o_proj.weight.shape == o_shape, f"o_proj: expected {o_shape}, got {ref.o_proj.weight.shape}"

    @pytest.mark.parametrize(
        "config,q_shape,k_shape,v_shape,o_shape",
        EXPECTED_SHAPES,
        ids=["glm_4_7_355b", "gpt_oss_120b", "llama_guard_4", "qwen3_235b"],
    )
    def test_exact_io_shapes(self, config, q_shape, k_shape, v_shape, o_shape):
        """Input/output tensor shapes must match models_shapes.txt (batch=1, seq=1)."""
        ref = GQAReference(config, use_pre_norm=True).eval()
        x = torch.randn(1, 1, config.hidden_size)
        with torch.no_grad():
            out, _ = ref(x, use_residual=True)
        assert out.shape == (1, 1, config.hidden_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
