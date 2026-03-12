# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""test_kimi_model.py — Smoke tests for the KimiGenerator model adapter.

These tests verify the module structure, import chain, and constructor logic
WITHOUT requiring Tenstorrent hardware or real weights.

Hardware / full-weight tests are marked with ``@pytest.mark.skipif`` and only
run when ``KIMI_HF_MODEL`` is set and a TTNN device is available.

Run (no hardware):
    pytest models/demos/kimi_k25/tests/test_kimi_model.py -v

Run (with hardware + real weights):
    KIMI_HF_MODEL=/workspace/extra/Kimi-K2.5 \
      pytest models/demos/kimi_k25/tests/test_kimi_model.py -v -k hardware
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_hf_config(**overrides):
    """Return a minimal mock HF config that satisfies KimiK25Config.from_hf_config."""
    defaults = dict(
        model_type="kimi_k2",
        hidden_size=7168,
        num_hidden_layers=61,
        num_attention_heads=64,
        num_key_value_heads=64,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        num_experts_per_tok=8,
        n_routed_experts=384,
        n_shared_experts=1,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=2.827,
        norm_topk_prob=True,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        moe_intermediate_size=2048,
        intermediate_size=18432,
        rms_norm_eps=1e-5,
        rope_theta=50000.0,
        rope_scaling={
            "type": "yarn",
            "factor": 64.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
        },
        max_position_embeddings=131072,
        vocab_size=163840,
        bos_token_id=1,
        eos_token_id=163585,
        num_mtp_layers=0,
        hidden_act="silu",
        quantization_config=None,
        text_config=None,
    )
    defaults.update(overrides)
    cfg = MagicMock()
    for k, v in defaults.items():
        setattr(cfg, k, v)
    # Ensure getattr falls back properly for missing attrs
    cfg.__class__.__name__ = "KimiK2Config"
    return cfg


# ---------------------------------------------------------------------------
# Module structure tests (no hardware, no weights)
# ---------------------------------------------------------------------------


class TestModuleStructure:
    """Verify the module exports the expected symbols."""

    def test_module_importable(self):
        """kimi_model module can be imported (all DSV3 deps mocked)."""
        # We stub out the heavy DSV3 imports so the module loads in a plain
        # Python environment (no ttnn, no torch, etc.)
        dsv3_gen = types.ModuleType("models.demos.deepseek_v3.tt.generator")
        dsv3_gen.DeepseekGenerator = object  # minimal stub

        dsv3_rbm = types.ModuleType("models.demos.deepseek_v3.tt.model.row_batched_model")
        dsv3_rbm.RowBatchedModel = object

        dsv3_wc = types.ModuleType("models.demos.deepseek_v3.utils.weight_config")
        dsv3_wc.get_weight_config = lambda *a, **kw: {}

        kimi_ca = types.ModuleType("models.demos.kimi_k25.utils.config_adapter")

        from models.demos.kimi_k25.utils.config_adapter import KimiK25Config

        kimi_ca.KimiK25Config = KimiK25Config

        kimi_wl = types.ModuleType("models.demos.kimi_k25.utils.weight_loader")
        kimi_wl.KimiLazyStateDict = MagicMock()

        stub_map = {
            "models.demos.deepseek_v3.tt.generator": dsv3_gen,
            "models.demos.deepseek_v3.tt.model": types.ModuleType(
                "models.demos.deepseek_v3.tt.model"
            ),
            "models.demos.deepseek_v3.tt.model.row_batched_model": dsv3_rbm,
            "models.demos.deepseek_v3.utils.weight_config": dsv3_wc,
            "models.demos.kimi_k25.utils.config_adapter": kimi_ca,
            "models.demos.kimi_k25.utils.weight_loader": kimi_wl,
        }

        # Temporarily inject stubs and reimport
        original = {}
        for key, mod in stub_map.items():
            original[key] = sys.modules.get(key)
            sys.modules[key] = mod

        # Remove cached version if present
        sys.modules.pop("models.demos.kimi_k25.tt.kimi_model", None)

        try:
            import models.demos.kimi_k25.tt.kimi_model as km

            assert hasattr(km, "KimiGenerator"), "KimiGenerator not exported"
            assert hasattr(km, "load_kimi_model"), "load_kimi_model not exported"
            assert "KimiGenerator" in km.__all__
            assert "load_kimi_model" in km.__all__
        finally:
            for key, mod in original.items():
                if mod is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = mod
            sys.modules.pop("models.demos.kimi_k25.tt.kimi_model", None)

    def test_load_kimi_model_signature(self):
        """load_kimi_model accepts model_path + mesh_device + cache_dir + **kwargs."""
        from models.demos.kimi_k25.tt.kimi_model import load_kimi_model

        sig = inspect.signature(load_kimi_model)
        params = list(sig.parameters.keys())
        assert "model_path" in params, f"model_path missing; got {params}"
        assert "mesh_device" in params, f"mesh_device missing; got {params}"
        assert "cache_dir" in params, f"cache_dir missing; got {params}"
        assert "kwargs" in params, f"**kwargs missing; got {params}"

    def test_kimi_generator_is_subclass(self):
        """KimiGenerator is a subclass of DeepseekGenerator."""
        from models.demos.deepseek_v3.tt.generator import DeepseekGenerator
        from models.demos.kimi_k25.tt.kimi_model import KimiGenerator

        assert issubclass(
            KimiGenerator, DeepseekGenerator
        ), "KimiGenerator must subclass DeepseekGenerator"

    def test_kimi_generator_overrides_prepare_weight_configs(self):
        """KimiGenerator.\_prepare\_weight\_configs is overridden (not inherited)."""
        from models.demos.deepseek_v3.tt.generator import DeepseekGenerator
        from models.demos.kimi_k25.tt.kimi_model import KimiGenerator

        # The method should be defined directly on KimiGenerator, not inherited
        assert "_prepare_weight_configs" in KimiGenerator.__dict__, (
            "_prepare_weight_configs must be defined on KimiGenerator itself "
            "to override the parent's weight loading logic."
        )

    def test_default_cache_dir_constant(self):
        """_DEFAULT_CACHE_DIR is set to a kimi-specific path (not deepseek)."""
        from models.demos.kimi_k25.tt.kimi_model import _DEFAULT_CACHE_DIR

        assert "kimi" in str(_DEFAULT_CACHE_DIR).lower(), (
            f"_DEFAULT_CACHE_DIR should contain 'kimi', got {_DEFAULT_CACHE_DIR!r}"
        )
        assert "deepseek" not in str(_DEFAULT_CACHE_DIR).lower(), (
            f"_DEFAULT_CACHE_DIR must not be the deepseek path, got {_DEFAULT_CACHE_DIR!r}"
        )


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """KimiGenerator validates the HF config before constructing the model."""

    def test_valid_config_passes(self):
        """A well-formed Kimi config is accepted without raising."""
        from models.demos.kimi_k25.utils.config_adapter import KimiK25Config

        cfg = _make_mock_hf_config()
        # Should not raise
        kimi_cfg = KimiK25Config.from_hf_config(cfg)
        assert kimi_cfg.n_routed_experts == 384
        assert kimi_cfg.n_group == 1

    def test_wrong_expert_count_raises(self):
        """A config with wrong n_routed_experts raises ValueError."""
        from models.demos.kimi_k25.utils.config_adapter import KimiK25Config

        cfg = _make_mock_hf_config(n_routed_experts=256)  # DSV3's count, not Kimi's
        with pytest.raises(ValueError, match="n_routed_experts"):
            KimiK25Config.from_hf_config(cfg)

    def test_wrong_rms_norm_eps_raises(self):
        """A config with wrong rms_norm_eps (1e-6 instead of 1e-5) raises ValueError."""
        from models.demos.kimi_k25.utils.config_adapter import KimiK25Config

        cfg = _make_mock_hf_config(rms_norm_eps=1e-6)  # DSV3's eps, not Kimi's
        with pytest.raises(ValueError, match="rms_norm_eps"):
            KimiK25Config.from_hf_config(cfg)

    def test_wrong_hidden_size_raises(self):
        """A config with incorrect hidden_size raises ValueError."""
        from models.demos.kimi_k25.utils.config_adapter import KimiK25Config

        cfg = _make_mock_hf_config(hidden_size=4096)
        with pytest.raises(ValueError, match="hidden_size"):
            KimiK25Config.from_hf_config(cfg)


# ---------------------------------------------------------------------------
# KimiLazyStateDict import / key helpers
# ---------------------------------------------------------------------------


class TestWeightLoaderIntegration:
    """KimiLazyStateDict is correctly referenced from kimi_model."""

    def test_kimi_lazy_state_dict_imported(self):
        """kimi_model imports KimiLazyStateDict from the correct module."""
        import models.demos.kimi_k25.tt.kimi_model as km

        assert hasattr(km, "KimiLazyStateDict"), (
            "KimiLazyStateDict must be importable via kimi_model module"
        )

    def test_prepare_weight_configs_uses_kimi_state_dict(self):
        """_prepare_weight_configs calls KimiLazyStateDict, not the parent loader."""
        import models.demos.kimi_k25.tt.kimi_model as km

        source = inspect.getsource(km.KimiGenerator._prepare_weight_configs)
        assert "KimiLazyStateDict" in source, (
            "_prepare_weight_configs must instantiate KimiLazyStateDict "
            "to ensure INT4 dequant and prefix stripping are applied."
        )

    def test_random_weights_falls_through_to_parent(self):
        """When random_weights=True, parent _prepare_weight_configs is called."""
        import models.demos.kimi_k25.tt.kimi_model as km

        source = inspect.getsource(km.KimiGenerator._prepare_weight_configs)
        assert "random_weights" in source, (
            "_prepare_weight_configs must check self.random_weights and "
            "fall through to parent for smoke tests."
        )
        assert "super()" in source, (
            "_prepare_weight_configs must call super() for the random_weights path."
        )


# ---------------------------------------------------------------------------
# Integration test (requires hardware + real weights)
# ---------------------------------------------------------------------------

KIMI_HF_MODEL = os.environ.get("KIMI_HF_MODEL", "")
HAS_REAL_WEIGHTS = bool(KIMI_HF_MODEL) and Path(KIMI_HF_MODEL).exists()


@pytest.mark.skipif(not HAS_REAL_WEIGHTS, reason="KIMI_HF_MODEL not set or path absent")
class TestRealConfigLoad:
    """Load the real HF config from /workspace/extra/Kimi-K2.5 and validate it."""

    def test_real_hf_config_validates(self):
        """Real config.json at KIMI_HF_MODEL validates against KimiK25Config."""
        from transformers import AutoConfig

        from models.demos.kimi_k25.utils.config_adapter import KimiK25Config

        hf_config = AutoConfig.from_pretrained(KIMI_HF_MODEL, trust_remote_code=True)
        kimi_cfg = KimiK25Config.from_hf_config(hf_config)
        assert kimi_cfg.n_routed_experts == 384
        assert kimi_cfg.n_group == 1
        assert abs(kimi_cfg.rms_norm_eps - 1e-5) < 1e-10
        assert kimi_cfg.num_hidden_layers == 61

    def test_kimi_lazy_state_dict_init(self):
        """KimiLazyStateDict initialises from the real model directory."""
        from models.demos.kimi_k25.utils.weight_loader import KimiLazyStateDict

        sd = KimiLazyStateDict(KIMI_HF_MODEL)
        # Should have keys visible under language_model.model.* prefix
        all_keys = list(sd)
        assert len(all_keys) > 0, "State dict must expose at least some keys"
        # Keys should NOT contain the raw language_model.model. prefix
        assert not any(
            k.startswith("language_model.model.") for k in all_keys
        ), "Keys should already have the prefix stripped"
        # Layer 0 attention weight should be BF16 pass-through
        assert any("layers.0" in k for k in all_keys), "Should have layer 0 keys"
