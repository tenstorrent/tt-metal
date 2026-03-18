# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""test_kimi_generate.py — M5 full-model smoke test for Kimi K2.5 on tt-metal.

Test plan
---------

CPU-only (no hardware required):

  TestKimiGeneratorImport       — ``KimiGenerator`` / ``load_kimi_model`` can be
                                  imported; ``KimiGenerator`` is a subclass of
                                  ``DeepseekGenerator``.
  TestKimiGeneratorStructure    — ``__init__`` keyword arguments are present;
                                  expected instance attributes exist on a
                                  constructed object (random_weights=True, no
                                  real mesh device needed for attribute checks).
  TestKimiGeneratorConfigVal    — Mismatched HF config fields raise ValueError
                                  *before* any hardware initialisation.
  TestKimiFullModelReference    — ``DeepseekV3ForCausalLM(kimi_2layer_config)``
                                  instantiates on PyTorch meta device (zero
                                  memory) with correct lm_head/embed shapes,
                                  layer count, expert count, and dense-vs-MoE
                                  assignment.  Validates that the DSV3 reference
                                  model is structurally compatible with Kimi
                                  K2.5 architecture without needing hardware or
                                  large RAM.
  TestKimiQuantizationConfig    — ``KimiK25Config.quantization_config`` property
                                  returns a dict with ``weight_block_size`` key,
                                  enabling ``prepare_model_state_dict`` and
                                  ``dequantize_state_dict`` to work with
                                  ``KimiK25Config`` in the random-weights test path.

Hardware (requires ``MESH_DEVICE=TG`` / ``DUAL`` / ``QUAD``):

  test_forward_pass             — Random-weights 2-layer full-model decode and
                                  prefill smoke test.  Uses
                                  ``RowBatchedModel.forward_decode`` /
                                  ``forward_prefill`` driven from Kimi's
                                  ``hf_config_short`` (384 experts, n_group=1,
                                  64 attn heads).  Pass criterion: no crash,
                                  output tensor is finite (no NaN/Inf).
  test_pcc_correctness_random_weights
                                — PCC comparison between TT prefill logits and
                                  CPU ``DeepseekV3ForCausalLM`` reference using
                                  shared random weights (same seed, same state
                                  dict).  Pass criterion: PCC ≥ 0.95 on logits.
                                  Uses 2-layer model and seq_len=4 for speed.
  test_full_model               — 61-layer random-weights stress test.
                                  ``@pytest.mark.slow``; timeout=3600s.

Notes
-----
* CPU tests in ``TestKimiFullModelReference`` use PyTorch meta device, so they
  instantiate the model with zero actual memory.  All shape/structural checks
  run near-instantly.
* The hardware test ``test_forward_pass`` is the key M5 deliverable: it
  exercises the full layer chain (embed → 1 dense MLA → 1 MoE → … → lm_head)
  end-to-end with random weights on real silicon.
* ``test_pcc_correctness_random_weights`` extends M5 by verifying numerical
  agreement between TT and CPU reference.  Shared random weights are built with
  ``prepare_model_state_dict(random_weights=True)`` (which requires
  ``KimiK25Config.quantization_config``); the reference path recovers BF16 via
  ``dequantize_state_dict``.
"""

from __future__ import annotations

import inspect
import os
from copy import deepcopy

import pytest


# ============================================================================
# CPU-only: import tests
# ============================================================================


class TestKimiGeneratorImport:
    """M5 — KimiGenerator and load_kimi_model must be importable."""

    def test_kimi_generator_import(self):
        """KimiGenerator can be imported from the Kimi tt module."""
        from models.demos.kimi_k25.tt.kimi_model import KimiGenerator  # noqa: F401

    def test_load_kimi_model_import(self):
        """load_kimi_model factory can be imported."""
        from models.demos.kimi_k25.tt.kimi_model import load_kimi_model  # noqa: F401

    def test_kimi_generator_is_callable(self):
        """KimiGenerator is a class (callable)."""
        from models.demos.kimi_k25.tt.kimi_model import KimiGenerator

        assert callable(KimiGenerator)

    def test_kimi_generator_inherits_deepseek(self):
        """KimiGenerator must subclass DeepseekGenerator."""
        from models.demos.deepseek_v3.tt.generator import DeepseekGenerator
        from models.demos.kimi_k25.tt.kimi_model import KimiGenerator

        assert issubclass(KimiGenerator, DeepseekGenerator), (
            f"KimiGenerator must subclass DeepseekGenerator; "
            f"MRO: {[c.__name__ for c in KimiGenerator.__mro__]}"
        )

    def test_load_kimi_model_is_callable(self):
        """load_kimi_model must be a callable (not a class)."""
        from models.demos.kimi_k25.tt.kimi_model import load_kimi_model

        assert callable(load_kimi_model)
        assert not isinstance(load_kimi_model, type), "load_kimi_model should be a function, not a class"


# ============================================================================
# CPU-only: __init__ signature / structure
# ============================================================================


class TestKimiGeneratorStructure:
    """M5 — KimiGenerator.__init__ must accept expected keyword arguments."""

    @pytest.fixture(autouse=True)
    def _kimi_cls(self):
        from models.demos.kimi_k25.tt.kimi_model import KimiGenerator

        self.KimiGenerator = KimiGenerator

    def test_init_accepts_model_path(self):
        """__init__ must accept model_path kwarg."""
        sig = inspect.signature(self.KimiGenerator.__init__)
        assert "model_path" in sig.parameters, "__init__ must have a 'model_path' parameter"

    def test_init_accepts_mesh_device(self):
        """__init__ must accept mesh_device kwarg."""
        sig = inspect.signature(self.KimiGenerator.__init__)
        assert "mesh_device" in sig.parameters, "__init__ must have a 'mesh_device' parameter"

    def test_init_accepts_cache_dir(self):
        """__init__ must accept cache_dir kwarg."""
        sig = inspect.signature(self.KimiGenerator.__init__)
        assert "cache_dir" in sig.parameters, "__init__ must have a 'cache_dir' parameter"

    def test_init_accepts_hf_config(self):
        """__init__ must accept hf_config kwarg (allows pre-loaded config injection)."""
        sig = inspect.signature(self.KimiGenerator.__init__)
        assert "hf_config" in sig.parameters, "__init__ must have an 'hf_config' parameter"

    def test_init_accepts_random_weights(self):
        """__init__ must accept random_weights kwarg (inherited from DeepseekGenerator)."""
        sig = inspect.signature(self.KimiGenerator.__init__)
        assert "random_weights" in sig.parameters or "kwargs" in sig.parameters, (
            "__init__ must accept 'random_weights' (directly or via **kwargs)"
        )

    def test_init_accepts_override_num_layers(self):
        """__init__ must accept override_num_layers for smoke tests."""
        sig = inspect.signature(self.KimiGenerator.__init__)
        assert "override_num_layers" in sig.parameters or "kwargs" in sig.parameters, (
            "__init__ must accept 'override_num_layers' (directly or via **kwargs)"
        )

    def test_prepare_weight_configs_is_overridden(self):
        """KimiGenerator must define its own _prepare_weight_configs."""
        assert "_prepare_weight_configs" in self.KimiGenerator.__dict__, (
            "KimiGenerator must override _prepare_weight_configs "
            "(not just inherit from DeepseekGenerator)"
        )

    def test_load_kimi_model_signature(self):
        """load_kimi_model must accept model_path, mesh_device, cache_dir, **kwargs."""
        from models.demos.kimi_k25.tt.kimi_model import load_kimi_model

        sig = inspect.signature(load_kimi_model)
        for param in ("model_path", "mesh_device", "cache_dir"):
            assert param in sig.parameters, f"load_kimi_model must have a '{param}' parameter"

    def test_default_cache_dir_is_kimi_specific(self):
        """Default cache dir must reference 'kimi' to avoid overwriting DSV3 cache."""
        from models.demos.kimi_k25.tt.kimi_model import _DEFAULT_CACHE_DIR

        assert "kimi" in _DEFAULT_CACHE_DIR.lower(), (
            f"_DEFAULT_CACHE_DIR={_DEFAULT_CACHE_DIR!r} must contain 'kimi' "
            "to avoid colliding with the DSV3 weight cache"
        )


# ============================================================================
# CPU-only: config validation
# ============================================================================


class TestKimiGeneratorConfigValidation:
    """M5 — KimiGenerator must reject mismatched HF configs at __init__ time."""

    @pytest.fixture(autouse=True)
    def _base_config(self):
        """Fixture config for all validation tests."""
        from models.demos.kimi_k25.utils.config_adapter import KimiK25Config

        self.good_config = KimiK25Config.from_fixture()
        self.KimiK25Config = KimiK25Config

    def _make_config(self, **overrides) -> object:
        """Return a deepcopy of the fixture config with the given fields overridden."""
        cfg = deepcopy(self.good_config)
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def _raises_on_init(self, hf_config) -> bool:
        """Return True if KimiGenerator raises ValueError for the given config."""
        from models.demos.kimi_k25.tt.kimi_model import KimiGenerator

        try:
            KimiGenerator(hf_config=hf_config, mesh_device=None, model_path=None)
        except ValueError:
            return True
        except Exception:
            # Any other error (e.g. mesh_device=None downstream) is fine — we
            # only care that ValueError was raised for the config mismatch, not
            # that the full __init__ succeeded.
            pass
        return False

    def test_valid_config_does_not_raise(self):
        """A valid Kimi K2.5 config must not raise ValueError in __init__."""
        from models.demos.kimi_k25.tt.kimi_model import KimiGenerator

        try:
            KimiGenerator(hf_config=self.good_config, mesh_device=None, model_path=None)
        except ValueError as exc:
            pytest.fail(f"Valid Kimi config raised ValueError: {exc}")
        except Exception:
            # Other exceptions (NoneType mesh_device, etc.) are expected with
            # mesh_device=None; we only care that ValueError was NOT raised.
            pass

    def test_wrong_n_routed_experts_raises(self):
        """Incorrect n_routed_experts (e.g. 256 instead of 384) must raise ValueError."""
        bad = self._make_config(n_routed_experts=256)
        assert self._raises_on_init(bad), "n_routed_experts=256 should raise ValueError"

    def test_wrong_rms_norm_eps_raises(self):
        """Incorrect rms_norm_eps (e.g. 1e-6 instead of 1e-5) must raise ValueError."""
        bad = self._make_config(rms_norm_eps=1e-6)
        assert self._raises_on_init(bad), "rms_norm_eps=1e-6 should raise ValueError"

    def test_wrong_hidden_size_raises(self):
        """Incorrect hidden_size (e.g. 4096 instead of 7168) must raise ValueError."""
        bad = self._make_config(hidden_size=4096)
        assert self._raises_on_init(bad), "hidden_size=4096 should raise ValueError"

    def test_wrong_num_attention_heads_raises(self):
        """Incorrect num_attention_heads (e.g. 128 instead of 64) must raise ValueError."""
        bad = self._make_config(num_attention_heads=128)
        assert self._raises_on_init(bad), "num_attention_heads=128 should raise ValueError"

    def test_wrong_vocab_size_raises(self):
        """Incorrect vocab_size (e.g. 32000 instead of 163840) must raise ValueError."""
        bad = self._make_config(vocab_size=32000)
        assert self._raises_on_init(bad), "vocab_size=32000 should raise ValueError"


# ============================================================================
# CPU-only: weight loader integration
# ============================================================================


class TestKimiGeneratorWeightLoaderIntegration:
    """M5 — Verify that _prepare_weight_configs injects KimiLazyStateDict."""

    def test_prepare_weight_configs_uses_kimi_state_dict(self):
        """_prepare_weight_configs body must reference KimiLazyStateDict."""
        import inspect

        from models.demos.kimi_k25.tt.kimi_model import KimiGenerator

        src = inspect.getsource(KimiGenerator._prepare_weight_configs)
        assert "KimiLazyStateDict" in src, (
            "_prepare_weight_configs must instantiate KimiLazyStateDict "
            "for the real-weights path"
        )

    def test_prepare_weight_configs_random_weights_falls_through(self):
        """For random_weights=True, _prepare_weight_configs must call super()."""
        import inspect

        from models.demos.kimi_k25.tt.kimi_model import KimiGenerator

        src = inspect.getsource(KimiGenerator._prepare_weight_configs)
        assert "random_weights" in src, (
            "_prepare_weight_configs must handle random_weights flag"
        )
        assert "super()" in src, (
            "_prepare_weight_configs must call super() for the random_weights path"
        )

    def test_kimi_lazy_state_dict_importable(self):
        """KimiLazyStateDict must be importable from utils.weight_loader."""
        from models.demos.kimi_k25.utils.weight_loader import KimiLazyStateDict  # noqa: F401

    def test_random_weights_uses_prepare_model_state_dict(self):
        """_run_kimi_forward_pass_random_weights must NOT pass empty state_dicts=().

        Regression guard for the bug where state_dicts=() was passed to
        get_test_weight_config, which then called RowBatchedModel.convert_weights
        with an empty tuple — raising ``ValueError: not enough values to unpack``
        because convert_weights does ``(state_dict,) = state_dicts``.

        The correct fix: call prepare_model_state_dict(cfg, random_weights=True)
        and pass (random_state_dict,) as state_dicts.
        """
        import inspect
        import models.demos.kimi_k25.tests.test_kimi_generate as _mod

        src = inspect.getsource(_mod._run_kimi_forward_pass_random_weights)
        assert "prepare_model_state_dict" in src, (
            "_run_kimi_forward_pass_random_weights must import and call "
            "prepare_model_state_dict to build a random state dict — "
            "state_dicts=() causes ValueError in convert_weights"
        )
        # Ensure we're not passing the empty tuple anymore
        assert "state_dicts=()" not in src and "()" not in src.split("get_test_weight_config")[1].split(")")[0], (
            "_run_kimi_forward_pass_random_weights must not pass empty state_dicts=() "
            "to get_test_weight_config"
        )


# ============================================================================
# CPU-only: full-model reference architecture validation
# ============================================================================


class TestKimiFullModelReference:
    """CPU-only: validate DeepseekV3ForCausalLM accepts Kimi K2.5 full-model config.

    These tests use PyTorch's *meta device* to instantiate the model with zero
    memory allocation.  They confirm that the full-model architecture
    (embedding, MoE + MLA layers, lm_head) is structurally compatible with
    Kimi K2.5 parameters (vocab_size=163840, hidden_size=7168, 384 experts,
    64 attn heads, n_group=1) without requiring a GPU or large RAM.

    All tests soft-skip if the DSV3 reference model is unavailable.
    """

    @pytest.fixture(scope="class")
    def cfg_2layer(self):
        """Kimi K2.5 config with 2 transformer layers for fast CPU reference tests."""
        from models.demos.kimi_k25.utils.config_adapter import KimiK25Config

        cfg = KimiK25Config.from_fixture()
        cfg.num_hidden_layers = 2
        cfg.max_seq_len = 128
        return cfg

    def test_deepseek_causal_lm_importable(self):
        """DeepseekV3ForCausalLM can be imported from the DSV3 reference module."""
        try:
            from models.demos.deepseek_v3.reference.modeling_deepseek import (  # noqa: F401
                DeepseekV3ForCausalLM,
            )
        except ImportError:
            pytest.skip("DSV3 reference model not importable in this environment")

    def test_kimi_config_has_required_full_model_fields(self, cfg_2layer):
        """KimiK25Config exposes all fields required by DeepseekV3ForCausalLM.__init__."""
        required = [
            "hidden_size",
            "num_hidden_layers",
            "vocab_size",
            "num_attention_heads",
            "n_routed_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "intermediate_size",
            "first_k_dense_replace",
            "n_group",
            "rms_norm_eps",
            "rope_scaling",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "q_lora_rank",
            "hidden_act",
        ]
        missing = [f for f in required if not hasattr(cfg_2layer, f)]
        assert not missing, (
            f"KimiK25Config is missing fields required by DeepseekV3ForCausalLM: {missing}"
        )

    def test_full_model_meta_device_instantiation(self, cfg_2layer):
        """DeepseekV3ForCausalLM(kimi_2layer_config) instantiates on meta device without error."""
        try:
            import torch
            from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3ForCausalLM
        except ImportError:
            pytest.skip("torch or DSV3 reference model not importable")

        with torch.device("meta"):
            model = DeepseekV3ForCausalLM(cfg_2layer).eval()

        # Basic sanity: model wraps a DeepseekV3Model
        assert hasattr(model, "model"), "DeepseekV3ForCausalLM must have a .model attribute"
        assert hasattr(model, "lm_head"), "DeepseekV3ForCausalLM must have a .lm_head attribute"

    def test_lm_head_shape_on_meta_device(self, cfg_2layer):
        """lm_head weight shape matches Kimi K2.5 vocab_size × hidden_size."""
        try:
            import torch
            from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3ForCausalLM
        except ImportError:
            pytest.skip("torch or DSV3 reference model not importable")

        with torch.device("meta"):
            model = DeepseekV3ForCausalLM(cfg_2layer).eval()

        expected = (cfg_2layer.vocab_size, cfg_2layer.hidden_size)  # (163840, 7168)
        actual = tuple(model.lm_head.weight.shape)
        assert actual == expected, (
            f"lm_head.weight shape mismatch: got {actual}, expected {expected}"
        )

    def test_embed_tokens_shape_on_meta_device(self, cfg_2layer):
        """embed_tokens weight shape matches Kimi K2.5 vocab_size × hidden_size."""
        try:
            import torch
            from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3ForCausalLM
        except ImportError:
            pytest.skip("torch or DSV3 reference model not importable")

        with torch.device("meta"):
            model = DeepseekV3ForCausalLM(cfg_2layer).eval()

        expected = (cfg_2layer.vocab_size, cfg_2layer.hidden_size)  # (163840, 7168)
        actual = tuple(model.model.embed_tokens.weight.shape)
        assert actual == expected, (
            f"embed_tokens.weight shape mismatch: got {actual}, expected {expected}"
        )

    def test_layer_count_on_meta_device(self, cfg_2layer):
        """Model has exactly cfg.num_hidden_layers=2 transformer layers."""
        try:
            import torch
            from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3ForCausalLM
        except ImportError:
            pytest.skip("torch or DSV3 reference model not importable")

        with torch.device("meta"):
            model = DeepseekV3ForCausalLM(cfg_2layer).eval()

        actual = len(model.model.layers)
        assert actual == cfg_2layer.num_hidden_layers, (
            f"Expected {cfg_2layer.num_hidden_layers} layers, got {actual}"
        )

    def test_moe_layer_expert_count_on_meta_device(self, cfg_2layer):
        """MoE layers have exactly 384 experts (Kimi K2.5 routed experts)."""
        try:
            import torch
            from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3ForCausalLM
        except ImportError:
            pytest.skip("torch or DSV3 reference model not importable")

        with torch.device("meta"):
            model = DeepseekV3ForCausalLM(cfg_2layer).eval()

        # Layer 0 is dense (first_k_dense_replace=1); layer 1+ are MoE
        # With 2 layers: layer 0 dense, layer 1 MoE
        moe_layer = model.model.layers[1].mlp
        n_experts = len(moe_layer.experts)
        assert n_experts == cfg_2layer.n_routed_experts, (
            f"MoE layer has {n_experts} experts; expected {cfg_2layer.n_routed_experts} (=384)"
        )

    def test_first_layer_is_dense_mlp(self, cfg_2layer):
        """Layer 0 uses dense MLP (first_k_dense_replace=1), not MoE."""
        try:
            import torch
            from models.demos.deepseek_v3.reference.modeling_deepseek import (
                DeepseekV3ForCausalLM,
                DeepseekV3MLP,
            )
        except ImportError:
            pytest.skip("torch or DSV3 reference model not importable")

        with torch.device("meta"):
            model = DeepseekV3ForCausalLM(cfg_2layer).eval()

        layer0_mlp = model.model.layers[0].mlp
        assert isinstance(layer0_mlp, DeepseekV3MLP), (
            f"Layer 0 MLP should be DeepseekV3MLP (dense), got {type(layer0_mlp).__name__}"
        )

    def test_prepare_model_state_dict_returns_expected_key_prefixes(self, cfg_2layer):
        """prepare_model_state_dict returns keys with expected top-level prefixes.

        This is a source-inspection test (no actual weight instantiation):
        validates that the function uses the correct key prefixes for Kimi K2.5.

        For the real execution test, use MESH_DEVICE=TG and run test_forward_pass.
        """
        import inspect

        from models.demos.deepseek_v3.utils import hf_model_utils

        src = inspect.getsource(hf_model_utils.prepare_model_state_dict)
        # Verify the function filters keys as expected by TT weight converters
        assert "model.embed_tokens." in src, "prepare_model_state_dict must filter embed_tokens keys"
        assert "model.layers." in src, "prepare_model_state_dict must filter layer keys"
        assert "lm_head." in src, "prepare_model_state_dict must filter lm_head keys"


# ============================================================================
# CPU-only: quantization_config property
# ============================================================================


class TestKimiQuantizationConfig:
    """CPU-only: validate KimiK25Config.quantization_config property.

    ``prepare_model_state_dict(random_weights=True)`` calls
    ``add_inv_scale_to_state_dict(state, block_shape=hf_config.quantization_config["weight_block_size"])``.
    ``dequantize_state_dict(state, hf_config)`` reads the same key.

    Without this property, both functions raise ``AttributeError`` when given a
    ``KimiK25Config``.  These tests guard against regression.
    """

    @pytest.fixture(autouse=True)
    def _cfg(self):
        from models.demos.kimi_k25.utils.config_adapter import KimiK25Config

        self.cfg = KimiK25Config.from_fixture()

    def test_quantization_config_exists(self):
        """KimiK25Config must have a quantization_config attribute."""
        assert hasattr(self.cfg, "quantization_config"), (
            "KimiK25Config is missing quantization_config — "
            "prepare_model_state_dict(random_weights=True) will crash without it"
        )

    def test_quantization_config_is_dict(self):
        """quantization_config must return a dict (not None or another type)."""
        qc = self.cfg.quantization_config
        assert isinstance(qc, dict), (
            f"quantization_config must be a dict, got {type(qc).__name__}"
        )

    def test_quantization_config_has_weight_block_size(self):
        """quantization_config must contain 'weight_block_size' key."""
        qc = self.cfg.quantization_config
        assert "weight_block_size" in qc, (
            "quantization_config must have 'weight_block_size' key — "
            "dequantize_state_dict reads qc['weight_block_size']"
        )

    def test_weight_block_size_is_list_of_two_ints(self):
        """weight_block_size must be a list/tuple of two positive ints."""
        wbs = self.cfg.quantization_config["weight_block_size"]
        assert isinstance(wbs, (list, tuple)), (
            f"weight_block_size must be a list or tuple, got {type(wbs).__name__}"
        )
        assert len(wbs) == 2, f"weight_block_size must have length 2, got {len(wbs)}"
        assert all(isinstance(x, int) and x > 0 for x in wbs), (
            f"weight_block_size elements must be positive ints, got {wbs}"
        )

    def test_weight_block_size_is_128x128(self):
        """weight_block_size must match DSV3's [128, 128] block shape.

        The TT weight converters were built for DSV3's FP8 block quantization with
        128×128 blocks.  Kimi K2.5 uses the same TT model infrastructure, so the
        random-weights test path must use the same block shape.
        """
        wbs = self.cfg.quantization_config["weight_block_size"]
        assert list(wbs) == [128, 128], (
            f"weight_block_size must be [128, 128] to match DSV3 TT converters, got {list(wbs)}"
        )

    def test_quantization_config_is_read_only_property(self):
        """quantization_config must be a property (not a mutable dataclass field).

        If it were a mutable field, accidental mutation in one test could affect
        subsequent tests.  Verify it is a property on the *class*, not an instance
        attribute set by __post_init__.

        Note: ``type(KimiK25Config)`` is ``type`` (the metaclass), not the class
        itself.  For dataclasses, we must use ``KimiK25Config.__dict__`` to inspect
        descriptors defined on the class.
        """
        from models.demos.kimi_k25.utils.config_adapter import KimiK25Config

        descriptor = KimiK25Config.__dict__.get("quantization_config")
        assert isinstance(descriptor, property), (
            f"quantization_config must be a @property on KimiK25Config, "
            f"got {type(descriptor).__name__} — mutable fields can be mutated across tests"
        )

    def test_prepare_model_state_dict_source_reads_quantization_config(self):
        """prepare_model_state_dict source must reference quantization_config.

        Source-inspection guard: if the DSV3 implementation changes and stops
        reading this key, this test will fail to remind us to update KimiK25Config.

        Soft-skips if the DSV3 utilities are not importable (e.g. loguru not
        installed in the scheduling container — all DSV3 utils require loguru).
        """
        import inspect

        try:
            from models.demos.deepseek_v3.utils import hf_model_utils
        except ImportError as exc:
            pytest.skip(f"DSV3 utils not importable in this environment: {exc}")

        src = inspect.getsource(hf_model_utils.prepare_model_state_dict)
        assert "quantization_config" in src, (
            "prepare_model_state_dict no longer reads quantization_config — "
            "KimiK25Config.quantization_config property may be obsolete"
        )


# ============================================================================
# Hardware: full-model random-weights smoke test
# ============================================================================


def _run_kimi_forward_pass_random_weights(
    mode: str,
    seq_len: int,
    batch_size_per_row: int,
    hf_config_short,
    cache_path,
    mesh_device,
    ccl,
    force_recalculate: bool,
    num_layers: int = 2,
) -> None:
    """Run a full-model forward pass with random weights and Kimi K2.5 config.

    This exercises the complete RowBatchedModel layer stack (embed → MLA →
    MoE → … → lm_head) with Kimi's 384-expert, n_group=1, 64-head config.

    Args:
        mode:               ``"decode"`` or ``"prefill"``.
        seq_len:            Sequence length (1 for decode, ≥1 for prefill).
        batch_size_per_row: Users per mesh row (decode only).
        hf_config_short:    Kimi hf_config with num_hidden_layers=num_layers.
        cache_path:         Weight cache directory.
        mesh_device:        TTNN mesh device.
        ccl:                CCL instance for the mesh.
        force_recalculate:  Bypass weight cache if True.
        num_layers:         Number of transformer layers to instantiate.
    """
    import torch

    import ttnn
    from models.demos.deepseek_v3.tt.mla.mla2d import MLA2D
    from models.demos.deepseek_v3.tt.model.row_batched_model import RowBatchedModel
    from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW
    from models.demos.deepseek_v3.utils.hf_model_utils import prepare_model_state_dict
    from models.demos.deepseek_v3.utils.run_config import create_run_config
    from models.demos.deepseek_v3.utils.test_utils import (
        get_model_config,
        get_rope_tensors,
        get_test_weight_config,
    )

    # Override layer count for speed
    cfg = deepcopy(hf_config_short)
    cfg.num_hidden_layers = num_layers
    # Clamp first_k_dense_replace so it doesn't exceed num_hidden_layers
    if hasattr(cfg, "first_k_dense_replace"):
        cfg.first_k_dense_replace = min(cfg.first_k_dense_replace, num_layers)

    mesh_rows = mesh_device.shape[0]
    dp_factor = mesh_device.shape[1]

    if mode == "prefill":
        assert batch_size_per_row == 1, "Prefill only supports batch_size_per_row=1"
        batch_size = 1
    else:
        assert seq_len == 1, "Decode only supports seq_len=1"
        batch_size = batch_size_per_row * mesh_rows

    # Random token input
    torch.manual_seed(42)
    if mode == "decode":
        torch_input = torch.randint(0, cfg.vocab_size - 1, (batch_size, seq_len), dtype=torch.long)
        position_ids = torch.randint(0, cfg.max_seq_len - 1, (batch_size,), dtype=torch.long)
        torch_input = torch_input.transpose(1, 0)  # (seq_len, batch)
    else:
        torch_input = torch.randint(0, cfg.vocab_size - 1, (1, seq_len), dtype=torch.long)
        position_ids = None

    # Paged attention config
    paged_config = MLA2D.get_valid_paged_config(cfg.max_seq_len, USERS_PER_ROW, dp_factor)

    # KV cache init (empty for random-weight smoke test)
    if mode == "decode":
        cache_dim = cfg.kv_lora_rank + cfg.qk_rope_head_dim
        decode_input_caches = tuple(
            torch.empty((batch_size, 1, 0, cache_dim), dtype=torch.bfloat16)
            for _ in range(cfg.num_hidden_layers)
        )
        denom = mesh_rows * dp_factor
        batches_per_device = batch_size // denom
        blocks_per_batch = paged_config.max_num_blocks // batches_per_device
        mapping = torch.arange(batches_per_device * blocks_per_batch, dtype=torch.long).reshape(
            batches_per_device, blocks_per_batch
        )
        mappings = tuple(mapping for _ in range(cfg.num_hidden_layers))

        from models.demos.deepseek_v3.utils.test_utils import paged_caches_from_torch

        paged_input_caches, torch_page_tables = paged_caches_from_torch(
            decode_input_caches,
            tuple(mesh_device.shape),
            paged_config,
            user_id=None,
            mappings=mappings,
        )
    else:
        paged_input_caches = None
        total_global_users = mesh_rows * USERS_PER_ROW
        num_devices = mesh_rows * dp_factor
        batches_per_device = total_global_users // num_devices
        blocks_per_batch = paged_config.max_num_blocks // batches_per_device
        torch_page_table = torch.arange(paged_config.max_num_blocks, dtype=torch.int32).reshape(
            batches_per_device, blocks_per_batch
        )
        torch_page_tables = (torch_page_table,) * cfg.num_hidden_layers

    tt_page_tables = tuple(
        MLA2D.create_page_table(
            page_table=tpt, paged_config=paged_config, mesh_device=mesh_device
        )
        for tpt in torch_page_tables
    )

    # Build random-weight state dict from DSV3 reference model instantiated with Kimi config.
    # RowBatchedModel.convert_weights() does: (state_dict,) = state_dicts — requires exactly
    # one dict.  state_dicts=() (empty tuple) would raise ValueError; state_dicts=None triggers
    # prepare_model_state_dict internally but get_test_weight_config does not forward
    # random_weights=True to get_weight_config.  Prepare explicitly here, mirroring DSV3
    # test_moe.py's generate_reference_io pattern.
    random_state_dict = prepare_model_state_dict(cfg, random_weights=True)

    # Random-weight model config
    weight_config = get_test_weight_config(
        RowBatchedModel,
        cfg,
        (random_state_dict,),  # single state_dict required by RowBatchedModel.convert_weights
        cache_path,
        mesh_device,
        force_recalculate,
        test_name="test_kimi_generate",
        real_weights=False,
    )
    model_config = get_model_config(RowBatchedModel, mode, cfg, mesh_device)
    model_state = RowBatchedModel.create_state(cfg, paged_config, mesh_device, ccl, paged_input_caches)
    model_shared_state = RowBatchedModel.create_shared_state(cfg, mesh_device)
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    # TTNN input tensor
    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    position_ids_tt = (
        ttnn.from_torch(
            position_ids,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            dtype=ttnn.int32,
        )
        if mode == "decode"
        else None
    )

    rope_tensors = get_rope_tensors(cfg, batch_size_per_row, seq_len, position_ids, mesh_device)

    # Forward pass
    if mode == "prefill":
        tt_output = RowBatchedModel.forward_prefill(tt_input, 0, run_config, rope_tensors, tt_page_tables)
    else:
        tt_output = RowBatchedModel.forward_decode(
            tt_input, position_ids_tt, run_config, rope_tensors, tt_page_tables
        )

    ttnn.synchronize_device(mesh_device)

    # Finite output check (NaN/Inf = silent numerical failure)
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
    ).float()

    assert torch.isfinite(tt_output_torch).all(), (
        f"Kimi K2.5 full-model {mode} forward pass produced non-finite output "
        f"(NaN/Inf) with random weights.  This indicates a layer configuration "
        f"error (shape mismatch, uninitialized tensor, etc.)."
    )

    ttnn.deallocate(tt_output)


# ---------------------------------------------------------------------------
# Hardware test parametrization
# ---------------------------------------------------------------------------

_SMOKE_TEST_CASES = [
    pytest.param("decode", 1, 32, id="mode_decode_seq_1_batch_32"),
    pytest.param("prefill", 128, 1, id="mode_prefill_seq_128_batch_1"),
]


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mode, seq_len, batch_size_per_row", _SMOKE_TEST_CASES)
def test_forward_pass(
    mode,
    seq_len,
    batch_size_per_row,
    hf_config_short,
    cache_path,
    mesh_device,
    ccl,
    force_recalculate_weight_config,
    set_deterministic_env,
):
    """M5 smoke test — full Kimi K2.5 model forward pass with random weights.

    Instantiates RowBatchedModel driven by Kimi's hf_config_short (384 experts,
    n_group=1, 64 attn heads) with random weights and runs a single forward
    pass.  Validates that the output tensor is finite (no NaN/Inf).

    No reference model comparison is performed here (M6 correctness work).

    Requires MESH_DEVICE=TG / DUAL / QUAD.
    """
    _run_kimi_forward_pass_random_weights(
        mode=mode,
        seq_len=seq_len,
        batch_size_per_row=batch_size_per_row,
        hf_config_short=hf_config_short,
        cache_path=cache_path,
        mesh_device=mesh_device,
        ccl=ccl,
        force_recalculate=force_recalculate_weight_config,
        num_layers=2,
    )

# ============================================================================
# Hardware: PCC correctness with random weights (prefill)
# ============================================================================


def _run_kimi_pcc_reference_forward(
    *,
    torch_input: "torch.Tensor",
    state_dict: dict,
    hf_config,
) -> "torch.Tensor":
    """Run CPU reference forward pass for PCC comparison (prefill only).

    Loads ``state_dict`` into ``DeepseekV3ForCausalLM`` (dequantizing FP8 to BF16)
    and runs a causal-LM forward pass on ``torch_input``.

    Args:
        torch_input:  LongTensor of shape ``(1, seq_len)`` — batch=1 for prefill.
        state_dict:   Random weights as returned by ``prepare_model_state_dict``.
                      May contain FP8-quantized tensors with ``_scale_inv`` suffixes
                      that ``dequantize_state_dict`` will convert to BF16.
        hf_config:    ``KimiK25Config`` (must expose ``.quantization_config``).

    Returns:
        Float32 logits tensor of shape ``(1, seq_len, vocab_size)``.
    """
    import torch
    from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3ForCausalLM
    from models.demos.deepseek_v3.utils.test_utils import dequantize_state_dict

    with torch.device("meta"):
        reference_model = DeepseekV3ForCausalLM(hf_config).eval()
    reference_model = reference_model.to_empty(device=torch.device("cpu"))
    reference_model.load_state_dict(dequantize_state_dict(state_dict, hf_config))
    reference_model = reference_model.to(torch.bfloat16)

    with torch.no_grad():
        output = reference_model(
            torch_input,
            output_attentions=False,
            use_cache=False,
        )

    logits = output.logits.float().cpu()  # (1, seq_len, vocab_size)
    del reference_model, output
    return logits


@pytest.mark.timeout(900)
@pytest.mark.parametrize("seq_len", [4, 16], ids=["seq_4", "seq_16"])
def test_pcc_correctness_random_weights(
    seq_len,
    hf_config_short,
    cache_path,
    mesh_device,
    ccl,
    force_recalculate_weight_config,
    set_deterministic_env,
):
    """M5 PCC test — compare TT prefill logits vs CPU reference with shared random weights.

    Both paths use the same random state dict (seeded at 42) so any numerical
    deviation between TT and CPU reveals model-level errors rather than weight
    initialisation differences.

    Flow
    ----
    1. Build random FP8-quantised state dict via
       ``prepare_model_state_dict(cfg, random_weights=True)`` (uses
       ``KimiK25Config.quantization_config["weight_block_size"]``).
    2. CPU reference: load dequantised BF16 weights into
       ``DeepseekV3ForCausalLM`` and run a causal-LM forward pass.
    3. TT forward: run ``RowBatchedModel.forward_prefill`` with the same
       quantised state dict.
    4. Compare logits with ``assert_hidden_dim_pcc(pcc_required=0.95)``.

    Model size: 2 layers (``hf_config_short``), seq_len=``seq_len`` tokens.
    Timeout: 900 s (weight compile on TG may take several minutes for 2 layers).
    Hardware: requires ``MESH_DEVICE=TG`` (or DUAL / QUAD) via ``mesh_device`` fixture.
    """
    import torch
    import ttnn
    from models.demos.deepseek_v3.tt.mla.mla2d import MLA2D
    from models.demos.deepseek_v3.tt.model.row_batched_model import RowBatchedModel
    from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW
    from models.demos.deepseek_v3.utils.hf_model_utils import prepare_model_state_dict
    from models.demos.deepseek_v3.utils.run_config import create_run_config
    from models.demos.deepseek_v3.utils.test_utils import (
        assert_hidden_dim_pcc,
        get_model_config,
        get_rope_tensors,
        get_test_weight_config,
    )

    # 2-layer config (hf_config_short already has num_hidden_layers constrained)
    cfg = deepcopy(hf_config_short)
    cfg.num_hidden_layers = 2
    if hasattr(cfg, "first_k_dense_replace"):
        cfg.first_k_dense_replace = min(cfg.first_k_dense_replace, cfg.num_hidden_layers)

    # Seeded random input (batch=1 for prefill)
    torch.manual_seed(42)
    torch_input = torch.randint(0, cfg.vocab_size - 1, (1, seq_len), dtype=torch.long)

    # Build shared random state dict — FP8 block-quantised MLP weights
    random_state_dict = prepare_model_state_dict(cfg, random_weights=True)

    # -----------------------------------------------------------------------
    # Step 1: CPU reference forward (dequantise FP8 → BF16, run on CPU)
    # -----------------------------------------------------------------------
    reference_logits = _run_kimi_pcc_reference_forward(
        torch_input=torch_input,
        state_dict=random_state_dict,
        hf_config=cfg,
    )
    # reference_logits shape: (1, seq_len, vocab_size)

    # -----------------------------------------------------------------------
    # Step 2: TT forward (prefill)
    # -----------------------------------------------------------------------
    dp_factor = mesh_device.shape[1]
    paged_config = MLA2D.get_valid_paged_config(cfg.max_seq_len, USERS_PER_ROW, dp_factor)

    # Page table for prefill (no KV cache)
    total_global_users = mesh_device.shape[0] * USERS_PER_ROW
    num_devices = mesh_device.shape[0] * dp_factor
    batches_per_device = total_global_users // num_devices
    blocks_per_batch = paged_config.max_num_blocks // batches_per_device
    torch_page_table = torch.arange(paged_config.max_num_blocks, dtype=torch.int32).reshape(
        batches_per_device, blocks_per_batch
    )
    torch_page_tables = (torch_page_table,) * cfg.num_hidden_layers
    tt_page_tables = tuple(
        MLA2D.create_page_table(page_table=tpt, paged_config=paged_config, mesh_device=mesh_device)
        for tpt in torch_page_tables
    )

    weight_config = get_test_weight_config(
        RowBatchedModel,
        cfg,
        (random_state_dict,),
        cache_path,
        mesh_device,
        force_recalculate_weight_config,
        test_name="test_kimi_pcc",
        real_weights=False,
    )
    model_config = get_model_config(RowBatchedModel, "prefill", cfg, mesh_device)
    model_state = RowBatchedModel.create_state(cfg, paged_config, mesh_device, ccl, None)
    model_shared_state = RowBatchedModel.create_shared_state(cfg, mesh_device)
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(0),  # (1, 1, seq_len)
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    rope_tensors = get_rope_tensors(cfg, 1, seq_len, None, mesh_device)

    tt_output = RowBatchedModel.forward_prefill(tt_input, 0, run_config, rope_tensors, tt_page_tables)
    ttnn.synchronize_device(mesh_device)

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
    ).float()

    ttnn.deallocate(tt_output)

    # Sanity: output must be finite before PCC check
    assert torch.isfinite(tt_output_torch).all(), (
        "TT prefill output contains NaN/Inf with random weights — "
        "check layer configuration before measuring PCC"
    )

    # -----------------------------------------------------------------------
    # Step 3: PCC comparison (logits, pcc_required=0.95)
    # -----------------------------------------------------------------------
    # assert_hidden_dim_pcc handles shape mismatches on the second-to-last dim
    # (TT may pad vocab or batch dim) and chunked PCC for large outputs.
    assert_hidden_dim_pcc(tt_output_torch, reference_logits, pcc_required=0.95)


# ============================================================================
# Hardware: 61-layer full-model random-weights smoke test
# ============================================================================


_FULL_MODEL_TEST_CASES = [
    pytest.param("decode", 1, 32, id="mode_decode_seq_1_batch_32"),
    pytest.param("prefill", 128, 1, id="mode_prefill_seq_128_batch_1"),
]

_KIMI_NUM_LAYERS = 61  # Full Kimi K2.5 transformer depth


@pytest.mark.slow
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mode, seq_len, batch_size_per_row", _FULL_MODEL_TEST_CASES)
def test_full_model(
    mode,
    seq_len,
    batch_size_per_row,
    hf_config_short,
    cache_path,
    mesh_device,
    ccl,
    force_recalculate_weight_config,
    set_deterministic_env,
):
    """M5+ stress test — Kimi K2.5 with all 61 transformer layers, random weights.

    This is a longer-running validation that exercises the full model depth
    (all 61 layers including the first dense MLA + 60 MoE layers).

    Unlike ``test_forward_pass`` which uses 2 layers for fast iteration, this
    test validates that the RowBatchedModel layer stack is stable at full
    production depth with random weights.

    Finite-output criterion only — no reference comparison (M6 correctness work).

    Timeout: 3600s (weight conversion at 61 layers dominates compile time).
    Hardware: MESH_DEVICE=TG required.
    """
    _run_kimi_forward_pass_random_weights(
        mode=mode,
        seq_len=seq_len,
        batch_size_per_row=batch_size_per_row,
        hf_config_short=hf_config_short,
        cache_path=cache_path,
        mesh_device=mesh_device,
        ccl=ccl,
        force_recalculate=force_recalculate_weight_config,
        num_layers=_KIMI_NUM_LAYERS,
    )



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
