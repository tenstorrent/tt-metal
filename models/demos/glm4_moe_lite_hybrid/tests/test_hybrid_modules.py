# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Test suite for the hybrid GLM-4.7-Flash implementation.

Tests are split into:
- Framework tests (no TTNN required): module lifecycle, replacement, config
- Integration tests (TTNN required): attention, MoE, cache, runner

Framework tests run anywhere. Integration tests require TT_ENABLE_HW_TESTS=1.
"""

from __future__ import annotations

import os

import pytest
import torch

from models.demos.glm4_moe_lite_hybrid.core.config import Glm4MoeLiteHParams
from models.demos.glm4_moe_lite_hybrid.core.module import TTNNModule
from models.demos.glm4_moe_lite_hybrid.core.module_replacement import register_module_replacement

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_hparams() -> Glm4MoeLiteHParams:
    return Glm4MoeLiteHParams(
        vocab_size=151552,
        hidden_size=2048,
        intermediate_size=10240,
        num_hidden_layers=47,
        num_attention_heads=32,
        num_key_value_heads=1,
        q_lora_rank=768,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_head_dim=256,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        partial_rotary_factor=1.0,
        rope_interleave=True,
        moe_intermediate_size=1408,
        n_routed_experts=64,
        n_shared_experts=1,
        num_experts_per_tok=4,
        first_k_dense_replace=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.8,
        n_group=1,
        topk_group=1,
        topk_method="noaux_tc",
    )


# ---------------------------------------------------------------------------
# Framework tests (no TTNN dependency)
# ---------------------------------------------------------------------------


class TestTTNNModule:
    def test_basic_lifecycle(self):
        module = TTNNModule()
        assert not module._preprocessed_weight
        assert not module._weights_on_device
        assert module.device is None
        assert module.module_name is not None

    def test_from_torch(self):
        torch_linear = torch.nn.Linear(10, 20)
        module = TTNNModule.from_torch(torch_linear)
        assert module.torch_layer is torch_linear
        assert isinstance(module, TTNNModule)

    def test_model_config(self):
        module = TTNNModule()
        module.set_model_config({"key": "value"})
        assert module.model_config["key"] == "value"

    def test_none_config(self):
        module = TTNNModule()
        module.set_model_config(None)
        assert module.model_config == {}

    def test_named_children(self):
        parent = TTNNModule()
        child1 = TTNNModule()
        child2 = TTNNModule()
        parent.child1 = child1
        parent.child2 = child2
        children = dict(parent.named_children())
        assert "child1" in children
        assert "child2" in children

    def test_named_children_skips_fallback(self):
        parent = TTNNModule()
        parent._fallback_torch_layer = torch.nn.Linear(10, 20)
        children = dict(parent.named_children())
        assert "_fallback_torch_layer" not in children

    def test_named_modules_dedup(self):
        parent = TTNNModule()
        child = TTNNModule()
        parent.a = child
        parent.b = child
        modules = list(parent.named_modules())
        module_objs = [m for _, m in modules]
        assert module_objs.count(child) == 1

    def test_to_device(self):
        module = TTNNModule()
        fake_device = object()
        result = module.to_device(fake_device)
        assert module.device is fake_device
        assert result is module

    def test_set_device_recursive(self):
        parent = TTNNModule()
        child = TTNNModule()
        parent.child = child
        fake_device = object()
        parent.set_device_recursive(fake_device)
        assert parent.device is fake_device
        assert child.device is fake_device

    def test_preprocess_weights_idempotent(self):
        call_count = 0

        class CountingModule(TTNNModule):
            def preprocess_weights_impl(self):
                nonlocal call_count
                call_count += 1

        module = CountingModule()
        module.preprocess_weights()
        module.preprocess_weights()
        assert call_count == 1

    def test_child_weight_propagation(self):
        parent = TTNNModule()
        child = TTNNModule()
        parent.child = child
        parent.preprocess_weights()
        assert child._preprocessed_weight

    def test_repr(self):
        parent = TTNNModule()
        parent.child = TTNNModule()
        r = repr(parent)
        assert "TTNNModule" in r
        assert "child" in r

    def test_forward_raises(self):
        module = TTNNModule()
        with pytest.raises(NotImplementedError):
            module.forward()


class TestModuleReplacement:
    def test_basic_replacement(self):
        class CustomLinear(TTNNModule):
            @classmethod
            def from_torch(cls, torch_layer):
                inst = cls()
                inst._fallback_torch_layer = torch_layer
                inst.in_features = torch_layer.in_features
                inst.out_features = torch_layer.out_features
                return inst

            def forward(self, x):
                return x

        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.Linear(20, 30),
        )
        result = register_module_replacement(model, {torch.nn.Linear: CustomLinear})
        assert len(result) == 2
        for _, module in result.items():
            assert isinstance(module, CustomLinear)
        assert isinstance(model[0], CustomLinear)
        assert isinstance(model[1], CustomLinear)

    def test_exclude_replacement(self):
        class CustomLinear(TTNNModule):
            @classmethod
            def from_torch(cls, torch_layer):
                inst = cls()
                inst._fallback_torch_layer = torch_layer
                return inst

            def forward(self, x):
                return x

        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.Linear(20, 30),
        )
        result = register_module_replacement(
            model,
            {torch.nn.Linear: CustomLinear},
            exclude={"0"},
        )
        assert len(result) == 1
        assert isinstance(model[0], torch.nn.Linear)
        assert isinstance(model[1], CustomLinear)

    def test_nested_replacement(self):
        class CustomLinear(TTNNModule):
            @classmethod
            def from_torch(cls, torch_layer):
                inst = cls()
                inst._fallback_torch_layer = torch_layer
                return inst

            def forward(self, x):
                return x

        inner = torch.nn.Sequential(torch.nn.Linear(10, 20))
        model = torch.nn.Sequential(inner, torch.nn.Linear(20, 30))
        result = register_module_replacement(model, {torch.nn.Linear: CustomLinear})
        assert len(result) == 2

    def test_empty_map(self):
        model = torch.nn.Sequential(torch.nn.Linear(10, 20))
        result = register_module_replacement(model, {})
        assert len(result) == 0

    def test_model_config_propagation(self):
        class CustomLinear(TTNNModule):
            @classmethod
            def from_torch(cls, torch_layer):
                inst = cls()
                inst._fallback_torch_layer = torch_layer
                return inst

            def forward(self, x):
                return x

        model = torch.nn.Sequential(torch.nn.Linear(10, 20))
        config = {"test_key": 42}
        result = register_module_replacement(model, {torch.nn.Linear: CustomLinear}, model_config=config)
        for _, module in result.items():
            assert module.model_config["test_key"] == 42


class TestConfig:
    def test_hparams_validate(self, default_hparams):
        default_hparams.validate()

    def test_hparams_derived(self, default_hparams):
        assert default_hparams.qk_head_dim == (default_hparams.qk_nope_head_dim + default_hparams.qk_rope_head_dim)

    def test_kvpe_dim(self, default_hparams):
        kvpe_dim = default_hparams.kv_lora_rank + default_hparams.qk_rope_head_dim
        assert kvpe_dim == 576

    def test_invalid_hparams(self):
        with pytest.raises(AssertionError):
            Glm4MoeLiteHParams(
                vocab_size=0,
                hidden_size=2048,
                intermediate_size=10240,
                num_hidden_layers=47,
                num_attention_heads=32,
                num_key_value_heads=1,
                q_lora_rank=768,
                kv_lora_rank=512,
                qk_nope_head_dim=192,
                qk_rope_head_dim=64,
                v_head_dim=128,
                qk_head_dim=256,
                rms_norm_eps=1e-6,
                rope_theta=10000.0,
                partial_rotary_factor=1.0,
                rope_interleave=True,
                moe_intermediate_size=1408,
                n_routed_experts=64,
                n_shared_experts=1,
                num_experts_per_tok=4,
                first_k_dense_replace=1,
                norm_topk_prob=True,
                routed_scaling_factor=1.8,
                n_group=1,
                topk_group=1,
                topk_method="noaux_tc",
            ).validate()

    def test_moe_params(self, default_hparams):
        assert default_hparams.n_routed_experts == 64
        assert default_hparams.num_experts_per_tok == 4
        assert default_hparams.n_shared_experts == 1
        assert default_hparams.routed_scaling_factor == 1.8
        assert default_hparams.first_k_dense_replace == 1


# ---------------------------------------------------------------------------
# Integration tests (require TTNN)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_HW_TESTS", "") != "1",
    reason="Hardware tests disabled (set TT_ENABLE_HW_TESTS=1)",
)
class TestTTNNIntegration:
    """Tests requiring the TTNN SDK and a TT device."""

    def test_kvpe_cache_on_device(self, default_hparams):
        import ttnn
        from models.demos.glm4_moe_lite_hybrid.modules.kvpe_cache import CompressedKVPECache, CompressedKVPECacheConfig

        device = ttnn.open_device(0)
        try:
            config = CompressedKVPECacheConfig(
                num_layers=1,
                max_num_blocks=4,
                block_size=64,
            )
            cache = CompressedKVPECache(default_hparams, config)
            cache.to_device(device, batch_size=1)
            assert cache.get_cache(0) is not None
            assert cache.page_table is not None
        finally:
            ttnn.close_device(device)

    def test_attention_module_creation(self):
        from models.demos.glm4_moe_lite_hybrid.modules.attention import HybridGlm4MLA

        mla = HybridGlm4MLA()
        assert isinstance(mla, TTNNModule)

    def test_moe_module_creation(self):
        from models.demos.glm4_moe_lite_hybrid.modules.moe import (
            HybridGlm4MoEExperts,
            HybridGlm4MoEMLP,
            HybridGlm4MoERouter,
            HybridGlm4MoERuntimeManager,
        )

        router = HybridGlm4MoERouter()
        experts = HybridGlm4MoEExperts()
        mlp = HybridGlm4MoEMLP()
        mgr = HybridGlm4MoERuntimeManager()
        assert isinstance(router, TTNNModule)
        assert isinstance(experts, TTNNModule)
        assert isinstance(mlp, TTNNModule)
        assert mgr.runtime is None

    @pytest.mark.skipif(
        os.environ.get("TT_ENABLE_LARGE_MODEL_TESTS", "") != "1",
        reason="Large model tests disabled",
    )
    def test_layer0_decode_pcc(self, default_hparams):
        """Layer 0 decode PCC >= 0.99 vs CPU reference."""
        pytest.skip("Requires model snapshot")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
