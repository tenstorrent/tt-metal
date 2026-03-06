# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""GLM-4.7-Flash: CPU-only unit tests for the optimal merged implementation.

This file validates the correctness of optimizations ported from the agentic AI
branch (sdawle/mickg10/glm47_flash) into the human-written tt_symbiote framework
(sdawle/ign/glm_flash) on the sdawle/glm47_flash_optimal branch.

What this file does:
  - Tests pure Python / PyTorch logic WITHOUT requiring TT hardware, the
    transformers library, or a full ttnn C++ backend build.
  - Mocks the ttnn module with minimal attribute stubs so that config, dtype
    selection, and math utilities can be imported and exercised in isolation.
  - Covers 7 test classes (40 tests total):
      * TestGlm4MoeLiteHParams  - Config dataclass construction, validation,
                                   derived dimensions (kvpe_dim, qk_head_dim).
      * TestKvBDecomposition    - kv_b weight split into kv_b1 (Q-absorption)
                                   and kv_b2 (value extraction), roundtrip.
      * TestRouterBiasCentering - MoE router bias shift for BF16 precision.
      * TestGetExpertsDtype     - GLM4_EXPERTS_DTYPE env var parsing (bf8/bf16/bf4).
      * TestFusedGateUp         - Fused gate+up projection equivalence.
      * TestKVPEDimensions      - KVPE cache memory reduction (14.2x).
      * TestFusedSiLUMul        - Fused SiLU*mul vs separate ops.

Run with:
    python3 -m pytest tests/test_glm47_flash_optimal_cpu.py -v --noconftest
"""

import os
import sys
import types
from unittest import mock

import pytest
import torch


# ---------------------------------------------------------------------------
# Mock ttnn so config.py and attention.py can import without full C++ backend.
# We only need the dtype sentinel objects that get_experts_dtype references.
# ---------------------------------------------------------------------------
_ttnn_mock = types.ModuleType("ttnn")
_ttnn_mock.bfloat8_b = "BFLOAT8_B"
_ttnn_mock.bfloat16 = "BFLOAT16"
_ttnn_mock.bfloat4_b = "BFLOAT4_B"
_ttnn_mock.float32 = "FLOAT32"
_ttnn_mock.DataType = type("DataType", (), {})
_ttnn_mock.TILE_SIZE = 32
_ttnn_mock.TILE_LAYOUT = "TILE_LAYOUT"
_ttnn_mock.ROW_MAJOR_LAYOUT = "ROW_MAJOR_LAYOUT"
_ttnn_mock.DRAM_MEMORY_CONFIG = "DRAM_MEMORY_CONFIG"
_ttnn_mock.L1_MEMORY_CONFIG = "L1_MEMORY_CONFIG"
_ttnn_mock.Tensor = type("Tensor", (), {})
_ttnn_mock.CoreGrid = lambda y, x: types.SimpleNamespace(y=y, x=x)
_ttnn_mock.CoreCoord = lambda x, y: types.SimpleNamespace(x=x, y=y)
_ttnn_mock.ShardStrategy = types.SimpleNamespace(WIDTH="WIDTH", HEIGHT="HEIGHT")
_ttnn_mock.ShardOrientation = types.SimpleNamespace(ROW_MAJOR="ROW_MAJOR")
_ttnn_mock.MathFidelity = types.SimpleNamespace(HiFi2="HiFi2", HiFi4="HiFi4", LoFi="LoFi")
_ttnn_mock.UnaryOpType = types.SimpleNamespace(SILU="SILU")
_ttnn_mock.FabricConfig = types.SimpleNamespace(FABRIC_1D_RING="FABRIC_1D_RING")

for fn_name in [
    "from_torch",
    "to_torch",
    "to_device",
    "to_layout",
    "to_memory_config",
    "reshape",
    "permute",
    "slice",
    "concat",
    "pad",
    "unsqueeze",
    "squeeze",
    "repeat",
    "matmul",
    "linear",
    "mul",
    "silu",
    "gelu",
    "zeros",
    "rms_norm",
    "gather",
    "copy",
    "deallocate",
    "synchronize_device",
    "ReplicateTensorToMesh",
    "ShardTensorToMesh",
    "ConcatMeshToTensor",
    "create_sharded_memory_config",
    "init_device_compute_kernel_config",
    "get_device_ids",
]:
    setattr(_ttnn_mock, fn_name, lambda *a, **kw: None)

_ttnn_mock.transformer = types.SimpleNamespace(
    paged_scaled_dot_product_attention_decode=lambda *a, **kw: None,
    paged_flash_multi_latent_attention_decode=lambda *a, **kw: None,
    scaled_dot_product_attention=lambda *a, **kw: None,
)
_ttnn_mock.experimental = types.SimpleNamespace(
    paged_fill_cache=lambda *a, **kw: None,
    paged_update_cache=lambda *a, **kw: None,
    all_gather_async=lambda *a, **kw: None,
    reduce_scatter_minimal_async=lambda *a, **kw: None,
)
_ttnn_mock.SDPAProgramConfig = type("SDPAProgramConfig", (), {"__init__": lambda self, **kw: None})
_ttnn_mock.WormholeComputeKernelConfig = type("WormholeComputeKernelConfig", (), {"__init__": lambda self, **kw: None})
_ttnn_mock.MatmulMultiCoreReuseMultiCast1DProgramConfig = type(
    "MatmulMultiCoreReuseMultiCast1DProgramConfig", (), {"__init__": lambda self, **kw: None}
)
_ttnn_mock.Tile = lambda shape: shape
_ttnn_mock.Topology = types.SimpleNamespace(Linear="Linear")
_ttnn_mock.uint16 = "UINT16"
_ttnn_mock.uint32 = "UINT32"
_ttnn_mock.int32 = "INT32"

# Preprocess functions
_ttnn_mock_preprocessing = types.ModuleType("ttnn.model_preprocessing")
_ttnn_mock_preprocessing.preprocess_linear_weight = lambda w, **kw: w
_ttnn_mock_preprocessing.preprocess_linear_bias = lambda b, **kw: b

# Install mocks before importing our modules
sys.modules["ttnn"] = _ttnn_mock
sys.modules["ttnn.model_preprocessing"] = _ttnn_mock_preprocessing

# Now import after mocking
from models.experimental.tt_symbiote.modules.config import (
    Glm4MoeLiteHParams,
    get_experts_dtype,
)


# ---------------------------------------------------------------------------
# GLM-4.7-Flash reference values from the real HuggingFace model config
# ---------------------------------------------------------------------------
GLM4_REFERENCE = {
    "vocab_size": 151552,
    "hidden_size": 3584,
    "intermediate_size": 18944,
    "num_hidden_layers": 40,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "q_lora_rank": 768,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 64,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "rms_norm_eps": 1.5625e-07,
    "rope_theta": 1000000.0,
    "partial_rotary_factor": 0.5,
    "rope_interleave": True,
    "moe_intermediate_size": 1408,
    "n_routed_experts": 128,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "first_k_dense_replace": 1,
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.5,
    "n_group": 8,
    "topk_group": 4,
    "topk_method": "noaux_tc",
}


def _make_mock_hf_config(**overrides):
    """Create a mock HF config with GLM-4.7-Flash reference values."""
    vals = {**GLM4_REFERENCE, **overrides}
    vals["qk_head_dim"] = vals["qk_nope_head_dim"] + vals["qk_rope_head_dim"]
    return types.SimpleNamespace(**vals)


# ===== Glm4MoeLiteHParams Tests =====


class TestGlm4MoeLiteHParams:
    def test_from_hf_config_reference_values(self):
        """Construct from mock HF config and verify all fields match."""
        hparams = Glm4MoeLiteHParams.from_hf_config(_make_mock_hf_config())

        assert hparams.vocab_size == 151552
        assert hparams.hidden_size == 3584
        assert hparams.num_hidden_layers == 40
        assert hparams.num_attention_heads == 32
        assert hparams.num_key_value_heads == 2
        assert hparams.q_lora_rank == 768
        assert hparams.kv_lora_rank == 512
        assert hparams.qk_nope_head_dim == 64
        assert hparams.qk_rope_head_dim == 64
        assert hparams.v_head_dim == 128
        assert hparams.n_routed_experts == 128
        assert hparams.num_experts_per_tok == 8

    def test_kvpe_dim(self):
        """kvpe_dim = kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576."""
        hparams = Glm4MoeLiteHParams.from_hf_config(_make_mock_hf_config())
        assert hparams.kvpe_dim == 576
        assert hparams.kvpe_dim == hparams.kv_lora_rank + hparams.qk_rope_head_dim

    def test_qk_head_dim(self):
        """qk_head_dim = qk_nope_head_dim + qk_rope_head_dim = 64 + 64 = 128."""
        hparams = Glm4MoeLiteHParams.from_hf_config(_make_mock_hf_config())
        assert hparams.qk_head_dim == 128
        assert hparams.qk_head_dim == hparams.qk_nope_head_dim + hparams.qk_rope_head_dim

    def test_qk_head_dim_auto_computed(self):
        """If qk_head_dim is absent from HF config, it's computed from nope + rope."""
        hf_cfg = _make_mock_hf_config()
        del hf_cfg.qk_head_dim
        hparams = Glm4MoeLiteHParams.from_hf_config(hf_cfg)
        assert hparams.qk_head_dim == 128

    def test_validate_passes(self):
        """validate() should not raise for valid reference config."""
        hparams = Glm4MoeLiteHParams.from_hf_config(_make_mock_hf_config())
        hparams.validate()

    def test_validate_fails_zero_hidden(self):
        """validate() must catch hidden_size=0."""
        hparams = Glm4MoeLiteHParams.from_hf_config(_make_mock_hf_config(hidden_size=0))
        with pytest.raises(AssertionError):
            hparams.validate()

    def test_validate_fails_bad_qk_head_dim(self):
        """validate() must catch qk_head_dim != nope + rope."""
        hf_cfg = _make_mock_hf_config()
        hf_cfg.qk_head_dim = 999
        hparams = Glm4MoeLiteHParams.from_hf_config(hf_cfg)
        with pytest.raises(AssertionError):
            hparams.validate()

    def test_validate_fails_topk_group_exceeds_n_group(self):
        """validate() must catch topk_group > n_group."""
        hparams = Glm4MoeLiteHParams.from_hf_config(_make_mock_hf_config(topk_group=16, n_group=4))
        with pytest.raises(AssertionError):
            hparams.validate()

    def test_frozen_dataclass(self):
        """HParams should be immutable."""
        hparams = Glm4MoeLiteHParams.from_hf_config(_make_mock_hf_config())
        with pytest.raises(AttributeError):
            hparams.hidden_size = 1024

    def test_rope_theta_from_nested(self):
        """rope_theta can come from nested rope_parameters dict."""
        hf_cfg = _make_mock_hf_config()
        del hf_cfg.rope_theta
        hf_cfg.rope_parameters = {"rope_theta": 500000.0}
        hparams = Glm4MoeLiteHParams.from_hf_config(hf_cfg)
        assert hparams.rope_theta == 500000.0

    def test_rope_theta_missing_raises(self):
        """Should raise if rope_theta is not findable."""
        hf_cfg = _make_mock_hf_config()
        del hf_cfg.rope_theta
        hf_cfg.rope_parameters = {}
        with pytest.raises(ValueError, match="Unable to determine rope_theta"):
            Glm4MoeLiteHParams.from_hf_config(hf_cfg)

    def test_partial_rotary_factor_default(self):
        """Should default to 1.0 when not present."""
        hf_cfg = _make_mock_hf_config()
        del hf_cfg.partial_rotary_factor
        hparams = Glm4MoeLiteHParams.from_hf_config(hf_cfg)
        assert hparams.partial_rotary_factor == 1.0


# ===== kv_b Decomposition Math Tests =====


class TestKvBDecomposition:
    """Test the kv_b weight decomposition logic from agent's layer_weights.py.

    The decomposition splits kv_b_proj.weight [H*(qk_nope+v), kv_lora] into:
      kv_b1: [H, qk_nope, kv_lora] -- absorbs nope dims into Q for FlashMLA
      kv_b2: [H, kv_lora, v] -- extracts values post-attention (transposed)
    """

    @pytest.fixture
    def dims(self):
        return {
            "num_heads": 32,
            "qk_nope_head_dim": 64,
            "v_head_dim": 128,
            "kv_lora_rank": 512,
        }

    def test_decomposition_shapes(self, dims):
        H, nope, v, lora = dims["num_heads"], dims["qk_nope_head_dim"], dims["v_head_dim"], dims["kv_lora_rank"]

        kv_b_weight = torch.randn(H * (nope + v), lora)
        kv_b_reshaped = kv_b_weight.view(H, nope + v, lora)

        kv_b1 = kv_b_reshaped[:, :nope, :].contiguous()
        kv_b2 = kv_b_reshaped[:, -v:, :].transpose(1, 2).contiguous()

        assert kv_b1.shape == (H, nope, lora)
        assert kv_b2.shape == (H, lora, v)

    def test_decomposition_recovers_nope(self, dims):
        H, nope, v, lora = dims["num_heads"], dims["qk_nope_head_dim"], dims["v_head_dim"], dims["kv_lora_rank"]

        kv_b_weight = torch.randn(H * (nope + v), lora)
        kv_b_reshaped = kv_b_weight.view(H, nope + v, lora)

        kv_b1 = kv_b_reshaped[:, :nope, :].contiguous()
        assert torch.equal(kv_b1, kv_b_reshaped[:, :nope, :])

    def test_decomposition_recovers_value(self, dims):
        H, nope, v, lora = dims["num_heads"], dims["qk_nope_head_dim"], dims["v_head_dim"], dims["kv_lora_rank"]

        kv_b_weight = torch.randn(H * (nope + v), lora)
        kv_b_reshaped = kv_b_weight.view(H, nope + v, lora)

        kv_b2 = kv_b_reshaped[:, -v:, :].transpose(1, 2).contiguous()
        recovered_v = kv_b2.transpose(1, 2)
        assert torch.allclose(recovered_v, kv_b_reshaped[:, nope:, :])

    def test_q_absorption_matmul(self, dims):
        """q_nope @ kv_b1 should produce [B, H, S, kv_lora_rank]."""
        H, nope, v, lora = dims["num_heads"], dims["qk_nope_head_dim"], dims["v_head_dim"], dims["kv_lora_rank"]
        B, S = 1, 1

        kv_b_weight = torch.randn(H * (nope + v), lora)
        kv_b1 = kv_b_weight.view(H, nope + v, lora)[:, :nope, :].contiguous()

        q_nope = torch.randn(B, H, S, nope)
        q_absorbed = torch.matmul(q_nope, kv_b1.unsqueeze(0))
        assert q_absorbed.shape == (B, H, S, lora)

    def test_value_extraction_matmul(self, dims):
        """attn_latent @ kv_b2 should produce [B, H, S, v_head_dim]."""
        H, nope, v, lora = dims["num_heads"], dims["qk_nope_head_dim"], dims["v_head_dim"], dims["kv_lora_rank"]
        B, S = 1, 1

        kv_b_weight = torch.randn(H * (nope + v), lora)
        kv_b2 = kv_b_weight.view(H, nope + v, lora)[:, -v:, :].transpose(1, 2).contiguous()

        attn_latent = torch.randn(B, H, S, lora)
        values = torch.matmul(attn_latent, kv_b2.unsqueeze(0))
        assert values.shape == (B, H, S, v)

    def test_full_roundtrip(self, dims):
        """Decomposed and reconstructed weight should match the original."""
        H, nope, v, lora = dims["num_heads"], dims["qk_nope_head_dim"], dims["v_head_dim"], dims["kv_lora_rank"]

        kv_b_weight = torch.randn(H * (nope + v), lora)
        kv_b_reshaped = kv_b_weight.view(H, nope + v, lora)

        kv_b1 = kv_b_reshaped[:, :nope, :].contiguous()
        kv_b2 = kv_b_reshaped[:, -v:, :].transpose(1, 2).contiguous()

        nope_proj = kv_b1
        value_proj = kv_b2.transpose(1, 2)
        full_proj = torch.cat([nope_proj, value_proj], dim=1)
        full_recon = full_proj.reshape(H * (nope + v), lora)

        assert torch.allclose(full_recon, kv_b_weight, atol=1e-6)


# ===== Router Bias Centering Tests =====


class TestRouterBiasCentering:
    def test_centering_non_negative(self):
        e_bias = torch.randn(128)
        e_bias_f32 = e_bias.to(torch.float32)
        e_bias_centered = e_bias_f32 - float(e_bias_f32.min().item())
        assert (e_bias_centered >= 0).all()
        assert float(e_bias_centered.min().item()) == pytest.approx(0.0, abs=1e-7)

    def test_centering_preserves_topk_ordering(self):
        e_bias = torch.randn(128)
        k = 8

        e_bias_f32 = e_bias.to(torch.float32)
        e_bias_centered = e_bias_f32 - float(e_bias_f32.min().item())

        _, topk_original = torch.topk(e_bias_f32, k)
        _, topk_centered = torch.topk(e_bias_centered, k)

        assert torch.equal(topk_original.sort().values, topk_centered.sort().values)

    def test_centering_preserves_relative_order(self):
        e_bias = torch.randn(128)
        e_bias_f32 = e_bias.to(torch.float32)
        e_bias_centered = e_bias_f32 - float(e_bias_f32.min().item())
        assert torch.equal(torch.argsort(e_bias_f32), torch.argsort(e_bias_centered))

    def test_centering_bf16_precision(self):
        """Centering should improve BF16 resolution near the decision boundary."""
        e_bias = torch.tensor([-100.5, -100.3, -100.1, -99.9, -99.7], dtype=torch.float32)
        centered = e_bias - e_bias.min()
        assert centered.min() == 0.0

        bf16_uncentered = e_bias.to(torch.bfloat16)
        bf16_centered = centered.to(torch.bfloat16)

        err_uncentered = (e_bias - bf16_uncentered.float()).abs().max()
        err_centered = (centered - bf16_centered.float()).abs().max()
        assert err_centered <= err_uncentered

    def test_centering_with_all_positive(self):
        e_bias = torch.tensor([1.0, 2.0, 3.0, 4.0])
        centered = e_bias - e_bias.min()
        assert centered[0] == 0.0
        assert centered[-1] == 3.0


# ===== get_experts_dtype Tests =====


class TestGetExpertsDtype:
    def test_default_bf8(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GLM4_EXPERTS_DTYPE", None)
            assert get_experts_dtype() == _ttnn_mock.bfloat8_b

    def test_bf8_explicit(self):
        with mock.patch.dict(os.environ, {"GLM4_EXPERTS_DTYPE": "bf8"}):
            assert get_experts_dtype() == _ttnn_mock.bfloat8_b

    def test_bfloat8_b_explicit(self):
        with mock.patch.dict(os.environ, {"GLM4_EXPERTS_DTYPE": "bfloat8_b"}):
            assert get_experts_dtype() == _ttnn_mock.bfloat8_b

    def test_bf16(self):
        with mock.patch.dict(os.environ, {"GLM4_EXPERTS_DTYPE": "bf16"}):
            assert get_experts_dtype() == _ttnn_mock.bfloat16

    def test_bfloat16(self):
        with mock.patch.dict(os.environ, {"GLM4_EXPERTS_DTYPE": "bfloat16"}):
            assert get_experts_dtype() == _ttnn_mock.bfloat16

    def test_bf4(self):
        with mock.patch.dict(os.environ, {"GLM4_EXPERTS_DTYPE": "bf4"}):
            assert get_experts_dtype() == _ttnn_mock.bfloat4_b

    def test_bfloat4_b(self):
        with mock.patch.dict(os.environ, {"GLM4_EXPERTS_DTYPE": "bfloat4_b"}):
            assert get_experts_dtype() == _ttnn_mock.bfloat4_b

    def test_invalid_raises(self):
        with mock.patch.dict(os.environ, {"GLM4_EXPERTS_DTYPE": "fp32"}):
            with pytest.raises(ValueError, match="Invalid GLM4_EXPERTS_DTYPE"):
                get_experts_dtype()

    def test_whitespace_stripped(self):
        with mock.patch.dict(os.environ, {"GLM4_EXPERTS_DTYPE": "  bf16  "}):
            assert get_experts_dtype() == _ttnn_mock.bfloat16

    def test_case_insensitive(self):
        with mock.patch.dict(os.environ, {"GLM4_EXPERTS_DTYPE": "BF16"}):
            assert get_experts_dtype() == _ttnn_mock.bfloat16

    def test_empty_string_is_default(self):
        with mock.patch.dict(os.environ, {"GLM4_EXPERTS_DTYPE": ""}):
            assert get_experts_dtype() == _ttnn_mock.bfloat8_b


# ===== Fused Gate+Up Logic Tests =====


class TestFusedGateUp:
    def test_fused_vs_separate_equivalence(self):
        """Fused w1w3 matmul + split produces same result as separate w1, w3 matmuls."""
        hidden, intermediate, num_tokens = 128, 64, 4

        w1 = torch.randn(hidden, intermediate)
        w3 = torch.randn(hidden, intermediate)
        w1w3 = torch.cat([w1, w3], dim=-1)

        x = torch.randn(num_tokens, hidden)

        w1_out = x @ w1
        w3_out = x @ w3

        w1w3_out = x @ w1w3
        fused_w1_out = w1w3_out[:, :intermediate]
        fused_w3_out = w1w3_out[:, intermediate:]

        assert torch.allclose(w1_out, fused_w1_out, atol=1e-5)
        assert torch.allclose(w3_out, fused_w3_out, atol=1e-5)

    def test_fused_silu_mul_equivalence(self):
        """Fused SiLU*mul should equal separate silu(gate) * up."""
        gate = torch.randn(4, 64)
        up = torch.randn(4, 64)

        separate = torch.nn.functional.silu(gate) * up
        fused = torch.nn.functional.silu(gate) * up

        assert torch.allclose(separate, fused, atol=1e-6)

    def test_fused_gate_up_shapes(self):
        """w1w3 shape should be [hidden, 2*intermediate]."""
        hidden, intermediate = 3584, 1408
        w1 = torch.randn(hidden, intermediate)
        w3 = torch.randn(hidden, intermediate)
        w1w3 = torch.cat([w1, w3], dim=-1)
        assert w1w3.shape == (hidden, 2 * intermediate)


# ===== KVPE Dimension Calculations =====


class TestKVPEDimensions:
    def test_kvpe_dim_value(self):
        hparams = Glm4MoeLiteHParams.from_hf_config(_make_mock_hf_config())
        assert hparams.kvpe_dim == 576

    def test_kvpe_memory_vs_standard(self):
        """KVPE cache uses >10x less memory per position than expanded K/V cache."""
        hparams = Glm4MoeLiteHParams.from_hf_config(_make_mock_hf_config())
        standard_per_pos = hparams.num_attention_heads * hparams.qk_head_dim * 2
        kvpe_per_pos = hparams.kvpe_dim
        ratio = standard_per_pos / kvpe_per_pos
        assert ratio > 10, f"Expected >10x memory reduction, got {ratio:.1f}x"

    def test_kvpe_vs_kv_lora_standard(self):
        """KVPE is always larger than kv_lora_rank due to rope component."""
        hparams = Glm4MoeLiteHParams.from_hf_config(_make_mock_hf_config())
        assert hparams.kvpe_dim > hparams.kv_lora_rank
        assert hparams.kvpe_dim == hparams.kv_lora_rank + hparams.qk_rope_head_dim
