# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Host-only unit tests for the GLM-5.3-Flash config / precision / weights surface.

No device: these pin the contract every device module reads, so a failure here is a cheap signal.
Ported from Blackhole-only assumptions where they were implicit; nothing here is arch-dependent.
"""

from __future__ import annotations

import json

import pytest

import ttnn
from models.experimental.glm53_flash.tt import precision
from models.experimental.glm53_flash.tt.config import (
    Glm5NextTextConfig,
    LayerSpec,
    round_up,
)
from models.experimental.glm53_flash.tt.indexer import check_dense_exact
from models.experimental.glm53_flash.tt.weights import cache_dtype

pytestmark = pytest.mark.use_module_device()


# ---------------------------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------------------------


def test_defaults_match_the_checkpoint():
    """The dataclass defaults are the real GLM-5.3-Flash dims, not placeholders."""
    c = Glm5NextTextConfig()
    assert c.vocab_size == 154880
    assert c.hidden_size == 4096
    assert c.num_hidden_layers == 45
    assert c.intermediate_size == 12288  # dense FFN
    assert c.moe_intermediate_size == 2048  # MoE FFN
    assert c.n_routed_experts == 288
    assert c.num_experts_per_tok == 8
    assert c.n_shared_experts == 1
    assert c.hc_mult == 4
    assert c.qk_rope_head_dim == 0  # NoPE
    assert c.qk_nope_head_dim == 256
    assert c.v_head_dim == 256
    assert c.kv_lora_rank == 512
    assert c.q_lora_rank == 1536
    assert c.swiglu_limit == 10.0
    assert c.index_topk == 2048
    assert c.index_kpool == 4


def test_layer_type_remap():
    """The legacy checkpoint spelling ``deepseek_sparse_attention`` becomes ``indexed_attention``."""
    c = Glm5NextTextConfig.__new__(Glm5NextTextConfig)
    assert "deepseek_sparse_attention" in {
        "deepseek_sparse_attention": 0,
    }
    # __post_init__ remaps both legacy spellings.
    cfg = Glm5NextTextConfig()
    cfg.layer_types = ["linear_attention", "deepseek_sparse_attention", "full_attention"]
    cfg.__post_init__()
    assert cfg.layer_types == ["linear_attention", "indexed_attention", "indexed_attention"]


def test_layer_kind_classification():
    """Layers 0-2 are kda_dense, 31 are kda_moe, 11 are mla_moe at 3, 7, 11, ... 43."""
    c = Glm5NextTextConfig()
    c.__post_init__()
    kinds = [c.layer_kind(i) for i in range(c.num_hidden_layers)]
    assert kinds[0] == "kda_dense"
    assert kinds[1] == "kda_dense"
    assert kinds[2] == "kda_dense"
    assert kinds[3] == "mla_moe"
    assert kinds[7] == "mla_moe"
    assert kinds[43] == "mla_moe"
    assert kinds[4] == "kda_moe"
    assert kinds.count("mla_moe") == 11
    assert kinds.count("kda_dense") == 3
    # 45 - 11 mla_moe - 3 kda_dense = 31
    assert kinds.count("kda_moe") == 31
    # 'mla_dense' is documented as not occurring in this checkpoint.
    assert kinds.count("mla_dense") == 0


def test_qk_head_dim_is_nope_total():
    """``qk_head_dim`` is the no-rope width only, since ``qk_rope_head_dim`` is 0."""
    c = Glm5NextTextConfig()
    assert c.qk_head_dim == c.qk_nope_head_dim + c.qk_rope_head_dim == 256


def test_num_local_experts_alias():
    c = Glm5NextTextConfig()
    assert c.num_local_experts == c.n_routed_experts == 288


def test_from_json_parses_text_config(tmp_path):
    """A checkpoint-shaped ``config.json`` (nested ``text_config``) loads, with overrides applied."""
    payload = {
        "text_config": {
            "vocab_size": 154880,
            "hidden_size": 4096,
            "num_hidden_layers": 45,
            "n_routed_experts": 288,
            "layer_types": ["linear_attention"] * 45,
        }
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(payload))
    cfg = Glm5NextTextConfig.from_json(str(p))
    assert cfg.hidden_size == 4096
    assert cfg.num_hidden_layers == 45
    cfg2 = Glm5NextTextConfig.from_json(str(p), num_hidden_layers=4)
    assert cfg2.num_hidden_layers == 4


def test_round_up():
    assert round_up(1, 32) == 32
    assert round_up(32, 32) == 32
    assert round_up(33, 32) == 64
    assert round_up(0, 32) == 0
    assert round_up(2048, 32) == 2048


def test_layer_spec_from_config():
    """``LayerSpec.from_config`` carries kind plus the per-kind widths the device modules read."""
    c = Glm5NextTextConfig()
    c.__post_init__()
    dense = LayerSpec.from_config(c, 0)
    assert dense.kind == "dense_kda"
    dsa = LayerSpec.from_config(c, 3)
    assert dsa.kind == "sparse_dsa"
    sparse = LayerSpec.from_config(c, 4)
    assert sparse.kind == "sparse_kda"


# ---------------------------------------------------------------------------------------------
# Precision / dtype map
# ---------------------------------------------------------------------------------------------


def test_weight_dtype_map_is_complete():
    """Every category the cache distinguishes is present, and the values are real ttnn dtypes."""
    assert set(precision.WEIGHT_DTYPES) == {"routed_expert", "projection_bfp8", "bf16", "fp32"}
    assert precision.WEIGHT_DTYPES["routed_expert"] == ttnn.bfloat4_b
    assert precision.WEIGHT_DTYPES["projection_bfp8"] == ttnn.bfloat8_b
    assert precision.WEIGHT_DTYPES["bf16"] == ttnn.bfloat16
    assert precision.WEIGHT_DTYPES["fp32"] == ttnn.float32


def test_activation_dtypes():
    """Residual streams and KDA state are fp32; sublayer inputs and the latent cache are bf16."""
    assert precision.STREAM_DTYPE == ttnn.float32
    assert precision.ACT_DTYPE == ttnn.bfloat16
    assert precision.LATENT_CACHE_DTYPE == ttnn.bfloat16
    assert precision.KDA_STATE_DTYPE == ttnn.float32


def test_cache_dtype_routing():
    """The suffix map routes experts to bf4, the projections to bf8, and A_log/dt_bias to fp32."""
    assert cache_dtype("mlp.experts.0.gate_proj") == ttnn.bfloat4_b
    assert cache_dtype("self_attn.q_proj.weight") == ttnn.bfloat8_b
    assert cache_dtype("self_attn.q_b_proj.weight") == ttnn.bfloat8_b
    assert cache_dtype("self_attn.kv_a_proj_with_mqa.weight") == ttnn.bfloat8_b
    assert cache_dtype("self_attn.A_log") == ttnn.float32
    assert cache_dtype("self_attn.dt_bias") == ttnn.float32
    assert cache_dtype("input_layernorm.weight") == ttnn.bfloat16
    # The fp32-strict set and the bf8 set must not overlap.
    from models.experimental.glm53_flash.tt.weights import _BFP8_SUFFIXES, _FP32_SUFFIXES

    assert not (set(_BFP8_SUFFIXES) & set(_FP32_SUFFIXES))


def test_compute_kernel_configs_build(device):
    """HiFi2/HiFi4 fp32-acc configs construct on this arch (Wormhole or Blackhole)."""
    for fn in (precision.hifi2_fp32, precision.hifi4_fp32):
        cfg = fn(device)
        assert cfg is not None
        assert cfg.fp32_dest_acc_en is True


def test_compute_kernel_configs_are_arch_typed(device):
    """The config object must be the arch's own type -- this is the guard that catches a
    Blackhole-authored config being handed to a Wormhole device."""
    cfg = precision.hifi4_fp32(device)
    arch = device.arch()
    expected = {
        ttnn.Arch.WORMHOLE_B0: "WormholeComputeKernelConfig",
        ttnn.Arch.BLACKHOLE: "BlackholeComputeKernelConfig",
    }[arch]
    assert type(cfg).__name__ == expected, f"{arch} produced {type(cfg).__name__}, expected {expected}"


# ---------------------------------------------------------------------------------------------
# Indexer gate
# ---------------------------------------------------------------------------------------------


def test_dense_exact_boundary(expect_error):
    """DSA dense attention is exact only where top-k selects every visible token (kv_len <= 2051)."""
    from models.experimental.glm53_flash.tt.config import DSA_DENSE_EXACT_MAX_KV
    from models.experimental.glm53_flash.tt.indexer import dense_selection_is_exact

    assert DSA_DENSE_EXACT_MAX_KV == 2051  # index_topk 2048 + (kpool-1) 3
    # Below the bound: no raise.
    for n in (1, 64, 2048, 2051):
        assert dense_selection_is_exact(n)
        check_dense_exact(3, n)
    # Above: must raise rather than silently computing dense attention.
    for n in (2052, 4096, 65536):
        assert not dense_selection_is_exact(n)
        with expect_error(NotImplementedError, "indexer top-k"):
            check_dense_exact(3, n)
