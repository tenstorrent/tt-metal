# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Kimi-K3.1 is a variant of the Kimi-K3 code path, not a copy of it.

``PREFILL_MODEL=kimi_k3_1`` must build K3's stack with K3.1's config, and must not pick up anything
recorded against the K3 checkpoint. Hardware-free: no mesh, no weights.
"""

from __future__ import annotations

import pytest

from models.demos.common.prefill.adapter import get_adapter
from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config
from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer
from models.demos.deepseek_v3_d_p.tt.mla.indexer import resolve_has_indexer
from models.demos.deepseek_v3_d_p.tt.runners.adapters.kimi_k3 import KimiK3Adapter

K3 = get_adapter("kimi_k3")
K31 = get_adapter("kimi_k3_1")

# K3-checkpoint artifacts: each would silently feed K3 weights, goldens or a K3 drafter into a K3.1 run.
_K3_ARTIFACTS = (
    "hf_model_default",
    "ttnn_cache_default",
    "prefill_trace_default",
    "test_prefill_trace_default",
    "kda_golden_default",
    "mla_trace_defaults",
    "dflash_model_default",
    "dflash_golden_default",
    "default_local_path",
    "hf_repo_id",
    "env_var",
    "ttnn_cache_env",
    "ref_cache_env",
    "mla_ref_cache_env",
)


def test_kimi_k3_1_runs_on_the_k3_code_path():
    assert isinstance(K31, KimiK3Adapter) and K31.name == "kimi_k3_1" and K31.model_config is KimiK31Config
    assert K31.transformer_cls is TtKimiK3Transformer
    # Inherited, not re-implemented: a K3 fix to any of these reaches K3.1.
    for method in ("build_runtime", "allocate_kv_cache", "layer_split_boundaries", "load_hf_config"):
        assert getattr(type(K31), method) is getattr(KimiK3Adapter, method), method


def test_only_kimi_k3_1_config_carries_the_indexer():
    k3, k31 = K3.load_hf_config(), K31.load_hf_config()
    assert not resolve_has_indexer(k3)
    assert resolve_has_indexer(k31)
    assert (k31.index_topk, k31.index_n_heads, k31.index_head_dim, k31.index_rope_dim) == (2048, 32, 128, 0)
    assert k31.num_hidden_layers == k3.num_hidden_layers == KimiK31Config.NUM_LAYERS


@pytest.mark.parametrize("attr", _K3_ARTIFACTS)
def test_kimi_k3_1_inherits_no_k3_artifact(attr):
    assert getattr(K31, attr) != getattr(K3, attr)


def test_kimi_k3_1_flags():
    assert not K31.supports_dflash  # Kimi-K3-DSpark targets K3
    assert not K31.passes_indexer_indices  # every DSA layer runs its own indexer
    assert K31.reference_attention_cls is None  # no K3.1 CPU reference; K3's MLA has no indexer


def test_kimi_k3_1_indexer_k_norm_uses_its_rms_norm_eps():
    """vLLM's KimiK3DSAIndexerKNorm takes config.rms_norm_eps (1e-5); DS-V3.2 / GLM keep the reference's 1e-6.
    The CPU reference reads the same config field the device indexer does."""
    from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32.reference import _model_args_from_hf
    from models.demos.deepseek_v3_d_p.reference.deepseek_v3_2_config import deepseek_v32_hf_config
    from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import glm_5_3_hf_config
    from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import kimi_k3_1_hf_config

    k31 = kimi_k3_1_hf_config()
    assert k31.index_k_norm_eps == KimiK31Config.RMS_NORM_EPS == 1e-5
    assert _model_args_from_hf(k31, 5120).index_k_norm_eps == 1e-5
    for config in (deepseek_v32_hf_config(), glm_5_3_hf_config()):
        assert not hasattr(config, "index_k_norm_eps")
        assert _model_args_from_hf(config, 5120).index_k_norm_eps == 1e-6
