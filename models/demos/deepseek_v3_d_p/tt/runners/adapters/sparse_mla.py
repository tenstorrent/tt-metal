# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Sparse-attention (DSA) prefill adapters: the shared ``SparseMLAPrefillAdapter`` base, DeepSeek-V3.2-Exp and Kimi-K3.1.

GLM-5.3 has its own serving adapter (``adapters/glm_5_3.py``);
what remains here is the test-only base + DeepSeek-V3.2-Exp.

Both add a lightning indexer (DeepSeek Sparse Attention) on top of the MLA + MoE
family. Two things set them apart from the dense ``DeepSeekV3Adapter`` / ``KimiK27Adapter``:

  * Their HF ``model_type`` (``deepseek_v32`` / ``glm_moe_dsa``) is NOT registered with
    transformers, so ``AutoConfig`` cannot load them — the config is HAND-BUILT via
    ``config_builder``. That builder is also the single source the sparse-MLA CPU reference
    derives its dims from, so device and truth stay in sync (see reference.cpu_deepseek_v32).
  * No prefill *serving* runtime is wired yet (P0 scope is sparse-MLA reference parity — see
    ``tests/sparse_mla/``). These adapters exist to feed the pytest ``variant`` fixture with
    config + metadata; ``allocate_kv_cache`` / ``build_runtime`` therefore raise. Wire those
    (and flip ``supports_pretrained``) when a sparse serving path lands.

Kept import-light like the rest of the adapters: the config builder is imported lazily
inside the ``config_builder`` property.
"""

from __future__ import annotations

from typing import Callable

from models.demos.common.prefill.adapter import PrefillRunParams
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config
from models.demos.deepseek_v3_d_p.tt.runners.adapters.mla import MLAPrefillAdapter


class SparseMLAPrefillAdapter(MLAPrefillAdapter):
    """Shared base for the DSA (sparse-attention) test variants.

    Serving is not wired: the HF config comes from a hand-built ``config_builder`` (AutoConfig
    can't load these model types) and the KV/runtime methods raise. Subclasses set the identity
    attributes and ``config_builder``.
    """

    supports_pretrained = False
    pretrained_mla_layer = None  # no reachable checkpoint, hence no MLA weights and no GPU trace
    # No golden prefill trace: these variants are exercised only by the sparse-MLA reference
    # tests, which build their own traces. Set explicitly so ``.prefill_trace_default`` is always
    # readable (the base declares it annotation-only).
    prefill_trace_default = None
    # Serving weight roots are unused (no runtime). Empty so an accidental read yields an obvious
    # empty path rather than an AttributeError.
    hf_model_default = ""
    ttnn_cache_default = ""

    def load_hf_config(self):
        """Return the hand-built HF-attribute config. ``AutoConfig`` (the base implementation)
        can't load these unregistered model types, so we go through ``config_builder`` — the same
        object conftest's ``config_only`` path resolves to."""
        return self.config_builder()

    def allocate_kv_cache(self, *, mesh_device, hf_config, params: PrefillRunParams):
        raise NotImplementedError(f"{self.name}: sparse-attention prefill serving not wired (test-only variant)")

    def build_runtime(self, *, mesh_device, hf_config, params: PrefillRunParams):
        raise NotImplementedError(f"{self.name}: sparse-attention prefill serving not wired (test-only variant)")


class DeepSeekV32Adapter(SparseMLAPrefillAdapter):
    # --- identity ---
    name = "deepseek_v32"
    # V3.2-Exp shares R1's MoE dims (256 experts, 3 dense layers); ``model_config`` is read only by
    # the full-transformer / prefill tests (which V3.2 doesn't run yet), so R1's dims class stands in.
    model_config = DeepSeekV3Config

    # --- test metadata ---
    hf_repo_id = "deepseek-ai/DeepSeek-V3.2-Exp"
    env_var = "DEEPSEEK_V32_HF_MODEL"
    mla_ref_cache_env = "DEEPSEEK_V32_MLA_REF_CACHE"
    mla_pcc_threshold = 0.996

    @property
    def config_builder(self) -> Callable:
        from models.demos.deepseek_v3_d_p.reference.deepseek_v3_2_config import deepseek_v32_hf_config

        return deepseek_v32_hf_config


class KimiK31Adapter(SparseMLAPrefillAdapter):
    """Kimi-K3.1 sparse-MLA (DSA) test variant: K3's gated NoPE MLA plus a NoPE lightning indexer.

    Test-only, like ``DeepSeekV32Adapter``: it feeds the sparse-MLA module tests. Serving the 93-layer
    KDA + DSA hybrid is out of scope here (``KimiK3Adapter`` owns the K3 runtime).
    """

    # --- identity ---
    name = "kimi_k3_1"

    model_config = KimiK31Config

    # --- test metadata ---
    hf_repo_id = "moonshotai/Kimi-K3.1-Pro"
    env_var = "KIMI_K31_HF_MODEL"
    mla_ref_cache_env = "KIMI_K31_MLA_REF_CACHE"
    # K3 is NoPE: the KV cache's second half carries no rotation (see KimiK3Adapter.kv_pe_interleave).
    kv_pe_interleave = False
    # None, deliberately: the only reader is conftest's pretrained_mla_layer_weights fixture, which would first
    # resolve the whole multi-TB checkpoint. The sparse-MLA tests load real weights through --ds-layer instead
    # (the first DSA layer is 3; layers 0-2 are KDA), which uses pretrained_attention_extract_only below.
    pretrained_mla_layer = None
    # The K3.1 shards are tens of GB each: fetch only the requested layer's attention tensors (range reads).
    pretrained_attention_extract_only = True

    @property
    def default_sparse_kv_cache_format(self):
        """Scaled-FP8 main KVPE cache (production format, as GLM-5.3); BF16 is a diagnostic only."""
        from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat

        return MlaKvCacheFormat.SCALED_FP8

    @property
    def config_builder(self) -> Callable:
        from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import kimi_k3_1_hf_config

        return kimi_k3_1_hf_config
