# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Sparse-attention (DSA) prefill adapters: the shared ``SparseMLAPrefillAdapter`` base + DeepSeek-V3.2-Exp.

GLM-5.1 / GLM-5.2 have their own serving adapters (``adapters/glm_5_1.py`` / ``adapters/glm_5_2.py``);
what remains here is the test-only base + DeepSeek-V3.2-Exp.

Both add a lightning indexer (DeepSeek Sparse Attention) on top of the MLA + MoE
family. Two things set them apart from the dense ``DeepSeekV3Adapter`` / ``KimiK26Adapter``:

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
from models.demos.deepseek_v3_d_p.tt.runners.adapters.mla import MLAPrefillAdapter


class SparseMLAPrefillAdapter(MLAPrefillAdapter):
    """Shared base for the DSA (sparse-attention) test variants.

    Serving is not wired: the HF config comes from a hand-built ``config_builder`` (AutoConfig
    can't load these model types) and the KV/runtime methods raise. Subclasses set the identity
    attributes and ``config_builder``.
    """

    supports_pretrained = False
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
