# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Kimi-K3 MLA adapter (test-only).

Kimi-K3 is a **hybrid**: 93 layers, of which only 24 are full-attention (MLA) layers and 69 are KDA
linear-attention layers. No KDA module exists in this package, so K3 cannot yet be served end to
end — per ``docs/KIMI_K3_MLA.md`` finding F6, a serving adapter would subclass
``PrefillModelAdapter`` directly and build its own runtime with a hybrid layer schedule.

This adapter therefore exists to give the MLA layer a first-class ``variant`` fixture: the test
suite's ``TEST_VARIANTS`` registers it locally (the same test-only pattern GLM-5.2 uses) and it is
deliberately **absent** from ``models/demos/common/prefill/adapter.py:ADAPTER_PATHS``, so nothing
can select it with ``PREFILL_MODEL=kimi_k3`` and get a half-built model.

It subclasses ``MLAPrefillAdapter`` for the config/cache/reference plumbing only. ``build_runtime``
and ``allocate_kv_cache`` are inherited but would size the KV cache to ``params.num_layers``, which
is wrong for 24-of-93 (F5); they are overridden to fail loudly rather than mislead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config
from models.demos.deepseek_v3_d_p.tt.runners.adapters.mla import MLAPrefillAdapter


class KimiK3Adapter(MLAPrefillAdapter):
    # --- identity ---
    name = "kimi_k3"
    model_config = KimiK3Config
    # Repo-local, dot-free (transformers' trust_remote_code import chokes on "." in a path). The dir
    # holds the TRIMMED upstream MLA reference, not a loadable full-model checkpoint.
    hf_model_default = "models/demos/deepseek_v3_d_p/reference/kimi_k3"
    default_gate_mode = "DEVICE_FP32"  # single expert group, as Kimi-K2.6

    # Single expert group + device gate: route routing-all-gather semaphores to L1_SMALL. Inherited
    # from the Kimi family; inert for the MLA-only tests but correct if a runtime is ever built.
    l1_small_size = 512
    routing_use_l1_small_for_semaphores = True

    # --- test metadata ---
    hf_repo_id = "moonshotai/Kimi-K3"
    env_var = "KIMI_K3_HF_MODEL"
    default_local_path = Path("models/demos/deepseek_v3_d_p/reference/kimi_k3")
    shared_path = None
    num_layers_to_download = 1
    ref_cache_env = "TT_KIMI_K3_PREFILL_HOST_REF_CACHE"
    mla_ref_cache_env = "KIMI_K3_MLA_REF_CACHE"
    ttnn_cache_env = "TT_KIMI_K3_PREFILL_TTNN_CACHE"
    # No K3 checkpoint is staged anywhere reachable, and the MXFP4 weights would need a dequant path
    # for the MoE side anyway (MLA itself is exempt: quantization_config.ignore covers self_attn).
    # False makes the pretrained fixtures SKIP rather than fail.
    supports_pretrained = False
    # Carried over from Kimi-K2.6 as a starting point; re-derive once measured.
    mla_pcc_threshold = 0.995

    @property
    def config_builder(self) -> Callable:
        """Hand-built config: upstream ``modeling_kimi_linear.py`` raises ImportError without
        ``fla-core``, so ``AutoConfig`` cannot load ``model_type: kimi_linear`` here."""
        return kimi_k3_hf_config

    @property
    def reference_attention_cls(self):
        from models.demos.deepseek_v3_d_p.reference.kimi_k3.modeling_kimi_k3_mla import KimiMLAAttention

        return KimiMLAAttention

    def allocate_kv_cache(self, **kwargs):
        raise NotImplementedError(
            "Kimi-K3 KV-cache allocation needs the 24-of-93 model-layer -> kv-slot map (F5 in "
            "docs/KIMI_K3_MLA.md); the inherited MLA allocator would size the cache to the full "
            "layer count. Use KimiK3Config.mla_kv_slot() when wiring this up."
        )

    def build_runtime(self, **kwargs):
        raise NotImplementedError(
            "Kimi-K3 has no prefill runtime: 69 of its 93 layers are KDA linear-attention layers "
            "with no TT implementation (F6 in docs/KIMI_K3_MLA.md). This adapter is MLA-test-only."
        )
