# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Kimi-K3 MLA + MoE adapter (test-only).

Kimi-K3 is a **hybrid**: 93 layers, of which only 24 are full-attention (MLA) layers and 69 are KDA
linear-attention layers. No KDA module exists in this package, so K3 cannot yet be served end to
end; a serving adapter would have to subclass ``PrefillModelAdapter`` directly and build its own
runtime with a hybrid layer schedule.

This adapter therefore exists to give the MLA and MoE layers a first-class ``variant`` fixture: the
test suite's ``TEST_VARIANTS`` registers it locally (the same test-only pattern GLM-5.2 uses) and it
is deliberately **absent** from ``models/demos/common/prefill/adapter.py:ADAPTER_PATHS``, so nothing
can select it with ``PREFILL_MODEL=kimi_k3`` and get a half-built model.

It subclasses ``MLAPrefillAdapter`` for the config/cache/reference plumbing only. ``build_runtime``
and ``allocate_kv_cache`` are inherited but would size the KV cache to ``params.num_layers``, which
is wrong for 24-of-93; they are overridden to fail loudly rather than mislead.

MoE scope (issue #51336): the latent-MoE structure -- routed experts at the reduced 3584 hidden,
896 experts / top-16, a latent RMSNorm, and one shared expert at 6144. Two deliberate limits:
  * the device path runs **SiLU**, not the checkpoint's SiTU-GLU, until the kernel in #51335 lands;
  * only the **gate** uses real checkpoint weights. Experts, shared expert and the latent
    projections use seeded random weights, because everything routed is MXFP4 and no dequantizer
    exists yet. Device PCC is therefore TT-vs-torch on identical seeded weights.
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
    num_layers_to_download = 1
    ref_cache_env = "TT_KIMI_K3_PREFILL_HOST_REF_CACHE"
    mla_ref_cache_env = "KIMI_K3_MLA_REF_CACHE"
    ttnn_cache_env = "TT_KIMI_K3_PREFILL_TTNN_CACHE"
    # Loading the staged checkpoint wholesale needs an MXFP4 -> bf16 dequantizer that does not exist
    # yet, so the pretrained fixtures stay skipped. The MoE gate is exempt: it is unquantized and read
    # through a prefix-filtered safe_open.
    supports_pretrained = False
    # Left as None: shared_path feeds conftest's state_dict fixture, which pytest would resolve --
    # loading all 1.5 TB -- before the supports_pretrained skip in the fixture body runs.
    shared_path = None

    # Device vs upstream KimiSparseMoeBlock. Held at test_kimi_k3_moe's final_output_pcc: the two
    # compare the same device tensor against references that agree to 1.7e-5, so they measure the
    # same thing and cannot carry different bars. The old 0.99 was measured on 2x4 (0.995692) and
    # does not transfer -- 8x4 spreads 896 experts over 32 chips instead of 8, so the top-16 combine
    # accumulates across 4x as many chips in the bf8 latent space, and both checks land together at
    # 0.9694 (reference 0.969434, final_output 0.969454). 0.965 keeps the usual ~0.005 of margin.
    moe_pcc_threshold = 0.965

    @property
    def config_builder(self) -> Callable:
        """Hand-built config: upstream ``modeling_kimi_linear.py`` raises ImportError without
        ``fla-core``, so ``AutoConfig`` cannot load ``model_type: kimi_linear`` here."""
        return kimi_k3_hf_config

    @property
    def reference_attention_cls(self):
        from models.demos.deepseek_v3_d_p.reference.kimi_k3.modeling_kimi_k3_mla import KimiMLAAttention

        return KimiMLAAttention

    @property
    def reference_moe_cls(self):
        """K3's MoE block: DeepSeek-V3's, wrapped in the shared low-rank latent projection pair.

        Note this reference computes **SiTU-GLU**, which no TT kernel implements yet (#51335). It is
        therefore the truth model for the host-side latent-structure test, not for device PCC -- the
        device runs SiLU, so device comparisons must use a SiLU-configured reference on both sides.
        """
        from models.demos.deepseek_v3_d_p.reference.kimi_k3.modeling_kimi_moe import KimiSparseMoeBlock

        return KimiSparseMoeBlock

    def allocate_kv_cache(self, **kwargs):
        raise NotImplementedError(
            "Kimi-K3 KV-cache allocation needs the 24-of-93 model-layer -> kv-slot map; the "
            "inherited MLA allocator would size the cache to the full 93-layer count. Use "
            "KimiK3Config.mla_kv_slot() when wiring this up."
        )

    def build_runtime(self, **kwargs):
        raise NotImplementedError(
            "Kimi-K3 has no prefill runtime: 69 of its 93 layers are KDA linear-attention layers "
            "with no TT implementation. This adapter is MLA-test-only."
        )
