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
896 experts / top-16, a latent RMSNorm, and one shared expert at 6144. Every FFN site runs the
checkpoint's **SiTU-GLU** on device: the routed experts through the fused kernel (#51351), the
shared expert and the layer-0 dense FFN through ttnn-level softcap/sigmoid/multiply (#53625), which
is correct but not yet tuned at their 6144 / 33792 widths. One deliberate limit remains:
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
    # The download fallback fetches layers 0..N-1, and pretrained_mla_layer is 3.
    num_layers_to_download = 4
    ref_cache_env = "TT_KIMI_K3_PREFILL_HOST_REF_CACHE"
    mla_ref_cache_env = "KIMI_K3_MLA_REF_CACHE"
    ttnn_cache_env = "TT_KIMI_K3_PREFILL_TTNN_CACHE"
    # Loading the staged checkpoint wholesale needs an MXFP4 -> bf16 dequantizer that does not exist
    # yet, so the full-transformer fixtures stay skipped. The MoE gate is exempt: it is unquantized and read
    # through a prefix-filtered safe_open.
    supports_pretrained = False
    # MLA alone is loadable: quantization_config.ignore covers self_attn, so those weights are bf16.
    # The first full-attention layer, not 0 -- layers 0-2 are KDA and hold no MLA tensors.
    pretrained_mla_layer = KimiK3Config.mla_layer_ids()[0]
    mla_trace_defaults = ("/mnt/models/deepseek-prefill-cache/golden/structured_traces/kimi_k3_100k_vllm",)
    # Left as None: shared_path feeds conftest's state_dict fixture, which pytest would resolve --
    # loading all 1.5 TB -- before the supports_pretrained skip in the fixture body runs.
    shared_path = None

    # Device vs upstream KimiSparseMoeBlock. Held at test_kimi_k3_moe's final_output_pcc: the two
    # compare the same device tensor against references that agree to 2e-5, so they measure the same
    # thing and cannot carry different bars. The 8x4 value is 0.9696 (reference 0.969567,
    # final_output 0.969585) -- 8x4 spreads 896 experts over 32 chips instead of 8, so the top-16
    # combine accumulates across 4x as many chips in the bf8 latent space than on the 2x4 proxy
    # (0.995693). 0.965 keeps the usual ~0.005 of margin.
    #
    # Moving the routed experts onto SiTU-GLU did not cost anything here: 8x4 went 0.969434 ->
    # 0.969567 and 2x4 0.995692 -> 0.995693. K3's realistic activations sit well inside both tanh
    # caps, so the bf4 accuracy cost that _K3_SATURATION_CASES documents does not arise at these
    # scales.
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

        The block applies one activation to routed and shared experts alike, which matches the
        device: ``run_reference_moe`` builds it with hidden_act="situ" for both halves.
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
