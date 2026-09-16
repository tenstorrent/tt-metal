# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Kimi-K2.7-Code prefill adapter.

Same architecture as DeepSeek-V3 (MLA + MoE), so it subclasses ``MLAPrefillAdapter``
and only overrides what differs: a single expert group with a device gate routes the
MoE routing all-gather's semaphores to L1_SMALL (so they don't pin the main-L1 floor
and clash with the next layer's MLA static CBs), which needs an L1_SMALL carve-out at
mesh-open time.
"""

from __future__ import annotations

from pathlib import Path

from models.demos.deepseek_v3_d_p.reference.kimi_k2_7_config import KimiK27Config
from models.demos.deepseek_v3_d_p.tt.runners.adapters.mla import MLAPrefillAdapter


class KimiK27Adapter(MLAPrefillAdapter):
    # --- identity & runner defaults ---
    name = "kimi_k2_7"
    model_config = KimiK27Config
    hf_model_default = "/mnt/models/moonshotai/Kimi-K2_7-Code-dequantized"
    ttnn_cache_default = "/mnt/models/moonshotai/Kimi-K2_7-Code-Cache/Kimi-K2_7-Code-Cache-prefill"
    default_gate_mode = "DEVICE_FP32"  # Kimi (1 expert group)
    prefill_trace_default = "/mnt/models/deepseek-prefill-cache/golden/structured_traces/vllm-kimi-k27-codedebug-56320"
    # Empty: https://github.com/tenstorrent/tt-metal/issues/54973
    mla_trace_defaults = ()

    # Single expert group + device gate: route routing-all-gather semaphores to L1_SMALL.
    # Routing consumes 512 B; leave 256 B for MLA high-bandwidth-gather semaphores.
    l1_small_size = 768
    routing_use_l1_small_for_semaphores = True

    # The Kimi-K2.x DFlash drafter checkpoint ($DFLASH_HF_MODEL) targets THIS architecture: its
    # num_target_layers=61 / hidden_size=7168 match, and it taps layer outputs (1, 12, 24, 35, 47, 58) of
    # it. No other model may enable PREFILL_DFLASH.
    supports_dflash = True

    # --- test metadata (HF download coordinates + PCC thresholds) ---
    hf_repo_id = "moonshotai/Kimi-K2.7-Code"
    env_var = "KIMI_K2_7_HF_MODEL"
    default_local_path = Path("/mnt/models/moonshotai/Kimi-K2_7-Code-dequantized")
    shared_path = None
    num_layers_to_download = 24
    ref_cache_env = "TT_KIMI_PREFILL_HOST_REF_CACHE"
    mla_ref_cache_env = "KIMI_MLA_REF_CACHE"
    ttnn_cache_env = "TT_KIMI_PREFILL_TTNN_CACHE"
    mla_pcc_threshold = 0.995
    moe_pcc_threshold = 0.971
    # vllm-traced golden: metadata.json + row-sharded kv_post_transform
    # (kv_cache/layer_N/rows_*.safetensors). resolve_trace_dir descends a run-hash subdir as needed.
    prefill_trace_layout = "chunked_group_a_v1"
    test_prefill_trace_default = (
        "/mnt/models/deepseek-prefill-cache/golden/structured_traces/vllm-kimi-k27-codedebug-56320"
    )

    @property
    def reference_model_cls(self):
        from models.demos.deepseek_v3_d_p.reference.kimi_k2_7.modeling_deepseek import DeepseekV3Model

        return DeepseekV3Model

    @property
    def reference_attention_cls(self):
        from models.demos.deepseek_v3_d_p.reference.kimi_k2_7.modeling_deepseek import DeepseekV3Attention

        return DeepseekV3Attention

    @property
    def reference_moe_cls(self):
        from models.demos.deepseek_v3_d_p.reference.kimi_k2_7.modeling_deepseek import DeepseekV3MoE

        return DeepseekV3MoE
