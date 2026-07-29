# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Kimi-K2.6 prefill adapter.

Same architecture as DeepSeek-V3 (MLA + MoE), so it subclasses ``MLAPrefillAdapter``
and only overrides what differs: a single expert group with a device gate routes the
MoE routing all-gather's semaphores to L1_SMALL (so they don't pin the main-L1 floor
and clash with the next layer's MLA static CBs), which needs an L1_SMALL carve-out at
mesh-open time.
"""

from __future__ import annotations

from pathlib import Path

from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.tt.runners.adapters.mla import MLAPrefillAdapter


class KimiK26Adapter(MLAPrefillAdapter):
    # --- identity & runner defaults ---
    name = "kimi_k2_6"
    model_config = KimiK26Config
    # Repo-local config (dot-free, in-tree). The runner only needs config dims; real
    # weights come from the TTNN cache. To use a different checkpoint, set
    # PREFILL_HF_MODEL to a dot-free path (transformers' trust_remote_code import
    # chokes on the "." in /mnt/models/moonshotai/Kimi-K2.6-dequantized).
    hf_model_default = "models/demos/deepseek_v3_d_p/reference/kimi_k2_6"
    ttnn_cache_default = "/mnt/models/Kimi-K2_6-Cache/Kimi-K2_6-Cache-prefill"
    default_gate_mode = "DEVICE_FP32"  # Kimi (1 expert group)
    prefill_trace_default = "/mnt/models/deepseek-prefill-cache/golden/kimi-26/kimi_longbook_56320"

    # Single expert group + device gate: route routing-all-gather semaphores to L1_SMALL.
    l1_small_size = 512
    routing_use_l1_small_for_semaphores = True

    # --- test metadata (HF download coordinates + PCC thresholds) ---
    hf_repo_id = "moonshotai/Kimi-K2.6"
    env_var = "KIMI_K2_6_HF_MODEL"
    default_local_path = Path("models/demos/deepseek_v3_d_p/reference/kimi_k2_6")
    shared_path = None
    num_layers_to_download = 24
    ref_cache_env = "TT_KIMI_PREFILL_HOST_REF_CACHE"
    mla_ref_cache_env = "KIMI_MLA_REF_CACHE"
    ttnn_cache_env = "TT_KIMI_PREFILL_TTNN_CACHE"
    mla_pcc_threshold = 0.995
    moe_pcc_threshold = 0.971
    # vllm-traced golden: metadata.json + row-sharded kv_post_transform
    # (kv_cache/layer_N/rows_*.safetensors). The test golden differs from the
    # serving one above; resolve_trace_dir descends a run-hash subdir as needed.
    prefill_trace_layout = "chunked_group_a_v1"
    test_prefill_trace_default = "/mnt/models/deepseek-prefill-cache/golden/structured_traces/kimi_debug_55k_vllm"

    @property
    def reference_model_cls(self):
        from models.demos.deepseek_v3_d_p.reference.kimi_k2_6.modeling_deepseek import DeepseekV3Model

        return DeepseekV3Model

    @property
    def reference_attention_cls(self):
        from models.demos.deepseek_v3_d_p.reference.kimi_k2_6.modeling_deepseek import DeepseekV3Attention

        return DeepseekV3Attention

    @property
    def reference_moe_cls(self):
        from models.demos.deepseek_v3_d_p.reference.kimi_k2_6.modeling_deepseek import DeepseekV3MoE

        return DeepseekV3MoE
