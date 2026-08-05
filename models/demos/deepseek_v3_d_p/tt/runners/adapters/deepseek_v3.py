# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""DeepSeek-V3 (DeepSeek-R1-0528) prefill adapter."""

from __future__ import annotations

from pathlib import Path

from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.runners.adapters.mla import MLAPrefillAdapter


class DeepSeekV3Adapter(MLAPrefillAdapter):
    # --- identity & runner defaults ---
    name = "deepseek_v3_d_p"
    model_config = DeepSeekV3Config
    hf_model_default = "models/demos/deepseek_v3/reference"
    ttnn_cache_default = "/mnt/models/DeepSeek-R1-0528-Cache/DeepSeek-R1-0528-Cache-prefill_secure"
    default_gate_mode = "DEVICE_FP32"
    prefill_trace_default = "/mnt/models/deepseek-prefill-cache/golden/longbook_qa_eng_prefill_56320_nopad"

    # --- test metadata (HF download coordinates + PCC thresholds) ---
    hf_repo_id = "deepseek-ai/DeepSeek-R1-0528"
    env_var = "DEEPSEEK_V3_HF_MODEL"
    default_local_path = Path("models/demos/deepseek_v3/reference")
    shared_path = Path("/proj_sw/user_dev/deepseek-ai/DeepSeek-R1-0528")
    num_layers_to_download = 24
    ref_cache_env = "TT_DS_PREFILL_HOST_REF_CACHE"
    mla_ref_cache_env = "DEEPSEEK_V3_MLA_REF_CACHE"
    ttnn_cache_env = "TT_DS_PREFILL_TTNN_CACHE"
    mla_pcc_threshold = 0.996
    moe_pcc_threshold = 0.982
    prefill_trace_layout = "single_file"
    # Stock fast tokenizer — no custom tokenizer code to import, so skip trust_remote_code (Kimi keeps it).
    tokenizer_trust_remote_code = False
    # No trust_remote_code, so config/tokenizer load fine from the HF snapshot dir — skip the flat copy
    # (which is wasteful and fails on a read-only HF cache mount). Kimi keeps it (inherits True).
    needs_flat_config_dir = False

    @property
    def reference_model_cls(self):
        from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3Model

        return DeepseekV3Model

    @property
    def reference_attention_cls(self):
        from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3Attention

        return DeepseekV3Attention

    @property
    def reference_moe_cls(self):
        from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MoE

        return DeepseekV3MoE
