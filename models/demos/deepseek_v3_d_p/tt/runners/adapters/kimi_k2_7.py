# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Kimi-K2.7 prefill adapter.

Architecturally identical to Kimi-K2.6 (same MLA + MoE dims, same reference model and
device knobs) — only the checkpoint differs. So it subclasses ``KimiK26Adapter`` and
overrides just the identity and the default cache/trace paths.
"""

from __future__ import annotations

from pathlib import Path

from models.demos.deepseek_v3_d_p.tt.runners.adapters.kimi_k2_6 import KimiK26Adapter


class KimiK27Adapter(KimiK26Adapter):
    # --- identity & runner defaults ---
    name = "kimi_k2_7"
    ttnn_cache_default = "/mnt/models/moonshotai/Kimi-K2_7-Code-Cache/Kimi-K2_7-Code-Cache-prefill"
    prefill_trace_default = "/mnt/models/deepseek-prefill-cache/golden/structured_traces/vllm-kimi-k27-codedebug-56320"

    # --- test metadata (HF download coordinates) ---
    env_var = "KIMI_K2_7_HF_MODEL"
    default_local_path = Path("/mnt/models/moonshotai/Kimi-K2_7-Code-dequantized")
    test_prefill_trace_default = (
        "/mnt/models/deepseek-prefill-cache/golden/structured_traces/vllm-kimi-k27-codedebug-56320"
    )
