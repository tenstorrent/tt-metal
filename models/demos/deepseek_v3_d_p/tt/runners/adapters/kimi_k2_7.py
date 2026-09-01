# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Kimi-K2.7 prefill adapter.

Architecturally identical to Kimi-K2.6 (same MLA + MoE dims, same reference model and
device knobs) — only the checkpoint differs. So it subclasses ``KimiK26Adapter`` and
overrides just the identity and the default cache/trace paths.
"""

from __future__ import annotations

from pathlib import Path

from models.demos.deepseek_v3_d_p.reference.kimi_k2_7_config import KimiK27Config
from models.demos.deepseek_v3_d_p.tt.runners.adapters.kimi_k2_6 import KimiK26Adapter


class KimiK27Adapter(KimiK26Adapter):
    # --- identity & runner defaults ---
    name = "kimi_k2_7"
    # Not inherited: KimiK26Adapter sets model_config = KimiK26Config, so without this every
    # variant=kimi_k2_7 test would read K2.7 constants in its decorators and K2.6 constants at
    # runtime through variant.model_config. The values are identical today -- this is about
    # having one place to change on the day they are not.
    model_config = KimiK27Config
    # Not inherited: KimiK26Adapter points hf_model_default at the in-tree reference/kimi_k2_6 dir, so a
    # PREFILL_MODEL=kimi_k2_7 run with no PREFILL_HF_MODEL would silently read K2.6's config. This dir
    # is dot-free (trust_remote_code chokes on "." in a path) and ships the auto_map modules.
    hf_model_default = "/mnt/models/moonshotai/Kimi-K2_7-Code-dequantized"
    ttnn_cache_default = "/mnt/models/moonshotai/Kimi-K2_7-Code-Cache/Kimi-K2_7-Code-Cache-prefill"
    prefill_trace_default = "/mnt/models/deepseek-prefill-cache/golden/structured_traces/vllm-kimi-k27-codedebug-56320"

    # --- test metadata (HF download coordinates) ---
    # Not inherited: K2.6's repo id would make the conftest fallback silently download K2.6
    # weights and cache them under kimi_k2_7_bh_32dev/ when neither $KIMI_K2_7_HF_MODEL nor
    # default_local_path is present -- a wrong checkpoint that looks right in the logs.
    hf_repo_id = "moonshotai/Kimi-K2.7-Code"
    env_var = "KIMI_K2_7_HF_MODEL"
    default_local_path = Path("/mnt/models/moonshotai/Kimi-K2_7-Code-dequantized")
    test_prefill_trace_default = (
        "/mnt/models/deepseek-prefill-cache/golden/structured_traces/vllm-kimi-k27-codedebug-56320"
    )
    # Empty: https://github.com/tenstorrent/tt-metal/issues/54973
    mla_trace_defaults = ()
