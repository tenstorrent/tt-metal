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
    # Not inherited from Kimi-K2.6: different checkpoint, and K2.7's vllm capture has no mla_io/
    # streams. BLOCKED ON an MLA trace for K2.7: test_mla.py's reference='trace' axis skips while
    # this is empty (K2.6 has two, under mla_sdpa_traces/). load_trace needs a dir holding
    # mla_io/mla_{input,output}_layer_<N>.safetensors for exactly one layer plus
    # kv_cache/layer_<N>.safetensors. Kimi-K3 registers a plain structured_traces/ dir that happens
    # to include mla_io/, so the existing capture flow can emit this -- K2.7's trace simply was not
    # captured with those streams enabled. Once staged, list it here and drop the skip in test_mla.py.
    mla_trace_defaults = ()
