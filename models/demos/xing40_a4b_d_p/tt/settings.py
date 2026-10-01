# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Every switch of xing40_a4b_d_p in one table (core/model_settings.py). Environment variable = ``XING_<NAME>`` unless
the entry is in ENV_NAMES (harness variables that predate the table keep their names). Nothing else in tt/ or
bringup/hooks.py reads the environment."""

from __future__ import annotations

import os

from models.demos.common.bringup.core.model_settings import Setting, Settings

TABLE = {
    # ---- attention (tt/attention.py)
    "KV_CACHE_DTYPE": Setting(
        "bfp8",
        ("bfp8", "bf16"),
        "owner 2026-10-01: bfp8_b TILE MLA latent cache (DeepSeek decode format), spec serving.kv_dtype; every ladder "
        "rung requires it. bf16 = the K.1 cache, for comparison",
    ),
    "MLA_K_CHUNK": Setting(256, None, "ring_mla k chunk (tokens); k512 overflows L1 with a bf16 cache (2.15 MB)"),
    "MLA_Q_CHUNK": Setting(
        0,
        None,
        "ring_mla q chunk; 0 = by fidelity: 64 at HiFi2, 32 at HiFi4 (P.1, chunk 5120 after 51200, per layer: HiFi4 "
        "q32/k256 25.9 ms; HiFi3 q32 20.6 ms (q64 23.6); HiFi2 q64/k256 17.9 ms (q32 18.7). q128/k256, q256/k128, "
        "q32/k384 overflow L1 at any fidelity)",
    ),
    "MLA_SDPA_FIDELITY": Setting(
        "HiFi2",
        ("HiFi2", "HiFi4"),
        "owner exception to rule 7 for the ring_mla matmuls only (P.1, 2026-10-01): HiFi2 fails the frozen "
        "C.moe.attention component test (L02 median row norm -0.0068 vs 0.004) but the owner accepts it on end-to-end "
        "accuracy (rung last / s56320 within 1e-3 of HiFi4, top5 1.0; supervision.md P.1)",
    ),
    "MLA_SDPA": Setting(
        "fork",
        ("fork", "source"),
        "fork = ttnn.bringup.ring_mla with fp32 DEST (latent-V streaming path); source = ttnn.transformer.ring_mla at "
        "bf16 DEST (owner 06:35 setting; its bf16 QK^T accumulation fails the component test's x2-input check)",
    ),
    "MLA_EXP_APPROX": Setting(
        False,
        None,
        "online-softmax correction exp: True = fp32-accurate exp, False = range-reduced polynomial (as before)",
    ),
    # ---- mHC / residual / norm / dense matmuls
    "HC_IMPL": Setting("fused", ("fused", "composed"), "mHC coefficients / collapse after the all_reduce (tt/mhc.py)"),
    "RESIDUAL_MIX": Setting("fused", ("fused", "addcmul", "matmul"), "mHC residual mix path (tt/residual.py)"),
    "MATMUL_FIDELITY": Setting(
        "HiFi4",
        ("HiFi4", "HiFi2"),
        "owner rule 7: HiFi4 for every matmul / norm / mHC compute config (q_a, norm, mhc, mlp, residual, MLA "
        "projections); the SDPA and the routed experts have their own switches",
    ),
    # ---- routed experts (tt/experts.py)
    "EXPERTS_MODE": Setting(
        "unified", ("unified", "loop"), "unified = one unified_routed_expert_ffn program; loop = per-expert ops"
    ),
    "EXPERTS_FIDELITY": Setting(
        "hifi2",
        ("hifi2", "hifi4"),
        "owner decision 2026-10-01 after P.3's A/B: experts 256.6 -> 242.4 ms, s56320 top5 1.0, final hidden 0.99803 "
        "with the bfp8 KV cache; hifi4 = the previous path",
    ),
    # ---- harness variables (names kept, see ENV_NAMES)
    "HYBRID": Setting(
        False, None, "debug: hybrid harness (CPU reference + DEVICE_STEPS on device, host in/out per step)"
    ),
    "BRINGUP_SPEC_SET": Setting(
        False, None, "BRINGUP_SPEC is set: use it when it is this model's (runners/adapter.py)"
    ),
    "HF_MODEL": Setting("", None, "checkpoint dir; empty = the bring-up spec's paths.hf"),
    "LAYERS": Setting("", None, "layers this rank serves, parse_layers syntax; empty = all"),
    "TTNN_CACHE": Setting(
        "@default", None, "weight cache root; @default = the adapter's ttnn_cache_default, empty = no cache"
    ),
}

ENV_NAMES = {
    "HYBRID": "BRINGUP_HYBRID",
    "BRINGUP_SPEC_SET": "BRINGUP_SPEC",
    "HF_MODEL": "PREFILL_HF_MODEL",
    "LAYERS": "PREFILL_XING_LAYERS",
    "TTNN_CACHE": "PREFILL_TTNN_CACHE",
}


class _XingSettings(Settings):
    def env_name(self, name: str) -> str:
        return ENV_NAMES.get(name, super().env_name(name))

    def get(self, name: str):
        if name == "BRINGUP_SPEC_SET":
            return bool(os.environ.get("BRINGUP_SPEC"))
        return super().get(name)


# Component and swap tests run at these values (framework rule); the shipped defaults above are judged end to end.
MAX_PRECISION = {
    "KV_CACHE_DTYPE": "bf16",
    "MLA_SDPA_FIDELITY": "HiFi4",
    "MATMUL_FIDELITY": "HiFi4",
    "EXPERTS_FIDELITY": "hifi4",
}

settings = _XingSettings("XING_", TABLE, max_precision=MAX_PRECISION)
