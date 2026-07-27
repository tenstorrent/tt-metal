# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Validated GLM-4.7-Flash performance defaults, shared by every entry point.

Single source of truth for the flag set that produces the published numbers
(51.3 ms/token bs=1 @ ISL=128 on a 32-chip WH Galaxy). Both
`scripts/debug_run_full_tt_greedy.py` and the vLLM entry point in
`generator_vllm.py` apply these, so a served model and a benchmark run
land on the same configuration.

Everything is applied with `os.environ.setdefault`, so an explicit value in the
environment always wins -- which is what makes A/B bisecting work
(`GLM4_MOE_LITE_SHARDED_NORM=0 python ...` still disables the sharded norm).

Must be called BEFORE `Glm4RuntimeConfig.from_env()` and before any weight
conversion, since the dtype knobs are read at weight-load time.
"""

from __future__ import annotations

import os

# Flags whose *code* default is off or conservative, but which are part of the
# validated winning configuration. These are the ones that genuinely change
# behaviour when this module is applied.
WINNING_DEFAULTS: dict[str, str] = {
    # --- Op fusion ---
    "GLM4_MOE_LITE_FUSE_QKV_A": "1",
    "GLM4_MOE_LITE_FUSE_SHARED_GATE_UP": "1",
    "GLM4_MOE_LITE_FUSE_MLP_MOE_REDUCE": "1",
    "GLM4_MOE_LITE_FUSED_ROUTER": "1",
    # --- Memory placement ---
    "GLM4_MOE_LITE_DECODE_L1_ACT": "1",
    "GLM4_MOE_LITE_EP_L1": "1",
    # --- Per-op overhead elimination ---
    "GLM4_MOE_LITE_SKIP_DEFENSIVE_CLONES": "1",
    "GLM4_MOE_LITE_SKIP_TYPECAST": "1",
    # --- Weight precision ---
    # Code default for dense weights is bf16; bf8 is a measured ~7% bs=1 decode
    # win with coherence verified (Rayleigh / Canberra / 17x24=408 prompts).
    "GLM4_MOE_LITE_DENSE_TT_DTYPE": "bf8",
    # --- Collectives (code defaults are 1 link / linear topology) ---
    "GLM4_MOE_LITE_CCL_NUM_LINKS": "4",
    "GLM4_MOE_LITE_CCL_TOPOLOGY": "ring",
    # --- Prefill ---
    "GLM4_MOE_LITE_BATCHED_PREFILL": "1",
}

# Already on in code. Restated so that dumping the environment shows the full
# effective configuration in one place, and so a future code-default change
# cannot silently move the benchmarked configuration out from under us.
CODE_DEFAULT_ON: dict[str, str] = {
    "GLM4_MOE_LITE_FUSED_COLLECTIVE_EPILOGUE": "1",
    "GLM4_MOE_LITE_BUFFERED_MOE_ALL_REDUCE": "1",
    "GLM4_MOE_LITE_FUSE_DOWN_ROUTING_SCALE": "1",
    "GLM4_MOE_LITE_SHARDED_NORM": "1",
    "GLM4_MOE_LITE_EXPLICIT_PROG_CFG": "1",
    "GLM4_MOE_LITE_ROUTER_L1": "1",
    "GLM4_MOE_LITE_NORM_L1": "1",
    "GLM4_MOE_LITE_EXPERTS_TT_DTYPE": "bf8",
}

# Gated experiments that build and run but must NOT be enabled: FUSED_KV_BRANCH is
# numerically incorrect, LMHEAD_SHARD is a measured bs=1 regression, TRACE_2CQ is
# neutral, and TP has a known accuracy regression. Pinned off so an inherited
# environment (docker, CI, a stale shell) cannot quietly turn them on in a
# production serving path.
PINNED_OFF: tuple[str, ...] = (
    "GLM4_MOE_LITE_FUSED_KV_BRANCH",
    "GLM4_MOE_LITE_LMHEAD_SHARD",
    "GLM4_MOE_LITE_TRACE_2CQ",
    "GLM4_MOE_LITE_TP",
)


def apply_perf_defaults(*, enable_moe: bool = False, pin_off_experiments: bool = False) -> dict[str, str]:
    """Apply the validated flag set via `setdefault`; return what this call changed.

    Args:
        enable_moe: also default `GLM4_MOE_LITE_ENABLE_MOE=1`. GLM-4.7-Flash is a
            MoE model and the routed experts are skipped entirely without it, so
            serving paths want this on.
        pin_off_experiments: also default the known-bad/neutral experiments in
            `PINNED_OFF` to "0". Recommended for serving; sweeps set them
            explicitly themselves.

    Returns:
        The subset of variables this call actually set (i.e. those not already
        present in the environment), for logging.
    """
    pending = {**WINNING_DEFAULTS, **CODE_DEFAULT_ON}
    if enable_moe:
        pending["GLM4_MOE_LITE_ENABLE_MOE"] = "1"
    if pin_off_experiments:
        pending.update({name: "0" for name in PINNED_OFF})

    applied: dict[str, str] = {}
    for key, value in pending.items():
        if key not in os.environ:
            applied[key] = value
        os.environ.setdefault(key, value)
    return applied


def overridden_defaults() -> dict[str, str]:
    """Return `{var: intended_default}` for defaults the environment disagrees with.

    Useful for logging: a deployment that exports `GLM4_MOE_LITE_DENSE_TT_DTYPE=bf16`
    will not hit the published numbers, and that should be visible in the log rather
    than discovered by a confusing benchmark result.
    """
    intended = {**WINNING_DEFAULTS, **CODE_DEFAULT_ON}
    return {key: value for key, value in intended.items() if os.environ.get(key, value).strip() != value}
