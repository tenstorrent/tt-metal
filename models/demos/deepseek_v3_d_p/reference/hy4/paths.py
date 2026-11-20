# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared Hy4-preview checkpoint and golden-trace locations.

The prefill adapter (#65918) should take ``HY4_PREFILL_TRACE_DEFAULT`` as its ``prefill_trace_default``
(``PREFILL_TRACE_DIR`` still overrides at run time), and the golden generator writes there by default.
Every location is overridable from the environment, so nothing here is pinned to one machine.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---- checkpoints ----------------------------------------------------------------------------------
HY4_HF_MODEL_ENV = "HY4_HF_MODEL"  # BF16 checkpoint dir (tencent/Hy4-preview)
HY4_FP8_MODEL_ENV = "HY4_FP8_HF_MODEL"  # official MXFP8 checkpoint dir (tencent/Hy4-preview-FP8)
HY4_HF_REPO_ID = "tencent/Hy4-preview"
HY4_FP8_HF_REPO_ID = "tencent/Hy4-preview-FP8"
# Searched in order after the env var. The first is where the BF16 release is staged on the bring-up
# hosts today; the weka paths follow the GLM-5.3 convention for CI machines.
HY4_HF_MODEL_CANDIDATES = (
    "/mnt/tt-data/Hy4-Preview",
    "/mnt/weka/model-weights/llm/tencent/Hy4-preview",
)
HY4_FP8_MODEL_CANDIDATES = (
    "/mnt/tt-data/Hy4-Preview-FP8",
    "/mnt/weka/model-weights/llm/tencent/Hy4-preview-FP8",
)

# ---- golden traces --------------------------------------------------------------------------------
HY4_GOLDEN_ROOT_ENV = "HY4_GOLDEN_ROOT"
HY4_GOLDEN_ROOT_DEFAULT = "/mnt/weka/model-cache/scratch/tencent/Hy4-preview-Cache/golden_traces"
# One trace per prompt length; names encode checkpoint precision, prompt source and length.
HY4_GOLDEN_TRACE_NAMES = {
    ("bf16", 4096): "hy4-bf16-longbench-4k",
    ("bf16", 32768): "hy4-bf16-longbench-32k",
    ("bf16", 131072): "hy4-bf16-longbench-128k",
    ("mxfp8", 4096): "hy4-mxfp8-longbench-4k",
    ("mxfp8", 32768): "hy4-mxfp8-longbench-32k",
    ("mxfp8", 131072): "hy4-mxfp8-longbench-128k",
}


def golden_root() -> Path:
    return Path(os.environ.get(HY4_GOLDEN_ROOT_ENV, HY4_GOLDEN_ROOT_DEFAULT))


def golden_trace_dir(precision: str = "bf16", seq_len: int = 32768) -> Path:
    name = HY4_GOLDEN_TRACE_NAMES.get((precision, seq_len), f"hy4-{precision}-{seq_len}")
    return golden_root() / name


# The trace the producer KV check uses by default: 32K is the first length the issue's "done" bar
# names, and it is long enough that the top-2,048 sparse selection actually drops positions.
HY4_PREFILL_TRACE_DEFAULT = str(golden_trace_dir("bf16", 32768))


def _first_existing(env: str, candidates) -> Path | None:
    value = os.environ.get(env)
    if value:
        return Path(value)
    for c in candidates:
        if (Path(c) / "config.json").exists():
            return Path(c)
    return None


def resolve_hf_model(precision: str = "bf16") -> Path:
    """The checkpoint dir for ``precision`` ("bf16" or "mxfp8"): env var first, then known locations."""
    env, candidates = (
        (HY4_HF_MODEL_ENV, HY4_HF_MODEL_CANDIDATES)
        if precision == "bf16"
        else (HY4_FP8_MODEL_ENV, HY4_FP8_MODEL_CANDIDATES)
    )
    path = _first_existing(env, candidates)
    if path is None:
        raise FileNotFoundError(f"no Hy4 {precision} checkpoint found: set {env} (searched {', '.join(candidates)})")
    return path
