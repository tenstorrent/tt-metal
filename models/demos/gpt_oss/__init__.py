# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""gpt-oss-20b runs with its release settings (the accuracy fixes) by default.

The model code reads its accuracy and speed switches from GPT_OSS_* environment variables, most of them when a module
is imported, and each switch is off when its variable is unset. This package's __init__ runs before any of those
modules is imported, so it fills in every variable that is not set already with the value that passes the GPQA eval
on QuietBox 2 (the tt-inference-server P300X2 entry), for gpt-oss-20b only.

- A variable that is already set is never changed. To turn one switch off, set it yourself (for example
  GPT_OSS_DECODE_SPLIT_NATIVE_FOLD=0, GPT_OSS_BOUNDARY_FIDELITY=HiFi2, or GPT_OSS_DENSE_ROUNDED_DECODE= ).
- GPT_OSS_RELEASE_DEFAULTS=0 turns all of them off (the old behavior: every switch off unless set).
- gpt-oss-120b gets none of them: the per-layer ranks and the rounded weights below are gpt-oss-20b's.
- The decode switches act only on the fused decode path (1x4 Blackhole mesh); elsewhere they are read and unused.
- The rounded weights are Hugging Face Hub locations at pinned revisions (tt/weight_source.py), downloaded once into
  the Hub cache. The experts use the K-GPTQ rounding for decode and prefill (GPT_OSS_PREFILL_KGPTQ), so the
  output-grouped package of GPT_OSS_DECODE_BF4_ROUNDED / GPT_OSS_PREFILL_BF4_ROUNDED is not needed.
"""

import os

_KGPTQ = "hf://jerrywangTT/gpt-oss-20b-qb2-kgptq@110a7d8d43e2a0ec9ac12bea3f6b681e2c2d0961"

# gpt-oss-20b's release settings (tt-inference-server workflows/model_specs/dev/llm.yaml, gpt-oss-20b on P300X2),
# with the K-GPTQ prefill experts in place of GPT_OSS_DECODE_BF4_ROUNDED + GPT_OSS_PREFILL_BF4_ROUNDED.
RELEASE_DEFAULTS_20B = {
    # dense weights (attention, router, LM head) GPTQ-rounded for the fused decode copies
    "GPT_OSS_DENSE_ROUNDED_DECODE": f"{_KGPTQ}/dense/dense_gptq_device.pt",
    # decode boundary RMSNorm multiplies at HiFi3
    "GPT_OSS_BOUNDARY_FIDELITY": "HiFi3",
    # prefill RMSNorm with FP32 accumulation; prefill residual stream in bf16
    "GPT_OSS_PREFILL_NORM_FP32": "1",
    "GPT_OSS_PREFILL_RESIDUAL_BF16": "1",
    # fused QKV + RoPE loads y / cos / sin exactly (the cheaper variant)
    "GPT_OSS_QKV_ROPE_EXACT": "2",
    # decode experts: the exact MXFP4 split, exact top-1 expert on the listed layers, K-GPTQ rounding elsewhere
    "GPT_OSS_DECODE_EXPERTS_SPLIT": "1",
    "GPT_OSS_DECODE_SPLIT_RANKS": "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,1,0,1,1",
    "GPT_OSS_DECODE_SPLIT_ROUND_DIR": f"{_KGPTQ}/kgptq_experts",
    "GPT_OSS_DECODE_SPLIT_ROUND_ONLY": "1",
    "GPT_OSS_DECODE_SPLIT_NATIVE": "1",
    # the native split's two accumulators folded in FP32 on the SFPU (the kernel fix)
    "GPT_OSS_DECODE_SPLIT_NATIVE_FOLD": "1",
    # decode speedups
    "GPT_OSS_DECODE_BOUNDARY_EARLY_SUM": "1",
    "GPT_OSS_DECODE_ROUTER_FAST_TOPK": "1",
    # prefill experts read the same K-GPTQ weights as decode (tile-transposed); short prompts padded into that path
    "GPT_OSS_PREFILL_KGPTQ": "1",
}

# model_config.py: HF_MODEL, else the first of these internal CI checkpoints that exists, else the 120b one
_CI_CHECKPOINTS = ("/mnt/MLPerf/tt_dnn-models/openai/gpt-oss-20b", "/mnt/MLPerf/tt_dnn-models/openai/gpt-oss-120b")


def _model_path():
    path = os.environ.get("HF_MODEL")
    if path:
        return path
    return next((p for p in _CI_CHECKPOINTS if os.path.exists(p)), _CI_CHECKPOINTS[-1])


def is_gpt_oss_20b(path):
    name = path.lower()
    return "20b" in name and "120b" not in name


def apply_release_defaults():
    """Set each unset GPT_OSS_* release variable (gpt-oss-20b only); returns the names it set."""
    if os.environ.get("GPT_OSS_RELEASE_DEFAULTS", "1") == "0" or not is_gpt_oss_20b(_model_path()):
        return []
    applied = [k for k in RELEASE_DEFAULTS_20B if k not in os.environ]
    for k in applied:
        os.environ[k] = RELEASE_DEFAULTS_20B[k]
    return applied


RELEASE_DEFAULTS_APPLIED = apply_release_defaults()
if RELEASE_DEFAULTS_APPLIED:
    from loguru import logger

    logger.info(
        f"gpt-oss-20b release settings on by default ({len(RELEASE_DEFAULTS_APPLIED)} GPT_OSS_* variables not set "
        "already; GPT_OSS_RELEASE_DEFAULTS=0 turns them off)"
    )
