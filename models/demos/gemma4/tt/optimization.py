# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4 profiling and optimization knobs."""

import os
from dataclasses import dataclass

import ttnn
from loguru import logger


_DTYPE_ALIASES = {
    "bf16": ("bf16", ttnn.bfloat16),
    "bfloat16": ("bf16", ttnn.bfloat16),
    "bfp8": ("bfp8", ttnn.bfloat8_b),
    "bfloat8_b": ("bfp8", ttnn.bfloat8_b),
    "bf8": ("bfp8", ttnn.bfloat8_b),
    "bfp4": ("bfp4", ttnn.bfloat4_b),
    "bfloat4_b": ("bfp4", ttnn.bfloat4_b),
    "bf4": ("bfp4", ttnn.bfloat4_b),
}


_PRECISION_PROFILE_ALIASES = {
    "bf16": "bf16",
    "all_bf16": "bf16",
    "all-bf16": "bf16",
    "mixed": "mixed_bfp8",
    "mixed_bfp8": "mixed_bfp8",
    "balanced": "mixed_bfp8",
}

_PRECISION_PROFILES = {
    # Known-good path.  Keep BF16 cache names unchanged so existing warmed caches
    # remain valid when GEMMA4_PRECISION_PROFILE=bf16 is requested.
    "bf16": {
        "attention_qkv": "bf16",
        "attention_o_proj": "bf16",
        "shared_mlp_gate": "bf16",
        "shared_mlp_up": "bf16",
        "shared_mlp_down": "bf16",
        "expert_gate": "bf16",
        "expert_up": "bf16",
        "expert_down": "bf16",
        "lm_head": "bf16",
    },
    # Conservative mixed profile from neighboring LLM evidence: projection-heavy
    # matmuls use BFP8, while embedding, norms, router/gating auxiliaries,
    # KV-cache, RoPE, and LM head stay BF16 in their owning modules.
    "mixed_bfp8": {
        "attention_qkv": "bfp8",
        "attention_o_proj": "bfp8",
        "shared_mlp_gate": "bfp8",
        "shared_mlp_up": "bfp8",
        "shared_mlp_down": "bfp8",
        "expert_gate": "bfp8",
        "expert_up": "bfp8",
        "expert_down": "bfp8",
        "lm_head": "bf16",
    },
}


@dataclass(frozen=True)
class PrecisionChoice:
    dtype: object
    cache_suffix: str
    canonical: str
    source: str


def env_weight_dtype(env_name: str, default):
    """Return ``(dtype, cache_suffix)`` for an optional dtype env override."""

    raw = os.getenv(env_name)
    if raw is None or raw == "":
        return default, ""
    key = raw.strip().lower()
    if key not in _DTYPE_ALIASES:
        valid = ", ".join(sorted(_DTYPE_ALIASES))
        raise ValueError(f"Unsupported {env_name}={raw!r}. Valid values: {valid}")
    canonical, dtype = _DTYPE_ALIASES[key]
    logger.info(f"Gemma4 override: {env_name}={canonical}")
    return dtype, f"_{canonical}"


def precision_profile_name() -> str:
    """Return the active Gemma4 precision profile name."""

    raw = os.getenv("GEMMA4_PRECISION_PROFILE", "mixed_bfp8")
    key = raw.strip().lower()
    profile = _PRECISION_PROFILE_ALIASES.get(key)
    if profile is None:
        valid = ", ".join(sorted(_PRECISION_PROFILE_ALIASES))
        raise ValueError(f"Unsupported GEMMA4_PRECISION_PROFILE={raw!r}. Valid values: {valid}")
    return profile


def _dtype_from_canonical(canonical: str):
    return _DTYPE_ALIASES[canonical][1]


def profile_weight_dtype(
    tensor_group: str,
    *,
    env_name: str | None = None,
    legacy_env_name: str | None = None,
) -> PrecisionChoice:
    """Resolve a weight dtype from a specific env var, legacy env var, or profile.

    Env overrides always get an explicit cache suffix, including ``_bf16``.  Profile
    defaults suffix only lower-precision tensors; the all-BF16 profile intentionally
    preserves the historical suffix-free cache names.
    """

    for candidate_env in (env_name, legacy_env_name):
        if not candidate_env:
            continue
        raw = os.getenv(candidate_env)
        if raw is None or raw == "":
            continue
        key = raw.strip().lower()
        if key not in _DTYPE_ALIASES:
            valid = ", ".join(sorted(_DTYPE_ALIASES))
            raise ValueError(f"Unsupported {candidate_env}={raw!r}. Valid values: {valid}")
        canonical, dtype = _DTYPE_ALIASES[key]
        logger.info(f"Gemma4 override: {candidate_env}={canonical} for {tensor_group}")
        return PrecisionChoice(dtype=dtype, cache_suffix=f"_{canonical}", canonical=canonical, source=candidate_env)

    profile = precision_profile_name()
    try:
        canonical = _PRECISION_PROFILES[profile][tensor_group]
    except KeyError as exc:
        valid = ", ".join(sorted(next(iter(_PRECISION_PROFILES.values()))))
        raise ValueError(f"Unknown Gemma4 precision tensor group {tensor_group!r}. Valid groups: {valid}") from exc

    suffix = "" if canonical == "bf16" else f"_{canonical}"
    return PrecisionChoice(
        dtype=_dtype_from_canonical(canonical),
        cache_suffix=suffix,
        canonical=canonical,
        source=f"profile:{profile}",
    )
