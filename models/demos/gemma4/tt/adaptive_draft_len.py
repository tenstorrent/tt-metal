# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Adaptive speculative draft length (K) selection for Gemma4.

Throughput is acceptance-limited: ``tok/s ≈ (mean_accept + 1) / iter_time``.
High K helps when the drafter matches the target (structured / long-context /
code); low K wastes less draft+verify work when acceptance collapses (creative /
open-ended).

Measured QB2 (P150x4, K=16) anchors from ``Legacy_Gemma4_performance.md``:

| Workload                         | mean accept / K | tok/s |
|----------------------------------|-----------------|-------|
| Structured / repetitive          | 15.6 / 16       | ~81   |
| Long-context analytical          | 10.77 / 16      | ~52   |
| Code                             | 7.8 / 16        | ~44   |
| Summarize                        | 2.9 / 16        | ~20   |
| Creative story (short prompt)    | 1.6 / 16        | ~14   |

This module picks a per-request K. The fused CCL trace bakes K at capture time,
so callers must ``release_fused_trace()`` before changing ``draft_len`` (see
``SpeculativeDecoder.set_draft_len``). Mid-request K changes are unsafe because
trace capture writes KV.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Literal

WorkloadClass = Literal["high_accept", "medium_accept", "low_accept"]


@dataclass(frozen=True)
class AdaptiveDraftLenConfig:
    """Env-tunable K bands for adaptive speculative decode."""

    enabled: bool = True
    k_high: int = 16
    k_mid: int = 8
    k_low: int = 6
    # Prompt-token thresholds (encoded length, not chars).
    long_prompt_tokens: int = 2048
    short_prompt_tokens: int = 128

    @classmethod
    def from_env(cls, default_draft_len: int | None = None) -> AdaptiveDraftLenConfig:
        """Build config from env; ``GEMMA4_SPEC_DRAFT_LEN`` is the high band default."""
        high = int(os.environ.get("GEMMA4_SPEC_DRAFT_LEN", str(default_draft_len or 16)))
        mid = int(os.environ.get("GEMMA4_SPEC_DRAFT_LEN_MID", "8"))
        low = int(os.environ.get("GEMMA4_SPEC_DRAFT_LEN_LOW", "6"))
        # Keep bands ordered and valid.
        high = max(1, high)
        mid = max(1, min(mid, high))
        low = max(1, min(low, mid))
        enabled_raw = os.environ.get("GEMMA4_SPEC_ADAPTIVE_K")
        if enabled_raw is None:
            enabled = True
        else:
            enabled = enabled_raw.lower() in ("1", "true", "yes")
        return cls(
            enabled=enabled,
            k_high=high,
            k_mid=mid,
            k_low=low,
            long_prompt_tokens=int(os.environ.get("GEMMA4_SPEC_ADAPTIVE_LONG_TOKENS", "2048")),
            short_prompt_tokens=int(os.environ.get("GEMMA4_SPEC_ADAPTIVE_SHORT_TOKENS", "128")),
        )


# Creative / open-ended cues → low K (acceptance collapses at K=16).
_CREATIVE_RE = re.compile(
    r"\b("
    r"story|stories|tale|fiction|novel|poem|poetry|lyrics|song|screenplay|"
    r"creative|imagine|fantasy|make[- ]believe|roleplay|role[- ]play|"
    r"write\s+(me\s+)?(a\s+)?(short\s+)?story|"
    r"tell\s+me\s+a\s+story|"
    r"once\s+upon\s+a\s+time"
    r")\b",
    re.IGNORECASE,
)

# Summarize / rewrite → mid K (measured ~2.9/16 accept).
_SUMMARIZE_RE = re.compile(
    r"\b("
    r"summarize|summarise|summary|tldr|tl;dr|eli5|"
    r"rewrite|paraphrase|condense|shorten|bullet\s*points?"
    r")\b",
    re.IGNORECASE,
)

# Code / structured → keep high K (measured ~7.8–15.6/16).
_CODE_RE = re.compile(
    r"\b("
    r"python|javascript|typescript|rust|golang|java|c\+\+|cpp|"
    r"function|class\s+\w+|def\s+\w+|import\s+\w+|```|"
    r"algorithm|merge[_ ]?sort|quick[_ ]?sort|leetcode|"
    r"unit\s+test|type\s+hints?|docstring|regex|sql\b|json\b"
    r")\b",
    re.IGNORECASE,
)

# Explicit structured / repetitive → high K.
_STRUCTURED_RE = re.compile(
    r"\b(" r"repeat|numbered|list\s+\d+|enumerate|" r"step[- ]by[- ]step|exactly\s+\d+\s+times" r")\b",
    re.IGNORECASE,
)


def classify_workload(
    prompt: str,
    *,
    prompt_tokens: int = 0,
    max_new_tokens: int = 0,
    short_prompt_tokens: int = 128,
    long_prompt_tokens: int = 2048,
) -> WorkloadClass:
    """Classify a prompt into an acceptance band for K selection."""
    text = prompt or ""
    if _STRUCTURED_RE.search(text) or _CODE_RE.search(text):
        return "high_accept"
    if _CREATIVE_RE.search(text):
        return "low_accept"
    if _SUMMARIZE_RE.search(text):
        return "medium_accept"
    # Short open-ended prompts with a large generation budget behave like creative.
    if prompt_tokens > 0 and prompt_tokens <= short_prompt_tokens and max_new_tokens >= 256:
        return "low_accept"
    if prompt_tokens >= long_prompt_tokens:
        return "high_accept"
    return "high_accept"


def select_adaptive_draft_len(
    prompt: str,
    *,
    prompt_tokens: int = 0,
    max_new_tokens: int = 0,
    config: AdaptiveDraftLenConfig | None = None,
    override: int | None = None,
) -> tuple[int, WorkloadClass | None]:
    """Pick draft length K for this request.

    Returns ``(k, workload_class)``. ``workload_class`` is None when adaptive
    selection is disabled or an explicit override is used.
    """
    if override is not None:
        k = int(override)
        if k < 1:
            raise ValueError(f"draft_len override must be >= 1, got {k}")
        return k, None

    cfg = config or AdaptiveDraftLenConfig.from_env()
    if not cfg.enabled:
        return cfg.k_high, None

    workload = classify_workload(
        prompt,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        short_prompt_tokens=cfg.short_prompt_tokens,
        long_prompt_tokens=cfg.long_prompt_tokens,
    )
    if workload == "low_accept":
        return cfg.k_low, workload
    if workload == "medium_accept":
        return cfg.k_mid, workload
    return cfg.k_high, workload
