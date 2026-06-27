"""Detect "TTNN kernel is missing" from agent failure traces.

When a HOT component (invoked in workload) doesn't graduate, there are
only two legitimate explanations:

  1. TTNN doesn't have the operation the component needs.
     → KERNEL_MISSING — flag it, don't keep retrying, allow demo emission
       (the component stays on CPU fallback; user gets a clear list of
        "ttnn ops we need")

  2. The LLM hasn't figured out how to write the implementation yet.
     → keep iterating; this should be transient

Anything else (random API failures, tooling bugs) is a transient issue
that more iterations would fix.

This module pattern-matches agent failure traces for kernel-missing
signals. Generic across TTNN ops; no hard-coded list of supported
operations.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple


# Patterns that strongly indicate "TTNN lacks the operation". Each
# pattern is paired with a capture group (or string) describing the
# missing op so we can persist a useful message in missing_kernels.json.
# Boundaries are kept loose (no trailing-newline requirement) so short
# error strings still match.
_KERNEL_MISSING_PATTERNS: List[Tuple[str, str]] = [
    # TT_FATAL with "not implemented" / "not supported" — optional op detail.
    # Op capture is GREEDY-to-EOL so "ttnn.embedding with float16" is kept
    # as one description rather than truncated at the first dot.
    (r"TT_FATAL[^\n]*?not implemented(?:[:\-\s]+(?P<op>[^\n]+))?", "{op}"),
    (r"TT_FATAL[^\n]*?not supported(?:[:\-\s]+(?P<op>[^\n]+))?", "{op}"),
    # Generic "not implemented" raised from a ttnn call
    (r"NotImplementedError[^\n]*?(ttnn\.[\w_\.]+)", "{1}"),
    # Bare "not implemented" / "not yet implemented" with optional context
    (r"\bnot (?:yet )?implemented\b(?:[:\-\s]+(?P<op>[^\n]+))?", "{op}"),
    # "no kernel" / "no matching kernel"
    (r"no (?:matching )?kernel(?:\s+for\s+(?P<op>[\w\.][\w\. ]*))?", "{op}"),
    # "unsupported op" / "operation not supported"
    (r"unsupported (?:op|operation)(?:[:\-\s]+(?P<op>[\w\.][\w\. ]*))?", "{op}"),
    # ttnn-specific phrasing:
    (r"ttnn does not support (?P<op>[\w\.\s]+)", "{op}"),
    (r"sparse_coo[^\n]*?not supported", "ttnn ops on sparse_coo tensors"),
]


def detect_kernel_missing(failure_text: str) -> Optional[str]:
    """If ``failure_text`` matches a kernel-missing pattern, return a
    short description of the missing op. Returns None otherwise.

    Examples that match:
      "TT_FATAL: feature not implemented: ttnn.embedding with float16"
        → "ttnn.embedding with float16"
      "NotImplementedError: ttnn.scaled_dot_product_attention(causal=True, ...)"
        → "ttnn.scaled_dot_product_attention"
      "RuntimeError: no kernel for ttnn.permute on sparse_coo"
        → "ttnn.permute"
    """
    if not failure_text:
        return None
    for pattern, template in _KERNEL_MISSING_PATTERNS:
        m = re.search(pattern, failure_text, re.IGNORECASE | re.DOTALL)
        if not m:
            continue
        # Extract the op description
        try:
            if "{op}" in template:
                op = (m.groupdict().get("op") or "").strip()
                desc = template.replace("{op}", op).strip() if op else "(operation unspecified)"
            elif "{1}" in template:
                op = m.group(1).strip()
                desc = template.replace("{1}", op).strip()
            else:
                desc = template
        except (IndexError, AttributeError):
            desc = "(operation unspecified)"
        return desc or "(operation unspecified)"
    return None


def is_kernel_missing_failure(failure_text: str) -> bool:
    """Boolean version of detect_kernel_missing — used for short-circuit
    checks in the auto-iterate loop."""
    return detect_kernel_missing(failure_text) is not None


def verify_ttnn_op_exists(op_name: str) -> Optional[bool]:
    """Check if ``op_name`` resolves to a real ttnn attribute.

    Returns:
      True  — op exists in ttnn (a false-positive kernel-missing label
              would be wrong; the agent failed for a non-kernel reason)
      False — op genuinely missing from ttnn (kernel-missing is correct)
      None  — couldn't verify (generic annotation, ttnn not importable,
              etc.). Caller should treat as "unverified" and not over-
              commit to either interpretation.

    Used by the auto-iterate loop to gate KERNEL_MISSING categorization:
    only persist KERNEL_MISSING if the op is verified MISSING. If the
    op exists (False positive), keep iterating or fall back to COLD.
    """
    if not op_name or not isinstance(op_name, str):
        return None
    clean = op_name.split("(")[0].strip()
    # Strip leading words that aren't part of the op (e.g. annotations)
    parts_after_ttnn = clean.split("ttnn.", 1)
    if len(parts_after_ttnn) != 2:
        return None
    op_path = "ttnn." + parts_after_ttnn[1].split(" ")[0].split(":")[0].rstrip(".,;")
    if not op_path.startswith("ttnn."):
        return None
    try:
        import ttnn
    except ImportError:
        return None
    parts = op_path[len("ttnn.") :].split(".")
    obj = ttnn
    for part in parts:
        if not part:
            return None
        if not hasattr(obj, part):
            return False
        obj = getattr(obj, part)
    return True
