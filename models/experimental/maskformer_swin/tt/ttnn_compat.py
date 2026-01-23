# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers for optional TT-NN availability.

The MaskFormer demo relies on TT-NN for accelerator execution, but many
development workflows (fallback parity, docs generation, unit tests) run on
CPUs without TT runtimes installed. Importing ``ttnn`` at module scope would
raise ``ModuleNotFoundError`` in those scenarios, so this module centralizes
the optional import and exposes lightweight helpers to query availability or
raise actionable errors when TT functionality is required.
"""

from __future__ import annotations

from typing import Any, Optional

try:  # pragma: no cover - exercised implicitly when TT-NN is installed
    import ttnn as _ttnn  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - CPU-only development environment
    _ttnn = None
else:  # pragma: no cover - validate we imported the real TTNN package
    # When running from the tt-metal repo without installing the python package, `import ttnn`
    # can resolve to the top-level `ttnn/` source directory as a namespace package, which is not
    # the runtime module and does not expose APIs like `from_torch`. Treat that as unavailable.
    if not hasattr(_ttnn, "from_torch"):
        _ttnn = None


ttnn = _ttnn  # re-export for convenience inside MaskFormer modules


def is_available() -> bool:
    """Return ``True`` when the TT-NN runtime is importable."""

    return ttnn is not None


def require_ttnn(action: str) -> Any:
    """Raise a helpful error when TT-NN functionality is requested but missing."""

    if ttnn is None:
        raise RuntimeError(f"TT-NN runtime is required to {action}. Install tt-metal / ttnn in this environment.")
    return ttnn


def get_default_dtype() -> Optional[Any]:
    """Best-effort default dtype (bf16 → fp16 → fp32) or ``None`` when unavailable."""

    for candidate in ("bfloat16", "float16", "float32"):
        dtype = getattr(ttnn, candidate, None)
        if dtype is not None:
            return dtype
    return None
