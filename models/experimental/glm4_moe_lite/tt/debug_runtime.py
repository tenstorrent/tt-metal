# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import Any

import ttnn


def _env_bool(name: str) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    return bool(raw) and raw not in {"0", "false", "no", "off"}


_ORIG_TTNN_DEALLOCATE = ttnn.deallocate


def _noop_deallocate(*args: Any, **kwargs: Any) -> None:
    # Keep signature-compatible with ttnn.deallocate; used only for debugging.
    return None


# Debug knob: disable all explicit deallocation calls in the GLM-4.7 TT stack.
#
# Motivation: some TTNN view-like ops may alias buffers without refcounting, and
# aggressive manual deallocation can become a use-after-free bug under async
# execution. When this happens, greedy decode can become nondeterministic or
# emit garbled tokens.
#
# This flag is intended ONLY for correctness isolation (short runs). Disabling
# deallocation can increase memory pressure and should not be used for perf runs.
if _env_bool("GLM4_MOE_LITE_DISABLE_DEALLOC"):
    ttnn.deallocate = _noop_deallocate  # type: ignore[assignment]
