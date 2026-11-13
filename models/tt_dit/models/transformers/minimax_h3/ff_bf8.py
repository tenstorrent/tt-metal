# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MINIMAX_H3_FF1_OUT_BF8: block-float-8 hand-off from ff1 (+ fused SwiGLU) to ff2 (default off).

With `MINIMAX_H3_FF1_OUT_BF8=1` the ff1 all-gather-matmul packs its SwiGLU output as bfloat8_b (an explicit output
dtype, which also overrides a bf16 output pin on ff1), so the [N, ffn_dim / tp] intermediate is written to and read
from DRAM at 1088 B per tile instead of 2048 B, and ff2 unpacks block-float in0 tiles. ff2's fused matmul +
reduce-scatter + gated residual is asked for a bf16 output explicitly, so the reduce-scatter partials and the residual
stream stay 16-bit.

Unset or `0`: `ff_kwargs()` is empty, so every call is today's call. Any other value raises at import.
Lossy: the SwiGLU output is rounded to 7-bit mantissas with a shared exponent per 16 values.
"""

from __future__ import annotations

import os

import ttnn

_VALUE = os.environ.get("MINIMAX_H3_FF1_OUT_BF8", "0").strip()
if _VALUE not in ("", "0", "1"):
    msg = f"MINIMAX_H3_FF1_OUT_BF8={_VALUE!r}: expected 0 or 1"
    raise ValueError(msg)
ENABLED = _VALUE == "1"

_logged = False


def ff_kwargs() -> dict:
    """Extra kwargs for the feed-forward call: ff1 output bfloat8_b and ff2 output bf16, or none."""
    global _logged
    if not ENABLED:
        return {}
    if not _logged:
        _logged = True
        print("[MINIMAX_H3_FF1_OUT_BF8] ff1 (+ SwiGLU) out -> bfloat8_b (ff2 input); ff2 out stays bf16 (lossy)")
    return {"ff1_output_dtype": ttnn.bfloat8_b, "ff2_output_dtype": ttnn.bfloat16}
