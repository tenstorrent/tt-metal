# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Built-in graphs used as gold cases and as a smoke test for the analyzer."""

from __future__ import annotations

from typing import Callable, Dict

from ..ir import Graph
from .ltx import ltx_block_bh_2x4, ltx_block_bh_4x8
from .sd35 import sd35_block, sd35_block_double_gather
from .synthetic import synthetic_redundancy

EXAMPLES: Dict[str, Callable[[], Graph]] = {
    "sd35_block": sd35_block,
    "sd35_block_double_gather": sd35_block_double_gather,
    "synthetic_redundancy": synthetic_redundancy,
    "ltx_block_bh_4x8": ltx_block_bh_4x8,
    "ltx_block_bh_2x4": ltx_block_bh_2x4,
}

DESCRIPTIONS: Dict[str, str] = {
    "sd35_block": "SD3.5-large joint block, sp=2/tp=4 -- every collective is necessary (precision test)",
    "sd35_block_double_gather": "same block using fused AGMM without dropping the explicit pre-gathers (12 -> 6)",
    "synthetic_redundancy": "dead / over-wide / participant-shrink / step-invariant collectives",
    "ltx_block_bh_4x8": "LTX-2.3 A+V block on Blackhole 4x8, Ring topology (fused AGMM path)",
    "ltx_block_bh_2x4": "LTX-2.3 A+V block on Blackhole 2x4, Linear topology (explicit-gather path)",
}


def load(name: str) -> Graph:
    if name not in EXAMPLES:
        raise KeyError("unknown example '%s' (have: %s)" % (name, ", ".join(sorted(EXAMPLES))))
    return EXAMPLES[name]()


__all__ = [
    "EXAMPLES",
    "DESCRIPTIONS",
    "load",
    "sd35_block",
    "sd35_block_double_gather",
    "synthetic_redundancy",
    "ltx_block_bh_4x8",
    "ltx_block_bh_2x4",
]
