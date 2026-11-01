# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-shape activation and output placement, matching how the model calls each matmul.

Getting this wrong is not a small error. On the LM head the same shape costs 3718 us
with the activation in DRAM and 8456 us with it in L1, because mcast_in0 re-broadcasts
the activation once per output column and a large L1 activation makes that dominant.
So placement is part of the shape's identity, not a benchmark setting, and it lives in
one table here rather than in a --xin flag that applies to everything.

From the planner's TP1+norm decode profile (STATUS.md line 351), per matmul:

    ffn_down    17408x5120  bfp8  L1   -> L1     88 cores  347.7 us
    ffn_gate/up 5120x17408  bfp4  L1   -> L1    110 cores  170.0 us
    gdn in-proj 5120x16480  bfp8  L1   -> L1     44 cores  232.5 us
    gdn out      6144x5120  bfp8  L1   -> DRAM   33 cores   88.4 us
    attn q+gate 5120x12288  bfp8  L1   -> L1     99 cores  167.6 us
    attn k, v   5120x1024   bfp8  L1   -> L1     33 cores   73.4 us
    attn o_proj  6144x5120  bfp8  DRAM -> L1     88 cores  123.3 us
    lm_head     5120x248320 bfp8  DRAM -> DRAM     -            -   (3718 us device)

The two DRAM entries are the projections whose neighbour in the graph is not an
L1-resident tile op: o_proj reads a tensor that the attention output leaves in DRAM,
and the LM head reads what the final norm leaves in DRAM. Everything else is L1 to
L1, which is what the block-masked scale fold will also want since its epilogue
consumes the partials immediately.

Usage::

    from .placement import act_mem, out_mem

    x = ttnn.from_torch(..., memory_config=act_mem(ttnn, name))
    y = ttnn.linear(x, w, memory_config=out_mem(ttnn, name))
"""

from __future__ import annotations

#: Shapes whose activation enters the matmul from DRAM instead of L1.
ACT_FROM_DRAM = frozenset({"lm_head", "attn_o"})

#: Shapes whose output leaves for DRAM instead of staying in L1.
OUT_TO_DRAM = frozenset({"gdn_out", "lm_head"})


def act_mem(ttnn, name: str):
    """Memory config for the activation of ``name``, as the model places it."""
    return ttnn.DRAM_MEMORY_CONFIG if name in ACT_FROM_DRAM else ttnn.L1_MEMORY_CONFIG


def out_mem(ttnn, name: str):
    """Memory config for the output of ``name``, as the model places it."""
    return ttnn.DRAM_MEMORY_CONFIG if name in OUT_TO_DRAM else ttnn.L1_MEMORY_CONFIG


def describe(name: str) -> str:
    """``L1->L1`` style label for a run record, so a number carries its placement."""
    return f"{'DRAM' if name in ACT_FROM_DRAM else 'L1'}->{'DRAM' if name in OUT_TO_DRAM else 'L1'}"
