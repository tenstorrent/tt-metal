# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Which MoE block TtMoe runs: the data movement that takes the gate's top-k to the routed experts and back.

Every block shares TtMoe's gate, TP gather, latent projections and shared expert, and returns the routed output in the
same layout ([1, 1, S, Hr / TP] bf16 TILE, reduce-scattered over the mesh columns on the hidden dim), so they swap
without touching the rest of the model:

    model config  MOE_BLOCK_IMPL = "auto"            (per model; absent -> "dispatch_combine")
    environment   TT_DS_PREFILL_MOE_BLOCK=dispatch_combine   (per run, for A/B; wins over the config)
    constructor   TtMoe(..., moe_block="all_gather")  (tests)

A new block: add its name here, build it in TtMoe.__init__ and route to it in TtMoe.forward.
"""

import os

MOE_BLOCK_ENV = "TT_DS_PREFILL_MOE_BLOCK"

MOE_BLOCKS = {
    # routing setup -> dispatch (each token to its K experts' chips) -> routed expert -> combine -> post-combine reduce
    "dispatch_combine": "tt_dispatch.py / tt_combine.py / tt_reduce.py",
    # all-gather the column's tokens -> on-device route plan -> flat expert (indexed) -> local reduce -> send back.
    # Needs the flat routed expert (Blackhole, <= 64 experts per chip); falls back to dispatch_combine otherwise.
    "all_gather": "tt_moe_ag.py",
    # all_gather on meshes with <= AUTO_ALL_GATHER_MAX_ROWS rows, dispatch_combine on taller ones. On a LoudBox
    # (2 x 4) the all-gather block is 6-15% less device time and ~18% less wall clock for Kimi-K2.7 / K3 / GLM-5.3
    # (tests/perf/test_moe_block_perf.py); on the Galaxy (8 x 4) every chip receives 7 mesh rows of tokens instead of 1
    # and the local reduce covers 4x the tokens, which is projected to cancel the gain (K2.7 4.6-5.9 ms against the
    # 4.95 ms dispatch_combine measures there, K3 4.9-5.7 against 4.96): dispatch_combine stays there until the Galaxy
    # perf leg (moe_block_perf) measures it.
    "auto": "all_gather on <= AUTO_ALL_GATHER_MAX_ROWS mesh rows, else dispatch_combine",
}

AUTO_ALL_GATHER_MAX_ROWS = 2

DEFAULT_MOE_BLOCK = "dispatch_combine"


def check_moe_block(name: str) -> str:
    if name not in MOE_BLOCKS:
        raise ValueError(f"unknown MoE block {name!r} (${MOE_BLOCK_ENV}); known: {sorted(MOE_BLOCKS)}")
    return name


def pick_moe_block(name: str, mesh_rows: int) -> str:
    """The concrete block for ``name`` on a mesh with ``mesh_rows`` rows (the dispatch axis): resolves "auto"."""
    check_moe_block(name)
    if name != "auto":
        return name
    return "all_gather" if mesh_rows <= AUTO_ALL_GATHER_MAX_ROWS else "dispatch_combine"


def resolve_moe_block(model_cfg=None) -> str:
    """``$TT_DS_PREFILL_MOE_BLOCK`` when set, else the model config's ``MOE_BLOCK_IMPL``, else "dispatch_combine"."""
    return check_moe_block(
        os.environ.get(MOE_BLOCK_ENV) or getattr(model_cfg, "MOE_BLOCK_IMPL", None) or DEFAULT_MOE_BLOCK
    )
