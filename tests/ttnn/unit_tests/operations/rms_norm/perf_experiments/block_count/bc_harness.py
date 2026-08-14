# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared plumbing for the `block_count` bake-off (P5: one block per core).

The isolated concept is ONE knob — `block_rows` (and therefore `num_blocks`) —
held against everything else the real op does.  Rather than fork the kernels,
this bench drives the SHIPPED program descriptor and overrides exactly the one
plan field, so `baseline` is literally the op's current approach (a
byte-identical program) and a candidate differs only in `B`.

`_plan` is looked up in `create_program_descriptor.__globals__` at call time, so
replacing that dict entry is the whole hook.  Patching the module object
imported by name would patch a SECOND import nobody runs (documented in
test_rms_norm_perf_decode.py).
"""

from __future__ import annotations

import os

import ttnn
from ttnn.operations.rms_norm.rms_norm import create_program_descriptor as _create_program_descriptor

PLAN_GLOBALS = _create_program_descriptor.__globals__
_ORIG_PLAN = PLAN_GLOBALS["_plan"]

TARGET_FIDELITY = ttnn.MathFidelity.HiFi2
TARGET_FP32_ACC = False

# The last plan a hook produced — so a test can report the geometry it measured.
LAST_PLAN = {}


def target_compute_config():
    """The user's precision contract — FIXED for every variant in this bench."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=TARGET_FIDELITY,
        fp32_dest_acc_en=TARGET_FP32_ACC,
        math_approx_mode=False,
    )


def _div_up(a, b):
    return (a + b - 1) // b


def make_hook(label, *, force_block_rows=None, force_in_depth=None):
    """A `_plan` replacement that prints the plan and optionally overrides B.

    The override is how the sweep separates "one block" from "coarser block":
    at a FIXED L1 budget only `block_rows` moves, so the budget's other effects
    (it also gates the in_depth rung) cannot confound the reading.
    """

    def _hook(device, input_tensor, *, has_gamma, bytes_):
        plan = _ORIG_PLAN(device, input_tensor, has_gamma=has_gamma, bytes_=bytes_)
        ladder_b, ladder_depth = plan["block_rows"], plan["in_depth"]
        if force_block_rows is not None:
            plan["block_rows"] = force_block_rows
        if force_in_depth is not None:
            plan["in_depth"] = force_in_depth
        b = plan["block_rows"]
        rows = plan["shard_rows"] if plan["sharded"] else _div_up(plan["row_tiles"], plan["num_row_groups"])
        combine = PLAN_GLOBALS["_combine_owners"]
        owners = combine(plan["num_hidden_slices"], b) if plan["num_hidden_slices"] > 1 else 1
        LAST_PLAN.clear()
        LAST_PLAN.update(plan)
        LAST_PLAN["rows_per_core"] = rows
        LAST_PLAN["num_blocks"] = _div_up(rows, b) if rows else 0
        LAST_PLAN["num_owners"] = owners
        print(
            f"PLAN[{label}] sharded={plan['sharded']} G={plan['num_row_groups']} "
            f"s={plan['num_hidden_slices']} S={plan['slice_hidden_tiles']} "
            f"rows/core={rows} ladder_B={ladder_b} B={b} "
            f"num_blocks={LAST_PLAN['num_blocks']} owners={owners} "
            f"in_depth={plan['in_depth']}(ladder {ladder_depth})",
            flush=True,
        )
        return plan

    return _hook


def make_split_budget_hook(label, *, search_mb, ladder_mb, force_block_rows=None):
    """Give the two CONSUMERS of `_l1_working_budget` different budgets.

    `_plan` calls it TWICE and the two calls do different jobs:

      call 0 (interleaved only) — the partition-search ADMISSION FILTER: a
              (s, slice_tiles) candidate is admitted only if its b==1 footprint
              fits.  A wider budget therefore admits FATTER hidden slices, and
              the score (occupancy, then fewer slices) then prefers them.
      call 1 — the BLOCK/DEPTH LADDER: the coarsest block that fits.  This is
              the one P5 is about.

    Widening both is what Refinement 4 measured; this hook widens only the
    ladder, which is the narrowest form of the change.  `_plan_sharded` makes a
    single call and it IS the ladder, so the sharded path always gets `ladder_mb`.
    """
    reserve = PLAN_GLOBALS["L1_RESERVE"]
    is_sharded = PLAN_GLOBALS["is_sharded"]

    def _hook(device, input_tensor, *, has_gamma, bytes_):
        sharded = is_sharded(input_tensor)
        calls = {"n": 0}

        def _budget(_dev):
            n = calls["n"]
            calls["n"] += 1
            mb = search_mb if (n == 0 and not sharded) else ladder_mb
            return int(mb * 1024 * 1024) - reserve

        saved = PLAN_GLOBALS["_l1_working_budget"]
        PLAN_GLOBALS["_l1_working_budget"] = _budget
        try:
            inner = make_hook(f"{label}", force_block_rows=force_block_rows)
            return inner(device, input_tensor, has_gamma=has_gamma, bytes_=bytes_)
        finally:
            PLAN_GLOBALS["_l1_working_budget"] = saved

    return _hook


def guard_no_ablation():
    """A stale /tmp/rms_norm_ablate_bits silently stubs stage payloads."""
    assert not os.path.exists("/tmp/rms_norm_ablate_bits"), "ablation hook present — delete /tmp/rms_norm_ablate_bits"
