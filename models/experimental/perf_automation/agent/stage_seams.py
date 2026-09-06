# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The names of the per-stage seams the perf engine binds -- in ONE place.

WHY THIS EXISTS. The seam set was spelled as string literals in six files: the adapter that calls
them, the contract that checks them, the mark injector, the op-signature probe, the perf-test
generator, and the emit-e2e prompt that tells a model to write them. Adding `_trace_items` in
August touched only the CONSUMER copies, so the generator never learned to emit it and the contract
never learned to ask for it -- and a seam nothing produces reads, downstream, as a stage that
retires exactly one item. Voxtral's audio encoder was therefore priced at 1 item instead of 1500,
its compute roof came out ~1500x low, and it was reported memory-bound when it is compute-bound.

The lists below are the tool's OWN protocol, not model vocabulary: they are suffixes appended to
whatever stage names the model itself declares. Nothing here names a stage, a component or a model.
"""

from __future__ import annotations

SETUP = "_trace_setup"
STEP = "_trace_step"
INPUTS = "_trace_inputs"
ITEMS = "_trace_items"

# A stage cannot be measured at all without these: setup does host prep outside the trace, step is
# the one fixed-shape call inside it.
REQUIRED = (SETUP, STEP)

# Absent, these degrade rather than break -- but each degrades silently, which is why the contract
# reports them: INPUTS costs the stage its own boundary, ITEMS costs it a real arithmetic ceiling.
OPTIONAL = (INPUTS, ITEMS)

ALL = REQUIRED + OPTIONAL


def hook(stage: str, seam: str) -> str:
    """The attribute a model exposes for `seam` on `stage`. The stage name comes from the model."""
    return "%s%s" % (stage, seam)
