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
# How many data-parallel groups share one call's items, each running its own items/SPLIT at the same
# time. Absent means 1: every chip group runs the whole call (replicated), which is also what a
# single-chip or TP-only pipeline is. Only the pipeline knows which of its stages it splits
# (Qwen-Image-Edit splits the denoise batch over its DP columns and runs the encoders replicated).
SPLIT = "_trace_split"
# How many chip groups split ONE REQUEST'S TOKENS in a call (sequence parallelism): each group runs its
# own tokens/SEQ_SPLIT slice of the same request at the same time and exchanges only attention K/V.
# Absent means 1: every group runs every token. Only the pipeline knows which stages it cuts along
# tokens -- a single-token step has nothing to cut.
SEQ_SPLIT = "_trace_seq_split"
# How many times ONE REQUEST runs this stage's step. Absent means 1: the step is the stage's whole work
# for a request. A loop the pipeline replays per request -- one token per decode step, one scheduler
# step per denoise step -- states its count, so the report can say that a full-pipeline pass timed each
# stage once while a request runs it N times. Only the pipeline knows N (its own schedule length).
REPEATS = "_trace_repeats"

# A stage cannot be measured at all without these: setup does host prep outside the trace, step is
# the one fixed-shape call inside it.
REQUIRED = (SETUP, STEP)

# Absent, these degrade rather than break -- but each degrades silently, which is why the contract
# reports them: INPUTS costs the stage its own boundary, ITEMS costs it a real arithmetic ceiling,
# SPLIT (on a stage that is split) prices it as if one chip group did the whole batch, SEQ_SPLIT (on a
# stage that cuts tokens) prices it as if one group ran the whole sequence.
OPTIONAL = (INPUTS, ITEMS, SPLIT, SEQ_SPLIT, REPEATS)

ALL = REQUIRED + OPTIONAL


# The tensor-parallel degree a pipeline runs at, when it states one -- read off the pipeline, never
# assumed from the mesh the operator typed: a 4x8 Galaxy can run TP=8 x DP=4 or TP=4 x DP=8, and only
# the model knows which (Qwen-Image-Edit: `self.tp = device.shape[TP_AXIS]`).
TP_ATTR = "tp"
# The sequence-parallel degree a pipeline runs at, when it states one -- the same rule: the mesh rows
# carry replicas as well as token groups, and only the pipeline knows how many of each.
SP_ATTR = "sp"


def hook(stage: str, seam: str) -> str:
    """The attribute a model exposes for `seam` on `stage`. The stage name comes from the model."""
    return "%s%s" % (stage, seam)
