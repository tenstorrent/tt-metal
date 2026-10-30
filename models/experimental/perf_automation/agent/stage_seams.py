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

import re

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
# The tensors a pipeline reads BY INDEX -- its lookup tables: token embeddings, position tables, any
# tensor a row is selected from -- as a method or attribute returning them, in any nesting of lists,
# tuples or dicts. A unit of work reads the ROWS an index selects from such a table, not the table,
# and nothing else can tell a table from a weight of the same shape: voxtral's embed_tokens and
# lm_head are both [131072, 3072], and an op the model wrote itself (a generic_op gather) carries no
# API name to key on. Only the pipeline knows, so it says, ONCE; every reader -- the dispatch hook
# that measures a stage's streamed bytes, the census that credits resident bytes to a subtree -- asks
# here. Unstated, every table is charged in full: the floor reads too high and the stack too short,
# the direction that keeps a run working rather than stopping it early.
GATHERED_TABLES_ATTR = "gathered_tables"


# A DECLARED STAGE THE MEASUREMENT COULD NOT TIME SAYS SO, on a line of its own:
# TRACE_STAGE_UNMEASURED[<stage>]=<why>. The adapter and trace_replay give such a stage no row so the
# others keep theirs -- and a whole-pipeline headline is then the sum of the stages that RAN. On a
# 32-chip port (2026-10-13) an edit crashed three of five stages, the sum of the two left (170 ms)
# was read as the full pipeline (12332 ms), banked as a -98.6% win and promoted to
# the committed best, after which every real reading was refused as +7136% diverged. The gate reads
# this line to know the reading is partial; printed and parsed only through the two helpers below.
UNMEASURED_MARK = "TRACE_STAGE_UNMEASURED"


def unmeasured_line(stage: str, why: str) -> str:
    """The marker line for `stage`. `why` is folded onto one line so a reader can split on `]=`."""
    return "%s[%s]=%s" % (UNMEASURED_MARK, stage, " ".join(str(why).split()) or "unknown")


def parse_unmeasured(line: str):
    """(stage, why) when `line` carries the marker, else None."""
    if UNMEASURED_MARK + "[" not in line:
        return None
    rest = line.split(UNMEASURED_MARK + "[", 1)[1]
    if "]=" not in rest:
        return None
    stage, why = rest.split("]=", 1)
    stage = stage.strip()
    return (stage, why.strip()) if stage else None


# WHAT A `why` MUST STILL SAY AFTER IT IS SHORTENED. The two writers of that field (the adapter for a
# stage that cannot prepare its inputs, trace_replay for one that cannot be timed) kept the first 120
# characters of the exception, which is the right length for a log line and the wrong one for this
# failure: a trace region too small for a stage raises a TT_FATAL whose head is the region assertion
# and whose SIZING sentence -- the only place the device says how many bytes the capture wanted --
# comes several lines later. Clipped to the head alone, the marker said a stage overflowed but never
# by how much, so the loops that grow the region off these lines (perf_test_gen._settle_perf_node,
# perf_mcp._grow_trace_region_and_retry) had nothing to size with and fell back to blind doubling:
# on Qwen-Image-Edit that is three re-runs of a 25-minute measurement to reach a size the first
# failure had already named. The pattern lives here, beside the field it has to survive in, and the
# region code reads it from here rather than keeping a second copy.
UNMEASURED_WHY_HEAD = 120
TRACE_REGION_NEED_RE = re.compile(r"Creating trace buffers of size (\d+)B.*?only (\d+)B is allocated")


def unmeasured_why(exc, head: int = UNMEASURED_WHY_HEAD) -> str:
    """`exc` folded to one line for `unmeasured_line`: its first `head` characters, plus whatever
    sizing the device reported past them. Nothing model-specific: the sentence is the device's."""
    text = " ".join(str(exc).split())
    short = text[:head]
    m = TRACE_REGION_NEED_RE.search(text)
    if m and m.end() > len(short):
        short = "%s ... %s" % (short.rstrip(), m.group(0))
    return short


def hook(stage: str, seam: str) -> str:
    """The attribute a model exposes for `seam` on `stage`. The stage name comes from the model."""
    return "%s%s" % (stage, seam)
