# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The tokens the op-signature probe writes into its sequence that are NOT ops. One definition.

THE SEQUENCE IS A WIRE FORMAT WITH TWO KINDS OF ENTRY. The probe appends an entry per dispatched
ttnn op, and interleaves bookkeeping tokens that say where a block began, where it ended, which
declared stage is running and which object made the call. Every reader therefore has to answer the
same question before it can do anything else: is this entry work, or is it structure?

WHY THIS MODULE EXISTS. The answer was written out four times -- as `_SIGNPOST_PREFIX` in the
probe, `_SIGNPOST_TOKEN` in run.py, `_SIGNPOST_PREFIX` again in perf_mcp.py, and as a literal
tuple inside run._is_control -- and the copies did not say the same thing. run.py excluded all four
prefixes; perf_mcp.py excluded only the block-START prefix, so a block-END token read as an op.

That divergence is not cosmetic. Both modules carry the same guard against DECOUPLED SIGNPOSTS: a
run whose markers all land in a clump with no ops after them delimits nothing, because every op
would attribute to block 0. The guard asks "is there any work after the first marker?" -- and with
END tokens counted as work, a clump of starts and ends answers yes to perf_mcp and no to run.py.
The same sequence was usable to one reader and unusable to the other, and the one that said yes
feeds check_lever_coverage, which then reports which blocks a dtype lever reached.

So the vocabulary lives here, the predicate lives here, and a marker invented tomorrow is added in
one place.
"""

from __future__ import annotations

# NOTHING IS IMPORTED HERE ON PURPOSE. run.py states at its top that "cc_optimize is not a package:
# these modules run both as scripts and as plain imports", which is why it loads tmpstate.py by file
# path rather than importing it. A vocabulary with no imports of its own can be loaded that same way
# from anywhere -- as a package module, as a top-level module, or by path -- so no reader has to
# re-spell a token because of how it happened to be started. caller_stacks takes its MARK from here
# for the same reason: the emitter of a token does not have to be the module that names it.

BLOCK = "PERF_BLOCK_SIGNPOST:"

# A BLOCK HAS TWO EDGES AND ONLY ONE WAS MARKED. With a start marker alone, every op emitted after
# the last block of a stack is attributed to that block -- and "after the last block" is the whole
# rest of the model. Measured on Voxtral 2026-08-13: a normal encoder block dispatches 20 ops and
# the LAST one was credited with 12573, i.e. 67% of the entire run, including embedding, rms_norm,
# silu, scaled_dot_product_attention_decode and argmax -- language-model decode ops that are not in
# the encoder at all. Coverage then concluded the encoder needs all 32 layers (it needs 1-2: all 32
# blocks are one class emitting identical ops), max() took 32 as the window, capping to 32 changed
# no work, and the run profiled the whole model.
BLOCK_END = "PERF_BLOCK_SIGNPOST_END:"

STAGE = "PERF_STAGE_SIGNPOST:"

# Emitted by caller_stacks per block ENTRY and EXIT, which is why it is a control token and not an
# op: on a 32-layer stack it is 64 entries that no device ever ran. caller_stacks imports this name
# as its MARK, so the string is written once and the module that writes it still owns its meaning.
CALLER = "PERF_CALLER:"


def control_prefixes() -> tuple:
    """Every prefix that marks structure rather than work, in the order a reader would try them."""
    return (BLOCK, BLOCK_END, STAGE, CALLER)


def is_control(tok) -> bool:
    """Is this entry bookkeeping the probe wrote, rather than an op the device ran?

    The op filter used to name ONE prefix, so every control token invented later was silently
    counted as work. That matters more than it sounds: the work signal is what decides INERT, so
    markers would inflate the very number used to prove a depth cap did nothing -- an uncappable
    model could look cappable purely from its own instrumentation. Caller markers are emitted per
    block ENTRY and EXIT, which on a 32-layer stack is 64 phantom ops.
    """
    return isinstance(tok, str) and tok.startswith(control_prefixes())
