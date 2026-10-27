# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Torch references for Kimi K3 attention residuals (AttnRes), in three files.

A **read** is the one thing AttnRes changes. A plain transformer hands each module
the residual stream itself — one running sum every module output is folded into.
AttnRes cuts that sum into 12-layer blocks and hands the module a *read* instead: a
softmax mixture of the current block's partial sum (``running_sum``) with one summed
representation per earlier block (``block_residual``, one entry appended per
**seal**), weighted by RMS-normalized scores against a per-site learned query.
``running_sum`` restarts from zero at every seal, so it only ever carries its own
block; the first sealed entry is the token embedding, which no block contributes to.
Writing is untouched — a module's output is still added into ``running_sum`` with
weight one. A 93-layer stack performs 186 reads: one before attention and one before
the MLP in every layer, plus one before the final norm, less layer 0's pre-attention
read, which has nothing sealed to mix and is skipped. The **candidates** of a read
are the tensors it mixes: the ``S`` sealed blocks plus the live partial sum,
``S + 1`` in all — 9 at the last read, since 93 layers give the embedding, seven
whole blocks, and a final block of nine.

The block size is not a tuning knob: it is ``attn_res_block_size`` in the checkpoint
config, and the query weights are trained against that partition.

``hf_attn_res.py`` is HuggingFace's ``_apply_attn_res``, vendored byte-identical
under ``LICENSE-Kimi-K3``. It is the one read here that nobody on this side wrote,
and so the only evidence that the definition was understood correctly. It computes
one softmax over the whole candidate set, against unfolded norm and projection
weights, so it can express neither the split the device implements nor the folded
query it holds — hence the two files below rather than this one alone.

``hf_walk.py`` drives that vendored read over a whole stack: which layers seal,
which reads see how many candidates, when the live stream is absent. The schedule
is the other thing no HuggingFace function exposes, because it lives in the model's
layer loop rather than in the read.

``attn_res.py`` is the folded torch form the device modules mirror: one query
carrying ``res_norm.weight * res_proj.weight`` and ``rsqrt`` pulled out of the dot.
It holds the read twice — ``attn_res`` reads the whole candidate set at once, while
``attn_res_inter_block`` + ``attn_res_merge`` split it the way the device op is
structured, scoring the sealed half separately from the live stream. Its walk drives
the one-shot form; the split is checked against that form rather than used by it.
Every device PCC gate is measured against this file.

So the schedule is written twice — ``hf_walk.HfStream`` and
``attn_res.AttnResStream`` — and that is deliberate. The two differ in exactly two
places: which read they call, and whether the query arrives folded or as the two
weights the checkpoint stores. ``seal``, ``accumulate`` and ``num_sealed`` are
identical in both, so the pair does not gate the schedule against itself. What it
gates is that the folded query equals ``norm * proj`` at every site of a whole stack
rather than at the single read the other gates cover. Collapsing the two into one class
parametrized by a read function would leave ``test_stack_matches_hf_walk`` comparing a
walk to itself. The schedule the pair shares is instead pinned to a written-down trace,
which catches a boundary that moves but not one that was read wrong to begin with.

``tests/`` beside these files is where all three shortcuts — the folded query, the
hoisted ``rsqrt``, the split read — are checked against forms that do not share
them. No device, milliseconds, and none of production's shapes: what is under test
is algebra and scheduling, neither of which the shape reaches.
"""
