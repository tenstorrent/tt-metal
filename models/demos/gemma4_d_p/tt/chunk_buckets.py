# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Choose a prefill chunk width per request from its prompt length.

Prefill time for one request splits into a cost paid once per chunk and a cost that grows
with the context preceding each chunk::

    T(P, C) = N * a(C) + slope(C) * N*(N-1)/2,   N = ceil(P / C)

``a(C)`` is strongly concave -- one 8192-token chunk is cheaper than four 2048-token ones for
the same tokens -- so a wider chunk is always better *for tokens that actually fill it*. The
whole case for a narrow bucket is the ``ceil``: a 4096-token prompt in a 32768-wide chunk pays
for 32768 tokens of attention and MLP. That is the term a per-request width removes.

Consequence: this is a per-request choice, pinned for the request's lifetime. The ring KV
layout is block-cyclic with period ``C`` (``ring_prefill``: local row ``chunk*L + j`` on rank
``r`` holds global token ``chunk*C + r*L + j``), and ring_joint SDPA reconstructs global
positions from the *current* chunk's width. Two requests at two widths are fine -- each owns
its cache slot -- but changing width inside one request silently reads the prefix at the wrong
positions.
"""

from __future__ import annotations

import math

# Measured on a BH Galaxy 8x4 (CP8/TP4), Gemma4-31B, branch kmabee/gemma4-swa-multihop-halo.
# (a, slope) in ms: `a` is the chunk-0 device time (== TTFT for a prompt that fits one chunk),
# `slope` the extra device time per preceding chunk. Both come from fitting the demo's own
# [traced_perf] per-chunk times; see tech_reports/Gemma4PrefillChunkSize/README.md, which
# reproduces every measured total to within 1%.
#
# These are one mesh shape on one branch. Re-measure with the chunk sweep before trusting them
# on another; the shape of the answer (concave `a`, steep `slope`) is what the selector needs,
# not the exact constants.
PREFILL_CHUNK_COST_MS = {
    2048: (131.3, 1.4584),
    4096: (174.2, 2.9966),
    8192: (242.7, 11.9831),
    16384: (443.7, 37.0900),
    32768: (928.2, 126.2286),
}


def modelled_prefill_ms(prompt_len, chunk_size, cost_table=None):
    """Modelled device time to prefill ``prompt_len`` tokens at ``chunk_size``.

    The prompt is padded up to whole chunks, which is where a too-wide bucket loses.
    """
    cost_table = PREFILL_CHUNK_COST_MS if cost_table is None else cost_table
    if chunk_size not in cost_table:
        raise KeyError(
            f"no measured cost for chunk {chunk_size}; measured widths are "
            f"{sorted(cost_table)}. Run the chunk sweep and add a row rather than "
            f"interpolating -- a(C) is concave and extrapolates badly."
        )
    if prompt_len <= 0:
        raise ValueError(f"prompt_len must be positive, got {prompt_len}")
    a, slope = cost_table[chunk_size]
    n = math.ceil(prompt_len / chunk_size)
    return n * a + slope * n * (n - 1) / 2


def select_chunk_size(prompt_len, chunk_sizes, cost_table=None):
    """Cheapest configured width for this prompt. Ties go to the narrower width."""
    if not chunk_sizes:
        raise ValueError("no prefill chunk widths configured")
    return min(sorted(chunk_sizes), key=lambda c: modelled_prefill_ms(prompt_len, c, cost_table))


def bucket_switch_points(chunk_sizes, max_prompt_len, cost_table=None, step=None):
    """Prompt lengths at which ``select_chunk_size`` changes its answer.

    Returns ``[(first_prompt_len, chunk_size), ...]`` covering ``[step, max_prompt_len]``, for
    logging the admission policy that a given bucket set actually implements. Sampled on a
    ``step`` grid (default: the narrowest width), so a boundary is accurate to one step.
    """
    widths = sorted(chunk_sizes)
    step = step or widths[0]
    plan, previous = [], None
    for prompt_len in range(step, max_prompt_len + 1, step):
        chosen = select_chunk_size(prompt_len, widths, cost_table)
        if chosen != previous:
            plan.append((prompt_len, chosen))
            previous = chosen
    return plan


def validate_chunk_sizes(chunk_sizes, cp_degree, max_seq_len):
    """Raise unless every configured width is a usable geometry. Returns them sorted."""
    from models.demos.gemma4_d_p.tt.model import prefill_chunk_geometry_error

    widths = tuple(sorted({int(c) for c in chunk_sizes}))
    if not widths:
        raise ValueError("no prefill chunk widths configured")
    for chunk_size in widths:
        error = prefill_chunk_geometry_error(chunk_size, cp_degree, max_seq_len)
        if error:
            raise ValueError(f"prefill chunk {chunk_size}: {error}")
    return widths
