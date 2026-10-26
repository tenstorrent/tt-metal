# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Prefill grouping for the multi-user pack (P3d-4): which rows of one batched prefill call run in a pack and which alone.

Host only and pure: no device, no ttnn call (the cost table of the single path is read from the prefill driver).  The
adapter (``qwen38_vllm._prefill_batch``) asks :func:`plan_prefill_groups` for groups of row indices:

* ``("pack", S, [rows])``: one pack of bucket ``S`` (``S`` rows per user, at most ``2048 // S`` users) holding those rows;
* ``("single", [row])``: the B=1 chain prefill and the lane import, as without the pack.

Rule: only a row of at most ``max(buckets)`` tokens can pack; the bucket of a pack is the smallest enabled ``S`` that holds
its longest row.  The rows are sorted by (length, index); a pack takes a contiguous run of that order; a dynamic program
minimizes the estimated total time (:func:`pack_ms` and :func:`single_ms`).  Ties go to the single path.  ``force=True``
(``QWEN38_PACK_FORCE=1``, tests and accuracy runs only) packs every eligible row, even alone.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

from models.demos.blackhole.qwen38_flash_next.tools.qwen38_prefill_driver import PASS_COST_MS, plan_cover

PACK_FORCE_VARIABLE = "QWEN38_PACK_FORCE"
PACK_ROWS = 2048  # the slab the pack runs on (ttnn/pack.py PACK_ROWS)

# Pack cost in ms (P3d-3 timing.json, B=16 adapter, median of 3, 2026-10-26, 4x Blackhole QuietBox 2): the part that does
# not depend on the user count (host prep + upload + replay + logits read) and the landing per landed user.
#   S=128: 70.7 + 2.3 + 570.1 + 13.9 = 657.0, landing 93.1 / 16 users = 5.8
#   S=256: 71.2 + 2.3 + 560.9 + 14.0 = 648.4, landing 53.8 /  8 users = 6.7
#   S=512: 71.5 + 2.3 + 564.9 + 14.1 = 652.8, landing 35.4 /  4 users = 8.8
#   S=1024: 71.0 + 2.2 + 574.2 + 13.7 = 661.1, landing 24.2 / 2 users = 12.1
PACK_FIXED_MS: Mapping[int, float] = {128: 657.0, 256: 648.4, 512: 652.8, 1024: 661.1}
PACK_LAND_MS: Mapping[int, float] = {128: 5.8, 256: 6.7, 512: 8.8, 1024: 12.1}

# Single path, per request, on top of PASS_COST_MS of its cover (P3d-3 measurements, same run: P32 122 ms, P128 216 ms,
# P1024 766 ms): the fixed part (seed, hand-off, last step, import) is 60 ms (P32: one padded short pass 62 -> 122; P128: one
# padded long pass 156 -> 216), and a slab pass adds its non-overlapped host preparation of 58 ms (P1024: one padded slab
# pass 648 -> 766 = 648 + 58 + 60).  A head under CHUNK_HEAD_MIN rows is teacher forced token by token at STEP_MS per token
# (an estimate from the B=1 decode step, not a P3d-3 measurement).
SINGLE_FIXED_MS = 60.0
SINGLE_SLAB_HOST_MS = 58.0
SINGLE_STEP_MS = 26.0
CHUNK_HEAD_MIN = 16  # qwen38_chat_session.CHUNK_PREFILL_MIN_ROWS

Group = tuple


def pack_force_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """``QWEN38_PACK_FORCE`` ("0" default, "1" = every eligible row packs); anything else is refused."""

    value = (os.environ if environ is None else environ).get(PACK_FORCE_VARIABLE, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{PACK_FORCE_VARIABLE} must be 0 or 1, got {value!r}")
    return value == "1"


def bucket_users(rows_per_user: int) -> int:
    """Users of one pack of bucket ``rows_per_user``: ``2048 // S``."""

    return PACK_ROWS // rows_per_user


def pack_ms(rows_per_user: int, users: int) -> float:
    """Estimated wall of one pack of ``users`` users of bucket ``rows_per_user``."""

    return PACK_FIXED_MS[rows_per_user] + users * PACK_LAND_MS[rows_per_user]


def single_ms(
    length: int, *, context: int = 32768, slab_rows: int = PACK_ROWS, long_chunks: bool = True, costs=None
) -> float:
    """Estimated wall of one single-path request of ``length`` tokens (cover of its ``length - 1`` head rows)."""

    head = length - 1
    if head < CHUNK_HEAD_MIN:
        return SINGLE_FIXED_MS + head * SINGLE_STEP_MS
    cover = plan_cover(
        head,
        position=0,
        context=context,
        slab_rows=slab_rows,
        long_chunks=long_chunks,
        costs=PASS_COST_MS if costs is None else costs,
    )
    table = PASS_COST_MS if costs is None else costs
    total = SINGLE_FIXED_MS
    for spec in cover:
        total += table[spec.kind] * (spec.rows / 2048 if spec.kind == "slab" else 1.0)
        if spec.kind == "slab":
            total += SINGLE_SLAB_HOST_MS
    return total


def default_pack_buckets(lanes: int | None) -> tuple[int, ...]:
    """The buckets enabled when ``QWEN38_PACK_BUCKETS`` is unset: with at most 8 lanes ``S = 128`` could never fill above
    half (G = 16), so it is dropped (its second GDN launch buffers, 226.5 MiB, and its 74 MiB trace); more lanes: all four.
    ``lanes=None`` (unknown): all four."""

    return (256, 512, 1024) if lanes is not None and lanes <= 8 else (128, 256, 512, 1024)


def plan_prefill_groups(
    lengths: Sequence[int],
    buckets: Sequence[int],
    lanes: int,
    *,
    force: bool = False,
    context: int = 32768,
) -> list[Group]:
    """Groups of the rows ``0 .. len(lengths) - 1``, each row in exactly one group, in execution order (ascending length of
    the group's first row; a pack is listed where its first row sorts).  See the module docstring for the rule."""

    lengths = [int(n) for n in lengths]
    buckets = sorted({int(b) for b in buckets})
    if any(n < 1 for n in lengths):
        raise ValueError(f"prompt lengths must be positive, got {lengths}")
    if len(lengths) > lanes:
        raise ValueError(f"{len(lengths)} rows exceed the {lanes} lanes")
    top = buckets[-1] if buckets else 0
    eligible = sorted((i for i, n in enumerate(lengths) if n <= top), key=lambda i: (lengths[i], i))
    ineligible = sorted((i for i, n in enumerate(lengths) if n > top), key=lambda i: (lengths[i], i))
    groups: list[Group] = []
    if eligible:
        groups += _plan_eligible(eligible, lengths, buckets, force, context)
    groups += [("single", [i]) for i in ineligible]
    return groups


def _bucket_for(length: int, buckets: Sequence[int]) -> int:
    return next(b for b in buckets if b >= length)


def _plan_eligible(order: list[int], lengths: list[int], buckets: list[int], force: bool, context: int) -> list[Group]:
    count = len(order)
    single = [single_ms(lengths[i], context=context) for i in order]
    # best[k] = (cost, previous k, group) for the first k rows of the sorted order
    best: list[tuple[float, int, Group] | None] = [None] * (count + 1)
    best[0] = (0.0, 0, ())
    for k in range(1, count + 1):
        options: list[tuple[float, int, Group]] = []
        if not force:  # the single path first: ties keep it (the strict comparison below)
            options.append((best[k - 1][0] + single[k - 1], k - 1, ("single", [order[k - 1]])))
        for j in range(k - 1, -1, -1):  # the pack of the sorted run j .. k - 1
            size = k - j
            bucket = _bucket_for(lengths[order[k - 1]], buckets)  # the longest row of the run is its last
            if size > bucket_users(bucket):
                break
            options.append((best[j][0] + pack_ms(bucket, size), j, ("pack", bucket, [order[t] for t in range(j, k)])))
        chosen = options[0]
        for option in options[1:]:
            if option[0] < chosen[0]:
                chosen = option
        best[k] = chosen
    groups: list[Group] = []
    k = count
    while k > 0:
        _cost, previous, group = best[k]
        groups.append(group)
        k = previous
    groups.reverse()
    return groups


def plan_cost_ms(groups: Sequence[Group], lengths: Sequence[int], *, context: int = 32768) -> float:
    """The estimated total of ``groups`` (the DP's objective; for logs and tests)."""

    total = 0.0
    for group in groups:
        if group[0] == "pack":
            total += pack_ms(group[1], len(group[2]))
        else:
            total += single_ms(lengths[group[1][0]], context=context)
    return total
