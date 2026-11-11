# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Fabric traffic of MoE dispatch and combine along the dispatch axis, as a placement proxy.

``contributions(counts)`` takes counts[chunk, source_row, expert] (tokens of one dispatch group that a source
row sends to each expert of that group) and returns (dispatch, combine), each
[chunk, expert, owner_row, 4 * rows] payload pages per link: for every expert and every candidate owner row,
the pages that expert would put on each directed link if it lived on that row. The optimizer adds these up to
score link congestion of a placement without re-routing.

Counts exclude packet headers, acknowledgments and DRAM contention: these rank placements, they do not
predict time.

Models:
  ring    TtMoe "fabric2d" transport on a torus axis: each token takes the shorter way round, split over two
          streams per direction; at exactly half way round it is split over all four streams (matches
          dispatch_fabric2d_assignments.cpp / combine_fabric2d_assignments.cpp for 8 rows).
  linear  "direct" transport on a non-wrapping axis (FABRIC_2D without torus): straight from source to owner,
          split evenly over ``num_links`` parallel links.
"""

from __future__ import annotations

import numpy as np


def ring_links(count: int, source: int, destination: int, rows: int) -> np.ndarray:
    """Pages on [4 streams, rows departing-hop positions] for one source->destination transfer."""
    result = np.zeros((4, rows), dtype=np.int64)
    if source == destination:
        return result
    cw, ccw = (destination - source) % rows, (source - destination) % rows
    half = rows / 2
    for stream in range(4):
        positive = stream % 2 == 0
        distance = cw if positive else ccw
        if distance > half:
            continue
        splits = 4 if distance == half else 2
        share = stream if distance == half else stream // 2
        pages = count * (share + 1) // splits - count * share // splits
        for hop in range(distance):
            result[stream, (source + (hop if positive else -hop)) % rows] += pages
    return result


def ring_contributions(counts: np.ndarray):
    frames, rows, experts = counts.shape
    half = rows / 2
    dispatch = np.zeros((frames, experts, rows, 4 * rows), dtype=np.int64)
    combine = np.zeros_like(dispatch)
    for source in range(rows):
        values = counts[:, source, :]
        for owner in range(rows):
            if source == owner:
                continue
            for output, start, end in ((dispatch, source, owner), (combine, owner, source)):
                for stream in range(4):
                    direction = 1 if stream % 2 == 0 else -1
                    distance = ((end - start) * direction) % rows
                    if distance > half:
                        continue
                    splits = 4 if distance == half else 2
                    share = stream if distance == half else stream // 2
                    amount = values * (share + 1) // splits - values * share // splits
                    for hop in range(distance):
                        output[:, :, owner, stream * rows + (start + direction * hop) % rows] += amount
    return dispatch, combine


def linear_contributions(counts: np.ndarray, num_links: int = 2):
    frames, rows, experts = counts.shape
    if not 1 <= num_links <= 2:
        raise ValueError("linear model supports 1 or 2 links (4 stream slots)")
    dispatch = np.zeros((frames, experts, rows, 4 * rows), dtype=np.int64)
    combine = np.zeros_like(dispatch)
    for source in range(rows):
        values = counts[:, source, :]
        for owner in range(rows):
            if source == owner:
                continue
            for output, start, end in ((dispatch, source, owner), (combine, owner, source)):
                direction = 1 if end > start else -1
                for link in range(num_links):
                    amount = values * (link + 1) // num_links - values * link // num_links
                    stream = (0 if direction > 0 else 2) + link
                    for hop in range(abs(end - start)):
                        output[:, :, owner, stream * rows + start + direction * hop] += amount
    return dispatch, combine


TRANSPORTS = {"ring": lambda counts, num_links: ring_contributions(counts), "linear": linear_contributions}


def contributions(counts: np.ndarray, transport: str, num_links: int = 2):
    try:
        fn = TRANSPORTS[transport]
    except KeyError:
        raise ValueError(f"unknown transport {transport!r}; known: {sorted(TRANSPORTS)}") from None
    return fn(counts, num_links)
