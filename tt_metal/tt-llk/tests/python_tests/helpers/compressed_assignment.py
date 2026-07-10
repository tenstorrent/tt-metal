# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Realistic B-compression assignments for the face-compressed matmul test.

Both generators are driven by the SAME per-format statistics (tile fraction +
mean run length), so a face-vs-tile perf comparison is controlled — the only
thing that differs is the granularity at which precision is assigned:

* ``generate_tile_assignment`` samples a 32x32-TILE assignment from a first-order
  Markov chain with geometric runs matching ``shares`` and ``mean_run``. This is
  what a tile-granular saliency allocator produces. Feed it to the 16x16-face
  kernel via ``expand_tiles_to_faces`` (homogeneous 2x2 replication — every face
  of a tile shares its format).

* ``generate_face_assignment`` refines that same tile model to 16x16 faces:
  tiles stay homogeneous except for isolated sub-tile precision flips a
  face-granular allocator would carve out (a lone off-format face inside a
  uniform region). Their density is set by ``switch_mult`` = the ratio of face
  switches to the homogeneous (tile-granular) baseline, where a switch is any
  change between adjacent faces (all transitions counted equally — the kernel's
  per-switch cost is a downstream concern, not modelled here). ``switch_mult`` =
  1.0 reproduces the homogeneous expansion; 2.0 is the isotropic-field estimate
  (refining the quantization cell 32->16 crosses domain boundaries ~2x as often
  — stereology: crossings scale with cell edge); 3-4 is a vertical-boundary /
  zigzag-heavy upper bracket.

Default statistics are DeepSeek-R1 (v3-D) whole-weight-tensor tile stats
(``DEEPSEEK_R1``); ``mean_run`` is in units of 32x32 tiles. Supply your own
``CompressionStats`` to model a different distribution / run-length profile.
"""

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CompressionStats:
    """Per-precision compression profile, ordered high->low precision.

    ``shares``: tile fraction per precision (need not sum to exactly 1 — it is
    renormalized). ``mean_run``: mean contiguous run length in 32x32 tiles."""

    names: tuple
    shares: tuple
    mean_run: tuple

    def __post_init__(self):
        assert len(self.names) == len(self.shares) == len(self.mean_run) >= 1
        assert all(s >= 0 for s in self.shares) and sum(self.shares) > 0
        assert all(r >= 1.0 for r in self.mean_run), "mean_run is in tiles, must be >= 1"


# DeepSeek-R1 (v3-D, 3.5 b/e) whole-weight-tensor tile statistics: bfp4 baseline
# with scattered bfp2/zero demotions. Switch rate Sum(share/run) ~= 9.3% and avg
# section 1/that ~= 10.75 tiles both fall out of these two rows.
DEEPSEEK_R1 = CompressionStats(
    names=("bfp4", "bfp2", "zero"),
    shares=(0.629, 0.269, 0.103),
    mean_run=(18.51, 5.91, 7.54),
)


def transition_matrix(shares, mean_run):
    """Row-stochastic transition matrix whose stationary distribution == ``shares``
    and whose per-state mean run length == ``mean_run`` (geometric runs).

    Diagonals p_ii = 1 - 1/run set the run lengths; off-diagonals are fixed by
    detailed balance (reversible chain — no directional preference), which is
    unique for up to 3 states. For 3 states the reversible conductances are
    q_ij = R/2 - r_k (r_i = share_i/run_i, R = sum r_i, k the third state)."""
    pi = np.asarray(shares, dtype=float)
    pi = pi / pi.sum()
    L = np.asarray(mean_run, dtype=float)
    n = len(pi)
    r = pi / L
    R = r.sum()
    P = np.diag(1.0 - 1.0 / L)
    if n == 1:
        return np.array([[1.0]])
    if n == 2:
        P[0, 1], P[1, 0] = 1.0 / L[0], 1.0 / L[1]
        return P
    if n == 3:
        q = np.zeros((3, 3))
        q[0, 1] = q[1, 0] = R / 2 - r[2]
        q[0, 2] = q[2, 0] = R / 2 - r[1]
        q[1, 2] = q[2, 1] = R / 2 - r[0]
        assert (q >= -1e-9).all(), (
            "detailed balance infeasible: one class switches too often "
            "(max run rate exceeds half the total). Supply an explicit matrix."
        )
        for i in range(3):
            for j in range(3):
                if i != j:
                    P[i, j] = q[i, j] / pi[i]
        return P
    raise NotImplementedError("supply an explicit transition matrix for >3 states")


def _sample_markov(n, stats, rng):
    """Sample ``n`` precision indices (0..len-1, ordered as ``stats``) from the
    Markov chain. Run-based: draw a state, a geometric run length (mean run_i),
    emit it, then jump to a different state via the embedded jump chain."""
    shares = np.asarray(stats.shares, dtype=float)
    shares = shares / shares.sum()
    L = np.asarray(stats.mean_run, dtype=float)
    k = len(shares)
    P = transition_matrix(shares, L)
    jump = P.copy()
    np.fill_diagonal(jump, 0.0)
    row = jump.sum(axis=1, keepdims=True)
    jump = np.divide(jump, row, out=np.zeros_like(jump), where=row > 0)

    out = np.empty(n, dtype=np.int64)
    pos = 0
    s = int(rng.choice(k, p=shares))
    while pos < n:
        run = int(rng.geometric(min(1.0, 1.0 / L[s])))
        run = min(run, n - pos)
        out[pos : pos + run] = s
        pos += run
        if row[s] > 0:
            s = int(rng.choice(k, p=jump[s]))
    return out


def _tile_codes(kt_tiles, ct_tiles, codes, stats, rng):
    """Row-major (kt_tiles x ct_tiles) tile assignment as an int array of format
    ``codes`` (codes[i] is the format for precision level i in ``stats``)."""
    assert len(codes) == len(stats.shares), "one code per precision level"
    idx = _sample_markov(kt_tiles * ct_tiles, stats, rng)
    tiles = np.asarray(codes, dtype=np.int64)[idx]
    # An all-zero (bfp0) assignment yields an empty B buffer the kernel can't
    # consume; seed one lowest-non-zero tile, matching assign_tile_matched.
    nonzero = [c for c in codes if c != 0]
    if nonzero and not (tiles != 0).any():
        tiles[0] = min(nonzero)
    return tiles


def generate_tile_assignment(kt_tiles, ct_tiles, codes, stats=DEEPSEEK_R1, seed=0):
    """A realistic 32x32-TILE assignment (row-major, length kt_tiles*ct_tiles)
    sampled from ``stats``. Expand with ``expand_tiles_to_faces`` to run it on the
    face kernel as a tile-granular-allocation baseline."""
    rng = np.random.default_rng(seed)
    return _tile_codes(kt_tiles, ct_tiles, codes, stats, rng).tolist()


def expand_tiles_to_faces(tile_codes, kt_tiles, ct_tiles):
    """Homogeneous 2x2 expansion of a 32x32-tile assignment to 16x16 faces,
    row-major over (2*kt_tiles, 2*ct_tiles) — the layout run_compressed expects.
    Every face of a tile inherits the tile's format (tile-granular allocation)."""
    arr = np.asarray(tile_codes, dtype=np.int64).reshape(kt_tiles, ct_tiles)
    return arr.repeat(2, axis=0).repeat(2, axis=1).flatten().tolist()


def _switch_count(faces):
    """Number of format switches: adjacent faces that differ, in row-major face
    order. All transitions count equally — the generator models the assignment,
    not any kernel's per-switch cost (which transitions are cheap/free is a
    perf-analysis concern applied downstream, not baked in here)."""
    return int((faces[1:] != faces[:-1]).sum())


def _inject_interior_flips(faces, switch_mult, codes, rng):
    """In-place: flip isolated interior faces to a different format so the face
    switch count reaches ``switch_mult`` x the homogeneous baseline.

    Each flip lands on a face whose two neighbours share its format, so it becomes
    a lone different-format face (A A A -> A B A) adding exactly two switches — the
    sub-tile precision spot a face-granular allocator carves out. The replacement
    is drawn uniformly from the other formats, and flips are kept >2 apart so they
    stay isolated. Needs >=2 formats (nothing to switch between otherwise)."""
    if switch_mult <= 1.0 or len(codes) < 2:
        return
    baseline = _switch_count(faces)
    n_flips = int(round((switch_mult - 1.0) * baseline / 2.0))
    if n_flips <= 0:
        return
    interior = (faces[1:-1] == faces[:-2]) & (faces[1:-1] == faces[2:])
    cand = np.nonzero(interior)[0] + 1
    rng.shuffle(cand)
    used = []
    for i in cand:
        if len(used) >= n_flips:
            break
        if all(abs(int(i) - u) > 2 for u in used):  # keep each flip isolated
            other = [c for c in codes if c != faces[i]]
            faces[i] = int(rng.choice(other))
            used.append(int(i))


def generate_face_assignment(kt_tiles, ct_tiles, codes, stats=DEEPSEEK_R1, switch_mult=2.0, seed=0):
    """A realistic 16x16-FACE assignment refined from the same tile model as
    ``generate_tile_assignment`` (same ``seed`` => same tile substrate, so the
    face and tile runs are a controlled comparison). ``switch_mult`` is the ratio
    of face switches to the homogeneous expansion (a switch = any change between
    adjacent faces, all transitions equal): 1.0 reproduces ``expand_tiles_to_faces``
    exactly, 2.0 is the isotropic estimate. Row-major over (2*kt_tiles, 2*ct_tiles),
    length 4*kt_tiles*ct_tiles."""
    rng = np.random.default_rng(seed)
    tiles = _tile_codes(kt_tiles, ct_tiles, codes, stats, rng)
    faces = np.asarray(expand_tiles_to_faces(tiles.tolist(), kt_tiles, ct_tiles), dtype=np.int64)
    _inject_interior_flips(faces, switch_mult, codes, rng)
    return faces.tolist()


# ---------------------------------------------------------------------------
# Exact-count generator.
#
# The Markov sampler above reproduces ``shares`` only asymptotically: at small
# tile counts (kt*ct in the tens) a whole assignment is just a handful of runs,
# so the realized format mix swings wildly seed-to-seed and rarely matches
# ``shares`` (e.g. a shape may come out bfp2-dominated even though bfp4 is the
# majority format). For small shapes where you want the target proportions on
# the nose, use this generator instead: it plants EXACTLY round(share_i * n)
# tiles of each format (largest-remainder rounding, so counts sum to n) and only
# uses ``mean_run`` to decide the run LAYOUT, not the counts.
#
# What is preserved vs. the Markov chain: per-format tile fraction (now exact)
# and per-format mean run length. What is dropped: the chain's off-diagonal
# transition *preferences* (which format tends to follow which) — runs here are
# ordered only to keep adjacent runs of different formats where feasible, with no
# directional bias. For the controlled tile-vs-face comparison that matters:
# feed the tiles to ``expand_tiles_to_faces`` (tile-granular) or to
# ``generate_exact_face_assignment`` (face-granular), exactly as with the Markov
# generators.


def _exact_counts(n, shares):
    """Integer per-class counts summing to ``n``, proportional to ``shares``
    (largest-remainder / Hamilton rounding: floor everyone, then hand the leftover
    to the largest fractional parts)."""
    pi = np.asarray(shares, dtype=float)
    pi = pi / pi.sum()
    raw = pi * n
    counts = np.floor(raw).astype(np.int64)
    leftover = n - int(counts.sum())
    for c in np.argsort(-(raw - counts))[:leftover]:
        counts[c] += 1
    return counts


def _partition_into_runs(count, mean_run, rng):
    """Split ``count`` (>=1) items into runs each >=1, averaging ``mean_run`` with
    geometric-shaped lengths. Returns a list of run lengths summing exactly to
    ``count``."""
    if count <= 0:
        return []
    k = min(count, max(1, int(round(count / mean_run))))
    lengths = [max(1, int(rng.geometric(min(1.0, 1.0 / mean_run)))) for _ in range(k)]
    # Correct the total to `count` exactly, keeping every run >= 1.
    diff = count - sum(lengths)
    while diff != 0:
        j = int(rng.integers(k))
        if diff > 0:
            lengths[j] += 1
            diff -= 1
        elif lengths[j] > 1:
            lengths[j] -= 1
            diff += 1
    return lengths


def _order_runs(classes, rng):
    """Order a multiset of run classes so adjacent runs differ in class where
    feasible — greedy: always emit the class with the most remaining runs that
    isn't the previous one (ties broken randomly). If one class holds more than
    half the runs a same-class pair is forced, which simply merges those two runs
    into a longer one; the exact counts are unaffected either way."""
    remaining = Counter(classes)
    order = []
    prev = -1
    for _ in range(len(classes)):
        cands = [c for c in remaining if remaining[c] > 0 and c != prev]
        if not cands:
            cands = [c for c in remaining if remaining[c] > 0]
        top = max(remaining[c] for c in cands)
        pick = int(rng.choice([c for c in cands if remaining[c] == top]))
        order.append(pick)
        remaining[pick] -= 1
        prev = pick
    return order


def _exact_tile_indices(n, stats, rng):
    """Length-``n`` array of class indices (0..k-1, ordered as ``stats``) with
    EXACT per-class counts round(share_i * n) and geometric run lengths averaging
    ``mean_run_i``, laid out so adjacent runs differ in class where feasible."""
    counts = _exact_counts(n, stats.shares)
    L = np.asarray(stats.mean_run, dtype=float)

    run_lengths = {c: deque(_partition_into_runs(int(counts[c]), L[c], rng)) for c in range(len(counts))}
    classes = [c for c, lens in run_lengths.items() for _ in lens]

    out = np.empty(n, dtype=np.int64)
    pos = 0
    for c in _order_runs(classes, rng):
        length = run_lengths[c].popleft()
        out[pos : pos + length] = c
        pos += length
    assert pos == n
    return out


def _exact_tile_codes(kt_tiles, ct_tiles, codes, stats, rng):
    """``_tile_codes`` counterpart backed by the exact-count sampler."""
    assert len(codes) == len(stats.shares), "one code per precision level"
    idx = _exact_tile_indices(kt_tiles * ct_tiles, stats, rng)
    tiles = np.asarray(codes, dtype=np.int64)[idx]
    # An all-zero (bfp0) assignment yields an empty B buffer the kernel can't
    # consume; seed one lowest-non-zero tile, matching _tile_codes.
    nonzero = [c for c in codes if c != 0]
    if nonzero and not (tiles != 0).any():
        tiles[0] = min(nonzero)
    return tiles


def generate_exact_tile_assignment(kt_tiles, ct_tiles, codes, stats=DEEPSEEK_R1, seed=0):
    """Exact-count counterpart of ``generate_tile_assignment``: a 32x32-TILE
    assignment whose per-format counts are round(share_i * kt*ct) exactly (no
    sampling noise), with run lengths averaging ``stats.mean_run``. Expand with
    ``expand_tiles_to_faces`` to run it on the face kernel."""
    rng = np.random.default_rng(seed)
    return _exact_tile_codes(kt_tiles, ct_tiles, codes, stats, rng).tolist()


def generate_exact_face_assignment(kt_tiles, ct_tiles, codes, stats=DEEPSEEK_R1, switch_mult=2.0, seed=0):
    """Exact-count counterpart of ``generate_face_assignment``: refines the exact
    tile substrate to 16x16 faces with isolated sub-tile precision flips at density
    ``switch_mult`` (1.0 reproduces ``expand_tiles_to_faces`` of the exact tiles).
    Same ``seed`` => same tile substrate as ``generate_exact_tile_assignment``, so
    face vs tile stays a controlled comparison."""
    rng = np.random.default_rng(seed)
    tiles = _exact_tile_codes(kt_tiles, ct_tiles, codes, stats, rng)
    faces = np.asarray(expand_tiles_to_faces(tiles.tolist(), kt_tiles, ct_tiles), dtype=np.int64)
    _inject_interior_flips(faces, switch_mult, codes, rng)
    return faces.tolist()
