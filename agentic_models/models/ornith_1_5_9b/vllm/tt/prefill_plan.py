# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure-host canonical prefill planning; no torch/TTNN or device side effects.

``total_length`` is the known logical request horizon, not cache capacity or the
current fragment end. Original chunks remain anchored at zero with independent
32768 linear / 16384 full-attention caps. A fragment carries the numerical profile
of each original whole-request chunk, including the linear511 / full512 split.
The runtime can reduce large-profile padding to8192 rows while retaining this
canonical boundary/profile identity; the version below identifies that policy.

An unaligned continuation requires saved layer INPUT rows from its preceding
32-token boundary and, for linear attention, conv/recurrent state at that same
boundary. Restore that state before recomputing the <=31-row overlap, then drop
overlap outputs. Full attention rewrites only the overlap K/V. These objects
must remain device tensors; this module merely describes their required spans.

Prefix state is profile-dependent. Compare profile tags before reuse; an unknown
future horizon does not establish compatibility. Tags are only a necessary
condition: token identity, weights, precision, mesh and cache ownership must
also match. This planner does NOT declare serving prefix-cache support. The
hybrid recurrent model uses the documented ``recurrent_state`` exclusion unless
the serving stack can preserve the required recurrent snapshots and input tails.
Within-request resumed/chunked prefill is independent of that declaration.
"""

from dataclasses import dataclass

MAX_CONTEXT = 262144
TILE_ROWS = 32
LINEAR_BUCKETS = (1, 32, 64, 128, 511, 512, 640, 768, 960, 1024, 4096, 8192, 32768)
FULL_PREFILL_BUCKETS = (32, 64, 128, 480, 512, 768, 1024, 4096, 8192, 16384)
CHUNK_CAPS = {"linear_attention": 32768, "full_attention": 16384}
PROFILE_VERSION = "ornith-tp4-normalized-prefill-v2-compact8192"


def _horizon(total_length):
    if isinstance(total_length, bool) or not isinstance(total_length, int) or not 1 <= total_length <= MAX_CONTEXT:
        raise ValueError(f"total_length must be an integer in [1,{MAX_CONTEXT}]")


def physical_bucket(kind, canonical_valid_count):
    """Select from a whole original chunk's count, never a fragment's count."""
    if kind not in CHUNK_CAPS:
        raise ValueError(f"Unsupported decoder kind: {kind}")
    if not isinstance(canonical_valid_count, int) or not 1 <= canonical_valid_count <= CHUNK_CAPS[kind]:
        raise ValueError("Canonical count exceeds this layer kind's original chunk cap")
    if kind == "linear_attention":
        buckets, required = LINEAR_BUCKETS, canonical_valid_count
    else:
        buckets = FULL_PREFILL_BUCKETS
        required = ((canonical_valid_count + TILE_ROWS - 1) // TILE_ROWS) * TILE_ROWS
    return next(n for n in buckets if n >= required)


@dataclass(frozen=True)
class ProfileTag:
    version: str
    kind: str
    chunk_start: int
    physical_bucket: int


@dataclass(frozen=True)
class BoundaryState:
    """Layer-local saved input tail and optional recurrent snapshot contract."""

    boundary: int
    tail_end: int
    profile_tag: ProfileTag
    requires_conv_recurrent: bool

    @property
    def tail_rows(self):
        return self.tail_end - self.boundary


@dataclass(frozen=True)
class PrefillSegment:
    canonical_chunk_start: int
    canonical_chunk_end: int
    profile_tag: ProfileTag
    absolute_aligned_start: int
    logical_start: int
    logical_end: int
    # Restore this before execution; discard initial_overlap rows of output.
    required_cached_boundary: BoundaryState | None = None
    # Snapshot before execution and retain input rows through tail_end. Its
    # boundary can equal required_cached_boundary.boundary for same-tile resumes.
    save_boundary_before: BoundaryState | None = None

    @property
    def physical_bucket(self):
        return self.profile_tag.physical_bucket

    @property
    def initial_overlap(self):
        return self.logical_start - self.absolute_aligned_start

    @property
    def valid_count(self):
        """Actual rows consumed by the adapter, INCLUDING cached overlap."""
        return self.logical_end - self.absolute_aligned_start


@dataclass(frozen=True)
class PrefillPlan:
    total_length: int
    start: int
    end: int
    kind: str
    segments: tuple[PrefillSegment, ...]


def plan_prefill(*, total_length, start, end, kind):
    """Plan [start,end) without changing the original per-kind chunk strategy.

    If a nonfinal fragment ends inside a tile, split before its last partial
    tile. The tail segment requests a snapshot at its aligned start, enabling
    the next fragment to replay a bounded overlap without double-advancing state.
    No split is added at a final request end; no future prefill needs that tail.
    """
    _horizon(total_length)
    if kind not in CHUNK_CAPS:
        raise ValueError(f"Unsupported decoder kind: {kind}")
    if not all(isinstance(n, int) and not isinstance(n, bool) for n in (start, end)):
        raise ValueError("Fragment boundaries must be integers")
    if not 0 <= start < end <= total_length:
        raise ValueError(f"Invalid fragment [{start},{end}) for horizon {total_length}")
    cap = CHUNK_CAPS[kind]
    save_at = end // TILE_ROWS * TILE_ROWS if end < total_length and end % TILE_ROWS else None
    segments = []
    cursor = start
    while cursor < end:
        chunk_start = cursor // cap * cap
        chunk_end = min(chunk_start + cap, total_length)
        bucket = physical_bucket(kind, chunk_end - chunk_start)
        tag = ProfileTag(PROFILE_VERSION, kind, chunk_start, bucket)
        aligned = cursor // TILE_ROWS * TILE_ROWS
        stop = min(end, chunk_end)
        if save_at is not None and cursor < save_at < stop:
            stop = save_at
        recurrent = kind == "linear_attention"
        needed = BoundaryState(aligned, cursor, tag, recurrent) if aligned < cursor else None
        saved = BoundaryState(aligned, end, tag, recurrent) if save_at == aligned and stop == end else None
        segment = PrefillSegment(chunk_start, chunk_end, tag, aligned, cursor, stop, needed, saved)
        assert 0 < segment.valid_count <= bucket
        assert 0 <= segment.initial_overlap < TILE_ROWS
        assert saved is None or 0 < saved.tail_rows < TILE_ROWS
        segments.append(segment)
        cursor = stop
    return PrefillPlan(total_length, start, end, kind, tuple(segments))


def prefix_profile_tags(*, total_length, prefix_end, kind):
    """Profiles that produced [0,prefix_end); no advertised cache capability."""
    _horizon(total_length)
    if kind not in CHUNK_CAPS or not isinstance(prefix_end, int) or not 0 <= prefix_end <= total_length:
        raise ValueError("Invalid prefix range or decoder kind")
    cap = CHUNK_CAPS[kind]
    return tuple(
        ProfileTag(PROFILE_VERSION, kind, begin, physical_bucket(kind, min(cap, total_length - begin)))
        for begin in range(0, prefix_end, cap)
    )


def prefix_profiles_compatible(*, cached_total_length, requested_total_length, prefix_end, kind):
    """Unknown horizons are incompatible; compatible profiles alone are insufficient."""
    if cached_total_length is None or requested_total_length is None:
        return False
    return prefix_profile_tags(
        total_length=cached_total_length, prefix_end=prefix_end, kind=kind
    ) == prefix_profile_tags(total_length=requested_total_length, prefix_end=prefix_end, kind=kind)
