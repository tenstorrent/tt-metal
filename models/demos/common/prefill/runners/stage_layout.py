# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""How the distributed ranks of a pipelined prefill run map onto pipeline stages.

A stage is one mesh. Usually one host (one rank) owns it, so rank == stage. A [RING,RING] seam torus spans two
neighbouring Galaxies: one stage, two ranks (mesh_host_rank 0 and 1 of the same mesh_id). Ranks are numbered
mesh-major, host-rank-minor, the order rr_pipeline_binding.py writes them in.

The host split of a stage cuts mesh axis 0 into contiguous row blocks, one per host (host_topology dims
[hosts, 1] in the mesh graph descriptor). Mesh coordinates keep the same (row, col) across a stage boundary,
so a boundary's D2D channels pair the ranks whose row blocks overlap.

ttnn-free, so unit-testable on a host without a device.
"""

import os
from bisect import bisect_right
from dataclasses import dataclass

STAGE_HOSTS_ENV = "PREFILL_STAGE_HOSTS"


@dataclass(frozen=True)
class BoundaryPair:
    """One D2D service across a stage boundary: ``sender_rank`` and ``receiver_rank`` connect the mesh
    coordinates whose axis-0 index is in ``rows`` (half-open) and every column."""

    sender_rank: int
    receiver_rank: int
    rows: tuple


@dataclass(frozen=True)
class StageLayout:
    """``hosts_per_stage[s]`` is the number of ranks (hosts) of stage ``s``."""

    hosts_per_stage: tuple

    def __post_init__(self):
        if not self.hosts_per_stage or any(h < 1 for h in self.hosts_per_stage):
            raise ValueError(f"every stage needs at least one host, got {self.hosts_per_stage}")

    @classmethod
    def single_host(cls, num_ranks: int) -> "StageLayout":
        return cls((1,) * num_ranks)

    @classmethod
    def from_env(cls, num_ranks: int, environ=None) -> "StageLayout":
        """``PREFILL_STAGE_HOSTS`` ("1,2,1") lists each stage's host count; unset means one host per stage."""
        raw = (os.environ if environ is None else environ).get(STAGE_HOSTS_ENV, "").strip()
        if not raw:
            return cls.single_host(num_ranks)
        layout = cls(tuple(int(field) for field in raw.split(",")))
        if layout.num_ranks != num_ranks:
            raise ValueError(
                f"{STAGE_HOSTS_ENV}={raw!r} describes {layout.num_ranks} ranks but the run has {num_ranks}"
            )
        return layout

    @property
    def num_stages(self) -> int:
        return len(self.hosts_per_stage)

    @property
    def num_ranks(self) -> int:
        return sum(self.hosts_per_stage)

    @property
    def is_single_host(self) -> bool:
        return all(h == 1 for h in self.hosts_per_stage)

    def first_rank(self, stage: int) -> int:
        return sum(self.hosts_per_stage[:stage])

    def ranks_of(self, stage: int) -> range:
        first = self.first_rank(stage)
        return range(first, first + self.hosts_per_stage[stage])

    def stage_of(self, rank: int) -> int:
        if not 0 <= rank < self.num_ranks:
            raise ValueError(f"rank {rank} outside 0..{self.num_ranks - 1}")
        return bisect_right(self.leader_ranks(), rank) - 1

    def host_rank_of(self, rank: int) -> int:
        """The rank's mesh_host_rank within its stage."""
        return rank - self.first_rank(self.stage_of(rank))

    def is_leader(self, rank: int) -> bool:
        """The one rank per stage that speaks for it in stage-wide collectives and reports (mesh_host_rank 0)."""
        return self.host_rank_of(rank) == 0

    def leader_ranks(self) -> list:
        return [self.first_rank(stage) for stage in range(self.num_stages)]

    def host_rows(self, rank: int, mesh_rows: int) -> tuple:
        """Half-open range of mesh axis-0 rows the rank's host owns."""
        stage = self.stage_of(rank)
        hosts = self.hosts_per_stage[stage]
        if mesh_rows % hosts:
            raise ValueError(f"stage {stage}: {mesh_rows} mesh rows do not split over {hosts} hosts")
        per_host = mesh_rows // hosts
        host = self.host_rank_of(rank)
        return host * per_host, (host + 1) * per_host

    def boundary_pairs(self, sender_stage: int, mesh_rows: int) -> list:
        """The D2D services across the boundary from ``sender_stage`` to the next stage, ordered by first row."""
        receiver_stage = sender_stage + 1
        if receiver_stage >= self.num_stages:
            raise ValueError(f"stage {sender_stage} is the last stage and has no outbound boundary")
        pairs = []
        for sender in self.ranks_of(sender_stage):
            s_lo, s_hi = self.host_rows(sender, mesh_rows)
            for receiver in self.ranks_of(receiver_stage):
                r_lo, r_hi = self.host_rows(receiver, mesh_rows)
                lo, hi = max(s_lo, r_lo), min(s_hi, r_hi)
                if lo < hi:
                    pairs.append(BoundaryPair(sender, receiver, (lo, hi)))
        return sorted(pairs, key=lambda p: p.rows)

    def inbound_pairs(self, rank: int, mesh_rows: int) -> list:
        """The boundary services this rank receives on; empty on the first stage."""
        stage = self.stage_of(rank)
        if stage == 0:
            return []
        return [p for p in self.boundary_pairs(stage - 1, mesh_rows) if p.receiver_rank == rank]

    def outbound_pairs(self, rank: int, mesh_rows: int) -> list:
        """The boundary services this rank sends on; empty on the last stage."""
        stage = self.stage_of(rank)
        if stage == self.num_stages - 1:
            return []
        return [p for p in self.boundary_pairs(stage, mesh_rows) if p.sender_rank == rank]


def collapse_to_stages(gathered: list, layout: StageLayout | None, mesh_rows: int) -> list:
    """One entry per pipeline stage from the per-rank entries an all-gather returns, each carrying the host that
    owns every chip.

    Every host of a multi-host stage gathers the same layer range and the stage's whole-mesh fabric node ids, so
    keeping all entries would list that range once per host. The stage's first rank's entry is kept, with
    ``chip_host_tags[row][col]`` added: the ``host_tag`` of the rank whose rows contain mesh row ``row``. A stage
    spanning two hosts thus keeps a per-chip host, not one host for the stage. ``layout`` None: every rank is a
    stage and the entries are returned as they are."""
    if layout is None:
        return gathered
    collapsed = []
    for entries in gathered:
        by_rank = {entry["rank"]: entry for entry in entries}
        stage_entries = []
        for stage in range(layout.num_stages):
            leader = dict(by_rank[layout.first_rank(stage)])
            cols = len(leader["fnids"][0])
            row_tags = {}
            for rank in layout.ranks_of(stage):
                lo, hi = layout.host_rows(rank, mesh_rows)
                row_tags.update({row: by_rank[rank]["host_tag"] for row in range(lo, hi)})
            leader["chip_host_tags"] = [[row_tags[row]] * cols for row in range(mesh_rows)]
            stage_entries.append(leader)
        collapsed.append(stage_entries)
    return collapsed


def chip_host_tag(stage_entry: dict, row: int, col: int) -> int:
    """The host tag of chip (row, col) of a gathered stage: its own when the stage records one per chip."""
    tags = stage_entry.get("chip_host_tags")
    return stage_entry["host_tag"] if tags is None else tags[row][col]


def single_peer(pairs: list, mesh_rows: int, direction: str, rank: int) -> int | None:
    """The one peer rank of a boundary side, None when the rank has no such boundary.

    One D2D service on a whole stage mesh connects one sender rank to one receiver rank over every mesh
    coordinate. A boundary where a rank's mesh coordinates are split across two peer ranks (a one-host
    stage next to a two-host seam stage) needs a service per row block, built on a submesh of that block."""
    if not pairs:
        return None
    peers = {(p.receiver_rank if direction == "outbound" else p.sender_rank) for p in pairs}
    covered = sum(hi - lo for lo, hi in (p.rows for p in pairs))
    if len(pairs) > 1 or covered != mesh_rows or len(peers) != 1:
        raise NotImplementedError(
            f"[pp rank {rank}] {direction} boundary needs {len(pairs)} D2D services over rows "
            f"{[p.rows for p in pairs]} of {mesh_rows}: D2DStreamService connects whole meshes rank to rank, "
            "so a boundary between stages with different host counts needs one service per host row block, on a submesh "
            "(tests/ttnn/distributed/test_rr_prefill_d2d_seam.py), and the runner does not yet move activations "
            "between the stage mesh and those submeshes"
        )
    return peers.pop()
