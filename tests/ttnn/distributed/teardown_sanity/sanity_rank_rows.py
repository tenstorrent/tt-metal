# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Helpers the tt-metal rank rows (test_context_py.py, test_mesh_py.py) need beyond the shared harness in
tests/tt_metal/distributed/multiprocess/teardown_sanity/sanity_common.py: the new-mode launch line (a solve first), a
one-galaxy descriptor to make two ranks disagree, and a per-rank error-line check."""

from __future__ import annotations

from pathlib import Path

import sanity_common
from sanity_common import MAPPING, MGD, ROOT, TTRUN, Result

# a one-galaxy Blackhole descriptor: as rank 1's descriptor it disagrees with rank 0's; alone it is a mock with no device
CONFLICT_DESC = (
    "tt_metal/third_party/tt-cluster-descriptors/blackhole/single_bh_galaxy_clus_desc/single_bh_galaxy_clus_desc.yaml"
)
MGD_QUAD = ROOT / "tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto"


def conflict_mapping(tmp: Path) -> Path:
    """The dual-galaxy mapping with rank 1's descriptor replaced by a different Blackhole machine: two ranks that
    disagree."""
    text = MAPPING.read_text()
    lines = [(f'  1: "{CONFLICT_DESC}"' if line.strip().startswith('1: "') else line) for line in text.splitlines()]
    p = tmp / "conflict_mapping.yaml"
    p.write_text("\n".join(lines) + "\n")
    return p


def rc_contains(r: Result, rank: int, needle: str) -> None:
    """Rank ``rank`` reported an error line containing ``needle`` (the harness keeps each rank's non-success lines)."""
    lines = r.rcs.get(rank, [])
    assert any(
        needle in x for x in lines
    ), f"rank {rank} did not report {needle!r}; it reported {lines}\n{r.log[-3000:]}"


def solve_cmd(mpi_extra: str = "", mgd: Path = MGD) -> list:
    """tt-run in new mode: phase 1 (generate_rank_bindings on the mock descriptors) then the ranks."""
    return [
        TTRUN,
        "--mesh-graph-descriptor",
        str(mgd),
        "--mock-cluster-rank-binding",
        str(MAPPING),
        "--mpi-args",
        ("--allow-run-as-root --oversubscribe " + mpi_extra).strip(),
        "--force-rediscovery",
        *sanity_common.probe_program(),  # the suite's rank program
    ]
