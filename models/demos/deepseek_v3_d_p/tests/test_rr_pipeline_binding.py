# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the [RING,RING] 4x4 prefill bindings: no devices, no ttnn.

Covers the split-ring rank table (generate_rank_bindings.py), the binding templates and descriptors for the
SC4 and two-Galaxy pipelines, and rr_pipeline_binding.py's torus search on a synthetic pod.
"""
import collections
import itertools
import re
from pathlib import Path

import pytest
import yaml

from models.demos.deepseek_v3_d_p.utils import rr_pipeline_binding as rr

REPO = Path(__file__).resolve().parents[4]
TOPO = REPO / "models/demos/common/prefill/runners/topology_configuration"
HOSTS = ["h0", "h1", "h2", "h3"]

@pytest.fixture(autouse=True)
def _repo_root(monkeypatch):
    # Descriptor and template paths are repo-relative, as on the command line.
    monkeypatch.chdir(REPO)


# --- rank table -----------------------------------------------------------------------------------


def test_sc4_rank_table_is_mesh_major_with_two_ranks_per_odd_stage():
    table = rr.grb.build_split_ring_rank_table(HOSTS, 8)
    assert [(m, r) for m, r, _, _ in table] == [(0, 0), (1, 0), (1, 1), (2, 0), (3, 0), (3, 1)] + [
        (4, 0),
        (5, 0),
        (5, 1),
        (6, 0),
        (7, 0),
        (7, 1),
    ]
    # Even stage: the host's own torus (slices 1+2). Odd stage: host k's slice 3 and host k+1's slice 0.
    assert table[0] == (0, 0, "h0", [1, 2])
    assert table[1] == (1, 0, "h0", [3])
    assert table[2] == (1, 1, "h1", [0])
    assert table[-1] == (7, 1, "h0", [0])


def test_two_galaxy_rank_table_is_torus_seam_torus():
    assert rr.grb.build_split_ring_rank_table(["a", "b"], 3) == [
        (0, 0, "a", [1, 2]),
        (1, 0, "a", [3]),
        (1, 1, "b", [0]),
        (2, 0, "b", [1, 2]),
    ]


def test_quad_split_table_is_the_eight_stage_split_ring_table():
    assert rr.grb.build_quad_split_rank_table(HOSTS) == rr.grb.build_split_ring_rank_table(HOSTS, 8)


@pytest.mark.parametrize("hosts, stages", [(HOSTS, 9), (["h0"], 2), (["h0", "h1"], 5)])
def test_rank_table_rejects_stages_that_do_not_fit(hosts, stages, expect_error):
    with expect_error(SystemExit, "1"):
        rr.grb.build_split_ring_rank_table(hosts, stages)


def test_every_stage_covers_16_distinct_chips_and_each_slice_is_used_once_per_host():
    table = rr.grb.build_split_ring_rank_table(HOSTS, 8)
    per_stage = {}
    per_host = {}
    for mesh, _, host, slices in table:
        per_stage[mesh] = per_stage.get(mesh, 0) + 8 * len(slices)
        per_host.setdefault(host, []).extend(slices)
    assert set(per_stage.values()) == {16}
    # Each Galaxy contributes its torus (1, 2) and one slice-3 and one slice-0 half to two seams.
    assert all(sorted(s) == [0, 1, 2, 3] for s in per_host.values())


# --- templates and descriptors --------------------------------------------------------------------

TEMPLATES = {
    "pipeline_prefill_request_sc4_rr_12rank_4x4_sp1.yaml": (8, "ci/sc4_4x4_rr_mgd.textproto"),
    "pipeline_prefill_request_sc2_rr_4rank_4x4_sp1.yaml": (3, "ci/sc2_4x4_rr_mgd.textproto"),
}


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_template_ranks_follow_the_rank_table(name):
    stages, mgd = TEMPLATES[name]
    binding = yaml.safe_load((TOPO / name).read_text())
    table = rr.grb.build_split_ring_rank_table(["a", "b", "c", "d"], stages)
    assert [(b["rank"], b["mesh_id"], b["mesh_host_rank"]) for b in binding["rank_bindings"]] == [
        (i, m, h) for i, (m, h, _, _) in enumerate(table)
    ]
    assert binding["mesh_graph_desc_path"].endswith(mgd)
    assert (REPO / binding["mesh_graph_desc_path"]).exists()
    assert binding["global_env"]["PREFILL_FABRIC_MODE"] == "2d_torus_xy"
    assert binding["global_env"]["PREFILL_SP_AXIS"] == "1"


def test_fill_rank_bindings_addresses_ranks_by_mesh_and_host_rank(expect_error):
    text = (TOPO / "pipeline_prefill_request_sc2_rr_4rank_4x4_sp1.yaml").read_text()
    table = rr.grb.build_split_ring_rank_table(["a", "b"], 3)
    filled = rr.fill_rank_bindings(
        text, [["1", "2"], ["3"], ["4", "5"], ["6"]], "t", mesh_ranks=[(m, h) for m, h, _, _ in table]
    )
    got = {b["rank"]: b["env_overrides"]["TT_VISIBLE_DEVICES"] for b in yaml.safe_load(filled)["rank_bindings"]}
    assert got == {0: "1,2", 1: "3", 2: "4,5", 3: "6"}
    with expect_error(SystemExit, "does not have exactly one rank"):
        rr.fill_rank_bindings(text, [["1"]] * 4, "t", mesh_ranks=[(0, 0), (1, 0), (1, 0), (2, 0)])


def mesh_graph(path):
    text = Path(path).read_text()
    hosts = dict(
        re.findall(
            r'name: "(M\d)"\s+arch: BLACKHOLE\s+device_topology \{[^}]*\}\s+host_topology\s+\{ dims: \[ (\d), 1 \]',
            text,
        )
    )
    return {
        "meshes": re.findall(r'instances \{ mesh \{ mesh_descriptor: "(M\d)" mesh_id: (\d+)', text),
        "host_dims": hosts,
        "dim_types": set(re.findall(r"dim_types: \[ (\w+), (\w+) \]", text)),
        "seams": re.findall(r"channels \{ count: (\d+) policy: RELAXED \}\s+assign_z_direction: true", text),
        "intra_mesh": re.findall(r"channels \{ count: ([1-7]) policy: (\w+) \}", text),
    }


@pytest.mark.parametrize("name, stages", [("sc4_4x4_rr", 8), ("sc2_4x4_rr", 3)])
def test_descriptors_alternate_single_and_two_host_rings(name, stages):
    g = mesh_graph(TOPO / f"ci/{name}_mgd.textproto")
    assert g["dim_types"] == {("RING", "RING")}
    assert g["host_dims"] == {"M0": "1", "M1": "2"}
    assert g["meshes"] == [(f"M{i % 2}", str(i)) for i in range(stages)]
    # A chain: one 8-link seam between consecutive stages, nothing closing the ring.
    assert g["seams"] == ["8"] * (stages - 1)
    # Both mesh kinds ask for 2 channels per edge.
    assert g["intra_mesh"] == [("2", "STRICT")] * 2


def test_pins_name_both_board_revisions_and_the_right_meshes():
    for name, even, odd in [("sc4_4x4_rr", "0,2,4,6", "1,3,5,7"), ("sc2_4x4_rr", "0,2", "1")]:
        text = (TOPO / f"ci/{name}_mgd.textproto").read_text()
        pinned = set(re.findall(r'mesh_id_regex: "([\d,]+)"', text))
        assert pinned == {even, odd}
        assert "BH_REV_AB" in text and "BH_REV_C" in text


def test_existing_tray_half_configs_are_still_there():
    for name in ("ci/sc4_4x4_sp1_mgd.textproto", "pipeline_prefill_request_sc4_8rank_4x4_sp1.yaml"):
        assert (TOPO / name).exists()
    assert "[ LINE, RING ]" in (TOPO / "ci/sc4_4x4_sp1_mgd.textproto").read_text()


# --- descriptor model -----------------------------------------------------------------------------


def synthetic_torus(extra_edges=(), links=2, weak=None):
    """Counts for a 4x4 torus on chips (0, 0)..(0, 15), `links` per edge; `weak` edge gets 1."""
    counts = {}
    for r in range(4):
        for c in range(4):
            a = (0, r * 4 + c)
            for b in ((0, r * 4 + (c + 1) % 4), (0, ((r + 1) % 4) * 4 + c)):
                counts[frozenset((a, b))] = links
    if weak:
        counts[frozenset(weak)] = 1
    counts.update({frozenset(e): 2 for e in extra_edges})
    return counts, {(0, i) for i in range(16)}


def test_find_torus_finds_a_torus_and_reports_its_weakest_edge():
    counts, chips = synthetic_torus()
    grid, min_links = rr.best_torus(chips, counts)
    assert min_links == 2 and sorted(c for row in grid for c in row) == sorted(chips)
    counts, chips = synthetic_torus(weak=((0, 0), (0, 1)))
    assert rr.best_torus(chips, counts)[1] == 1
    assert rr.find_torus(chips, counts, min_links=2) is None


def test_find_torus_rejects_a_ring_of_16_and_a_missing_wrap():
    counts, chips = synthetic_torus()
    del counts[frozenset(((0, 0), (0, 3)))]
    assert rr.best_torus(chips, counts) == (None, 0)
    line = {frozenset(((0, i), (0, (i + 1) % 16))): 2 for i in range(16)}
    assert rr.best_torus(chips, line) == (None, 0)


class FakeGalaxy:
    """The parts of assemble_pipeline_binding.Galaxy that the torus search reads: 32 chips, logical id == chip id."""

    def __init__(self, name, index):
        self.name = name
        self.logical = {c: c for c in range(32)}
        self.unique = {c: 1000 * index + c for c in range(32)}
        self.links = collections.Counter()
        self.remote = collections.Counter()


# slice id -> logical ids, in the shape the discovery gtest writes it
FAKE_SLICES = {s: set(range(8 * s, 8 * s + 8)) for s in range(4)}


def fake_pod():
    """Two Galaxies cabled h0 -> h1: each has an in-Galaxy torus on slices 1+2, and h0's slice 3 joins h1's slice 0
    as a seam torus (rows 0-1 on h0, rows 2-3 on h1), 2 links on every torus edge."""
    h0, h1 = FakeGalaxy("h0", 0), FakeGalaxy("h1", 1)
    for g in (h0, h1):
        for r, c in itertools.product(range(4), range(4)):
            for dr, dc in ((0, 1), (1, 0)):
                g.links[frozenset((8 + r * 4 + c, 8 + ((r + dr) % 4) * 4 + (c + dc) % 4))] = 2

    def seam_chip(cell):
        return (h0, 24 + cell) if cell < 8 else (h1, cell - 8)

    for r, c in itertools.product(range(4), range(4)):
        for dr, dc in ((0, 1), (1, 0)):
            (ga, a), (gb, b) = seam_chip(r * 4 + c), seam_chip(((r + dr) % 4) * 4 + (c + dc) % 4)
            if ga is gb:
                ga.links[frozenset((a, b))] = 2
            else:
                ga.remote[(a, gb.unique[b])] = 2
                gb.remote[(b, ga.unique[a])] = 2
    return h0, h1


def test_check_finds_tori_on_every_stage_and_the_cabling_direction():
    h0, h1 = fake_pod()
    table = rr.grb.build_split_ring_rank_table(["h0", "h1"], 3)
    rows, _ = rr.stage_report([h0, h1], table, FAKE_SLICES)
    assert [(kind, links) for _, kind, _, links in rows] == [("in-Galaxy torus", 2), ("seam torus", 2), ("in-Galaxy torus", 2)]
    found = rr.ring_orders([h1, h0], 3, FAKE_SLICES)
    assert [[g.name for g in order] for _, order in found] == [["h0", "h1"]]


def test_slice_map_is_read_from_the_discovery_output(tmp_path):
    path = tmp_path / "slice_to_pcie_device_mapping.yaml"
    path.write_text("device_mapping:\n  hostA:\n    0: [0, 1]\n    3: [6, 7]\n  hostB:\n    0: [9]\n")
    assert rr.load_slices(path) == {0: {0, 1}, 3: {6, 7}}
