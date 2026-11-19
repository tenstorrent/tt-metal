# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the [RING,RING] 4x4 prefill bindings: no devices, no ttnn.

Covers the split-ring rank table (generate_rank_bindings.py), the binding templates and descriptors for the
SC4 and two-Galaxy pipelines, and rr_pipeline_binding.py's descriptor model. Cases that read cabling
descriptors skip without the tt-cluster-descriptors submodule.
"""
import re
from pathlib import Path

import pytest
import yaml

from models.demos.deepseek_v3_d_p.utils import rr_pipeline_binding as rr
from models.demos.deepseek_v3_d_p.utils.assemble_pipeline_binding import Galaxy, load_mapping

REPO = Path(__file__).resolve().parents[4]
TOPO = REPO / "models/demos/common/prefill/runners/topology_configuration"
DESCRIPTORS = REPO / "tt_metal/third_party/tt-cluster-descriptors/superclusters/blackhole"
SC4_REVC_MAPPING = DESCRIPTORS / "SC20_32x4_revC_subtorus_aisleC/SC4_32x4_revC_subtorus_aisleC_mapping.yaml"
SC4_REVAB_MAPPING = DESCRIPTORS / "SC36_32x4_revAB_subtorus_aisleC/SC4_32x4_revAB_subtorus_aisleC_mapping.yaml"
SC4_REVAB_FLAT_MAPPING = DESCRIPTORS / "SC16_32x4_revAB_aisleD/SC4_32x4_revAB_aisleD_mapping.yaml"
HOSTS = ["h0", "h1", "h2", "h3"]

needs_descriptors = pytest.mark.skipif(
    not SC4_REVC_MAPPING.exists(), reason="tt-cluster-descriptors submodule not checked out"
)


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


def test_slice_positions_swap_trays_2_and_3_on_rev_c():
    assert {t for t, _ in rr.grb.slice_positions(3, rev_c=False)} == {2, 4}
    assert {t for t, _ in rr.grb.slice_positions(3, rev_c=True)} == {3, 4}
    assert {t for t, _ in rr.grb.slice_positions(0, rev_c=False)} == {1, 3}
    assert {t for t, _ in rr.grb.slice_positions(0, rev_c=True)} == {1, 2}
    assert {a for _, a in rr.grb.slice_positions(1, rev_c=True)} == {3, 4, 7, 8}
    assert {a for _, a in rr.grb.slice_positions(0, rev_c=True)} == {1, 2, 5, 6}


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


@needs_descriptors
def test_revc_pod_has_eight_tori_at_two_links_and_eight_link_seams():
    galaxies = [Galaxy(d) for d in load_mapping(SC4_REVC_MAPPING)]
    assert all(g.rev_c for g in galaxies)
    table, ranks = rr.predicted_rank_table(galaxies, 8)
    rows, boundaries = rr.stage_report(galaxies, table)
    assert [(kind, links) for _, kind, _, links in rows] == [("in-Galaxy torus", 2), ("seam torus", 2)] * 4
    assert boundaries == [(i, i + 1, 8) for i in range(7)]
    # The in-Galaxy torus is the slice 1+2 chips: ASICs 3,4,7,8 of trays 1..4 = logical ids 2,3,6,7,... on a Rev C host.
    assert ranks[0]["devices"] == "2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31"
    assert [r["host"] for r in ranks[:3]] == [galaxies[0].name, galaxies[0].name, galaxies[1].name]


@needs_descriptors
def test_revab_pod_uses_the_rev_ab_slice_trays():
    galaxies = [Galaxy(d) for d in load_mapping(SC4_REVAB_MAPPING)]
    assert not any(g.rev_c for g in galaxies)
    table, _ = rr.predicted_rank_table(galaxies, 8)
    rows, boundaries = rr.stage_report(galaxies, table)
    assert all(links >= 2 for *_, links in rows)
    assert [n for *_, n in boundaries] == [8] * 7
    g = galaxies[0]
    assert {chip_tray for chip_tray, _ in (rr.chip_position(g, c) for c in rr.slice_chips(g, 3))} == {2, 4}


@needs_descriptors
def test_wrong_host_order_is_not_a_torus_ring():
    galaxies = [Galaxy(d) for d in load_mapping(SC4_REVC_MAPPING)]
    swapped = [galaxies[0], galaxies[2], galaxies[1], galaxies[3]]
    table, _ = rr.predicted_rank_table(swapped, 8)
    rows, _ = rr.stage_report(swapped, table)
    assert any(links == 0 for *_, links in rows)
    # Only the cabling ring itself fits, in the direction slice 3 -> slice 0.
    found = rr.ring_orders(swapped, 8)
    assert [[g.name for g in order] for _, order in found] == [[g.name for g in galaxies]]


@needs_descriptors
def test_flat_aisle_pod_has_no_subtorus_tori():
    # A non-subtorus cabling (revAB aisle D) has links but no C4xC4 on the carve's chips.
    if not SC4_REVAB_FLAT_MAPPING.exists():
        pytest.skip("flat aisle D mapping missing")
    galaxies = [Galaxy(d) for d in load_mapping(SC4_REVAB_FLAT_MAPPING)]
    assert rr.ring_orders(galaxies, 8) == []


@needs_descriptors
def test_two_galaxy_pair_chain(tmp_path):
    out = tmp_path / "pair.yaml"
    rankfile = tmp_path / "pair.rf"
    names = [Galaxy(d).name for d in load_mapping(SC4_REVC_MAPPING)]
    rc = rr.main(
        [
            "predict",
            "--mapping",
            str(SC4_REVC_MAPPING),
            "--host",
            names[1],
            "--host",
            names[2],
            "--stages",
            "3",
            "--out",
            str(out),
            "--rankfile-out",
            str(rankfile),
        ]
    )
    assert rc == 0
    binding = yaml.safe_load(out.read_text())
    assert [(b["mesh_id"], b["mesh_host_rank"]) for b in binding["rank_bindings"]] == [(0, 0), (1, 0), (1, 1), (2, 0)]
    assert binding["mesh_graph_desc_path"].endswith("ci/sc2_4x4_rr_mgd.textproto")
    assert all(b["env_overrides"]["TT_VISIBLE_DEVICES"] for b in binding["rank_bindings"])
    assert rankfile.read_text().splitlines() == [
        f"rank 0={names[1]} slot=0",
        f"rank 1={names[1]} slot=1",
        f"rank 2={names[2]} slot=0",
        f"rank 3={names[2]} slot=1",
    ]


@needs_descriptors
def test_pair_order_search_finds_the_direction_of_the_cabling():
    galaxies = [Galaxy(d) for d in load_mapping(SC4_REVC_MAPPING)]
    forward = [galaxies[1], galaxies[2]]
    found = rr.ring_orders(forward, 3)
    assert [[g.name for g in order] for _, order in found] == [[galaxies[1].name, galaxies[2].name]]
