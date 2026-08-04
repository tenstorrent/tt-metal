# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Tests for the collective-redundancy analyzer. Pure Python, no ttnn, no device.

Run with:  pytest models/tt_dit/tools/dit_analyzer/tests/test_dit_analyzer.py
or:        python3 models/tt_dit/tools/dit_analyzer/tests/test_dit_analyzer.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dit_analyzer import GraphBuilder, analyze_graph, load_graph  # noqa: E402
from dit_analyzer.capture import Trace, TraceOp, trace_to_graph  # noqa: E402
from dit_analyzer.ir import Dist, Graph, Mesh  # noqa: E402
from dit_analyzer.region import Box, RegionSet  # noqa: E402
from dit_analyzer.report import render_report, render_states  # noqa: E402

SP, TP = 0, 1
MESH = Mesh(shape=(2, 4), axis_names=("sp", "tp"))


# -----------------------------------------------------------------------------
# region algebra
# -----------------------------------------------------------------------------
def test_region_shard_and_cover():
    shape = [1, 1024, 2048]
    full = RegionSet.full(shape)
    shards = [RegionSet.shard(shape, 2, i, 4) for i in range(4)]
    assert sum(s.volume for s in shards) == full.volume
    union = shards[0]
    for s in shards[1:]:
        union = union.union(s)
    assert union == full
    assert full.covers(shards[1])
    assert not shards[1].covers(full)
    assert shards[0].intersect(shards[1]).is_empty


def test_region_subtract_and_merge():
    shape = [8, 8]
    full = RegionSet.full(shape)
    hole = RegionSet.of(Box(((2, 4), (2, 4))))
    rest = full.subtract(hole)
    assert rest.volume == 64 - 4
    assert rest.union(hole) == full
    assert not rest.covers(hole)
    # adjacent boxes on one axis merge back into a single box
    left = RegionSet.of(Box(((0, 8), (0, 4))))
    right = RegionSet.of(Box(((0, 8), (4, 8))))
    assert len(left.union(right).boxes) == 1


def test_region_map_axis():
    r = RegionSet.of(Box(((0, 1), (0, 32), (64, 128))))
    shifted = r.map_axis(2, lambda lo, hi: (lo - 64, hi - 64))
    assert shifted.bounds(2) == (0, 64)


# -----------------------------------------------------------------------------
# forward / backward on a real block shape
# -----------------------------------------------------------------------------
def test_sd35_block_has_no_findings():
    """Precision test: every collective in the real block is load-bearing."""
    report = analyze_graph(load_graph("example:sd35_block"))
    assert report.findings == [], [f.title for f in report.findings]
    assert report.diagnostics == [], [str(d) for d in report.diagnostics]
    assert len(report.necessary) == len(report.views)
    # 6 TP gathers + 2 SP gathers inside the ring SDPA + 2 reduce-scatters
    assert len({v.node.id for v in report.views}) == 10


def test_sd35_double_gather_finds_six_redundant_gathers():
    """Recall test: the 12-collectives-could-be-6 pattern."""
    report = analyze_graph(load_graph("example:sd35_block_double_gather"))
    flagged = [f for f in report.findings if f.rule == "unused_gather"]
    assert len(flagged) == 6, [f.title for f in report.findings]
    assert all(f.confidence == "provable" for f in flagged)
    assert all(f.severity == "high" for f in flagged)
    # every flagged collective is the gather fused inside an AGMM
    for f in flagged:
        node = report.graph.node(f.nodes[0])
        assert node.op == "all_gather"
        assert node.fused_in and node.fused_in.startswith("agmm:")
        assert f.proof["redundant_pair"][1] == node.id
        assert f.calls == 38
    # and it is worth reporting: >1 GiB of link traffic per forward
    assert sum(f.bytes_per_forward for f in flagged) > 1 << 30
    # the necessary collectives are still recognised as necessary
    assert len({v.node.id for v in report.necessary}) == 10


def test_ltx_ring_block_finds_the_gate_qkv_double_gather():
    """Phase 5: LTX-2.3 on BH 4x8 (Ring). One duplicate per attention instance.

    On Ring, `use_nonfused_agmm` is False, so `_compute_gate` and `to_qkv`/`to_q`
    each take the fused AGMM path and each gathers the same activation over TP.
    """
    report = analyze_graph(load_graph("example:ltx_block_bh_4x8"))
    dups = [f for f in report.findings if f.rule == "duplicate_gather"]
    assert len(dups) == 6, [f.title for f in report.findings]
    assert [f.rule for f in report.findings] == ["duplicate_gather"] * 6
    assert report.diagnostics == [], [str(d) for d in report.diagnostics]

    for f in dups:
        flagged, earlier = (report.graph.node(n) for n in f.nodes)
        # the flagged gather is the one fused inside the Q / QKV projection
        assert flagged.op == "all_gather" and flagged.fused_in.startswith("agmm:")
        assert ".to_q" in flagged.fused_in
        # ... and it duplicates the gate projection's fused gather
        assert earlier.fused_in.startswith("agmm:") and "to_gate_logits" in earlier.fused_in
        # same attention instance on both sides
        assert flagged.fused_in.split(":")[1].split(".")[0] == earlier.fused_in.split(":")[1].split(".")[0]
        assert f.confidence == "provable" and f.calls == 48

    # the three video-sized ones dominate: >100 GiB of link traffic per forward
    video_sized = [f for f in dups if f.bytes_per_call > (100 << 20)]
    assert len(video_sized) == 3
    assert sum(f.bytes_per_forward for f in video_sized) > 100 * (1 << 30)


def test_ltx_linear_block_is_clean():
    """Same source file on BH 2x4 (Linear): one explicit gather, nothing redundant."""
    report = analyze_graph(load_graph("example:ltx_block_bh_2x4"))
    assert report.findings == [], [f.title for f in report.findings]
    assert report.diagnostics == [], [str(d) for d in report.diagnostics]
    assert len(report.necessary) == len(report.views)


def test_states_table_mentions_every_device_in_the_group():
    report = analyze_graph(load_graph("example:sd35_block"))
    text = render_states(report, node_filter="ag_spatial_pre_attn")
    assert "available before" in text and "needed downstream" in text
    for dev in (0, 1, 2, 3):
        assert "\n  %d " % dev in text
    assert "NECESSARY" in text


def test_report_renders_for_every_example():
    for name in (
        "sd35_block",
        "sd35_block_double_gather",
        "synthetic_redundancy",
        "ltx_block_bh_4x8",
        "ltx_block_bh_2x4",
    ):
        text = render_report(analyze_graph(load_graph("example:" + name)), states=True, link_bw_gbs=12.5)
        assert "graph: " in text and "collectives analyzed" in text


# -----------------------------------------------------------------------------
# individual rules
# -----------------------------------------------------------------------------
def test_synthetic_example_covers_each_rule():
    report = analyze_graph(load_graph("example:synthetic_redundancy"))
    rules = {f.rule for f in report.findings}
    assert {"dead_collective", "overwide_gather", "participant_shrink", "invariant_collective"} <= rules
    overwide = next(f for f in report.findings if f.rule == "overwide_gather")
    assert overwide.proof["wasted_fraction"] == 0.5
    shrink = next(f for f in report.findings if f.rule == "participant_shrink")
    assert shrink.proof["devices_needing_remote_data"] in ([0], [4])


def test_duplicate_gather_is_proved_by_value_identity():
    b = GraphBuilder("dup", MESH)
    x = b.input("x", [1, 512, 1024], shard={TP: 2})
    w = b.param("w", [1024, 1024], shard={TP: 1})
    g1 = b.all_gather(x, dim=2, mesh_axis=TP, label="first")
    y1 = b.matmul(g1, w, label="mm1")
    g2 = b.all_gather(x, dim=2, mesh_axis=TP, label="second")  # same value, again
    y2 = b.matmul(g2, w, label="mm2")
    report = analyze_graph(b.finish([y1, y2]))
    dups = [f for f in report.findings if f.rule == "duplicate_gather"]
    assert len(dups) == 1, [f.title for f in report.findings]
    assert dups[0].confidence == "provable"
    assert dups[0].proof["equivalent_symbol"] == g1.id
    assert "value_id" in dups[0].proof


def test_compute_between_gathers_blocks_the_duplicate_claim():
    """A pointwise op mints a new value, so the second gather is necessary."""
    b = GraphBuilder("no_dup", MESH)
    x = b.input("x", [1, 512, 1024], shard={TP: 2})
    scale = b.input("scale", [1, 1, 1024], shard={TP: 2})
    w = b.param("w", [1024, 1024], shard={TP: 1})
    g1 = b.all_gather(x, dim=2, mesh_axis=TP, label="first")
    y1 = b.matmul(g1, w, label="mm1")
    x2 = b.mul(x, scale, label="rescaled")
    g2 = b.all_gather(x2, dim=2, mesh_axis=TP, label="second")
    y2 = b.matmul(g2, w, label="mm2")
    report = analyze_graph(b.finish([y1, y2]))
    assert [f.rule for f in report.findings] == []


def test_reduce_scatter_is_not_flagged_as_unused():
    """The pre-state is an unreduced partial sum, so 'already local' proves nothing."""
    b = GraphBuilder("rs", MESH)
    x = b.input("x", [1, 512, 4096], shard={TP: 2})
    w = b.param("w", [4096, 1024], shard={TP: 0})  # row-parallel: K fractured
    out = b.matmul_rs(x, w, mesh_axis=TP, dim=-1, label="rowlinear")
    report = analyze_graph(b.finish([out]))
    assert [f.rule for f in report.findings] == []
    assert any(v.reduces for v in report.views)


def test_gather_of_partial_sum_is_diagnosed():
    b = GraphBuilder("bad", MESH)
    x = b.input("x", [1, 512, 4096], shard={TP: 2})
    w = b.param("w", [4096, 1024], shard={TP: 0})
    partial = b.matmul(x, w, label="rowlinear")  # partial sums over tp
    gathered = b.all_gather(partial, dim=2, mesh_axis=TP, label="wrong_gather")
    report = analyze_graph(b.finish([gathered]))
    codes = {d.code for d in report.diagnostics}
    assert "GATHER_OF_PARTIAL" in codes


def test_unknown_op_taints_confidence():
    b = GraphBuilder("unknown", MESH)
    x = b.input("x", [1, 512, 1024], shard={TP: 2})
    opaque = b.unknown("ttnn.some_new_op", [x], [1, 512, 1024], label="mystery")
    g = b.all_gather(opaque, dim=2, mesh_axis=TP, label="after_unknown")
    out = b.pointwise("gelu", [g], label="act")
    report = analyze_graph(b.finish([out]))
    assert any(d.code == "UNKNOWN_OP" for d in report.diagnostics)
    assert all(f.confidence == "suspicious" for f in report.findings), [(f.rule, f.confidence) for f in report.findings]


def test_mergeable_hint_is_not_a_redundancy_finding():
    report = analyze_graph(load_graph("example:sd35_block"))
    assert report.findings == []
    assert report.hints, "expected batching hints on the real block"
    assert all(h.rule == "mergeable_collectives" and h.bytes_per_call == 0 for h in report.hints)
    # k/v gathers inside one ring-SDPA kernel must not be suggested for merging
    for h in report.hints:
        a, bb = (report.graph.node(n) for n in h.nodes)
        assert not (a.fused_in and a.fused_in == bb.fused_in)


def test_findings_are_ranked_by_severity_then_traffic():
    report = analyze_graph(load_graph("example:synthetic_redundancy"))
    keys = [f.rank_key for f in report.findings]
    assert keys == sorted(keys)


# -----------------------------------------------------------------------------
# IR plumbing
# -----------------------------------------------------------------------------
def test_graph_json_round_trip_is_stable():
    graph = load_graph("example:sd35_block_double_gather")
    again = Graph.from_json(graph.to_json())
    assert again.to_json() == graph.to_json()
    before = analyze_graph(graph)
    after = analyze_graph(again)
    assert [f.title for f in before.findings] == [f.title for f in after.findings]


def test_mesh_groups():
    assert MESH.groups(TP) == [(0, 1, 2, 3), (4, 5, 6, 7)]
    assert MESH.groups(SP) == [(0, 4), (1, 5), (2, 6), (3, 7)]
    assert MESH.group_of(6, SP) == (2, 6)
    assert MESH.index_in_group(6, SP) == 1


def test_trace_to_graph_lifts_local_shapes_and_flags_assumptions():
    """The offline half of capture: per-device shapes -> logical shapes."""
    trace = Trace(mesh_shape=(2, 4), axis_names=("sp", "tp"), name="fake", steps=4)
    trace.entries = {0: "in0", 1: "in1"}
    trace.ops.append(
        TraceOp(
            op="all_gather",
            call="ttnn.experimental.all_gather_async",
            inputs=[0],
            outputs=[2],
            in_shapes=[[1, 512, 256]],  # per-device
            out_shapes=[[1, 512, 1024]],
            dtypes=["bf16"],
            attrs={"dim": 2, "cluster_axis": 1},
            loc="models/tt_dit/blocks/x.py:10",
        )
    )
    trace.ops.append(
        TraceOp(
            op="matmul",
            call="ttnn.experimental.minimal_matmul",
            inputs=[2, 1],
            outputs=[3],
            in_shapes=[[1, 512, 1024], [1024, 256]],
            out_shapes=[[1, 512, 256]],
            dtypes=["bf16"],
            attrs={},
            loc="models/tt_dit/layers/linear.py:296",
        )
    )
    graph = trace_to_graph(
        trace,
        placements={"in0": Dist.make(Mesh(shape=(2, 4)), {0: 1, 1: 2}), "in1": Dist.make(Mesh(shape=(2, 4)), {1: 1})},
        params=["in1"],
    )
    # in0 local [1,512,256] with sp on dim1 and tp on dim2 -> logical [1,1024,1024]
    in0 = graph.symbol([s for s in graph.symbols if s.startswith("in0")][0])
    assert in0.shape == (1, 1024, 1024)
    report = analyze_graph(graph)
    assert len({v.node.id for v in report.views}) == 1
    assert "assumptions" not in graph.meta  # both entries were declared

    graph2 = trace_to_graph(trace)  # nothing declared
    assert graph2.meta["assumptions"]


def test_report_declares_trust_and_flags_shim_belief():
    # Requirement: every report must state its provenance, and say "the shim
    # believes" whenever a finding rests on shim-computed (not device-verified)
    # shapes -- and that must survive a JSON round trip (dump then analyze).
    from dit_analyzer.ir import Graph
    from dit_analyzer.report import render_report, render_trust

    b = GraphBuilder(name="hw", mesh=MESH)
    x = b.input("x", [1, 512, 1024], shard={TP: 2})
    g = b.finish([b.all_gather(x, dim=2, mesh_axis=TP, label="ag")])
    assert g.provenance == "hand-written"  # builder default
    assert "hand-transcribed" in render_trust(g)
    assert "SHIM BELIEVES" not in render_report(analyze_graph(g))

    g.provenance = "dry-run"
    assert "SHIM BELIEVES" in render_trust(g)
    assert "SHIM BELIEVES" in render_report(analyze_graph(g))  # in the report body
    assert Graph.from_json(g.to_json()).provenance == "dry-run"  # survives serialization

    unknown = Graph(name="u", mesh=MESH)
    assert "unverified" in render_trust(unknown)


def test_reshape_preserves_a_shard_on_a_kept_trailing_axis():
    # the VAE's [B,H,W,C] <-> [B,1,H*W,C] merges leading axes but keeps the
    # channel axis; a shard on it must survive, not degrade to replicated+taint.
    b = GraphBuilder(name="vae_reshape", mesh=MESH)
    x = b.input("x", [1, 4, 4, 8], shard={TP: 3})  # channels fractured on tp
    merged = b.view(x, [1, 1, 16, 8], label="merge_hw")  # H,W -> H*W, C kept
    g = b.all_gather(merged, dim=3, mesh_axis=TP, label="ag")  # real gather of the shard
    report = analyze_graph(b.finish([b.pointwise("silu", [g], label="out")]))
    codes = {d.code for d in report.diagnostics}
    assert "OPAQUE_RESHAPE" not in codes, codes  # the shard was tracked through
    assert "GATHER_OF_REPLICATED" not in codes, codes  # so the gather isn't seen as a no-op
    # and a shard that IS on a reshaped (merged) axis must still be opaque
    b2 = GraphBuilder(name="vae_reshape_opaque", mesh=MESH)
    y = b2.input("y", [1, 4, 4, 8], shard={TP: 1})  # fractured on a merged spatial axis
    b2.view(y, [1, 1, 16, 8], label="merge_hw")
    report2 = analyze_graph(b2.finish([y]))
    assert "OPAQUE_RESHAPE" in {d.code for d in report2.diagnostics}


def test_shard_chunk_matches_ttnn_torch_chunk_semantics():
    from dit_analyzer.region import RegionSet, shard_chunk_count, shard_chunk_size

    # even split: the common tt_dit case
    assert shard_chunk_size(40, 4) == 10
    assert shard_chunk_count(40, 4) == 4
    # uneven: ttnn's chunk_ndim gives ceil(E/n) to leading devices and the
    # remainder to one trailing device (torch.chunk), NOT an even ceil to all
    assert shard_chunk_size(38, 4) == 10  # ceil(38/4)
    assert shard_chunk_count(38, 4) == 4  # 10,10,10,8 -> 4 non-empty shards
    # the region shards tile the axis exactly, disjoint and complete
    shards = [RegionSet.shard((1, 38), 1, i, 4) for i in range(4)]
    assert [s.bounds(1) for s in shards] == [(0, 10), (10, 20), (20, 30), (30, 38)]
    assert sum(s.volume for s in shards) == 38  # last shard is short, not padded
    # the empty-device case ttnn TT_FATALs on a 2D mesh
    assert shard_chunk_size(5, 4) == 2  # ceil(5/4)
    assert shard_chunk_count(5, 4) == 3  # 2,2,1 -> device 3 gets nothing


def test_uneven_shard_local_shape_follows_ttnn_or_refuses(monkeypatch=None):
    import dit_analyzer.dryrun.context as context_mod
    from dit_analyzer.dryrun.tensor import local_shape
    from dit_analyzer.ir import Dist, Mesh

    mesh = Mesh(shape=(1, 4), axis_names=("sp", "tp"))
    saved = context_mod.CTX.mesh
    context_mod.CTX.mesh = mesh
    try:
        # tp-sharded on axis 1: even -> exact, uneven -> ttnn chunk size (ceil)
        dist = Dist.make(mesh, {1: 1})
        assert local_shape((1, 40), dist) == (1, 10)
        assert local_shape((1, 38), dist) == (1, 10)  # device-0 chunk, ceil(38/4)
        # the empty-device case is refused, matching ttnn's TT_FATAL
        try:
            local_shape((1, 5), dist)
            raise AssertionError("expected NotImplementedError for the empty-device shard")
        except NotImplementedError as exc:
            assert "empty" in str(exc)
    finally:
        context_mod.CTX.mesh = saved


def test_padded_volume_rounds_the_innermost_two_axes_to_a_tile():
    from dit_analyzer.region import TILE, Box, RegionSet

    assert TILE == 32
    # a num_heads-column gather ships a full tile row, not 8 logical columns
    b = Box.full((1, 40, 8))
    assert b.volume == 1 * 40 * 8
    assert b.padded_volume() == 1 * 64 * 32  # 40->64, 8->32
    # only the last two axes are tiled; the leading axis is untouched
    assert Box.full((5, 32, 32)).padded_volume() == 5 * 32 * 32
    # tile-aligned extents (the LTX video activation) are unchanged, so the
    # oracle findings keep their byte counts
    assert Box.full((1, 38912, 4096)).padded_volume() == 1 * 38912 * 4096
    # region padding sums per box
    rs = RegionSet(3, [Box.full((1, 40, 8))])
    assert rs.padded_volume() == 1 * 64 * 32


def test_block_float_bytes_carry_exponent_overhead():
    from dit_analyzer.ir import TensorSymbol, elem_bytes_for

    # whole-byte formats are exact
    assert elem_bytes_for("bf16") == 2.0
    assert elem_bytes_for("fp32") == 4.0
    # block-float carries +1/16 byte of shared exponent per element, which
    # whole-byte accounting (the old int table) dropped -- so a bfp8_b gather
    # was undercounted by ~6%.
    assert elem_bytes_for("bfp8_b") == 1.0 + 1.0 / 16
    assert elem_bytes_for("bfp4_b") == 0.5 + 1.0 / 16
    assert elem_bytes_for("bf8_b") == elem_bytes_for("bfp8_b")  # legacy alias
    assert elem_bytes_for("unknown-tag") == 2.0  # conservative default
    # and it reaches the symbol's byte estimate
    sym = TensorSymbol(id="w", shape=(32, 32), dtype="bfp8_b")
    assert sym.elem_bytes == 1.0 + 1.0 / 16
    assert sym.bytes_of(sym.full_region()) == 32 * 32 * (1.0 + 1.0 / 16)


def _tests():
    return [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]


if __name__ == "__main__":
    failed = 0
    for name, fn in _tests():
        try:
            fn()
            print("PASS %s" % name)
        except AssertionError as exc:
            failed += 1
            print("FAIL %s: %s" % (name, exc))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("ERROR %s: %r" % (name, exc))
    print("\n%d/%d passed" % (len(_tests()) - failed, len(_tests())))
    sys.exit(1 if failed else 0)
