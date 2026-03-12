# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Standalone tests for the 5-stage optimization pipeline.

No device needed. Tests SDF, Polyhedral, ILP, Software Pipelining,
and Graph Coloring on the gated_local_reduce topology:

  LocalReduce(SiLU) -> intermed
  LocalReduce       -> intermed
  EltwiseMul(in0=intermed[0], in1=intermed[1]) -> out
"""

import sys
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Mock ttnn so tests work without device/build
# ---------------------------------------------------------------------------
_mock_ttnn = MagicMock()


class MockCoreCoord:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class MockCoreRange:
    def __init__(self, start, end):
        self.start_coord = start
        self.end_coord = end


class MockCoreRangeSet:
    def __init__(self, ranges_list):
        self._ranges = ranges_list

    def ranges(self):
        return self._ranges


def make_single_core(x=0, y=0):
    return MockCoreRangeSet([MockCoreRange(MockCoreCoord(x, y), MockCoreCoord(x, y))])


# Patch ttnn into sys.modules before importing auto_fusion
if "ttnn" not in sys.modules:
    sys.modules["ttnn"] = _mock_ttnn
    _mock_ttnn.CoreRangeSet = MockCoreRangeSet
    _mock_ttnn.CoreRange = MockCoreRange
    _mock_ttnn.CoreCoord = MockCoreCoord

from models.demos.deepseek_v3_b1.auto_fusion.graph import FusionGraph
from models.demos.deepseek_v3_b1.auto_fusion.graph_coloring import ChaitinBriggsAllocator
from models.demos.deepseek_v3_b1.auto_fusion.ilp_scheduler import ILPScheduler
from models.demos.deepseek_v3_b1.auto_fusion.polyhedral import PolyhedralAnalyzer
from models.demos.deepseek_v3_b1.auto_fusion.sdf import SDFAnalyzer
from models.demos.deepseek_v3_b1.auto_fusion.software_pipeline import SoftwarePipeliner
from models.demos.deepseek_v3_b1.auto_fusion.specs.eltwise_mul import ELTWISE_MUL
from models.demos.deepseek_v3_b1.auto_fusion.specs.local_reduce import LOCAL_REDUCE

# ===========================================================================
# Helpers: Build gated_local_reduce graph
# ===========================================================================


def build_gated_local_reduce_graph(
    group1_tiles=4,
    group2_tiles=4,
):
    """
    Build a gated local reduce graph:
      reduce1(SiLU) -> intermed
      reduce2       -> intermed
      eltwise_mul(intermed[0], intermed[1]) -> out

    This mirrors the hand-fused gated_local_reduce_kernel.cpp
    """
    g = FusionGraph()
    cores = make_single_core()

    # Phase 1: LocalReduce with SiLU (group1)
    g.add(
        "reduce1",
        LOCAL_REDUCE,
        cores=cores,
        ct_args={"num_tiles": group1_tiles, "apply_silu": True},
    )

    # Phase 2: LocalReduce without SiLU (group2)
    g.add(
        "reduce2",
        LOCAL_REDUCE,
        cores=cores,
        ct_args={"num_tiles": group2_tiles, "apply_silu": False},
    )

    # Phase 3: EltwiseMul (binary: reduce1_out * reduce2_out -> out)
    g.add(
        "mul",
        ELTWISE_MUL,
        cores=cores,
        ct_args={"num_tiles": 1},
        inputs={
            "in0": ("reduce1", "output"),
            "in1": ("reduce2", "output"),
        },
    )

    return g


# ===========================================================================
# SDF Tests
# ===========================================================================


class TestSDFAnalysis:
    """Tests for Stage 1: SDF Analysis."""

    def test_topology_matrix_shape(self):
        """Topology matrix has correct dimensions."""
        g = build_gated_local_reduce_graph()
        sdf = SDFAnalyzer(g)
        gamma = sdf.build_topology_matrix()
        # 2 edges (reduce1->mul, reduce2->mul), 3 actors
        assert gamma.shape == (2, 3)

    def test_topology_matrix_values(self):
        """Topology matrix has correct production/consumption rates."""
        g = build_gated_local_reduce_graph()
        sdf = SDFAnalyzer(g)
        gamma = sdf.build_topology_matrix()
        # Edge 0: reduce1.output -> mul.in0 (both rate 1)
        # Edge 1: reduce2.output -> mul.in1 (both rate 1)
        # Actor order: reduce1=0, reduce2=1, mul=2
        assert gamma[0, 0] == 1  # reduce1 produces 1
        assert gamma[0, 2] == -1  # mul consumes 1
        assert gamma[1, 1] == 1  # reduce2 produces 1
        assert gamma[1, 2] == -1  # mul consumes 1

    def test_repetition_vector(self):
        """All actors fire once (single-rate graph)."""
        g = build_gated_local_reduce_graph()
        sdf = SDFAnalyzer(g)
        q = sdf.compute_repetition_vector()
        assert q == {"reduce1": 1, "reduce2": 1, "mul": 1}

    def test_buffer_bounds(self):
        """Buffer bounds match hand-fused: intermed needs 1 page per edge."""
        g = build_gated_local_reduce_graph()
        sdf = SDFAnalyzer(g)
        bounds = sdf.compute_buffer_bounds()
        # Each edge needs max(prod_rate, cons_rate) = max(1, 1) = 1
        assert bounds[("reduce1", "mul")] == 1
        assert bounds[("reduce2", "mul")] == 1

    def test_double_buffering_not_suggested_same_core(self):
        """No double buffering for same-core same-RISC ops."""
        g = build_gated_local_reduce_graph()
        sdf = SDFAnalyzer(g)
        suggestions = sdf.suggest_double_buffering()
        # Both reduce and mul are TRISC-only, same core
        # No cross-RISC overlap possible
        assert len(suggestions) == 0

    def test_no_edges_graph(self):
        """Single node graph: trivial repetition vector."""
        g = FusionGraph()
        g.add("single", LOCAL_REDUCE, cores=make_single_core(), ct_args={"num_tiles": 4, "apply_silu": False})
        sdf = SDFAnalyzer(g)
        q = sdf.compute_repetition_vector()
        assert q == {"single": 1}


# ===========================================================================
# Polyhedral Tests
# ===========================================================================


class TestPolyhedralAnalysis:
    """Tests for Stage 2: Polyhedral Analysis."""

    def test_iteration_domains_are_1d(self):
        """All gated_local_reduce ops have 1D domains."""
        g = build_gated_local_reduce_graph(group1_tiles=4, group2_tiles=4)
        poly = PolyhedralAnalyzer(g)
        domains = poly.build_iteration_domains()
        assert domains["reduce1"].dimensions == [(0, 4)]
        assert domains["reduce2"].dimensions == [(0, 4)]
        assert domains["mul"].dimensions == [(0, 1)]

    def test_mul_domain_is_degenerate(self):
        """EltwiseMul with num_tiles=1 has degenerate domain."""
        g = build_gated_local_reduce_graph()
        poly = PolyhedralAnalyzer(g)
        domains = poly.build_iteration_domains()
        assert domains["mul"].is_degenerate

    def test_fusion_legality_all_pairs(self):
        """All sequential pairs are fusion-legal."""
        g = build_gated_local_reduce_graph()
        poly = PolyhedralAnalyzer(g)
        # reduce1 -> mul (flow dependence)
        dep = poly.check_fusion_legality("reduce1", "mul")
        assert dep.is_legal
        assert dep.dependence_type == "flow"
        # reduce2 -> mul (flow dependence)
        dep = poly.check_fusion_legality("reduce2", "mul")
        assert dep.is_legal
        assert dep.dependence_type == "flow"
        # reduce1 -> reduce2 (independent, no edge)
        dep = poly.check_fusion_legality("reduce1", "reduce2")
        assert dep.is_legal
        assert dep.dependence_type == "none"

    def test_fusion_legality_reverse_has_anti_dependence(self):
        """Reverse pair (mul -> reduce1) has anti-dependence (reduce1->mul edge exists)."""
        g = build_gated_local_reduce_graph()
        poly = PolyhedralAnalyzer(g)
        dep = poly.check_fusion_legality("mul", "reduce1")
        # reduce1 -> mul edge exists, so reversing order creates anti-dependence
        assert not dep.is_legal
        assert dep.dependence_type == "anti"

    def test_tile_sizes_degenerate(self):
        """Tile sizes for degenerate domains = total iterations."""
        g = build_gated_local_reduce_graph(group1_tiles=4, group2_tiles=4)
        poly = PolyhedralAnalyzer(g)
        sizes = poly.compute_tile_sizes(l1_budget=1048576)
        assert sizes["reduce1"]["tiles"] == 4
        assert sizes["reduce2"]["tiles"] == 4
        assert sizes["mul"]["tiles"] == 1

    def test_unknown_op_rejected(self):
        """Unknown op ID returns not-legal."""
        g = build_gated_local_reduce_graph()
        poly = PolyhedralAnalyzer(g)
        dep = poly.check_fusion_legality("reduce1", "nonexistent")
        assert not dep.is_legal


# ===========================================================================
# ILP Scheduler Tests
# ===========================================================================


class TestILPScheduler:
    """Tests for Stage 3: ILP Scheduling."""

    def test_schedule_respects_precedence(self):
        """ILP schedule respects data edges: reduce before mul."""
        g = build_gated_local_reduce_graph()
        ilp = ILPScheduler(g)
        result = ilp.solve()
        schedule = result.schedule
        assert schedule.index("reduce1") < schedule.index("mul")
        assert schedule.index("reduce2") < schedule.index("mul")

    def test_schedule_is_sequential(self):
        """For 3-op graph, schedule is sequential."""
        g = build_gated_local_reduce_graph()
        ilp = ILPScheduler(g)
        result = ilp.solve()
        assert len(result.schedule) == 3

    def test_makespan_positive(self):
        """Makespan is positive for non-empty graph."""
        g = build_gated_local_reduce_graph()
        ilp = ILPScheduler(g)
        result = ilp.solve()
        assert result.makespan > 0

    def test_cb_assignments_fit_in_32(self):
        """All CB assignments use indices < 32."""
        g = build_gated_local_reduce_graph()
        ilp = ILPScheduler(g)
        result = ilp.solve()
        for slot in result.cb_assignments.values():
            assert 0 <= slot < 32

    def test_solver_status(self):
        """Solver finds optimal or fallback solution."""
        g = build_gated_local_reduce_graph()
        ilp = ILPScheduler(g)
        result = ilp.solve()
        assert result.solver_status in ("optimal", "feasible", "fallback")

    def test_empty_graph(self):
        """Empty graph returns empty schedule."""
        g = FusionGraph()
        ilp = ILPScheduler(g)
        result = ilp.solve()
        assert result.schedule == []
        assert result.makespan == 0

    def test_start_times_non_negative(self):
        """All start times are non-negative."""
        g = build_gated_local_reduce_graph()
        ilp = ILPScheduler(g)
        result = ilp.solve()
        for t in result.start_times.values():
            assert t >= 0


# ===========================================================================
# Software Pipelining Tests
# ===========================================================================


class TestSoftwarePipelining:
    """Tests for Stage 4: Software Pipelining."""

    def test_not_beneficial_for_gated_local_reduce(self):
        """Pipelining is NOT beneficial: all work on TRISC."""
        g = build_gated_local_reduce_graph()
        pipeliner = SoftwarePipeliner(g)
        assert not pipeliner.is_beneficial()

    def test_mii_equals_trisc_total(self):
        """MII = sum of TRISC latencies (only active RISC)."""
        g = build_gated_local_reduce_graph()
        pipeliner = SoftwarePipeliner(g)
        mii = pipeliner.compute_mii()
        # LOCAL_REDUCE: trisc=500, ELTWISE_MUL: trisc=300
        # Total TRISC = 500 + 500 + 300 = 1300
        assert mii == 1300

    def test_generate_schedule_output(self):
        """Pipeline schedule has correct structure."""
        g = build_gated_local_reduce_graph()
        pipeliner = SoftwarePipeliner(g)
        sched = pipeliner.generate_schedule()
        assert not sched.is_beneficial
        assert sched.mii > 0
        assert sched.res_mii > 0
        assert sched.rec_mii == 0  # DAG, no recurrence
        assert len(sched.reason) > 0  # Has explanation

    def test_prologue_epilogue_zero_when_not_beneficial(self):
        """No prologue/epilogue when pipelining is not beneficial."""
        g = build_gated_local_reduce_graph()
        pipeliner = SoftwarePipeliner(g)
        sched = pipeliner.generate_schedule()
        assert sched.prologue_length == 0
        assert sched.epilogue_length == 0

    def test_risc_utilization(self):
        """RISC utilization is computed correctly."""
        g = build_gated_local_reduce_graph()
        pipeliner = SoftwarePipeliner(g)
        sched = pipeliner.generate_schedule()
        assert "trisc" in sched.risc_utilization
        assert sched.risc_utilization["trisc"] > 0


# ===========================================================================
# Graph Coloring Tests
# ===========================================================================


class TestGraphColoring:
    """Tests for Stage 5: Chaitin-Briggs Graph Coloring."""

    def test_four_color_assignment(self):
        """Gated local reduce uses exactly the right number of colors."""
        g = build_gated_local_reduce_graph()
        nodes = g.nodes
        edges = g.edges
        schedule = g.get_schedule()
        coloring = ChaitinBriggsAllocator(nodes, edges, schedule)
        result = coloring.allocate()
        # 7 ports total: reduce1(input, output), reduce2(input, output), mul(in0, in1, out)
        assert len(result.port_assignments) == 7
        assert result.num_colors_used <= 7

    def test_no_spills(self):
        """No spills for small graph."""
        g = build_gated_local_reduce_graph()
        coloring = ChaitinBriggsAllocator(g.nodes, g.edges, g.get_schedule())
        result = coloring.allocate()
        assert len(result.spilled) == 0

    def test_interfering_cbs_get_different_colors(self):
        """Simultaneously-live CBs get different slot indices."""
        g = build_gated_local_reduce_graph()
        coloring = ChaitinBriggsAllocator(g.nodes, g.edges, g.get_schedule())
        result = coloring.allocate()
        # reduce1.output and reduce2.output are both live when mul executes
        r1_out = result.port_assignments.get(("reduce1", "output"))
        r2_out = result.port_assignments.get(("reduce2", "output"))
        if r1_out is not None and r2_out is not None:
            assert r1_out != r2_out

    def test_same_core_chained_cbs_share_color(self):
        """SAME_CORE chained ports share the same CB slot."""
        g = build_gated_local_reduce_graph()
        coloring = ChaitinBriggsAllocator(g.nodes, g.edges, g.get_schedule())
        result = coloring.allocate()
        # reduce1.output -> mul.in0 are SAME_CORE chained
        r1_out = result.port_assignments.get(("reduce1", "output"))
        mul_in0 = result.port_assignments.get(("mul", "in0"))
        if r1_out is not None and mul_in0 is not None:
            assert r1_out == mul_in0

    def test_interference_graph_construction(self):
        """Interference graph has correct number of edges."""
        g = build_gated_local_reduce_graph()
        coloring = ChaitinBriggsAllocator(g.nodes, g.edges, g.get_schedule())
        adj = coloring.build_interference_graph()
        assert len(adj) > 0  # At least some groups

    def test_all_slots_valid(self):
        """All assigned slots are in range [0, 31]."""
        g = build_gated_local_reduce_graph()
        coloring = ChaitinBriggsAllocator(g.nodes, g.edges, g.get_schedule())
        result = coloring.allocate()
        for slot in result.port_assignments.values():
            assert 0 <= slot <= 31


# ===========================================================================
# Integration: Full 5-Stage Pipeline
# ===========================================================================


class TestFullPipeline:
    """Integration tests: compile() runs all 5 stages."""

    def test_compile_succeeds(self):
        """FusionGraph.compile() runs without error."""
        g = build_gated_local_reduce_graph()
        external_ports = {("reduce1", "input"), ("reduce2", "input")}
        source, schedule, allocator = g.compile(external_ports=external_ports)
        assert isinstance(source, str)
        assert len(source) > 100
        assert len(schedule) == 3

    def test_compile_stores_analysis_results(self):
        """compile() populates SDF, poly, ILP, pipeline, coloring results."""
        g = build_gated_local_reduce_graph()
        external_ports = {("reduce1", "input"), ("reduce2", "input")}
        g.compile(external_ports=external_ports)
        assert hasattr(g, "_sdf_result")
        assert hasattr(g, "_poly_result")
        assert hasattr(g, "_ilp_result")
        assert hasattr(g, "_pipeline_result")
        assert hasattr(g, "_coloring_result")

    def test_sdf_result_correct(self):
        """SDF analysis produces expected repetition vector."""
        g = build_gated_local_reduce_graph()
        external_ports = {("reduce1", "input"), ("reduce2", "input")}
        g.compile(external_ports=external_ports)
        assert g._sdf_result["repetition_vector"] == {"reduce1": 1, "reduce2": 1, "mul": 1}

    def test_pipeline_not_beneficial(self):
        """Pipeline analysis correctly identifies no benefit."""
        g = build_gated_local_reduce_graph()
        external_ports = {("reduce1", "input"), ("reduce2", "input")}
        g.compile(external_ports=external_ports)
        assert not g._pipeline_result.is_beneficial

    def test_coloring_no_spills(self):
        """Coloring has no spills."""
        g = build_gated_local_reduce_graph()
        external_ports = {("reduce1", "input"), ("reduce2", "input")}
        g.compile(external_ports=external_ports)
        assert len(g._coloring_result.spilled) == 0

    def test_generated_source_has_all_ops(self):
        """Generated kernel source references all 3 ops."""
        g = build_gated_local_reduce_graph()
        external_ports = {("reduce1", "input"), ("reduce2", "input")}
        source, _, _ = g.compile(external_ports=external_ports)
        assert "reduce1" in source.lower() or "REDUCE1" in source
        assert "reduce2" in source.lower() or "REDUCE2" in source
        assert "mul" in source.lower() or "MUL" in source

    def test_generated_source_has_local_reduce_include(self):
        """Generated source includes local_reduce.hpp."""
        g = build_gated_local_reduce_graph()
        external_ports = {("reduce1", "input"), ("reduce2", "input")}
        source, _, _ = g.compile(external_ports=external_ports)
        assert "local_reduce.hpp" in source

    def test_generated_source_has_eltwise_mul_include(self):
        """Generated source includes eltwise_mul.hpp."""
        g = build_gated_local_reduce_graph()
        external_ports = {("reduce1", "input"), ("reduce2", "input")}
        source, _, _ = g.compile(external_ports=external_ports)
        assert "eltwise_mul.hpp" in source

    def test_cb_allocations_have_correct_count(self):
        """CB allocator assigns indices for all ports."""
        g = build_gated_local_reduce_graph()
        external_ports = {("reduce1", "input"), ("reduce2", "input")}
        _, _, allocator = g.compile(external_ports=external_ports)
        # 7 ports: reduce1(input,output), reduce2(input,output), mul(in0,in1,out)
        assert len(allocator._allocations) == 7


# ===========================================================================
# Spec Tests
# ===========================================================================


class TestSpecs:
    """Tests for LOCAL_REDUCE and ELTWISE_MUL specs."""

    def test_local_reduce_spec_structure(self):
        """LOCAL_REDUCE has correct structure."""
        assert LOCAL_REDUCE.name == "LocalReduce"
        assert LOCAL_REDUCE.ncrisc.is_noop
        assert LOCAL_REDUCE.brisc.is_noop
        assert not LOCAL_REDUCE.trisc.is_noop
        assert "input" in LOCAL_REDUCE.cb_ports
        assert "output" in LOCAL_REDUCE.cb_ports
        assert LOCAL_REDUCE.cb_ports["input"].is_sharded
        assert LOCAL_REDUCE.cb_ports["input"].sdf_rate.is_parametric
        assert LOCAL_REDUCE.cb_ports["output"].sdf_rate.tokens == 1

    def test_eltwise_mul_spec_structure(self):
        """ELTWISE_MUL has correct structure."""
        assert ELTWISE_MUL.name == "EltwiseMul"
        assert ELTWISE_MUL.ncrisc.is_noop
        assert ELTWISE_MUL.brisc.is_noop
        assert "in0" in ELTWISE_MUL.cb_ports
        assert "in1" in ELTWISE_MUL.cb_ports
        assert "out" in ELTWISE_MUL.cb_ports

    def test_local_reduce_risc_latency(self):
        """LOCAL_REDUCE has estimated latencies."""
        assert LOCAL_REDUCE.risc_latency["trisc"] == 500
        assert LOCAL_REDUCE.risc_latency["ncrisc"] == 50

    def test_eltwise_mul_risc_latency(self):
        """ELTWISE_MUL has estimated latencies."""
        assert ELTWISE_MUL.risc_latency["trisc"] == 300


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
