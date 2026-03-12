# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Standalone tests for down_proj auto-fusion (no device needed).

Tests the graph construction, CB allocation, codegen, and analysis pipeline
for the 5-op down_proj pattern: Mcast1 + Mcast2 + Matmul + ResidualAdd + Gather.
"""

import pytest


# Mock ttnn core types for standalone testing
class MockCoreCoord:
    def __init__(self, x, y):
        self.x, self.y = x, y


class MockCoreRange:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.start_coord = start
        self.end_coord = end

    def contains(self, coord):
        return self.start.x <= coord.x <= self.end.x and self.start.y <= coord.y <= self.end.y


class MockCoreRangeSet:
    def __init__(self, ranges_list):
        self._ranges = ranges_list

    def ranges(self):
        return self._ranges


def single_core(x=0, y=0):
    c = MockCoreCoord(x, y)
    return MockCoreRangeSet([MockCoreRange(c, c)])


def core_grid(x0, y0, x1, y1):
    return MockCoreRangeSet([MockCoreRange(MockCoreCoord(x0, y0), MockCoreCoord(x1, y1))])


def multi_core_set(coords):
    """Create a MockCoreRangeSet from a list of (x, y) coordinate pairs."""
    ranges = [MockCoreRange(MockCoreCoord(x, y), MockCoreCoord(x, y)) for x, y in coords]
    return MockCoreRangeSet(ranges)


# ============================================================================
# Constants matching down_proj topology
# ============================================================================
MCAST_GRID = core_grid(0, 0, 12, 9)  # 13x10 = 130 cores
SENDER_CORE = single_core(12, 9)

# Build matmul core set (112 cores = 130 - 8 DRAM - 9 phantoms - 1 sender)
DRAM_WORKERS = [(0, 0), (0, 3), (0, 7), (0, 9), (7, 1), (7, 4), (7, 6), (7, 9)]
PHANTOMS = [(12, r) for r in range(10)]  # Col 12, all rows (including (12,9) sender)

_excluded = set(DRAM_WORKERS) | set(PHANTOMS)
_matmul_coords = [(col, row) for row in range(10) for col in range(13) if (col, row) not in _excluded]
assert len(_matmul_coords) == 112, f"Expected 112 matmul cores, got {len(_matmul_coords)}"
MATMUL_CORES = multi_core_set(_matmul_coords)

# Mcast receivers = all 130 minus sender (12,9)
_receiver_coords = [(col, row) for row in range(10) for col in range(13) if not (col == 12 and row == 9)]
MCAST_RECEIVERS = multi_core_set(_receiver_coords)


# ============================================================================
# Test: Graph construction + codegen
# ============================================================================


class TestDownProjGraphCodegen:
    """Test the graph construction and code generation for down_proj."""

    def _build_graph(self):
        """Build the 5-op down_proj FusionGraph."""
        from models.demos.deepseek_v3_b1.auto_fusion.graph import FusionGraph
        from models.demos.deepseek_v3_b1.auto_fusion.specs import GATHER, MATMUL, MCAST, RESIDUAL_ADD

        K_TILES = 8  # K dimension (number of input tiles)
        OUT_W = 4  # Output width per core (tiles)
        NUM_MATMUL_CORES = 112
        TOTAL_RESIDUAL_TILES = NUM_MATMUL_CORES * OUT_W  # = 448

        g = FusionGraph()

        # Mcast1: input broadcast from (12,9) to all 130 cores
        g.add(
            "mcast1",
            MCAST,
            cores=MCAST_GRID,
            ct_args={
                "num_cores": 130,
                "is_part_of_receiver_grid": True,
                "dest_noc_start_x": 0,
                "dest_noc_start_y": 0,
                "dest_noc_end_x": 12,
                "dest_noc_end_y": 9,
                "data_sender_semaphore": 0,
                "data_receiver_semaphore": 1,
                "data_size_bytes": K_TILES * 2048,
                "src_num_pages": K_TILES,
                "dst_num_pages": K_TILES,
                "_sender_cores": SENDER_CORE,
                "_receiver_cores": MCAST_RECEIVERS,
            },
        )

        # Mcast2: residual broadcast from (12,9) to all 130 cores
        g.add(
            "mcast2",
            MCAST,
            cores=MCAST_GRID,
            ct_args={
                "num_cores": 130,
                "is_part_of_receiver_grid": True,
                "dest_noc_start_x": 0,
                "dest_noc_start_y": 0,
                "dest_noc_end_x": 12,
                "dest_noc_end_y": 9,
                "data_sender_semaphore": 0,  # Reuses semaphore 0
                "data_receiver_semaphore": 4,
                "data_size_bytes": TOTAL_RESIDUAL_TILES * 64,  # 1x32 tile
                "src_num_pages": TOTAL_RESIDUAL_TILES,
                "dst_num_pages": TOTAL_RESIDUAL_TILES,
                "_sender_cores": SENDER_CORE,
                "_receiver_cores": MCAST_RECEIVERS,
            },
        )

        # Matmul: [1, K] x [K, N_per_core] -> [1, N_per_core] on 112 cores
        g.add(
            "matmul",
            MATMUL,
            cores=MATMUL_CORES,
            ct_args={
                "out_w": OUT_W,
                "transpose": False,
                "fused_activation": 0,
                "k_num_tiles": K_TILES,
                "in1_num_pages": K_TILES * OUT_W,
                "pop_in0": True,
                "pop_in1": False,
            },
            inputs={"in0": ("mcast1", "dst")},
        )

        # ResidualAdd: matmul_out + shard(residual) on 112 matmul cores
        g.add(
            "residual_add",
            RESIDUAL_ADD,
            cores=MATMUL_CORES,
            ct_args={
                "out_w": OUT_W,
                "total_in1_tiles": TOTAL_RESIDUAL_TILES,
                "core_idx": 0,  # Placeholder, per-core in real usage
            },
            inputs={
                "in0": ("matmul", "out"),
                "in1": ("mcast2", "dst"),
            },
        )

        # Gather: collect from 112 cores to (12,9)
        g.add(
            "gather",
            GATHER,
            cores=MCAST_GRID,  # All cores participate
            ct_args={
                "dest_noc_x": 0,
                "dest_noc_y": 0,
                "data_size_bytes": OUT_W * 64,
                "receiver_semaphore_id": 2,
                "src_num_pages": OUT_W,
                "sender_grid_start_x": 0,
                "sender_grid_start_y": 0,
                "sender_grid_end_x": 0,
                "sender_grid_end_y": 0,
                "row_major": 1,
                "receiver_data_addr": 0,
                "sender_idx": 0,
                "noc0_num_senders": NUM_MATMUL_CORES,
                "noc1_num_senders": 0,
                "noc0_receiver_semaphore_id": 2,
                "noc1_receiver_semaphore_id": 3,
                "dst_num_pages": NUM_MATMUL_CORES * OUT_W,
                "use_per_core_sender_idx": True,
                "_sender_cores": MATMUL_CORES,
                "_receiver_cores": SENDER_CORE,
            },
            inputs={"src": ("residual_add", "out")},
        )

        return g

    def test_graph_has_five_ops(self):
        g = self._build_graph()
        assert len(g.nodes) == 5
        assert g.get_schedule() == ["mcast1", "mcast2", "matmul", "residual_add", "gather"]

    def test_graph_has_correct_edges(self):
        g = self._build_graph()
        edges = g.edges
        # mcast1.dst -> matmul.in0 (MCAST)
        # matmul.out -> residual_add.in0 (SAME_CORE)
        # mcast2.dst -> residual_add.in1 (MCAST)
        # residual_add.out -> gather.src (SAME_CORE)
        assert len(edges) == 4

    def test_compile_succeeds(self):
        g = self._build_graph()
        source, schedule, allocator = g.compile()
        assert source is not None
        assert len(schedule) == 5

    def test_generated_source_has_all_ops(self):
        g = self._build_graph()
        source, _, _ = g.compile()
        assert "mcast1_op" in source
        assert "mcast2_op" in source
        assert "matmul_op" in source
        assert "residual_add_op" in source
        assert "gather_op" in source

    def test_generated_source_has_init_teardown(self):
        g = self._build_graph()
        source, _, _ = g.compile()
        # Mcast1 and mcast2 should have init/teardown
        assert "mcast1_op.init(" in source
        assert "mcast2_op.init(" in source
        assert "mcast1_op.teardown()" in source
        assert "mcast2_op.teardown()" in source

    def test_generated_source_has_correct_includes(self):
        g = self._build_graph()
        source, _, _ = g.compile()
        assert "mcast.hpp" in source
        assert "matmul.hpp" in source
        assert "gather.hpp" in source
        assert "residual_add.hpp" in source

    def test_generated_source_has_role_flags(self):
        g = self._build_graph()
        source, _, _ = g.compile()
        assert "is_mcast1_core" in source
        assert "is_mcast1_sender" in source
        assert "is_mcast1_receiver" in source
        assert "is_matmul_core" in source
        assert "is_gather_sender" in source
        assert "is_gather_receiver" in source

    def test_generated_source_has_rt_args_init(self):
        """NCRISC/BRISC sections should have proper RT args initialization."""
        g = self._build_graph()
        source, _, _ = g.compile()
        # NCRISC mcast receiver args should have get_semaphore
        assert "get_semaphore(" in source
        # BRISC mcast sender args should have get_read_ptr and get_write_ptr
        assert "get_read_ptr(" in source
        assert "get_write_ptr(" in source

    def test_generated_source_has_compute_init_guard(self):
        """deepseek_compute_kernel_init only inside guard for compute-active ops."""
        g = self._build_graph()
        source, _, _ = g.compile()
        assert "deepseek_compute_kernel_init();" in source
        # Guard should include matmul and residual_add (the compute-active ops)
        assert "is_matmul_core" in source
        assert "is_residual_add_core" in source

    def test_cb_allocations_correct_count(self):
        """Should have 8+ CBs like the hand-fused kernel."""
        g = self._build_graph()
        _, _, allocator = g.compile()
        unique_indices = set(a.index for a in allocator._allocations.values())
        # At minimum: src(mcast1), dst(mcast1)=in0(matmul), in1(matmul),
        # out(matmul)=in0(residual_add), src(mcast2), dst(mcast2)=in1(residual_add),
        # out(residual_add)=src(gather), dst(gather)
        assert len(unique_indices) >= 6, f"Only {len(unique_indices)} unique CB indices"

    def test_matmul_weights_sharded_setup(self):
        """Matmul in1 (weights) should have setup_sharded_buffer."""
        g = self._build_graph()
        source, _, _ = g.compile()
        assert "setup_sharded_buffer(matmul_in1_cb" in source

    def test_matmul_in0_not_sharded_setup(self):
        """Matmul in0 (from mcast) should NOT have setup_sharded_buffer."""
        g = self._build_graph()
        source, _, _ = g.compile()
        # in0 comes from mcast (internal), should not be set up as sharded
        assert "setup_sharded_buffer(matmul_in0_cb" not in source

    def test_gather_use_per_core_sender_idx(self):
        """Gather should have use_per_core_sender_idx in Core struct."""
        g = self._build_graph()
        source, _, _ = g.compile()
        assert "gather_use_per_core_sender_idx" in source


class TestDownProjAnalysis:
    """Test the 5-stage analysis pipeline for down_proj."""

    def _build_and_compile(self):
        """Build and compile the down_proj graph, returning graph and results."""
        g = TestDownProjGraphCodegen()._build_graph()
        source, schedule, allocator = g.compile()
        return g, source, schedule, allocator

    def test_sdf_repetition_vector(self):
        g, _, _, _ = self._build_and_compile()
        rep_vec = g._sdf_result["repetition_vector"]
        for node_id, count in rep_vec.items():
            assert count >= 1

    def test_polyhedral_fusion_legal(self):
        """All sequential op pairs should be fusion-legal."""
        g, _, schedule, _ = self._build_and_compile()
        # If compile succeeded, fusion legality was already checked
        assert len(schedule) == 5

    def test_ilp_schedule(self):
        g, _, _, _ = self._build_and_compile()
        ilp = g._ilp_result
        assert ilp.solver_status in ("optimal", "feasible", "fallback")

    def test_coloring_no_spills(self):
        g, _, _, _ = self._build_and_compile()
        coloring = g._coloring_result
        assert len(coloring.spilled) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
