# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Standalone tests for auto-fusion code generation.

These tests verify the codegen output WITHOUT requiring a device.
They validate that:
1. The FusionGraph correctly builds a topology
2. CB allocation assigns indices without conflicts
3. The generated C++ kernel source has correct structure
4. The generated source follows the hand-fused kernel patterns

Run with: python -m pytest models/demos/deepseek_v3_b1/auto_fusion/tests/test_codegen_standalone.py -xvs
"""


import pytest

from models.demos.deepseek_v3_b1.auto_fusion.cb_allocator import CBAllocator
from models.demos.deepseek_v3_b1.auto_fusion.codegen import UnifiedKernelCodegen
from models.demos.deepseek_v3_b1.auto_fusion.types import (
    CBDirection,
    CBPortSpec,
    CorePlacement,
    DataEdge,
    MicroOpSpec,
    OpNode,
    RISCContract,
    TransferType,
)

# ============================================================================
# Minimal mock CoreRangeSet for standalone testing (no ttnn dependency)
# ============================================================================


class MockCoreCoord:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class MockCoreRange:
    def __init__(self, start, end):
        self.start_coord = start
        self.end_coord = end


class MockCoreRangeSet:
    def __init__(self, ranges):
        self._ranges = ranges

    def ranges(self):
        return self._ranges

    def num_cores(self):
        count = 0
        for cr in self._ranges:
            w = cr.end_coord.x - cr.start_coord.x + 1
            h = cr.end_coord.y - cr.start_coord.y + 1
            count += w * h
        return count


def single_core(x=0, y=0):
    c = MockCoreCoord(x, y)
    return MockCoreRangeSet([MockCoreRange(c, c)])


def core_grid(x0, y0, x1, y1):
    return MockCoreRangeSet([MockCoreRange(MockCoreCoord(x0, y0), MockCoreCoord(x1, y1))])


# ============================================================================
# Minimal spec fixtures for testing
# ============================================================================

SIMPLE_COMPUTE_SPEC = MicroOpSpec(
    name="SimpleCompute",
    header="unified_kernels/simple.hpp",
    namespace="test_ops",
    struct_name="SimpleCompute",
    ncrisc=RISCContract(
        ct_args_type="test_ops::SimpleCompute::ReaderCTArgs",
        rt_args_type="test_ops::SimpleCompute::ReaderArgs",
        setup_sharded=["input"],
        is_noop=True,
    ),
    brisc=RISCContract(
        ct_args_type="test_ops::SimpleCompute::WriterCTArgs",
        rt_args_type="test_ops::SimpleCompute::WriterArgs",
        is_noop=True,
    ),
    trisc=RISCContract(
        ct_args_type="test_ops::SimpleCompute::ComputeCTArgs<{num_tiles}>",
        rt_args_type="test_ops::SimpleCompute::ComputeArgs",
        named_ct_args=["input_cb", "output_cb", "num_tiles"],
        rt_args_fields=[
            ("input_cb", "ct:input_cb"),
            ("output_cb", "ct:output_cb"),
        ],
        cb_reads=["input"],
        cb_writes=["output"],
    ),
    cb_ports={
        "input": CBPortSpec(CBDirection.INPUT, is_sharded=True),
        "output": CBPortSpec(CBDirection.OUTPUT),
    },
    op_template="Op<{CTArgs}, {is_active}, true>",
)

SIMPLE_DATAFLOW_SPEC = MicroOpSpec(
    name="SimpleDataflow",
    header="unified_kernels/dataflow.hpp",
    namespace="test_ops",
    struct_name="SimpleDataflow",
    ncrisc=RISCContract(
        ct_args_type="test_ops::SimpleDataflow::ReaderCTArgs",
        rt_args_type="test_ops::SimpleDataflow::ReaderArgs",
        setup_sharded=["src"],
        is_noop=True,
    ),
    brisc=RISCContract(
        ct_args_type="test_ops::SimpleDataflow::WriterCTArgs",
        rt_args_type="test_ops::SimpleDataflow::WriterArgs",
        is_noop=True,
    ),
    trisc=RISCContract(
        ct_args_type="test_ops::SimpleDataflow::ComputeCTArgs",
        rt_args_type="test_ops::SimpleDataflow::ComputeArgs",
        is_noop=True,
    ),
    cb_ports={
        "src": CBPortSpec(CBDirection.INPUT, is_sharded=True),
        "dst": CBPortSpec(CBDirection.OUTPUT),
    },
    op_template="Op<{CTArgs}, {is_active}>",
)


# ============================================================================
# Graph construction tests
# ============================================================================


class TestFusionGraphConstruction:
    """Test that FusionGraph correctly builds topology."""

    def test_single_node(self):
        """Single node graph works."""
        # Patch the _cores_to_set to work with mocks
        import models.demos.deepseek_v3_b1.auto_fusion.graph as graph_mod
        from models.demos.deepseek_v3_b1.auto_fusion.graph import FusionGraph

        orig = graph_mod._cores_to_set
        graph_mod._cores_to_set = lambda cores: {
            (x, y)
            for cr in cores.ranges()
            for x in range(cr.start_coord.x, cr.end_coord.x + 1)
            for y in range(cr.start_coord.y, cr.end_coord.y + 1)
        }
        try:
            g = FusionGraph()
            g.add("op1", SIMPLE_COMPUTE_SPEC, single_core(), ct_args={"num_tiles": 3})
            assert len(g.nodes) == 1
            assert g.get_node("op1").id == "op1"
            assert g.get_schedule() == ["op1"]
        finally:
            graph_mod._cores_to_set = orig

    def test_linear_chain(self):
        """Two-node linear chain with SAME_CORE edge."""
        import models.demos.deepseek_v3_b1.auto_fusion.graph as graph_mod
        from models.demos.deepseek_v3_b1.auto_fusion.graph import FusionGraph

        orig = graph_mod._cores_to_set
        graph_mod._cores_to_set = lambda cores: {
            (x, y)
            for cr in cores.ranges()
            for x in range(cr.start_coord.x, cr.end_coord.x + 1)
            for y in range(cr.start_coord.y, cr.end_coord.y + 1)
        }
        try:
            g = FusionGraph()
            g.add("op1", SIMPLE_COMPUTE_SPEC, single_core(), ct_args={"num_tiles": 3})
            g.add(
                "op2",
                SIMPLE_COMPUTE_SPEC,
                single_core(),
                ct_args={"num_tiles": 3},
                inputs={"input": ("op1", "output")},
            )
            assert len(g.nodes) == 2
            assert len(g.edges) == 1
            assert g.edges[0].transfer == TransferType.SAME_CORE
        finally:
            graph_mod._cores_to_set = orig

    def test_cross_core_edge(self):
        """Edge between different cores is detected as non-SAME_CORE."""
        import models.demos.deepseek_v3_b1.auto_fusion.graph as graph_mod
        from models.demos.deepseek_v3_b1.auto_fusion.graph import FusionGraph

        orig = graph_mod._cores_to_set
        graph_mod._cores_to_set = lambda cores: {
            (x, y)
            for cr in cores.ranges()
            for x in range(cr.start_coord.x, cr.end_coord.x + 1)
            for y in range(cr.start_coord.y, cr.end_coord.y + 1)
        }
        try:
            g = FusionGraph()
            g.add("op1", SIMPLE_COMPUTE_SPEC, single_core(0, 0), ct_args={"num_tiles": 3})
            g.add(
                "op2",
                SIMPLE_COMPUTE_SPEC,
                core_grid(1, 0, 3, 0),
                ct_args={"num_tiles": 3},
                inputs={"input": ("op1", "output")},
            )
            assert len(g.edges) == 1
            assert g.edges[0].transfer == TransferType.MCAST  # 1 -> many
        finally:
            graph_mod._cores_to_set = orig

    def test_duplicate_id_rejected(self):
        """Duplicate op IDs are rejected."""
        from models.demos.deepseek_v3_b1.auto_fusion.graph import FusionGraph

        g = FusionGraph()
        g.add("op1", SIMPLE_COMPUTE_SPEC, single_core())
        with pytest.raises(ValueError, match="Duplicate"):
            g.add("op1", SIMPLE_COMPUTE_SPEC, single_core())

    def test_unknown_source_rejected(self):
        """Reference to unknown source node is rejected."""
        import models.demos.deepseek_v3_b1.auto_fusion.graph as graph_mod
        from models.demos.deepseek_v3_b1.auto_fusion.graph import FusionGraph

        orig = graph_mod._cores_to_set
        graph_mod._cores_to_set = lambda cores: {
            (x, y)
            for cr in cores.ranges()
            for x in range(cr.start_coord.x, cr.end_coord.x + 1)
            for y in range(cr.start_coord.y, cr.end_coord.y + 1)
        }
        try:
            g = FusionGraph()
            with pytest.raises(ValueError, match="Unknown source"):
                g.add(
                    "op1",
                    SIMPLE_COMPUTE_SPEC,
                    single_core(),
                    inputs={"input": ("nonexistent", "output")},
                )
        finally:
            graph_mod._cores_to_set = orig


# ============================================================================
# CB allocation tests
# ============================================================================


class TestCBAllocator:
    """Test CB index allocation."""

    def test_single_op_allocation(self):
        """Single op gets unique CB indices per port."""
        node = OpNode(
            id="op1",
            spec=SIMPLE_COMPUTE_SPEC,
            placement=CorePlacement(single_core(), "is_op1_core"),
            ct_args={"num_tiles": 3},
            rt_args={},
        )
        alloc = CBAllocator([node], [], ["op1"])
        result = alloc.allocate()

        assert ("op1", "input") in result
        assert ("op1", "output") in result
        assert result[("op1", "input")].index != result[("op1", "output")].index

    def test_same_core_chaining(self):
        """SAME_CORE edge shares CB index between output and input."""
        node1 = OpNode(
            id="op1",
            spec=SIMPLE_COMPUTE_SPEC,
            placement=CorePlacement(single_core(), "is_op1_core"),
            ct_args={},
            rt_args={},
        )
        node2 = OpNode(
            id="op2",
            spec=SIMPLE_COMPUTE_SPEC,
            placement=CorePlacement(single_core(), "is_op2_core"),
            ct_args={},
            rt_args={},
        )
        edge = DataEdge("op1", "output", "op2", "input", TransferType.SAME_CORE)

        alloc = CBAllocator([node1, node2], [edge], ["op1", "op2"])
        result = alloc.allocate()

        # Output of op1 shares CB with input of op2
        assert result[("op1", "output")].index == result[("op2", "input")].index

    def test_cross_core_separate_cbs(self):
        """Cross-core edges get separate CB indices."""
        node1 = OpNode(
            id="op1",
            spec=SIMPLE_COMPUTE_SPEC,
            placement=CorePlacement(single_core(0, 0), "is_op1_core"),
            ct_args={},
            rt_args={},
        )
        node2 = OpNode(
            id="op2",
            spec=SIMPLE_COMPUTE_SPEC,
            placement=CorePlacement(single_core(1, 0), "is_op2_core"),
            ct_args={},
            rt_args={},
        )
        edge = DataEdge("op1", "output", "op2", "input", TransferType.MCAST)

        alloc = CBAllocator([node1, node2], [edge], ["op1", "op2"])
        result = alloc.allocate()

        # Output of op1 and input of op2 should have different CBs (different cores)
        assert result[("op1", "output")].index != result[("op2", "input")].index

    def test_no_index_exceeds_31(self):
        """All allocated indices are in range [0, 31]."""
        nodes = []
        for i in range(10):
            nodes.append(
                OpNode(
                    id=f"op{i}",
                    spec=SIMPLE_COMPUTE_SPEC,
                    placement=CorePlacement(single_core(), f"is_op{i}_core"),
                    ct_args={},
                    rt_args={},
                )
            )
        alloc = CBAllocator(nodes, [], [f"op{i}" for i in range(10)])
        result = alloc.allocate()

        for key, a in result.items():
            assert 0 <= a.index < 32, f"CB index {a.index} out of range for {key}"


# ============================================================================
# Codegen tests
# ============================================================================


class MockGraph:
    """Mock graph for codegen testing."""

    def __init__(self, nodes, edges):
        self._nodes = nodes
        self._edges = edges

    @property
    def nodes(self):
        return self._nodes

    @property
    def edges(self):
        return self._edges


class TestCodegen:
    """Test unified kernel C++ code generation."""

    def _make_node(self, op_id, spec, cores=None, ct_args=None, cb_bindings=None):
        node = OpNode(
            id=op_id,
            spec=spec,
            placement=CorePlacement(cores or single_core(), f"is_{op_id}_core"),
            ct_args=ct_args or {},
            rt_args={},
        )
        if cb_bindings:
            node.cb_bindings = cb_bindings
        return node

    def test_single_op_generates_valid_structure(self):
        """Single-op kernel has correct structure."""
        node = self._make_node(
            "rmsnorm",
            SIMPLE_COMPUTE_SPEC,
            ct_args={"num_tiles": 3},
            cb_bindings={"input": 0, "output": 1},
        )
        graph = MockGraph([node], [])
        allocs = {
            ("rmsnorm", "input"): type("A", (), {"index": 0})(),
            ("rmsnorm", "output"): type("A", (), {"index": 1})(),
        }

        codegen = UnifiedKernelCodegen(graph, ["rmsnorm"], allocs)
        source = codegen.generate()

        # Verify structural elements
        assert "// Auto-generated fused kernel" in source
        assert '#include "../../../unified_kernels/kernel_op_api.hpp"' in source
        assert '#include "../../../unified_kernels/kernel_utils.hpp"' in source
        assert '#include "../../../unified_kernels/simple.hpp"' in source
        assert "struct Core {" in source
        assert "is_rmsnorm_core" in source
        assert "void kernel_main() {" in source
        assert "#if defined(COMPILE_FOR_NCRISC)" in source
        assert "#elif defined(COMPILE_FOR_BRISC)" in source
        assert "#elif defined(COMPILE_FOR_TRISC)" in source
        assert "#endif" in source
        assert "deepseek_compute_kernel_init();" in source
        # Profiler zones are off by default
        assert 'DeviceZoneScopedN("RMSNORM")' not in source

        # Verify profiler zones when enabled
        codegen_profiled = UnifiedKernelCodegen(graph, ["rmsnorm"], allocs, emit_profiler_zones=True)
        source_profiled = codegen_profiled.generate()
        assert 'DeviceZoneScopedN("RMSNORM")' in source_profiled

    def test_two_op_chain_generates_both_ops(self):
        """Two-op chain includes both op invocations in order."""
        node1 = self._make_node(
            "op1",
            SIMPLE_COMPUTE_SPEC,
            ct_args={"num_tiles": 3},
            cb_bindings={"input": 0, "output": 1},
        )
        node2 = self._make_node(
            "op2",
            SIMPLE_COMPUTE_SPEC,
            ct_args={"num_tiles": 3},
            cb_bindings={"input": 1, "output": 2},
        )
        graph = MockGraph([node1, node2], [])
        allocs = {}

        codegen = UnifiedKernelCodegen(graph, ["op1", "op2"], allocs)
        source = codegen.generate()

        # Both ops appear in execution body (as comments, profiler zones off by default)
        assert "// op1" in source
        assert "// op2" in source

        # op1 appears before op2 in the source
        pos1 = source.index("// op1")
        pos2 = source.index("// op2")
        assert pos1 < pos2

    def test_op_with_init_teardown(self):
        """Op with has_init/has_teardown gets init/teardown calls."""
        spec_with_init = MicroOpSpec(
            name="Persistent",
            header="unified_kernels/persistent.hpp",
            namespace="test_ops",
            struct_name="Persistent",
            ncrisc=RISCContract(
                ct_args_type="test_ops::Persistent::ReaderCTArgs",
                rt_args_type="test_ops::Persistent::ReaderArgs",
                is_noop=True,
            ),
            brisc=RISCContract(
                ct_args_type="test_ops::Persistent::WriterCTArgs",
                rt_args_type="test_ops::Persistent::WriterArgs",
                is_noop=True,
            ),
            trisc=RISCContract(
                ct_args_type="test_ops::Persistent::ComputeCTArgs",
                rt_args_type="test_ops::Persistent::ComputeArgs",
                is_noop=True,
            ),
            cb_ports={"src": CBPortSpec(CBDirection.INPUT)},
            op_template="Op<{CTArgs}, {is_active}>",
            has_init=True,
            has_teardown=True,
        )

        node = self._make_node("mcast", spec_with_init, cb_bindings={"src": 0})
        graph = MockGraph([node], [])
        codegen = UnifiedKernelCodegen(graph, ["mcast"], {})
        source = codegen.generate()

        # Profiler zones off by default, but init/teardown still emitted
        assert "mcast_op.init(mcast_args)" in source
        assert "mcast_op.teardown()" in source

    def test_trisc_rtargs_fields_emitted(self):
        """TRISC RT args with fields produce aggregate initialization."""
        node = self._make_node(
            "rmsnorm",
            SIMPLE_COMPUTE_SPEC,
            ct_args={"num_tiles": 3},
            cb_bindings={"input": 0, "output": 1},
        )
        graph = MockGraph([node], [])
        codegen = UnifiedKernelCodegen(graph, ["rmsnorm"], {})
        source = codegen.generate()

        # Should have .input_cb and .output_cb fields in TRISC section
        assert ".input_cb =" in source
        assert ".output_cb =" in source

    def test_ctargs_template_params_substituted(self):
        """Template parameters in CTArgs types are substituted."""
        node = self._make_node(
            "compute",
            SIMPLE_COMPUTE_SPEC,
            ct_args={"num_tiles": 7},
            cb_bindings={"input": 0, "output": 1},
        )
        graph = MockGraph([node], [])
        codegen = UnifiedKernelCodegen(graph, ["compute"], {})
        source = codegen.generate()

        # The TRISC CTArgs should reference the named CT arg
        assert "compute_CTArgs" in source

    def test_multiple_includes_deduplicated(self):
        """Same header included by multiple ops appears only once."""
        node1 = self._make_node("op1", SIMPLE_COMPUTE_SPEC, cb_bindings={"input": 0, "output": 1})
        node2 = self._make_node("op2", SIMPLE_COMPUTE_SPEC, cb_bindings={"input": 2, "output": 3})
        graph = MockGraph([node1, node2], [])
        codegen = UnifiedKernelCodegen(graph, ["op1", "op2"], {})
        source = codegen.generate()

        # simple.hpp should appear exactly once
        count = source.count("simple.hpp")
        assert count == 1


# ============================================================================
# Integration test: compile a graph matching RMSNorm -> Mcast -> Matmul
# ============================================================================


class TestRMSNormMcastMatmulCodegen:
    """
    Test code generation for a 3-op chain matching the first stages
    of the attention block: RMSNorm -> Mcast -> Matmul.
    """

    def test_three_op_chain_codegen(self):
        """Generate source for rmsnorm -> mcast -> matmul chain."""
        from models.demos.deepseek_v3_b1.auto_fusion.specs.matmul import MATMUL
        from models.demos.deepseek_v3_b1.auto_fusion.specs.rmsnorm import RMSNORM

        # Create nodes manually (skip graph to avoid ttnn dependency)
        rmsnorm_node = OpNode(
            id="rmsnorm",
            spec=RMSNORM,
            placement=CorePlacement(single_core(0, 0), "is_rmsnorm_core"),
            ct_args={
                "fp32_acc": True,
                "num_tiles": 48,
                "rsqrt_fast_approx": True,
                "epsilon": 0x358637BD,  # float_to_uint32(1e-6)
                "scalar": 0x3A1D4952,  # float_to_uint32(1/sqrt(7168))
            },
            rt_args={},
        )
        rmsnorm_node.cb_bindings = {"input": 0, "gamma": 1, "output": 2}

        matmul_node = OpNode(
            id="matmul",
            spec=MATMUL,
            placement=CorePlacement(core_grid(0, 0, 7, 3), "is_matmul_core"),
            ct_args={
                "out_w": 3,
                "transpose": False,
                "fused_activation": 0,
                "k_num_tiles": 48,
            },
            rt_args={},
        )
        matmul_node.cb_bindings = {"in0": 2, "in1": 3, "out": 4}

        graph = MockGraph([rmsnorm_node, matmul_node], [])
        allocs = {}
        codegen = UnifiedKernelCodegen(graph, ["rmsnorm", "matmul"], allocs)
        source = codegen.generate()

        # Verify includes
        assert "rmsnorm.hpp" in source
        assert "matmul.hpp" in source

        # Verify Core struct has both role flags
        assert "is_rmsnorm_core" in source
        assert "is_matmul_core" in source

        # Verify both ops appear in execution body in order
        rmsnorm_pos = source.index("rmsnorm_op;")
        matmul_pos = source.index("matmul_op;")
        assert rmsnorm_pos < matmul_pos

        # Verify TRISC section has CTArgs for both
        assert "rmsnorm_CTArgs" in source
        assert "matmul_CTArgs" in source

        # Verify RMSNorm TRISC CTArgs template has ComputeCTArgs
        assert "RMSNorm::ComputeCTArgs<" in source

        # Verify Matmul TRISC CTArgs template has ComputeCTArgs
        assert "Matmul::ComputeCTArgs<" in source

        # Verify sharded buffer setup in NCRISC section
        assert "setup_sharded_buffer" in source

        # Print source for manual inspection
        print("\n" + "=" * 80)
        print("Generated 2-op fused kernel (RMSNorm -> Matmul):")
        print("=" * 80)
        print(source)

    def test_common_runtime_args_codegen(self):
        """RMSNorm common RT args generate get_common_arg_val calls."""
        from models.demos.deepseek_v3_b1.auto_fusion.specs.rmsnorm import RMSNORM

        rmsnorm_node = OpNode(
            id="rmsnorm",
            spec=RMSNORM,
            placement=CorePlacement(single_core(0, 0), "is_rmsnorm_core"),
            ct_args={
                "fp32_acc": 0,
                "num_tiles": 12,
                "rsqrt_fast_approx": 0,
                "epsilon": 0x358637BD,
                "scalar": 0x3A1D4952,
            },
            rt_args={},
        )
        rmsnorm_node.cb_bindings = {"input": 0, "gamma": 1, "output": 2}

        graph = MockGraph([rmsnorm_node], [])
        codegen = UnifiedKernelCodegen(graph, ["rmsnorm"], {})
        source = codegen.generate()

        # epsilon uses get_common_arg_val<uint32_t>(0)
        assert "get_common_arg_val<uint32_t>(0)" in source
        # scalar uses get_common_arg_val<float>(1)
        assert "get_common_arg_val<float>(1)" in source
        # Both are in the TRISC section as aggregate init fields
        assert ".epsilon = get_common_arg_val<uint32_t>(0)" in source
        assert ".scalar = get_common_arg_val<float>(1)" in source

    def test_common_runtime_args_base_offset_multi_op(self):
        """Two ops with common RT args get correct base offsets."""
        from models.demos.deepseek_v3_b1.auto_fusion.specs.rmsnorm import RMSNORM

        node1 = OpNode(
            id="rmsnorm1",
            spec=RMSNORM,
            placement=CorePlacement(single_core(0, 0), "is_rmsnorm1_core"),
            ct_args={"fp32_acc": 0, "num_tiles": 12, "rsqrt_fast_approx": 0},
            rt_args={},
        )
        node1.cb_bindings = {"input": 0, "gamma": 1, "output": 2}

        node2 = OpNode(
            id="rmsnorm2",
            spec=RMSNORM,
            placement=CorePlacement(single_core(0, 0), "is_rmsnorm2_core"),
            ct_args={"fp32_acc": 0, "num_tiles": 12, "rsqrt_fast_approx": 0},
            rt_args={},
        )
        node2.cb_bindings = {"input": 3, "gamma": 4, "output": 5}

        graph = MockGraph([node1, node2], [])
        codegen = UnifiedKernelCodegen(graph, ["rmsnorm1", "rmsnorm2"], {})
        source = codegen.generate()

        # First op: base=0, epsilon at 0, scalar at 1
        assert ".epsilon = get_common_arg_val<uint32_t>(0)" in source
        assert ".scalar = get_common_arg_val<float>(1)" in source
        # Second op: base=2 (first op has 2 common RT args), epsilon at 2, scalar at 3
        assert ".epsilon = get_common_arg_val<uint32_t>(2)" in source
        assert ".scalar = get_common_arg_val<float>(3)" in source


# ============================================================================
# Liveness analysis tests
# ============================================================================


class TestLivenessAnalysis:
    """Test CB liveness interval computation."""

    def test_sharded_input_lives_until_consuming_step(self):
        """Sharded input is alive from step 0 to its op's step."""
        node = OpNode(
            id="op1",
            spec=SIMPLE_COMPUTE_SPEC,
            placement=CorePlacement(single_core(), "is_op1_core"),
            ct_args={},
            rt_args={},
        )
        alloc = CBAllocator([node], [], ["op1"])
        intervals = alloc._compute_liveness()

        # input is sharded → [0, 0] (step 0 is op1's step)
        assert intervals[("op1", "input")] == (0, 0)

    def test_output_lives_to_end_of_schedule(self):
        """Output without consumers lives to end of schedule."""
        node = OpNode(
            id="op1",
            spec=SIMPLE_COMPUTE_SPEC,
            placement=CorePlacement(single_core(), "is_op1_core"),
            ct_args={},
            rt_args={},
        )
        alloc = CBAllocator([node], [], ["op1"])
        intervals = alloc._compute_liveness()

        # output → [0, 1] (step 0 to num_steps=1)
        assert intervals[("op1", "output")] == (0, 1)

    def test_same_core_edge_merges_intervals(self):
        """SAME_CORE edge merges producer and consumer intervals."""
        node1 = OpNode(
            id="op1",
            spec=SIMPLE_COMPUTE_SPEC,
            placement=CorePlacement(single_core(), "is_op1_core"),
            ct_args={},
            rt_args={},
        )
        node2 = OpNode(
            id="op2",
            spec=SIMPLE_COMPUTE_SPEC,
            placement=CorePlacement(single_core(), "is_op2_core"),
            ct_args={},
            rt_args={},
        )
        edge = DataEdge("op1", "output", "op2", "input", TransferType.SAME_CORE)

        alloc = CBAllocator([node1, node2], [edge], ["op1", "op2"])
        intervals = alloc._compute_liveness()

        # Merged: op1.output [0, 2] ∪ op2.input [0, 1] = [0, 2]
        assert intervals[("op1", "output")] == intervals[("op2", "input")]
        assert intervals[("op1", "output")][0] == 0
        assert intervals[("op1", "output")][1] == 2

    def test_cross_core_edge_extends_source_lifetime(self):
        """Cross-core edge extends source to cover consumer step."""
        node1 = OpNode(
            id="op1",
            spec=SIMPLE_COMPUTE_SPEC,
            placement=CorePlacement(single_core(0, 0), "is_op1_core"),
            ct_args={},
            rt_args={},
        )
        node2 = OpNode(
            id="op2",
            spec=SIMPLE_COMPUTE_SPEC,
            placement=CorePlacement(single_core(1, 0), "is_op2_core"),
            ct_args={},
            rt_args={},
        )
        edge = DataEdge("op1", "output", "op2", "input", TransferType.MCAST)

        alloc = CBAllocator([node1, node2], [edge], ["op1", "op2"])
        intervals = alloc._compute_liveness()

        # op1.output: initially [0, 2], extended to max(2, step_of_op2=1) = 2
        assert intervals[("op1", "output")][1] >= 1

    def test_index_reuse_for_non_overlapping_cbs(self):
        """Non-overlapping CBs can reuse the same index."""
        # 3 sequential ops, each with input+output
        # op0.output [0,3], op0.input [0,0]
        # op1.output [1,3], op1.input [0,1]  (sharded)
        # op2.output [2,3], op2.input [0,2]  (sharded)
        # With edges: op0→op1 (SAME_CORE), op1→op2 (SAME_CORE)
        # op0.output and op1.input share an index
        # op1.output and op2.input share an index
        nodes = []
        for i in range(3):
            nodes.append(
                OpNode(
                    id=f"op{i}",
                    spec=SIMPLE_COMPUTE_SPEC,
                    placement=CorePlacement(single_core(), f"is_op{i}_core"),
                    ct_args={},
                    rt_args={},
                )
            )
        edges = [
            DataEdge("op0", "output", "op1", "input", TransferType.SAME_CORE),
            DataEdge("op1", "output", "op2", "input", TransferType.SAME_CORE),
        ]
        alloc = CBAllocator(nodes, edges, ["op0", "op1", "op2"])
        result = alloc.allocate()

        # op0.output == op1.input (chained)
        assert result[("op0", "output")].index == result[("op1", "input")].index
        # op1.output == op2.input (chained)
        assert result[("op1", "output")].index == result[("op2", "input")].index
        # Total unique indices should be ≤ 4 (op0.input, chain1, chain2, op2.output)
        unique_indices = {a.index for a in result.values()}
        assert len(unique_indices) <= 4


# ============================================================================
# L1 memory packing tests
# ============================================================================


class TestL1MemoryPacking:
    """Test L1 memory pool packing for intermediate CBs."""

    def test_non_overlapping_cbs_share_memory(self):
        """Two intermediate CBs with non-overlapping lifetimes share L1."""
        # op0 produces output (step 0, dies at step 0)
        # op1 produces output (step 1, lives to end)
        # op0's output can share memory with op1's output
        nodes = []
        for i in range(2):
            nodes.append(
                OpNode(
                    id=f"op{i}",
                    spec=SIMPLE_COMPUTE_SPEC,
                    placement=CorePlacement(single_core(), f"is_op{i}_core"),
                    ct_args={},
                    rt_args={},
                )
            )
        # Edge: op0.output → op1.input (SAME_CORE), so op0.output lives to step 1
        edge = DataEdge("op0", "output", "op1", "input", TransferType.SAME_CORE)

        alloc = CBAllocator(nodes, [edge], ["op0", "op1"])
        result = alloc.allocate()

        # Mark all as non-external
        for a in result.values():
            a.is_external = False

        # op0.input [0,0] and op1.output [1,2] don't overlap
        pool_size = alloc.pack_l1(
            {
                ("op0", "input"): 1024,
                ("op1", "output"): 2048,
            }
        )

        # They should share memory: pool = max(1024, 2048) = 2048 (not 3072)
        op0_input = result[("op0", "input")]
        op1_output = result[("op1", "output")]
        assert op0_input.pool_offset == 0 or op1_output.pool_offset == 0
        assert pool_size <= 2048  # Not 1024+2048=3072

    def test_overlapping_cbs_get_separate_memory(self):
        """Two intermediate CBs with overlapping lifetimes need separate L1."""
        node = OpNode(
            id="op1",
            spec=SIMPLE_COMPUTE_SPEC,
            placement=CorePlacement(single_core(), "is_op1_core"),
            ct_args={},
            rt_args={},
        )
        alloc = CBAllocator([node], [], ["op1"])
        result = alloc.allocate()

        # Both input and output of same op overlap (both alive at step 0)
        for a in result.values():
            a.is_external = False

        pool_size = alloc.pack_l1(
            {
                ("op1", "input"): 1024,
                ("op1", "output"): 2048,
            }
        )

        # They overlap → need separate memory → pool = 1024+2048 = 3072
        assert pool_size == 3072

    def test_external_cbs_not_packed(self):
        """External (tensor-backed) CBs are excluded from L1 packing."""
        node = OpNode(
            id="op1",
            spec=SIMPLE_COMPUTE_SPEC,
            placement=CorePlacement(single_core(), "is_op1_core"),
            ct_args={},
            rt_args={},
        )
        alloc = CBAllocator([node], [], ["op1"])
        result = alloc.allocate(external_ports={("op1", "input"), ("op1", "output")})

        pool_size = alloc.pack_l1(
            {
                ("op1", "input"): 1024,
                ("op1", "output"): 2048,
            }
        )

        # Both external → nothing to pack
        assert pool_size == 0

    def test_five_phase_attention_like_packing(self):
        """Simulate 5-phase pipeline: RMSNorm → Mcast → Matmul → Gather → RMSNorm2.

        Tests that intermediate CBs are packed efficiently, similar to
        the hand-fused attention block's L1 reuse pattern.
        """
        # Create a simple 2-port spec for each phase
        spec = SIMPLE_COMPUTE_SPEC

        nodes = []
        for i, name in enumerate(["rmsnorm", "mcast", "matmul", "gather", "rmsnorm2"]):
            nodes.append(
                OpNode(
                    id=name,
                    spec=spec,
                    placement=CorePlacement(single_core(), f"is_{name}_core"),
                    ct_args={},
                    rt_args={},
                )
            )

        # Linear chain: each output feeds next input (SAME_CORE)
        edges = [
            DataEdge("rmsnorm", "output", "mcast", "input", TransferType.SAME_CORE),
            DataEdge("mcast", "output", "matmul", "input", TransferType.SAME_CORE),
            DataEdge("matmul", "output", "gather", "input", TransferType.SAME_CORE),
            DataEdge("gather", "output", "rmsnorm2", "input", TransferType.SAME_CORE),
        ]

        alloc = CBAllocator(nodes, edges, ["rmsnorm", "mcast", "matmul", "gather", "rmsnorm2"])
        result = alloc.allocate()

        # Mark all as internal (no external tensors)
        for a in result.values():
            a.is_external = False

        # Set sizes: 2KB per intermediate CB
        cb_sizes = {}
        for key in result:
            cb_sizes[key] = 2048

        pool_size = alloc.pack_l1(cb_sizes)

        # With 5 ops, each with 2 ports = 10 CBs.
        # Sequential chain means many can share memory.
        # Naive allocation: 10 × 2KB = 20KB
        # With packing: should be significantly less
        naive_total = 10 * 2048
        assert pool_size < naive_total, f"L1 packing should save memory: {pool_size} vs naive {naive_total}"

        # Print the packing summary for inspection
        print(f"\n5-phase pipeline: pool={pool_size}B " f"(naive={naive_total}B, saving {naive_total - pool_size}B)")
        print(alloc.get_liveness_summary())

    def test_cb_index_reuse_across_phases(self):
        """CB indices are reused when lifetimes don't overlap."""
        # 4 sequential ops with SAME_CORE chaining
        nodes = [
            OpNode(
                id=f"op{i}",
                spec=SIMPLE_COMPUTE_SPEC,
                placement=CorePlacement(single_core(), f"is_op{i}_core"),
                ct_args={},
                rt_args={},
            )
            for i in range(4)
        ]
        edges = [DataEdge(f"op{i}", "output", f"op{i+1}", "input", TransferType.SAME_CORE) for i in range(3)]

        alloc = CBAllocator(nodes, edges, [f"op{i}" for i in range(4)])
        result = alloc.allocate()

        unique_indices = {a.index for a in result.values()}

        # 4 ops × 2 ports = 8 logical CBs, but with chaining and index reuse
        # we should need fewer than 8 indices
        # Chains: (op0.out, op1.in), (op1.out, op2.in), (op2.out, op3.in)
        # Free: op0.in, op3.out
        # That's 5 logical groups, but with index reuse:
        # op0.input [0,0] can share with op3.output after op0.input dies
        assert len(unique_indices) <= 5

        # All indices in valid range
        for idx in unique_indices:
            assert 0 <= idx <= 31


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
