# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone tests for Sequential Kernel Chaining Infrastructure

These tests don't require ttnn to be imported, allowing testing of the
core infrastructure logic independently.
"""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch


# Mock ttnn before importing sequential
sys.modules["ttnn"] = MagicMock()
sys.modules["ttnn._ttnn"] = MagicMock()
sys.modules["ttnn._ttnn.program_descriptor"] = MagicMock()


class TestCBInfo:
    """Tests for CBInfo dataclass."""

    def test_creation(self):
        """Test creating CBInfo."""
        from models.experimental.ops.descriptors.sequential import CBInfo

        cb = CBInfo(
            original_index=5,
            total_size=4096,
            data_format="Float16_b",
            page_size=2048,
            core_ranges="mock_ranges",
        )

        assert cb.original_index == 5
        assert cb.total_size == 4096
        assert cb.page_size == 2048
        assert cb.data_format == "Float16_b"
        assert cb.core_ranges == "mock_ranges"


class TestPhaseInfo:
    """Tests for PhaseInfo dataclass."""

    def test_creation(self):
        """Test creating PhaseInfo."""
        from models.experimental.ops.descriptors.sequential import PhaseInfo, CBInfo

        phase = PhaseInfo(
            phase_idx=0,
            op_descriptor=MagicMock(),
            cb_info={0: CBInfo(0, 1024, None, 1024, None)},
        )

        assert phase.phase_idx == 0
        assert 0 in phase.cb_info
        assert phase.cb_info == {0: CBInfo(0, 1024, None, 1024, None)}

    def test_default_empty_cb_info(self):
        """Test PhaseInfo with default empty cb_info."""
        from models.experimental.ops.descriptors.sequential import PhaseInfo

        phase = PhaseInfo(phase_idx=1, op_descriptor=MagicMock())
        assert phase.cb_info == {}


class TestExtractCBInfo:
    """Tests for extract_cb_info function."""

    def test_extract_from_descriptor(self):
        """Test extracting CB info from a mock descriptor."""
        from models.experimental.ops.descriptors.sequential import extract_cb_info

        # Create mock format descriptor
        fmt_desc = MagicMock()
        fmt_desc.buffer_index = 3
        fmt_desc.data_format = "Float16_b"
        fmt_desc.page_size = 2048

        # Create mock CB descriptor
        cb_desc = MagicMock()
        cb_desc.total_size = 8192
        cb_desc.core_ranges = "mock_ranges"
        cb_desc.format_descriptors = [fmt_desc]

        # Create mock program descriptor
        prog_desc = MagicMock()
        prog_desc.cbs = [cb_desc]

        result = extract_cb_info(prog_desc)

        assert 3 in result
        assert result[3].original_index == 3
        assert result[3].total_size == 8192
        assert result[3].page_size == 2048

    def test_extract_multiple_cbs(self):
        """Test extracting multiple CBs."""
        from models.experimental.ops.descriptors.sequential import extract_cb_info

        fmt1 = MagicMock(buffer_index=0, data_format="F16", page_size=1024)
        fmt2 = MagicMock(buffer_index=16, data_format="F32", page_size=2048)

        cb1 = MagicMock(total_size=2048, core_ranges="r1", format_descriptors=[fmt1])
        cb2 = MagicMock(total_size=4096, core_ranges="r2", format_descriptors=[fmt2])

        prog_desc = MagicMock(cbs=[cb1, cb2])

        result = extract_cb_info(prog_desc)

        assert len(result) == 2
        assert 0 in result
        assert 16 in result
        assert result[0].page_size == 1024
        assert result[16].page_size == 2048


class TestExtractCBNames:
    """Tests for extract_cb_names_from_kernel function."""

    def test_extract_cb_names(self):
        """Test extracting CB names from kernel descriptor."""
        from models.experimental.ops.descriptors.sequential import extract_cb_names_from_kernel

        kernel = MagicMock()
        kernel.named_compile_time_args = [
            ("cb_in", 0),
            ("cb_out", 16),
            ("cb_scaler", 2),
            ("some_other_arg", 42),
        ]

        result = extract_cb_names_from_kernel(kernel)

        assert result == {
            "cb_in": 0,
            "cb_out": 16,
            "cb_scaler": 2,
        }
        assert "some_other_arg" not in result


class TestSequentialChainBuilderBasic:
    """Basic tests for SequentialChainBuilder."""

    def test_creation(self):
        """Test creating a builder."""
        from models.experimental.ops.descriptors.sequential import SequentialChainBuilder

        builder = SequentialChainBuilder()
        assert len(builder.phases) == 0
        assert builder._built is False

    def test_add_phase(self):
        """Test adding phases to the builder."""
        from models.experimental.ops.descriptors.sequential import SequentialChainBuilder

        builder = SequentialChainBuilder()

        mock_desc = MagicMock()
        mock_desc.descriptor = MagicMock(cbs=[])

        builder.add_phase(mock_desc)
        assert len(builder.phases) == 1
        assert builder.phases[0].phase_idx == 0

        builder.add_phase(mock_desc)
        assert len(builder.phases) == 2
        assert builder.phases[1].phase_idx == 1

    def test_method_chaining(self):
        """Test that builder methods return self."""
        from models.experimental.ops.descriptors.sequential import SequentialChainBuilder

        builder = SequentialChainBuilder()

        mock_desc = MagicMock()
        mock_desc.descriptor = MagicMock(cbs=[])

        result = builder.add_phase(mock_desc).add_phase(mock_desc)

        assert result is builder
        assert len(builder.phases) == 2

    def test_single_phase_returns_original(self):
        """Test that single-phase build returns original descriptor."""
        from models.experimental.ops.descriptors.sequential import SequentialChainBuilder

        builder = SequentialChainBuilder()

        mock_desc = MagicMock()
        mock_desc.descriptor = MagicMock(cbs=[])

        builder.add_phase(mock_desc)
        result = builder.build(device=MagicMock())

        assert result is mock_desc

    def test_build_empty_raises(self):
        """Test that building empty chain raises ValueError."""
        from models.experimental.ops.descriptors.sequential import SequentialChainBuilder

        builder = SequentialChainBuilder()

        with pytest.raises(ValueError, match="no phases"):
            builder.build(device=MagicMock())

    def test_build_twice_raises(self):
        """Test that building twice raises ValueError."""
        from models.experimental.ops.descriptors.sequential import SequentialChainBuilder

        builder = SequentialChainBuilder()
        mock_desc = MagicMock()
        mock_desc.descriptor = MagicMock(cbs=[])

        builder.add_phase(mock_desc)
        builder.build(device=MagicMock())

        with pytest.raises(ValueError, match="already been built"):
            builder.build(device=MagicMock())


class TestChainDescriptorsConvenience:
    """Tests for the chain_descriptors convenience function."""

    def test_single_descriptor(self):
        """Test that single descriptor returns it unchanged."""
        from models.experimental.ops.descriptors.sequential import chain_descriptors

        mock_desc = MagicMock()
        mock_desc.descriptor = MagicMock(cbs=[])

        result = chain_descriptors([mock_desc], device=MagicMock())
        assert result is mock_desc

    def test_empty_list_raises(self):
        """Test that empty list raises ValueError."""
        from models.experimental.ops.descriptors.sequential import chain_descriptors

        with pytest.raises(ValueError, match="no phases"):
            chain_descriptors([], device=MagicMock())


class TestParallelChainsStandalone:
    """Standalone tests for parallel chain functions."""

    def test_create_parallel_chain_empty(self):
        """Test creating parallel chains with empty list."""
        from models.experimental.ops.descriptors.sequential import create_parallel_chain_descriptors

        result = create_parallel_chain_descriptors([], device=MagicMock())
        assert result == []

    def test_create_parallel_chain_single_ops(self):
        """Test creating parallel chains with single-op chains."""
        from models.experimental.ops.descriptors.sequential import create_parallel_chain_descriptors

        mock_desc1 = MagicMock()
        mock_desc1.descriptor = MagicMock(cbs=[])
        mock_desc2 = MagicMock()
        mock_desc2.descriptor = MagicMock(cbs=[])

        chains = [[mock_desc1], [mock_desc2]]
        result = create_parallel_chain_descriptors(chains, device=MagicMock())

        # Single-op chains should return original descriptors
        assert len(result) == 2
        assert result[0] is mock_desc1
        assert result[1] is mock_desc2


class TestSourceTransformations:
    """Tests for kernel source transformation functions."""

    def test_prefix_named_args_phase0_unchanged(self):
        """Test that phase 0 source is unchanged."""
        from models.experimental.ops.descriptors.sequential import _prefix_named_args_in_source

        source = 'constexpr uint32_t cb = get_named_compile_time_arg_val("cb_in");'
        result = _prefix_named_args_in_source(source, 0)
        assert result == source

    def test_prefix_named_args_phase1(self):
        """Test that phase 1 gets prefixed named args."""
        from models.experimental.ops.descriptors.sequential import _prefix_named_args_in_source

        source = 'constexpr uint32_t cb = get_named_compile_time_arg_val("cb_in");'
        result = _prefix_named_args_in_source(source, 1)
        assert 'get_named_compile_time_arg_val("phase1_cb_in")' in result

    def test_prefix_named_args_phase2(self):
        """Test phase 2 prefix."""
        from models.experimental.ops.descriptors.sequential import _prefix_named_args_in_source

        source = 'constexpr uint32_t blk = get_named_compile_time_arg_val("blk");'
        result = _prefix_named_args_in_source(source, 2)
        assert 'get_named_compile_time_arg_val("phase2_blk")' in result

    def test_offset_runtime_args_phase0_unchanged(self):
        """Test that phase 0 runtime args are unchanged."""
        from models.experimental.ops.descriptors.sequential import _offset_runtime_args_in_source

        source = "uint32_t val = get_arg_val<uint32_t>(3);"
        result = _offset_runtime_args_in_source(source, 0)
        assert result == source

    def test_offset_runtime_args_phase1(self):
        """Test phase 1 runtime arg offsetting."""
        from models.experimental.ops.descriptors.sequential import _offset_runtime_args_in_source

        source = "uint32_t val = get_arg_val<uint32_t>(3);"
        result = _offset_runtime_args_in_source(source, 1)
        assert "__phase1_rt_offset + 3" in result
        assert "phase1_rt_arg_offset" in result

    def test_offset_runtime_args_variable_init(self):
        """Test that incrementing variable init pattern is offset."""
        from models.experimental.ops.descriptors.sequential import _offset_runtime_args_in_source

        source = "uint32_t rt_args_idx = 0;"
        result = _offset_runtime_args_in_source(source, 1)
        assert "__phase1_rt_offset" in result
        # The variable should be initialized to the offset, not 0
        assert "= 0;" not in result or "__phase1_rt_offset" in result

    def test_offset_compile_time_args_phase0_unchanged(self):
        """Test that phase 0 positional args are unchanged."""
        from models.experimental.ops.descriptors.sequential import _offset_compile_time_args_in_source

        source = "uint32_t blk = get_compile_time_arg_val(0);"
        result = _offset_compile_time_args_in_source(source, 0, 0)
        assert result == source

    def test_offset_compile_time_args_phase1(self):
        """Test that phase 1 positional args are offset."""
        from models.experimental.ops.descriptors.sequential import _offset_compile_time_args_in_source

        source = "uint32_t blk = get_compile_time_arg_val(0);\nuint32_t mode = get_compile_time_arg_val(1);"
        result = _offset_compile_time_args_in_source(source, 1, 3)
        assert "get_compile_time_arg_val(3)" in result
        assert "get_compile_time_arg_val(4)" in result

    def test_offset_compile_time_args_tensor_accessor(self):
        """Test that TensorAccessorArgs<N> is also offset."""
        from models.experimental.ops.descriptors.sequential import _offset_compile_time_args_in_source

        source = "constexpr auto src_args = TensorAccessorArgs<2>();"
        result = _offset_compile_time_args_in_source(source, 1, 5)
        assert "TensorAccessorArgs<7>" in result

    def test_transform_phase_source_combines_all(self):
        """Test that _transform_phase_source applies all transforms."""
        from models.experimental.ops.descriptors.sequential import _transform_phase_source

        source = (
            'constexpr uint32_t cb = get_named_compile_time_arg_val("cb_in");\n'
            "uint32_t blk = get_compile_time_arg_val(0);\n"
            "constexpr auto args = TensorAccessorArgs<2>();\n"
            "uint32_t val = get_arg_val<uint32_t>(3);\n"
        )
        result = _transform_phase_source(source, 1, ct_arg_offset=5)
        assert "phase1_cb_in" in result
        assert "get_compile_time_arg_val(5)" in result  # offset from 0
        assert "TensorAccessorArgs<7>" in result  # offset from 2
        assert "__phase1_rt_offset + 3" in result


class TestIfdefResolution:
    """Tests for preprocessor directive resolution."""

    def test_resolve_ifdef_rmsnorm_defined(self):
        """Test resolving #ifdef RMSNORM when it's defined."""
        from models.experimental.ops.descriptors.sequential import _resolve_ifdef_directives

        source = """
int a = 1;
#ifdef RMSNORM
int b = 2;
#else
int c = 3;
#endif
int d = 4;
"""
        result = _resolve_ifdef_directives(source, {"RMSNORM"})
        assert "int a = 1" in result
        assert "int b = 2" in result
        assert "int c = 3" not in result
        assert "int d = 4" in result

    def test_resolve_ifdef_rmsnorm_not_defined(self):
        """Test resolving #ifdef RMSNORM when it's not defined."""
        from models.experimental.ops.descriptors.sequential import _resolve_ifdef_directives

        source = """
int a = 1;
#ifdef RMSNORM
int b = 2;
#else
int c = 3;
#endif
int d = 4;
"""
        result = _resolve_ifdef_directives(source, set())
        assert "int a = 1" in result
        assert "int b = 2" not in result
        assert "int c = 3" in result
        assert "int d = 4" in result

    def test_resolve_ifdef_fuse_gamma_defined(self):
        """Test resolving #ifdef FUSE_GAMMA when it's defined."""
        from models.experimental.ops.descriptors.sequential import _resolve_ifdef_directives

        source = """
int a = 1;
#ifdef FUSE_GAMMA
int gamma_code = 2;
#endif
int b = 3;
"""
        result = _resolve_ifdef_directives(source, {"FUSE_GAMMA"})
        assert "int gamma_code = 2" in result
        assert "int a = 1" in result
        assert "int b = 3" in result

    def test_resolve_ifdef_fuse_gamma_not_defined(self):
        """Test resolving #ifdef FUSE_GAMMA when it's not defined."""
        from models.experimental.ops.descriptors.sequential import _resolve_ifdef_directives

        source = """
int a = 1;
#ifdef FUSE_GAMMA
int gamma_code = 2;
#endif
int b = 3;
"""
        result = _resolve_ifdef_directives(source, set())
        assert "int gamma_code = 2" not in result
        assert "int a = 1" in result
        assert "int b = 3" in result

    def test_resolve_leaves_unknown_defines(self):
        """Test that unknown defines are left untouched."""
        from models.experimental.ops.descriptors.sequential import _resolve_ifdef_directives

        source = """
#ifdef SOME_OTHER_FLAG
int a = 1;
#endif
"""
        result = _resolve_ifdef_directives(source, set())
        # Unknown directive should pass through
        assert "#ifdef SOME_OTHER_FLAG" in result
        assert "int a = 1" in result
        assert "#endif" in result


class TestKernelBodyExtraction:
    """Tests for kernel body extraction."""

    def test_extract_simple_body(self):
        """Test extracting a simple kernel body."""
        from models.experimental.ops.descriptors.sequential import _extract_kernel_body_for_fusion

        source = """
#include "api.h"

void some_helper() {
    // helper code
}

void kernel_main() {
    int x = 1;
    int y = 2;
    compute(x, y);
}

void other_func() {
    // other
}
"""
        body = _extract_kernel_body_for_fusion(source)

        assert "int x = 1" in body
        assert "int y = 2" in body
        assert "compute(x, y)" in body
        assert "helper code" not in body
        assert "other" not in body

    def test_extract_nested_braces(self):
        """Test extracting body with nested braces."""
        from models.experimental.ops.descriptors.sequential import _extract_kernel_body_for_fusion

        source = """
void kernel_main() {
    for (int i = 0; i < 10; i++) {
        if (i > 5) {
            do_something();
        }
    }
}
"""
        body = _extract_kernel_body_for_fusion(source)
        assert "do_something" in body
        assert "for" in body

    def test_extract_empty_body(self):
        """Test extracting from source with no kernel_main."""
        from models.experimental.ops.descriptors.sequential import _extract_kernel_body_for_fusion

        source = """
void other_function() {
    int x = 1;
}
"""
        body = _extract_kernel_body_for_fusion(source)
        assert body.strip() == ""


class TestCollectIncludes:
    """Tests for include collection."""

    def test_collect_unique_includes(self):
        """Test collecting unique includes from multiple sources."""
        from models.experimental.ops.descriptors.sequential import _collect_includes

        sources = [
            '#include "common.h"\n#include "phase0.h"\nvoid kernel_main() {}',
            '#include "common.h"\n#include "phase1.h"\nvoid kernel_main() {}',
        ]

        includes = _collect_includes(sources)

        assert len(includes) == 3
        assert '#include "common.h"' in includes
        assert '#include "phase0.h"' in includes
        assert '#include "phase1.h"' in includes


class TestCollectDefines:
    """Tests for define collection."""

    def test_collect_defines_before_main(self):
        """Test collecting defines only before kernel_main."""
        from models.experimental.ops.descriptors.sequential import _collect_defines

        sources = [
            "#define FOO 1\n#define BAR 2\nvoid kernel_main() {\n#define INSIDE 3\n}",
        ]

        defines = _collect_defines(sources)
        define_strs = [d.strip() for d in defines]

        assert "#define FOO 1" in define_strs
        assert "#define BAR 2" in define_strs
        assert "#define INSIDE 3" not in define_strs


class TestInlineLocalIncludes:
    """Tests for local include inlining."""

    def test_inlines_local_include(self, tmp_path):
        """Test inlining a local include file."""
        from models.experimental.ops.descriptors.sequential import _inline_local_includes

        # Create a local header file
        header = tmp_path / "utils.h"
        header.write_text("#pragma once\nint helper() { return 42; }\n")

        source = '#include "utils.h"\nvoid kernel_main() {}\n'
        result = _inline_local_includes(source, str(tmp_path))

        assert "int helper()" in result
        assert '#include "utils.h"' not in result
        # pragma once should be stripped
        assert "#pragma once" not in result

    def test_leaves_path_includes(self):
        """Test that includes with paths are left unchanged."""
        from models.experimental.ops.descriptors.sequential import _inline_local_includes

        source = '#include "api/dataflow/dataflow_api.h"\nvoid kernel_main() {}\n'
        result = _inline_local_includes(source, "/some/dir")

        # Path includes should remain
        assert '#include "api/dataflow/dataflow_api.h"' in result

    def test_no_kernel_dir_returns_unchanged(self):
        """Test that None kernel_dir returns source unchanged."""
        from models.experimental.ops.descriptors.sequential import _inline_local_includes

        source = '#include "utils.h"\nvoid kernel_main() {}\n'
        result = _inline_local_includes(source, None)
        assert result == source


class TestMergeNamedCompileTimeArgs:
    """Tests for named compile-time arg merging."""

    def test_phase0_keeps_original_names(self):
        """Test that phase 0 keeps original arg names."""
        from models.experimental.ops.descriptors.sequential import _merge_named_compile_time_args

        phase_kernels = [
            {"reader": MagicMock(named_compile_time_args=[("cb_in", 0), ("blk", 4)])},
        ]

        result = _merge_named_compile_time_args(phase_kernels, "reader")
        names = dict(result)
        assert "cb_in" in names
        assert "blk" in names
        assert names["cb_in"] == 0
        assert names["blk"] == 4

    def test_phase1_gets_prefixed_names(self):
        """Test that phase 1+ args get prefixed."""
        from models.experimental.ops.descriptors.sequential import _merge_named_compile_time_args

        phase_kernels = [
            {"reader": MagicMock(named_compile_time_args=[("cb_in", 0)])},
            {"reader": MagicMock(named_compile_time_args=[("cb_in", 0)])},
        ]

        result = _merge_named_compile_time_args(phase_kernels, "reader")
        names = dict(result)
        assert "cb_in" in names  # Phase 0
        assert "phase1_cb_in" in names  # Phase 1

    def test_rt_arg_offsets_added(self):
        """Test that runtime arg offsets are added for phase 1+."""
        from models.experimental.ops.descriptors.sequential import _merge_named_compile_time_args

        phase_kernels = [
            {"reader": MagicMock(named_compile_time_args=[("cb_in", 0)])},
            {"reader": MagicMock(named_compile_time_args=[("cb_in", 0)])},
        ]

        rt_offsets = {0: 0, 1: 10}
        result = _merge_named_compile_time_args(phase_kernels, "reader", rt_offsets)
        names = dict(result)
        assert "phase1_rt_arg_offset" in names
        assert names["phase1_rt_arg_offset"] == 10

    def test_barrier_config_added(self):
        """Test that barrier config named args are added."""
        from models.experimental.ops.descriptors.sequential import _merge_named_compile_time_args, BarrierConfig

        phase_kernels = [
            {"reader": MagicMock(named_compile_time_args=[])},
        ]

        bc = BarrierConfig(
            num_cores=2,
            core0_phys_x=1,
            core0_phys_y=1,
            mcast_start_x=1,
            mcast_start_y=1,
            mcast_end_x=2,
            mcast_end_y=1,
        )
        result = _merge_named_compile_time_args(
            phase_kernels,
            "reader",
            barrier_rt_offset=10,
            barrier_config=bc,
        )
        names = dict(result)
        assert "barrier_rt_offset" in names
        assert names["barrier_rt_offset"] == 10
        assert "num_barrier_cores" in names
        assert names["num_barrier_cores"] == 2


class TestComputeRuntimeArgOffsets:
    """Tests for runtime arg offset computation."""

    @staticmethod
    def _make_runtime_args_view(args_per_core):
        """Create a mock RuntimeArgsView.

        args_per_core: list of lists, one per core.
        rv[col_idx] -> col_proxy, col_proxy[0] -> VectorUInt32
        """
        view = MagicMock()
        view.__len__ = MagicMock(return_value=len(args_per_core))
        cols = []
        for args in args_per_core:
            col = MagicMock()
            col.__getitem__ = MagicMock(return_value=args)
            cols.append(col)
        view.__getitem__ = MagicMock(side_effect=lambda i: cols[i])
        return view

    def test_basic_offsets(self):
        """Test basic runtime arg offset computation."""
        from models.experimental.ops.descriptors.sequential import _compute_runtime_arg_offsets

        kernel0 = MagicMock()
        kernel0.runtime_args = self._make_runtime_args_view([[1, 2, 3, 4, 5]])

        kernel1 = MagicMock()
        kernel1.runtime_args = self._make_runtime_args_view([[10, 20, 30]])

        phase_kernels = [
            {"reader": kernel0},
            {"reader": kernel1},
        ]

        offsets = _compute_runtime_arg_offsets(phase_kernels, "reader")
        assert offsets[0] == 0
        assert offsets[1] == 5  # Phase 0 had 5 args

    def test_offsets_with_missing_kernel(self):
        """Test offsets when a phase has no kernel of that type."""
        from models.experimental.ops.descriptors.sequential import _compute_runtime_arg_offsets

        kernel0 = MagicMock()
        kernel0.runtime_args = self._make_runtime_args_view([[1, 2, 3]])

        phase_kernels = [
            {"reader": kernel0},
            {"reader": None},
        ]

        offsets = _compute_runtime_arg_offsets(phase_kernels, "reader")
        assert offsets[0] == 0
        assert offsets[1] == 3


class TestConcatenateRuntimeArgs:
    """Tests for runtime arg concatenation."""

    @staticmethod
    def _make_runtime_args_view(args_per_core):
        """Create a mock RuntimeArgsView."""
        view = MagicMock()
        view.__len__ = MagicMock(return_value=len(args_per_core))
        cols = []
        for args in args_per_core:
            col = MagicMock()
            col.__getitem__ = MagicMock(return_value=args)
            cols.append(col)
        view.__getitem__ = MagicMock(side_effect=lambda i: cols[i])
        return view

    @staticmethod
    def _make_core_ranges():
        """Create a mock CoreRangeSet with one core at (0,0)."""
        core_range = MagicMock()
        core_range.start.x = 0
        core_range.start.y = 0
        core_range.end.x = 0
        core_range.end.y = 0
        core_range_set = MagicMock()
        core_range_set.ranges.return_value = [core_range]
        return core_range_set

    def test_basic_concatenation(self):
        """Test basic runtime arg concatenation across phases."""
        from models.experimental.ops.descriptors.sequential import _concatenate_runtime_args

        core_ranges = self._make_core_ranges()

        kernel0 = MagicMock()
        kernel0.runtime_args = self._make_runtime_args_view([[1, 2, 3]])
        kernel0.core_ranges = core_ranges

        kernel1 = MagicMock()
        kernel1.runtime_args = self._make_runtime_args_view([[10, 20]])
        kernel1.core_ranges = core_ranges

        phase_kernels = [
            {"reader": kernel0},
            {"reader": kernel1},
        ]

        result = _concatenate_runtime_args(phase_kernels, "reader")
        assert len(result) == 1  # One core
        core_coord, args = result[0]
        assert args == [1, 2, 3, 10, 20]


class TestMergeCompileTimeArgs:
    """Tests for compile-time arg concatenation."""

    def test_concatenates_all_phases(self):
        """Test that compile-time args from all phases are concatenated."""
        from models.experimental.ops.descriptors.sequential import _merge_compile_time_args

        phase_kernels = [
            {"reader": MagicMock(compile_time_args=[10, 20, 30])},
            {"reader": MagicMock(compile_time_args=[40, 50])},
        ]

        merged, offsets = _merge_compile_time_args(phase_kernels, "reader")
        assert merged == [10, 20, 30, 40, 50]
        assert offsets == {0: 0, 1: 3}

    def test_handles_missing_kernel(self):
        """Test offset computation with a missing kernel."""
        from models.experimental.ops.descriptors.sequential import _merge_compile_time_args

        phase_kernels = [
            {"reader": MagicMock(compile_time_args=[10, 20])},
            {"reader": None},
            {"reader": MagicMock(compile_time_args=[30])},
        ]

        merged, offsets = _merge_compile_time_args(phase_kernels, "reader")
        assert merged == [10, 20, 30]
        assert offsets == {0: 0, 1: 2, 2: 2}


class TestMergeDefines:
    """Tests for define merging."""

    def test_common_defines_kept(self):
        """Test that common defines (REDUCE_OP etc.) are kept once."""
        from models.experimental.ops.descriptors.sequential import _merge_defines

        phase_kernels = [
            {"compute": MagicMock(defines=[("REDUCE_OP", "PoolType::SUM"), ("CUSTOM", "1")])},
            {"compute": MagicMock(defines=[("REDUCE_OP", "PoolType::SUM"), ("CUSTOM", "2")])},
        ]

        result = _merge_defines(phase_kernels, "compute")
        names = [name for name, _ in result]

        # REDUCE_OP should appear once
        assert names.count("REDUCE_OP") == 1
        # CUSTOM from phase 0 keeps original name, phase 1 gets prefixed
        assert "CUSTOM" in names
        assert "PHASE1_CUSTOM" in names

    def test_source_level_defines_excluded(self):
        """Test that source-level defines (RMSNORM, FUSE_GAMMA, etc.) are excluded."""
        from models.experimental.ops.descriptors.sequential import _merge_defines

        phase_kernels = [
            {
                "compute": MagicMock(
                    defines=[
                        ("RMSNORM", "1"),
                        ("FUSE_PRE_ADD", "1"),
                        ("FUSE_GAMMA", "1"),
                        ("FUSE_BETA", "1"),
                        ("NORMAL", "val"),
                    ]
                )
            },
        ]

        result = _merge_defines(phase_kernels, "compute")
        names = [name for name, _ in result]

        assert "RMSNORM" not in names
        assert "FUSE_PRE_ADD" not in names
        assert "FUSE_GAMMA" not in names
        assert "FUSE_BETA" not in names
        assert "NORMAL" in names


class TestValidateFp32Consistency:
    """Tests for fp32_dest_acc_en validation."""

    def test_consistent_fp32_passes(self):
        """Test that consistent fp32 settings pass validation."""
        from models.experimental.ops.descriptors.sequential import _validate_fp32_consistency

        mock_kernel1 = MagicMock()
        mock_kernel1.config = MagicMock(fp32_dest_acc_en=True)

        mock_kernel2 = MagicMock()
        mock_kernel2.config = MagicMock(fp32_dest_acc_en=True)

        desc1 = MagicMock()
        desc1.descriptor.kernels = [mock_kernel1]
        desc2 = MagicMock()
        desc2.descriptor.kernels = [mock_kernel2]

        # Should not raise
        _validate_fp32_consistency([desc1, desc2])

    def test_inconsistent_fp32_raises(self):
        """Test that inconsistent fp32 settings raise ValueError."""
        from models.experimental.ops.descriptors.sequential import _validate_fp32_consistency

        mock_kernel1 = MagicMock()
        mock_kernel1.config = MagicMock(fp32_dest_acc_en=True)

        mock_kernel2 = MagicMock()
        mock_kernel2.config = MagicMock(fp32_dest_acc_en=False)

        desc1 = MagicMock()
        desc1.descriptor.kernels = [mock_kernel1]
        desc2 = MagicMock()
        desc2.descriptor.kernels = [mock_kernel2]

        with pytest.raises(ValueError, match="fp32_dest_acc_en mismatch"):
            _validate_fp32_consistency([desc1, desc2])


class TestKernelClassification:
    """Tests for kernel type classification."""

    @staticmethod
    def _setup_config_types():
        """Set up real classes on mocked ttnn so isinstance() works."""
        import ttnn

        # Create real classes for config descriptors (isinstance needs real types)
        class _Compute:
            pass

        class _Reader:
            pass

        class _Writer:
            pass

        ttnn.ComputeConfigDescriptor = _Compute
        ttnn.ReaderConfigDescriptor = _Reader
        ttnn.WriterConfigDescriptor = _Writer
        return _Compute, _Reader, _Writer

    def test_classify_compute(self):
        """Test classifying a compute kernel."""
        from models.experimental.ops.descriptors.sequential import _classify_kernel

        Compute, _, _ = self._setup_config_types()
        kernel = MagicMock()
        kernel.config = Compute()
        assert _classify_kernel(kernel) == "compute"

    def test_classify_reader(self):
        """Test classifying a reader kernel."""
        from models.experimental.ops.descriptors.sequential import _classify_kernel

        _, Reader, _ = self._setup_config_types()
        kernel = MagicMock()
        kernel.config = Reader()
        assert _classify_kernel(kernel) == "reader"

    def test_classify_writer(self):
        """Test classifying a writer kernel."""
        from models.experimental.ops.descriptors.sequential import _classify_kernel

        _, _, Writer = self._setup_config_types()
        kernel = MagicMock()
        kernel.config = Writer()
        assert _classify_kernel(kernel) == "writer"


class TestCollectPreMainCode:
    """Tests for pre-main code collection."""

    def test_collects_namespace_code(self):
        """Test that namespace code before kernel_main is collected."""
        from models.experimental.ops.descriptors.sequential import _collect_pre_main_code

        source = """#include "api.h"
#define FOO 1

namespace my_ns {
    constexpr int val = 42;
}

void kernel_main() {
    // body
}
"""
        result = _collect_pre_main_code(source)
        assert "namespace my_ns" in result
        assert "constexpr int val = 42" in result
        # Should not include includes, defines, or comments
        assert "#include" not in result
        assert "#define" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
