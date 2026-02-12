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
from unittest.mock import MagicMock


# Mock ttnn before importing the descriptor modules (which do `import ttnn` at
# module level).  We snapshot sys.modules BEFORE the mock, import the descriptor
# modules under the mock, keep private references to them, then FULLY UNDO
# everything in sys.modules so that test files collected later (e.g.
# test_sequential.py) re-import from scratch with the real ttnn.
#
# The standalone test functions below access the mock-backed modules via the
# private module-level references (_mock_sequential / _mock_cpp_parser) instead
# of doing a bare `from models.experimental.ops... import X`.

_modules_before = set(sys.modules)
_ttnn_mocked_keys = ["ttnn", "ttnn._ttnn", "ttnn._ttnn.program_descriptor"]
_ttnn_originals = {k: sys.modules.get(k) for k in _ttnn_mocked_keys}

for _k in _ttnn_mocked_keys:
    sys.modules[_k] = MagicMock()

import models.experimental.ops.descriptors.sequential as _mock_sequential  # noqa: E402
import models.experimental.ops.descriptors.cpp_parser as _mock_cpp_parser  # noqa: E402

# Remove every module that was added during the mock window.
for _k in set(sys.modules) - _modules_before:
    del sys.modules[_k]

# Restore the original ttnn entries (or remove mock entries).
for _k in _ttnn_mocked_keys:
    if _ttnn_originals[_k] is not None:
        sys.modules[_k] = _ttnn_originals[_k]
    else:
        sys.modules.pop(_k, None)

del _modules_before, _ttnn_originals, _ttnn_mocked_keys, _k


class TestCBInfo:
    """Tests for CBInfo dataclass."""

    def test_creation(self):
        """Test creating CBInfo."""
        CBInfo = _mock_sequential.CBInfo

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
        PhaseInfo = _mock_sequential.PhaseInfo
        CBInfo = _mock_sequential.CBInfo

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
        PhaseInfo = _mock_sequential.PhaseInfo

        phase = PhaseInfo(phase_idx=1, op_descriptor=MagicMock())
        assert phase.cb_info == {}


class TestExtractCBInfo:
    """Tests for extract_cb_info function."""

    def test_extract_from_descriptor(self):
        """Test extracting CB info from a mock descriptor."""
        extract_cb_info = _mock_sequential.extract_cb_info

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
        cb_desc.has_global_circular_buffer.return_value = False

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
        extract_cb_info = _mock_sequential.extract_cb_info

        fmt1 = MagicMock(buffer_index=0, data_format="F16", page_size=1024)
        fmt2 = MagicMock(buffer_index=16, data_format="F32", page_size=2048)

        cb1 = MagicMock(total_size=2048, core_ranges="r1", format_descriptors=[fmt1])
        cb1.has_global_circular_buffer.return_value = False
        cb2 = MagicMock(total_size=4096, core_ranges="r2", format_descriptors=[fmt2])
        cb2.has_global_circular_buffer.return_value = False

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
        extract_cb_names_from_kernel = _mock_sequential.extract_cb_names_from_kernel

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
        SequentialChainBuilder = _mock_sequential.SequentialChainBuilder

        builder = SequentialChainBuilder()
        assert len(builder.phases) == 0
        assert builder._built is False

    def test_add_phase(self):
        """Test adding phases to the builder."""
        SequentialChainBuilder = _mock_sequential.SequentialChainBuilder

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
        SequentialChainBuilder = _mock_sequential.SequentialChainBuilder

        builder = SequentialChainBuilder()

        mock_desc = MagicMock()
        mock_desc.descriptor = MagicMock(cbs=[])

        result = builder.add_phase(mock_desc).add_phase(mock_desc)

        assert result is builder
        assert len(builder.phases) == 2

    def test_single_phase_returns_original(self):
        """Test that single-phase build returns original descriptor."""
        SequentialChainBuilder = _mock_sequential.SequentialChainBuilder

        builder = SequentialChainBuilder()

        mock_desc = MagicMock()
        mock_desc.descriptor = MagicMock(cbs=[])

        builder.add_phase(mock_desc)
        result = builder.build(device=MagicMock())

        assert result is mock_desc

    def test_build_empty_raises(self):
        """Test that building empty chain raises ValueError."""
        SequentialChainBuilder = _mock_sequential.SequentialChainBuilder

        builder = SequentialChainBuilder()

        with pytest.raises(ValueError, match="no phases"):
            builder.build(device=MagicMock())

    def test_build_twice_raises(self):
        """Test that building twice raises ValueError."""
        SequentialChainBuilder = _mock_sequential.SequentialChainBuilder

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
        chain_descriptors = _mock_sequential.chain_descriptors

        mock_desc = MagicMock()
        mock_desc.descriptor = MagicMock(cbs=[])

        result = chain_descriptors([mock_desc], device=MagicMock())
        assert result is mock_desc

    def test_empty_list_raises(self):
        """Test that empty list raises ValueError."""
        chain_descriptors = _mock_sequential.chain_descriptors

        with pytest.raises(ValueError, match="no phases"):
            chain_descriptors([], device=MagicMock())


class TestParallelChainsStandalone:
    """Standalone tests for parallel chain functions."""

    def test_create_parallel_chain_empty(self):
        """Test creating parallel chains with empty list."""
        create_parallel_chain_descriptors = _mock_sequential.create_parallel_chain_descriptors

        result = create_parallel_chain_descriptors([], device=MagicMock())
        assert result == []

    def test_create_parallel_chain_single_ops(self):
        """Test creating parallel chains with single-op chains."""
        create_parallel_chain_descriptors = _mock_sequential.create_parallel_chain_descriptors

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
        _prefix_named_args_in_source = _mock_sequential._prefix_named_args_in_source

        source = 'constexpr uint32_t cb = get_named_compile_time_arg_val("cb_in");'
        result = _prefix_named_args_in_source(source, 0)
        assert result == source

    def test_prefix_named_args_phase1(self):
        """Test that phase 1 gets prefixed named args."""
        _prefix_named_args_in_source = _mock_sequential._prefix_named_args_in_source

        source = 'constexpr uint32_t cb = get_named_compile_time_arg_val("cb_in");'
        result = _prefix_named_args_in_source(source, 1)
        assert 'get_named_compile_time_arg_val("phase1_cb_in")' in result

    def test_prefix_named_args_phase2(self):
        """Test phase 2 prefix."""
        _prefix_named_args_in_source = _mock_sequential._prefix_named_args_in_source

        source = 'constexpr uint32_t blk = get_named_compile_time_arg_val("blk");'
        result = _prefix_named_args_in_source(source, 2)
        assert 'get_named_compile_time_arg_val("phase2_blk")' in result

    def test_offset_runtime_args_phase0_unchanged(self):
        """Test that phase 0 runtime args are unchanged."""
        _offset_runtime_args_in_source = _mock_sequential._offset_runtime_args_in_source

        source = "uint32_t val = get_arg_val<uint32_t>(3);"
        result = _offset_runtime_args_in_source(source, 0)
        assert result == source

    def test_offset_runtime_args_phase1(self):
        """Test phase 1 runtime arg offsetting."""
        _offset_runtime_args_in_source = _mock_sequential._offset_runtime_args_in_source

        source = "uint32_t val = get_arg_val<uint32_t>(3);"
        result = _offset_runtime_args_in_source(source, 1)
        assert "__phase1_rt_offset + 3" in result
        assert "phase1_rt_arg_offset" in result

    def test_offset_runtime_args_variable_init(self):
        """Test that incrementing variable init pattern is offset."""
        _offset_runtime_args_in_source = _mock_sequential._offset_runtime_args_in_source

        source = "uint32_t rt_args_idx = 0;"
        result = _offset_runtime_args_in_source(source, 1)
        assert "__phase1_rt_offset" in result
        # The variable should be initialized to the offset, not 0
        assert "= 0;" not in result or "__phase1_rt_offset" in result

    def test_offset_compile_time_args_phase0_unchanged(self):
        """Test that phase 0 positional args are unchanged."""
        _offset_compile_time_args_in_source = _mock_sequential._offset_compile_time_args_in_source

        source = "uint32_t blk = get_compile_time_arg_val(0);"
        result = _offset_compile_time_args_in_source(source, 0, 0)
        assert result == source

    def test_offset_compile_time_args_phase1(self):
        """Test that phase 1 positional args are offset."""
        _offset_compile_time_args_in_source = _mock_sequential._offset_compile_time_args_in_source

        source = "uint32_t blk = get_compile_time_arg_val(0);\nuint32_t mode = get_compile_time_arg_val(1);"
        result = _offset_compile_time_args_in_source(source, 1, 3)
        assert "get_compile_time_arg_val(3)" in result
        assert "get_compile_time_arg_val(4)" in result

    def test_offset_compile_time_args_tensor_accessor(self):
        """Test that TensorAccessorArgs<N> is also offset."""
        _offset_compile_time_args_in_source = _mock_sequential._offset_compile_time_args_in_source

        source = "constexpr auto src_args = TensorAccessorArgs<2>();"
        result = _offset_compile_time_args_in_source(source, 1, 5)
        assert "TensorAccessorArgs<7>" in result

    def test_transform_phase_source_combines_all(self):
        """Test that _transform_phase_source applies all transforms."""
        _transform_phase_source = _mock_sequential._transform_phase_source

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
        resolve_ifdef_directives = _mock_cpp_parser.resolve_ifdef_directives

        source = """
int a = 1;
#ifdef RMSNORM
int b = 2;
#else
int c = 3;
#endif
int d = 4;
"""
        result = resolve_ifdef_directives(source, {"RMSNORM"})
        assert "int a = 1" in result
        assert "int b = 2" in result
        assert "int c = 3" not in result
        assert "int d = 4" in result

    def test_resolve_ifdef_rmsnorm_not_defined(self):
        """Test resolving #ifdef RMSNORM when it's not defined."""
        resolve_ifdef_directives = _mock_cpp_parser.resolve_ifdef_directives

        source = """
int a = 1;
#ifdef RMSNORM
int b = 2;
#else
int c = 3;
#endif
int d = 4;
"""
        result = resolve_ifdef_directives(source, set())
        assert "int a = 1" in result
        assert "int b = 2" not in result
        assert "int c = 3" in result
        assert "int d = 4" in result

    def test_resolve_ifdef_fuse_gamma_defined(self):
        """Test resolving #ifdef FUSE_GAMMA when it's defined."""
        resolve_ifdef_directives = _mock_cpp_parser.resolve_ifdef_directives

        source = """
int a = 1;
#ifdef FUSE_GAMMA
int gamma_code = 2;
#endif
int b = 3;
"""
        result = resolve_ifdef_directives(source, {"FUSE_GAMMA"})
        assert "int gamma_code = 2" in result
        assert "int a = 1" in result
        assert "int b = 3" in result

    def test_resolve_ifdef_fuse_gamma_not_defined(self):
        """Test resolving #ifdef FUSE_GAMMA when it's not defined."""
        resolve_ifdef_directives = _mock_cpp_parser.resolve_ifdef_directives

        source = """
int a = 1;
#ifdef FUSE_GAMMA
int gamma_code = 2;
#endif
int b = 3;
"""
        result = resolve_ifdef_directives(source, set())
        assert "int gamma_code = 2" not in result
        assert "int a = 1" in result
        assert "int b = 3" in result

    def test_resolve_leaves_unknown_defines(self):
        """Test that unknown defines are left untouched."""
        resolve_ifdef_directives = _mock_cpp_parser.resolve_ifdef_directives

        source = """
#ifdef SOME_OTHER_FLAG
int a = 1;
#endif
"""
        result = resolve_ifdef_directives(source, set())
        # Unknown directive should pass through
        assert "#ifdef SOME_OTHER_FLAG" in result
        assert "int a = 1" in result
        assert "#endif" in result


class TestKernelBodyExtraction:
    """Tests for kernel body extraction."""

    def test_extract_simple_body(self):
        """Test extracting a simple kernel body."""
        extract_kernel_body = _mock_cpp_parser.extract_kernel_body

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
        body = extract_kernel_body(source)

        assert "int x = 1" in body
        assert "int y = 2" in body
        assert "compute(x, y)" in body
        assert "helper code" not in body
        assert "other" not in body

    def test_extract_nested_braces(self):
        """Test extracting body with nested braces."""
        extract_kernel_body = _mock_cpp_parser.extract_kernel_body

        source = """
void kernel_main() {
    for (int i = 0; i < 10; i++) {
        if (i > 5) {
            do_something();
        }
    }
}
"""
        body = extract_kernel_body(source)
        assert "do_something" in body
        assert "for" in body

    def test_extract_empty_body(self):
        """Test extracting from source with no kernel_main."""
        extract_kernel_body = _mock_cpp_parser.extract_kernel_body

        source = """
void other_function() {
    int x = 1;
}
"""
        body = extract_kernel_body(source)
        assert body.strip() == ""


class TestCollectIncludes:
    """Tests for include collection."""

    def test_collect_unique_includes(self):
        """Test collecting unique includes from multiple sources."""
        collect_includes = _mock_cpp_parser.collect_includes

        sources = [
            '#include "common.h"\n#include "phase0.h"\nvoid kernel_main() {}',
            '#include "common.h"\n#include "phase1.h"\nvoid kernel_main() {}',
        ]

        includes = collect_includes(sources)

        assert len(includes) == 3
        assert '#include "common.h"' in includes
        assert '#include "phase0.h"' in includes
        assert '#include "phase1.h"' in includes


class TestCollectDefines:
    """Tests for define collection."""

    def test_collect_defines_before_main(self):
        """Test collecting defines only before kernel_main."""
        collect_defines = _mock_cpp_parser.collect_defines

        sources = [
            "#define FOO 1\n#define BAR 2\nvoid kernel_main() {\n#define INSIDE 3\n}",
        ]

        defines = collect_defines(sources)
        define_strs = [d.strip() for d in defines]

        assert "#define FOO 1" in define_strs
        assert "#define BAR 2" in define_strs
        assert "#define INSIDE 3" not in define_strs


class TestInlineLocalIncludes:
    """Tests for local include inlining."""

    def test_inlines_local_include(self, tmp_path):
        """Test inlining a local include file."""
        inline_local_includes = _mock_cpp_parser.inline_local_includes

        # Create a local header file
        header = tmp_path / "utils.h"
        header.write_text("#pragma once\nint helper() { return 42; }\n")

        source = '#include "utils.h"\nvoid kernel_main() {}\n'
        result = inline_local_includes(source, str(tmp_path))

        assert "int helper()" in result
        assert '#include "utils.h"' not in result
        # pragma once should be stripped
        assert "#pragma once" not in result

    def test_leaves_path_includes(self):
        """Test that includes with paths are left unchanged."""
        inline_local_includes = _mock_cpp_parser.inline_local_includes

        source = '#include "api/dataflow/dataflow_api.h"\nvoid kernel_main() {}\n'
        result = inline_local_includes(source, "/some/dir")

        # Path includes should remain
        assert '#include "api/dataflow/dataflow_api.h"' in result

    def test_no_kernel_dir_returns_unchanged(self):
        """Test that None kernel_dir returns source unchanged."""
        inline_local_includes = _mock_cpp_parser.inline_local_includes

        source = '#include "utils.h"\nvoid kernel_main() {}\n'
        result = inline_local_includes(source, None)
        assert result == source


class TestMergeNamedCompileTimeArgs:
    """Tests for named compile-time arg merging."""

    def test_phase0_keeps_original_names(self):
        """Test that phase 0 keeps original arg names."""
        _merge_named_compile_time_args = _mock_sequential._merge_named_compile_time_args

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
        _merge_named_compile_time_args = _mock_sequential._merge_named_compile_time_args

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
        _merge_named_compile_time_args = _mock_sequential._merge_named_compile_time_args

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
        _merge_named_compile_time_args = _mock_sequential._merge_named_compile_time_args
        BarrierConfig = _mock_sequential.BarrierConfig

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


def _make_mock_core_ranges():
    """Create a mock CoreRangeSet with one core at (0,0)."""
    core_range = MagicMock()
    core_range.start.x = 0
    core_range.start.y = 0
    core_range.end.x = 0
    core_range.end.y = 0
    core_range_set = MagicMock()
    core_range_set.ranges.return_value = [core_range]
    return core_range_set


def _make_mock_runtime_args_view(args_per_core):
    """Create a mock RuntimeArgsView with coordinate-based 2D indexing.

    args_per_core: list of lists, one per core.
    rv[x][y] -> VectorUInt32 of args for CoreCoord(x, y).
    Mock uses a function-based getitem to handle MagicMock keys
    (MagicMock.__index__() returns 1, not 0, causing IndexError with lists).
    """
    view = MagicMock()
    view.__len__ = MagicMock(return_value=len(args_per_core))

    def get_col(_x):
        col = MagicMock()
        col.__getitem__ = MagicMock(return_value=args_per_core[0])
        return col

    view.__getitem__ = MagicMock(side_effect=get_col)
    return view


class TestComputeRuntimeArgOffsets:
    """Tests for runtime arg offset computation."""

    def test_basic_offsets(self):
        """Test basic runtime arg offset computation."""
        _compute_runtime_arg_offsets = _mock_sequential._compute_runtime_arg_offsets

        core_ranges = _make_mock_core_ranges()

        kernel0 = MagicMock()
        kernel0.runtime_args = _make_mock_runtime_args_view([[1, 2, 3, 4, 5]])
        kernel0.core_ranges = core_ranges

        kernel1 = MagicMock()
        kernel1.runtime_args = _make_mock_runtime_args_view([[10, 20, 30]])
        kernel1.core_ranges = core_ranges

        phase_kernels = [
            {"reader": kernel0},
            {"reader": kernel1},
        ]

        offsets = _compute_runtime_arg_offsets(phase_kernels, "reader")
        assert offsets[0] == 0
        assert offsets[1] == 5  # Phase 0 had 5 args

    def test_offsets_with_missing_kernel(self):
        """Test offsets when a phase has no kernel of that type."""
        _compute_runtime_arg_offsets = _mock_sequential._compute_runtime_arg_offsets

        core_ranges = _make_mock_core_ranges()

        kernel0 = MagicMock()
        kernel0.runtime_args = _make_mock_runtime_args_view([[1, 2, 3]])
        kernel0.core_ranges = core_ranges

        phase_kernels = [
            {"reader": kernel0},
            {"reader": None},
        ]

        offsets = _compute_runtime_arg_offsets(phase_kernels, "reader")
        assert offsets[0] == 0
        assert offsets[1] == 3


class TestConcatenateRuntimeArgs:
    """Tests for runtime arg concatenation."""

    def test_basic_concatenation(self):
        """Test basic runtime arg concatenation across phases."""
        _concatenate_runtime_args = _mock_sequential._concatenate_runtime_args

        core_ranges = _make_mock_core_ranges()

        kernel0 = MagicMock()
        kernel0.runtime_args = _make_mock_runtime_args_view([[1, 2, 3]])
        kernel0.core_ranges = core_ranges

        kernel1 = MagicMock()
        kernel1.runtime_args = _make_mock_runtime_args_view([[10, 20]])
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
        _merge_compile_time_args = _mock_sequential._merge_compile_time_args

        phase_kernels = [
            {"reader": MagicMock(compile_time_args=[10, 20, 30])},
            {"reader": MagicMock(compile_time_args=[40, 50])},
        ]

        merged, offsets = _merge_compile_time_args(phase_kernels, "reader")
        assert merged == [10, 20, 30, 40, 50]
        assert offsets == {0: 0, 1: 3}

    def test_handles_missing_kernel(self):
        """Test offset computation with a missing kernel."""
        _merge_compile_time_args = _mock_sequential._merge_compile_time_args

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
        _merge_defines = _mock_sequential._merge_defines

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
        _merge_defines = _mock_sequential._merge_defines

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
        _validate_fp32_consistency = _mock_sequential._validate_fp32_consistency

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
        _validate_fp32_consistency = _mock_sequential._validate_fp32_consistency

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


class TestCategorizePreMain:
    """Tests for tree-sitter-based pre-main code categorization."""

    def test_categorizes_namespace_and_functions(self):
        """Test that namespace code, functions, and variables are categorized."""
        categorize_pre_main = _mock_cpp_parser.categorize_pre_main

        source = """#include "api.h"
#define FOO 1

namespace my_ns {
    constexpr int val = 42;
}

ALWI void ACQ() { acquire_dst(); }

constexpr uint32_t MAX_TILES = 64;

void kernel_main() {
    // body
}
"""
        blocks = categorize_pre_main(source)
        kinds = [b.kind for b in blocks]
        assert "namespace" in kinds
        assert "function" in kinds
        assert "variable" in kinds
        # Find the function block
        func_blocks = [b for b in blocks if b.kind == "function"]
        assert len(func_blocks) == 1
        assert func_blocks[0].name == "ACQ"
        # Find the variable block
        var_blocks = [b for b in blocks if b.kind == "variable"]
        assert len(var_blocks) == 1
        assert var_blocks[0].name == "MAX_TILES"

    def test_skips_includes_defines_comments(self):
        """Pre-main should not include #include, #define, or comments."""
        categorize_pre_main = _mock_cpp_parser.categorize_pre_main

        source = """#include <cstdint>
#define FOO 1
// a comment
/* block comment */
namespace g = norm;
void kernel_main() {}
"""
        blocks = categorize_pre_main(source)
        for b in blocks:
            assert not b.text.startswith("#include")
            assert not b.text.startswith("#define")
            assert not b.text.startswith("//")

    def test_function_with_alwi(self):
        """ALWI-prefixed functions are correctly detected."""
        categorize_pre_main = _mock_cpp_parser.categorize_pre_main

        source = """ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }
void kernel_main() {}
"""
        blocks = categorize_pre_main(source)
        func_blocks = [b for b in blocks if b.kind == "function"]
        assert len(func_blocks) == 2
        names = {b.name for b in func_blocks}
        assert "ACQ" in names
        assert "REL" in names

    def test_namespace_alias(self):
        """Namespace aliases are categorized correctly."""
        categorize_pre_main = _mock_cpp_parser.categorize_pre_main

        source = """namespace generic = norm::kernel_util::generic;
namespace kutil = norm::kernel_util;
void kernel_main() {}
"""
        blocks = categorize_pre_main(source)
        assert len(blocks) == 2
        for b in blocks:
            assert b.kind == "namespace_alias"

    def test_using_declaration(self):
        """Using declarations are categorized correctly."""
        categorize_pre_main = _mock_cpp_parser.categorize_pre_main

        source = """using std::uint32_t;
void kernel_main() {}
"""
        blocks = categorize_pre_main(source)
        assert len(blocks) == 1
        assert blocks[0].kind == "using"

    def test_template_function(self):
        """Template functions are categorized as functions."""
        categorize_pre_main = _mock_cpp_parser.categorize_pre_main

        source = """template <typename T>
inline void process(T x) {
    return;
}
void kernel_main() {}
"""
        blocks = categorize_pre_main(source)
        func_blocks = [b for b in blocks if b.kind == "function"]
        assert len(func_blocks) == 1
        assert func_blocks[0].name == "process"

    def test_struct_definition(self):
        """Struct definitions are categorized as struct."""
        categorize_pre_main = _mock_cpp_parser.categorize_pre_main

        source = """struct Config {
    int x;
    int y;
};
void kernel_main() {}
"""
        blocks = categorize_pre_main(source)
        struct_blocks = [b for b in blocks if b.kind == "struct"]
        assert len(struct_blocks) == 1

    def test_variable_types(self):
        """Various variable declaration types are detected."""
        categorize_pre_main = _mock_cpp_parser.categorize_pre_main

        source = """constexpr uint32_t MAX_TILES = 64;
static uint32_t counter = 0;
volatile float* data_ptr;
void kernel_main() {}
"""
        blocks = categorize_pre_main(source)
        var_blocks = [b for b in blocks if b.kind == "variable"]
        assert len(var_blocks) == 3
        var_names = {b.name for b in var_blocks}
        assert "MAX_TILES" in var_names
        assert "counter" in var_names
        assert "data_ptr" in var_names

    def test_empty_source(self):
        """Empty source returns no blocks."""
        categorize_pre_main = _mock_cpp_parser.categorize_pre_main

        blocks = categorize_pre_main("void kernel_main() {}")
        assert len(blocks) == 0

    def test_stops_at_kernel_main(self):
        """Should not include code from inside or after kernel_main."""
        categorize_pre_main = _mock_cpp_parser.categorize_pre_main

        source = """int before = 1;
void kernel_main() {
    int inside = 2;
}
int after = 3;
"""
        blocks = categorize_pre_main(source)
        all_text = " ".join(b.text for b in blocks)
        assert "before" in all_text
        assert "inside" not in all_text
        assert "after" not in all_text


class TestCollectAllPreMainCode:
    """Tests for robust pre-main merging across phases.

    Functions and global variables are prefixed with phaseN_ for ALL phases
    (including phase 0) to avoid redefinition errors.  Shared items like
    namespace blocks and using declarations are deduped.
    """

    def _make_source(self, pre_main_body: str) -> str:
        """Wrap pre-main code in a minimal kernel source."""
        return f"{pre_main_body}\n\nvoid kernel_main() {{\n    // body\n}}\n"

    def test_single_phase_functions_prefixed(self):
        """Single phase: functions get phase0_ prefix."""
        _collect_all_pre_main_code = _mock_sequential._collect_all_pre_main_code

        source = self._make_source("ALWI void ACQ() { acquire_dst(); }")
        result, names = _collect_all_pre_main_code([(0, source)])
        assert "phase0_ACQ" in result
        assert names == {0: ["ACQ"]}

    def test_single_phase_shared_items_no_prefix(self):
        """Single phase: namespace aliases and using declarations NOT prefixed."""
        _collect_all_pre_main_code = _mock_sequential._collect_all_pre_main_code

        source = self._make_source("namespace g = norm;")
        result, names = _collect_all_pre_main_code([(0, source)])
        assert "namespace g = norm;" in result
        assert 0 not in names  # no phase-specific items

    def test_identical_functions_both_prefixed(self):
        """Identical function from two phases: both emitted with distinct prefixes."""
        _collect_all_pre_main_code = _mock_sequential._collect_all_pre_main_code

        source = self._make_source("namespace g = norm;\n\nALWI void ACQ() { acquire_dst(); }")
        result, names = _collect_all_pre_main_code([(0, source), (1, source)])
        # Namespace alias: deduped (shared)
        assert result.count("namespace g = norm") == 1
        # Function: each phase gets its own prefixed copy
        assert "phase0_ACQ" in result
        assert "phase1_ACQ" in result
        assert names[0] == ["ACQ"]
        assert names[1] == ["ACQ"]

    def test_different_helpers_both_prefixed(self):
        """Different helper functions from different ops are both prefixed."""
        _collect_all_pre_main_code = _mock_sequential._collect_all_pre_main_code

        source0 = self._make_source("ALWI void helper_a() { return; }")
        source1 = self._make_source("ALWI void helper_b() { return; }")
        result, names = _collect_all_pre_main_code([(0, source0), (1, source1)])
        assert "phase0_helper_a" in result
        assert "phase1_helper_b" in result

    def test_same_signature_different_body_both_emitted(self):
        """Same function signature but different body: both emitted with prefixes."""
        _collect_all_pre_main_code = _mock_sequential._collect_all_pre_main_code

        source0 = self._make_source("inline void process() {\n    int x = 1;\n}")
        source1 = self._make_source("inline void process() {\n    int x = 2;\n}")
        result, names = _collect_all_pre_main_code([(0, source0), (1, source1)])
        # Both versions kept with phase prefixes
        assert "phase0_process" in result
        assert "phase1_process" in result
        assert "int x = 1" in result
        assert "int x = 2" in result

    def test_global_vars_all_phases_prefixed(self):
        """Global variables from ALL phases get phase-prefixed."""
        _collect_all_pre_main_code = _mock_sequential._collect_all_pre_main_code

        source0 = self._make_source("uint32_t counter = 0;")
        source1 = self._make_source("uint32_t counter = 0;")
        result, names = _collect_all_pre_main_code([(0, source0), (1, source1)])
        assert "uint32_t phase0_counter = 0;" in result
        assert "uint32_t phase1_counter = 0;" in result
        assert "counter" in names[0]
        assert "counter" in names[1]

    def test_namespace_blocks_deduped_by_signature(self):
        """Namespace blocks with same name are deduped (shared)."""
        _collect_all_pre_main_code = _mock_sequential._collect_all_pre_main_code

        ns_block = "namespace my_ns {\n" "using T = uint32_t;\n" "inline void f() { return; }\n" "}  // namespace my_ns"
        source0 = self._make_source(ns_block)
        source1 = self._make_source(ns_block)
        result, _ = _collect_all_pre_main_code([(0, source0), (1, source1)])
        # The opening "namespace my_ns {" should appear exactly once
        assert result.count("namespace my_ns {") == 1

    def test_different_namespaces_both_kept(self):
        """Different namespace blocks are both kept (shared, not prefixed)."""
        _collect_all_pre_main_code = _mock_sequential._collect_all_pre_main_code

        source0 = self._make_source("namespace ns_a {\ninline void fa() { return; }\n}")
        source1 = self._make_source("namespace ns_b {\ninline void fb() { return; }\n}")
        result, _ = _collect_all_pre_main_code([(0, source0), (1, source1)])
        assert "ns_a" in result
        assert "ns_b" in result

    def test_mixed_content(self):
        """Mix of aliases, functions, and globals from multiple phases."""
        _collect_all_pre_main_code = _mock_sequential._collect_all_pre_main_code

        source0 = self._make_source(
            "namespace g = norm;\n\n" "ALWI void ACQ() { acquire_dst(); }\n\n" "uint32_t state = 0;"
        )
        source1 = self._make_source(
            "namespace g = norm;\n\n"
            "ALWI void ACQ() { acquire_dst(); }\n\n"
            "uint32_t state = 0;\n\n"
            "ALWI void extra() { return; }"
        )
        result, names = _collect_all_pre_main_code([(0, source0), (1, source1)])
        # Alias: deduped (shared)
        assert result.count("namespace g = norm") == 1
        # Function ACQ: both phases prefixed
        assert "phase0_ACQ" in result
        assert "phase1_ACQ" in result
        # Global var: both phases prefixed
        assert "uint32_t phase0_state = 0;" in result
        assert "uint32_t phase1_state = 0;" in result
        # Extra function from phase 1: prefixed
        assert "phase1_extra" in result
        # Phase names include both functions and globals
        assert set(names[0]) == {"ACQ", "state"}
        assert set(names[1]) == {"ACQ", "state", "extra"}

    def test_phase_names_returned(self):
        """phase_names dict correctly tracks all prefixed names per phase."""
        _collect_all_pre_main_code = _mock_sequential._collect_all_pre_main_code

        source0 = self._make_source("ALWI void ACQ() { acquire_dst(); }\n\nuint32_t x = 1;")
        source1 = self._make_source("ALWI void REL() { release_dst(); }\n\nfloat y = 2.0;")
        _, names = _collect_all_pre_main_code([(0, source0), (1, source1)])
        assert set(names[0]) == {"ACQ", "x"}
        assert set(names[1]) == {"REL", "y"}

    def test_empty(self):
        """Empty input returns empty string and empty dict."""
        _collect_all_pre_main_code = _mock_sequential._collect_all_pre_main_code

        result, names = _collect_all_pre_main_code([])
        assert result == ""
        assert names == {}

    def test_block_comments_stripped(self):
        """Block comments in pre-main are stripped before processing."""
        _collect_all_pre_main_code = _mock_sequential._collect_all_pre_main_code

        source = self._make_source("/**\n * @brief Helper\n */\n" "ALWI void helper() { return; }")
        result, _ = _collect_all_pre_main_code([(0, source)])
        assert "@brief" not in result
        assert "phase0_helper" in result

    def test_pragma_stripped(self):
        """#pragma directives are stripped from pre-main."""
        _collect_all_pre_main_code = _mock_sequential._collect_all_pre_main_code

        source = self._make_source("#pragma once\n\nnamespace g = norm;")
        result, _ = _collect_all_pre_main_code([(0, source)])
        assert "#pragma" not in result
        assert "namespace g = norm" in result


class TestCrossOpPreMainStress:
    """Stress tests for pre-main merging with diverse real-world kernel patterns.

    Each test simulates fusing compute kernels from different ops by running
    realistic kernel source through the pre-main merging pipeline and verifying
    structural validity of the output.
    """

    # ---------------------------------------------------------------
    # Realistic kernel source templates (mimicking actual op kernels)
    # ---------------------------------------------------------------

    # layernorm compute: namespace aliases + short ALWI helpers + inlined numeric.h
    LAYERNORM_COMPUTE = """\
#include <cstdint>
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL
#include "compute_kernel_api.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/layernorm.h"
#include <tt-metalium/constants.hpp>

namespace generic = norm::kernel_util::generic;
namespace kutil = norm::kernel_util;
namespace numeric = kutil::compute::numeric;
namespace policies = kutil::compute::policies;

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

void kernel_main() {
    uint32_t NCHt = get_arg_val<uint32_t>(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    ACQ();
    REL();
}
"""

    # rmsnorm_post_allgather compute: same REDUCE defines, multi-line ACQ/REL
    RMSNORM_POST_COMPUTE = """\
#include <cstdint>
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/layernorm.h"

ALWI void ACQ() {
    tile_regs_acquire();
    tile_regs_wait();
}
ALWI void REL() {
    tile_regs_commit();
    tile_regs_release();
}

void kernel_main() {
    uint32_t NCHt = get_arg_val<uint32_t>(0);
    ACQ();
    REL();
}
"""

    # matmul compute: minimal pre-main, just a using declaration
    MATMUL_COMPUTE = """\
#include <cstdint>
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"

using std::uint32_t;

// matmul C=A*B using dims MK*KN = MN (row major order)
void kernel_main() {
    constexpr int onetile = 1;
    int dst_tile_index = 0;
}
"""

    # batchnorm compute: ALWI helper with many params, moreh_common include
    BATCHNORM_COMPUTE = """\
#include "compute_kernel_api/eltwise_binary.h"
#include <cstdint>

ALWI void batchnorm_bcast_tiles(
    uint32_t cb_bcast,
    uint32_t cb_other,
    uint32_t freq,
    uint32_t tile_start,
    uint32_t cb_batch_var,
    uint32_t cb_eps,
    uint32_t cb_den,
    uint32_t cb_weight,
    uint32_t cb_bias,
    uint32_t cb_tmp_1,
    uint32_t cb_output_0,
    uint32_t weight_has,
    uint32_t bias_has) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;
    uint32_t weight_has_value = weight_has;
    uint32_t bias_has_value = bias_has;
    auto cb_affine_or_out = (weight_has_value || bias_has_value) ? cb_tmp_1 : cb_output_0;
    cb_reserve_back(cb_den, onetile);
    cb_wait_front(cb_batch_var, onetile);
    tile_regs_acquire();
    tile_regs_commit();
    tile_regs_wait();
    tile_regs_release();
    cb_pop_front(cb_batch_var, onetile);
    cb_push_back(cb_den, onetile);
}

void kernel_main() {
    batchnorm_bcast_tiles(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 1);
}
"""

    # eltwise binary_ng compute: ALWI process_tile with macros + inlined utils
    ELTWISE_BINARY_COMPUTE = """\
#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"

#define IS_EMPTY(...) 0
#define PROCESS_ACTIVATIONS(op, i) /* noop */
#define HAS_ACTIVATIONS(op) 0
#define BCAST_OP LHS
#define OTHER_OP RHS
#define BCAST_OP_0 LHS
#define BCAST_OP_1 RHS

ALWI void process_tile(
    tt::CBIndex cb_pre_lhs,
    tt::CBIndex cb_post_lhs,
    tt::CBIndex cb_pre_rhs,
    tt::CBIndex cb_post_rhs,
    tt::CBIndex cb_out,
    uint32_t freq,
    uint32_t tile_start,
    uint32_t num_tiles_per_cycle) {
    using namespace ckernel;
    cb_wait_front(cb_post_lhs, num_tiles_per_cycle);
    cb_reserve_back(cb_out, num_tiles_per_cycle);
    tile_regs_acquire();
    tile_regs_commit();
    tile_regs_wait();
    tile_regs_release();
    cb_push_back(cb_out, num_tiles_per_cycle);
}

void kernel_main() {
    process_tile(0, 1, 2, 3, 4, 8, 0, 1);
}
"""

    # untilize compute: constexpr helper function
    UNTILIZE_COMPUTE = """\
#include <cstdint>

constexpr uint32_t compute_num_blocks_per_column(uint32_t per_core_block_tile_cnt, uint32_t max_bct) {
    for (uint32_t bct = max_bct; bct >= 1; --bct) {
        if (per_core_block_tile_cnt % bct == 0) {
            return per_core_block_tile_cnt / bct;
        }
    }
    return 1;
}

void kernel_main() {
    constexpr uint32_t n = compute_num_blocks_per_column(8, 4);
}
"""

    # groupnorm compute: no pre-main (everything inside kernel_main)
    GROUPNORM_COMPUTE = """\
#include <cstdint>
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_SCALAR
#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/bcast.h"

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
}
"""

    # eltwise unary sfpu: no pre-main, many includes
    ELTWISE_SFPU_COMPUTE = """\
#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/trigonometry.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
}
"""

    # numeric.h inlined (simplified): nested namespace with template functions
    NUMERIC_INLINED_COMPUTE = """\
#include <cstdint>
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/eltwise_binary.h"
#include <type_traits>
#include <array>

namespace policies = norm::kernel_util::compute::policies;
namespace generic = norm::kernel_util::generic;

namespace norm::kernel_util::compute::numeric {

namespace detail {

constexpr uint32_t dst0 = 0;

inline void scale_dest(uint32_t dst, uint32_t scalar) {
    binop_with_scalar_tile_init();
    mul_unary_tile(dst, scalar);
}

template <typename Block>
inline void reduce_block(uint32_t cb_in, uint32_t cb_scaler, uint32_t cb_out, const Block& block) {
    constexpr uint32_t onetile = 1;
    reduce_init_delta<false>(cb_in, cb_scaler);
    cb_wait_front(cb_scaler, onetile);
    cb_reserve_back(cb_out, onetile);
    tile_regs_acquire();
    tile_regs_commit();
    tile_regs_wait();
    tile_regs_release();
    cb_push_back(cb_out, onetile);
    reduce_revert_delta(cb_in);
}

}  // namespace detail
}  // namespace norm::kernel_util::compute::numeric

namespace generic = norm::kernel_util::generic;
namespace kutil = norm::kernel_util;
namespace numeric = kutil::compute::numeric;

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

void kernel_main() {
    ACQ();
    REL();
}
"""

    # Kernel with global/static variables in pre-main
    GLOBALS_COMPUTE = """\
#include <cstdint>

static uint32_t phase_counter = 0;
volatile uint32_t sync_flag = 0;
constexpr uint32_t MAX_TILES = 64;

ALWI void reset_state() {
    phase_counter = 0;
    sync_flag = 0;
}

void kernel_main() {
    reset_state();
    phase_counter++;
}
"""

    # Kernel with Doxygen block comments (should be stripped)
    DOXYGEN_COMPUTE = """\
#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"

/**
 * @brief Compute two-stage NOC addresses for distributed reduce.
 * @tparam row_major Whether cores are indexed row-major
 * @tparam N Number of remote workers
 * @param addrs Array to populate
 * @param x Starting X coordinate
 */
template <bool row_major, uint32_t N>
inline void compute_noc_addrs(uint32_t* addrs, uint32_t x, uint32_t y) {
    for (uint32_t i = 0; i < N; ++i) {
        addrs[i] = x + i;
    }
}

/**
 * @file Short helper
 */
ALWI void sync() { /* noop */ }

void kernel_main() {
    uint32_t addrs[4];
    compute_noc_addrs<true, 4>(addrs, 0, 0);
    sync();
}
"""

    # ---------------------------------------------------------------
    # Structural validation helpers
    # ---------------------------------------------------------------

    @staticmethod
    def _assert_balanced_braces(code: str, label: str = ""):
        """Verify all braces are balanced in the merged code."""
        depth = 0
        for i, ch in enumerate(code):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            assert depth >= 0, f"Unbalanced '}}' at position {i} in {label}"
        assert depth == 0, f"Unclosed '{{' (depth={depth}) in {label}"

    @staticmethod
    def _assert_no_orphaned_doxygen(code: str, label: str = ""):
        """Verify no orphaned Doxygen artifacts (from broken block comment stripping)."""
        for lineno, line in enumerate(code.split("\n"), 1):
            stripped = line.strip()
            # Orphaned Doxygen line starts with * but is not inside a block comment
            assert not (
                stripped.startswith("* @") or stripped == "*/"
            ), f"Orphaned Doxygen artifact at line {lineno} in {label}: {stripped!r}"

    @staticmethod
    def _assert_no_duplicate_functions(code: str, label: str = ""):
        """Check that no function signature appears more than once."""
        import re

        # Match function-like declarations: type name(...)
        sigs = re.findall(
            r"(?:ALWI|inline|FORCE_INLINE|constexpr)\s+\w+\s+(\w+)\s*\(",
            code,
        )
        seen = {}
        for sig in sigs:
            seen[sig] = seen.get(sig, 0) + 1
        dupes = {k: v for k, v in seen.items() if v > 1}
        assert not dupes, f"Duplicate function signatures in {label}: {dupes}"

    def _merge_sources(self, *sources):
        """Run sources through the pre-main merging pipeline."""
        _collect_all_pre_main_code = _mock_sequential._collect_all_pre_main_code

        indexed = list(enumerate(sources))
        result, _ = _collect_all_pre_main_code(indexed)
        return result

    def _validate(self, result: str, label: str):
        """Run all structural validations."""
        self._assert_balanced_braces(result, label)
        self._assert_no_orphaned_doxygen(result, label)
        self._assert_no_duplicate_functions(result, label)

    # ---------------------------------------------------------------
    # Test 1: layernorm + matmul (namespace aliases vs using decl)
    # ---------------------------------------------------------------
    def test_layernorm_plus_matmul(self):
        """Fuse layernorm (aliases + ALWI) with matmul (using decl, minimal)."""
        result = self._merge_sources(self.LAYERNORM_COMPUTE, self.MATMUL_COMPUTE)
        self._validate(result, "LN+matmul")
        # LN pre-main content (shared items)
        assert "namespace generic = norm::kernel_util::generic;" in result
        assert "namespace kutil = norm::kernel_util;" in result
        # LN functions get phase0_ prefix
        assert "phase0_ACQ" in result
        assert "phase0_REL" in result
        # Matmul pre-main content (shared item)
        assert "using std::uint32_t;" in result

    # ---------------------------------------------------------------
    # Test 2: layernorm + batchnorm (short helpers vs long helper)
    # ---------------------------------------------------------------
    def test_layernorm_plus_batchnorm(self):
        """Fuse layernorm (short ACQ/REL) with batchnorm (13-param ALWI)."""
        result = self._merge_sources(self.LAYERNORM_COMPUTE, self.BATCHNORM_COMPUTE)
        self._validate(result, "LN+batchnorm")
        assert "phase0_ACQ" in result
        assert "phase1_batchnorm_bcast_tiles" in result
        # Shared items
        assert "namespace generic" in result

    # ---------------------------------------------------------------
    # Test 3: layernorm + eltwise binary_ng (aliases vs macro-heavy)
    # ---------------------------------------------------------------
    def test_layernorm_plus_eltwise_binary(self):
        """Fuse layernorm (namespace aliases) with eltwise binary (macros + process_tile)."""
        result = self._merge_sources(self.LAYERNORM_COMPUTE, self.ELTWISE_BINARY_COMPUTE)
        self._validate(result, "LN+eltwise_binary")
        assert "phase0_ACQ" in result
        assert "phase1_process_tile" in result
        assert "namespace generic" in result

    # ---------------------------------------------------------------
    # Test 4: rmsnorm_post + groupnorm (same REDUCE defines, different ACQ/REL)
    # ---------------------------------------------------------------
    def test_rmsnorm_post_plus_groupnorm(self):
        """Fuse rmsnorm_post (multi-line ACQ/REL) with groupnorm (no pre-main)."""
        result = self._merge_sources(self.RMSNORM_POST_COMPUTE, self.GROUPNORM_COMPUTE)
        self._validate(result, "rmsnorm+groupnorm")
        # rmsnorm pre-main has multi-line ACQ/REL, prefixed with phase0_
        assert "phase0_ACQ" in result
        assert "tile_regs_acquire" in result
        assert "tile_regs_release" in result
        # groupnorm has no pre-main, so nothing new added

    # ---------------------------------------------------------------
    # Test 5: layernorm + untilize (aliases + helpers vs constexpr function)
    # ---------------------------------------------------------------
    def test_layernorm_plus_untilize(self):
        """Fuse layernorm (complex) with untilize (constexpr helper function)."""
        result = self._merge_sources(self.LAYERNORM_COMPUTE, self.UNTILIZE_COMPUTE)
        self._validate(result, "LN+untilize")
        assert "phase0_ACQ" in result
        assert "phase1_compute_num_blocks_per_column" in result

    # ---------------------------------------------------------------
    # Test 6: rmsnorm_post + layernorm (conflicting ACQ/REL signatures)
    # ---------------------------------------------------------------
    def test_rmsnorm_post_plus_layernorm(self):
        """Fuse rmsnorm_post (multi-line ACQ/REL) with layernorm (single-line ACQ/REL).

        Both define ACQ() and REL() with different bodies. Per-phase prefixing
        gives each its own version, avoiding silent drops or redefinition errors.
        """
        result = self._merge_sources(self.RMSNORM_POST_COMPUTE, self.LAYERNORM_COMPUTE)
        self._validate(result, "rmsnorm+LN")
        # rmsnorm's multi-line version (phase 0)
        assert "phase0_ACQ" in result
        assert "tile_regs_acquire" in result
        # layernorm's single-line version (phase 1) — now kept with its own prefix
        assert "phase1_ACQ" in result
        assert "acquire_dst" in result
        # Namespace aliases from LN should still be present (shared)
        assert "namespace generic" in result

    # ---------------------------------------------------------------
    # Test 7: 3-phase: layernorm + batchnorm + eltwise binary
    # ---------------------------------------------------------------
    def test_three_phase_ln_batchnorm_eltwise(self):
        """Three-phase fusion: LN + batchnorm + eltwise binary."""
        result = self._merge_sources(
            self.LAYERNORM_COMPUTE,
            self.BATCHNORM_COMPUTE,
            self.ELTWISE_BINARY_COMPUTE,
        )
        self._validate(result, "LN+batchnorm+eltwise")
        assert "phase0_ACQ" in result
        assert "phase1_batchnorm_bcast_tiles" in result
        assert "phase2_process_tile" in result
        assert "namespace generic" in result

    # ---------------------------------------------------------------
    # Test 8: 4-phase: matmul + layernorm + batchnorm + untilize
    # ---------------------------------------------------------------
    def test_four_phase_diverse_ops(self):
        """Four-phase fusion with maximally diverse ops."""
        result = self._merge_sources(
            self.MATMUL_COMPUTE,
            self.LAYERNORM_COMPUTE,
            self.BATCHNORM_COMPUTE,
            self.UNTILIZE_COMPUTE,
        )
        self._validate(result, "matmul+LN+batchnorm+untilize")
        # matmul (shared item)
        assert "using std::uint32_t;" in result
        # LN (shared + prefixed)
        assert "namespace generic" in result
        assert "phase1_ACQ" in result
        # batchnorm
        assert "phase2_batchnorm_bcast_tiles" in result
        # untilize
        assert "phase3_compute_num_blocks_per_column" in result

    # ---------------------------------------------------------------
    # Test 9: Inlined numeric.h namespace + Doxygen + globals
    # ---------------------------------------------------------------
    def test_complex_inlined_header_plus_doxygen_plus_globals(self):
        """Complex inlined namespace (numeric.h) + Doxygen comments + globals.

        Exercises: nested namespaces, block comment stripping, global
        variable prefixing for all phases, template function dedup.
        """
        result = self._merge_sources(
            self.NUMERIC_INLINED_COMPUTE,
            self.DOXYGEN_COMPUTE,
            self.GLOBALS_COMPUTE,
        )
        self._validate(result, "numeric+doxygen+globals")
        # numeric.h namespace block (shared)
        assert "norm::kernel_util::compute::numeric" in result
        assert "scale_dest" in result
        assert "reduce_block" in result
        # Doxygen comments should be stripped
        assert "@brief" not in result
        assert "@tparam" not in result
        assert "@param" not in result
        # Doxygen kernel's template function should be present (in namespace, shared)
        assert "compute_noc_addrs" in result
        # Globals: ALL phases prefixed
        assert "phase2_phase_counter" in result
        assert "phase2_sync_flag" in result

    # ---------------------------------------------------------------
    # Test 10: 5-phase stress: all different ops
    # ---------------------------------------------------------------
    def test_five_phase_maximum_diversity(self):
        """Five-phase fusion with maximum op diversity.

        Exercises every pre-main pattern simultaneously: namespace aliases,
        using declarations, ALWI short/long helpers, constexpr functions,
        template functions, nested namespaces, global variables, Doxygen.
        """
        result = self._merge_sources(
            self.LAYERNORM_COMPUTE,  # aliases + short ACQ/REL
            self.BATCHNORM_COMPUTE,  # long ALWI helper
            self.ELTWISE_BINARY_COMPUTE,  # ALWI process_tile
            self.UNTILIZE_COMPUTE,  # constexpr helper
            self.GLOBALS_COMPUTE,  # globals + ALWI reset_state
        )
        self._validate(result, "5-phase-stress")

        # Phase 0 (LN): namespace aliases (shared) and helpers (prefixed)
        assert "namespace generic = norm::kernel_util::generic;" in result
        assert "phase0_ACQ" in result
        assert "phase0_REL" in result

        # Phase 1 (batchnorm): long helper function (prefixed)
        assert "phase1_batchnorm_bcast_tiles" in result

        # Phase 2 (eltwise binary): process_tile (prefixed)
        assert "phase2_process_tile" in result

        # Phase 3 (untilize): constexpr function (prefixed)
        assert "phase3_compute_num_blocks_per_column" in result

        # Phase 4 (globals): prefixed global variables + helper
        assert "phase4_phase_counter" in result
        assert "phase4_sync_flag" in result
        assert "phase4_reset_state" in result


class TestReplaceInCodeOnly:
    """Tests for tree-sitter-based code-aware replacement."""

    def test_replaces_in_code(self):
        replace_in_code_only = _mock_cpp_parser.replace_in_code_only

        source = "int ACQ = 1;\nACQ();"
        result = replace_in_code_only(source, "ACQ", "phase0_ACQ")
        assert "phase0_ACQ = 1" in result
        assert "phase0_ACQ()" in result

    def test_skips_string_literals(self):
        replace_in_code_only = _mock_cpp_parser.replace_in_code_only

        source = 'const char* s = "ACQ is here";\nint ACQ = 1;'
        result = replace_in_code_only(source, "ACQ", "phase0_ACQ")
        assert '"ACQ is here"' in result  # String unchanged
        assert "phase0_ACQ = 1" in result  # Code changed

    def test_skips_comments(self):
        replace_in_code_only = _mock_cpp_parser.replace_in_code_only

        source = "// ACQ comment\nint ACQ = 1;"
        result = replace_in_code_only(source, "ACQ", "phase0_ACQ")
        assert "// ACQ comment" in result  # Comment unchanged
        assert "phase0_ACQ = 1" in result  # Code changed

    def test_skips_block_comments(self):
        replace_in_code_only = _mock_cpp_parser.replace_in_code_only

        source = "/* ACQ in block */\nint ACQ = 1;"
        result = replace_in_code_only(source, "ACQ", "phase0_ACQ")
        assert "/* ACQ in block */" in result  # Block comment unchanged
        assert "phase0_ACQ = 1" in result  # Code changed

    def test_word_boundary(self):
        replace_in_code_only = _mock_cpp_parser.replace_in_code_only

        source = "int ACQ = 1;\nint ACQUIRE = 2;"
        result = replace_in_code_only(source, "ACQ", "phase0_ACQ")
        assert "phase0_ACQ = 1" in result
        assert "ACQUIRE" in result  # Not replaced (different word)


class TestExtractKernelBody:
    """Tests for tree-sitter-based kernel body extraction."""

    def test_extract_body_with_alwi_kernel_main(self):
        extract_kernel_body = _mock_cpp_parser.extract_kernel_body

        source = """
ALWI void kernel_main() {
    int x = 1;
    compute(x);
}
"""
        body = extract_kernel_body(source)
        assert "int x = 1" in body
        assert "compute(x)" in body

    def test_extract_body_with_string_braces(self):
        """kernel_main body with string literal containing braces."""
        extract_kernel_body = _mock_cpp_parser.extract_kernel_body

        source = """
void kernel_main() {
    const char* msg = "{ json }";
    int y = 2;
}
"""
        body = extract_kernel_body(source)
        assert "{ json }" in body
        assert "int y = 2" in body

    def test_standard_kernel_main(self):
        extract_kernel_body = _mock_cpp_parser.extract_kernel_body

        source = "void kernel_main() {\n    int x = 1;\n}"
        body = extract_kernel_body(source)
        assert "int x = 1" in body

    def test_no_kernel_main(self):
        extract_kernel_body = _mock_cpp_parser.extract_kernel_body

        source = "void other_function() { int x = 1; }"
        body = extract_kernel_body(source)
        assert body == ""


class TestIfdefRobustness:
    """Tests for extended #ifdef resolution patterns."""

    def test_not_defined_c_style(self):
        """Test C++ standard !defined(X) syntax."""
        resolve_ifdef_directives = _mock_cpp_parser.resolve_ifdef_directives

        source = """
int a = 1;
#if !defined(RMSNORM)
int b = 2;
#endif
int c = 3;
"""
        result = resolve_ifdef_directives(source, set())
        assert "int b = 2" in result

        result2 = resolve_ifdef_directives(source, {"RMSNORM"})
        assert "int b = 2" not in result2

    def test_not_defined_no_parens(self):
        """Test !defined X without parentheses."""
        resolve_ifdef_directives = _mock_cpp_parser.resolve_ifdef_directives

        source = """
#if !defined FUSE_PRE_ADD
int no_pre_add = 1;
#endif
"""
        result = resolve_ifdef_directives(source, set())
        assert "int no_pre_add = 1" in result

        result2 = resolve_ifdef_directives(source, {"FUSE_PRE_ADD"})
        assert "int no_pre_add = 1" not in result2

    def test_if_0_if_1(self):
        """Test #if 0 and #if 1 unconditional blocks."""
        resolve_ifdef_directives = _mock_cpp_parser.resolve_ifdef_directives

        source = """
int a = 1;
#if 0
int dead_code = 999;
#endif
#if 1
int always_here = 42;
#endif
int b = 2;
"""
        result = resolve_ifdef_directives(source, set())
        assert "int dead_code = 999" not in result
        assert "int always_here = 42" in result
        assert "int a = 1" in result
        assert "int b = 2" in result

    def test_line_continuation(self):
        """Test backslash line continuation in #if directives."""
        resolve_ifdef_directives = _mock_cpp_parser.resolve_ifdef_directives

        source = """int a = 1;
#if defined FUSE_GAMMA \\
|| defined FUSE_BETA
int gamma_or_beta = 1;
#endif
int b = 2;"""
        result = resolve_ifdef_directives(source, {"FUSE_GAMMA"})
        assert "int gamma_or_beta = 1" in result

    def test_mixed_defined_and_not_defined(self):
        """Test #if defined(A) && !defined(B)."""
        resolve_ifdef_directives = _mock_cpp_parser.resolve_ifdef_directives

        source = """
#if defined(RMSNORM) && !defined(FUSE_PRE_ADD)
int rms_no_preadd = 1;
#endif
"""
        # RMSNORM defined, FUSE_PRE_ADD not defined -> true
        result = resolve_ifdef_directives(source, {"RMSNORM"})
        assert "int rms_no_preadd = 1" in result

        # Both defined -> false (because !defined(FUSE_PRE_ADD) is false)
        result2 = resolve_ifdef_directives(source, {"RMSNORM", "FUSE_PRE_ADD"})
        assert "int rms_no_preadd = 1" not in result2


class TestCategorizePreMainVariables:
    """Tests for variable detection in categorize_pre_main."""

    def test_pointer_types(self):
        categorize_pre_main = _mock_cpp_parser.categorize_pre_main

        source = "uint32_t* ptr = nullptr;\nvoid kernel_main() {}"
        blocks = categorize_pre_main(source)
        var_blocks = [b for b in blocks if b.kind == "variable"]
        assert len(var_blocks) == 1
        assert var_blocks[0].name == "ptr"

    def test_constexpr_types(self):
        categorize_pre_main = _mock_cpp_parser.categorize_pre_main

        source = "constexpr uint32_t MAX_TILES = 64;\nvoid kernel_main() {}"
        blocks = categorize_pre_main(source)
        var_blocks = [b for b in blocks if b.kind == "variable"]
        assert len(var_blocks) == 1
        assert var_blocks[0].name == "MAX_TILES"

    def test_static_variables(self):
        categorize_pre_main = _mock_cpp_parser.categorize_pre_main

        source = "static uint32_t counter = 0;\nvoid kernel_main() {}"
        blocks = categorize_pre_main(source)
        var_blocks = [b for b in blocks if b.kind == "variable"]
        assert len(var_blocks) == 1
        assert var_blocks[0].name == "counter"

    def test_functions_not_variables(self):
        categorize_pre_main = _mock_cpp_parser.categorize_pre_main

        source = "ALWI void ACQ() { acquire_dst(); }\nvoid kernel_main() {}"
        blocks = categorize_pre_main(source)
        var_blocks = [b for b in blocks if b.kind == "variable"]
        assert len(var_blocks) == 0


class TestRuntimeArgsPattern:
    """Tests for the runtime args variable offset pattern."""

    def test_matches_arg_variables(self):
        _offset_runtime_args_in_source = _mock_sequential._offset_runtime_args_in_source

        source = "uint32_t rt_args_idx = 0;\nint x = get_arg_val<uint32_t>(0);"
        result = _offset_runtime_args_in_source(source, 1)
        assert "__phase1_rt_offset" in result
        # rt_args_idx should be offset
        assert "rt_args_idx = __phase1_rt_offset" in result

    def test_does_not_match_unrelated_variables(self):
        _offset_runtime_args_in_source = _mock_sequential._offset_runtime_args_in_source

        source = "uint32_t counter = 0;\nuint32_t flags = 0;"
        result = _offset_runtime_args_in_source(source, 1)
        # counter and flags should NOT be offset (no "arg" or "rt" in name)
        assert "counter = 0" in result
        assert "flags = 0" in result


class TestPrefixPhaseNamesSkipsStrings:
    """Tests for _prefix_phase_names_in_source respecting strings/comments."""

    def test_does_not_replace_in_string(self):
        _prefix_phase_names_in_source = _mock_sequential._prefix_phase_names_in_source

        source = 'const char* msg = "ACQ called";\nACQ();'
        result = _prefix_phase_names_in_source(source, 0, ["ACQ"])
        # ACQ in string should stay unchanged
        assert '"ACQ called"' in result
        # ACQ in code should be prefixed
        assert "phase0_ACQ();" in result

    def test_does_not_replace_in_comment(self):
        _prefix_phase_names_in_source = _mock_sequential._prefix_phase_names_in_source

        source = "// Call ACQ to acquire\nACQ();"
        result = _prefix_phase_names_in_source(source, 0, ["ACQ"])
        assert "// Call ACQ to acquire" in result
        assert "phase0_ACQ();" in result

    def test_does_not_replace_in_block_comment(self):
        _prefix_phase_names_in_source = _mock_sequential._prefix_phase_names_in_source

        source = "/* ACQ description */\nACQ();"
        result = _prefix_phase_names_in_source(source, 0, ["ACQ"])
        assert "/* ACQ description */" in result
        assert "phase0_ACQ();" in result


class TestFunctionDetectionRobust:
    """Tests for function detection via categorize_pre_main."""

    def test_function_with_braces_in_string(self):
        """Function with braces in string is still detected as function."""
        categorize_pre_main = _mock_cpp_parser.categorize_pre_main

        source = 'void foo() {\n    const char* s = "no { braces }";\n}\nvoid kernel_main() {}'
        blocks = categorize_pre_main(source)
        func_blocks = [b for b in blocks if b.kind == "function"]
        assert len(func_blocks) == 1
        assert func_blocks[0].name == "foo"

    def test_scope_qualified_function(self):
        """Scope-qualified function names are extracted correctly."""
        categorize_pre_main = _mock_cpp_parser.categorize_pre_main

        source = "void MyClass::process() {\n    return;\n}\nvoid kernel_main() {}"
        blocks = categorize_pre_main(source)
        func_blocks = [b for b in blocks if b.kind == "function"]
        assert len(func_blocks) == 1
        assert func_blocks[0].name == "process"


class TestInlineLocalIncludesRelative:
    """Tests for relative path include inlining."""

    def test_relative_path_inlined(self, tmp_path):
        """Includes with relative paths are resolved and inlined."""
        inline_local_includes = _mock_cpp_parser.inline_local_includes

        # Create directory structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        header = subdir / "helper.h"
        header.write_text("#pragma once\nint helper_val = 42;")

        source = '#include "subdir/helper.h"\nvoid kernel_main() {}'
        result = inline_local_includes(source, str(tmp_path))
        assert "int helper_val = 42" in result
        assert "#pragma once" not in result

    def test_local_include_still_works(self, tmp_path):
        """Local includes (no path separator) still work."""
        inline_local_includes = _mock_cpp_parser.inline_local_includes

        header = tmp_path / "local.h"
        header.write_text("int local_val = 1;")

        source = '#include "local.h"\nvoid kernel_main() {}'
        result = inline_local_includes(source, str(tmp_path))
        assert "int local_val = 1" in result


class TestCategorizePreMainStopsAtKernelMain:
    """Test that categorize_pre_main stops at kernel_main with attributes."""

    def test_stops_at_alwi_kernel_main(self):
        categorize_pre_main = _mock_cpp_parser.categorize_pre_main

        source = """namespace g = norm;

ALWI void kernel_main() {
    int x = 1;
}
"""
        blocks = categorize_pre_main(source)
        all_text = " ".join(b.text for b in blocks)
        assert "namespace g = norm" in all_text or any("norm" in b.text for b in blocks)
        assert "int x = 1" not in all_text


class TestCBPoolAllocator:
    """Tests for CBPoolAllocator pool-based CB slot allocation."""

    def test_basic_allocation_same_config_reuses_slot(self):
        """Two phases with identical CB configs should reuse the same slot."""
        CBPoolAllocator = _mock_sequential.CBPoolAllocator
        CBInfo = _mock_sequential.CBInfo

        pool = CBPoolAllocator(max_slots=32)

        # Phase 0: CB index 0, format "F16", page_size=1024
        cb_info_0 = {
            0: CBInfo(0, 2048, "F16", 1024, None, "Default"),
        }
        pool.allocate_phase(0, cb_info_0, set())

        # Phase 1: CB index 0, same config
        cb_info_1 = {
            0: CBInfo(0, 4096, "F16", 1024, None, "Default"),
        }
        pool.allocate_phase(1, cb_info_1, set())

        # Both phases should map CB 0 to the same slot
        assert pool.get_remap(0)[0] == pool.get_remap(1)[0]
        # Total size should be max(2048, 4096) = 4096
        slot_idx = pool.get_remap(0)[0]
        assert pool._slots[slot_idx].total_size == 4096

    def test_different_page_size_gets_separate_slot(self):
        """Different page sizes on the same CB index should get separate slots."""
        CBPoolAllocator = _mock_sequential.CBPoolAllocator
        CBInfo = _mock_sequential.CBInfo

        pool = CBPoolAllocator(max_slots=32)

        # Phase 0: CB 0, page_size=1024
        cb_info_0 = {
            0: CBInfo(0, 2048, "F16", 1024, None, "Default"),
        }
        pool.allocate_phase(0, cb_info_0, set())

        # Phase 1: CB 0, page_size=2048 (different)
        cb_info_1 = {
            0: CBInfo(0, 4096, "F16", 2048, None, "Default"),
        }
        pool.allocate_phase(1, cb_info_1, set())

        # Should get different slots
        slot_0 = pool.get_remap(0)[0]
        slot_1 = pool.get_remap(1)[0]
        assert slot_0 != slot_1

    def test_different_data_format_gets_separate_slot(self):
        """Different data formats on the same CB index should get separate slots."""
        CBPoolAllocator = _mock_sequential.CBPoolAllocator
        CBInfo = _mock_sequential.CBInfo

        pool = CBPoolAllocator(max_slots=32)

        cb_info_0 = {0: CBInfo(0, 2048, "F16", 1024, None, "Default")}
        pool.allocate_phase(0, cb_info_0, set())

        cb_info_1 = {0: CBInfo(0, 4096, "F32", 1024, None, "Default")}
        pool.allocate_phase(1, cb_info_1, set())

        assert pool.get_remap(0)[0] != pool.get_remap(1)[0]

    def test_different_unpack_to_dest_mode_gets_separate_slot(self):
        """Different unpack_to_dest_mode on same (format, page_size) gets separate slots."""
        CBPoolAllocator = _mock_sequential.CBPoolAllocator
        CBInfo = _mock_sequential.CBInfo

        pool = CBPoolAllocator(max_slots=32)

        cb_info_0 = {0: CBInfo(0, 2048, "F16", 1024, None, "Default")}
        pool.allocate_phase(0, cb_info_0, set())

        cb_info_1 = {0: CBInfo(0, 2048, "F16", 1024, None, "UnpackToDestFp32")}
        pool.allocate_phase(1, cb_info_1, set())

        assert pool.get_remap(0)[0] != pool.get_remap(1)[0]

    def test_overflow_raises_error(self):
        """Exceeding max_slots raises ValueError."""
        CBPoolAllocator = _mock_sequential.CBPoolAllocator
        CBInfo = _mock_sequential.CBInfo

        pool = CBPoolAllocator(max_slots=3)

        # Allocate 3 different configs
        for i in range(3):
            cb_info = {0: CBInfo(0, 1024, f"F{i}", 1024, None, "Default")}
            pool.allocate_phase(i, cb_info, set())

        # 4th different config should overflow
        with pytest.raises(ValueError, match="CB pool overflow"):
            cb_info = {0: CBInfo(0, 1024, "F99", 1024, None, "Default")}
            pool.allocate_phase(3, cb_info, set())

    def test_phantom_cb_reservation(self):
        """Phantom CBs get identity-mapped and don't collide with real allocations."""
        CBPoolAllocator = _mock_sequential.CBPoolAllocator
        CBInfo = _mock_sequential.CBInfo

        pool = CBPoolAllocator(max_slots=32)

        # Phase 0: real CB at index 0, phantom at index 18
        cb_info_0 = {0: CBInfo(0, 2048, "F16", 1024, None, "Default")}
        pool.allocate_phase(0, cb_info_0, phantom_cb_indices={18})

        # Phase 1: real CB at index 0 (same config), real CB at index 5
        cb_info_1 = {
            0: CBInfo(0, 2048, "F16", 1024, None, "Default"),
            5: CBInfo(5, 2048, "F32", 2048, None, "Default"),
        }
        pool.allocate_phase(1, cb_info_1, phantom_cb_indices=set())

        # Phantom index 18 should be identity-mapped
        assert pool.get_remap(0)[18] == 18
        # Real CB 5 should NOT be allocated at index 18 (reserved by phantom)
        slot_5 = pool.get_remap(1)[5]
        assert slot_5 != 18

    def test_multiple_cbs_per_phase(self):
        """Within a phase, each CB gets its own slot even with same config."""
        CBPoolAllocator = _mock_sequential.CBPoolAllocator
        CBInfo = _mock_sequential.CBInfo

        pool = CBPoolAllocator(max_slots=32)

        # Phase 0 has 3 CBs (CB 0 and CB 16 have same config)
        cb_info_0 = {
            0: CBInfo(0, 2048, "F16", 1024, None, "Default"),
            5: CBInfo(5, 4096, "F32", 2048, None, "Default"),
            16: CBInfo(16, 1024, "F16", 1024, None, "Default"),
        }
        pool.allocate_phase(0, cb_info_0, set())

        # CB 0 and CB 16 have the same config but are in the SAME phase,
        # so they must get separate slots (they hold different data concurrently)
        assert pool.get_remap(0)[0] != pool.get_remap(0)[16]
        # CB 5 has different config — separate slot
        assert pool.get_remap(0)[5] != pool.get_remap(0)[0]
        # All 3 CBs get unique slots
        slots = {pool.get_remap(0)[i] for i in [0, 5, 16]}
        assert len(slots) == 3

    def test_cross_phase_sharing(self):
        """CBs with same config across different phases should share a slot."""
        CBPoolAllocator = _mock_sequential.CBPoolAllocator
        CBInfo = _mock_sequential.CBInfo

        pool = CBPoolAllocator(max_slots=32)

        # Phase 0: CB 0
        cb_info_0 = {0: CBInfo(0, 2048, "F16", 1024, None, "Default")}
        pool.allocate_phase(0, cb_info_0, set())

        # Phase 1: CB 0 with same config
        cb_info_1 = {0: CBInfo(0, 2048, "F16", 1024, None, "Default")}
        pool.allocate_phase(1, cb_info_1, set())

        # Phase 1's CB 0 should reuse phase 0's slot
        assert pool.get_remap(0)[0] == pool.get_remap(1)[0]

    def test_cross_phase_sharing_different_index(self):
        """CBs at different indices but same config across phases share a slot."""
        CBPoolAllocator = _mock_sequential.CBPoolAllocator
        CBInfo = _mock_sequential.CBInfo

        pool = CBPoolAllocator(max_slots=32)

        # Phase 0: CB 5 with config_a
        cb_info_0 = {5: CBInfo(5, 2048, "F16", 1024, None, "Default")}
        pool.allocate_phase(0, cb_info_0, set())

        # Phase 1: CB 0 with same config_a (different original index)
        cb_info_1 = {0: CBInfo(0, 2048, "F16", 1024, None, "Default")}
        pool.allocate_phase(1, cb_info_1, set())

        # Phase 1's CB 0 should reuse phase 0's CB 5 slot
        assert pool.get_remap(0)[5] == pool.get_remap(1)[0]

    def test_get_all_slot_indices(self):
        """get_all_slot_indices returns all allocated slots."""
        CBPoolAllocator = _mock_sequential.CBPoolAllocator
        CBInfo = _mock_sequential.CBInfo

        pool = CBPoolAllocator(max_slots=32)

        cb_info_0 = {
            0: CBInfo(0, 2048, "F16", 1024, None, "Default"),
            5: CBInfo(5, 4096, "F32", 2048, None, "Default"),
        }
        pool.allocate_phase(0, cb_info_0, set())

        indices = pool.get_all_slot_indices()
        assert len(indices) == 2
        assert all(idx >= 0 for idx in indices)

    def test_named_arg_cb_remapping(self):
        """CB-reference named args (cb_*) should get remapped values."""
        merge_fn = _mock_sequential._merge_named_compile_time_args

        # Create mock kernel with named compile-time args
        kernel = MagicMock()
        kernel.named_compile_time_args = [
            ("cb_in0", 0),
            ("cb_out", 16),
            ("blk", 4),  # Not a CB arg, should NOT be remapped
        ]

        phase_kernels = [{"reader": kernel}]
        # Remap: CB 0 -> slot 3, CB 16 -> slot 7
        phase_remaps = [{0: 3, 16: 7}]

        result = merge_fn(
            phase_kernels,
            "reader",
            phase_remaps=phase_remaps,
        )

        result_dict = dict(result)
        assert result_dict["cb_in0"] == 3  # Remapped from 0 to 3
        assert result_dict["cb_out"] == 7  # Remapped from 16 to 7
        assert result_dict["blk"] == 4  # Not remapped (not a cb_ arg)

    def test_named_arg_cb_remapping_phase1_prefix(self):
        """Phase 1+ named args should be prefixed AND remapped."""
        merge_fn = _mock_sequential._merge_named_compile_time_args

        kernel0 = MagicMock()
        kernel0.named_compile_time_args = [("cb_in0", 0)]
        kernel1 = MagicMock()
        kernel1.named_compile_time_args = [("cb_in0", 0)]

        phase_kernels = [{"reader": kernel0}, {"reader": kernel1}]
        # Phase 0: CB 0 -> slot 0, Phase 1: CB 0 -> slot 5
        phase_remaps = [{0: 0}, {0: 5}]

        result = merge_fn(
            phase_kernels,
            "reader",
            phase_remaps=phase_remaps,
        )

        result_dict = dict(result)
        assert result_dict["cb_in0"] == 0  # Phase 0: identity
        assert result_dict["phase1_cb_in0"] == 5  # Phase 1: remapped + prefixed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
