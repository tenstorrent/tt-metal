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


# Mock ttnn before importing the fusion subpackage modules (which do
# `import ttnn` at module level).  We snapshot sys.modules BEFORE the mock,
# import the fusion modules under the mock, keep private references to them,
# then FULLY UNDO everything in sys.modules so that test files collected later
# (e.g. test_sequential.py) re-import from scratch with the real ttnn.
#
# The standalone test functions below access the mock-backed modules via the
# private module-level references (_mock_common / _mock_cb_allocator /
# _mock_codegen / _mock_fusion / _mock_graph) instead of doing a bare
# `from models.experimental.ops... import X`.

_modules_before = set(sys.modules)
_ttnn_mocked_keys = ["ttnn", "ttnn._ttnn", "ttnn._ttnn.program_descriptor"]
_ttnn_originals = {k: sys.modules.get(k) for k in _ttnn_mocked_keys}

for _k in _ttnn_mocked_keys:
    sys.modules[_k] = MagicMock()

import models.experimental.ops.descriptors.fusion.common as _mock_common  # noqa: E402
import models.experimental.ops.descriptors.fusion.cb_allocator as _mock_cb_allocator  # noqa: E402
import models.experimental.ops.descriptors.fusion.codegen as _mock_codegen  # noqa: E402
import models.experimental.ops.descriptors.fusion.fusion as _mock_fusion  # noqa: E402
import models.experimental.ops.descriptors.fusion.graph as _mock_graph  # noqa: E402

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
        CBInfo = _mock_cb_allocator.CBInfo

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
        PhaseInfo = _mock_cb_allocator.PhaseInfo
        CBInfo = _mock_cb_allocator.CBInfo

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
        PhaseInfo = _mock_cb_allocator.PhaseInfo

        phase = PhaseInfo(phase_idx=1, op_descriptor=MagicMock())
        assert phase.cb_info == {}


class TestExtractCBInfo:
    """Tests for extract_cb_info function."""

    def test_extract_from_descriptor(self):
        """Test extracting CB info from a mock descriptor."""
        extract_cb_info = _mock_cb_allocator.extract_cb_info

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
        extract_cb_info = _mock_cb_allocator.extract_cb_info

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
        extract_cb_names_from_kernel = _mock_cb_allocator.extract_cb_names_from_kernel

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


class TestOpGraphBuilderBasic:
    """Basic tests for OpGraphBuilder."""

    def _make_mock_op(self):
        """Create a mock OpDescriptor with realistic kernel core ranges."""
        mock_desc = MagicMock()
        mock_desc.descriptor = MagicMock(cbs=[])
        kernel = MagicMock()
        kernel.core_ranges = _make_mock_core_ranges()
        mock_desc.descriptor.kernels = [kernel]
        return mock_desc

    def test_creation(self):
        """Test creating a builder."""
        OpGraphBuilder = _mock_graph.OpGraphBuilder
        OpNode = _mock_graph.OpNode

        mock_desc = self._make_mock_op()
        builder = OpGraphBuilder(OpNode(mock_desc))
        assert builder._built is False

    def test_single_node_returns_fused_op(self):
        """Test that single-node build returns FusedOp wrapping the original."""
        OpGraphBuilder = _mock_graph.OpGraphBuilder
        OpNode = _mock_graph.OpNode
        FusedOp = _mock_fusion.FusedOp

        mock_desc = self._make_mock_op()
        result = OpGraphBuilder(OpNode(mock_desc)).build(device=MagicMock())

        assert type(result).__name__ == "FusedOp"
        assert result.descriptor is mock_desc.descriptor
        assert result.input_tensors is mock_desc.input_tensors
        assert result.output_tensors is mock_desc.output_tensors

    def test_build_twice_raises(self):
        """Test that building twice raises ValueError."""
        OpGraphBuilder = _mock_graph.OpGraphBuilder
        OpNode = _mock_graph.OpNode

        mock_desc = self._make_mock_op()
        builder = OpGraphBuilder(OpNode(mock_desc))
        builder.build(device=MagicMock())

        with pytest.raises(ValueError, match="Already built"):
            builder.build(device=MagicMock())


class TestSourceTransformations:
    """Tests for kernel source transformation functions."""

    def test_prefix_named_args_phase0_unchanged(self):
        """Test that phase 0 source is unchanged."""
        _prefix_named_args_in_source = _mock_codegen._prefix_named_args_in_source

        source = 'constexpr uint32_t cb = get_named_compile_time_arg_val("cb_in");'
        result = _prefix_named_args_in_source(source, 0)
        assert result == source

    def test_prefix_named_args_phase1(self):
        """Test that phase 1 gets prefixed named args."""
        _prefix_named_args_in_source = _mock_codegen._prefix_named_args_in_source

        source = 'constexpr uint32_t cb = get_named_compile_time_arg_val("cb_in");'
        result = _prefix_named_args_in_source(source, 1)
        assert 'get_named_compile_time_arg_val("phase_1_cb_in")' in result

    def test_prefix_named_args_phase2(self):
        """Test phase 2 prefix."""
        _prefix_named_args_in_source = _mock_codegen._prefix_named_args_in_source

        source = 'constexpr uint32_t blk = get_named_compile_time_arg_val("blk");'
        result = _prefix_named_args_in_source(source, 2)
        assert 'get_named_compile_time_arg_val("phase_2_blk")' in result

    def test_emit_rt_arg_wrapper_bakes_offset(self):
        """Test that wrapper function bakes the literal offset value."""
        _emit_rt_arg_wrapper = _mock_codegen._emit_rt_arg_wrapper
        result = _emit_rt_arg_wrapper(1, 12)
        joined = "\n".join(result)
        assert "arg_idx + 12" in joined
        assert "phase_1_get_arg_val" in joined
        # Wrapper should NOT contain a #define (that's separate)
        assert "#define" not in joined
        assert "get_named_compile_time_arg_val" not in joined

    def test_emit_rt_arg_define_and_undef(self):
        """Test that define/undef emit the correct preprocessor directives."""
        _emit_rt_arg_define = _mock_codegen._emit_rt_arg_define
        _emit_rt_arg_undef = _mock_codegen._emit_rt_arg_undef
        assert _emit_rt_arg_define(1) == "#define get_arg_val phase_1_get_arg_val"
        assert _emit_rt_arg_define(2) == "#define get_arg_val phase_2_get_arg_val"
        assert _emit_rt_arg_undef() == "#undef get_arg_val"

    def test_offset_compile_time_args_phase0_unchanged(self):
        """Test that phase 0 positional args are unchanged."""
        _offset_compile_time_args_in_source = _mock_codegen._offset_compile_time_args_in_source

        source = "uint32_t blk = get_compile_time_arg_val(0);"
        result = _offset_compile_time_args_in_source(source, 0, 0)
        assert result == source

    def test_offset_compile_time_args_phase1(self):
        """Test that phase 1 positional args are offset."""
        _offset_compile_time_args_in_source = _mock_codegen._offset_compile_time_args_in_source

        source = "uint32_t blk = get_compile_time_arg_val(0);\nuint32_t mode = get_compile_time_arg_val(1);"
        result = _offset_compile_time_args_in_source(source, 1, 3)
        assert "get_compile_time_arg_val(3)" in result
        assert "get_compile_time_arg_val(4)" in result

    def test_offset_compile_time_args_tensor_accessor(self):
        """Test that TensorAccessorArgs<N> is also offset."""
        _offset_compile_time_args_in_source = _mock_codegen._offset_compile_time_args_in_source

        source = "constexpr auto src_args = TensorAccessorArgs<2>();"
        result = _offset_compile_time_args_in_source(source, 1, 5)
        assert "TensorAccessorArgs<7>" in result

    def test_transform_phase_source_combines_all(self):
        """Test that _transform_phase_source applies all transforms.

        Note: runtime arg offsetting is now handled by #define/#undef redirect
        (see _emit_rt_arg_define/_emit_rt_arg_undef), NOT by source-level rewriting in
        _transform_phase_source. So get_arg_val calls are left unchanged.
        """
        _transform_phase_source = _mock_codegen._transform_phase_source

        source = (
            'constexpr uint32_t cb = get_named_compile_time_arg_val("cb_in");\n'
            "uint32_t blk = get_compile_time_arg_val(0);\n"
            "constexpr auto args = TensorAccessorArgs<2>();\n"
            "uint32_t val = get_arg_val<uint32_t>(3);\n"
        )
        result = _transform_phase_source(source, 1, ct_arg_offset=5)
        assert "phase_1_cb_in" in result
        assert "get_compile_time_arg_val(5)" in result  # offset from 0
        assert "TensorAccessorArgs<7>" in result  # offset from 2
        # get_arg_val is NOT rewritten in source — redirect is done via #define
        assert "get_arg_val<uint32_t>(3)" in result


class TestKernelBodyExtraction:
    """Tests for kernel body extraction."""

    def test_extract_simple_body(self):
        """Test extracting a simple kernel body."""
        extract_kernel_body = _mock_codegen.extract_kernel_body

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
        extract_kernel_body = _mock_codegen.extract_kernel_body

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
        extract_kernel_body = _mock_codegen.extract_kernel_body

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
        collect_includes = _mock_codegen.collect_includes

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
        collect_defines = _mock_codegen.collect_defines

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
        inline_local_includes = _mock_codegen.inline_local_includes

        # Create a local header file
        header = tmp_path / "utils.h"
        header.write_text("#pragma once\nint helper() { return 42; }\n")

        source = '#include "utils.h"\nvoid kernel_main() {}\n'
        headers, remaining = inline_local_includes(source, str(tmp_path))

        assert len(headers) == 1
        assert "int helper()" in headers[0][1]
        assert '#include "utils.h"' not in remaining
        # pragma once should be stripped
        assert "#pragma once" not in headers[0][1]

    def test_leaves_path_includes(self):
        """Test that includes with paths are left unchanged."""
        inline_local_includes = _mock_codegen.inline_local_includes

        source = '#include "api/dataflow/dataflow_api.h"\nvoid kernel_main() {}\n'
        headers, remaining = inline_local_includes(source, "/some/dir")

        # Path includes should remain (no local file found)
        assert headers == []
        assert '#include "api/dataflow/dataflow_api.h"' in remaining

    def test_no_kernel_dir_returns_unchanged(self):
        """Test that None kernel_dir returns source unchanged."""
        inline_local_includes = _mock_codegen.inline_local_includes

        source = '#include "utils.h"\nvoid kernel_main() {}\n'
        headers, remaining = inline_local_includes(source, None)
        assert headers == []
        assert remaining == source


class TestMergeNamedCompileTimeArgs:
    """Tests for named compile-time arg merging."""

    def test_phase0_keeps_original_names(self):
        """Test that phase 0 keeps original arg names."""
        _merge_named_compile_time_args = _mock_codegen._merge_named_compile_time_args

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
        _merge_named_compile_time_args = _mock_codegen._merge_named_compile_time_args

        phase_kernels = [
            {"reader": MagicMock(named_compile_time_args=[("cb_in", 0)])},
            {"reader": MagicMock(named_compile_time_args=[("cb_in", 0)])},
        ]

        result = _merge_named_compile_time_args(phase_kernels, "reader")
        names = dict(result)
        assert "cb_in" in names  # Phase 0
        assert "phase_1_cb_in" in names  # Phase 1

    def test_rt_arg_offsets_not_in_named_args(self):
        """RT arg offsets are baked into source, not passed as named compile-time args."""
        _merge_named_compile_time_args = _mock_codegen._merge_named_compile_time_args

        phase_kernels = [
            {"reader": MagicMock(named_compile_time_args=[("cb_in", 0)])},
            {"reader": MagicMock(named_compile_time_args=[("cb_in", 0)])},
        ]

        result = _merge_named_compile_time_args(phase_kernels, "reader")
        names = dict(result)
        assert "phase1_rt_arg_offset" not in names

    def test_barrier_rt_offset_added(self):
        """Test that barrier_rt_offset named arg is added."""
        _merge_named_compile_time_args = _mock_codegen._merge_named_compile_time_args

        phase_kernels = [
            {"reader": MagicMock(named_compile_time_args=[])},
        ]

        result = _merge_named_compile_time_args(
            phase_kernels,
            "reader",
            barrier_rt_offset=10,
        )
        names = dict(result)
        assert "barrier_rt_offset" in names
        assert names["barrier_rt_offset"] == 10


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
        _compute_and_concatenate_runtime_args = _mock_codegen._compute_and_concatenate_runtime_args

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

        offsets, _ = _compute_and_concatenate_runtime_args(phase_kernels, "reader")
        assert offsets[0] == 0
        assert offsets[1] == 5  # Phase 0 had 5 args

    def test_offsets_with_missing_kernel(self):
        """Test offsets when a phase has no kernel of that type."""
        _compute_and_concatenate_runtime_args = _mock_codegen._compute_and_concatenate_runtime_args

        core_ranges = _make_mock_core_ranges()

        kernel0 = MagicMock()
        kernel0.runtime_args = _make_mock_runtime_args_view([[1, 2, 3]])
        kernel0.core_ranges = core_ranges

        phase_kernels = [
            {"reader": kernel0},
            {"reader": None},
        ]

        offsets, _ = _compute_and_concatenate_runtime_args(phase_kernels, "reader")
        assert offsets[0] == 0
        assert offsets[1] == 3


class TestConcatenateRuntimeArgs:
    """Tests for runtime arg concatenation."""

    def test_basic_concatenation(self):
        """Test basic runtime arg concatenation across phases."""
        _compute_and_concatenate_runtime_args = _mock_codegen._compute_and_concatenate_runtime_args

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

        _, result = _compute_and_concatenate_runtime_args(phase_kernels, "reader")
        assert len(result) == 1  # One core
        core_coord, args = result[0]
        assert args == [1, 2, 3, 10, 20]


class TestMergeCompileTimeArgs:
    """Tests for compile-time arg concatenation."""

    def test_concatenates_all_phases(self):
        """Test that compile-time args from all phases are concatenated."""
        _merge_compile_time_args = _mock_codegen._merge_compile_time_args

        phase_kernels = [
            {"reader": MagicMock(compile_time_args=[10, 20, 30])},
            {"reader": MagicMock(compile_time_args=[40, 50])},
        ]

        merged, offsets = _merge_compile_time_args(phase_kernels, "reader")
        assert merged == [10, 20, 30, 40, 50]
        assert offsets == {0: 0, 1: 3}

    def test_handles_missing_kernel(self):
        """Test offset computation with a missing kernel."""
        _merge_compile_time_args = _mock_codegen._merge_compile_time_args

        phase_kernels = [
            {"reader": MagicMock(compile_time_args=[10, 20])},
            {"reader": None},
            {"reader": MagicMock(compile_time_args=[30])},
        ]

        merged, offsets = _merge_compile_time_args(phase_kernels, "reader")
        assert merged == [10, 20, 30]
        assert offsets == {0: 0, 1: 2, 2: 2}


class TestValidateFp32Consistency:
    """Tests for fp32_dest_acc_en validation."""

    def test_consistent_fp32_passes(self):
        """Test that consistent fp32 settings pass validation."""
        _validate_fp32_consistency = _mock_codegen._validate_fp32_consistency

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
        _validate_fp32_consistency = _mock_codegen._validate_fp32_consistency

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


class TestExtractKernelBody:
    """Tests for kernel body extraction via brace matching."""

    def test_extract_body_with_alwi_kernel_main(self):
        extract_kernel_body = _mock_codegen.extract_kernel_body

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
        extract_kernel_body = _mock_codegen.extract_kernel_body

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
        extract_kernel_body = _mock_codegen.extract_kernel_body

        source = "void kernel_main() {\n    int x = 1;\n}"
        body = extract_kernel_body(source)
        assert "int x = 1" in body

    def test_no_kernel_main(self):
        extract_kernel_body = _mock_codegen.extract_kernel_body

        source = "void other_function() { int x = 1; }"
        body = extract_kernel_body(source)
        assert body == ""

    def test_raw_string_with_braces(self):
        """Raw string literal with braces should not confuse brace matching."""
        extract_kernel_body = _mock_codegen.extract_kernel_body

        source = """
void kernel_main() {
    const char* s = R"({ "key": "value" })";
    int x = 1;
}
"""
        body = extract_kernel_body(source)
        assert "int x = 1" in body
        assert 'R"(' in body

    def test_raw_string_with_delimiter(self):
        """Raw string with custom delimiter containing braces and quotes."""
        extract_kernel_body = _mock_codegen.extract_kernel_body

        source = '''
void kernel_main() {
    const char* s = R"foo(
        }}} """ {{{
    )foo";
    int done = 1;
}
'''
        body = extract_kernel_body(source)
        assert "int done = 1" in body

    def test_prefixed_raw_string(self):
        """LR, uR, UR, u8R raw string prefixes."""
        extract_kernel_body = _mock_codegen.extract_kernel_body

        source = """
void kernel_main() {
    auto a = LR"({ braces })";
    auto b = uR"({ braces })";
    auto c = UR"({ braces })";
    auto d = u8R"({ braces })";
    int ok = 1;
}
"""
        body = extract_kernel_body(source)
        assert "int ok = 1" in body

    def test_identifier_ending_in_R_not_raw_string(self):
        """Variable named fooR followed by string is not a raw string."""
        extract_kernel_body = _mock_codegen.extract_kernel_body

        source = """
void kernel_main() {
    int fooR = 1;
    const char* s = "normal { string }";
    int ok = 1;
}
"""
        body = extract_kernel_body(source)
        assert "int ok = 1" in body

    def test_line_comment_with_braces(self):
        """Braces in line comments should be ignored."""
        extract_kernel_body = _mock_codegen.extract_kernel_body

        source = """
void kernel_main() {
    // this has a } that should not close the function
    int x = 1;
}
"""
        body = extract_kernel_body(source)
        assert "int x = 1" in body

    def test_block_comment_with_braces(self):
        """Braces in block comments should be ignored."""
        extract_kernel_body = _mock_codegen.extract_kernel_body

        source = """
void kernel_main() {
    /* } } } */
    int x = 1;
}
"""
        body = extract_kernel_body(source)
        assert "int x = 1" in body

    def test_char_literal_with_brace(self):
        """Char literal containing a brace character."""
        extract_kernel_body = _mock_codegen.extract_kernel_body

        source = """
void kernel_main() {
    char c = '}';
    int x = 1;
}
"""
        body = extract_kernel_body(source)
        assert "int x = 1" in body

    def test_escaped_quote_in_string(self):
        """Escaped quote followed by brace in string."""
        extract_kernel_body = _mock_codegen.extract_kernel_body

        source = r"""
void kernel_main() {
    const char* s = "escaped \" } quote";
    int x = 1;
}
"""
        body = extract_kernel_body(source)
        assert "int x = 1" in body

    def test_deeply_nested_braces(self):
        """Multiple levels of nested braces."""
        extract_kernel_body = _mock_codegen.extract_kernel_body

        source = """
void kernel_main() {
    for (int i = 0; i < 10; i++) {
        if (i > 5) {
            while (true) {
                break;
            }
        }
    }
    int done = 1;
}
"""
        body = extract_kernel_body(source)
        assert "int done = 1" in body
        assert "while" in body

    def test_mixed_comments_and_strings(self):
        """Body with interspersed comments, strings, and braces."""
        extract_kernel_body = _mock_codegen.extract_kernel_body

        source = """
void kernel_main() {
    // comment with }
    const char* a = "string with { and }";
    /* block
       comment } with } braces */
    char c = '{';
    if (true) {
        int x = 1;
    }
}
"""
        body = extract_kernel_body(source)
        assert "int x = 1" in body
        assert "char c" in body


class TestRuntimeArgRedirect:
    """Tests for the #define/#undef runtime arg redirect approach."""

    def test_wrapper_bakes_literal_offset(self):
        """Wrapper for phase 2 should bake the literal offset."""
        _emit_rt_arg_wrapper = _mock_codegen._emit_rt_arg_wrapper
        result = _emit_rt_arg_wrapper(2, 18)
        joined = "\n".join(result)
        assert "arg_idx + 18" in joined
        assert "phase_2_get_arg_val" in joined
        assert "get_named_compile_time_arg_val" not in joined

    def test_define_and_undef(self):
        """Define/undef should emit correct preprocessor directives."""
        _emit_rt_arg_define = _mock_codegen._emit_rt_arg_define
        _emit_rt_arg_undef = _mock_codegen._emit_rt_arg_undef
        assert _emit_rt_arg_define(2) == "#define get_arg_val phase_2_get_arg_val"
        assert _emit_rt_arg_undef() == "#undef get_arg_val"


class TestInlineLocalIncludesRelative:
    """Tests for relative path include inlining."""

    def test_relative_path_inlined(self, tmp_path):
        """Includes with relative paths are resolved and inlined."""
        inline_local_includes = _mock_codegen.inline_local_includes

        # Create directory structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        header = subdir / "helper.h"
        header.write_text("#pragma once\nint helper_val = 42;")

        source = '#include "subdir/helper.h"\nvoid kernel_main() {}'
        headers, remaining = inline_local_includes(source, str(tmp_path))
        assert len(headers) == 1
        assert "int helper_val = 42" in headers[0][1]
        assert "#pragma once" not in headers[0][1]

    def test_local_include_still_works(self, tmp_path):
        """Local includes (no path separator) still work."""
        inline_local_includes = _mock_codegen.inline_local_includes

        header = tmp_path / "local.h"
        header.write_text("int local_val = 1;")

        source = '#include "local.h"\nvoid kernel_main() {}'
        headers, remaining = inline_local_includes(source, str(tmp_path))
        assert len(headers) == 1
        assert "int local_val = 1" in headers[0][1]


class TestCBPoolAllocator:
    """Tests for CBPoolAllocator pool-based CB slot allocation."""

    def test_basic_allocation_same_config_reuses_slot(self):
        """Two phases with identical CB configs should reuse the same slot."""
        CBPoolAllocator = _mock_cb_allocator.CBPoolAllocator
        CBInfo = _mock_cb_allocator.CBInfo

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
        CBPoolAllocator = _mock_cb_allocator.CBPoolAllocator
        CBInfo = _mock_cb_allocator.CBInfo

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
        CBPoolAllocator = _mock_cb_allocator.CBPoolAllocator
        CBInfo = _mock_cb_allocator.CBInfo

        pool = CBPoolAllocator(max_slots=32)

        cb_info_0 = {0: CBInfo(0, 2048, "F16", 1024, None, "Default")}
        pool.allocate_phase(0, cb_info_0, set())

        cb_info_1 = {0: CBInfo(0, 4096, "F32", 1024, None, "Default")}
        pool.allocate_phase(1, cb_info_1, set())

        assert pool.get_remap(0)[0] != pool.get_remap(1)[0]

    def test_different_unpack_to_dest_mode_gets_separate_slot(self):
        """Different unpack_to_dest_mode on same (format, page_size) gets separate slots."""
        CBPoolAllocator = _mock_cb_allocator.CBPoolAllocator
        CBInfo = _mock_cb_allocator.CBInfo

        pool = CBPoolAllocator(max_slots=32)

        cb_info_0 = {0: CBInfo(0, 2048, "F16", 1024, None, "Default")}
        pool.allocate_phase(0, cb_info_0, set())

        cb_info_1 = {0: CBInfo(0, 2048, "F16", 1024, None, "UnpackToDestFp32")}
        pool.allocate_phase(1, cb_info_1, set())

        assert pool.get_remap(0)[0] != pool.get_remap(1)[0]

    def test_overflow_raises_error(self):
        """Exceeding max_slots raises ValueError."""
        CBPoolAllocator = _mock_cb_allocator.CBPoolAllocator
        CBInfo = _mock_cb_allocator.CBInfo

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
        CBPoolAllocator = _mock_cb_allocator.CBPoolAllocator
        CBInfo = _mock_cb_allocator.CBInfo

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
        CBPoolAllocator = _mock_cb_allocator.CBPoolAllocator
        CBInfo = _mock_cb_allocator.CBInfo

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
        CBPoolAllocator = _mock_cb_allocator.CBPoolAllocator
        CBInfo = _mock_cb_allocator.CBInfo

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
        CBPoolAllocator = _mock_cb_allocator.CBPoolAllocator
        CBInfo = _mock_cb_allocator.CBInfo

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
        CBPoolAllocator = _mock_cb_allocator.CBPoolAllocator
        CBInfo = _mock_cb_allocator.CBInfo

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
        merge_fn = _mock_codegen._merge_named_compile_time_args

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
        merge_fn = _mock_codegen._merge_named_compile_time_args

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
        assert result_dict["phase_1_cb_in0"] == 5  # Phase 1: remapped + prefixed


def _make_mock_core_range_set(coord_ranges):
    """Create a mock CoreRangeSet from a list of ((sx,sy),(ex,ey)) pairs.

    Each pair defines a rectangular CoreRange from (sx,sy) to (ex,ey).
    The mock supports .ranges() iteration with .start.x/y and .end.x/y
    attributes, matching the real CoreRangeSet API used by
    _core_range_set_to_coords.
    """
    ranges = []
    for (sx, sy), (ex, ey) in coord_ranges:
        cr = MagicMock()
        cr.start.x = sx
        cr.start.y = sy
        cr.end.x = ex
        cr.end.y = ey
        ranges.append(cr)
    crs = MagicMock()
    crs.ranges.return_value = ranges
    return crs


def _make_op_with_cores(core_range_set):
    """Create a mock OpDescriptor whose kernels use the given core ranges.

    The returned mock has .descriptor.kernels[0].core_ranges set to the
    provided CoreRangeSet, matching the structure expected by
    _get_node_core_range().
    """
    kernel = MagicMock()
    kernel.core_ranges = core_range_set
    descriptor = MagicMock()
    descriptor.kernels = [kernel]
    op = MagicMock()
    op.descriptor = descriptor
    return op


# Monkeypatch _get_node_core_range: the real implementation round-trips through
# _coords_to_core_range_set (which uses mocked ttnn.CoreRange/CoreRangeSet,
# producing unusable MagicMock objects).  Since each test mock op has exactly
# one kernel, returning that kernel's core_ranges directly is equivalent and
# keeps the mock CoreRangeSet that _core_range_set_to_coords can iterate.
_mock_graph._get_node_core_range = lambda node: node.op.descriptor.kernels[0].core_ranges


class TestOpGraphTopologyValidation:
    """Tests for OpGraph topology validation."""

    def test_overlapping_siblings_rejected(self):
        """Two children of root with overlapping core ranges should be rejected."""
        OpGraphBuilder = _mock_graph.OpGraphBuilder
        OpNode = _mock_graph.OpNode

        # Root covers (0,0)-(3,0); children overlap at (1,0) and (2,0)
        root_op = _make_op_with_cores(_make_mock_core_range_set([((0, 0), (3, 0))]))
        child_a = OpNode(_make_op_with_cores(_make_mock_core_range_set([((0, 0), (2, 0))])))
        child_b = OpNode(_make_op_with_cores(_make_mock_core_range_set([((1, 0), (3, 0))])))
        root = OpNode(root_op, children=[child_a, child_b])

        builder = OpGraphBuilder(root)
        with pytest.raises(ValueError, match="overlapping cores"):
            builder._validate_topology()

    def test_child_outside_parent_rejected(self):
        """Child range that extends beyond its parent should be rejected."""
        OpGraphBuilder = _mock_graph.OpGraphBuilder
        OpNode = _mock_graph.OpNode

        # Intermediate parent covers (0,0)-(3,0), but one child extends to (5,0)
        parent_op = _make_op_with_cores(_make_mock_core_range_set([((0, 0), (3, 0))]))
        leaf_a = OpNode(_make_op_with_cores(_make_mock_core_range_set([((0, 0), (1, 0))])))
        leaf_b = OpNode(_make_op_with_cores(_make_mock_core_range_set([((2, 0), (5, 0))])))
        parent_node = OpNode(parent_op, children=[leaf_a, leaf_b])

        # Root covers the full superset so the parent is valid under root
        root_op = _make_op_with_cores(_make_mock_core_range_set([((0, 0), (5, 0))]))
        root = OpNode(root_op, children=[parent_node])

        builder = OpGraphBuilder(root)
        with pytest.raises(ValueError, match="outside parent range"):
            builder._validate_topology()

    def test_overlapping_children_rejected(self):
        """Sibling children with overlapping ranges should be rejected."""
        OpGraphBuilder = _mock_graph.OpGraphBuilder
        OpNode = _mock_graph.OpNode

        # Parent covers (0,0)-(5,0), children overlap at (2,0) and (3,0)
        parent_op = _make_op_with_cores(_make_mock_core_range_set([((0, 0), (5, 0))]))
        leaf_a = OpNode(_make_op_with_cores(_make_mock_core_range_set([((0, 0), (3, 0))])))
        leaf_b = OpNode(_make_op_with_cores(_make_mock_core_range_set([((2, 0), (5, 0))])))
        root = OpNode(parent_op, children=[leaf_a, leaf_b])

        builder = OpGraphBuilder(root)
        with pytest.raises(ValueError, match="overlapping cores"):
            builder._validate_topology()

    def test_leaf_nodes_accepted(self):
        """Leaf nodes (children with no further descendants) should be accepted."""
        OpGraphBuilder = _mock_graph.OpGraphBuilder
        OpNode = _mock_graph.OpNode

        root_op = _make_op_with_cores(_make_mock_core_range_set([((0, 0), (3, 0))]))
        child_a = OpNode(_make_op_with_cores(_make_mock_core_range_set([((0, 0), (1, 0))])))
        child_b = OpNode(_make_op_with_cores(_make_mock_core_range_set([((2, 0), (3, 0))])))
        root = OpNode(root_op, children=[child_a, child_b])

        builder = OpGraphBuilder(root)
        # Should not raise
        builder._validate_topology()

    def test_valid_topology_accepted(self):
        """Valid 2-branch split should pass validation."""
        OpGraphBuilder = _mock_graph.OpGraphBuilder
        OpNode = _mock_graph.OpNode

        root_op = _make_op_with_cores(_make_mock_core_range_set([((0, 0), (3, 0))]))
        child_a = OpNode(_make_op_with_cores(_make_mock_core_range_set([((0, 0), (1, 0))])))
        child_b = OpNode(_make_op_with_cores(_make_mock_core_range_set([((2, 0), (3, 0))])))
        root = OpNode(root_op, children=[child_a, child_b])

        builder = OpGraphBuilder(root)
        # Should not raise
        builder._validate_topology()

    def test_valid_nested_topology_accepted(self):
        """Valid nested branching should pass validation."""
        OpGraphBuilder = _mock_graph.OpGraphBuilder
        OpNode = _mock_graph.OpNode

        # Tree: root(0-5) -> mid(0-3, children=[leaf(0-1), leaf(2-3)]), leaf(4-5)
        leaf_0_1 = OpNode(_make_op_with_cores(_make_mock_core_range_set([((0, 0), (1, 0))])))
        leaf_2_3 = OpNode(_make_op_with_cores(_make_mock_core_range_set([((2, 0), (3, 0))])))
        mid_0_3 = OpNode(
            _make_op_with_cores(_make_mock_core_range_set([((0, 0), (3, 0))])),
            children=[leaf_0_1, leaf_2_3],
        )
        leaf_4_5 = OpNode(_make_op_with_cores(_make_mock_core_range_set([((4, 0), (5, 0))])))
        root_op = _make_op_with_cores(_make_mock_core_range_set([((0, 0), (5, 0))]))
        root = OpNode(root_op, children=[mid_0_3, leaf_4_5])

        builder = OpGraphBuilder(root)
        # Should not raise
        builder._validate_topology()

    def test_partial_coverage_accepted(self):
        """Children that don't fully tile the parent should be accepted.

        Unused parent cores simply don't participate in child phases.
        """
        OpGraphBuilder = _mock_graph.OpGraphBuilder
        OpNode = _mock_graph.OpNode

        # Parent covers (0,0)-(5,0) = 6 cores, but children only use 4
        parent_op = _make_op_with_cores(_make_mock_core_range_set([((0, 0), (5, 0))]))
        child_a = OpNode(_make_op_with_cores(_make_mock_core_range_set([((0, 0), (1, 0))])))
        child_b = OpNode(_make_op_with_cores(_make_mock_core_range_set([((2, 0), (3, 0))])))
        # Cores (4,0)-(5,0) intentionally unused
        root = OpNode(parent_op, children=[child_a, child_b])

        builder = OpGraphBuilder(root)
        # Should not raise -- partial coverage is allowed
        builder._validate_topology()

    def test_intermediate_node_with_children_accepted(self):
        """Intermediate node (has children) is valid -- it runs its op then branches."""
        OpGraphBuilder = _mock_graph.OpGraphBuilder
        OpNode = _mock_graph.OpNode

        # Intermediate node has an op AND children -- every OpNode has exactly one op
        mid_op = _make_op_with_cores(_make_mock_core_range_set([((0, 0), (3, 0))]))
        leaf_a = OpNode(_make_op_with_cores(_make_mock_core_range_set([((0, 0), (1, 0))])))
        leaf_b = OpNode(_make_op_with_cores(_make_mock_core_range_set([((2, 0), (3, 0))])))
        root = OpNode(mid_op, children=[leaf_a, leaf_b])

        builder = OpGraphBuilder(root)
        # Should not raise
        builder._validate_topology()


class TestEffectiveLeafRange:
    """Tests for OpGraphBuilder._effective_leaf_range."""

    def test_leaf_node_returns_own_range(self):
        """Leaf node should return its own core coords."""
        OpNode = _mock_graph.OpNode
        OpGraphBuilder = _mock_graph.OpGraphBuilder

        node = OpNode(_make_op_with_cores(_make_mock_core_range_set([((0, 0), (1, 0))])))

        result = OpGraphBuilder._effective_leaf_range(node)
        assert result == {(0, 0), (1, 0)}

    def test_intermediate_node_returns_leaf_union(self):
        """Intermediate node should return union of all descendant leaf ranges."""
        OpNode = _mock_graph.OpNode
        OpGraphBuilder = _mock_graph.OpGraphBuilder

        leaf_a = OpNode(_make_op_with_cores(_make_mock_core_range_set([((0, 0), (1, 0))])))
        leaf_b = OpNode(_make_op_with_cores(_make_mock_core_range_set([((4, 0), (5, 0))])))
        # Gap at (2,0)-(3,0)
        node = OpNode(
            _make_op_with_cores(_make_mock_core_range_set([((0, 0), (5, 0))])),
            children=[leaf_a, leaf_b],
        )

        result = OpGraphBuilder._effective_leaf_range(node)
        assert result == {(0, 0), (1, 0), (4, 0), (5, 0)}
        # Cores (2,0) and (3,0) NOT included -- they're unused

    def test_nested_effective_range(self):
        """Deeply nested tree should return only the leaf-level core coords."""
        OpNode = _mock_graph.OpNode
        OpGraphBuilder = _mock_graph.OpGraphBuilder

        # Tree: mid(0-3) -> [leaf(0-1), leaf(2-3)]
        # Effective = {0,1,2,3}
        leaf_0_1 = OpNode(_make_op_with_cores(_make_mock_core_range_set([((0, 0), (1, 0))])))
        leaf_2_3 = OpNode(_make_op_with_cores(_make_mock_core_range_set([((2, 0), (3, 0))])))
        mid = OpNode(
            _make_op_with_cores(_make_mock_core_range_set([((0, 0), (3, 0))])),
            children=[leaf_0_1, leaf_2_3],
        )

        result = OpGraphBuilder._effective_leaf_range(mid)
        assert result == {(0, 0), (1, 0), (2, 0), (3, 0)}


class TestCBRestoreVerification:
    """Tests for _verify_cb_restore (module-level function)."""

    def test_verify_passes_on_correct_restore(self):
        """Verification should pass when state matches."""
        _verify_cb_restore = _mock_cb_allocator._verify_cb_restore

        cb = MagicMock()
        cb.total_size = 4096
        fmt = MagicMock()
        fmt.buffer_index = 5

        saved = [{"cb": cb, "total_size": 4096, "fmt": [(fmt, 5)]}]
        # Should not raise
        _verify_cb_restore(saved)

    def test_verify_fails_on_total_size_mismatch(self):
        """Verification should fail when total_size doesn't match."""
        _verify_cb_restore = _mock_cb_allocator._verify_cb_restore

        cb = MagicMock()
        cb.total_size = 8192  # Different from saved!
        fmt = MagicMock()
        fmt.buffer_index = 5

        saved = [{"cb": cb, "total_size": 4096, "fmt": [(fmt, 5)]}]
        with pytest.raises(RuntimeError, match="total_size"):
            _verify_cb_restore(saved)

    def test_verify_fails_on_buffer_index_mismatch(self):
        """Verification should fail when buffer_index doesn't match."""
        _verify_cb_restore = _mock_cb_allocator._verify_cb_restore

        cb = MagicMock()
        cb.total_size = 4096
        fmt = MagicMock()
        fmt.buffer_index = 10  # Different from saved!

        saved = [{"cb": cb, "total_size": 4096, "fmt": [(fmt, 5)]}]
        with pytest.raises(RuntimeError, match="buffer_index"):
            _verify_cb_restore(saved)


class TestCompileTimePerformance:
    """Tests that build pipeline performance doesn't regress catastrophically.

    These measure the Python-side build pipeline time (CB pool allocation,
    source generation, merge) with generous thresholds to avoid CI flakiness.
    """

    def _make_mock_phase(self, cb_indices=None, kernel_source="void kernel_main() {}"):
        """Create a mock OpDescriptor with realistic CB/kernel structure."""
        if cb_indices is None:
            cb_indices = [0, 1, 16]

        descriptor = MagicMock()

        # Mock CB descriptors
        cbs = []
        for idx in cb_indices:
            cb = MagicMock()
            cb.total_size = 2048
            cb.core_ranges = _make_mock_core_ranges()
            cb.has_global_circular_buffer.return_value = False
            cb.has_buffer.return_value = False

            fmt = MagicMock()
            fmt.buffer_index = idx
            fmt.data_format = "Float16_b"
            fmt.page_size = 2048
            cb.format_descriptors = [fmt]
            cbs.append(cb)
        descriptor.cbs = cbs

        # Mock kernels
        reader = MagicMock()
        reader.core_ranges = _make_mock_core_ranges()
        reader.kernel_source = kernel_source
        reader.defines = {}
        reader.compile_time_args = [32, 4]
        reader.named_compile_time_args = [("cb_in", 0), ("cb_out", 16)]
        reader.runtime_args = _make_mock_runtime_args_view([[1, 2, 3]])

        writer = MagicMock()
        writer.core_ranges = _make_mock_core_ranges()
        writer.kernel_source = kernel_source
        writer.defines = {}
        writer.compile_time_args = [32]
        writer.named_compile_time_args = [("cb_in", 0), ("cb_out", 16)]
        writer.runtime_args = _make_mock_runtime_args_view([[4, 5]])

        compute = MagicMock()
        compute.core_ranges = _make_mock_core_ranges()
        compute.kernel_source = kernel_source
        compute.defines = {}
        compute.compile_time_args = [8, 1]
        compute.named_compile_time_args = [("cb_in", 0), ("cb_out", 16)]
        compute.runtime_args = _make_mock_runtime_args_view([[6]])

        descriptor.kernels = [reader, writer, compute]

        op = _mock_fusion.OpDescriptor(
            descriptor=descriptor,
            input_tensors=[MagicMock()],
            output_tensors=[MagicMock()],
        )
        return op

    def _run_cb_pool_allocation(self, n_phases):
        """Run CB pool allocation pipeline for n_phases, return elapsed time."""
        import time

        CBPoolAllocator = _mock_cb_allocator.CBPoolAllocator
        extract = _mock_cb_allocator.extract_cb_info

        pool = CBPoolAllocator()
        phases = [self._make_mock_phase() for _ in range(n_phases)]

        start = time.perf_counter()
        for phase_idx, phase in enumerate(phases):
            cb_info = extract(phase.descriptor)
            pool.allocate_phase(phase_idx, cb_info, phantom_cb_indices=set())
        elapsed = time.perf_counter() - start
        return elapsed

    def test_two_phase_build_time(self):
        """2-phase fused chain build should complete quickly."""
        elapsed = self._run_cb_pool_allocation(2)
        assert elapsed < 5.0, f"2-phase CB allocation took {elapsed:.2f}s (limit: 5s)"

    def test_four_phase_build_time(self):
        """4-phase fused chain build should complete in reasonable time."""
        elapsed = self._run_cb_pool_allocation(4)
        assert elapsed < 10.0, f"4-phase CB allocation took {elapsed:.2f}s (limit: 10s)"

    def test_build_time_scales_linearly(self):
        """Build time should scale roughly linearly with phase count, not quadratically."""
        times = {}
        for n_phases in [2, 4, 8]:
            times[n_phases] = self._run_cb_pool_allocation(n_phases)

        # 8-phase should be no more than 8x the 2-phase time (linear scaling).
        # With quadratic scaling it would be 16x. Use generous 10x limit.
        if times[2] > 0.001:  # Only check if measurable
            ratio = times[8] / times[2]
            assert ratio < 10.0, (
                f"Scaling appears super-linear: 2-phase={times[2]:.4f}s, "
                f"8-phase={times[8]:.4f}s, ratio={ratio:.1f}x (limit: 10x)"
            )


def _make_mock_core_ranges_multi(coords):
    """Create a mock CoreRangeSet for multiple cores.

    coords: list of (x, y) tuples.
    """
    ranges = []
    for x, y in coords:
        cr = MagicMock()
        cr.start.x = x
        cr.start.y = y
        cr.end.x = x
        cr.end.y = y
        ranges.append(cr)
    crs = MagicMock()
    crs.ranges.return_value = ranges
    return crs


def _make_mock_runtime_args_view_multi(args_by_coord):
    """Create a mock RuntimeArgsView with coordinate-based 2D indexing.

    args_by_coord: dict mapping (x, y) -> list of int args.
    rv[x][y] -> args list for that core.
    """
    view = MagicMock()

    def get_col(x):
        col = MagicMock()

        def get_row(y):
            key = (x, y)
            if key in args_by_coord:
                return args_by_coord[key]
            raise KeyError(f"No args for core ({x},{y})")

        col.__getitem__ = MagicMock(side_effect=get_row)
        return col

    view.__getitem__ = MagicMock(side_effect=get_col)
    return view


class _SimpleCoreCoord:
    """Simple CoreCoord mock with proper .x/.y attributes."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"CoreCoord({self.x}, {self.y})"


class TestRuntimeArgsPadding:
    """Tests that _compute_and_concatenate_runtime_args pads correctly."""

    def _patch_core_coord(self):
        """Temporarily set ttnn.CoreCoord to return objects with proper .x/.y."""
        original = _mock_codegen.ttnn.CoreCoord
        _mock_codegen.ttnn.CoreCoord = _SimpleCoreCoord
        return original

    def _restore_core_coord(self, original):
        _mock_codegen.ttnn.CoreCoord = original

    def test_padding_matches_offsets(self):
        """Core A has 5 args, core B has 3 — both should pad to 5 for phase offset alignment."""
        original = self._patch_core_coord()
        try:
            _compute_and_concatenate_runtime_args = _mock_codegen._compute_and_concatenate_runtime_args

            coords = [(0, 0), (1, 0)]
            core_ranges = _make_mock_core_ranges_multi(coords)

            # Phase 0: core (0,0) has 5 args, core (1,0) has 3 args
            kernel0 = MagicMock()
            kernel0.runtime_args = _make_mock_runtime_args_view_multi(
                {
                    (0, 0): [1, 2, 3, 4, 5],
                    (1, 0): [10, 20, 30],
                }
            )
            kernel0.core_ranges = core_ranges

            # Phase 1: both cores have 2 args
            kernel1 = MagicMock()
            kernel1.runtime_args = _make_mock_runtime_args_view_multi(
                {
                    (0, 0): [100, 200],
                    (1, 0): [300, 400],
                }
            )
            kernel1.core_ranges = core_ranges

            phase_kernels = [{"reader": kernel0}, {"reader": kernel1}]

            offsets, result = _compute_and_concatenate_runtime_args(phase_kernels, "reader")

            assert offsets[0] == 0
            assert offsets[1] == 5  # max_args for phase 0 = 5

            # Find args for each core
            result_dict = {(c.x, c.y): args for c, args in result}

            # Core (0,0): [1,2,3,4,5] + [100,200] = 7 args
            assert result_dict[(0, 0)] == [1, 2, 3, 4, 5, 100, 200]

            # Core (1,0): [10,20,30] + 2 padding zeros + [300,400] = 7 args
            assert result_dict[(1, 0)] == [10, 20, 30, 0, 0, 300, 400]

            # Both cores have same total length
            assert len(result_dict[(0, 0)]) == len(result_dict[(1, 0)])
        finally:
            self._restore_core_coord(original)

    def test_missing_core_padded(self):
        """A core with no args in a phase gets zero-padded to max width."""
        original = self._patch_core_coord()
        try:
            _compute_and_concatenate_runtime_args = _mock_codegen._compute_and_concatenate_runtime_args

            coords = [(0, 0), (1, 0)]
            core_ranges = _make_mock_core_ranges_multi(coords)

            # Phase 0: only core (0,0) has args
            kernel0 = MagicMock()
            kernel0.runtime_args = _make_mock_runtime_args_view_multi(
                {
                    (0, 0): [1, 2, 3],
                }
            )
            kernel0.core_ranges = core_ranges

            phase_kernels = [{"reader": kernel0}]

            _, result = _compute_and_concatenate_runtime_args(
                phase_kernels,
                "reader",
                target_core_range=core_ranges,
            )

            result_dict = {(c.x, c.y): args for c, args in result}

            # Core (0,0) has [1,2,3]
            assert result_dict[(0, 0)] == [1, 2, 3]

            # Core (1,0) has [0,0,0] — padded to phase max
            assert result_dict[(1, 0)] == [0, 0, 0]
        finally:
            self._restore_core_coord(original)

    def test_uniform_args_no_padding(self):
        """When all cores have same arg count, no padding is needed."""
        original = self._patch_core_coord()
        try:
            _compute_and_concatenate_runtime_args = _mock_codegen._compute_and_concatenate_runtime_args

            coords = [(0, 0), (1, 0)]
            core_ranges = _make_mock_core_ranges_multi(coords)

            kernel0 = MagicMock()
            kernel0.runtime_args = _make_mock_runtime_args_view_multi(
                {
                    (0, 0): [1, 2, 3],
                    (1, 0): [4, 5, 6],
                }
            )
            kernel0.core_ranges = core_ranges

            phase_kernels = [{"reader": kernel0}]
            _, result = _compute_and_concatenate_runtime_args(phase_kernels, "reader")

            result_dict = {(c.x, c.y): args for c, args in result}
            assert result_dict[(0, 0)] == [1, 2, 3]
            assert result_dict[(1, 0)] == [4, 5, 6]
        finally:
            self._restore_core_coord(original)


class TestCBArgNaming:
    """Tests for _is_cb_named_arg and centralized CB naming convention."""

    def test_valid_cb_arg_identified(self):
        """Standard CB args with valid index are identified."""
        _is_cb_named_arg = _mock_cb_allocator._is_cb_named_arg
        assert _is_cb_named_arg("cb_in", 0) is True
        assert _is_cb_named_arg("cb_out", 16) is True
        assert _is_cb_named_arg("cb_scaler", 2) is True
        assert _is_cb_named_arg("cb_gamma", 31) is True

    def test_non_cb_prefix_excluded(self):
        """Args without cb_ prefix are not identified as CB args."""
        _is_cb_named_arg = _mock_cb_allocator._is_cb_named_arg
        assert _is_cb_named_arg("blk", 4) is False
        assert _is_cb_named_arg("Ht", 8) is False
        assert _is_cb_named_arg("num_tiles", 16) is False

    def test_out_of_range_value_excluded(self):
        """CB-prefixed args with values outside [0,31] are excluded."""
        _is_cb_named_arg = _mock_cb_allocator._is_cb_named_arg
        assert _is_cb_named_arg("cb_debug_flag", 100) is False
        assert _is_cb_named_arg("cb_count", -1) is False
        assert _is_cb_named_arg("cb_large", 32) is False
        assert _is_cb_named_arg("cb_str", "not_an_int") is False


class TestCBStateSaveRestore:
    """Tests for module-level _save_cb_state / _restore_cb_state."""

    def test_save_and_restore_round_trip(self):
        """Save/restore should preserve original values after mutation."""
        _save_cb_state = _mock_cb_allocator._save_cb_state
        _restore_cb_state = _mock_cb_allocator._restore_cb_state
        _verify_cb_restore = _mock_cb_allocator._verify_cb_restore

        # Create a mock ProgramDescriptor with one CB
        fmt = MagicMock()
        fmt.buffer_index = 5
        cb = MagicMock()
        cb.total_size = 4096
        cb.format_descriptors = [fmt]
        prog_desc = MagicMock()
        prog_desc.cbs = [cb]

        saved = _save_cb_state([prog_desc])

        # Mutate (simulating what _build_fused_descriptor does)
        cb.total_size = 8192
        fmt.buffer_index = 10

        # Restore
        _restore_cb_state(saved)

        assert cb.total_size == 4096
        assert fmt.buffer_index == 5

        # Verify should pass
        _verify_cb_restore(saved)

    def test_deduplicates_shared_cbs(self):
        """When multiple phases share the same CB object, save it once."""
        _save_cb_state = _mock_cb_allocator._save_cb_state

        fmt = MagicMock()
        fmt.buffer_index = 3
        cb = MagicMock()
        cb.total_size = 2048
        cb.format_descriptors = [fmt]

        # Two program descriptors sharing the SAME CB object
        prog1 = MagicMock()
        prog1.cbs = [cb]
        prog2 = MagicMock()
        prog2.cbs = [cb]

        saved = _save_cb_state([prog1, prog2])
        # Only saved once despite appearing in two programs
        assert len(saved) == 1


class TestSemaphoreInitialValue:
    """Tests that semaphore reset uses initial_value, not hardcoded 0."""

    def test_nonzero_initial_value_in_source(self):
        """Generated source should reset semaphore to initial_value=5, not 0."""
        _generate_fused_source = _mock_codegen._generate_fused_source

        # Create minimal mock data for a 2-phase chain
        core_ranges = _make_mock_core_ranges()

        # source_type must be the exact same object as ttnn.KernelDescriptor.SourceType.SOURCE_CODE
        source_code_type = _mock_codegen.ttnn.KernelDescriptor.SourceType.SOURCE_CODE

        kernel = MagicMock()
        kernel.kernel_source = "void kernel_main() { /* reader */ }"
        kernel.source_type = source_code_type
        kernel.defines = []
        kernel.core_ranges = core_ranges
        kernel.named_compile_time_args = [("cb_in", 0)]

        PhaseInfo = _mock_cb_allocator.PhaseInfo
        CBInfo = _mock_cb_allocator.CBInfo

        phase0 = PhaseInfo(
            phase_idx=0,
            op_descriptor=MagicMock(),
            cb_info={0: CBInfo(0, 2048, MagicMock(), 2048, core_ranges, False, MagicMock())},
        )
        phase1 = PhaseInfo(
            phase_idx=1,
            op_descriptor=MagicMock(),
            cb_info={0: CBInfo(0, 2048, MagicMock(), 2048, core_ranges, False, MagicMock())},
        )
        phases = [phase0, phase1]

        phase_kernels = [{"reader": kernel}, {"reader": kernel}]

        MultiBarrierSpec = _mock_common.MultiBarrierSpec
        BarrierSegment = _mock_common.BarrierSegment
        BarrierConfig = _mock_common.BarrierConfig

        seg = BarrierSegment(
            config=BarrierConfig(
                num_cores=1,
                core0_phys_x=1,
                core0_phys_y=1,
                mcast_start_x=1,
                mcast_start_y=1,
                mcast_end_x=1,
                mcast_end_y=1,
            ),
            arrive_addr=3000,
            release_addr=4000,
        )
        multi_barrier = MultiBarrierSpec(
            segments=[seg],
            compute_done_addr=1000,
            writer_done_addr=2000,
            transition_map={0: (0, 0)},
        )

        result = _generate_fused_source(
            phase_kernels,
            "reader",
            phases,
            {0: 0, 1: 0},  # ct_offsets
            [[0]],  # per_phase_cb_slots
            risc_type="riscv_0",
            role_label="reader",
            rebind_info={},
            op_semaphore_info=[(3, 5)],  # sem_id=3, initial_value=5
            multi_barrier=multi_barrier,
        )

        assert result is not None
        assert "get_semaphore(3)) = 5;" in result
        assert "get_semaphore(3)) = 0;" not in result

    def test_mixed_initial_values(self):
        """Multiple semaphores with different initial values get their own values."""
        _generate_fused_source = _mock_codegen._generate_fused_source

        source_code_type = _mock_codegen.ttnn.KernelDescriptor.SourceType.SOURCE_CODE

        core_ranges = _make_mock_core_ranges()
        kernel = MagicMock()
        kernel.kernel_source = "void kernel_main() { /* reader */ }"
        kernel.source_type = source_code_type
        kernel.defines = []
        kernel.core_ranges = core_ranges
        kernel.named_compile_time_args = [("cb_in", 0)]

        PhaseInfo = _mock_cb_allocator.PhaseInfo
        CBInfo = _mock_cb_allocator.CBInfo

        phase0 = PhaseInfo(0, MagicMock(), {0: CBInfo(0, 2048, MagicMock(), 2048, core_ranges, False, MagicMock())})
        phase1 = PhaseInfo(1, MagicMock(), {0: CBInfo(0, 2048, MagicMock(), 2048, core_ranges, False, MagicMock())})

        phase_kernels = [{"reader": kernel}, {"reader": kernel}]

        MultiBarrierSpec = _mock_common.MultiBarrierSpec
        BarrierSegment = _mock_common.BarrierSegment
        BarrierConfig = _mock_common.BarrierConfig

        seg = BarrierSegment(
            config=BarrierConfig(
                num_cores=1,
                core0_phys_x=1,
                core0_phys_y=1,
                mcast_start_x=1,
                mcast_start_y=1,
                mcast_end_x=1,
                mcast_end_y=1,
            ),
            arrive_addr=3000,
            release_addr=4000,
        )
        multi_barrier = MultiBarrierSpec(
            segments=[seg],
            compute_done_addr=1000,
            writer_done_addr=2000,
            transition_map={0: (0, 0)},
        )

        result = _generate_fused_source(
            phase_kernels,
            "reader",
            [phase0, phase1],
            {0: 0, 1: 0},
            [[0]],  # per_phase_cb_slots
            risc_type="riscv_0",
            role_label="reader",
            rebind_info={},
            op_semaphore_info=[(2, 0), (7, 3)],  # sem 2 -> 0, sem 7 -> 3
            multi_barrier=multi_barrier,
        )

        assert result is not None
        assert "get_semaphore(2)) = 0;" in result
        assert "get_semaphore(7)) = 3;" in result


class TestMustMatchDefines:
    """Tests for MUST_MATCH_DEFINES validation in _collect_phase_defines."""

    def test_matching_accepted(self):
        """Same REDUCE_OP value across phases should be accepted."""
        _collect = _mock_codegen._collect_phase_defines

        phase_kernels = [
            {"compute": MagicMock(defines=[("REDUCE_OP", "0"), ("REDUCE_DIM", "1")])},
            {"compute": MagicMock(defines=[("REDUCE_OP", "0"), ("REDUCE_DIM", "1")])},
        ]

        must_match, _ = _collect(phase_kernels, "compute")
        names = [name for name, _ in must_match]
        assert "REDUCE_OP" in names
        assert "REDUCE_DIM" in names

    def test_mismatched_rejected(self):
        """Different REDUCE_OP values across phases should raise ValueError."""
        _collect = _mock_codegen._collect_phase_defines

        phase_kernels = [
            {"compute": MagicMock(defines=[("REDUCE_OP", "0")])},
            {"compute": MagicMock(defines=[("REDUCE_OP", "1")])},
        ]

        with pytest.raises(ValueError, match="REDUCE_OP.*inconsistent"):
            _collect(phase_kernels, "compute")

    def test_present_in_one_phase_only(self):
        """MUST_MATCH define present in only one phase should not raise and should be in must_match."""
        _collect = _mock_codegen._collect_phase_defines

        phase_kernels = [
            {"compute": MagicMock(defines=[("REDUCE_OP", "0"), ("BCAST_LLKOP", "2")])},
            {"compute": MagicMock(defines=[])},
        ]

        must_match, _ = _collect(phase_kernels, "compute")
        must_match_names = [name for name, _ in must_match]
        assert "REDUCE_OP" in must_match_names
        assert "BCAST_LLKOP" in must_match_names


# =============================================================================
# Sequential / Parallel High-Level API Tests
# =============================================================================


def _make_mock_op(name="op"):
    """Create a mock OpDescriptor instance for tree-structure tests.

    Uses the real OpDescriptor NamedTuple with mock field values.
    """
    OpDescriptor = _mock_fusion.OpDescriptor
    return OpDescriptor(
        descriptor=MagicMock(name=f"{name}_desc"),
        input_tensors=[MagicMock(name=f"{name}_in")],
        output_tensors=[MagicMock(name=f"{name}_out")],
    )


def _tree_shape(node):
    """Return a nested tuple representing the tree shape.

    Each node becomes (node.op, [child_shapes...]).  This makes it easy
    to assert tree structure without comparing MagicMock internals.
    """
    if type(node).__name__ != "OpNode":
        raise TypeError(f"Expected OpNode, got {type(node)}")
    return (node.op, [_tree_shape(c) for c in node.children])


class TestSequentialParallelAPI:
    """Tests for the Sequential/Parallel high-level API."""

    def test_linear_chain(self):
        """Sequential(a, b, c) produces a→b→c chain."""
        _resolve = _mock_fusion._resolve

        a, b, c = _make_mock_op("a"), _make_mock_op("b"), _make_mock_op("c")
        seq = _mock_fusion.Sequential(a, b, c)
        nodes = _resolve(seq)

        assert len(nodes) == 1
        root = nodes[0]
        assert root.op is a
        assert len(root.children) == 1
        assert root.children[0].op is b
        assert len(root.children[0].children) == 1
        assert root.children[0].children[0].op is c
        assert root.children[0].children[0].children == []

    def test_stem_and_branches(self):
        """Sequential(a, Parallel(b, c)) produces a→[b, c]."""
        _resolve = _mock_fusion._resolve

        a, b, c = _make_mock_op("a"), _make_mock_op("b"), _make_mock_op("c")
        seq = _mock_fusion.Sequential(a, _mock_fusion.Parallel(b, c))
        nodes = _resolve(seq)

        assert len(nodes) == 1
        root = nodes[0]
        assert root.op is a
        assert len(root.children) == 2
        assert root.children[0].op is b
        assert root.children[1].op is c

    def test_nested_sequential_flattening(self):
        """Sequential(Sequential(a, b), c) flattens to a→b→c."""
        _resolve = _mock_fusion._resolve

        a, b, c = _make_mock_op("a"), _make_mock_op("b"), _make_mock_op("c")
        inner = _mock_fusion.Sequential(a, b)
        outer = _mock_fusion.Sequential(inner, c)
        nodes = _resolve(outer)

        assert len(nodes) == 1
        root = nodes[0]
        assert root.op is a
        assert len(root.children) == 1
        assert root.children[0].op is b
        assert len(root.children[0].children) == 1
        assert root.children[0].children[0].op is c

    def test_nested_branches(self):
        """Sequential(a, Parallel(Sequential(b, c), d)) produces correct tree."""
        _resolve = _mock_fusion._resolve
        Sequential = _mock_fusion.Sequential
        Parallel = _mock_fusion.Parallel

        a, b, c, d = [_make_mock_op(n) for n in "abcd"]
        seq = Sequential(a, Parallel(Sequential(b, c), d))
        nodes = _resolve(seq)

        assert len(nodes) == 1
        root = nodes[0]
        assert root.op is a
        assert len(root.children) == 2
        # First branch: b→c
        assert root.children[0].op is b
        assert len(root.children[0].children) == 1
        assert root.children[0].children[0].op is c
        # Second branch: d (leaf)
        assert root.children[1].op is d
        assert root.children[1].children == []

    def test_add_method(self):
        """Incremental .add() matches inline construction."""
        _resolve = _mock_fusion._resolve
        Sequential = _mock_fusion.Sequential
        Parallel = _mock_fusion.Parallel

        a, b, c = _make_mock_op("a"), _make_mock_op("b"), _make_mock_op("c")

        # Inline
        inline = Sequential(a, b, c)
        inline_nodes = _resolve(inline)

        # Incremental
        incremental = Sequential(a)
        incremental.add(b)
        incremental.add(c)
        inc_nodes = _resolve(incremental)

        # Same shape
        assert _tree_shape(inline_nodes[0]) == _tree_shape(inc_nodes[0])

    def test_add_chaining(self):
        """s.add(a).add(b) returns self."""
        Sequential = _mock_fusion.Sequential
        a, b, c = _make_mock_op("a"), _make_mock_op("b"), _make_mock_op("c")

        s = Sequential(a)
        result = s.add(b).add(c)
        assert result is s
        assert len(s._items) == 3

    def test_single_op(self):
        """Sequential(op) produces a single OpNode(op)."""
        _resolve = _mock_fusion._resolve
        Sequential = _mock_fusion.Sequential

        a = _make_mock_op("a")
        nodes = _resolve(Sequential(a))
        assert len(nodes) == 1
        assert nodes[0].op is a
        assert nodes[0].children == []

    def test_parallel_requires_two(self):
        """Parallel(op) raises ValueError."""
        Parallel = _mock_fusion.Parallel
        a = _make_mock_op("a")
        with pytest.raises(ValueError, match="at least 2"):
            Parallel(a)

    def test_parallel_in_middle_errors(self):
        """Sequential(a, Parallel(b, c), d) raises ValueError."""
        _resolve = _mock_fusion._resolve
        Sequential = _mock_fusion.Sequential
        Parallel = _mock_fusion.Parallel

        a, b, c, d = [_make_mock_op(n) for n in "abcd"]
        seq = Sequential(a, Parallel(b, c), d)
        with pytest.raises(ValueError, match="diverge"):
            _resolve(seq)

    def test_empty_sequential(self):
        """Sequential() raises ValueError."""
        Sequential = _mock_fusion.Sequential
        with pytest.raises(ValueError, match="at least 1"):
            Sequential()

    def test_parallel_add(self):
        """Parallel(a, b).add(c) adds a third branch."""
        _resolve = _mock_fusion._resolve
        Parallel = _mock_fusion.Parallel

        a, b, c = _make_mock_op("a"), _make_mock_op("b"), _make_mock_op("c")
        p = Parallel(a, b)
        p.add(c)
        nodes = _resolve(p)
        assert len(nodes) == 3
        assert nodes[0].op is a
        assert nodes[1].op is b
        assert nodes[2].op is c

    def test_deep_nested_split(self):
        """Split of a split: Sequential(a, Parallel(Sequential(b, Parallel(c, d)), e))."""
        _resolve = _mock_fusion._resolve
        Sequential = _mock_fusion.Sequential
        Parallel = _mock_fusion.Parallel

        a, b, c, d, e = [_make_mock_op(n) for n in "abcde"]
        seq = Sequential(a, Parallel(Sequential(b, Parallel(c, d)), e))
        nodes = _resolve(seq)

        assert len(nodes) == 1
        root = nodes[0]
        assert root.op is a
        assert len(root.children) == 2

        # First branch: b→[c, d]
        branch0 = root.children[0]
        assert branch0.op is b
        assert len(branch0.children) == 2
        assert branch0.children[0].op is c
        assert branch0.children[1].op is d

        # Second branch: e (leaf)
        branch1 = root.children[1]
        assert branch1.op is e
        assert branch1.children == []

    def test_merge_build_results(self):
        """_merge_build_results deduplicates inputs, concatenates outputs, unions semaphores."""
        _merge = _mock_graph._merge_build_results
        _BuildResult = _mock_common._BuildResult

        shared_input = MagicMock(name="shared_in")
        in_a = MagicMock(name="in_a")
        in_b = MagicMock(name="in_b")
        out_a = MagicMock(name="out_a")
        out_b = MagicMock(name="out_b")
        sem_a = MagicMock(name="sem_a")
        sem_b = MagicMock(name="sem_b")

        r_a = _BuildResult(
            descriptor=MagicMock(name="desc_a"),
            input_tensors=[shared_input, in_a],
            output_tensors=[out_a],
            semaphores=(sem_a,),
        )
        r_b = _BuildResult(
            descriptor=MagicMock(name="desc_b"),
            input_tensors=[shared_input, in_b],
            output_tensors=[out_b],
            semaphores=(sem_b,),
        )

        merged = _merge([r_a, r_b])

        # shared_input deduped — only 3 unique inputs
        assert len(merged.input_tensors) == 3
        assert merged.input_tensors[0] is shared_input
        assert merged.input_tensors[1] is in_a
        assert merged.input_tensors[2] is in_b

        # Outputs concatenated
        assert len(merged.output_tensors) == 2
        assert merged.output_tensors[0] is out_a
        assert merged.output_tensors[1] is out_b

        # Semaphores unioned
        assert len(merged.semaphores) == 2
        assert merged.semaphores[0] is sem_a
        assert merged.semaphores[1] is sem_b

    def test_merge_build_results_single(self):
        """_merge_build_results with single result returns it directly."""
        _merge = _mock_graph._merge_build_results
        _BuildResult = _mock_common._BuildResult
        r = _BuildResult(
            descriptor=MagicMock(name="desc"),
            input_tensors=[MagicMock(name="in")],
            output_tensors=[MagicMock(name="out")],
        )
        assert _merge([r]) is r

    def test_resolve_raw_op_descriptor(self):
        """_resolve(OpDescriptor) returns [OpNode(op)]."""
        _resolve = _mock_fusion._resolve
        OpNode = _mock_graph.OpNode

        op = _make_mock_op("raw")
        nodes = _resolve(op)
        assert len(nodes) == 1
        assert type(nodes[0]).__name__ == "OpNode"
        assert nodes[0].op is op

    def test_parallel_add_chaining(self):
        """p.add(c).add(d) returns self for chaining."""
        Parallel = _mock_fusion.Parallel
        a, b, c, d = [_make_mock_op(n) for n in "abcd"]
        p = Parallel(a, b)
        result = p.add(c).add(d)
        assert result is p
        assert len(p._items) == 4

    def test_unsupported_type_raises(self):
        """_resolve with unsupported type raises TypeError."""
        _resolve = _mock_fusion._resolve
        with pytest.raises(TypeError, match="Unsupported"):
            _resolve(42)

    def test_resolve_rejects_fused_op(self):
        """_resolve(FusedOp) raises TypeError — prevents nesting."""
        _resolve = _mock_fusion._resolve
        FusedOp = _mock_fusion.FusedOp
        OpDescriptor = _mock_fusion.OpDescriptor
        fused = FusedOp(
            op=OpDescriptor(
                descriptor=MagicMock(name="desc"),
                input_tensors=[],
                output_tensors=[],
            ),
        )
        with pytest.raises(TypeError, match="FusedOp cannot be nested"):
            _resolve(fused)

    def test_multi_level_sequential_flattening(self):
        """Sequential(Sequential(a, Sequential(b, c)), d) deeply flattens."""
        _resolve = _mock_fusion._resolve
        Sequential = _mock_fusion.Sequential

        a, b, c, d = [_make_mock_op(n) for n in "abcd"]
        inner_inner = Sequential(b, c)
        inner = Sequential(a, inner_inner)
        outer = Sequential(inner, d)
        nodes = _resolve(outer)

        assert len(nodes) == 1
        root = nodes[0]
        # Should be a→b→c→d
        assert root.op is a
        assert root.children[0].op is b
        assert root.children[0].children[0].op is c
        assert root.children[0].children[0].children[0].op is d


class TestPhaseNameGeneration:
    """Tests for phase name propagation and tracy output in generated source."""

    def test_phase_namespace_comment_with_name(self):
        """Phase namespace comment includes op name when provided."""
        gen = _mock_codegen._generate_phase_namespace
        source = "void kernel_main() { int x = 1; }"
        lines = gen(0, "", source, [], 0, phase_name="rms_norm")
        # Banner: lines[0] = separator, lines[1] = label, lines[2] = separator
        assert "====" in lines[0]
        assert "Phase 0: rms_norm" in lines[1]
        assert "====" in lines[2]

    def test_phase_namespace_comment_without_name(self):
        """Phase namespace comment omits op name when empty."""
        gen = _mock_codegen._generate_phase_namespace
        source = "void kernel_main() { int x = 1; }"
        lines = gen(0, "", source, [], 0, phase_name="")
        assert "Phase 0" in lines[1]
        assert ":" not in lines[1].split("Phase 0")[1]

    def test_device_zone_scoped_emitted_with_name(self):
        """DeviceZoneScopedN is emitted inside run() when phase_name is set."""
        gen = _mock_codegen._generate_phase_namespace
        source = "void kernel_main() { int x = 1; }"
        lines = gen(0, "", source, [], 0, phase_name="layer_norm")
        joined = "\n".join(lines)
        assert 'DeviceZoneScopedN("layer_norm");' in joined
        # Should be inside run()
        run_idx = next(i for i, l in enumerate(lines) if "void run()" in l)
        zone_idx = next(i for i, l in enumerate(lines) if "DeviceZoneScopedN" in l)
        assert zone_idx == run_idx + 1

    def test_device_zone_scoped_not_emitted_without_name(self):
        """DeviceZoneScopedN is NOT emitted when phase_name is empty."""
        gen = _mock_codegen._generate_phase_namespace
        source = "void kernel_main() { int x = 1; }"
        lines = gen(0, "", source, [], 0, phase_name="")
        joined = "\n".join(lines)
        assert "DeviceZoneScopedN" not in joined

    def test_phase_name_default_empty(self):
        """Default phase_name is empty (backward compat)."""
        gen = _mock_codegen._generate_phase_namespace
        source = "void kernel_main() { int x = 1; }"
        lines = gen(0, "", source, [], 0)
        joined = "\n".join(lines)
        assert "DeviceZoneScopedN" not in joined

    def test_op_descriptor_name_field(self):
        """OpDescriptor has a name field with empty default."""
        OpDescriptor = _mock_fusion.OpDescriptor
        # 3-arg form still works (backward compat)
        op = OpDescriptor("desc", ["in"], ["out"])
        assert op.name == ""
        # 4-arg form sets name
        op2 = OpDescriptor("desc", ["in"], ["out"], "matmul")
        assert op2.name == "matmul"


class TestGlobalCircularBufferPassThrough:
    """Tests for GlobalCircularBuffer support in the fusion framework.

    A GlobalCB-backed CBDescriptor has two sets of format descriptors:
    - format_descriptors (local): regular CB slots, pool-allocated normally
    - remote_format_descriptors (remote): GlobalCB slots, reserved only
    """

    def _make_regular_cb(self, buffer_index=0, data_format="F16", page_size=1024, total_size=2048):
        """Create a mock regular (non-GlobalCB) CBDescriptor."""
        fmt = MagicMock(buffer_index=buffer_index, data_format=data_format, page_size=page_size)
        cb = MagicMock(
            total_size=total_size,
            core_ranges="mock_ranges",
            format_descriptors=[fmt],
            remote_format_descriptors=[],
        )
        cb.has_global_circular_buffer.return_value = False
        cb.has_buffer.return_value = False
        return cb

    def _make_global_cb(
        self,
        local_index=1,
        remote_index=31,
        data_format="F16",
        page_size=1024,
        total_size=4096,
    ):
        """Create a mock GlobalCB-backed CBDescriptor with both local and remote indices."""
        local_fmt = MagicMock(buffer_index=local_index, data_format=data_format, page_size=page_size)
        remote_fmt = MagicMock(buffer_index=remote_index, data_format=data_format, page_size=page_size)
        cb = MagicMock(
            total_size=total_size,
            core_ranges="mock_ranges",
            format_descriptors=[local_fmt],
            remote_format_descriptors=[remote_fmt],
        )
        cb.has_global_circular_buffer.return_value = True
        cb.has_buffer.return_value = False
        return cb

    def _make_remote_only_global_cb(self, remote_index=31, data_format="F16", page_size=1024, total_size=4096):
        """Create a mock GlobalCB-backed CBDescriptor with ONLY remote descriptors (no local)."""
        remote_fmt = MagicMock(buffer_index=remote_index, data_format=data_format, page_size=page_size)
        cb = MagicMock(
            total_size=total_size,
            core_ranges="mock_ranges",
            format_descriptors=[],
            remote_format_descriptors=[remote_fmt],
        )
        cb.has_global_circular_buffer.return_value = True
        cb.has_buffer.return_value = False
        return cb

    def test_extract_cb_info_skips_global_cb_remote(self):
        """extract_cb_info should extract local format descriptors but NOT remote ones."""
        extract_cb_info = _mock_cb_allocator.extract_cb_info

        regular_cb = self._make_regular_cb(buffer_index=0)
        global_cb = self._make_global_cb(local_index=1, remote_index=31)

        prog_desc = MagicMock(cbs=[regular_cb, global_cb])
        result = extract_cb_info(prog_desc)

        # Regular CB index 0 extracted
        assert 0 in result
        # GlobalCB local index 1 also extracted
        assert 1 in result
        # Remote index 31 NOT extracted (not in format_descriptors)
        assert 31 not in result

    def test_extract_cb_info_remote_only_global_cb_produces_no_cbinfo(self):
        """A GlobalCB with only remote_format_descriptors should produce no CBInfo."""
        extract_cb_info = _mock_cb_allocator.extract_cb_info

        remote_only_cb = self._make_remote_only_global_cb(remote_index=31)
        prog_desc = MagicMock(cbs=[remote_only_cb])

        result = extract_cb_info(prog_desc)
        assert len(result) == 0

    def test_extract_remote_cb_indices(self):
        """_extract_remote_cb_indices should return remote buffer indices from GlobalCB-backed CBs."""
        fn = _mock_cb_allocator._extract_remote_cb_indices

        regular_cb = self._make_regular_cb(buffer_index=0)
        global_cb = self._make_global_cb(local_index=1, remote_index=31)

        prog_desc = MagicMock(cbs=[regular_cb, global_cb])
        result = fn(prog_desc)

        assert result == {31}

    def test_extract_remote_cb_indices_no_global_cbs(self):
        """_extract_remote_cb_indices returns empty set when no GlobalCBs present."""
        fn = _mock_cb_allocator._extract_remote_cb_indices

        regular_cb = self._make_regular_cb(buffer_index=0)
        prog_desc = MagicMock(cbs=[regular_cb])

        result = fn(prog_desc)
        assert result == set()

    def test_reserve_index_prevents_collision(self):
        """reserve_index should prevent pool from allocating at reserved slot."""
        CBPoolAllocator = _mock_cb_allocator.CBPoolAllocator
        CBInfo = _mock_cb_allocator.CBInfo

        pool = CBPoolAllocator(max_slots=32)

        # Reserve slot 31 (GlobalCB remote index)
        pool.reserve_index(31)

        # Allocate a phase with a CB that naturally wants index 31
        cb_info = {31: CBInfo(31, 2048, "F16", 1024, None, "Default")}
        pool.allocate_phase(0, cb_info, set())

        # CB 31 should be remapped to a different slot (31 is reserved)
        remap = pool.get_remap(0)
        assert remap[31] != 31, "Reserved index should not be reused by pool allocation"

    def test_reserve_index_not_in_remap(self):
        """Reserved indices should NOT appear in any phase's remap (no CB reset)."""
        CBPoolAllocator = _mock_cb_allocator.CBPoolAllocator
        CBInfo = _mock_cb_allocator.CBInfo

        pool = CBPoolAllocator(max_slots=32)
        pool.reserve_index(31)

        # Phase 0: regular CBs only
        cb_info_0 = {0: CBInfo(0, 2048, "F16", 1024, None, "Default")}
        pool.allocate_phase(0, cb_info_0, set())

        # Phase 1: also regular CBs only
        cb_info_1 = {0: CBInfo(0, 2048, "F16", 1024, None, "Default")}
        pool.allocate_phase(1, cb_info_1, set())

        # Index 31 should not appear in any remap
        for phase_idx in range(2):
            remap = pool.get_remap(phase_idx)
            assert 31 not in remap.values(), f"Reserved index 31 found in phase {phase_idx} remap values"

    def test_per_phase_cb_slots_excludes_reserved(self):
        """per_phase_cb_slots (derived from remaps) should exclude reserved GlobalCB indices."""
        CBPoolAllocator = _mock_cb_allocator.CBPoolAllocator
        CBInfo = _mock_cb_allocator.CBInfo

        pool = CBPoolAllocator(max_slots=32)
        pool.reserve_index(31)

        cb_info_0 = {0: CBInfo(0, 2048, "F16", 1024, None, "Default")}
        pool.allocate_phase(0, cb_info_0, set())

        # Compute per_phase_cb_slots the same way builder.py does
        per_phase_cb_slots = []
        for i in range(1):
            remap = pool.get_remap(i)
            slots = sorted(set(remap.values()))
            per_phase_cb_slots.append(slots)

        # Slot 31 should NOT be in any phase's CB slot list (no reset for it)
        for slots in per_phase_cb_slots:
            assert 31 not in slots, "Reserved GlobalCB index should not be in per_phase_cb_slots"

    def test_global_cb_local_index_pool_allocated(self):
        """GlobalCB's local format_descriptor index should be pool-allocated normally."""
        CBPoolAllocator = _mock_cb_allocator.CBPoolAllocator
        CBInfo = _mock_cb_allocator.CBInfo

        pool = CBPoolAllocator(max_slots=32)
        pool.reserve_index(31)  # Reserve remote index

        # Phase 0: GlobalCB local index 1 + regular CB index 0
        cb_info_0 = {
            0: CBInfo(0, 2048, "F16", 1024, None, "Default"),
            1: CBInfo(1, 4096, "F16", 1024, None, "Default"),  # GlobalCB local part
        }
        pool.allocate_phase(0, cb_info_0, set())

        # Phase 1: regular CB index 1 with same config
        cb_info_1 = {
            1: CBInfo(1, 2048, "F16", 1024, None, "Default"),
        }
        pool.allocate_phase(1, cb_info_1, set())

        # Local index 1 should be pool-allocated and reused across phases
        slot_p0 = pool.get_remap(0)[1]
        slot_p1 = pool.get_remap(1)[1]
        assert slot_p0 == slot_p1, "GlobalCB local index should share slot with compatible CB"

    def test_build_merged_includes_remote_only_global_cb(self):
        """build_merged_cb_descriptors should include remote-only GlobalCB descriptors."""
        CBPoolAllocator = _mock_cb_allocator.CBPoolAllocator
        CBInfo = _mock_cb_allocator.CBInfo
        PhaseInfo = _mock_cb_allocator.PhaseInfo
        OpDescriptor = _mock_cb_allocator.OpDescriptor

        pool = CBPoolAllocator(max_slots=32)
        pool.reserve_index(31)

        # Phase with a regular CB and a remote-only GlobalCB
        regular_cb = self._make_regular_cb(buffer_index=0)
        remote_only_cb = self._make_remote_only_global_cb(remote_index=31)

        mock_descriptor = MagicMock()
        mock_descriptor.cbs = [regular_cb, remote_only_cb]

        mock_op_desc = MagicMock()
        mock_op_desc.descriptor = mock_descriptor

        # Extract CB info (only regular CB gets extracted).
        # source_fmt/source_cb must be set (build_merged_cb_descriptors uses them).
        fmt0 = regular_cb.format_descriptors[0]
        cb_info = {0: CBInfo(0, 2048, "F16", 1024, None, False, source_fmt=fmt0, source_cb=regular_cb)}
        phase = PhaseInfo(phase_idx=0, op_descriptor=mock_op_desc, cb_info=cb_info)

        pool.allocate_phase(0, cb_info, set())

        merged = pool.build_merged_cb_descriptors([phase])

        # Both the pool-allocated CB and the remote-only GlobalCB should be in merged
        assert len(merged) == 2
        # The remote-only GlobalCB (pass-through) must appear by identity
        assert remote_only_cb in merged

    def test_build_merged_includes_local_plus_remote_global_cb(self):
        """A GlobalCB with both local+remote descriptors should appear once in merged."""
        CBPoolAllocator = _mock_cb_allocator.CBPoolAllocator
        CBInfo = _mock_cb_allocator.CBInfo
        PhaseInfo = _mock_cb_allocator.PhaseInfo

        pool = CBPoolAllocator(max_slots=32)
        pool.reserve_index(31)

        # GlobalCB with local index 1 and remote index 31
        global_cb = self._make_global_cb(local_index=1, remote_index=31)
        regular_cb = self._make_regular_cb(buffer_index=0)

        mock_descriptor = MagicMock()
        mock_descriptor.cbs = [regular_cb, global_cb]

        mock_op_desc = MagicMock()
        mock_op_desc.descriptor = mock_descriptor

        fmt0 = regular_cb.format_descriptors[0]
        fmt1 = global_cb.format_descriptors[0]
        cb_info = {
            0: CBInfo(0, 2048, "F16", 1024, None, False, source_fmt=fmt0, source_cb=regular_cb),
            1: CBInfo(1, 4096, "F16", 1024, None, False, source_fmt=fmt1, source_cb=global_cb),  # from local fmt
        }
        phase = PhaseInfo(phase_idx=0, op_descriptor=mock_op_desc, cb_info=cb_info)
        pool.allocate_phase(0, cb_info, set())

        merged = pool.build_merged_cb_descriptors([phase])

        # Pool CBs + GlobalCB pass-through (GlobalCB local is pool-allocated,
        # but GlobalCB also appears via pass-through if has_global_circular_buffer)
        assert len(merged) >= 2
        # GlobalCB descriptor (pass-through) appears by identity
        assert global_cb in merged


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
