# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone tests for Sequential Kernel Chaining Infrastructure.

Tests the core fusion infrastructure logic without requiring a device.
"""

import pytest
from types import SimpleNamespace


import models.experimental.ops.descriptors.fusion.common as _common
import models.experimental.ops.descriptors.fusion.cb_allocator as _cb_alloc
import models.experimental.ops.descriptors.fusion.codegen as _codegen
import models.experimental.ops.descriptors.fusion.fusion as _fusion
import models.experimental.ops.descriptors.fusion.graph as _graph
import models.experimental.ops.descriptors.fusion.codegen.barrier as _barrier_mod
import models.experimental.ops.descriptors.fusion.codegen.source_gen as _source_gen_mod


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_PLACEHOLDER = object()  # don't-care sentinel for fields that are never read


def _ns(**kw):
    """Shorthand for SimpleNamespace."""
    return SimpleNamespace(**kw)


def _make_core_ranges(coords=None):
    """Create a mock CoreRangeSet. Default: single core at (0,0)."""
    if coords is None:
        coords = [((0, 0), (0, 0))]
    ranges = [_ns(start=_ns(x=sx, y=sy), end=_ns(x=ex, y=ey)) for (sx, sy), (ex, ey) in coords]
    return _ns(ranges=lambda: ranges)


def _make_core_ranges_multi(coords):
    """Create a mock CoreRangeSet from individual (x,y) coords."""
    return _make_core_ranges([((x, y), (x, y)) for x, y in coords])


class _RTArgsCol:
    """Column in a RuntimeArgsView: col[y] -> args list."""

    def __init__(self, fn):
        self._fn = fn

    def __getitem__(self, y):
        return self._fn(y)


class _RTArgsView:
    """Minimal stand-in for RuntimeArgsView with view[x][y] indexing."""

    def __init__(self, data):
        self._data = data  # either list-of-lists or dict-of-(x,y)->list

    def __len__(self):
        return len(self._data) if isinstance(self._data, list) else 0

    def __getitem__(self, x):
        if isinstance(self._data, dict):
            return _RTArgsCol(lambda y, _x=x: self._data[(_x, y)])
        # Legacy: single-core mode — ignore x, return first core's args
        return _RTArgsCol(lambda y: self._data[0])


def _make_rt_args_view(args_per_core):
    return _RTArgsView(args_per_core)


def _make_rt_args_view_multi(args_by_coord):
    return _RTArgsView(args_by_coord)


def _make_op_with_cores(core_range_set):
    """Create a mock OpDescriptor whose kernels use the given core ranges."""
    return _ns(descriptor=_ns(kernels=[_ns(core_ranges=core_range_set)]))


def _make_mock_op(name="op"):
    """Create a mock OpDescriptor NamedTuple for high-level API tests."""
    return _fusion.OpDescriptor(
        descriptor=_ns(__name__=f"{name}_desc"),
        input_tensors=[_ns(__name__=f"{name}_in")],
        output_tensors=[_ns(__name__=f"{name}_out")],
    )


def _make_barrier_spec(num_segments=1, has_reset_done=True):
    """Create a minimal MultiBarrierSpec for source generation tests."""
    segs = [
        _common.BarrierSegment(
            config=_common.BarrierConfig(num_release_cores=1, num_arrive_cores=1, core0_phys_x=1, core0_phys_y=1),
            arrive_addr=3000 + i * 1000,
            release_addr=4000 + i * 1000,
        )
        for i in range(num_segments)
    ]
    kwargs = dict(
        segments=segs,
        compute_done_addr=1000,
        writer_done_addr=2000,
        transition_map={0: (0, 0)},
    )
    if has_reset_done:
        kwargs["reset_done_addr"] = 5000
    return _common.MultiBarrierSpec(**kwargs)


def _make_two_phase_kernel_setup():
    """Create a 2-phase kernel setup for fused source generation tests."""
    core_ranges = _make_core_ranges()
    source_code_type = _codegen.ttnn.KernelDescriptor.SourceType.SOURCE_CODE
    kernel = _ns(
        kernel_source="void kernel_main() { /* reader */ }",
        source_type=source_code_type,
        defines=[],
        core_ranges=core_ranges,
        named_compile_time_args=[("cb_in", 0)],
    )
    phases = [
        _cb_alloc.PhaseInfo(
            i, _ns(name=f"phase_{i}"), {0: _cb_alloc.CBInfo(0, 2048, "F16", 2048, core_ranges, False, "Default")}
        )
        for i in range(2)
    ]
    return phases, [{"reader": kernel}, {"reader": kernel}]


def _make_cb_mock(
    buffer_index,
    data_format="F16",
    page_size=1024,
    total_size=2048,
    has_global=False,
    has_buffer=False,
    remote_format_descriptors=None,
):
    """Create a CB descriptor stand-in."""
    fmt = _ns(
        buffer_index=buffer_index,
        data_format=data_format,
        page_size=page_size,
        data_format_as_uint8=hash(data_format) & 0xFF,
    )
    return _ns(
        total_size=total_size,
        core_ranges="mock",
        address_offset=0,
        format_descriptors=[fmt],
        remote_format_descriptors=remote_format_descriptors or [],
        has_global_circular_buffer=lambda: has_global,
        has_buffer=lambda: has_buffer,
    )


# Monkeypatch _get_node_core_range for graph topology tests.
_graph._get_node_core_range = lambda node: node.op.descriptor.kernels[0].core_ranges


# ---------------------------------------------------------------------------
# CB Allocator: data classes, extraction, pool allocation
# ---------------------------------------------------------------------------


class TestCBDataClasses:
    """Tests for CBInfo, PhaseInfo, CBPoolKey data classes."""

    def test_cb_info_creation(self):
        cb = _cb_alloc.CBInfo(
            original_index=5, total_size=4096, data_format="Float16_b", page_size=2048, core_ranges="mock"
        )
        assert (cb.original_index, cb.total_size, cb.page_size) == (5, 4096, 2048)

    def test_phase_info_default_empty_cb_info(self):
        phase = _cb_alloc.PhaseInfo(phase_idx=1, op_descriptor=_PLACEHOLDER)
        assert phase.cb_info == {}

    def test_pool_key_equality_and_inequality(self):
        CI = _cb_alloc.CBInfo
        a = CI(0, 4096, 7, 2048, "r", has_buffer=False, unpack_to_dest_mode="Default")
        b = CI(5, 8192, 7, 2048, "r", has_buffer=False, unpack_to_dest_mode="Default")
        c = CI(0, 4096, 7, 4096, "r", has_buffer=False, unpack_to_dest_mode="Default")
        assert a.pool_key == b.pool_key
        assert a.pool_key != c.pool_key


class TestExtractCBInfo:
    """Tests for extract_cb_info and extract_cb_names_from_kernel."""

    def test_extract_single_and_multiple_cbs(self):
        cb1 = _make_cb_mock(0, "F16", 1024, 2048)
        cb2 = _make_cb_mock(16, "F32", 2048, 4096)
        result = _cb_alloc.extract_cb_info(_ns(cbs=[cb1, cb2]))
        assert len(result) == 2
        assert result[0].page_size == 1024
        assert result[16].page_size == 2048

    def test_extract_cb_names_filters_cb_prefix(self):
        kernel = _ns(named_compile_time_args=[("cb_in", 0), ("cb_out", 16), ("other", 42)])
        result = _cb_alloc.extract_cb_names_from_kernel(kernel)
        assert result == {"cb_in": 0, "cb_out": 16}


class TestCBArgNaming:
    """Tests for _is_cb_named_arg."""

    @pytest.mark.parametrize(
        "name,val,expected",
        [
            ("cb_in", 0, True),
            ("cb_out", 16, True),
            ("cb_gamma", 31, True),
            ("blk", 4, False),
            ("num_tiles", 16, False),
            ("cb_debug_flag", 100, False),
            ("cb_large", 32, False),
            ("cb_str", "not_an_int", False),
        ],
    )
    def test_is_cb_named_arg(self, name, val, expected):
        assert _cb_alloc._is_cb_named_arg(name, val) is expected


class TestCBPoolAllocator:
    """Tests for CBPoolAllocator pool-based CB slot allocation."""

    def _alloc(self, pool, phase_idx, cb_infos, phantoms=None):
        pool.allocate_phase(phase_idx, cb_infos, phantoms or set())

    def test_same_config_reuses_slot(self):
        pool = _cb_alloc.CBPoolAllocator()
        CI = _cb_alloc.CBInfo
        self._alloc(pool, 0, {0: CI(0, 2048, "F16", 1024, None, "Default")})
        self._alloc(pool, 1, {0: CI(0, 4096, "F16", 1024, None, "Default")})
        assert pool.get_remap(0)[0] == pool.get_remap(1)[0]
        assert pool._slots[pool.get_remap(0)[0]].total_size == 4096

    @pytest.mark.parametrize(
        "field,val0,val1",
        [
            ("page_size", 1024, 2048),
            ("data_format", "F16", "F32"),
            ("unpack_to_dest_mode", "Default", "UnpackToDestFp32"),
        ],
    )
    def test_different_config_gets_separate_slot(self, field, val0, val1):
        pool = _cb_alloc.CBPoolAllocator()
        CI = _cb_alloc.CBInfo
        kwargs0 = dict(
            original_index=0,
            total_size=2048,
            data_format="F16",
            page_size=1024,
            core_ranges=None,
            unpack_to_dest_mode="Default",
        )
        kwargs1 = dict(kwargs0)
        kwargs0[field] = val0
        kwargs1[field] = val1
        self._alloc(pool, 0, {0: CI(**kwargs0)})
        self._alloc(pool, 1, {0: CI(**kwargs1)})
        assert pool.get_remap(0)[0] != pool.get_remap(1)[0]

    def test_overflow_raises_on_projection(self):
        pool = _cb_alloc.CBPoolAllocator()
        CI = _cb_alloc.CBInfo
        for i in range(33):
            self._alloc(pool, i, {0: CI(0, 1024, f"F{i}", 1024, None, "Default")})
        assert len(pool._slots) == 33
        with pytest.raises(ValueError, match="CB pool overflow"):
            pool.project_to_group(list(range(33)), set())

    def test_phantom_cb_reservation(self):
        pool = _cb_alloc.CBPoolAllocator()
        CI = _cb_alloc.CBInfo
        self._alloc(pool, 0, {0: CI(0, 2048, "F16", 1024, None, "Default")}, phantoms={18})
        self._alloc(
            pool, 1, {0: CI(0, 2048, "F16", 1024, None, "Default"), 5: CI(5, 2048, "F32", 2048, None, "Default")}
        )
        assert pool.get_remap(0)[18] == 18
        assert pool.get_remap(1)[5] != 18

    def test_multiple_cbs_per_phase_get_unique_slots(self):
        pool = _cb_alloc.CBPoolAllocator()
        CI = _cb_alloc.CBInfo
        cb_info = {
            0: CI(0, 2048, "F16", 1024, None, "Default"),
            5: CI(5, 4096, "F32", 2048, None, "Default"),
            16: CI(16, 1024, "F16", 1024, None, "Default"),
        }
        self._alloc(pool, 0, cb_info)
        slots = {pool.get_remap(0)[i] for i in [0, 5, 16]}
        assert len(slots) == 3

    def test_cross_phase_sharing_different_index(self):
        pool = _cb_alloc.CBPoolAllocator()
        CI = _cb_alloc.CBInfo
        self._alloc(pool, 0, {5: CI(5, 2048, "F16", 1024, None, "Default")})
        self._alloc(pool, 1, {0: CI(0, 2048, "F16", 1024, None, "Default")})
        assert pool.get_remap(0)[5] == pool.get_remap(1)[0]

    def test_project_to_group_basic(self):
        pool = _cb_alloc.CBPoolAllocator()
        CI = _cb_alloc.CBInfo
        self._alloc(pool, 0, {0: CI(0, 2048, "F16", 1024, None, "Default")})
        self._alloc(pool, 1, {0: CI(0, 2048, "F32", 2048, None, "Default")})
        self._alloc(pool, 2, {0: CI(0, 2048, "F16", 1024, None, "Default")})
        proj = pool.project_to_group([0, 2], set())
        assert len(proj.phase_remaps) == 2
        assert proj.phase_remaps[0][0] == proj.phase_remaps[1][0]

    def test_project_shared_ops_consistent(self):
        pool = _cb_alloc.CBPoolAllocator()
        CI = _cb_alloc.CBInfo
        self._alloc(pool, 0, {0: CI(0, 2048, "F16", 1024, None, "Default")})
        self._alloc(pool, 1, {0: CI(0, 2048, "F32", 2048, None, "Default")})
        self._alloc(pool, 2, {0: CI(0, 2048, "F8", 512, None, "Default")})
        proj_a = pool.project_to_group([0, 1], set())
        proj_b = pool.project_to_group([0, 2], set())
        assert proj_a.phase_remaps[0] == proj_b.phase_remaps[0]

    def test_global_pool_unlimited_slots(self):
        pool = _cb_alloc.CBPoolAllocator()
        CI = _cb_alloc.CBInfo
        for i in range(35):
            self._alloc(pool, i, {0: CI(0, 1024, f"F{i}", 1024, None, "Default")})
        assert len(pool._slots) == 35
        pool.project_to_group([0, 1, 2, 3, 4], set())  # Should succeed
        with pytest.raises(ValueError, match="CB pool overflow"):
            pool.project_to_group(list(range(35)), set())


class TestCBStateSaveRestore:
    """Tests for _save_cb_state / _restore_cb_state / _verify_cb_restore."""

    def test_round_trip(self):
        fmt = _ns(buffer_index=5)
        cb = _ns(total_size=4096, core_ranges="mock", format_descriptors=[fmt])
        saved = _cb_alloc._save_cb_state([_ns(cbs=[cb])])
        cb.total_size, fmt.buffer_index = 8192, 10
        _cb_alloc._restore_cb_state(saved)
        assert (cb.total_size, fmt.buffer_index) == (4096, 5)
        _cb_alloc._verify_cb_restore(saved)

    def test_deduplicates_shared_cbs(self):
        fmt = _ns(buffer_index=3)
        cb = _ns(total_size=2048, core_ranges="mock", format_descriptors=[fmt])
        saved = _cb_alloc._save_cb_state([_ns(cbs=[cb]), _ns(cbs=[cb])])
        assert len(saved) == 1

    def test_verify_fails_on_mismatch(self):
        cb = _ns(total_size=8192, core_ranges="mock", format_descriptors=[_ns(buffer_index=5)])
        fmt = cb.format_descriptors[0]
        saved = [{"cb": cb, "total_size": 4096, "core_ranges": "mock", "fmt": [(fmt, 5)]}]
        with pytest.raises(RuntimeError, match="total_size"):
            _cb_alloc._verify_cb_restore(saved)

        cb.total_size = 4096
        fmt.buffer_index = 10
        with pytest.raises(RuntimeError, match="buffer_index"):
            _cb_alloc._verify_cb_restore(saved)


class TestAliasGroups:
    """Tests for alias group permutation and cache."""

    def test_permuted_order_reuses_slots(self):
        pool = _cb_alloc.CBPoolAllocator()
        CI = _cb_alloc.CBInfo

        def _info(idx, fmt, ps, fi):
            return CI(idx, 2048, fmt, ps, "r", alias_group=0, source_fmt=_ns(buffer_index=fi), source_cb=_PLACEHOLDER)

        pool.allocate_phase(0, {4: _info(4, "F16", 1024, 0), 5: _info(5, "F32", 2048, 1)}, set())
        pool.allocate_phase(1, {4: _info(4, "F32", 2048, 2), 5: _info(5, "F16", 1024, 3)}, set())
        r0, r1 = pool.get_remap(0), pool.get_remap(1)
        assert {r0[4], r0[5]} == {r1[4], r1[5]}
        assert r1[4] == r0[5] and r1[5] == r0[4]

    def test_different_keys_no_reuse(self):
        pool = _cb_alloc.CBPoolAllocator()
        CI = _cb_alloc.CBInfo

        def _info(idx, fmt, ps, fi):
            return CI(idx, 2048, fmt, ps, "r", alias_group=0, source_fmt=_ns(buffer_index=fi), source_cb=_PLACEHOLDER)

        pool.allocate_phase(0, {4: _info(4, "F16", 1024, 0), 5: _info(5, "F32", 2048, 1)}, set())
        pool.allocate_phase(1, {4: _info(4, "INT8", 512, 2), 5: _info(5, "BFP4", 256, 3)}, set())
        assert set(pool.get_remap(0).values()).isdisjoint(set(pool.get_remap(1).values()))

    def test_unique_alias_groups_cache(self):
        pool = _cb_alloc.CBPoolAllocator()
        CI = _cb_alloc.CBInfo

        def _phase():
            return {
                0: CI(0, 2048, "F16", 1024, "r", alias_group=0, source_fmt=_ns(buffer_index=0), source_cb=_PLACEHOLDER),
                1: CI(1, 4096, "F32", 2048, "r", alias_group=0, source_fmt=_ns(buffer_index=1), source_cb=_PLACEHOLDER),
            }

        pool.allocate_phase(0, _phase(), set())
        assert len(pool._unique_alias_groups) == 1
        pool.allocate_phase(1, _phase(), set())
        assert len(pool._unique_alias_groups) == 1  # Reuse doesn't grow


# ---------------------------------------------------------------------------
# Codegen: source transformations, kernel body extraction, includes/defines
# ---------------------------------------------------------------------------


class TestSourceTransformations:
    """Tests for kernel source transformation functions."""

    @pytest.mark.parametrize("phase,expected_prefix", [(0, '"cb_in"'), (1, '"phase_1_cb_in"'), (2, '"phase_2_blk"')])
    def test_prefix_named_args(self, phase, expected_prefix):
        fn = _codegen._prefix_named_args_in_source
        if phase == 2:
            source = 'constexpr uint32_t blk = get_named_compile_time_arg_val("blk");'
        else:
            source = 'constexpr uint32_t cb = get_named_compile_time_arg_val("cb_in");'
        result = fn(source, phase)
        assert f"get_named_compile_time_arg_val({expected_prefix})" in result

    def test_emit_rt_arg_wrapper_and_defines(self):
        joined = "\n".join(_codegen._emit_rt_arg_wrapper(2, 18))
        assert "arg_idx + 18" in joined
        assert "phase_2_get_arg_val" in joined and "phase_2_get_arg_addr" in joined

        assert _codegen._emit_rt_arg_define(1) == [
            "#define get_arg_val phase_1_get_arg_val",
            "#define get_arg_addr phase_1_get_arg_addr",
        ]
        assert _codegen._emit_rt_arg_undef() == ["#undef get_arg_val", "#undef get_arg_addr"]

    def test_offset_compile_time_args(self):
        fn = _codegen._offset_compile_time_args_in_source
        source = "uint32_t a = get_compile_time_arg_val(0);\nuint32_t b = get_compile_time_arg_val(1);"
        assert fn(source, 0, 0) == source  # Phase 0 unchanged
        result = fn(source, 1, 3)
        assert "get_compile_time_arg_val(3)" in result and "get_compile_time_arg_val(4)" in result

    def test_offset_tensor_accessor(self):
        result = _codegen._offset_compile_time_args_in_source(
            "constexpr auto src_args = TensorAccessorArgs<2>();", 1, 5
        )
        assert "TensorAccessorArgs<7>" in result

    def test_transform_phase_source_combines_all(self):
        source = (
            'constexpr uint32_t cb = get_named_compile_time_arg_val("cb_in");\n'
            "uint32_t blk = get_compile_time_arg_val(0);\n"
            "constexpr auto args = TensorAccessorArgs<2>();\n"
            "uint32_t val = get_arg_val<uint32_t>(3);\n"
        )
        result = _codegen._transform_phase_source(source, 1, ct_arg_offset=5)
        assert "phase_1_cb_in" in result
        assert "get_compile_time_arg_val(5)" in result
        assert "TensorAccessorArgs<7>" in result
        assert "get_arg_val<uint32_t>(3)" in result  # NOT rewritten in source


class TestExtractKernelBody:
    """Tests for kernel body extraction via brace matching."""

    @pytest.mark.parametrize(
        "source,expected_in,expected_not_in",
        [
            # Standard
            ("void kernel_main() {\n    int x = 1;\n}", ["int x = 1"], []),
            # ALWI prefix
            ("ALWI void kernel_main() {\n    compute(x);\n}", ["compute(x)"], []),
            # Nested braces
            (
                "void kernel_main() {\n    for (int i=0;i<10;i++) { if (i>5) { do_something(); } }\n}",
                ["do_something", "for"],
                [],
            ),
            # No kernel_main
            ("void other_function() { int x = 1; }", [], []),
            # String literal with braces
            (
                'void kernel_main() {\n    const char* msg = "{ json }";\n    int y = 2;\n}',
                ["{ json }", "int y = 2"],
                [],
            ),
            # Line comment with braces
            ("void kernel_main() {\n    // this has a } brace\n    int x = 1;\n}", ["int x = 1"], []),
            # Block comment with braces
            ("void kernel_main() {\n    /* } } } */\n    int x = 1;\n}", ["int x = 1"], []),
            # Char literal with brace
            ("void kernel_main() {\n    char c = '}';\n    int x = 1;\n}", ["int x = 1"], []),
            # Raw string
            (
                'void kernel_main() {\n    const char* s = R"({ \\"key\\" })";\n    int x = 1;\n}',
                ["int x = 1", 'R"('],
                [],
            ),
        ],
    )
    def test_extract(self, source, expected_in, expected_not_in):
        body = _codegen.extract_kernel_body(source)
        for s in expected_in:
            assert s in body
        for s in expected_not_in:
            assert s not in body
        if not expected_in and not expected_not_in:
            assert body == ""

    def test_raw_string_with_delimiter(self):
        source = (
            'void kernel_main() {\n    const char* s = R"foo(\n        }}} """ {{{\n    )foo";\n    int done = 1;\n}'
        )
        assert "int done = 1" in _codegen.extract_kernel_body(source)


class TestCollectIncludesAndDefines:
    """Tests for include/define collection."""

    def test_collect_unique_includes(self):
        sources = [
            '#include "common.h"\n#include "phase0.h"\nvoid kernel_main() {}',
            '#include "common.h"\n#include "phase1.h"\nvoid kernel_main() {}',
        ]
        includes = _codegen.collect_includes(sources)
        assert len(includes) == 3

    def test_collect_defines_before_main(self):
        defines = _codegen.collect_defines(["#define FOO 1\n#define BAR 2\nvoid kernel_main() {\n#define INSIDE 3\n}"])
        strs = [d.strip() for d in defines]
        assert "#define FOO 1" in strs and "#define INSIDE 3" not in strs


class TestInlineLocalIncludes:
    """Tests for local include inlining."""

    def test_inlines_local_and_relative(self, tmp_path):
        (tmp_path / "utils.h").write_text("#pragma once\nint helper() { return 42; }\n")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "helper.h").write_text("#pragma once\nint helper_val = 42;")

        # Local include
        headers, remaining = _codegen.inline_local_includes(
            '#include "utils.h"\nvoid kernel_main() {}\n', str(tmp_path)
        )
        assert len(headers) == 1 and "int helper()" in headers[0][1] and "#pragma once" not in headers[0][1]

        # Relative path include
        headers, remaining = _codegen.inline_local_includes(
            '#include "subdir/helper.h"\nvoid kernel_main() {}', str(tmp_path)
        )
        assert len(headers) == 1 and "int helper_val = 42" in headers[0][1]

    def test_leaves_system_includes(self):
        headers, remaining = _codegen.inline_local_includes(
            '#include "api/dataflow/dataflow_api.h"\nvoid kernel_main() {}\n', "/some/dir"
        )
        assert headers == [] and '#include "api/dataflow/dataflow_api.h"' in remaining

    def test_no_kernel_dir_unchanged(self):
        source = '#include "utils.h"\nvoid kernel_main() {}\n'
        headers, remaining = _codegen.inline_local_includes(source, None)
        assert headers == [] and remaining == source


# ---------------------------------------------------------------------------
# Codegen: argument merging
# ---------------------------------------------------------------------------


class TestArgMerging:
    """Tests for compile-time, runtime, and named arg merging."""

    def test_merge_named_compile_time_args(self):
        fn = _codegen._merge_named_compile_time_args
        phase_kernels = [
            {"reader": _ns(named_compile_time_args=[("cb_in", 0), ("blk", 4)])},
            {"reader": _ns(named_compile_time_args=[("cb_in", 0)])},
        ]
        result = dict(fn(phase_kernels, "reader"))
        assert result["cb_in"] == 0 and result["blk"] == 4  # Phase 0
        assert result["phase_1_cb_in"] == 0  # Phase 1 prefixed

    def test_named_arg_cb_remapping(self):
        fn = _codegen._merge_named_compile_time_args
        k0 = _ns(named_compile_time_args=[("cb_in0", 0)])
        k1 = _ns(named_compile_time_args=[("cb_in0", 0)])
        result = dict(fn([{"reader": k0}, {"reader": k1}], "reader", phase_remaps=[{0: 0}, {0: 5}]))
        assert result["cb_in0"] == 0 and result["phase_1_cb_in0"] == 5

    def test_barrier_rt_offset_added(self):
        result = dict(
            _codegen._merge_named_compile_time_args(
                [{"reader": _ns(named_compile_time_args=[])}], "reader", barrier_rt_offset=10
            )
        )
        assert result["barrier_rt_offset"] == 10

    def test_merge_compile_time_args(self):
        phase_kernels = [
            {"reader": _ns(compile_time_args=[10, 20, 30])},
            {"reader": None},
            {"reader": _ns(compile_time_args=[40])},
        ]
        merged, offsets = _codegen._merge_compile_time_args(phase_kernels, "reader")
        assert merged == [10, 20, 30, 40] and offsets == {0: 0, 1: 3, 2: 3}

    def test_compute_and_concatenate_runtime_args(self):
        fn = _codegen._compute_and_concatenate_runtime_args
        cr = _make_core_ranges()
        k0 = _ns(runtime_args=_make_rt_args_view([[1, 2, 3, 4, 5]]), core_ranges=cr)
        k1 = _ns(runtime_args=_make_rt_args_view([[10, 20, 30]]), core_ranges=cr)
        offsets, result = fn([{"reader": k0}, {"reader": k1}], "reader")
        assert offsets == {0: 0, 1: 5}
        _, args = result[0]
        assert args == [1, 2, 3, 4, 5, 10, 20, 30]

    def test_validate_fp32_consistency(self):
        fn = _codegen._validate_fp32_consistency

        def _desc(fp32):
            return _ns(descriptor=_ns(kernels=[_ns(config=_ns(fp32_dest_acc_en=fp32))]))

        fn([_desc(True), _desc(True)])  # Should not raise
        with pytest.raises(ValueError, match="fp32_dest_acc_en mismatch"):
            fn([_desc(True), _desc(False)])


class TestRuntimeArgsPadding:
    """Tests that _compute_and_concatenate_runtime_args pads correctly."""

    def _patch(self):
        orig = _codegen.ttnn.CoreCoord
        _codegen.ttnn.CoreCoord = _SimpleCoreCoord
        return orig

    def test_padding_matches_offsets(self):
        orig = self._patch()
        try:
            fn = _codegen._compute_and_concatenate_runtime_args
            cr = _make_core_ranges_multi([(0, 0), (1, 0)])
            k0 = _ns(
                runtime_args=_make_rt_args_view_multi({(0, 0): [1, 2, 3, 4, 5], (1, 0): [10, 20, 30]}), core_ranges=cr
            )
            k1 = _ns(runtime_args=_make_rt_args_view_multi({(0, 0): [100, 200], (1, 0): [300, 400]}), core_ranges=cr)
            offsets, result = fn([{"reader": k0}, {"reader": k1}], "reader")
            assert offsets[1] == 5
            d = {(c.x, c.y): args for c, args in result}
            assert d[(0, 0)] == [1, 2, 3, 4, 5, 100, 200]
            assert d[(1, 0)] == [10, 20, 30, 0, 0, 300, 400]
        finally:
            _codegen.ttnn.CoreCoord = orig

    def test_missing_core_padded(self):
        orig = self._patch()
        try:
            fn = _codegen._compute_and_concatenate_runtime_args
            cr = _make_core_ranges_multi([(0, 0), (1, 0)])
            k0 = _ns(runtime_args=_make_rt_args_view_multi({(0, 0): [1, 2, 3]}), core_ranges=cr)
            _, result = fn([{"reader": k0}], "reader", target_core_range=cr)
            d = {(c.x, c.y): args for c, args in result}
            assert d[(0, 0)] == [1, 2, 3] and d[(1, 0)] == [0, 0, 0]
        finally:
            _codegen.ttnn.CoreCoord = orig


class _SimpleCoreCoord:
    """Simple CoreCoord stand-in with .x/.y attributes."""

    def __init__(self, x, y):
        self.x, self.y = x, y


class TestMustMatchDefines:
    """Tests for MUST_MATCH_DEFINES validation in _collect_phase_defines."""

    def test_matching_accepted(self):
        must_match, _ = _codegen._collect_phase_defines(
            [
                {"compute": _ns(defines=[("REDUCE_OP", "0"), ("REDUCE_DIM", "1")])},
                {"compute": _ns(defines=[("REDUCE_OP", "0"), ("REDUCE_DIM", "1")])},
            ],
            "compute",
        )
        assert {"REDUCE_OP", "REDUCE_DIM"} <= {n for n, _ in must_match}

    def test_mismatched_rejected(self):
        with pytest.raises(ValueError, match="REDUCE_OP.*inconsistent"):
            _codegen._collect_phase_defines(
                [
                    {"compute": _ns(defines=[("REDUCE_OP", "0")])},
                    {"compute": _ns(defines=[("REDUCE_OP", "1")])},
                ],
                "compute",
            )

    def test_present_in_one_phase_only(self):
        must_match, _ = _codegen._collect_phase_defines(
            [
                {"compute": _ns(defines=[("REDUCE_OP", "0"), ("BCAST_LLKOP", "2")])},
                {"compute": _ns(defines=[])},
            ],
            "compute",
        )
        assert {"REDUCE_OP", "BCAST_LLKOP"} <= {n for n, _ in must_match}


# ---------------------------------------------------------------------------
# Graph: OpGraphBuilder topology validation
# ---------------------------------------------------------------------------


class TestOpGraphBuilder:
    """Tests for OpGraphBuilder: creation, single-node, build-twice, topology."""

    def _op(self, coords):
        return _make_op_with_cores(_make_core_ranges(coords))

    def _make_buildable_op(self):
        """Create an op with the fields OpGraphBuilder.build() actually reads."""
        desc = _ns(
            descriptor=_ns(cbs=[], kernels=[_ns(core_ranges=_make_core_ranges())]),
            input_tensors=[],
            output_tensors=[],
        )
        return desc

    def test_single_node_returns_fused_op(self):
        desc = self._make_buildable_op()
        result = _graph.OpGraphBuilder(_graph.OpNode(desc)).build(device=None)
        assert type(result).__name__ == "FusedOp"

    def test_build_twice_raises(self):
        desc = self._make_buildable_op()
        builder = _graph.OpGraphBuilder(_graph.OpNode(desc))
        builder.build(device=None)
        with pytest.raises(ValueError, match="Already built"):
            builder.build(device=None)

    def test_overlapping_siblings_rejected(self):
        root = _graph.OpNode(
            self._op([((0, 0), (3, 0))]),
            children=[
                _graph.OpNode(self._op([((0, 0), (2, 0))])),
                _graph.OpNode(self._op([((1, 0), (3, 0))])),
            ],
        )
        with pytest.raises(ValueError, match="overlapping cores"):
            _graph.OpGraphBuilder(root)._validate_topology()

    def test_valid_disjoint_children(self):
        root = _graph.OpNode(
            self._op([((0, 0), (3, 0))]),
            children=[
                _graph.OpNode(self._op([((0, 0), (1, 0))])),
                _graph.OpNode(self._op([((2, 0), (3, 0))])),
            ],
        )
        _graph.OpGraphBuilder(root)._validate_topology()  # Should not raise

    def test_child_wider_than_parent_accepted(self):
        root = _graph.OpNode(
            self._op([((0, 0), (1, 0))]),
            children=[
                _graph.OpNode(self._op([((0, 0), (2, 0))])),
                _graph.OpNode(self._op([((3, 0), (5, 0))])),
            ],
        )
        _graph.OpGraphBuilder(root)._validate_topology()  # Should not raise

    def test_valid_nested_topology(self):
        leaf_0_1 = _graph.OpNode(self._op([((0, 0), (1, 0))]))
        leaf_2_3 = _graph.OpNode(self._op([((2, 0), (3, 0))]))
        mid = _graph.OpNode(self._op([((0, 0), (3, 0))]), children=[leaf_0_1, leaf_2_3])
        leaf_4_5 = _graph.OpNode(self._op([((4, 0), (5, 0))]))
        root = _graph.OpNode(self._op([((0, 0), (5, 0))]), children=[mid, leaf_4_5])
        _graph.OpGraphBuilder(root)._validate_topology()  # Should not raise

    def test_partial_coverage_accepted(self):
        root = _graph.OpNode(
            self._op([((0, 0), (5, 0))]),
            children=[
                _graph.OpNode(self._op([((0, 0), (1, 0))])),
                _graph.OpNode(self._op([((2, 0), (3, 0))])),
            ],
        )
        _graph.OpGraphBuilder(root)._validate_topology()  # Cores (4-5) unused, OK


class TestNarrowWideTopology:
    """Tests for narrow->wide topology: no-op entries and aligned phase counts."""

    def test_noop_entries_in_walk(self):
        parent = _make_op_with_cores(_make_core_ranges([((0, 0), (1, 0))]))
        child = _make_op_with_cores(_make_core_ranges([((0, 0), (3, 0))]))
        root = _graph.OpNode(parent, children=[_graph.OpNode(child)])
        groups, unique_ops = _graph.OpGraphBuilder(root)._compute_core_groups()
        assert len(unique_ops) == 2
        assert any(phase is _common._NOOP_OP for g in groups for phase in g.phases)

    def test_groups_have_aligned_phase_counts(self):
        parent = _make_op_with_cores(_make_core_ranges([((0, 0), (1, 0))]))
        child = _make_op_with_cores(_make_core_ranges([((0, 0), (3, 0))]))
        root = _graph.OpNode(parent, children=[_graph.OpNode(child)])
        groups, _ = _graph.OpGraphBuilder(root)._compute_core_groups()
        assert len({len(g.phases) for g in groups}) == 1

    def test_disjoint_parent_child_trailing_barrier(self):
        """Parent {0,1} → children {2,3}, {4,5}: completely disjoint.

        Parent cores have 1 phase with a trailing barrier (no fake exit
        phase).  Child cores have 2 phases [_NOOP_OP, real_op].
        """
        parent = _make_op_with_cores(_make_core_ranges([((0, 0), (1, 0))]))
        child_a = _make_op_with_cores(_make_core_ranges([((2, 0), (3, 0))]))
        child_b = _make_op_with_cores(_make_core_ranges([((4, 0), (5, 0))]))
        root = _graph.OpNode(parent, children=[_graph.OpNode(child_a), _graph.OpNode(child_b)])
        groups, _ = _graph.OpGraphBuilder(root)._compute_core_groups()

        # Parent cores (0,1): 1 real phase + trailing barrier
        parent_group = None
        for g in groups:
            coords = _common._core_range_set_to_coords(g.core_range)
            if (0, 0) in coords:
                parent_group = g
                break
        assert parent_group is not None
        assert len(parent_group.phases) == 1
        assert parent_group.phases[0] is not _common._NOOP_OP
        assert parent_group.has_trailing_barrier
        assert len(parent_group.barrier_scopes) == 1  # trailing barrier scope

        # Child cores: 2 phases [_NOOP_OP, real_op], no trailing barrier
        for g in groups:
            coords = _common._core_range_set_to_coords(g.core_range)
            if (2, 0) in coords or (4, 0) in coords:
                assert len(g.phases) == 2
                assert g.phases[0] is _common._NOOP_OP
                assert g.phases[1] is not _common._NOOP_OP
                assert not g.has_trailing_barrier

    def test_disjoint_barrier_scope_includes_parent(self):
        """Disjoint parent-child: barrier scope must include parent cores."""
        parent = _make_op_with_cores(_make_core_ranges([((0, 0), (1, 0))]))
        child = _make_op_with_cores(_make_core_ranges([((2, 0), (3, 0))]))
        root = _graph.OpNode(parent, children=[_graph.OpNode(child)])
        groups, _ = _graph.OpGraphBuilder(root)._compute_core_groups()

        # The barrier scope at transition 0 should include ALL 4 cores
        for g in groups:
            if g.barrier_scopes:
                scope_coords = _common._core_range_set_to_coords(g.barrier_scopes[0])
                assert (0, 0) in scope_coords, "Parent core must be in barrier scope"
                assert (1, 0) in scope_coords, "Parent core must be in barrier scope"
                assert (2, 0) in scope_coords, "Child core must be in barrier scope"
                assert (3, 0) in scope_coords, "Child core must be in barrier scope"

    def test_disjoint_arrive_scope_is_parent_cores(self):
        """Disjoint parent-child: arrive scope should be parent cores only."""
        parent = _make_op_with_cores(_make_core_ranges([((0, 0), (1, 0))]))
        child = _make_op_with_cores(_make_core_ranges([((2, 0), (3, 0))]))
        root = _graph.OpNode(parent, children=[_graph.OpNode(child)])
        groups, _ = _graph.OpGraphBuilder(root)._compute_core_groups()

        # Find a group with a barrier arrive scope and check it
        for g in groups:
            if g.barrier_arrive_scopes:
                arrive_coords = _common._core_range_set_to_coords(g.barrier_arrive_scopes[0])
                # Parent cores should be arrive cores
                assert (0, 0) in arrive_coords
                assert (1, 0) in arrive_coords
                # Child cores should NOT be arrive cores (they had _NOOP_OP)
                assert (2, 0) not in arrive_coords
                assert (3, 0) not in arrive_coords

    def test_partial_disjoint_no_exit_for_overlapping(self):
        """Parent {0,1} → child {1,2,3}: core 0 exits, core 1 continues.

        Core 0 has 1 phase with a trailing barrier.  Core 1 continues
        (2 phases, no trailing barrier).
        """
        parent = _make_op_with_cores(_make_core_ranges([((0, 0), (1, 0))]))
        child = _make_op_with_cores(_make_core_ranges([((1, 0), (3, 0))]))
        root = _graph.OpNode(parent, children=[_graph.OpNode(child)])
        groups, _ = _graph.OpGraphBuilder(root)._compute_core_groups()

        # Core 0: 1 phase with trailing barrier (not 2 phases with exit noop)
        for g in groups:
            coords = _common._core_range_set_to_coords(g.core_range)
            if (0, 0) in coords and (1, 0) not in coords:
                assert len(g.phases) == 1
                assert g.phases[0] is not _common._NOOP_OP
                assert g.has_trailing_barrier

        # Core 1: 2 phases, no trailing barrier (it continues to child)
        for g in groups:
            coords = _common._core_range_set_to_coords(g.core_range)
            if (1, 0) in coords and (0, 0) not in coords:
                assert len(g.phases) == 2
                assert not g.has_trailing_barrier


class TestEffectiveLeafRange:
    """Tests for OpGraphBuilder._effective_leaf_range."""

    def test_leaf_and_intermediate(self):
        elf = _graph.OpGraphBuilder._effective_leaf_range

        leaf = _graph.OpNode(_make_op_with_cores(_make_core_ranges([((0, 0), (1, 0))])))
        assert elf(leaf) == {(0, 0), (1, 0)}

        leaf_a = _graph.OpNode(_make_op_with_cores(_make_core_ranges([((0, 0), (1, 0))])))
        leaf_b = _graph.OpNode(_make_op_with_cores(_make_core_ranges([((4, 0), (5, 0))])))
        mid = _graph.OpNode(_make_op_with_cores(_make_core_ranges([((0, 0), (5, 0))])), children=[leaf_a, leaf_b])
        assert elf(mid) == {(0, 0), (1, 0), (4, 0), (5, 0)}


# ---------------------------------------------------------------------------
# Sequential / Parallel high-level API
# ---------------------------------------------------------------------------


def _tree_shape(node):
    """Return nested tuple of (op, [children]) for shape comparison."""
    return (node.op, [_tree_shape(c) for c in node.children])


class TestSequentialParallelAPI:
    """Tests for Sequential/Parallel resolution."""

    def test_linear_chain(self):
        a, b, c = [_make_mock_op(n) for n in "abc"]
        nodes = _fusion._resolve(_fusion.Sequential(a, b, c))
        assert len(nodes) == 1
        r = nodes[0]
        assert r.op is a and r.children[0].op is b and r.children[0].children[0].op is c

    def test_stem_and_branches(self):
        a, b, c = [_make_mock_op(n) for n in "abc"]
        nodes = _fusion._resolve(_fusion.Sequential(a, _fusion.Parallel(b, c)))
        assert nodes[0].op is a and len(nodes[0].children) == 2

    def test_nested_sequential_flattening(self):
        a, b, c, d = [_make_mock_op(n) for n in "abcd"]
        nodes = _fusion._resolve(_fusion.Sequential(_fusion.Sequential(a, _fusion.Sequential(b, c)), d))
        r = nodes[0]
        assert r.op is a and r.children[0].op is b
        assert r.children[0].children[0].op is c and r.children[0].children[0].children[0].op is d

    def test_deep_nested_split(self):
        S, P = _fusion.Sequential, _fusion.Parallel
        a, b, c, d, e = [_make_mock_op(n) for n in "abcde"]
        nodes = _fusion._resolve(S(a, P(S(b, P(c, d)), e)))
        r = nodes[0]
        assert r.op is a and len(r.children) == 2
        assert r.children[0].op is b and len(r.children[0].children) == 2
        assert r.children[1].op is e

    def test_add_chaining(self):
        S = _fusion.Sequential
        a, b, c = [_make_mock_op(n) for n in "abc"]
        s = S(a)
        assert s.add(b).add(c) is s and len(s._items) == 3

    def test_parallel_requires_two_items(self):
        with pytest.raises(ValueError, match="at least 2"):
            _fusion.Parallel(_make_mock_op("a"))

    def test_parallel_in_middle_errors(self):
        S, P = _fusion.Sequential, _fusion.Parallel
        a, b, c, d = [_make_mock_op(n) for n in "abcd"]
        with pytest.raises(ValueError, match="diverge"):
            _fusion._resolve(S(a, P(b, c), d))

    def test_empty_sequential_errors(self):
        with pytest.raises(ValueError, match="at least 1"):
            _fusion.Sequential()

    def test_resolve_raw_op_and_unsupported_type(self):
        op = _make_mock_op("raw")
        nodes = _fusion._resolve(op)
        assert len(nodes) == 1 and nodes[0].op is op
        with pytest.raises(TypeError, match="Unsupported"):
            _fusion._resolve(42)

    def test_resolve_rejects_fused_op(self):
        fused = _fusion.FusedOp(op=_fusion.OpDescriptor(_PLACEHOLDER, [], []))
        with pytest.raises(TypeError, match="FusedOp cannot be nested"):
            _fusion._resolve(fused)

    def test_merge_build_results(self, monkeypatch):
        BR = _common._BuildResult
        shared = object()
        in_a, in_b = object(), object()
        out_a, out_b = object(), object()
        sem_a, sem_b = object(), object()
        # Stub merge_program_descriptors — real one needs C++ ProgramDescriptor objects
        monkeypatch.setattr(_graph, "ttnn", _ns(merge_program_descriptors=lambda descs: descs[0]))
        r_a = BR(_PLACEHOLDER, [shared, in_a], [out_a], (sem_a,))
        r_b = BR(_PLACEHOLDER, [shared, in_b], [out_b], (sem_b,))
        merged = _graph._merge_build_results([r_a, r_b])
        assert len(merged.input_tensors) == 3  # shared deduped
        assert len(merged.output_tensors) == 2
        assert len(merged.semaphores) == 2
        assert _graph._merge_build_results([r_a]) is r_a

    def test_op_descriptor_name_field(self):
        OD = _fusion.OpDescriptor
        assert OD("desc", ["in"], ["out"]).name == ""
        assert OD("desc", ["in"], ["out"], "matmul").name == "matmul"


# ---------------------------------------------------------------------------
# Phase name generation and DeviceZoneScopedN
# ---------------------------------------------------------------------------


class TestPhaseNameGeneration:
    """Tests for phase name propagation in generated source."""

    def test_phase_namespace_with_and_without_name(self):
        gen = _codegen._generate_phase_namespace
        source = "void kernel_main() { int x = 1; }"

        lines = gen(0, "", source, [], 0, phase_name="rms_norm")
        assert "Phase 0: rms_norm" in lines[1]
        joined = "\n".join(lines)
        assert 'DeviceZoneScopedN("rms_norm");' in joined

        lines = gen(0, "", source, [], 0, phase_name="")
        assert "Phase 0" in lines[1]
        assert "DeviceZoneScopedN" not in "\n".join(lines)

    def test_default_no_zone(self):
        lines = _codegen._generate_phase_namespace(0, "", "void kernel_main() { int x = 1; }", [], 0)
        assert "DeviceZoneScopedN" not in "\n".join(lines)


# ---------------------------------------------------------------------------
# Fused source generation (semaphores, has_writer)
# ---------------------------------------------------------------------------


class TestSemaphoreInitialValue:
    """Tests that semaphore reset uses initial_value, not hardcoded 0."""

    def _gen(self, op_semaphore_info):
        phases, phase_kernels = _make_two_phase_kernel_setup()
        return _codegen._generate_fused_source(
            phase_kernels,
            "reader",
            phases,
            {0: 0, 1: 0},
            [[0]],
            risc_type="riscv_1",
            role_label="reader",
            rebind_info={},
            op_semaphore_info=op_semaphore_info,
            multi_barrier=_make_barrier_spec(has_reset_done=False),
        )

    def test_nonzero_initial_value(self):
        result = self._gen([(3, 5)])
        assert "get_semaphore(3)) = 5;" in result
        assert "get_semaphore(3)) = 0;" not in result

    def test_mixed_initial_values(self):
        result = self._gen([(2, 0), (7, 3)])
        assert "get_semaphore(2)) = 0;" in result
        assert "get_semaphore(7)) = 3;" in result


class TestHasWriterFlag:
    """Tests for has_writer flag in barrier generation."""

    @pytest.mark.parametrize(
        "has_compute,has_writer,expect_in,expect_not_in",
        [
            (True, True, ["writer_done", "compute_done", "reset_done"], []),
            (True, False, ["compute_done", "reset_done"], ["writer_done"]),
            (False, True, ["writer_done", "reset_done"], ["compute_done"]),
            (False, False, ["reset_done"], ["compute_done", "writer_done"]),
        ],
    )
    def test_state_vars(self, has_compute, has_writer, expect_in, expect_not_in):
        text = "\n".join(
            _barrier_mod._emit_state_vars(
                "riscv_1", is_coordinator=True, has_compute=has_compute, has_writer=has_writer
            )
        )
        for s in expect_in:
            assert s in text
        for s in expect_not_in:
            assert s not in text

    @pytest.mark.parametrize(
        "has_compute,has_writer,wait_compute,wait_writer",
        [
            (True, True, True, True),
            (True, False, True, False),
            (False, True, False, True),
            (False, False, False, False),
        ],
    )
    def test_local_sync_coordinator(self, has_compute, has_writer, wait_compute, wait_writer):
        text = "\n".join(
            _barrier_mod._emit_local_sync_coordinator(
                has_compute=has_compute,
                dispatch=[],
                op_semaphore_info=[],
                has_writer=has_writer,
            )
        )
        assert ("noc_semaphore_wait_min(compute_done" in text) == wait_compute
        assert ("noc_semaphore_wait_min(writer_done" in text) == wait_writer

    @pytest.mark.parametrize(
        "has_compute,has_writer",
        [
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ],
    )
    def test_init_coordinator_rt_offsets(self, has_compute, has_writer):
        text = "\n".join(
            _barrier_mod._emit_init_coordinator(
                has_compute=has_compute,
                num_segments=0,
                op_semaphore_info=None,
                has_writer=has_writer,
            )
        )
        idx = 0
        if has_compute:
            assert "compute_done = reinterpret_cast" in text
            idx += 1
        else:
            assert "compute_done = reinterpret_cast" not in text
        if has_writer:
            assert "writer_done = reinterpret_cast" in text
            idx += 1
        else:
            assert "writer_done = reinterpret_cast" not in text
        assert f"rt_offset + {idx})" in text  # reset_done is always last

    @pytest.mark.parametrize("has_writer", [True, False])
    def test_barrier_namespace_full(self, has_writer):
        mb = _make_barrier_spec()
        text = "\n".join(
            _barrier_mod._generate_barrier_namespace(
                "riscv_1",
                mb,
                rebind_info={},
                sources=[(0, ""), (1, "")],
                per_phase_cb_slots=[[0], [0]],
                op_semaphore_info=None,
                has_compute=True,
                has_writer=has_writer,
            )
        )
        assert ("volatile tt_l1_ptr uint32_t* writer_done;" in text) == has_writer
        assert "volatile tt_l1_ptr uint32_t* compute_done;" in text
        assert ("noc_semaphore_wait_min(writer_done" in text) == has_writer

    def test_fused_source_passes_has_writer(self):
        phases, phase_kernels = _make_two_phase_kernel_setup()
        mb = _make_barrier_spec()
        for hw, expect_writer in [(False, False), (True, True)]:
            result = _codegen._generate_fused_source(
                phase_kernels,
                "reader",
                phases,
                {0: 0, 1: 0},
                [[0], [0]],
                risc_type="riscv_1",
                role_label="reader",
                rebind_info={},
                multi_barrier=mb,
                has_compute=True,
                has_writer=hw,
            )
            assert ("volatile tt_l1_ptr uint32_t* writer_done;" in result) == expect_writer
            assert "volatile tt_l1_ptr uint32_t* compute_done;" in result


class TestCoordinatorOnlySource:
    """Tests for barrier-only coordinator source generation."""

    @pytest.mark.parametrize(
        "has_compute,has_writer",
        [
            (True, False),
            (False, False),
            (True, True),
        ],
    )
    def test_coordinator_only_source(self, has_compute, has_writer):
        mb = _make_barrier_spec()
        source = _source_gen_mod._generate_coordinator_only_source(
            multi_barrier=mb,
            rebind_info={},
            all_phase_indices=[0, 1],
            per_phase_cb_slots=[[0], [0]],
            op_semaphore_info=None,
            has_compute=has_compute,
            has_writer=has_writer,
            phase_names={0: "op_a", 1: "op_b"},
        )
        assert "void kernel_main()" in source
        assert "barrier::init();" in source
        assert "barrier::sync();" in source
        assert "phase_0::run()" not in source
        assert "Phase 0: op_a (no-op for this RISC)" in source
        assert ("volatile tt_l1_ptr uint32_t* compute_done;" in source) == has_compute
        assert ("volatile tt_l1_ptr uint32_t* writer_done;" in source) == has_writer

    def test_coordinator_only_includes(self):
        mb = _make_barrier_spec()
        source = _source_gen_mod._generate_coordinator_only_source(
            multi_barrier=mb,
            rebind_info={},
            all_phase_indices=[0, 1],
            per_phase_cb_slots=[[0], [0]],
            op_semaphore_info=None,
            has_compute=True,
            has_writer=True,
            phase_names={},
        )
        assert '#include "dataflow_api.h"' in source
        assert "#include <array>" in source


# ---------------------------------------------------------------------------
# GlobalCircularBuffer support
# ---------------------------------------------------------------------------


class TestGlobalCircularBuffer:
    """Tests for GlobalCircularBuffer support in the fusion framework."""

    def _regular_cb(self, idx=0):
        return _make_cb_mock(idx)

    def _global_cb(self, local_idx=1, remote_idx=31):
        local_fmt = _ns(
            buffer_index=local_idx, data_format="F16", page_size=1024, data_format_as_uint8=hash("F16") & 0xFF
        )
        remote_fmt = _ns(
            buffer_index=remote_idx, data_format="F16", page_size=1024, data_format_as_uint8=hash("F16") & 0xFF
        )
        return _ns(
            total_size=4096,
            core_ranges="mock",
            address_offset=0,
            format_descriptors=[local_fmt],
            remote_format_descriptors=[remote_fmt],
            has_global_circular_buffer=lambda: True,
            has_buffer=lambda: False,
        )

    def _remote_only_cb(self, remote_idx=31):
        remote_fmt = _ns(
            buffer_index=remote_idx, data_format="F16", page_size=1024, data_format_as_uint8=hash("F16") & 0xFF
        )
        return _ns(
            total_size=4096,
            core_ranges="mock",
            address_offset=0,
            format_descriptors=[],
            remote_format_descriptors=[remote_fmt],
            has_global_circular_buffer=lambda: True,
            has_buffer=lambda: False,
        )

    def test_extract_cb_info_skips_remote(self):
        result = _cb_alloc.extract_cb_info(_ns(cbs=[self._regular_cb(0), self._global_cb(1, 31)]))
        assert 0 in result and 1 in result and 31 not in result

    def test_extract_remote_only_no_cbinfo(self):
        assert len(_cb_alloc.extract_cb_info(_ns(cbs=[self._remote_only_cb()]))) == 0

    def test_extract_remote_cb_indices(self):
        assert _cb_alloc._extract_remote_cb_indices(_ns(cbs=[self._regular_cb(), self._global_cb()])) == {31}
        assert _cb_alloc._extract_remote_cb_indices(_ns(cbs=[self._regular_cb()])) == set()

    def test_reserve_index_prevents_collision(self):
        pool = _cb_alloc.CBPoolAllocator()
        pool.reserve_index(31)
        CI = _cb_alloc.CBInfo
        pool.allocate_phase(0, {31: CI(31, 2048, "F16", 1024, None, "Default")}, set())
        assert pool.get_remap(0)[31] != 31

    def test_reserve_index_not_in_remap_values(self):
        pool = _cb_alloc.CBPoolAllocator()
        pool.reserve_index(31)
        CI = _cb_alloc.CBInfo
        pool.allocate_phase(0, {0: CI(0, 2048, "F16", 1024, None, "Default")}, set())
        pool.allocate_phase(1, {0: CI(0, 2048, "F16", 1024, None, "Default")}, set())
        for i in range(2):
            assert 31 not in pool.get_remap(i).values()

    def test_local_index_pool_allocated(self):
        pool = _cb_alloc.CBPoolAllocator()
        pool.reserve_index(31)
        CI = _cb_alloc.CBInfo
        pool.allocate_phase(
            0, {0: CI(0, 2048, "F16", 1024, None, "Default"), 1: CI(1, 4096, "F16", 1024, None, "Default")}, set()
        )
        pool.allocate_phase(1, {1: CI(1, 2048, "F16", 1024, None, "Default")}, set())
        assert pool.get_remap(0)[1] == pool.get_remap(1)[1]

    def test_build_merged_includes_global_cbs(self, monkeypatch):
        # Stub CBDescriptor — real one rejects fake core_ranges
        monkeypatch.setattr(
            _cb_alloc,
            "ttnn",
            _ns(
                CBDescriptor=lambda: _ns(),
                CBFormatDescriptor=lambda **kw: _ns(**kw),
            ),
        )
        pool = _cb_alloc.CBPoolAllocator()
        pool.reserve_index(31)
        CI, PI = _cb_alloc.CBInfo, _cb_alloc.PhaseInfo

        regular = self._regular_cb(0)
        remote_only = self._remote_only_cb(31)
        mock_op = _ns(descriptor=_ns(cbs=[regular, remote_only]))

        fmt0 = regular.format_descriptors[0]
        cb_info = {0: CI(0, 2048, "F16", 1024, None, False, source_fmt=fmt0, source_cb=regular)}
        pool.allocate_phase(0, cb_info, set())
        merged = pool.build_merged_cb_descriptors([PI(0, mock_op, cb_info)])
        assert len(merged) == 2 and remote_only in merged


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


class TestCompileTimePerformance:
    """Sanity check that CB pool allocation doesn't regress catastrophically."""

    def test_build_time_scales_linearly(self):
        import time

        def _alloc(n):
            pool = _cb_alloc.CBPoolAllocator()
            start = time.perf_counter()
            for i in range(n):
                cbs = {}
                for j in [0, 1, 16]:
                    cb = _make_cb_mock(j, "Float16_b", 2048, 2048)
                    cbs.update(_cb_alloc.extract_cb_info(_ns(cbs=[cb])))
                pool.allocate_phase(i, cbs, set())
            return time.perf_counter() - start

        t2, t8 = _alloc(2), _alloc(8)
        assert t8 < 10.0, f"8-phase took {t8:.2f}s"
        if t2 > 0.001:
            assert t8 / t2 < 20.0, f"Ratio {t8/t2:.1f}x exceeds 20x"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
