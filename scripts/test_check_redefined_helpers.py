# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for check_redefined_helpers.py."""

import os
import sys
import textwrap

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_redefined_helpers as chk  # noqa: E402
from precommit_common import REPO_ROOT as REPO  # noqa: E402


def names(path, body, lines=None):
    text = textwrap.dedent(body)
    return [item[2] for item in chk.find_violations(path, text, lines)]


def test_host_function_copies_are_flagged():
    found = names(
        "ttnn/cpp/ttnn/operations/example/host.cpp",
        """
        inline uint32_t align_up(uint32_t a, uint32_t align) { return (a + align - 1) & ~(align - 1); }
        constexpr std::size_t round_up(std::size_t a, std::size_t multiple) { return a; }
        uint32_t div_up(uint32_t n, uint32_t d) { return (n + d - 1) / d; }
        constexpr auto round_down(int a, int b) { return a; }
        uint32_t round_up_to_tile(uint32_t value, uint32_t tile_size) { return value; }
        """,
    )
    assert found == ["align_up", "round_up", "div_up", "round_down", "round_up_to_tile"]


def test_calls_are_not_definitions():
    found = names(
        "tt_metal/impl/device/firmware/risc_firmware_initializer.cpp",
        """
        void init() {
            worker.resize(tt::round_up(tensix_grid_size.x, 4), 0);
            return tt::div_up(a, b);
            auto n = tt::constants::TILE_HEIGHT;
            uint32_t pages = div_up(size, page_size);
        }
        """,
    )
    assert found == []


def test_device_round_up_is_flagged_and_div_up_is_not():
    found = names(
        "ttnn/cpp/ttnn/operations/example/device/kernels/reader.cpp",
        """
        FORCE_INLINE uint32_t div_up(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
        FORCE_INLINE uint32_t align_down(uint32_t a, uint32_t b) { return a & ~(b - 1); }
        FORCE_INLINE uint32_t round_up(uint32_t a, uint32_t b) { return b * div_up(a, b); }
        constexpr uint32_t TILE_HEIGHT = 32;
        constexpr uint32_t TILE_WIDTH = get_arg(args::tile_width);
        constexpr uint32_t TILE_HW = TILE_HEIGHT * TILE_WIDTH;
        """,
    )
    assert found == ["round_up", "TILE_HEIGHT"]


def test_device_advice_points_at_alignment_header():
    text = "FORCE_INLINE uint32_t round_up(uint32_t a, uint32_t b) { return a; }\n"
    found = chk.find_violations("tt_metal/hw/inc/api/example.h", text)
    assert len(found) == 1
    assert "api/alignment.h" in found[0][3]


def test_tile_constant_from_an_argument_is_allowed():
    found = names(
        "ttnn/ops/device/kernels/dataflow/reader.cpp",
        """
        constexpr uint32_t TILE_HEIGHT = get_arg(args::tile_height);
        constexpr uint32_t TILE_WIDTH = get_compile_time_arg_val(3);
        """,
    )
    assert found == []


def test_literal_tile_constant_on_the_host():
    found = chk.find_violations(
        "tt_metal/programming_examples/contributed/vecadd/vecadd.cpp",
        "constexpr uint32_t TILE_WIDTH = 32;\nconstexpr uint32_t TILE_HEIGHT = 32;\n",
    )
    assert [item[2] for item in found] == ["TILE_WIDTH", "TILE_HEIGHT"]
    assert "tt-metalium/constants.hpp" in found[0][3]


def test_l1_budget_formula_is_flagged_and_the_canonical_call_is_not():
    found = names(
        "ttnn/cpp/ttnn/operations/example/device/factory.cpp",
        """
        uint32_t max_l1_size =
            device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(HalMemType::L1);
        uint32_t half_l1_budget =
            (device->l1_size_per_core() / 2) - device->allocator()->get_base_allocator_addr(HalMemType::L1);
        uint32_t other = operations::data_movement::get_max_l1_space(input_tensor);
        """,
    )
    assert found == ["max L1 budget"]


def test_same_line_escape_hatch():
    found = names(
        "ttnn/cpp/ttnn/operations/example/host.hpp",
        """
        inline uint32_t align_up(uint32_t a, uint32_t b) { return a; }  // allow-redefined-helper: hostdev packet layout
        inline uint32_t round_up(uint32_t a, uint32_t b) { return a; }  /* allow-redefined-helper: shared layout */
        """,
    )
    assert found == []


def test_escape_hatch_inside_a_string_does_not_apply():
    found = names(
        "ttnn/cpp/ttnn/operations/example/host.hpp",
        """
        inline uint32_t align_up(uint32_t a, uint32_t b) { const char* msg = "allow-redefined-helper"; return a; }
        inline uint32_t round_up(uint32_t a, uint32_t b) { const char* msg = "// allow-redefined-helper"; return a; }
        """,
    )
    assert found == ["align_up", "round_up"]


def test_define_is_visible_and_a_commented_define_is_not():
    found = names(
        "ttnn/cpp/ttnn/operations/example/host.hpp",
        r"""
        #define TILE_HEIGHT 32
        /*
        #define TILE_WIDTH 32
        */
        // escaped-newline comment \
        constexpr uint32_t FACE_HEIGHT = 16;
        """,
    )
    assert found == ["TILE_HEIGHT"]


def test_only_added_lines_are_reported():
    body = "inline uint32_t align_up(uint32_t a, uint32_t b) { return a; }\nuint32_t div_up(uint32_t a, uint32_t b) { return a; }\n"
    assert names("ttnn/new.cpp", body, lines={2}) == ["div_up"]
    assert names("ttnn/new.cpp", body, lines=set()) == []


def test_canonical_headers_are_clean():
    for rel in (
        "tt_metal/api/tt-metalium/math.hpp",
        "tt_metal/api/tt-metalium/constants.hpp",
        "tt_metal/hw/inc/api/tile_constants.h",
        "tt_metal/hw/inc/api/alignment.h",
        "ttnn/cpp/ttnn/operations/core/program_cache_l1.hpp",
    ):
        with open(os.path.join(REPO, rel), encoding="utf-8") as handle:
            assert chk.find_violations(rel, handle.read()) == []


def test_real_copies_match_the_catalog():
    cases = [
        (
            "tt_metal/impl/buffers/tensor_prefetcher_manager.cpp",
            "inline uint32_t align_up(uint32_t a, uint32_t align) { return (a + align - 1) & ~(align - 1); }\n",
            "align_up",
        ),
        (
            "ttnn/cpp/ttnn/operations/ccl/common/uops/ccl_command.hpp",
            "constexpr std::size_t round_up(std::size_t a, std::size_t multiple) { return ((a + multiple - 1) / multiple) * multiple; }\n",
            "round_up",
        ),
        (
            "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/writer_unary_unpad_sharded_to_interleaved.cpp",
            "void kernel_main() {\n    constexpr uint32_t TILE_HEIGHT = 32;  // TODO: use common source of truth\n}\n",
            "TILE_HEIGHT",
        ),
    ]
    for rel, snippet, name in cases:
        found = [item[2] for item in chk.find_violations(rel, snippet)]
        assert name in found, rel


def test_unrelated_suffixed_names_are_not_flagged():
    found = names(
        "tt_metal/hw/inc/api/compute/experimental/2_0/internal/llk_descriptor.h",
        """
        constexpr std::uint32_t round_up_l1_words(std::uint32_t bytes) { return (bytes + 15u) >> 4; }
        uint32_t align_to_slice_and_sync(uint32_t block_idx, uint32_t sender_id) { return block_idx; }
        """,
    )
    assert found == []


def test_compilation_context_classification():
    assert chk.side_of("tt_metal/impl/device/firmware/risc_firmware_initializer.cpp") == "host"
    assert chk.side_of("tt_metal/hw/firmware/src/tt-1xx/brisck.cc") == "device"
    assert chk.side_of("tt-train/sources/ttml/metal/common/dataflow_utils.hpp") == "device"
    assert chk.side_of("tt-train/sources/ttml/metal/common/compute_utils.hpp") == "device"
    assert chk.side_of("ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp") == "device"
    assert chk.side_of("ttnn/cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp") == "both"
    assert names(
        "tt_metal/impl/device/firmware/risc_firmware_initializer.cpp",
        "uint32_t div_up(uint32_t a, uint32_t b) { return (a + b - 1) / b; }\n",
    ) == ["div_up"]
    dev_violations = chk.find_violations(
        "tt-train/sources/ttml/metal/common/dataflow_utils.hpp",
        "inline constexpr uint32_t round_up(uint32_t a, uint32_t b) { return a; }\n",
    )
    assert len(dev_violations) == 1
    assert "api/alignment.h" in dev_violations[0][3]


def test_multiline_declarations_are_detected():
    body = textwrap.dedent(
        """\
        inline constexpr uint32_t
        round_up(uint32_t a, uint32_t b = {}) {
            return a;
        }
        constexpr uint32_t TILE_HEIGHT =
            32;
        """
    )
    assert [item[2] for item in chk.find_violations("ttnn/cpp/ttnn/operations/example/host.cpp", body)] == [
        "round_up",
        "TILE_HEIGHT",
    ]
    assert [item[2] for item in chk.find_violations("ttnn/cpp/ttnn/operations/example/host.cpp", body, {1})] == [
        "round_up"
    ]
    assert [item[2] for item in chk.find_violations("ttnn/cpp/ttnn/operations/example/host.cpp", body, {3})] == [
        "round_up"
    ]
    assert [item[2] for item in chk.find_violations("ttnn/cpp/ttnn/operations/example/host.cpp", body, {6})] == [
        "TILE_HEIGHT"
    ]


def test_attributed_and_template_return_declarations_are_detected():
    assert (
        names(
            "ttnn/cpp/ttnn/operations/example/device/kernels/reader.cpp",
            """
        __attribute__((always_inline)) constexpr uint32_t align(uint32_t addr, uint32_t alignment) {
            return addr;
        }
        """,
        )
        == ["align"]
    )
    assert (
        names(
            "ttnn/cpp/ttnn/operations/example/host.hpp",
            """
        template <typename T1, typename T2>
        constexpr std::common_type_t<T1, T2> align(T1 addr, T2 alignment) {
            return addr;
        }
        """,
        )
        == ["align"]
    )


def test_constant_values_are_verified():
    found = names(
        "ttnn/cpp/ttnn/operations/example/device/kernels/reader.cpp",
        """
        #define TILE_WIDTH get_compile_time_arg_val(3)
        #define TILE_HEIGHT 32
        constexpr uint32_t TILE_HW = 32;
        constexpr uint32_t TILE_HW = TILE_HEIGHT * TILE_WIDTH;
        constexpr uint32_t FACE_HEIGHT = 8;
        constexpr uint32_t FACE_HW = 16 * 16;
        """,
    )
    assert found == ["TILE_HEIGHT", "FACE_HW"]
    assert (
        names(
            "ttnn/cpp/ttnn/operations/example/host.cpp",
            """
        constexpr uint32_t TILE_HW = tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH;
        """,
        )
        == ["TILE_HW"]
    )


def test_l1_check_ignores_raw_limit_and_detects_first_statement_in_function():
    assert (
        names(
            "tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp",
            "void validate(const IDevice* device) { uint32_t max_l1_size = device->l1_size_per_core(); }\n",
        )
        == []
    )
    assert (
        names(
            "ttnn/cpp/ttnn/operations/example/device/factory.cpp",
            "void validate(const IDevice* device) { uint32_t max_l1_size = device->l1_size_per_core(); }\n",
        )
        == []
    )
    body = textwrap.dedent(
        """\
        void build_program(IDevice* device) {
            uint32_t max_l1_size =
                device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(HalMemType::L1);
        }
        """
    )
    assert [
        item[:3] for item in chk.find_violations("ttnn/cpp/ttnn/operations/example/device/factory.cpp", body, {2, 3})
    ] == [(2, "l1", "max L1 budget")]
    assert [
        item[:3] for item in chk.find_violations("ttnn/cpp/ttnn/operations/example/device/factory.cpp", body, {3})
    ] == [(2, "l1", "max L1 budget")]


def test_git_rev_rejects_option_injection():
    assert chk.safe_git_rev("abc123def") == "abc123def"
    assert chk.safe_git_rev("origin/main") == "origin/main"
    assert chk.safe_git_rev("HEAD^") == "HEAD^"
    assert chk.safe_git_rev("--output=/tmp/x") is None
    assert chk.safe_git_rev("-c") is None
    assert chk.safe_git_rev("HEAD;rm") is None
    assert chk.safe_git_rev("rev\n--version") is None


def test_paths_outside_the_repo_are_refused():
    assert chk.safe_repo_path("/etc/passwd") is None
    inside = chk.safe_repo_path("scripts/check_redefined_helpers.py")
    assert inside is not None
    assert inside.endswith(os.path.join("scripts", "check_redefined_helpers.py"))


def test_diff_is_split_by_path_without_passing_the_path_to_git():
    diff = """\
diff --git a/ttnn/a.cpp b/ttnn/a.cpp
--- a/ttnn/a.cpp
+++ b/ttnn/a.cpp
@@ -1,0 +1,1 @@
+uint32_t div_up(uint32_t a, uint32_t b);
diff --git a/old b/ttnn/b.cpp
--- a/old
+++ b/ttnn/b.cpp
@@ -0,0 +4,1 @@
+constexpr uint32_t TILE_HEIGHT = 32;
"""
    found = chk.added_lines_by_path(diff)
    assert found["ttnn/a.cpp"] == {1}
    assert found["ttnn/b.cpp"] == {4}


def test_diff_parser_counts_only_additions(monkeypatch):
    diff = """\
@@ -10,2 +10,3 @@
 kept
-old
+uint32_t div_up(uint32_t a, uint32_t b);
+constexpr uint32_t TILE_HEIGHT = 32;
 kept-too
"""
    assert chk.parse_added_lines(diff) == {11, 12}

    head_diff = (
        "diff --git a/ttnn/a.cpp b/ttnn/a.cpp\n--- a/ttnn/a.cpp\n+++ b/ttnn/a.cpp\n@@ -1,0 +1,1 @@\n+int x;\n"
        "diff --git a/ttnn/b.cpp b/ttnn/b.cpp\n--- a/ttnn/b.cpp\n+++ b/ttnn/b.cpp\n@@ -1,0 +3,1 @@\n+int y;\n"
    )
    monkeypatch.delenv("PRE_COMMIT_FROM_REF", raising=False)
    monkeypatch.delenv("PRE_COMMIT_TO_REF", raising=False)
    monkeypatch.setattr(chk, "_git_diff", lambda args: head_diff if args == ["HEAD"] else None)
    assert chk.commit_added_lines() == {"ttnn/a.cpp": {1}, "ttnn/b.cpp": {3}}


def test_list_mentions_both_sides(capsys):
    assert chk.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "tt-metalium/math.hpp" in out
    assert "api/tile_constants.h" in out
    assert "get_max_l1_space" in out
