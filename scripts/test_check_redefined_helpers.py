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
        FORCE_INLINE uint32_t round_up(uint32_t a, uint32_t b) { return b * div_up(a, b); }
        constexpr uint32_t TILE_HEIGHT = 32;
        constexpr uint32_t TILE_WIDTH = get_arg(args::tile_width);
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
        """
        #define TILE_HEIGHT 32
        /*
        #define TILE_WIDTH 32
        */
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
    cases = {
        "tt_metal/impl/buffers/tensor_prefetcher_manager.cpp": "align_up",
        "ttnn/cpp/ttnn/operations/ccl/common/uops/ccl_command.hpp": "round_up",
        "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/writer_unary_unpad_sharded_to_interleaved.cpp": "TILE_HEIGHT",
    }
    for rel, name in cases.items():
        with open(os.path.join(REPO, rel), encoding="utf-8") as handle:
            found = [item[2] for item in chk.find_violations(rel, handle.read())]
        assert name in found, rel


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


def test_diff_parser_counts_only_additions():
    diff = """\
@@ -10,2 +10,3 @@
 kept
-old
+uint32_t div_up(uint32_t a, uint32_t b);
+constexpr uint32_t TILE_HEIGHT = 32;
 kept-too
"""
    assert chk.parse_added_lines(diff) == {11, 12}


def test_list_mentions_both_sides(capsys):
    assert chk.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "tt-metalium/math.hpp" in out
    assert "api/tile_constants.h" in out
    assert "get_max_l1_space" in out
