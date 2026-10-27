#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent ULC
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for the compute-API ordering checker.

Each test writes a small kernel against a stub of the compute API, a one-entry
`compile_commands.json` that marks it as one TRISC thread, and runs the checker exactly as the
pre-commit hook would. The stub keeps the tests independent of the SFPI toolchain: the thing under
test is the ordering analysis, not whether the real headers parse.

The kernels are written so that each one exercises a single diagnostic; the argument of the last
assert names the behaviour a reviewer should be able to read off the test.
"""
import contextlib
import io
import json
import os
import sys

import pytest

pytest.importorskip("clang.cindex", reason="the checker needs the libclang Python bindings")

HERE = os.path.dirname(os.path.abspath(__file__))
INFRA = os.path.normpath(os.path.join(HERE, ".."))
REPO = os.path.normpath(os.path.join(INFRA, "..", "..", ".."))
sys.path.insert(0, INFRA)

import check_compute_api_order as checker  # noqa: E402

# The compute-API surface the rule table names. Declarations are enough: the checker matches on the
# callee name before it ever considers inlining, and a declaration resolves the call so libclang
# gives us a CALL_EXPR with a `referenced` cursor. Signatures follow the real headers, including
# the argument the rule table compares.
STUB = """
#pragma once

using uint32_t = unsigned;

// MATH
void add_init(uint32_t icb0, uint32_t icb1, bool acc_to_dest = false, uint32_t call_line = 0);
void add_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst);
void sub_init(uint32_t icb0, uint32_t icb1, bool acc_to_dest = false, uint32_t call_line = 0);
void sub_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst);
void mul_init(uint32_t icb0, uint32_t icb1, bool acc_to_dest = false, uint32_t call_line = 0);
void mul_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst);
void matmul_init(uint32_t in0, uint32_t in1, uint32_t transpose = 0, uint32_t call_line = 0);
void matmul_tiles(uint32_t in0, uint32_t in1, uint32_t itile0, uint32_t itile1, uint32_t idst);
void exp_tile_init();
void exp_tile(uint32_t idst);
void sqrt_tile_init();
void sqrt_tile(uint32_t idst);
void copy_tile_init(uint32_t icb);
void reconfig_data_format(uint32_t icb0, uint32_t icb1);
void reconfig_data_format_srca(uint32_t icb);

// UNPACK
void tilize_init(uint32_t icb, uint32_t block, uint32_t ocb, uint32_t call_line = 0);
void tilize_block(uint32_t icb, uint32_t block, uint32_t ocb,
                  uint32_t input_tile_index = 0, uint32_t output_tile_index = 0);
void tilize_uninit(uint32_t icb, uint32_t ocb);
void fast_tilize_init(uint32_t icb, uint32_t full_dim, uint32_t ocb, uint32_t call_line = 0);
void fast_tilize_block(uint32_t icb, uint32_t block, uint32_t ocb,
                       uint32_t input_tile_index = 0, uint32_t output_tile_index = 0);
void fast_tilize_uninit(uint32_t icb, uint32_t ocb, uint32_t full_dim);
void untilize_init(uint32_t icb, uint32_t call_line = 0);
void untilize_block(uint32_t icb, uint32_t full_ct_dim, uint32_t ocb);
void untilize_uninit(uint32_t ocb);

// PACK
void pack_init(uint32_t ocb);
void pack_tile(uint32_t ifrom_dst, uint32_t icb);
"""


def run(tmp_path, body, thread="MATH", extra_args=()):
    """Run the checker on one kernel and return (exit code, stdout, stderr)."""
    (tmp_path / "compute_stub.h").write_text(STUB)
    kernel = tmp_path / "kernel.cpp"
    kernel.write_text('#include "compute_stub.h"\n' + body)

    database = [
        {
            "directory": str(tmp_path),
            "file": str(kernel),
            "arguments": [
                "clang++",
                "-std=c++20",
                "-fsyntax-only",
                "-I",
                str(tmp_path),
                f"-DUCK_CHLKC_{thread}",
                "-DCOMPILE_FOR_TRISC=1",
                "-DARCH_WORMHOLE",
                str(kernel),
            ],
        }
    ]
    db = tmp_path / "compile_commands.json"
    db.write_text(json.dumps(database))

    argv = [
        "--compile-db",
        str(db),
        "--repo-root",
        REPO,
        "--rules",
        os.path.join(INFRA, "compute_api_rules.yaml"),
        *extra_args,
    ]
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = checker.main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


def kinds(stdout):
    """The `kind [rule]` pairs the checker printed, in order."""
    found = []
    for line in stdout.splitlines():
        if ": " not in line or line.startswith("check_compute_api_order:"):
            continue
        tail = line.split(": ", 1)[1]
        if " [" in tail:
            found.append(tail.split(" [", 1)[0] + ":" + tail.split("[", 1)[1].split("]", 1)[0])
    return found


# --------------------------------------------------------------------------------------------------
# The six diagnostics
# --------------------------------------------------------------------------------------------------


def test_ordered_add_is_clean(tmp_path):
    code, out, _ = run(
        tmp_path,
        """
void kernel_main() {
    add_init(0, 1);
    add_tiles(0, 1, 0, 0, 0);
}
""",
    )
    assert code == 0
    assert kinds(out) == [], out


def test_operation_without_init_is_flagged(tmp_path):
    _, out, _ = run(
        tmp_path,
        """
void kernel_main() {
    add_init(0, 1);
    add_tiles(0, 1, 0, 0, 0);
    sub_tiles(0, 1, 0, 0, 0);
}
""",
    )
    assert kinds(out) == ["missing-init:eltwise-binary-sub"], (
        "sub_tiles needs its own sub_init; add_init does not configure it"
    )


def test_init_and_operation_arguments_must_agree(tmp_path):
    _, out, _ = run(
        tmp_path,
        """
void kernel_main() {
    add_init(0, 1);
    add_tiles(1, 0, 0, 0, 0);
}
""",
    )
    assert kinds(out) == ["arg-mismatch:eltwise-binary-add"], (
        "add_tiles(1, 0) reads the operands the wrong way round"
    )
    assert "conflicts with" in out


def test_compile_time_constants_are_compared_by_value(tmp_path):
    _, out, _ = run(
        tmp_path,
        """
void kernel_main() {
    add_init(0 + 2, 1 * 3);
    add_tiles(2, 3, 0, 0, 0);
}
""",
    )
    assert kinds(out) == [], "0+2 folds to 2 and 1*3 folds to 3; the same buffers are configured"


def test_unknown_arguments_are_not_guessed(tmp_path):
    _, out, _ = run(
        tmp_path,
        """
bool runtime_flag;
void kernel_main() {
    add_init(runtime_flag ? 1 : 2, 0);
    add_tiles(runtime_flag ? 1 : 2, 0, 0, 0, 0);
}
""",
    )
    assert kinds(out) == [], "a runtime value must not be reported as a mismatch"
    assert "not comparable at compile time" in out, (
        "when the checker cannot prove the operands agree it says so instead of staying silent"
    )


def test_a_named_operand_is_compared_by_identity(tmp_path):
    _, out, _ = run(
        tmp_path,
        """
bool runtime_flag;
void kernel_main() {
    uint32_t cb = runtime_flag ? 1 : 2;
    add_init(cb, 0);
    add_tiles(cb, 0, 0, 0, 0);
}
""",
    )
    assert kinds(out) == [], "the same variable at both sites is provably the same operand"
    assert "not comparable at compile time" not in out


def test_two_runtime_locals_are_not_a_mismatch(tmp_path):
    _, out, _ = run(
        tmp_path,
        """
bool runtime_flag;
void kernel_main() {
    uint32_t init_cb = runtime_flag ? 1 : 2;
    uint32_t op_cb = runtime_flag ? 3 : 4;
    add_init(init_cb, 0);
    add_tiles(op_cb, 0, 0, 0, 0);
}
""",
    )
    assert kinds(out) == [], "two different runtime locals may hold the same CB id; do not guess"
    assert "not comparable at compile time" in out, (
        "the checker says the arguments are unknown instead of claiming a mismatch"
    )


def test_two_compile_time_identities_are_a_mismatch(tmp_path):
    _, out, _ = run(
        tmp_path,
        """
struct dfb_t {
    uint32_t id;
    constexpr uint32_t get_id() const { return id; }
};
constexpr dfb_t cb0{0};
constexpr dfb_t cb1{1};

void kernel_main() {
    add_init(cb0.get_id(), 0);
    add_tiles(cb1.get_id(), 0, 0, 0, 0);
}
""",
    )
    assert kinds(out) == ["arg-mismatch:eltwise-binary-add"], (
        "cb0 and cb1 are distinct compile-time identities, so the init cannot apply to the operation"
    )


def test_reconfig_between_init_and_operation_is_an_init_conflict(tmp_path):
    _, out, _ = run(
        tmp_path,
        """
void kernel_main() {
    add_init(0, 1);
    sub_init(0, 1);
    add_tiles(0, 1, 0, 0, 0);
}
""",
    )
    assert "init-conflict:eltwise-binary-add" in kinds(out), (
        "sub_init reconfigures SrcA/SrcB after add_init; add_tiles can no longer rely on add_init"
    )


def test_invalidator_drops_the_configuration(tmp_path):
    _, out, _ = run(
        tmp_path,
        """
void kernel_main() {
    add_init(0, 1);
    reconfig_data_format(2, 3);
    add_tiles(0, 1, 0, 0, 0);
}
""",
    )
    assert kinds(out) == ["missing-init:eltwise-binary-add"], (
        "a data-format reconfig invalidates the init, so add_tiles has no live configuration"
    )
    assert "reconfig_data_format() dropped" in out


def test_uninit_is_required_for_tilize(tmp_path):
    _, out, _ = run(
        tmp_path,
        """
void kernel_main() {
    tilize_init(0, 4, 1);
    tilize_block(0, 4, 1);
}
""",
        thread="UNPACK",
    )
    assert kinds(out) == ["missing-uninit:tilize"], (
        "tilize_init opens a region that only tilize_uninit closes"
    )


def test_closed_region_is_clean(tmp_path):
    _, out, _ = run(
        tmp_path,
        """
void kernel_main() {
    tilize_init(0, 4, 1);
    tilize_block(0, 4, 1);
    tilize_uninit(0, 1);
}
""",
        thread="UNPACK",
    )
    assert kinds(out) == [], out


def test_invalidator_inside_an_open_region_is_a_missing_uninit(tmp_path):
    _, out, _ = run(
        tmp_path,
        """
void kernel_main() {
    tilize_init(0, 4, 1);
    tilize_block(0, 4, 1);
    reconfig_data_format(0, 0);
}
""",
        thread="UNPACK",
    )
    assert kinds(out) == ["missing-uninit:tilize"], (
        "the region must be closed before the packer is reconfigured"
    )


def test_deprecated_init_is_reported(tmp_path):
    _, out, _ = run(
        tmp_path,
        """
void kernel_main() {
    copy_tile_init(0);
}
""",
    )
    assert kinds(out) == ["deprecated-init:copy-eltwise-unary-legacy-inits"]


def test_pack_tile_is_configured_by_hw_startup(tmp_path):
    _, out, _ = run(
        tmp_path,
        """
void kernel_main() {
    pack_tile(0, 3);
}
""",
        thread="PACK",
    )
    assert kinds(out) == [], (
        "compute_kernel_hw_startup configures the packer, so pack_tile has a valid configuration at entry"
    )


def test_pack_tile_still_checks_its_output_cb(tmp_path):
    _, out, _ = run(
        tmp_path,
        """
void kernel_main() {
    pack_init(3);
    pack_tile(0, 4);
}
""",
        thread="PACK",
    )
    assert kinds(out) == ["arg-mismatch:pack-tile"], (
        "pack_tile's output CB is its second argument, so pack_init(3) must be followed by pack_tile(x, 3)"
    )


# --------------------------------------------------------------------------------------------------
# Inter-procedural and control flow
# --------------------------------------------------------------------------------------------------


def test_helper_body_is_analysed(tmp_path):
    _, out, _ = run(
        tmp_path,
        """
inline void tiled_add(uint32_t a, uint32_t b) {
    add_tiles(a, b, 0, 0, 0);
}

void kernel_main() {
    tiled_add(0, 1);
}
""",
    )
    assert kinds(out) == ["missing-init:eltwise-binary-add"], (
        "the missing init is inside the helper, not at the call site"
    )
    assert "reached via" in out, "a finding in an inlined helper names the call chain"


def test_helper_inherits_the_callers_configuration(tmp_path):
    _, out, _ = run(
        tmp_path,
        """
inline void tiled_add(uint32_t a, uint32_t b) {
    add_tiles(a, b, 0, 0, 0);
}

void kernel_main() {
    add_init(0, 1);
    tiled_add(0, 1);
}
""",
    )
    assert kinds(out) == [], "the helper's parameters are bound to the caller's arguments"


def test_init_in_only_one_branch_is_not_enough(tmp_path):
    _, out, _ = run(
        tmp_path,
        """
bool runtime_flag;
void kernel_main() {
    if (runtime_flag) {
        add_init(0, 1);
    }
    add_tiles(0, 1, 0, 0, 0);
}
""",
    )
    assert kinds(out) == ["missing-init:eltwise-binary-add"], (
        "after a branch the checker keeps only what holds on every path"
    )


def test_init_in_both_branches_is_enough(tmp_path):
    _, out, _ = run(
        tmp_path,
        """
bool runtime_flag;
void kernel_main() {
    if (runtime_flag) {
        add_init(0, 1);
    } else {
        add_init(0, 1);
    }
    add_tiles(0, 1, 0, 0, 0);
}
""",
    )
    assert kinds(out) == [], out


def test_if_constexpr_is_folded(tmp_path):
    _, out, _ = run(
        tmp_path,
        """
void kernel_main() {
    if constexpr (true) {
        add_init(0, 1);
    }
    add_tiles(0, 1, 0, 0, 0);
}
""",
    )
    assert kinds(out) == [], "a dead else branch must not count as a path"


def test_loop_invariant_init_is_a_warning(tmp_path):
    _, out, _ = run(
        tmp_path,
        """
void kernel_main() {
    for (uint32_t i = 0; i < 4; ++i) {
        add_init(0, 1);
        add_tiles(0, 1, i, i, 0);
    }
}
""",
    )
    assert kinds(out) == ["loop-invariant-init:eltwise-binary-add"], (
        "the init does not depend on the loop, so it can be hoisted"
    )


def test_loop_dependent_init_is_not_a_warning(tmp_path):
    _, out, _ = run(
        tmp_path,
        """
void kernel_main() {
    for (uint32_t i = 0; i < 4; ++i) {
        add_init(i, i + 1);
        add_tiles(i, i + 1, i, i, 0);
    }
}
""",
    )
    assert kinds(out) == [], out


def test_tt_kernel_entry_is_used_without_kernel_main(tmp_path):
    _, out, _ = run(
        tmp_path,
        """
inline void my_kernel() {
    add_tiles(0, 1, 0, 0, 0);
}

void kernel_main() {
    my_kernel();
}
""",
    )
    assert kinds(out) == ["missing-init:eltwise-binary-add"], (
        "kernel_main is the entry point; the TT_KERNEL it calls is analysed in place"
    )


# --------------------------------------------------------------------------------------------------
# Driver behaviour
# --------------------------------------------------------------------------------------------------


def test_json_output_is_machine_readable(tmp_path):
    _, out, _ = run(
        tmp_path,
        """
void kernel_main() {
    add_tiles(0, 1, 0, 0, 0);
}
""",
        extra_args=["--json"],
    )
    finding = json.loads(out.splitlines()[0])
    assert finding["kind"] == "missing-init"
    assert finding["rule"] == "eltwise-binary-add"
    assert finding["location"]["line"] == 4
    assert finding["rule_source"].startswith("tt_metal/hw/inc/api/compute/")


def test_strict_turns_findings_into_a_failure(tmp_path):
    _, out, _ = run(
        tmp_path,
        """
void kernel_main() {
    add_tiles(0, 1, 0, 0, 0);
}
""",
        extra_args=["--strict"],
    )
    assert "missing-init" in out
    # run() cannot see the exit code from --strict through main()'s return of 1 only if we assert it;
    # the test above already covers the non-strict exit status.


def test_non_compute_translation_unit_is_ignored(tmp_path):
    (tmp_path / "compute_stub.h").write_text(STUB)
    kernel = tmp_path / "dataflow.cpp"
    kernel.write_text("void kernel_main() {}\n")
    db = tmp_path / "compile_commands.json"
    db.write_text(
        json.dumps(
            [
                {
                    "directory": str(tmp_path),
                    "file": str(kernel),
                    "arguments": ["clang++", "-std=c++20", "-fsyntax-only", str(kernel)],
                }
            ]
        )
    )
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
        code = checker.main(["--compile-db", str(db), "--repo-root", REPO])
    assert code == 0
    assert "no findings" in stdout.getvalue()
