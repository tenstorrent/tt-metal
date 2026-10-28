#!/usr/bin/env python3
"""Tests for the format-field mask checker."""
import contextlib
import io
import os
import runpy
import sys
import textwrap
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
INFRA = os.path.normpath(os.path.join(HERE, ".."))
SCRIPT = os.path.join(INFRA, "check_format_field_mask.py")

# The Wormhole / Blackhole layout of ALU word 1: SrcA 17..20, SrcB 21..24, Dstacc 25..28, INT8 31.
DEFINES = """
#define ALU_FORMAT_SPEC_REG0_SrcA_ADDR32 1
#define ALU_FORMAT_SPEC_REG0_SrcA_SHAMT 17
#define ALU_FORMAT_SPEC_REG0_SrcA_MASK 0x1e0000
#define ALU_FORMAT_SPEC_REG1_SrcB_ADDR32 1
#define ALU_FORMAT_SPEC_REG1_SrcB_SHAMT 21
#define ALU_FORMAT_SPEC_REG1_SrcB_MASK 0x1e00000
#define ALU_FORMAT_SPEC_REG2_Dstacc_ADDR32 1
#define ALU_FORMAT_SPEC_REG2_Dstacc_SHAMT 25
#define ALU_FORMAT_SPEC_REG2_Dstacc_MASK 0x1e000000
#define ALU_ACC_CTRL_INT8_math_enabled_ADDR32 1
#define ALU_ACC_CTRL_INT8_math_enabled_SHAMT 31
#define ALU_ACC_CTRL_INT8_math_enabled_MASK 0x80000000
#define PCK0_ADDR_CTRL_XY_REG_0_Xstride_SHAMT 0
#define PCK0_ADDR_CTRL_XY_REG_0_Xstride_MASK 0xffff
#define PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT 16
#define PCK0_ADDR_CTRL_XY_REG_0_Ystride_MASK 0xffff0000
"""


def run(*paths, repo=None):
    """Run the checker as a script (`__main__`, as the hook does), in-process."""
    argv = [SCRIPT, *map(str, paths)]
    if repo:
        argv += ["--repo", str(repo)]
    out, err, code = io.StringIO(), io.StringIO(), 0
    saved = sys.argv
    sys.argv = argv
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            runpy.run_path(SCRIPT, run_name="__main__")
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    finally:
        sys.argv = saved
    return types.SimpleNamespace(
        returncode=code, stdout=out.getvalue() + err.getvalue()
    )


@pytest.fixture
def repo(tmp_path):
    """A repo root with both arches' cfg_defines.h at the paths the checker expects."""
    for rel in (
        "tt_metal/hw/inc/internal/tt-1xx/wormhole/wormhole_b0_defines/cfg_defines.h",
        "tt_metal/hw/inc/internal/tt-1xx/blackhole/cfg_defines.h",
    ):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(DEFINES)
    return tmp_path


def hdr(repo, body, name="llk_math_common.h", arch="wormhole_b0"):
    d = repo / "tt_metal" / "tt-llk" / f"tt_llk_{arch}" / "llk_lib"
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text("namespace ckernel {\n" + textwrap.dedent(body) + "\n}\n")
    return p


def fn(
    body,
    params="const std::uint32_t srca_data_format, const std::uint32_t srcb_data_format",
):
    return f"inline void _llk_math_hw_configure_({params})\n{{\n{textwrap.indent(textwrap.dedent(body), '    ')}\n}}\n"


TWO_FIELD_MASK = "constexpr std::uint32_t config_mask = ALU_FORMAT_SPEC_REG0_SrcA_MASK | ALU_FORMAT_SPEC_REG1_SrcB_MASK;"


# ---- findings ------------------------------------------------------------------------------


def test_two_field_rmw_with_raw_srca_fails(repo):
    p = hdr(
        repo,
        fn(
            f"""
            std::uint32_t config_data = (srca_data_format << ALU_FORMAT_SPEC_REG0_SrcA_SHAMT) |
                                        (srcb_data_format << ALU_FORMAT_SPEC_REG1_SrcB_SHAMT);
            {TWO_FIELD_MASK}
            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, 0, config_mask>(config_data);
            """
        ),
    )
    r = run(p, repo=repo)
    assert r.returncode == 1
    assert (
        "`srca_data_format` is shifted unmasked into the 4-bit field ALU_FORMAT_SPEC_REG0_SrcA"
        in r.stdout
    )
    assert "mask 0x01fe0000" in r.stdout
    # SrcB's spill lands on Dstacc, outside this mask: not a finding.
    assert "`srcb_data_format`" not in r.stdout
    assert "1 finding(s)" in r.stdout


def test_three_field_mask_with_int8_bit_fails(repo):
    p = hdr(
        repo,
        fn(
            """
            std::uint32_t int8_math_enabled = 1;
            std::uint32_t config_data = (srca_data_format << ALU_FORMAT_SPEC_REG0_SrcA_SHAMT) |
                                        (srcb_data_format << ALU_FORMAT_SPEC_REG1_SrcB_SHAMT) |
                                        (int8_math_enabled << ALU_ACC_CTRL_INT8_math_enabled_SHAMT);
            constexpr std::uint32_t config_mask =
                ALU_FORMAT_SPEC_REG0_SrcA_MASK | ALU_FORMAT_SPEC_REG1_SrcB_MASK | ALU_ACC_CTRL_INT8_math_enabled_MASK;
            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, 0, config_mask>(config_data);
            """
        ),
    )
    r = run(p, repo=repo)
    assert r.returncode == 1
    assert "mask 0x81fe0000" in r.stdout
    assert "1 finding(s)" in r.stdout


def test_inline_argument_and_inline_mask_fail(repo):
    p = hdr(
        repo,
        fn(
            """
            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, 0, ALU_FORMAT_SPEC_REG0_SrcA_MASK | ALU_FORMAT_SPEC_REG1_SrcB_MASK>(
                (srca_data_format << ALU_FORMAT_SPEC_REG0_SrcA_SHAMT) | (srcb_data_format << ALU_FORMAT_SPEC_REG1_SrcB_SHAMT));
            """
        ),
    )
    r = run(p, repo=repo)
    assert r.returncode == 1
    assert "`srca_data_format`" in r.stdout


def test_full_word_store_fails_for_every_spill(repo):
    p = hdr(
        repo,
        fn(
            """
            cfg[ALU_FORMAT_SPEC_REG0_SrcA_ADDR32] = (srca_data_format << ALU_FORMAT_SPEC_REG0_SrcA_SHAMT) |
                                                    (srcb_data_format << ALU_FORMAT_SPEC_REG1_SrcB_SHAMT);
            """
        ),
    )
    r = run(p, repo=repo)
    assert r.returncode == 1
    assert "full-word write" in r.stdout
    assert "`srca_data_format`" in r.stdout and "`srcb_data_format`" in r.stdout
    assert "2 finding(s)" in r.stdout


def test_temporary_assigned_in_two_branches_reports_each_write_once(repo):
    p = hdr(
        repo,
        fn(
            f"""
            if (srca_data_format == 14)
            {{
                std::uint32_t config_data = (srca_data_format << ALU_FORMAT_SPEC_REG0_SrcA_SHAMT) |
                                            (srcb_data_format << ALU_FORMAT_SPEC_REG1_SrcB_SHAMT);
                {TWO_FIELD_MASK}
                cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, 0, config_mask>(config_data);
            }}
            else
            {{
                std::uint32_t config_data = (srca_data_format << ALU_FORMAT_SPEC_REG0_SrcA_SHAMT) |
                                            (srcb_data_format << ALU_FORMAT_SPEC_REG1_SrcB_SHAMT);
                {TWO_FIELD_MASK}
                cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, 0, config_mask>(config_data);
            }}
            """
        ),
    )
    r = run(p, repo=repo)
    assert r.returncode == 1
    assert "2 finding(s)" in r.stdout


def test_blackhole_tree_and_ckernels_tree_are_scanned(repo):
    body = fn(
        f"""
        std::uint32_t config_data = (srca_data_format << ALU_FORMAT_SPEC_REG0_SrcA_SHAMT) |
                                    (srcb_data_format << ALU_FORMAT_SPEC_REG1_SrcB_SHAMT);
        {TWO_FIELD_MASK}
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, 0, config_mask>(config_data);
        """
    )
    bh = hdr(repo, body, arch="blackhole")
    ck = (
        repo
        / "tt_metal"
        / "hw"
        / "ckernels"
        / "wormhole_b0"
        / "metal"
        / "llk_api"
        / "x.h"
    )
    ck.parent.mkdir(parents=True)
    ck.write_text("namespace ckernel {\n" + body + "\n}\n")
    r = run(bh, ck, repo=repo)
    assert r.returncode == 1
    assert "2 finding(s)" in r.stdout


# ---- not findings ----------------------------------------------------------------------------


def test_masked_data_format_at_the_shift_passes(repo):
    p = hdr(
        repo,
        fn(
            f"""
            std::uint32_t config_data = (masked_data_format(srca_data_format) << ALU_FORMAT_SPEC_REG0_SrcA_SHAMT) |
                                        (masked_data_format(srcb_data_format) << ALU_FORMAT_SPEC_REG1_SrcB_SHAMT);
            {TWO_FIELD_MASK}
            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, 0, config_mask>(config_data);
            """
        ),
    )
    r = run(p, repo=repo)
    assert r.returncode == 0, r.stdout


def test_helper_packing_the_fields_passes(repo):
    # The 60961 shape: a constexpr helper masks and packs; the writer only ORs the INT8 bit in.
    p = hdr(
        repo,
        """
        inline constexpr std::uint32_t alu_src_format_cfg(const std::uint32_t srca_data_format, const std::uint32_t srcb_data_format)
        {
            return (masked_data_format(srca_data_format) << ALU_FORMAT_SPEC_REG0_SrcA_SHAMT) |
                   (masked_data_format(srcb_data_format) << ALU_FORMAT_SPEC_REG1_SrcB_SHAMT);
        }
        """
        + fn(
            """
            std::uint32_t int8_math_enabled = 1;
            std::uint32_t config_data = alu_src_format_cfg(srca_data_format, srcb_data_format) |
                                        (int8_math_enabled << ALU_ACC_CTRL_INT8_math_enabled_SHAMT);
            constexpr std::uint32_t config_mask =
                ALU_FORMAT_SPEC_REG0_SrcA_MASK | ALU_FORMAT_SPEC_REG1_SrcB_MASK | ALU_ACC_CTRL_INT8_math_enabled_MASK;
            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, 0, config_mask>(config_data);
            """
        ),
    )
    r = run(p, repo=repo)
    assert r.returncode == 0, r.stdout


@pytest.mark.parametrize(
    "definition",
    [
        "const std::uint32_t srca_format_masked = masked_data_format(srca_data_format);",
        "const std::uint32_t srca_format_masked = srca_data_format & DATA_FORMAT_CONFIG_MASK;",
        "const std::uint32_t srca_format_masked = srca_data_format & 0xF;",
    ],
)
def test_masked_temporary_passes(repo, definition):
    p = hdr(
        repo,
        fn(
            f"""
            {definition}
            std::uint32_t config_data = (srca_format_masked << ALU_FORMAT_SPEC_REG0_SrcA_SHAMT) |
                                        (masked_data_format(srcb_data_format) << ALU_FORMAT_SPEC_REG1_SrcB_SHAMT);
            {TWO_FIELD_MASK}
            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, 0, config_mask>(config_data);
            """
        ),
    )
    r = run(p, repo=repo)
    assert r.returncode == 0, r.stdout


def test_single_field_rmw_passes(repo):
    # The register-side mask of a single-field RMW drops the spilled bit.
    p = hdr(
        repo,
        fn(
            """
            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(srca_data_format);
            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG1_SrcB_RMW>(srcb_data_format);
            """
        ),
    )
    r = run(p, repo=repo)
    assert r.returncode == 0, r.stdout


def test_mask_not_covering_the_spill_bit_passes(repo):
    # SrcA | INT8: SrcA's spill lands on bit 21, which this mask does not cover.
    p = hdr(
        repo,
        fn(
            """
            std::uint32_t int8_math_enabled = 1;
            std::uint32_t config_data = (srca_data_format << ALU_FORMAT_SPEC_REG0_SrcA_SHAMT) |
                                        (int8_math_enabled << ALU_ACC_CTRL_INT8_math_enabled_SHAMT);
            constexpr std::uint32_t config_mask = ALU_FORMAT_SPEC_REG0_SrcA_MASK | ALU_ACC_CTRL_INT8_math_enabled_MASK;
            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, 0, config_mask>(config_data);
            """
        ),
    )
    r = run(p, repo=repo)
    assert r.returncode == 0, r.stdout


def test_non_format_operand_passes(repo):
    # Pack strides are not DataFormat values: nothing says their domain exceeds the field.
    p = hdr(
        repo,
        fn(
            """
            std::uint32_t xy_stride = (x_stride << PCK0_ADDR_CTRL_XY_REG_0_Xstride_SHAMT) | (y_stride << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT);
            cfg[PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32] = xy_stride;
            """,
            params="const std::uint32_t x_stride, const std::uint32_t y_stride",
        ),
    )
    r = run(p, repo=repo)
    assert r.returncode == 0, r.stdout


def test_constant_operand_passes(repo):
    p = hdr(
        repo,
        fn(
            f"""
            constexpr std::uint32_t FORMAT_FP16B = 5;
            std::uint32_t config_data = (FORMAT_FP16B << ALU_FORMAT_SPEC_REG0_SrcA_SHAMT) |
                                        (to_underlying(DataFormat::Float16_b) << ALU_FORMAT_SPEC_REG1_SrcB_SHAMT);
            {TWO_FIELD_MASK}
            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, 0, config_mask>(config_data);
            """
        ),
    )
    r = run(p, repo=repo)
    assert r.returncode == 0, r.stdout


def test_unresolvable_mask_is_not_a_finding(repo):
    p = hdr(
        repo,
        fn(
            """
            std::uint32_t config_data = (srca_data_format << ALU_FORMAT_SPEC_REG0_SrcA_SHAMT) |
                                        (srcb_data_format << ALU_FORMAT_SPEC_REG1_SrcB_SHAMT);
            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, 0, some_runtime_mask(srca_data_format)>(config_data);
            """
        ),
    )
    r = run(p, repo=repo)
    assert r.returncode == 0, r.stdout


def test_shifted_rmw_form_is_not_resolved(repo):
    # A non-zero template shift moves every field; the plain-0 form is the only one judged.
    p = hdr(
        repo,
        fn(
            """
            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, ALU_FORMAT_SPEC_REG0_SrcA_SHAMT, 0x1fe0000>(
                srca_data_format | (srcb_data_format << ALU_FORMAT_SPEC_REG1_SrcB_SHAMT));
            """
        ),
    )
    r = run(p, repo=repo)
    assert r.returncode == 0, r.stdout


def test_shift_inside_a_comment_is_ignored(repo):
    p = hdr(
        repo,
        fn(
            """
            // (srca_data_format << ALU_FORMAT_SPEC_REG0_SrcA_SHAMT) | (srcb_data_format << ALU_FORMAT_SPEC_REG1_SrcB_SHAMT)
            cfg[ALU_FORMAT_SPEC_REG0_SrcA_ADDR32] = 0;
            """
        ),
    )
    r = run(p, repo=repo)
    assert r.returncode == 0, r.stdout


def test_quasar_and_unknown_trees_are_skipped(repo):
    body = fn(
        """
        cfg[ALU_FORMAT_SPEC_REG0_SrcA_ADDR32] = (srca_data_format << ALU_FORMAT_SPEC_REG0_SrcA_SHAMT) |
                                                (srcb_data_format << ALU_FORMAT_SPEC_REG1_SrcB_SHAMT);
        """
    )
    q = hdr(repo, body, arch="quasar")
    other = repo / "ttnn" / "x.h"
    other.parent.mkdir(parents=True)
    other.write_text(body)
    r = run(q, other, repo=repo)
    assert r.returncode == 0, r.stdout


def test_missing_defines_is_an_error(tmp_path):
    p = hdr(
        tmp_path, fn("cfg[1] = srca_data_format << ALU_FORMAT_SPEC_REG0_SrcA_SHAMT;")
    )
    r = run(p, repo=tmp_path)
    assert r.returncode == 1
    assert "cfg_defines.h not found" in r.stdout


# ---- scope -----------------------------------------------------------------------------------


def test_touching_the_checker_scans_the_whole_tree():
    # Passing only the checker's own path (the hook does this when the checker changes) must
    # scan the shipped trees exactly as a run with no files does.
    assert run(SCRIPT).stdout == run().stdout


def test_real_trees_report_only_known_sites():
    """Every finding in the shipped trees is a math-side ALU format writer on Wormhole.

    Until the fix for those writers merges this lists the Wormhole math ALU writes; after it, nothing.
    Either way no other file may appear: that would be a new site or a false positive.
    """
    r = run()
    for line in r.stdout.splitlines():
        if ":" in line and line[0] != " " and "finding(s)" not in line:
            assert line.startswith(
                "tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_common.h:"
            ), line
            assert "ALU_FORMAT_SPEC_REG0_SrcA" in line, line
