#!/usr/bin/env python3
"""Tests for the cross-thread config-write checker.

Each test encodes a mistake the checker made during development, so a regression is caught rather
than rediscovered. Run: python3 -m pytest tt_metal/tt-llk/infra/tests/ -q
"""
import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "check_cross_thread_cfg.py")

spec = importlib.util.spec_from_file_location("chk", SCRIPT)
chk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chk)

# A miniature cfg_defines.h. Word 2 deliberately mirrors the real hazard: a PACK-owned field
# (STACC_RELU) sharing a word with a MATH-owned one (Zero_Flag_disabled_src).
DEFINES = """
#define ALU_ACC_CTRL_Zero_Flag_disabled_src_ADDR32 2
#define ALU_ACC_CTRL_Zero_Flag_disabled_src_SHAMT 0
#define ALU_ACC_CTRL_Zero_Flag_disabled_src_MASK 0x1
#define STACC_RELU_ApplyRelu_ADDR32 2
#define STACC_RELU_ApplyRelu_SHAMT 2
#define STACC_RELU_ApplyRelu_MASK 0x3c
#define ALU_FORMAT_SPEC_REG0_SrcA_ADDR32 1
#define ALU_FORMAT_SPEC_REG0_SrcA_SHAMT 17
#define ALU_FORMAT_SPEC_REG0_SrcA_MASK 0x1e0000
#define ALU_FORMAT_SPEC_REG0_SrcAUnsigned_ADDR32 1
#define ALU_FORMAT_SPEC_REG0_SrcAUnsigned_SHAMT 15
#define ALU_FORMAT_SPEC_REG0_SrcAUnsigned_MASK 0x8000
"""


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "cfg_defines.h").write_text(DEFINES)
    d = tmp_path / "llk_lib"
    d.mkdir()
    return tmp_path, d


def run(tmp_path, arch="wormhole_b0", extra=()):
    return subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--defs",
            str(tmp_path / "cfg_defines.h"),
            "--tree",
            str(tmp_path),
            "--arch",
            arch,
            *extra,
        ],
        capture_output=True,
        text=True,
    )


def write(d, name, body):
    (d / name).write_text(textwrap.dedent(body))


def test_defines_parse():
    """_MASK is already shifted into position; SHAMT must not be applied again."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".h", delete=False) as f:
        f.write(DEFINES)
    defs = chk.load_defs(f.name)
    assert defs["ALU_FORMAT_SPEC_REG0_SrcA"] == (1, 17, 0x1E0000)


def test_whole_word_write_clobbering_another_threads_field(tree):
    """The relu-class bug: PACK whole-word write destroys a MATH-owned field in the same word."""
    tmp, d = tree
    write(
        d,
        "llk_math_x.h",
        """
        inline void _llk_math_a_() {
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1);
        }
        """,
    )
    write(
        d,
        "llk_pack_x.h",
        """
        inline void _llk_pack_b_() {
            TTI_WRCFG(p_gpr::TMP0, p_cfg::WRCFG_32b, STACC_RELU_ApplyRelu_ADDR32);
        }
        """,
    )
    r = run(tmp)
    assert "CLOBBER" in r.stdout, r.stdout
    assert r.returncode == 1


def test_disjoint_bits_same_word_are_safe(tree):
    """Different bits of one word from two threads: safe, the config RMW is per-byte atomic.

    This is the false positive that made a moved-to-MATH field look like a race: the unpack side
    addresses the word by the SrcA name but masks only the Unsigned bit.
    """
    tmp, d = tree
    write(
        d,
        "llk_math_x.h",
        """
        inline void _llk_math_a_() {
            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(fmt);
        }
        """,
    )
    write(
        d,
        "llk_unpack_x.h",
        """
        inline void _llk_unpack_b_() {
            constexpr std::uint32_t alu_mask = ALU_FORMAT_SPEC_REG0_SrcAUnsigned_MASK;
            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, 0, alu_mask>(v);
        }
        """,
    )
    r = run(tmp)
    assert r.returncode == 0, r.stdout


def test_same_bits_two_threads_is_reported(tree):
    """Same field from two threads with no mutex: reported."""
    tmp, d = tree
    for fn, th in (("llk_math_x.h", "math"), ("llk_unpack_x.h", "unpack")):
        write(
            d,
            fn,
            f"""
            inline void _llk_{th}_a_() {{
                cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1);
            }}
            """,
        )
    r = run(tmp)
    assert "SAME-FIELD" in r.stdout, r.stdout


def test_mutex_is_tracked_by_state_not_by_line_distance(tree):
    """A protected write can sit far below its acquire; a fixed lookback window misses it.

    Both writers take the mutex here -- a mutex held by only one side protects nothing, so the
    one-sided version of this fixture is a DIFFERENT case (see the test below).
    """
    tmp, d = tree
    filler = "\n".join(f"    // filler {i}" for i in range(30))
    for fn, th in (("llk_math_x.h", "math"), ("llk_unpack_x.h", "unpack")):
        write(
            d,
            fn,
            f"""
            inline void _llk_{th}_a_() {{
                t6_mutex_acquire(mutex::REG_RMW);
            {filler}
                cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1);
                t6_mutex_release(mutex::REG_RMW);
            }}
            """,
        )
    r = run(tmp)
    assert "SAME-FIELD" not in r.stdout, r.stdout
    assert r.returncode == 0


def test_mutex_on_only_one_side_is_not_protection(tree):
    """One thread holding the mutex does not protect the thread that does not take it."""
    tmp, d = tree
    write(
        d,
        "llk_math_x.h",
        """
        inline void _llk_math_a_() {
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1);
        }
        """,
    )
    write(
        d,
        "llk_unpack_x.h",
        """
        inline void _llk_unpack_b_() {
            t6_mutex_acquire(mutex::REG_RMW);
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1);
            t6_mutex_release(mutex::REG_RMW);
        }
        """,
    )
    r = run(tmp)
    assert "SAME-FIELD" in r.stdout, r.stdout
    assert "llk_math_x.h" in r.stdout, "the unguarded side is the one to report"


def test_unsupported_arch_refuses_instead_of_reporting_clean(tree):
    """Quasar writes config via cfg_rmw/RMWCIB; a zero there would be vacuous, not clean."""
    tmp, d = tree
    write(d, "llk_math_x.h", "inline void _llk_math_a_() { }\n")
    r = run(tmp, arch="quasar")
    assert r.returncode == 2 and "REFUSING" in r.stdout, r.stdout


def test_vacuous_zero_refused_when_write_api_unrecognised(tree):
    """Headers present but no masked write recognised => the parser is blind; refuse."""
    tmp, d = tree
    write(
        d,
        "llk_math_x.h",
        """
        inline void _llk_math_a_() {
            cfg_rmw(SOME_OTHER_API, 1);
        }
        """,
    )
    r = run(tmp)
    assert r.returncode == 2 and "REFUSING" in r.stdout, r.stdout
