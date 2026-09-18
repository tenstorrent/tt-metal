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
#define WIDE_BASE_Word0_ADDR32 0
#define WIDE_BASE_Word0_SHAMT 0
#define WIDE_BASE_Word0_MASK 0xf
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


# --- addressing: the three pitfalls the module docstring calls out -------------------------------


def _math_owns_word(d, field="ALU_FORMAT_SPEC_REG0_SrcA"):
    write(
        d,
        "llk_math_owner.h",
        f"""
        inline void _llk_math_owner_() {{
            cfg_reg_rmw_tensix<{field}_RMW>(v);
        }}
        """,
    )


def test_wrcfg_128b_spans_four_words(tree):
    """WRCFG_128b writes an aligned group of FOUR words; a 32b write to the same base writes one."""
    tmp, d = tree
    # Word 2 is MATH-owned; the 128b write is based at word 0, so its span reaches it.
    _math_owns_word(d, "ALU_ACC_CTRL_Zero_Flag_disabled_src")
    write(
        d,
        "llk_pack_wide.h",
        """
        inline void _llk_pack_wide_() {
            TTI_WRCFG(p_gpr::TMP0, p_cfg::WRCFG_128b, WIDE_BASE_Word0_ADDR32);
        }
        """,
    )
    assert "CLOBBER" in run(tmp).stdout, "the 128b span must reach word 2"

    (d / "llk_pack_wide.h").unlink()
    write(
        d,
        "llk_pack_narrow.h",
        """
        inline void _llk_pack_narrow_() {
            TTI_WRCFG(p_gpr::TMP0, p_cfg::WRCFG_32b, WIDE_BASE_Word0_ADDR32);
        }
        """,
    )
    r = run(tmp)
    assert (
        r.returncode == 0
    ), f"a 32b write at word 0 must not reach word 2:\n{r.stdout}"


def test_literal_offset_is_added_to_the_base_word(tree):
    """`cfg[BASE + 1]` targets the next word, not the base and not 'unresolved'."""
    tmp, d = tree
    _math_owns_word(d)  # owns bits in word 1
    write(
        d,
        "llk_pack_off.h",
        """
        inline void _llk_pack_off_() {
            cfg[WIDE_BASE_Word0_ADDR32 + 1] = v;
        }
        """,
    )
    r = run(tmp)
    assert "CLOBBER" in r.stdout, f"BASE + 1 must resolve to word 1:\n{r.stdout}"
    assert "UNRESOLVED" not in r.stdout, r.stdout


def test_variable_offset_is_reported_unresolved(tree):
    """`cfg[BASE + i]` is not statically known: report it, never silently treat it as safe."""
    tmp, d = tree
    _math_owns_word(d, "WIDE_BASE_Word0")  # owns bits in word 0
    write(
        d,
        "llk_pack_var.h",
        """
        inline void _llk_pack_var_() {
            cfg[WIDE_BASE_Word0_ADDR32 + i] = v;
        }
        """,
    )
    r = run(tmp)
    assert "UNRESOLVED" in r.stdout, f"a variable offset must be reported:\n{r.stdout}"


# --- write mechanisms ----------------------------------------------------------------------------


def test_literal_mode_and_mop_word_wrcfg_are_seen(tree):
    """The mode is spelled as a bare bit at many sites, and TT_OP_WRCFG builds an executing MOP word."""
    tmp, d = tree
    _math_owns_word(d, "ALU_ACC_CTRL_Zero_Flag_disabled_src")  # word 2
    write(
        d,
        "llk_pack_literal.h",
        """
        inline void _llk_pack_literal_() {
            TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, STACC_RELU_ApplyRelu_ADDR32);
        }
        """,
    )
    assert "CLOBBER" in run(tmp).stdout, "literal mode 0 is WRCFG_32b"

    (d / "llk_pack_literal.h").unlink()
    write(
        d,
        "llk_pack_mop.h",
        """
        inline void _llk_pack_mop_() {
            static constexpr std::uint32_t w =
                TT_OP_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, STACC_RELU_ApplyRelu_ADDR32);
        }
        """,
    )
    assert "CLOBBER" in run(tmp).stdout, "a WRCFG built into a MOP word still executes"


def test_mask_resolves_to_the_nearest_preceding_definition(tree):
    """`config_mask` is redeclared per function; the one above the write governs, not the file's first."""
    tmp, d = tree
    _math_owns_word(d)  # MATH masks the SrcA bits of word 1
    write(
        d,
        "llk_unpack_masks.h",
        """
        inline void _llk_unpack_first_() {
            const std::uint32_t config_mask = ALU_FORMAT_SPEC_REG0_SrcAUnsigned_MASK;
            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, SH, config_mask>(v);
        }

        inline void _llk_unpack_second_() {
            const std::uint32_t config_mask = ALU_FORMAT_SPEC_REG0_SrcA_MASK;
            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, SH, config_mask>(v);
        }
        """,
    )
    r = run(tmp)
    # Resolving the second write against the first function's mask reads it as the disjoint
    # Unsigned bit and the SAME-FIELD race disappears.
    assert "SAME-FIELD" in r.stdout, r.stdout


# --- unclassifiable threads ----------------------------------------------------------------------


def test_unknown_thread_is_reported_as_a_note_and_does_not_fail(tree):
    """A write whose thread cannot be determined is surfaced, but never fails the commit.

    thread_of() keys on the llk_unpack/llk_math/llk_pack file and function naming. A write
    dispatched by a ThreadId template parameter, or one in a shared header, matches neither and
    used to be dropped silently -- the one outcome the rest of this checker is written to avoid.
    It is a gap in the tool's coverage, not evidence of a hazard, so it reports and exits clean.
    """
    tmp, d = tree
    _math_owns_word(d)  # MATH owns the SrcA bits of word 1
    write(
        d,
        "shared_helper.h",
        """
        inline void set_dest_acc_by_dispatch() {
            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(v);
        }
        """,
    )
    r = run(tmp)
    assert (
        r.returncode == 0
    ), f"an unclassified write must not fail the commit:\n{r.stdout}"
    assert "whose thread could not be determined" in r.stdout, r.stdout
    assert "shared_helper.h" in r.stdout, r.stdout


def test_unknown_thread_does_not_become_an_owner(tree):
    """The unclassified write must not be attributed to a thread of its own.

    If it entered the ownership map under a null thread, every genuine write to the same word
    would see it as another thread's claim and report against it -- turning a coverage gap into
    a wave of false findings.
    """
    tmp, d = tree
    write(
        d,
        "llk_math_owner.h",
        """
        inline void _llk_math_owner_() {
            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(v);
        }
        """,
    )
    write(
        d,
        "shared_helper.h",
        """
        inline void set_by_dispatch() {
            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(v);
        }
        """,
    )
    r = run(tmp)
    assert r.returncode == 0, r.stdout
    assert (
        "SAME-FIELD" not in r.stdout
    ), f"the note must not create a finding:\n{r.stdout}"
