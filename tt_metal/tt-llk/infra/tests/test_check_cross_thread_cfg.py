#!/usr/bin/env python3
"""Tests for the cross-thread config-write checker.

Each test encodes a mistake the checker made during development, so a regression is caught rather
than rediscovered. Run: python3 -m pytest tt_metal/tt-llk/infra/tests/ -q
"""

import contextlib
import importlib.util
import io
import os
import runpy
import sys
import textwrap
import types

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
    """Run the checker as a script (`__main__`, as the hook does), in-process."""
    argv = [
        SCRIPT,
        "--defs",
        str(tmp_path / "cfg_defines.h"),
        "--tree",
        str(tmp_path),
        "--arch",
        arch,
        *extra,
    ]
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
        returncode=code, stdout=out.getvalue(), stderr=err.getvalue()
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
    # The guard 30 lines up is seen: MATH consumes the field, so its guarded write is sound and
    # not reported; UNPACK does not consume it, so its guarded write is -- as a guarded one.
    assert "llk_math_x.h" not in r.stdout, r.stdout
    assert "llk_unpack_x.h" in r.stdout and "under mutex::REG_RMW" in r.stdout, r.stdout


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
    # ...but the checker cannot tell whether it conflicts, so it must not block a commit
    assert r.returncode == 0 and "advisory" in r.stdout, r.stdout


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


# --- template arguments are read whole -----------------------------------------------------------
# In each case PACK writes bit 0 of word 2, which MATH owns (Zero_Flag_disabled_src), so the
# write must be judged against MATH's bits rather than dropped or read with the wrong mask.
MATH_OWNS_BIT0 = """
inline void _llk_math_owner_() {
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1);
}
"""


@pytest.mark.parametrize(
    "write_line",
    [
        # OR'd mask on one line: the address-only fallback used to record ApplyRelu's mask alone
        "cfg_reg_rmw_tensix<STACC_RELU_ApplyRelu_ADDR32, 0, STACC_RELU_ApplyRelu_MASK | ALU_ACC_CTRL_Zero_Flag_disabled_src_MASK>(v);",
        # the same call split across lines, as clang-format leaves the long ones
        "cfg_reg_rmw_tensix<\n    STACC_RELU_ApplyRelu_ADDR32,\n    0,\n    STACC_RELU_ApplyRelu_MASK | ALU_ACC_CTRL_Zero_Flag_disabled_src_MASK>(v);",
        # a plain `_MASK` macro as the mask argument
        "cfg_reg_rmw_tensix<STACC_RELU_ApplyRelu_ADDR32, 0, ALU_ACC_CTRL_Zero_Flag_disabled_src_MASK>(v);",
        # a literal offset on the address and a literal mask: word 1 + 1 is word 2
        "cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32 + 1, 0, 0x1>(v);",
        # a mask constant defined in another header of the tree
        "cfg_reg_rmw_tensix<STACC_RELU_ApplyRelu_ADDR32, 0, SHARED_BIT0_MASK>(v);",
    ],
    ids=[
        "or-mask",
        "multi-line",
        "field-mask-macro",
        "offset-and-literal",
        "tree-constant",
    ],
)
def test_masked_write_arguments_resolve(tree, write_line):
    tmp, d = tree
    write(d, "llk_math_x.h", MATH_OWNS_BIT0)
    write(d, "ckernel_consts.h", "constexpr std::uint32_t SHARED_BIT0_MASK = 0x1;\n")
    body = textwrap.indent(write_line, " " * 12)
    write(d, "llk_pack_x.h", f"\ninline void _llk_pack_b_() {{\n{body}\n}}\n")
    r = run(tmp)
    assert "SAME-FIELD" in r.stdout and "llk_pack_x.h" in r.stdout, r.stdout
    assert "UNRESOLVED" not in r.stdout, r.stdout


def test_ambiguous_tree_constant_stays_unresolved(tree):
    """A mask name given two values in the tree is not guessed."""
    tmp, d = tree
    write(d, "llk_math_x.h", MATH_OWNS_BIT0)
    write(d, "a.h", "constexpr std::uint32_t TWO_WAY_MASK = 0x1;\n")
    write(d, "b.h", "constexpr std::uint32_t TWO_WAY_MASK = 0x4;\n")
    write(
        d,
        "llk_pack_x.h",
        """
        inline void _llk_pack_b_() {
            cfg_reg_rmw_tensix<STACC_RELU_ApplyRelu_ADDR32, 0, TWO_WAY_MASK>(v);
        }
        """,
    )
    r = run(tmp)
    assert "UNRESOLVED" in r.stdout and "SAME-FIELD" not in r.stdout, r.stdout


@pytest.mark.parametrize(
    "write_line",
    [
        "cfg_rmw(STACC_RELU_ApplyRelu_RMW, v);",
        "cfg_rmw_gpr(STACC_RELU_ApplyRelu_ADDR32, STACC_RELU_ApplyRelu_SHAMT, STACC_RELU_ApplyRelu_MASK, p_gpr::TMP0);",
        "cfg_write(STACC_RELU_ApplyRelu_ADDR32, v);",
    ],
    ids=["cfg_rmw", "cfg_rmw_gpr", "cfg_write"],
)
def test_mmio_config_helpers_clobber_the_whole_word(tree, write_line):
    """On WH/BH these are a RISC load + full-word store: disjoint bits are NOT safe through them."""
    tmp, d = tree
    write(d, "llk_math_x.h", MATH_OWNS_BIT0)
    write(d, "llk_pack_x.h", f"inline void _llk_pack_b_() {{ {write_line} }}\n")
    r = run(tmp)
    assert "CLOBBER" in r.stdout and "llk_pack_x.h" in r.stdout, r.stdout


def test_mmio_helper_forwarding_its_own_parameters_is_not_a_write(tree):
    """`cfg_rmw_gpr` calls `cfg_rmw(cfg_addr32, ...)`: no field is named, so there is nothing to judge."""
    tmp, d = tree
    write(d, "llk_math_x.h", MATH_OWNS_BIT0)
    write(
        d,
        "llk_pack_x.h",
        """
        inline void _llk_pack_b_(std::uint32_t cfg_addr32) {
            cfg_rmw(cfg_addr32, cfg_shamt, cfg_mask, wrdata);
        }
        """,
    )
    assert "CLOBBER" not in run(tmp).stdout


def test_wrcfg_128b_unaligned_base_writes_the_aligned_group(tree):
    """128b mode writes Config[(CfgIndex & ~3) .. +3]: word 3 as a base covers words 0-3, not 3-6."""
    tmp, d = tree
    write(d, "llk_math_x.h", MATH_OWNS_BIT0)
    write(
        d,
        "llk_pack_x.h",
        """
        inline void _llk_pack_b_() {
            TTI_WRCFG(p_gpr::TMP0, p_cfg::WRCFG_128b, ALU_FORMAT_SPEC_REG0_SrcA_ADDR32 + 2);
        }
        """,
    )
    r = run(tmp)
    assert "CLOBBER" in r.stdout and "config word 2" in r.stdout, r.stdout


def test_sfpu_kernel_is_classified_as_math(tree):
    """SFPU instructions are issued by the math thread; its kernels carry no llk_math in the name."""
    tmp, d = tree
    sfpu = tmp / "common" / "inc" / "sfpu"
    sfpu.mkdir(parents=True)
    write(
        sfpu,
        "ckernel_sfpu_x.h",
        """
        inline void enter_block() {
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1);
        }
        """,
    )
    write(
        d,
        "llk_unpack_x.h",
        """
        inline void _llk_unpack_a_() {
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(0);
        }
        """,
    )
    r = run(tmp)
    assert "SAME-FIELD" in r.stdout and "MATH RMWs" in r.stdout, r.stdout
    assert "could not be determined" not in r.stdout, r.stdout
    # the thread is inferred from the file's location, so the finding is shown but never blocks
    assert r.returncode == 0 and "advisory" in r.stdout, r.stdout


def test_hpp_header_contributes_ownership(tree):
    """The hooks trigger on .hpp, so the ownership walk must read .hpp too."""
    tmp, d = tree
    write(d, "llk_math_x.hpp", MATH_OWNS_BIT0)
    write(
        d,
        "llk_pack_x.h",
        """
        inline void _llk_pack_b_() {
            TTI_WRCFG(p_gpr::TMP0, p_cfg::WRCFG_32b, STACC_RELU_ApplyRelu_ADDR32);
        }
        """,
    )
    assert "CLOBBER" in run(tmp).stdout


def test_thread_section_field_does_not_alias_into_the_consumer_note(tree):
    """ThreadConfig fields share numbers with Config words (both bases are 0); they are not
    other fields of the shared word."""
    tmp, d = tree
    (tmp / "cfg_defines.h").write_text(
        "// Registers for THREAD\n"
        "#define THREAD_ONLY_Field_ADDR32 2\n#define THREAD_ONLY_Field_SHAMT 0\n#define THREAD_ONLY_Field_MASK 0x100\n"
        "// Registers for ALU\n" + DEFINES
    )
    (tmp / "consumers.yaml").write_text(
        "wormhole_b0:\n  THREAD_ONLY_Field:\n  - UNPACK\n  ALU_ACC_CTRL_Zero_Flag_disabled_src:\n  - MATH\n"
    )
    write(d, "llk_math_x.h", MATH_OWNS_BIT0)
    write(
        d,
        "llk_pack_x.h",
        """
        inline void _llk_pack_b_() {
            TTI_WRCFG(p_gpr::TMP0, p_cfg::WRCFG_32b, STACC_RELU_ApplyRelu_ADDR32);
        }
        """,
    )
    r = run(tmp, extra=["--consumers", str(tmp / "consumers.yaml")])
    assert "CLOBBER" in r.stdout, r.stdout
    assert (
        "consumed by MATH" in r.stdout
        and "UNPACK" not in r.stdout.split("consumed by")[1]
    ), r.stdout


# --- the hook never blocks a commit on what it did not cause ---------------------------------------
PACK_CLOBBERS_WORD2 = """
inline void _llk_pack_b_() {
    TTI_WRCFG(p_gpr::TMP0, p_cfg::WRCFG_32b, STACC_RELU_ApplyRelu_ADDR32);
}
"""


def _debt_tree(tree):
    """A tree carrying an unbaselined CLOBBER: PACK whole-word-writes MATH's word 2."""
    tmp, d = tree
    write(d, "llk_math_x.h", MATH_OWNS_BIT0)
    write(d, "llk_pack_x.h", PACK_CLOBBERS_WORD2)
    write(d, "llk_unpack_y.h", "inline void _llk_unpack_y_() {}\n")
    return tmp, d


def test_unrelated_commit_is_not_blocked_by_existing_debt(tree):
    tmp, d = _debt_tree(tree)
    r = run(tmp, extra=[str(d / "llk_unpack_y.h")])
    assert r.returncode == 0 and "CLOBBER" not in r.stdout, r.stdout


def test_commit_is_answerable_for_the_file_it_touches(tree):
    tmp, d = _debt_tree(tree)
    r = run(tmp, extra=[str(d / "llk_pack_x.h")])
    assert r.returncode == 1 and "CLOBBER" in r.stdout, r.stdout


def test_commit_is_answerable_for_a_conflict_it_causes(tree):
    """Adding the MATH owner is what turns the untouched PACK write into a clobber."""
    tmp, d = _debt_tree(tree)
    r = run(tmp, extra=[str(d / "llk_math_x.h")])
    assert r.returncode == 1 and "llk_pack_x.h" in r.stdout, r.stdout


def test_touching_a_verdict_input_checks_the_whole_tree(tree):
    """Dropping a baseline entry without fixing the site must fail, even though no header changed."""
    tmp, d = _debt_tree(tree)
    baseline = tmp.parent / f"{tmp.name}_baseline.txt"
    baseline.write_text("# emptied\n")
    r = run(tmp, extra=["--baseline", str(baseline), str(baseline)])
    assert r.returncode == 1 and "CLOBBER" in r.stdout, r.stdout


def test_inferred_owner_never_blocks_a_certain_writer(tree):
    """UNPACK's write conflicts only with an SFPU (inferred MATH) owner: shown, not blocking."""
    tmp, d = tree
    sfpu = tmp / "common" / "inc" / "sfpu"
    sfpu.mkdir(parents=True)
    write(
        sfpu,
        "ckernel_sfpu_x.h",
        MATH_OWNS_BIT0.replace("_llk_math_owner_", "enter_block"),
    )
    write(
        d,
        "llk_unpack_x.h",
        """
        inline void _llk_unpack_a_() {
            cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(0);
        }
        """,
    )
    r = run(tmp)
    assert (
        r.returncode == 0 and "advisory" in r.stdout and "UNPACK RMWs" in r.stdout
    ), r.stdout


def test_masked_write_at_a_variable_offset_is_advisory(tree):
    tmp, d = tree
    write(d, "llk_math_x.h", MATH_OWNS_BIT0)
    write(
        d,
        "llk_pack_x.h",
        """
        template <std::uint32_t N>
        inline void _llk_pack_b_() {
            cfg_reg_rmw_tensix<STACC_RELU_ApplyRelu_ADDR32 + N, 0, 0x1>(v);
        }
        """,
    )
    r = run(tmp)
    assert (
        r.returncode == 0 and "UNRESOLVED" in r.stdout and "advisory" in r.stdout
    ), r.stdout


# --- a guarded write is sound only when its thread consumes the field ------------------------------
def _guarded_pair(d):
    for fn, th in (("llk_math_x.h", "math"), ("llk_unpack_x.h", "unpack")):
        write(
            d,
            fn,
            f"""
            inline void _llk_{th}_a_() {{
                t6_mutex_acquire(mutex::REG_RMW);
                cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1);
                t6_mutex_release(mutex::REG_RMW);
            }}
            """,
        )


def _consumers(tmp, readers):
    y = tmp.parent / f"{tmp.name}_consumers.yaml"
    body = "".join(f"  - {t}\n" for t in readers)
    y.write_text(
        "wormhole_b0:\n"
        + (
            f"  ALU_ACC_CTRL_Zero_Flag_disabled_src:\n{body}"
            if readers
            else "  Unrelated_Field:\n  - MATH\n"
        )
    )
    return ["--consumers", str(y)]


def test_guarded_write_by_a_non_consumer_is_blocking(tree):
    tmp, d = tree
    _guarded_pair(d)
    r = run(tmp, extra=_consumers(tmp, ["MATH"]))
    assert r.returncode == 1 and "llk_unpack_x.h" in r.stdout, r.stdout
    assert "llk_math_x.h:" not in r.stdout, "the consuming writer is sound"


def test_guarded_writes_by_consumers_are_not_reported(tree):
    tmp, d = tree
    _guarded_pair(d)
    r = run(tmp, extra=_consumers(tmp, ["MATH", "UNPACK"]))
    assert r.returncode == 0 and "SAME-FIELD" not in r.stdout, r.stdout


def test_guarded_write_with_no_recorded_reader_is_advisory(tree):
    """An unrecorded field is UNKNOWN, not 'safe' and not 'violated': shown, never blocking."""
    tmp, d = tree
    _guarded_pair(d)
    r = run(tmp, extra=_consumers(tmp, []))
    assert (
        r.returncode == 0 and "advisory" in r.stdout and "may not consume" in r.stdout
    ), r.stdout


def test_unclassified_note_is_scoped_to_the_commit(tree):
    """pre-commit runs the hook once per batch of files; a whole-tree note would repeat per batch."""
    tmp, d = tree
    write(d, "llk_math_x.h", MATH_OWNS_BIT0)
    write(
        d,
        "shared_helper.h",
        "inline void set_by_dispatch() { cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(v); }\n",
    )
    write(d, "llk_unpack_y.h", "inline void _llk_unpack_y_() {}\n")
    assert (
        "could not be determined"
        not in run(tmp, extra=[str(d / "llk_unpack_y.h")]).stdout
    )
    assert (
        "could not be determined" in run(tmp, extra=[str(d / "shared_helper.h")]).stdout
    )
    assert (
        "could not be determined" in run(tmp).stdout
    ), "a whole-tree run lists every one"
