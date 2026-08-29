# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Regression: the LLK test firmware must configure Blackhole instruction gathering
the same way tt-metal firmware does.

Blackhole boots with instruction gathering enabled. tt-metal firmware disables it
on every RISC around Tensix (``configure_gathering()`` in
``tt_metal/hw/inc/internal/firmware_common.h``). ``ckernel.h``'s
``load_replay_buf()`` is the complement: it brackets the replay-record window with
disable/enable, but only ``#if defined(ENABLE_GATHERING)``.

Those are two halves of one contract -- either gathering is off globally, or it is on
and the record window is bracketed. Before ``configure_gathering()`` was added to
``do_crt0()``, the test harness implemented neither half: it never defines
ENABLE_GATHERING (so no bracketing) and never wrote cfg0 (so no global disable), which
left the tests running in a configuration that never ships.

This test asserts on the built firmware rather than on behaviour, because the
gathering hazard is a codegen lottery: a kernel that happens to be laid out safely
passes with gathering on, so a passing functional test proves nothing about the
configuration. CSR 0x7c0 is write-only in this codebase (every use is
``csrrs zero``/``csrrc zero``), so it cannot be read back and checked at runtime.
"""

import struct

from conftest import blackhole_only
from helpers.format_config import DataFormat
from helpers.param_config import input_output_formats
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig

# RISC-V encodings of the two CSR ops in configure_gathering(). Both hard-code t1
# (x6) in the asm template, so the encodings are fixed:
#   csr[11:0]<<20 | rs1<<15 | funct3<<12 | rd<<7 | opcode(0x73)
CSRRS_ZERO_CFG0_T1 = 0x7C032073  # csrrs zero, 0x7c0, t1   (funct3=010)
CSRRC_ZERO_CFG0_T1 = 0x7C033073  # csrrc zero, 0x7c0, t1   (funct3=011)

# configure_gathering() issues csrrs twice (bit 1, then bit 18) and csrrc once (bit 1).
# Counted as a lower bound so an unrelated cfg0 write (tt-metal's
# configure_l1_data_cache() uses the same encoding) cannot cause a spurious failure.
# That tolerance alone would admit a false pass -- another writer could satisfy the
# bound while configure_gathering() itself had been dropped -- so the ordered triple
# must also appear inside one short window, which is what proves this sequence, and
# not merely some cfg0 traffic, reached the binary. The source-level sequence is
# pinned separately by test_harness_gathering_sequence_matches_tt_metal below.
MIN_CSRRS = 2
MIN_CSRRC = 1
# The three words span 7 words when emitted (one asm block, so the compiler cannot
# interleave); allow slack for scheduling changes without admitting unrelated writes.
SEQUENCE_WINDOW_WORDS = 16


def _executable_words(elf_path):
    """Every 4-byte aligned word of every executable PROGBITS section."""
    data = elf_path.read_bytes()
    assert data[:4] == b"\x7fELF" and data[4] == 1, f"{elf_path} is not a 32-bit ELF"

    (e_shoff,) = struct.unpack_from("<I", data, 0x20)
    e_shentsize, e_shnum, _ = struct.unpack_from("<HHH", data, 0x2E)

    words = []
    for i in range(e_shnum):
        _, sh_type, sh_flags, _, sh_offset, sh_size = struct.unpack_from(
            "<IIIIII", data, e_shoff + i * e_shentsize
        )
        SHT_PROGBITS, SHF_EXECINSTR = 1, 0x4
        if sh_type != SHT_PROGBITS or not (sh_flags & SHF_EXECINSTR):
            continue
        for off in range(sh_offset, sh_offset + sh_size - 3, 4):
            words.append(struct.unpack_from("<I", data, off)[0])
    return words


def _has_gathering_sequence(words):
    """True if csrrs, csrrs, csrrc appear in that order inside one short window.

    configure_gathering() emits them from a single asm block, so they land together.
    Requiring the ordered triple is what distinguishes "this sequence was emitted"
    from "the ELF happens to contain enough cfg0 writes".
    """
    for i, word in enumerate(words):
        if word != CSRRS_ZERO_CFG0_T1:
            continue
        window = words[i + 1 : i + SEQUENCE_WINDOW_WORDS]
        if CSRRS_ZERO_CFG0_T1 not in window:
            continue
        after_second = window[window.index(CSRRS_ZERO_CFG0_T1) + 1 :]
        if CSRRC_ZERO_CFG0_T1 in after_second:
            return True
    return False


@blackhole_only
def test_gathering_disabled_in_all_risc_firmware():
    """Every RISC's firmware must carry tt-metal's gathering-disable sequence."""
    # Any kernel will do: the sequence under test lives in do_crt0(), which every
    # RISC's firmware runs. This mirrors test_dest_copy's invocation so the build
    # is a known-good one.
    formats = input_output_formats([DataFormat.Float16_b], same=True)[0]
    input_dimensions = [32, 32]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    configuration = TestConfig(
        "sources/debug_dest_copy.cpp",
        formats,
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
    )
    # Build only -- this asserts on the binary, so no device is needed
    # (runs under --compile-producer as well as on hardware).
    configuration.prepare()

    elf_dir = (
        TestConfig.ARTEFACTS_DIR
        / configuration.test_name
        / configuration.variant_id
        / "elf"
    )
    elves = {name: elf_dir / f"{name}.elf" for name in ("unpack", "math", "pack")}
    elves["brisc"] = TestConfig.SHARED_ELF_DIR / "brisc.elf"

    missing = [n for n, p in elves.items() if not p.is_file()]
    assert not missing, f"firmware ELFs were not built: {missing} (looked in {elf_dir})"

    for name, path in elves.items():
        words = _executable_words(path)
        n_set = words.count(CSRRS_ZERO_CFG0_T1)
        n_clear = words.count(CSRRC_ZERO_CFG0_T1)
        assert n_set >= MIN_CSRRS and n_clear >= MIN_CSRRC, (
            f"{name}.elf does not carry the gathering-disable sequence "
            f"(csrrs 0x7c0,t1 x{n_set}, csrrc 0x7c0,t1 x{n_clear}; "
            f"expected at least x{MIN_CSRRS} / x{MIN_CSRRC}). "
            "do_crt0() must call configure_gathering() so the tests run in the same "
            "gathering configuration as tt-metal firmware."
        )
        assert _has_gathering_sequence(words), (
            f"{name}.elf contains cfg0 writes but not configure_gathering()'s ordered "
            f"csrrs/csrrs/csrrc triple within {SEQUENCE_WINDOW_WORDS} words. The counts "
            "above are met by some other cfg0 writer, not by do_crt0()."
        )


@blackhole_only
def test_harness_gathering_sequence_matches_tt_metal():
    """The harness copy must not drift from tt-metal's configure_gathering()."""
    root = TestConfig.LLK_ROOT.parent.parent  # tt-llk/ -> tt_metal/ -> repo root
    metal = (root / "tt_metal/hw/inc/internal/firmware_common.h").read_text()
    harness = (TestConfig.LLK_ROOT / "tests/helpers/include/boot.h").read_text()

    def csr_ops(text):
        # The ordered cfg0 CSR operations inside configure_gathering().
        body = text.split("configure_gathering()", 1)[-1].split("\n}", 1)[0]
        return [
            line.strip()
            for line in body.splitlines()
            if "0x7c0" in line or "slli" in line or line.strip().startswith("li ")
        ]

    assert csr_ops(harness) == csr_ops(metal), (
        "boot.h's configure_gathering() has drifted from tt-metal's in "
        "firmware_common.h; keep the two in sync."
    )
