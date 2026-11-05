# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Regression: the LLK test firmware must configure Blackhole instruction gathering
the same way tt-metal firmware does.

Blackhole boots with instruction gathering enabled, and tt-metal leaves it on: the build
defines ENABLE_GATHERING unless TT_METAL_DISABLE_GATHERING=1. With the define,
``ckernel.h``'s ``load_replay_buf()`` brackets every replay-record window with
disable/enable. Without it, firmware disables gathering on every RISC around Tensix
(``configure_gathering()`` in ``tt_metal/hw/inc/internal/firmware_common.h``).

Those are two halves of one contract: either gathering is off globally, or it is on
and the record window is bracketed. The test firmware must implement the same one as
tt-metal for the same switch, and this test is what holds it to that.

This test asserts on the built firmware rather than on behaviour, because the
gathering hazard is a codegen lottery: a kernel that happens to be laid out safely
passes with gathering on, so a passing functional test proves nothing about the
configuration. CSR 0x7c0 is write-only in this codebase (every use is
``csrrs zero``/``csrrc zero``), so it cannot be read back and checked at runtime.
"""

import os
from pathlib import Path

import pytest
from conftest import blackhole_only
from helpers.format_config import DataFormat
from helpers.param_config import input_output_formats
from helpers.replay_record_scan import _executable_words
from helpers.replay_record_scan import check_elf_records as _check_elf_records
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig

# Vendored tt-llk: parents[4] is tt_metal. A standalone checkout has no hw/inc,
# and both the ELF compile and the firmware diff need that tree.
if not (Path(__file__).resolve().parents[4] / "hw" / "inc").is_dir():
    pytest.skip(
        "tt-metal hw/inc is absent; gathering compiles and diffs firmware against that tree",
        allow_module_level=True,
    )

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
def test_firmware_gathering_matches_build_config():
    """With gathering disabled (TT_METAL_DISABLE_GATHERING=1), every RISC's firmware must carry
    tt-metal's gathering-disable sequence; with gathering on (the default), BRISC's must not.

    BRISC only: it records nothing, so a disable_gathering() bracket around a record cannot
    look like the boot sequence there. run_hardware_free.sh runs this test in both
    configurations."""
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

    if TestConfig.gathering_build():
        assert not _has_gathering_sequence(_executable_words(elves["brisc"])), (
            "brisc.elf carries configure_gathering()'s gathering-disable sequence in a build "
            "with ENABLE_GATHERING: boot.h must leave gathering on, as tt-metal firmware does."
        )
        return

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


# _calculate_where_'s plain-SFPLOAD path (DISABLE_SFPLOADMACRO) records 6 instructions: three
# SFPLOADs, SFPSETCC, SFPENCC, SFPSTORE. Its SFPLOADMACRO paths record 3 and 4, so the length
# tells the site under test apart from the two that were already bracketed.
WHERE_PLAIN_SFPLOAD_RECORD_LEN = 6


def _assert_records_bracketed(elf_path):
    """Every replay record of ``elf_path`` sits inside the bracket; returns the records."""
    records, unbracketed = _check_elf_records(elf_path)
    assert not unbracketed, (
        f"{elf_path.name}: replay record(s) not bracketed by disable_gathering()/"
        "enable_gathering() with ENABLE_GATHERING defined; record through "
        "ckernel::load_replay_buf instead of a raw lltt::record / TTI_REPLAY:\n  "
        + "\n  ".join(unbracketed)
    )
    return records


def _math_elf(configuration):
    """Build ``configuration`` and return its math ELF."""
    # Build only: the assertions are on the binary.
    configuration.prepare()
    math_elf = (
        TestConfig.ARTEFACTS_DIR
        / configuration.test_name
        / configuration.variant_id
        / "elf"
        / "math.elf"
    )
    assert math_elf.is_file(), f"math ELF was not built: {math_elf}"
    return math_elf


@blackhole_only
def test_replay_records_are_gathering_bracketed():
    """With ENABLE_GATHERING, every replay record in the accurate bf16 exp kernel must sit
    inside disable_gathering()/enable_gathering(), as load_replay_buf emits it.

    Functional sweeps cannot catch a raw TTI_REPLAY record here: on a Blackhole p100a the
    unbracketed record produced bit-identical exp output with gathering on, so only the
    binary shows whether the bracket is there.
    """
    from helpers.llk_params import (
        ApproximationMode,
        DestAccumulation,
        FastMode,
        MathOperation,
    )
    from test_eltwise_unary_sfpu import build_eltwise_unary_sfpu_config

    # bf16 Dest + APPROX_MODE=No selects _sfpu_exp_21f_bf16_tti_, the replayed body.
    formats = input_output_formats([DataFormat.Float16_b], same=True)[0]
    input_dimensions = [32, 32]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    configuration = build_eltwise_unary_sfpu_config(
        formats,
        DestAccumulation.No,
        ApproximationMode.No,
        MathOperation.Exp,
        input_dimensions,
        src_A,
        src_B,
        tile_cnt_A,
        tile_cnt_B,
        test_name="sources/eltwise_unary_sfpu_gathering_test.cpp",
        fast_mode=FastMode.No,
    )
    records = _assert_records_bracketed(_math_elf(configuration))
    assert (
        records
    ), "no replay record found in the exp math ELF; the bf16 TTI path moved"


@blackhole_only
def test_welfords_replay_record_is_gathering_bracketed():
    """With ENABLE_GATHERING, _program_welfords_replay_buffer_'s record must sit inside
    disable_gathering()/enable_gathering(); only the math ELF can show that."""
    from helpers.llk_params import DestAccumulation

    formats = input_output_formats([DataFormat.Float16_b], same=True)[0]
    input_dimensions = [32, 32]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    configuration = TestConfig(
        "sources/welfords_sfpu_gathering_test.cpp",
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
        dest_acc=DestAccumulation.No,
    )
    records = _assert_records_bracketed(_math_elf(configuration))
    assert records, (
        "no replay record found in the welfords math ELF; "
        "_llk_math_welfords_sfpu_init_ no longer programs the replay buffer"
    )


@blackhole_only
def test_where_without_sfploadmacro_replay_record_is_gathering_bracketed():
    """With ENABLE_GATHERING and DISABLE_SFPLOADMACRO, _calculate_where_'s plain-SFPLOAD
    record must sit inside disable_gathering()/enable_gathering() like its two siblings.
    """
    from helpers.llk_params import (
        ApproximationMode,
        DestAccumulation,
        MathOperation,
        PerfRunType,
        Transpose,
    )
    from helpers.test_variant_parameters import (
        APPROX_MODE,
        DEST_SYNC,
        ITERATIONS,
        LOOP_FACTOR,
        NUM_BLOCKS,
        NUM_FACES,
        NUM_TILES_IN_BLOCK,
        PERF_RUN_TYPE,
        SFPU_TERNARY_OP,
        SFPU_TERNARY_SCALAR,
        TILE_COUNT,
        UNPACK_TRANS_FACES,
        UNPACK_TRANS_WITHIN_FACE,
    )
    from helpers.tile_constants import MAX_NUM_FACES

    # bf16 is the LO16 where path; the record is the same for every supported format.
    formats = input_output_formats([DataFormat.Float16_b], same=True)[0]
    input_dimensions = [32, 32]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    configuration = TestConfig(
        "sources/sfpu_ternary_gathering_nomacro_test.cpp",
        formats,
        templates=[
            SFPU_TERNARY_OP(MathOperation.TTNNWhere),
            SFPU_TERNARY_SCALAR(0x40000000),
            APPROX_MODE(ApproximationMode.No),
            ITERATIONS(8),
            DEST_SYNC(),
            TILE_COUNT(tile_cnt_A),
            LOOP_FACTOR(1),
            NUM_FACES(num_faces=MAX_NUM_FACES),
            UNPACK_TRANS_FACES(Transpose.No),
            UNPACK_TRANS_WITHIN_FACE(Transpose.No),
            PERF_RUN_TYPE(PerfRunType.L1_TO_L1),
        ],
        runtimes=[NUM_BLOCKS(tile_cnt_A), NUM_TILES_IN_BLOCK(1)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            buffer_C=src_A,
            stimuli_C_format=formats.input_format,
            tile_count_C=tile_cnt_A,
        ),
        unpack_to_dest=False,
        dest_acc=DestAccumulation.No,
        compile_time_formats=True,
    )
    records = _assert_records_bracketed(_math_elf(configuration))
    assert any(length == WHERE_PLAIN_SFPLOAD_RECORD_LEN for _, length in records), (
        f"no replay record of length {WHERE_PLAIN_SFPLOAD_RECORD_LEN} in the where math ELF "
        f"(record lengths found: {[length for _, length in records]}); DISABLE_SFPLOADMACRO no "
        "longer selects the plain-SFPLOAD path of _calculate_where_, so this test is not "
        "covering it"
    )


# One test variant per recording kernel path, built with gathering on (the Blackhole default)
# under --compile-producer; every record in its ELF must be bracketed. (kernel, node id, ELF)
GATHERING_BUILD_VARIANTS = [
    (
        "transpose_dest 32-bit",
        "test_transpose_dest.py::test_transpose_dest_float[fmt_dest_acc_math_transp_unpack_to_dest:"
        "(InputOutputFormat[A:Float32,B:Float32,out:Float32], <DestAccumulation.Yes: True>, "
        "<Transpose.Yes: True>, True)]",
        "math",
    ),
    (
        "gcd (shared with lcm)",
        "test_eltwise_binary_sfpu.py::test_eltwise_binary_sfpu_int_uniform[mathop:SfpuGcd-dest_acc:Yes]",
        "math",
    ),
    (
        "quant",
        "test_eltwise_binary_sfpu.py::test_eltwise_binary_sfpu_quant[mathop:SfpuQuant-sign_magnitude:False]",
        "math",
    ),
    (
        "requant",
        "test_eltwise_binary_sfpu.py::test_eltwise_binary_sfpu_quant[mathop:SfpuRequant-sign_magnitude:False]",
        "math",
    ),
    (
        "dequant",
        "test_eltwise_binary_sfpu.py::test_eltwise_binary_sfpu_quant[mathop:SfpuDequant-sign_magnitude:False]",
        "math",
    ),
    (
        "binary bcast column (broadcast helper)",
        "test_eltwise_binary_sfpu.py::test_eltwise_binary_sfpu_bcast[formats:Float16_b->Float16_b-"
        "bcast_dim:Column-mathop:SfpuElwsub-dest_acc:No]",
        "math",
    ),
    (
        "sfpu reduce max/min uint16 in a 32-bit Dest (manual swap chain)",
        "test_sfpu_reduce.py::test_sfpu_reduce[formats:UInt16->UInt16-mathop:ReduceColumn-dest_acc:Yes-"
        "input_bounds:(0, 1000)-reduce_pool:Max-dimension_combinations:[32, 32]-reduced_extent:15]",
        "math",
    ),
    (
        "sfpu reduce max/min float (LOADMACRO swap chain)",
        "test_sfpu_reduce.py::test_sfpu_reduce[formats:Float16_b->Float16_b-mathop:ReduceColumn-"
        "dest_acc:No-input_bounds:(-1000, 1000)-reduce_pool:Max-dimension_combinations:[32, 32]-"
        "reduced_extent:15]",
        "math",
    ),
    (
        "sfpu reduce sum/avg float (tree reduce)",
        "test_sfpu_reduce.py::test_sfpu_reduce[formats:Float16_b->Float16_b-mathop:ReduceColumn-"
        "dest_acc:No-input_bounds:(-1000, 1000)-reduce_pool:Sum-dimension_combinations:[32, 32]-"
        "reduced_extent:15]",
        "math",
    ),
    (
        "sfpu reduce sum/avg integer (tree reduce)",
        "test_sfpu_reduce.py::test_sfpu_reduce[formats:Int32->Int32-mathop:ReduceColumn-dest_acc:Yes-"
        "input_bounds:(-1000, 1000)-reduce_pool:Sum-dimension_combinations:[32, 32]-reduced_extent:1]",
        "math",
    ),
    (
        "sfpu reduce max/min signed int32 (swap chain)",
        "test_sfpu_reduce.py::test_sfpu_reduce[formats:Int32->Int32-mathop:ReduceColumn-dest_acc:Yes-"
        "input_bounds:(-1000, 1000)-reduce_pool:Max-dimension_combinations:[32, 32]-reduced_extent:1]",
        "math",
    ),
    (
        "sfpu reduce_custom (sdpa 4x2 subblock)",
        "test_sfpu_reduce_sdpa.py::test_sfpu_reduce_sdpa[stimuli:default-mathop:ReduceColumn]",
        "pack",  # the sdpa reduce runs its SFPU work on the pack thread
    ),
    (
        "topk_xl",
        "test_topk_xl.py::test_topk_xl[K:512-num_chunks:1-partial_tail:False]",
        "math",
    ),
    (
        "topk_xl K=2048 (face transposes from Dst, msb removal, index separation)",
        "test_topk_xl.py::test_topk_xl[K:2048-num_chunks:1-partial_tail:False]",
        "math",
    ),
    (
        "topk_xl index separation",
        "test_topk_xl.py::test_topk_xl_separate_indices[K:512-core_id:0-group_shift:16]",
        "math",
    ),
    (
        "topk_xl unfused macro",
        "test_topk_xl_unfused_macro.py::test_topk_xl_unfused_macro_chained[K:512-num_chunks:2]",
        "math",
    ),
    (
        # Also the canary for compiler-generated records: its scrub helper's unrolled SFPU fill
        # loop becomes a raw record unless the gathering build passes -mno-tt-tensix-optimize-replay.
        "topk_xl split",
        "test_topk_xl_split.py::test_topk_xl_split[k512-chunks1-tail512-rows1-positive-bf16-seg0]",
        "pack",  # the split sort records on the pack thread
    ),
    (
        "generalized moe gate shipping config (single-face transposes)",
        "test_generalized_moe_gate.py::test_generalized_moe_gate_shipping_config[dest_sync:Half-num_faces:4]",
        "math",
    ),
    (
        "generalized moe gate grouped (step1 transpose)",
        "test_generalized_moe_gate.py::test_generalized_moe_gate_grouped[seed:128-approx:No]",
        "math",
    ),
    (
        "generalized moe gate eltwise binary reload",
        "test_generalized_moe_gate.py::test_generalized_moe_gate_binary[math_op:Elwadd-reload:True]",
        "math",
    ),
    (
        "reduce_block_max_row (reduce_custom)",
        "test_reduce_block_max.py::test_reduce_block_max[formats:Float16_b->Float16_b-block_ct_dim:1-num_faces:4]",
        "math",
    ),
    (
        "reduce_block_max_row fp32 Dest",
        "test_reduce_block_max.py::test_reduce_block_max[formats:Float16_b->Float32-block_ct_dim:1-num_faces:4]",
        "math",
    ),
    (
        "reduce_block_max_row runtime (reduce_runtime_custom)",
        "test_reduce_block_max.py::test_reduce_block_max_runtime[formats:Float16_b->Float16_b-block_ct_dim:1-num_faces:4]",
        "math",
    ),
    (
        "reduce_block_max_row runtime fp32 Dest",
        "test_reduce_block_max.py::test_reduce_block_max_runtime[formats:Float16_b->Float32-block_ct_dim:1-num_faces:4]",
        "math",
    ),
    (
        "pack block contiguous (runtime-length record)",
        "test_pack_tiny_tile_block.py::test_pack_tiny_tile_block[formats:Float16_b->Float16_b-dest_acc:No-"
        "tile_dims:(1, 32)-num_tiles:1]",
        "pack",
    ),
    (
        "pack fast tilize",
        "test_fast_tilize_full.py::test_fast_tilize_full[formats:Float16_b->Bfp8_b-dest_acc:No-dimensions:(1, 3)]",
        "pack",
    ),
    (
        "exp approximate without the negative clamp (LOADMACRO record)",
        "test_eltwise_unary_sfpu.py::test_exponential_clamp_negative[False]",
        "math",
    ),
]


@blackhole_only
@pytest.mark.parametrize(
    "kernel, node_id, recording_elf",
    GATHERING_BUILD_VARIANTS,
    ids=[kernel for kernel, _, _ in GATHERING_BUILD_VARIANTS],
)
def test_kernel_replay_records_are_gathering_bracketed(
    kernel, node_id, recording_elf, tmp_path
):
    """Built with gathering on, every replay record in the kernel's ELFs must sit
    inside disable_gathering()/enable_gathering(): the record is the one known incompatibility
    with instruction gathering, and only the binary shows whether the bracket is there.
    """
    import subprocess
    import sys

    python_tests = TestConfig.LLK_ROOT / "tests" / "python_tests"
    env = dict(os.environ, RUNNER_TEMP=str(tmp_path))
    env.pop("TT_METAL_DISABLE_GATHERING", None)
    # An ordinary --compile-producer session in a private build tree, not another hardware-free
    # session (that would build under its own run id instead of RUNNER_TEMP/tt-llk-build).
    env.pop("LLK_HARDWARE_FREE", None)
    env.pop("LLK_HARDWARE_FREE_RUN_ID", None)
    # No xdist: one variant, and the build directory is private to this test.
    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            node_id,
            "--compile-producer",
            "-p",
            "no:cacheprovider",
            "-q",
        ],
        cwd=python_tests,
        env=env,
        capture_output=True,
        text=True,
    )
    # pytest exit 4 = usage error, 5 = nothing collected: the node id no longer exists.
    assert build.returncode not in (4, 5), (
        f"{kernel}: stale node id {node_id!r}; update GATHERING_BUILD_VARIANTS to the test's "
        f"current parametrisation:\n{build.stdout[-2000:]}\n{build.stderr[-1000:]}"
    )
    assert (
        build.returncode == 0
    ), f"{kernel}: the gathering build of {node_id} failed:\n{build.stdout[-4000:]}\n{build.stderr[-2000:]}"

    # Gathering on is the default tree (TestConfig.resolve_artefacts_path).
    elf_dirs = sorted((tmp_path / "tt-llk-build" / "sources").glob("*/*/elf"))
    assert elf_dirs, f"{kernel}: no ELF directory produced under {tmp_path}"
    recording_elf_records = 0
    unbracketed = []
    for elf_dir in elf_dirs:
        for thread in ("unpack", "math", "pack"):
            elf = elf_dir / f"{thread}.elf"
            assert elf.is_file(), f"{kernel}: {elf} was not built"
            records, bad = _check_elf_records(elf)
            if thread == recording_elf:
                recording_elf_records += len(records)
            unbracketed += [f"{thread}.elf: {line}" for line in bad]
    assert recording_elf_records, (
        f"{kernel}: no replay record found in {recording_elf}.elf of {node_id}; the variant no "
        "longer reaches the recording kernel, so it is not covering it"
    )
    assert not unbracketed, (
        f"{kernel}: replay record(s) not bracketed by disable_gathering()/enable_gathering() with "
        "ENABLE_GATHERING defined; record through ckernel::load_replay_buf instead of a raw "
        "lltt::record / TTI_REPLAY:\n  " + "\n  ".join(unbracketed)
    )


@blackhole_only
def test_mac_replay_records_are_gathering_bracketed():
    """mac_init's record (no harness test drives the kernel; a build-only driver instantiates
    the bf16 and fp32 inits) must sit inside disable_gathering()/enable_gathering()."""
    from helpers.llk_params import DestAccumulation

    formats = input_output_formats([DataFormat.Float16_b], same=True)[0]
    input_dimensions = [32, 32]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    configuration = TestConfig(
        "sources/sfpu_mac_gathering_test.cpp",
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
        dest_acc=DestAccumulation.No,
    )
    # Build only: the assertions are on the binary.
    configuration.prepare()
    math_elf = (
        TestConfig.ARTEFACTS_DIR
        / configuration.test_name
        / configuration.variant_id
        / "elf"
        / "math.elf"
    )
    assert math_elf.is_file(), f"math ELF was not built: {math_elf}"
    records, unbracketed = _check_elf_records(math_elf)
    # The compiler emits each init inline and out of line, so count the body lengths, not records.
    assert {length for _, length in records} == {5, 6}, (
        f"expected the fp32 (5) and bf16 (6) mac_init records in the math ELF, found lengths "
        f"{sorted(length for _, length in records)}"
    )
    assert not unbracketed, (
        "mac_init record(s) not bracketed by disable_gathering()/enable_gathering():\n  "
        + "\n  ".join(unbracketed)
    )
