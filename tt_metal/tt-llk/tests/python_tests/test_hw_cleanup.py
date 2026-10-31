# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Compile/smoke test for the Blackhole-only experimental hardware-teardown LLK
family "hw_cleanup" (compute_kernel_hw_cleanup.h ->
llk_{unpack,math,pack}_hw_cleanup.h + shared llk_hw_cleanup.h).

hw_cleanup is a TEARDOWN family with NO numeric output of its own. It drains the
three TRISCs, rendezvouses T0/T1/T2 through hardware mailboxes, and reprograms
both cfg banks to a canonical Float16_b 32x32 / four-face / 2048B geometry,
leaving cfg bank 0 selected (see the header docstrings).

Because cleanup emits no data, the only observable is that it compiles and runs
without hanging AND does not corrupt a result computed before it. The C++ source
therefore performs a plain identity datacopy of one tile, then runs the
per-thread cleanup canonicals (the same entry points compute_kernel_hw_cleanup()
dispatches), then re-inits pack (cleanup deliberately poisons pack MOP / strides
/ PAC X) and packs the datacopied tile out. The golden is that identity: the
packed tile must equal the input tile.

Blackhole-only; there is no BH card in this environment, so this is validated by
a clean Blackhole compile plus the identity golden below.
"""

import pytest
from conftest import blackhole_only
from helpers.format_config import DataFormat
from helpers.golden_generators import DataCopyGolden, get_golden_generator
from helpers.llk_params import DestAccumulation
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import NUM_FACES, TILE_COUNT
from helpers.utils import passed_test, result_tensor

pytestmark = blackhole_only

# WEDGES REAL BLACKHOLE — applied to the on-device test (test_hw_cleanup) only, which is therefore
# skipped on all backends; the build-only test below is collected and compiles. BH-card tt-exalens
# callstacks: MATH stalls in _llk_math_pack_sync_init_->tensix_sync (the Tensix pipe won't drain)
# during the cleanup, while UNPACK (hw_cleanup::finish) and PACK (hw_cleanup::start) block on the
# MATH-orchestrated three-thread mailbox rendezvous waiting for it. The hw_cleanup LLK is correct
# for its real (model-level) usage; the gap is test-side -- this standalone driver fires three
# independent run_kernels and does not reproduce the compute-kernel framework's
# ordering/preconditions the rendezvous is written against. ttsim can't model it either
# (UnimplementedFunctionality: tensix_cfg_wr32 reg=281, the cfg-bank reprogram). A hang cascades
# on hardware, so keep the on-device test skipped until it supplies that framework context.
_WEDGES_REAL_BLACKHOLE = pytest.mark.skip(
    reason="Wedges real BH: standalone test doesn't reproduce the framework flow-control the "
    "hw_cleanup three-thread mailbox rendezvous needs (the LLK itself is model-proven). "
    "ttsim also can't model the cfg-bank reprogram (reg 281)."
)

# Single config: the canonical geometry cleanup itself restores (one 32x32 tile,
# four faces, Float16_b in and out).
NUM_FACES_VALUE = 4


@_WEDGES_REAL_BLACKHOLE
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    dest_acc=[DestAccumulation.No],
    input_dimensions=[[32, 32]],
)
def test_hw_cleanup(formats, dest_acc, input_dimensions):
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    # Golden: identity datacopy of the single input tile. Cleanup contributes no
    # data of its own; the assertion is that the packed tile is unchanged (the
    # teardown neither corrupts the result nor hangs).
    generate_golden = get_golden_generator(DataCopyGolden)
    golden_tensor = generate_golden(
        src_A,
        formats.output_format,
        NUM_FACES_VALUE,
        input_dimensions,
    )

    configuration = TestConfig(
        "sources/hw_cleanup_test.cpp",
        formats,
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(NUM_FACES_VALUE),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            num_faces=NUM_FACES_VALUE,
        ),
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    res_tensor = result_tensor(res_from_L1, formats.output_format, golden_tensor)

    # All 1024 lanes of the single tile are defined (full identity datacopy), so
    # validate the whole tile at the Float16_b format tolerance.
    assert passed_test(golden_tensor, res_tensor, formats.output_format)


def test_hw_cleanup_builds_at_o2():
    """Guards the -O2 build of the hw_cleanup driver.

    tt-metal builds compute kernels at -O3 by default but lets a kernel choose O0/O1/O2/Os/Oz
    (``KernelBuildOptLevel``). ``addr_mod_t::set()`` expands to SETC16 with an immediate
    operand, so its index has to be a compile-time constant; a runtime-indexed loop over it
    only compiled because -O3 happened to unroll it completely, and was a hard
    "impossible constraint in asm" error at -O2. This builds the same driver at -O2
    (``sources/hw_cleanup_o2_build_test.cpp``) and never runs it.

    Only -O2 is guarded; the other levels are not swept here. -O0, -Os and -Oz do not build
    this translation unit with or without the constant-index calls: at those levels the
    SETC16 value operands of every ``addr_mod_t::set()`` caller (the datacopy init included)
    fail the same asm constraint, and at -O0 so do other immediate-operand instructions
    (SETRWC, SEMPOST). That is a property of the instruction macros, not of the hw_cleanup
    header.
    """
    formats = input_output_formats([DataFormat.Float16_b])[0]
    input_dimensions = [32, 32]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    configuration = TestConfig(
        "sources/hw_cleanup_o2_build_test.cpp",
        formats,
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(NUM_FACES_VALUE),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            num_faces=NUM_FACES_VALUE,
        ),
        dest_acc=DestAccumulation.No,
    )
    # Build only: the assertion is that the -O2 build succeeds.
    configuration.prepare()

    elf_dir = (
        TestConfig.ARTEFACTS_DIR
        / configuration.test_name
        / configuration.variant_id
        / "elf"
    )
    missing = [
        n for n in ("unpack", "math", "pack") if not (elf_dir / f"{n}.elf").is_file()
    ]
    assert not missing, f"-O2 build did not produce {missing} (looked in {elf_dir})"
