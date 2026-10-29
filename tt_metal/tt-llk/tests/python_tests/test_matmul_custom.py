# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.device import BootMode, LLKAssertException
from helpers.format_config import DataFormat, is_dest_acc_needed
from helpers.golden_generators import (
    TILE_DIM,
    MatmulGolden,
    TransposeGolden,
    get_golden_generator,
)
from helpers.llk_params import DestAccumulation, MathFidelity, Transpose, format_dict
from helpers.matmul_sweep import (
    calculate_matmul_output_faces,
    generate_matmul_dimension_combinations,
    generate_tile_dims,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import (
    StimuliSpec,
    convert_to_l1_view,
    generate_face_matmul_data,
    generate_stimuli,
)
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    IN_TILE_DIMS,
    MATH_FIDELITY,
    NUM_FACES,
    PARTIAL_FACE,
    THROTTLE_LEVEL,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
)
from helpers.tile_constants import MAX_FACE_R_DIM, MAX_NUM_FACES
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test

# Throttle levels supported by the no-mop matmul math LLK. Throttle only inserts
# NOPs between MVMULs to cap compute throughput, so the numeric result is IDENTICAL
# for every level -> the golden is the same MatmulGolden as the non-throttled path.
#
# Blackhole implements run_throttled_sequence_no_mop<1..5>, so it sweeps 0-5.
# Wormhole B0's LLK static_asserts THROTTLE_LEVEL == 0 (llk_math_matmul_custom_no_mop.h:
# "Wormhole custom no-mop matmul only supports THROTTLE_LEVEL == 0") -- levels 1-5 have no
# throttle sequences and do not compile there. Any non-Blackhole arch is therefore level 0
# only; throttle 1-5 on WH is unsupported by design, not an untested gap. This gate is
# arch-explicit (not a bare else) so a future arch that gains throttle sequences must opt in.
if get_chip_architecture() == ChipArchitecture.BLACKHOLE:
    THROTTLE_LEVELS = [0, 1, 2, 3, 4, 5]
else:
    THROTTLE_LEVELS = [0]


def _dest_bank_max_tiles(formats, dest_acc):
    """Max result-tile count for a (format, dest_acc) pair on DestSync.Half.

    A 32-bit destination register (dest_acc=Yes, or a format the harness forces
    onto 32-bit dest) holds 4 tiles; the 16-bit destination holds 8. Mirrors
    perf_matmul._dest_bank_max_tiles so test and perf coverage share the rule.
    """
    if is_dest_acc_needed(formats) or dest_acc == DestAccumulation.Yes:
        return 4
    return 8


MATMUL_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float32,
        DataFormat.Bfp8_b,
    ]
)
DEST_ACC_MODES = [DestAccumulation.No, DestAccumulation.Yes]
# A 16x32 / 8x32 in0 or 32x16 in1 tiny tile holds two faces of the 32x32 tile.
TINY_TILE_NUM_FACES = 2


def _run_matmul_custom(
    math_fidelity,
    formats,
    dest_acc,
    input_A_dimensions,
    input_B_dimensions,
    throttle_level: int,
    boot_mode: BootMode,
    in0_tile_r_dim: int = TILE_DIM,
    in1_tile_c_dim: int = TILE_DIM,
    transpose: Transpose = Transpose.No,
):
    """Run one no-mop matmul variant and compare it against the golden.

    ``in0_tile_r_dim=16`` selects a 16x32 in0 tiny tile (two horizontal faces, f0/f1), 8 an
    8x32 one (partial face); ``in1_tile_c_dim=16`` a 32x16 in1 tiny tile (two vertical faces,
    f0/f2). The input dimensions stay in 32x32 blocks: the tiny tile occupies the active
    faces of each block and the rest is zero, the same convention as test_math_matmul's
    tiny-tile sweep.
    ``transpose`` transposes in1 (faces and within faces) on both the unpack and the math
    thread, and the golden is computed against the transposed in1, as in test_math_matmul.
    """
    torch_format = format_dict[formats.output_format]

    # Calculate all matmul dimensions using helper function
    matmul_dims = generate_tile_dims(
        (input_A_dimensions, input_B_dimensions),
        in0_tile_r_dim=in0_tile_r_dim,
        in1_tile_c_dim=in1_tile_c_dim,
    )
    is_tiny_in0 = in0_tile_r_dim < TILE_DIM
    in0_face_r_dim = min(in0_tile_r_dim, MAX_FACE_R_DIM)
    num_faces_in0 = TINY_TILE_NUM_FACES if is_tiny_in0 else MAX_NUM_FACES
    num_faces_in1 = MAX_NUM_FACES if in1_tile_c_dim == TILE_DIM else TINY_TILE_NUM_FACES
    num_faces = calculate_matmul_output_faces(
        num_faces_in0, num_faces_in1, is_in0_horizontal=True
    )
    full_tiles = num_faces_in0 == MAX_NUM_FACES and num_faces_in1 == MAX_NUM_FACES

    if full_tiles:
        sfpu_false_spec = StimuliSpec.uniform(low=0.0, high=1.0)
        src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
            stimuli_format_A=formats.input_format,
            input_dimensions_A=input_A_dimensions,
            stimuli_format_B=formats.input_format,
            input_dimensions_B=input_B_dimensions,
            spec_A=sfpu_false_spec,
            spec_B=sfpu_false_spec,
        )
    else:
        assert (
            formats.input_format != DataFormat.Bfp8_b
        ), "tiny-tile stimuli are generated as plain blocks and tilized here"
        src_A = generate_face_matmul_data(
            num_faces=num_faces_in0,
            stimuli_format=formats.input_format,
            input_dimensions=input_A_dimensions,
            is_matrix_A=True,  # in0 (SrcB) uses f0,f1 in 2-face mode
            face_r_dim=in0_face_r_dim,
        )
        src_B = generate_face_matmul_data(
            num_faces=num_faces_in1,
            stimuli_format=formats.input_format,
            input_dimensions=input_B_dimensions,
            is_matrix_A=False,  # in1 (SrcA) uses f0,f2 in 2-face mode
        )
        tile_cnt_A = matmul_dims.tile_cnt_in0
        tile_cnt_B = matmul_dims.tile_cnt_in1

    src_B_golden = src_B
    if transpose == Transpose.Yes:
        t_matrix = get_golden_generator(TransposeGolden)
        src_B_golden = t_matrix.transpose_faces_multi_tile(
            src_B,
            formats.input_format,
            num_tiles=tile_cnt_B,
            tilize=True,
            input_dimensions=input_B_dimensions,
        )
        src_B_golden = t_matrix.transpose_within_faces_multi_tile(
            src_B_golden,
            formats.input_format,
            num_tiles=tile_cnt_B,
            untilize=True,
            input_dimensions=input_B_dimensions,
        )

    generate_golden = get_golden_generator(MatmulGolden)
    golden_tensor = generate_golden(
        src_A,
        src_B_golden,
        formats.output_format,
        math_fidelity,
        input_A_dimensions=input_A_dimensions,
        input_B_dimensions=input_B_dimensions,
        # Golden cannot model FPU strided for tilized data computation, so we tilize output after computation
        tilize=True,
        input_A_format=formats.input_format,
        input_B_format=formats.input_format,
    )

    if formats.input_format != DataFormat.Bfp8_b:
        tilized_A = tilize_block(
            src_A, dimensions=input_A_dimensions, stimuli_format=formats.input_format
        )
        tilized_B = tilize_block(
            src_B, dimensions=input_B_dimensions, stimuli_format=formats.input_format
        )
    else:
        # BFP8 format requires special handling for tilization
        tilized_A = src_A
        tilized_B = src_B

    if not full_tiles:
        # Condense each 32x32 block to the faces the tiny tile actually occupies.
        tilized_A = convert_to_l1_view(
            tilized_A, input_A_dimensions, tile_dimensions=[in0_tile_r_dim, TILE_DIM]
        )
        tilized_B = convert_to_l1_view(
            tilized_B, input_B_dimensions, tile_dimensions=[TILE_DIM, in1_tile_c_dim]
        )

    configuration = TestConfig(
        "sources/matmul_custom_test.cpp",
        formats,
        templates=[MATH_FIDELITY(math_fidelity), THROTTLE_LEVEL(throttle_level)],
        runtimes=[
            NUM_FACES(num_faces, num_faces_in0, num_faces_in1),
            TILE_COUNT(matmul_dims.output_tile_cnt),
            CRK_TILE_DIMM(matmul_dims.ct_dim, matmul_dims.rt_dim, matmul_dims.kt_dim),
            IN_TILE_DIMS(in0_tile_r_dim, TILE_DIM, TILE_DIM, in1_tile_c_dim),
            # Same per-thread partial-face flags as test_math_matmul's tiny-tile sweep: the
            # in0 (SrcB) unpack and the pack run partial-face for a sub-32-row in0 tile,
            # the math only below one face row.
            PARTIAL_FACE(
                partial_a=is_tiny_in0,
                partial_face_pack=is_tiny_in0,
                partial_b=False,
                partial_face_math=in0_tile_r_dim < MAX_FACE_R_DIM,
            ),
            UNPACK_TRANS_FACES(transpose),
        ],
        variant_stimuli=StimuliConfig(
            tilized_A.flatten(),
            formats.input_format,
            tilized_B.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=matmul_dims.output_tile_cnt,
        ),
        dest_acc=dest_acc,
        boot_mode=boot_mode,
    )

    res_from_L1 = configuration.run().result

    if not full_tiles:
        golden_tensor = convert_to_l1_view(
            golden_tensor,
            matmul_dims.output_dimensions,
            tile_dimensions=[in0_tile_r_dim, in1_tile_c_dim],
        )

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    if num_faces < MAX_NUM_FACES:
        # Only the active faces of each result tile are packed; compare those.
        num_elements_per_tile = in0_tile_r_dim * in1_tile_c_dim
        for i in range(matmul_dims.output_tile_cnt):
            start = i * (TILE_DIM * TILE_DIM)
            assert passed_test(
                golden_tensor[start : start + num_elements_per_tile],
                res_tensor[start : start + num_elements_per_tile],
                formats.output_format,
            ), f"Assert on tile {i}/{matmul_dims.output_tile_cnt} against golden failed"
    else:
        assert passed_test(
            golden_tensor, res_tensor, formats.output_format
        ), "Assert against golden failed"


@parametrize(
    math_fidelity=[
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
    formats=MATMUL_FORMATS,
    dest_acc=DEST_ACC_MODES,
    # dimensions depends on (formats, dest_acc): the max result-tile count is set by
    # which dest register the pair lands in (see _dest_bank_max_tiles). Kept as its
    # own axis -- rather than a packed (format, dest_acc, dims) tuple -- so test and
    # perf coverage line up axis-for-axis (cf. perf_matmul.matmul_combos).
    dimensions=lambda formats, dest_acc: generate_matmul_dimension_combinations(
        _dest_bank_max_tiles(formats, dest_acc)
    ),
)
def test_matmul_custom(
    math_fidelity,
    formats,
    dest_acc,
    dimensions,
    boot_mode=BootMode.DEFAULT,
):
    input_A_dimensions, input_B_dimensions = dimensions
    _run_matmul_custom(
        math_fidelity,
        formats,
        dest_acc,
        input_A_dimensions,
        input_B_dimensions,
        throttle_level=0,
        boot_mode=boot_mode,
    )


# Representative fidelity x format subset for the throttle sweep. Throttle inserts
# NOPs between MVMULs without changing the numeric result, so it is orthogonal to
# format/fidelity/dims; a small subset per throttle level keeps the sweep tractable
# while still crossing LoFi (single-phase) and HiFi (multi-phase) code paths and
# both 16-bit (Float16_b) and 32-bit (Float32) operands.
THROTTLE_FORMATS = input_output_formats([DataFormat.Float16_b, DataFormat.Float32])
# Single 32x32 output tile (ct=rt=kt=1). This is the regime the throttled no-mop
# matmul is designed and used in (the SDPA full-tile path); the hand-written
# throttle sequences replay a fixed single-tile MVMUL walk. Multi-tile / multi-K
# accumulation is validated at throttle 0 by test_matmul_custom above.
THROTTLE_DIMS = ([32, 32], [32, 32])


@parametrize(
    throttle_level=THROTTLE_LEVELS,
    math_fidelity=[MathFidelity.LoFi, MathFidelity.HiFi4],
    formats=THROTTLE_FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_matmul_custom_throttle(
    throttle_level,
    math_fidelity,
    formats,
    dest_acc,
    boot_mode=BootMode.DEFAULT,
):
    # Throttle levels 4 and 5 only advance the fidelity-phase counter on the
    # final MVMUL of the sequence (they use ADDR_MOD_4 with fidelity.incr=0 at
    # the phase boundary, unlike levels 1-3 which use the fidelity-incrementing
    # ADDR_MOD_5/6), so for a multi-phase (HiFi) matmul the throttled image
    # would collapse the extra phases. The LLK therefore drops throttling there
    # (matmul_no_mop_use_throttled) and runs the unthrottled image, so every
    # level is covered at both fidelities here.
    input_A_dimensions, input_B_dimensions = THROTTLE_DIMS
    _run_matmul_custom(
        math_fidelity,
        formats,
        dest_acc,
        input_A_dimensions,
        input_B_dimensions,
        throttle_level=throttle_level,
        boot_mode=boot_mode,
    )


@skip_for_wormhole
@skip_for_quasar
def test_matmul_custom_transpose_in1_32x16_rejected(
    request, boot_mode=BootMode.DEFAULT
):
    """transpose with a 32x16 in1 has no address-mod handling in the matmul LLK; the regular
    _llk_math_matmul_init_ rejects the combination under LLK asserts. The no-mop init shares
    the tiny-tile address mods with it and must reject it the same way, rather than run the
    kernel with the transpose silently ignored. Without the assert the kernel completes and
    the result is compared against the A*B^T golden, which it cannot match.

    Defined before the tiny-tile tests so that, in file order, the tests that follow it on
    the same core prove the rejected kernel leaves nothing pending: in this case the test
    kernel loads no sources, returns from the math thread right after the (asserting) init,
    and polls MATH_PACK with a bound on the pack thread instead of waiting on it."""
    if os.environ.get("TT_LLK_DISABLE_ASSERTS") == "1":
        pytest.skip("expects an LLK assert; asserts are disabled in this run")
    if request.config.getoption("--run-simulator"):
        pytest.skip(
            "expects an LLK assert; on ttsim the math thread never reaches KERNEL_COMPLETE "
            "after the EBREAK, so the assert is only detected at the pytest timeout"
        )

    input_A_dimensions, input_B_dimensions = THROTTLE_DIMS
    with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
        LLKAssertException, match=r"llk_math_matmul_custom_no_mop\.h"
    ):
        _run_matmul_custom(
            MathFidelity.LoFi,
            input_output_formats([DataFormat.Float16_b])[0],
            DestAccumulation.No,
            input_A_dimensions,
            input_B_dimensions,
            throttle_level=0,
            boot_mode=boot_mode,
            in1_tile_c_dim=16,
            transpose=Transpose.Yes,
        )


# Tiny in0 tiles: 16x32 (one face row, two faces), the geometry the Blackhole SDPA tiny-tile
# path feeds this LLK through the operand tensor shape, and 8x32 (a partial face, the 4-MVMUL
# image). The throttled replay sequences are hand-written for a full 32x32 tile, so on these
# geometries the no-mop init must record the unthrottled geometry-aware replay and the
# execute must replay it (the fallback the regular _llk_math_matmul_init_ takes). Because the
# execute makes the same decision, the fallback holds at every level and fidelity, including
# HiFi at throttle levels 4/5 -- where full tiles also fall back to the unthrottled image,
# covered by the full-tile test above.
TINY_IN0_THROTTLE_LEVELS = [level for level in THROTTLE_LEVELS if level > 0]
TINY_IN0_ROWS = [16, 8]


@pytest.mark.nightly
@skip_for_wormhole
@skip_for_quasar
@parametrize(
    throttle_level=TINY_IN0_THROTTLE_LEVELS,
    in0_tile_r_dim=TINY_IN0_ROWS,
    math_fidelity=[MathFidelity.LoFi, MathFidelity.HiFi4],
    formats=THROTTLE_FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_matmul_custom_throttle_tiny_in0(
    throttle_level,
    in0_tile_r_dim,
    math_fidelity,
    formats,
    dest_acc,
    boot_mode=BootMode.DEFAULT,
):
    input_A_dimensions, input_B_dimensions = THROTTLE_DIMS
    _run_matmul_custom(
        math_fidelity,
        formats,
        dest_acc,
        input_A_dimensions,
        input_B_dimensions,
        throttle_level=throttle_level,
        boot_mode=boot_mode,
        in0_tile_r_dim=in0_tile_r_dim,
    )


# Smoke subset of the nightly sweep above: the SDPA 16x32 geometry at one level that takes
# the fallback below 4 and one above it, on both fidelity paths.
@skip_for_wormhole
@skip_for_quasar
@parametrize(
    throttle_level=[level for level in THROTTLE_LEVELS if level in (1, 4)],
    math_fidelity=[MathFidelity.LoFi, MathFidelity.HiFi4],
)
def test_matmul_custom_throttle_tiny_in0_smoke(
    throttle_level,
    math_fidelity,
    boot_mode=BootMode.DEFAULT,
):
    input_A_dimensions, input_B_dimensions = THROTTLE_DIMS
    _run_matmul_custom(
        math_fidelity,
        input_output_formats([DataFormat.Float16_b])[0],
        DestAccumulation.No,
        input_A_dimensions,
        input_B_dimensions,
        throttle_level=throttle_level,
        boot_mode=boot_mode,
        in0_tile_r_dim=MAX_FACE_R_DIM,
    )


# 32x16 in1 tiny tile (two vertical faces, f0/f2) with a full in0: the other non-full
# geometry the fallback applies to (the 8-MVMUL image). Throttle 0 covers the image itself,
# 1 and 3 the fallback dispatch on both fidelity paths.
@skip_for_wormhole
@skip_for_quasar
@parametrize(
    throttle_level=[level for level in THROTTLE_LEVELS if level in (0, 1, 3)],
    math_fidelity=[MathFidelity.LoFi, MathFidelity.HiFi4],
)
def test_matmul_custom_throttle_tiny_in1(
    throttle_level,
    math_fidelity,
    boot_mode=BootMode.DEFAULT,
):
    input_A_dimensions, input_B_dimensions = THROTTLE_DIMS
    _run_matmul_custom(
        math_fidelity,
        input_output_formats([DataFormat.Float16_b])[0],
        DestAccumulation.No,
        input_A_dimensions,
        input_B_dimensions,
        throttle_level=throttle_level,
        boot_mode=boot_mode,
        in1_tile_c_dim=16,
    )


# in1 transposed (faces and within faces) against a full 32x32 in0 and against the tiny
# 16x32 / 8x32 in0 (the SDPA QK^T pairing): the positive counterpart of the rejection test
# below the throttle test, checking the transpose address-mod branches of the full-tile and
# tiny-tile no-mop images and the A*B^T golden on hardware.
@parametrize(
    throttle_level=[level for level in THROTTLE_LEVELS if level in (0, 1)],
    in0_tile_r_dim=[TILE_DIM] + TINY_IN0_ROWS,
    math_fidelity=[MathFidelity.LoFi, MathFidelity.HiFi4],
)
def test_matmul_custom_transpose(
    throttle_level,
    in0_tile_r_dim,
    math_fidelity,
    boot_mode=BootMode.DEFAULT,
):
    if (
        in0_tile_r_dim < TILE_DIM
        and get_chip_architecture() != ChipArchitecture.BLACKHOLE
    ):
        pytest.skip("the Wormhole no-mop matmul supports full 32x32 tiles only")

    input_A_dimensions, input_B_dimensions = THROTTLE_DIMS
    _run_matmul_custom(
        math_fidelity,
        input_output_formats([DataFormat.Float16_b])[0],
        DestAccumulation.No,
        input_A_dimensions,
        input_B_dimensions,
        throttle_level=throttle_level,
        boot_mode=boot_mode,
        in0_tile_r_dim=in0_tile_r_dim,
        transpose=Transpose.Yes,
    )
