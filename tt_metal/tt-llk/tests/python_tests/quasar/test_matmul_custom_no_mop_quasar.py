# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Quasar bring-up test for the MOP-less matmul LLK.

Exercises _llk_math_matmul_init_no_mop_ / _llk_math_matmul_block_no_mop_. The MOP-based matmul
programs BANK0 with [REPLAY(0, len), matmul_op] x FIDELITY_PHASES (matmul_op_last closing the
last phase); the no-MOP path issues that identical stream from the RISC core so MOP BANK0 stays
free for a fused op. Same replay image and addrmods, so the golden is the ordinary MatmulGolden
any mismatch means the hand-issued stream diverged from what the MOP expands to.

Axes mirror the Blackhole/Wormhole no-MOP test (test_matmul_custom.py): fidelity x format x
dest_acc x dimensions. Two things differ from that test, both forced by the arch:

  * Formats are Quasar's matmul set (test_matmul_quasar.MATMUL_FORMAT), not BH/WH's.
  * The MXFP4_2x variant needs its own test because it changes the replay length (8 MVMULs per
    tile instead of 16) and moves the two closing MVMULs to different addrmod slots. It is the
    variant GPT-OSS needs, so it cannot go untested.

Throttling is not swept: Quasar has no throttled MVMUL sequences, and the LLK API static_asserts
THROTTLE_LEVEL == 0 (same on Wormhole).
"""

import pytest
import torch
from helpers.constraints import get_valid_dest_accumulation_modes
from helpers.data_format_inference import data_formats
from helpers.device import BootMode
from helpers.format_config import DataFormat, InputOutputFormat, is_dest_acc_needed
from helpers.golden_generators import (
    MatmulGolden,
    get_golden_generator,
    quantize_mx_tensor_chunked,
)
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathFidelity,
    Transpose,
    format_dict,
)
from helpers.matmul_sweep import (
    generate_matmul_dimension_combinations,
    generate_tile_dims,
)
from helpers.param_config import (
    DEST_SYNC_TILE_LIMITS,
    input_output_formats,
    parametrize,
    runtime,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    DEST_SYNC,
    ENABLE_2X_FORMAT,
    IMPLIED_MATH_FORMAT,
    MATH_FIDELITY,
    NUM_FACES,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

# The no-MOP LLK replays a four-face MVMUL walk, so it is full-32x32-tile only (the same
# restriction the regular Quasar matmul carries, tt-metal #45208).
NUM_FACES_PER_TILE = 4

# The kernel drives DestSync.Half, matching test_matmul_custom.py. Full-sync only widens the dest
# capacity; it does not touch the REPLAY/MVMUL issue path this test is about.
DEST_SYNC_MODE = DestSync.Half

MATH_FIDELITIES = [
    MathFidelity.LoFi,
    MathFidelity.HiFi2,
    MathFidelity.HiFi3,
    MathFidelity.HiFi4,
]

# Plain (non-2x) formats out of Quasar's matmul set. MX formats are covered by the 2x test below,
# which needs the MX golden machinery anyway.
MATMUL_FORMATS = input_output_formats([DataFormat.Float16, DataFormat.Float16_b])


def _dest_bank_max_tiles(formats, dest_acc):
    """Max result-tile count for a (format, dest_acc) pair on DestSync.Half.

    A 32-bit destination register (dest_acc=Yes, or a format the harness forces onto 32-bit dest)
    holds half as many tiles as the 16-bit one. Mirrors test_matmul_custom._dest_bank_max_tiles.
    """
    capacity_divisor = (
        2 if (is_dest_acc_needed(formats) or dest_acc == DestAccumulation.Yes) else 1
    )
    return DEST_SYNC_TILE_LIMITS[DEST_SYNC_MODE] // capacity_divisor


def _run_matmul_custom_no_mop(
    math_fidelity,
    formats,
    dest_acc,
    implied_math_format,
    input_A_dimensions,
    input_B_dimensions,
    enable_2x_format: bool,
    boot_mode: BootMode,
):
    torch_format = format_dict[formats.output_format]

    stimuli_spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_A_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_B_dimensions,
        spec_A=stimuli_spec,
        spec_B=stimuli_spec,
        output_format=formats.output_format,
    )

    tilized_A = tilize_block(
        src_A, dimensions=input_A_dimensions, stimuli_format=formats.input_format
    )
    tilized_B = tilize_block(
        src_B, dimensions=input_B_dimensions, stimuli_format=formats.input_format
    )

    matmul_dims = generate_tile_dims((input_A_dimensions, input_B_dimensions))

    # MX inputs are quantized onto their lattice by the unpacker, so golden has to see
    # quantized operands rather than the raw stimuli (mirrors test_matmul_quasar).
    src_A_golden = src_A
    src_B_golden = src_B
    if formats.input_format.is_mx_format():
        tilized_A_golden = quantize_mx_tensor_chunked(
            tilized_A.flatten().to(torch.bfloat16), formats.input_format
        ).reshape(tilized_A.shape)
        tilized_B_golden = quantize_mx_tensor_chunked(
            tilized_B.flatten().to(torch.bfloat16), formats.input_format
        ).reshape(tilized_B.shape)
        src_A_golden = untilize_block(
            tilized_A_golden,
            stimuli_format=formats.input_format,
            dimensions=input_A_dimensions,
        )
        src_B_golden = untilize_block(
            tilized_B_golden,
            stimuli_format=formats.input_format,
            dimensions=input_B_dimensions,
        )

    # 2x register-format opt-in has to flow through inference; only disable inference for plain MX
    # formats, where there is nothing to infer.
    disable_format_inference = (
        formats.input_format.is_mx_format() and formats.register_format_hint is None
    )

    formats_config = data_formats(
        input_format=formats.input_format,
        input_format_B=formats.input_format_B,
        output_format=formats.output_format,
        is_fp32_dest_acc_en=dest_acc,
        num_iterations=1,
        unpacking_to_dest=False,
        disable_format_inference=disable_format_inference,
        register_format_hint=formats.register_format_hint,
    )[0]
    pack_src_format = formats_config.pack_src

    generate_golden = get_golden_generator(MatmulGolden)
    golden_tensor = generate_golden(
        src_A_golden,
        src_B_golden,
        formats.output_format,
        math_fidelity,
        input_A_dimensions=input_A_dimensions,
        input_B_dimensions=input_B_dimensions,
        # Golden cannot model FPU strided for tilized data computation, so we tilize output after computation
        tilize=True,
        input_A_format=formats.input_format,
        input_B_format=formats.input_format,
        # For accumulation of results in matmul we require to calculate in pack_src_format.
        math_format=pack_src_format,
        dest_acc=dest_acc,
    )

    configuration = TestConfig(
        test_name="sources/quasar/matmul_custom_no_mop_quasar_test.cpp",
        formats=formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            IMPLIED_MATH_FORMAT(implied_math_format),
            ENABLE_2X_FORMAT(enable_2x_format),
            DEST_SYNC(DEST_SYNC_MODE),
            UNPACK_TRANS_FACES(Transpose.No),
        ],
        runtimes=[
            CRK_TILE_DIMM(matmul_dims.ct_dim, matmul_dims.rt_dim, matmul_dims.kt_dim),
            TILE_COUNT(matmul_dims.output_tile_cnt),
            NUM_FACES(NUM_FACES_PER_TILE, NUM_FACES_PER_TILE, NUM_FACES_PER_TILE),
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
            num_faces=NUM_FACES_PER_TILE,
        ),
        unpack_to_dest=False,
        dest_acc=dest_acc,
        disable_format_inference=disable_format_inference,
        boot_mode=boot_mode,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    # For MX outputs, model the packer: quantize the golden onto the MX lattice (from the math /
    # pack_src format the result was produced in) so the comparison validates the device's MX output
    # quantization, not just matmul-math-to-MX precision.
    if formats.output_format.is_mx_format():
        golden_tensor = quantize_mx_tensor_chunked(
            golden_tensor.to(format_dict[pack_src_format]), formats.output_format
        ).to(torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"


@pytest.mark.quasar
@parametrize(
    math_fidelity=MATH_FIDELITIES,
    formats=MATMUL_FORMATS,
    dest_acc=lambda formats: get_valid_dest_accumulation_modes(formats),
    dimensions=runtime(
        lambda formats, dest_acc: generate_matmul_dimension_combinations(
            _dest_bank_max_tiles(formats, dest_acc)
        )
    ),
)
def test_matmul_custom_no_mop_quasar(
    math_fidelity,
    formats,
    dest_acc,
    dimensions,
    boot_mode=BootMode.DEFAULT,
):
    input_A_dimensions, input_B_dimensions = dimensions
    _run_matmul_custom_no_mop(
        math_fidelity,
        formats,
        dest_acc,
        # The metal llk_api wrapper configures the ALU format state with IMPLIED_MATH_FORMAT off,
        # so the non-MX path is tested in that mode.
        ImpliedMathFormat.No,
        input_A_dimensions,
        input_B_dimensions,
        enable_2x_format=False,
        boot_mode=boot_mode,
    )


# MXFP4 unpacked as MxFp4_2x_A/B in the src registers: SrcA expands two sub-datums per element, so a
# tile takes 8 MVMULs instead of 16 and the two closing MVMULs move to ADDR_MOD_4/ADDR_MOD_5 (from
# ADDR_MOD_5/ADDR_MOD_3). That is exactly the wiring the no-MOP path has to reproduce by hand, so it
# gets its own sweep. MxFp4 is input-only here, the packer never writes it. So the output stays a
# plain 16-bit float format.
MATMUL_2X_FORMATS = [
    InputOutputFormat(DataFormat.MxFp4, DataFormat.Float16_b, register_format_hint=hint)
    for hint in (DataFormat.MxFp4_2x_A, DataFormat.MxFp4_2x_B)
]

# One tile-shape per K depth is enough: the 2x variant only changes the per-tile MVMUL sequence, and
# the block/K walk around it is shared with the plain path covered above.
MATMUL_2X_DIMENSIONS = [
    ([32, 32], [32, 32]),
    ([32, 64], [64, 32]),
    ([64, 64], [64, 64]),
]


@pytest.mark.quasar
@parametrize(
    math_fidelity=MATH_FIDELITIES,
    formats=MATMUL_2X_FORMATS,
    dest_acc=lambda formats: get_valid_dest_accumulation_modes(formats),
    dimensions=MATMUL_2X_DIMENSIONS,
)
def test_matmul_custom_no_mop_quasar_mxfp4_2x(
    math_fidelity,
    formats,
    dest_acc,
    dimensions,
    boot_mode=BootMode.DEFAULT,
):
    input_A_dimensions, input_B_dimensions = dimensions
    _run_matmul_custom_no_mop(
        math_fidelity,
        formats,
        dest_acc,
        # MX formats carry their exponent in the data, so the math format has to be implied.
        ImpliedMathFormat.Yes,
        input_A_dimensions,
        input_B_dimensions,
        enable_2x_format=True,
        boot_mode=boot_mode,
    )
