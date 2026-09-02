# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Regression guard: a REDUCE_ROW MAX must survive a Src zero-substitution flag clobber.

REDUCE_ROW MAX is the reduce path the flag hoist changed, and reduce_row_perform_transpose is its
mov phase: it moves the pooled row DEST -> SrcB (MOVD2B/TRNSPSRCB) and adds it back with ELWADD.
(It is not the only reduce path with a mov phase -- REDUCE_SCALAR moves via MOVD2B + 4x MOVB2A --
but scalar's flag value is unchanged by that hoist, so it is not under test here.) Those readers need
ALU_ACC_CTRL_Zero_Flag_disabled_src SET (PRESERVE / no zero substitution): with it clear, a datum
whose low byte is zero is flushed to 0 mid-reduction. That is the failure #46511 describes as
"bf16 values with a zero low byte got flushed mid-sum and layernorm drifted off".

#46511 deliberately (re)asserts the op-need in the EXECUTE path "so it survives an
llk_math_hw_configure that runs after the op init". This test pins that behaviour down: it
clobbers the flag after reduce_init the way a real tt-metal compute kernel would, then reduces.

Two axes matter:

* ``clobber`` -- how the state gets polluted (see ZERO_FLAG_CLOBBER in
  helpers/test_variant_parameters.py). Mode 4 is a raw forced flush, so a pass cannot be
  explained away as "the clobber never moved the bit".
* ``fill_constant`` -- every reduced datum is a single constant, so the REDUCE_ROW MAX result is
  analytically the constant itself regardless of tile layout, and a flushed datum shows up as 0
  rather than as a small numerical drift. TRIGGER_CONSTANTS have a zero low byte in Float16_b
  (the datum the flag flushes); CONTROL_CONSTANTS do not, and must pass in every mode.

Blackhole-only: the hoist-to-init this guards against (and the execute-path re-assert that fixes
it) are in tt_llk_blackhole/llk_lib/llk_math_reduce.h.

MEASURED STATUS (Blackhole p300a, 2026-09-01) -- read this before trusting the test to discriminate.
The whole 70-case matrix passes BOTH with and without the execute-path re-assert, including the raw
forced-flush mode on every zero-low-byte constant. The clobber is not the reason: the kernel reads
ALU_ACC_CTRL_Zero_Flag_disabled_src back from the config register and skips the reduce if the
pollution did not land, and inverting that comparison makes the clobbered cases fail while the
control case still passes -- so the flag really is observed CLEAR (zero-substituting) across the
reduce, and the reduce still matches golden anyway.

So on Blackhole the flag has no observable effect on this path for the float formats REDUCE_ROW MAX
supports, and the PRESERVE that main, #46511 and the execute-path re-assert all maintain here is
currently unfalsifiable. This test therefore pins the STATE MACHINE (the clobber lands; the op still
matches golden), not a reproducible miscompute. Two things would make it bite, neither reachable
through this path today: integer operands (the documented hazard is 16-bit ints -- but UInt16
already gets PRESERVE from the operand-driven default, and GMPOOL/reduce golden are float-only), and
the layernorm-level drift #46511 actually observed, which needs a ttnn test rather than an LLK one.
"""

import torch
from conftest import blackhole_only
from helpers.format_config import DataFormat
from helpers.golden_generators import ReduceGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    MathFidelity,
    MathOperation,
    ReduceDimension,
    ReducePool,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    IN_FACE_DIMS,
    INPUT_TILE_CNT,
    MATH_FIDELITY,
    MATH_OP,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    NUM_TILES_IN_BLOCK,
    OUTPUT_TILE_CNT,
    ZERO_FLAG_CLOBBER,
    ZERO_FLAG_CLOBBER_PER_TILE,
)
from helpers.tile_shape import construct_tile_shape
from helpers.utils import passed_test

pytestmark = blackhole_only

# Float16_b encodings with a zero low byte -- exactly the datum the zero-substitution flag flushes.
# (The value cited in llk_math_reduce.h is 0x4400; note that is 512.0, not 768.0 -- 768.0 encodes as
# 0x4440, low byte 0x40, so it is NOT a trigger.)
TRIGGER_CONSTANTS = (2.0, 8.0, 32.0, 512.0)

# Same magnitude range, non-zero low byte. These must pass under every clobber mode; if they ever
# fail, the test is measuring something other than zero substitution.
CONTROL_CONSTANTS = (1.5, 12.0, 768.0)

# 0=none, 1=reconfig_data_format, 2=llk_math_hw_configure, 3=fp8 copy init, 4=raw forced flush.
CLOBBER_MODES = (0, 1, 2, 3, 4)


@parametrize(
    # Float16_b in/out with dest_acc=No is the vulnerable configuration: the flag is ignored when
    # fp32 DEST accumulation is on, so a 32-bit format would make the test vacuous.
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    fill_constant=list(TRIGGER_CONSTANTS + CONTROL_CONSTANTS),
    clobber=list(CLOBBER_MODES),
    clobber_per_tile=[False, True],
    tile_dimensions=[[32, 32]],
)
def test_reduce_row_max_zero_flag_clobber(
    formats,
    fill_constant,
    clobber,
    clobber_per_tile,
    tile_dimensions,
):
    tile_shape = construct_tile_shape(tile_dimensions)
    input_dimensions = [
        64,
        32,
    ]  # 2 tiles, so the per-tile re-assert is exercised more than once

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=tile_dimensions,
        tile_dimensions=tile_dimensions,
    )

    # Constant fill: the row MAX is analytically the constant for every row, whatever the face
    # layout, so a flushed datum reads as 0 instead of as numerical drift.
    src_A = torch.full_like(src_A, float(fill_constant))
    src_B = torch.full((tile_shape.total_tile_size(),), 1)  # MAX divides by 1

    generate_golden = get_golden_generator(ReduceGolden)
    golden_tensor = generate_golden(
        src_A,
        ReduceDimension.Row,
        ReducePool.Max,
        formats.output_format,
        tile_cnt_A,
        reduce_to_one=False,
        tile_shape=tile_shape,
        input_format=formats.input_format,
    )

    configuration = TestConfig(
        "sources/reduce_zero_flag_clobber_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=MathOperation.ReduceRow, pool_type=ReducePool.Max),
            MATH_FIDELITY(MathFidelity.HiFi4),
            ZERO_FLAG_CLOBBER(zero_flag_clobber=clobber),
            ZERO_FLAG_CLOBBER_PER_TILE(zero_flag_clobber_per_tile=clobber_per_tile),
        ],
        runtimes=[
            IN_FACE_DIMS(
                tile_shape.face_r_dim,
                tile_shape.face_c_dim,
                tile_shape.face_r_dim,
                tile_shape.face_c_dim,
            ),
            INPUT_TILE_CNT(tile_cnt_A),
            OUTPUT_TILE_CNT(tile_cnt_A),
            NUM_TILES_IN_BLOCK(tile_cnt_A),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim),
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
            num_faces=tile_shape.total_num_faces(),
            face_r_dim=tile_shape.face_r_dim,
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
        ),
        dest_acc=DestAccumulation.No,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        tile_shape=tile_shape,
    ), (
        f"REDUCE_ROW MAX drifted after a zero-flag clobber "
        f"(mode={clobber}, per_tile={clobber_per_tile}, fill={fill_constant})"
    )
