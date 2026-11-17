# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Quasar softsign: exceptional inputs mixed with ordinary ones across a row pair.

calculate_softsign evaluates two SFPU rows per iteration with their reciprocal chains
interleaved, and each row has its own "is this lane exceptional" predicate that decides whether
the lane keeps the LUT seed or takes the Newton-Raphson steps. A row must get the result it
would get alone, whatever its partner row holds in the same lane. The fuser softsign cases stay
finite and below 65000, so they never make the two predicates disagree.
"""

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DataCopyType,
    DestAccumulation,
    DestSync,
    MathOperation,
    PerfRunType,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import (
    generate_quasar_sfpu_format_variants,
    input_output_formats,
    parametrize,
)
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    LOOP_FACTOR,
    MATH_OP,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    TYPECAST_FORMATS,
    UNPACKER_ENGINE_SEL,
)
from helpers.tile_constants import FACE_C_DIM, MAX_FACE_R_DIM, MAX_NUM_FACES
from helpers.utils import result_tensor

SOFTSIGN_FORMATS = input_output_formats([DataFormat.Float32, DataFormat.Float16_b])

# The reciprocal LUT returns 0 from this denominator up (ckernel_sfpu_recip.h), so softsign
# returns 0 there instead of +-1, on the per-row path as on the paired one.
RECIP_CUTOFF = 2.0**126

# Inputs whose lanes the kernel treats differently from an ordinary one, or sit next to that
# boundary: +-0, +-inf, the reciprocal cutoff and its neighbours. NaNs are planted by bit
# pattern (see _softsign_stimulus).
SPECIAL_VALUES = (
    0.0,
    -0.0,
    float("inf"),
    float("-inf"),
    2.0**125,
    -(2.0**125),
    RECIP_CUTOFF,
    -RECIP_CUTOFF,
    1.5 * RECIP_CUTOFF,
    -1.5 * RECIP_CUTOFF,
    2.0**127,
    -(2.0**127),
)
NAN_SLOTS = 2  # one +NaN and one -NaN slot on top of SPECIAL_VALUES
SIGNED_NAN_BITS = {
    DataFormat.Float32: (torch.int32, 0x7FC00000, -0x00400000),
    DataFormat.Float16_b: (torch.int16, 0x7FC0, -0x0040),
}

# An SFPU row is two Dest rows of a face, and the kernel pairs SFPU rows, so face rows 4k..4k+1
# and 4k+2..4k+3 are the two halves of one pair. Per face: specials in the first half only, in
# the second half only, in both, and in neither.
ROWS_PER_PAIR_HALF = 2
SPECIAL_ROW_GROUPS = (True, False, False, True, True, True, False, False)

# Relative bound on the ordinary lanes, per (approximation mode, 32-bit Dest). The accurate
# kernel on a 32-bit Dest is the one case where a lane that wrongly kept its LUT seed (about
# seven bits) is visible, so its bound is a few fp32 steps; the rest are LUT- or Dest-limited.
ORDINARY_RTOL = {
    (ApproximationMode.No, True): 8 * 2.0**-23,
    (ApproximationMode.No, False): 2 * 2.0**-7,
    (ApproximationMode.Yes, True): 2.0**-6,
    (ApproximationMode.Yes, False): 2.0**-6,
}


def _softsign_stimulus(input_format: DataFormat) -> torch.Tensor:
    """One 32x32 tile, face by face: special and ordinary lanes interleaved (see module doc)."""
    torch_format = format_dict[input_format]
    int_dtype, pos_nan, neg_nan = SIGNED_NAN_BITS[input_format]
    slots = len(SPECIAL_VALUES) + NAN_SLOTS
    gen = torch.Generator().manual_seed(65384)
    face_elems = MAX_FACE_R_DIM * FACE_C_DIM
    flat = (
        torch.rand(MAX_NUM_FACES * face_elems, generator=gen, dtype=torch.float32)
        * 10.0
        - 5.0
    ).to(torch_format)
    bits = flat.view(int_dtype)
    for face in range(MAX_NUM_FACES):
        for row in range(MAX_FACE_R_DIM):
            if not SPECIAL_ROW_GROUPS[row // ROWS_PER_PAIR_HALF]:
                continue
            # Every other column, shifted per row and face: a special lane has ordinary
            # neighbours in its own row, and each special meets every pair position.
            for col in range(row % 2, FACE_C_DIM, 2):
                slot = (col // 2 + 3 * row + 5 * face) % slots
                index = face * face_elems + row * FACE_C_DIM + col
                if slot < len(SPECIAL_VALUES):
                    flat[index] = SPECIAL_VALUES[slot]
                else:
                    bits[index] = (pos_nan, neg_nan)[slot - len(SPECIAL_VALUES)]
    return flat


def _softsign_variants():
    return generate_quasar_sfpu_format_variants(
        MathOperation.Softsign, SOFTSIGN_FORMATS
    )


@pytest.mark.quasar
@parametrize(
    format_variant=_softsign_variants(),
    approx_mode=[ApproximationMode.No, ApproximationMode.Yes],
    dest_sync=[DestSync.Half, DestSync.Full],
)
def test_sfpu_softsign_row_pair_exceptions_quasar(
    format_variant, approx_mode, dest_sync
):
    formats = format_variant.formats
    src_A = _softsign_stimulus(formats.input_format)
    # src_B is unused by a unary op, but StimuliConfig requires an operand-B buffer.
    src_B = torch.zeros_like(src_A)

    configuration = create_test_or_perf_config(
        is_perf=False,
        run_types=(PerfRunType.L1_TO_L1,),
        test_config_kwargs={
            "test_name": "sources/quasar/eltwise_unary_sfpu_quasar_test.cpp",
            "formats": formats,
            "templates": [
                MATH_OP(mathop=MathOperation.Softsign),
                APPROX_MODE(approx_mode),
                DATA_COPY_TYPE(DataCopyType.A2D),
                UNPACKER_ENGINE_SEL(
                    UnpackerEngine.UnpDest
                    if format_variant.unpack_to_dest
                    else UnpackerEngine.UnpA
                ),
                DEST_SYNC(dest_sync),
                TYPECAST_FORMATS(),
            ],
            "runtimes": [
                TILE_COUNT(1),
                NUM_FACES(MAX_NUM_FACES),
                TEST_FACE_DIMS(),
                DEST_INDEX(0),
                LOOP_FACTOR(1),
            ],
            "variant_stimuli": StimuliConfig(
                src_A,
                formats.input_format,
                src_B,
                formats.input_format,
                formats.output_format,
                tile_count_A=1,
                tile_count_B=1,
                tile_count_res=1,
                num_faces=MAX_NUM_FACES,
            ),
            "unpack_to_dest": format_variant.unpack_to_dest,
            "dest_acc": format_variant.dest_acc,
        },
    )
    format_variant.apply_formats(configuration.formats_config)

    res_from_L1 = configuration.run().result
    result = result_tensor(res_from_L1, formats.output_format, src_A).to(torch.float64)

    x = src_A.to(torch.float64)
    is_32bit_dest = format_variant.dest_acc == DestAccumulation.Yes

    # +-inf and NaN: the seed is 0 (or NaN), and inf * 0 is NaN as on the per-row path.
    non_finite = ~torch.isfinite(x)
    assert torch.all(
        torch.isnan(result[non_finite])
    ), "softsign of +-inf / NaN must be NaN"

    # From the reciprocal cutoff up the seed is 0 and so is the result.
    past_cutoff = torch.isfinite(x) & (x.abs() >= RECIP_CUTOFF)
    assert torch.all(
        result[past_cutoff] == 0
    ), "softsign past the reciprocal cutoff must be 0"

    # Every other lane, +-0 and 2**125 included, is x / (1 + |x|) to the kernel's own accuracy,
    # whatever its partner row holds.
    ordinary = torch.isfinite(x) & (x.abs() < RECIP_CUTOFF)
    golden = x[ordinary] / (1 + x[ordinary].abs())
    close = torch.isclose(
        result[ordinary],
        golden,
        rtol=ORDINARY_RTOL[(approx_mode, is_32bit_dest)],
        atol=0.0,
    )
    assert torch.all(close), (
        f"{int((~close).sum())} ordinary lanes off the per-row result, e.g. inputs "
        f"{x[ordinary][~close][:8].tolist()} gave {result[ordinary][~close][:8].tolist()}"
    )


@pytest.mark.quasar
def test_softsign_stimulus_mixes_every_pair_position():
    """Each special sits in the first and in the second half of a row pair, beside ordinary lanes."""
    rows = _softsign_stimulus(DataFormat.Float32).reshape(
        MAX_NUM_FACES, MAX_FACE_R_DIM, FACE_C_DIM
    )
    special = ~torch.isfinite(rows) | (rows == 0) | (rows.abs() >= 2.0**125)
    assert bool(special.any(dim=2).logical_and((~special).any(dim=2)).any())
    half = (torch.arange(MAX_FACE_R_DIM) // ROWS_PER_PAIR_HALF) % 2
    bits = rows.view(torch.int32)
    for position in (0, 1):
        present = set(bits[:, half == position][special[:, half == position]].tolist())
        wanted = set(
            torch.tensor(SPECIAL_VALUES, dtype=torch.float32).view(torch.int32).tolist()
        ) | set(SIGNED_NAN_BITS[DataFormat.Float32][1:])
        assert wanted <= present, f"pair half {position} misses {wanted - present}"
