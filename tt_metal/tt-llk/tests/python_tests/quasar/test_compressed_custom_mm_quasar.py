# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Quasar bring-up test for the compressed_custom_mm LLK.

Exercises _llk_unpack_compressed_custom_mm_ / _llk_math_compressed_custom_mm_, the Quasar port of
the Blackhole compressed_custom_mm (test_compressed_custom_mm.py): in0 [M, kt_dim * 32] with
M in {1, 2, 4, 8}, times in1 [kt_dim * 32, ct_dim * 32] whose tiles each carry their own MX format,
with all-zero tiles not stored. The unpacker takes every weight tile's address and format from the
UNPACR0 ARG_FIFO, fed from the metadata on the RISC-V.

Same host-side approach as the Blackhole test: all three inputs are prepacked L1 images, and the
golden multiplies the dequantized in1, so MX rounding is folded into the golden instead of being
charged against PCC.

Axes:
  * Weight assignment. MX is not a plain format axis here: switching format per tile is the
    feature under test. Each uniform format checks one decode path on its own, "mixed" switches
    formats and interleaves zero tiles (unpacked as zeroed SrcA banks), "zero_rows" adds k-rows
    with no stored tile at all.
  * (ct_dim, kt_dim, M): the grid corners plus odd sizes. The whole output is at most 16 tiles of
    M rows, one dest section by contract, so there is no dest bank switch to cover.
  * Fidelity: LoFi only. MX weights unpack to Float16_b, which get_valid_math_fidelities caps at LoFi.

test_compressed_custom_mm_quasar_deepseek runs the realistic case: the DeepSeek shapes and weight
distribution of the Blackhole test_matmul_custom_compressed_deepseek.
"""

import pytest
import torch
from helpers.compressed_mx_utils import (
    ZERO_TILE,
    assign_random,
    assign_uniform,
    assign_with_zero_rows,
    pack_compressed_in1,
)
from helpers.compressed_utils import DEEPSEEK_T420, FMT_CODE, generate_exact_assignment
from helpers.constraints import get_valid_math_fidelities
from helpers.custom_mm_utils import dense_result_rowmajor, pack_in0_faces
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TILE_DIM, MatmulGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    format_dict,
    format_tile_sizes,
)
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    DEST_SYNC,
    IN_FACE_DIMS,
    MATH_FIDELITY,
    NUM_FACES,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    TILE_COUNT,
)
from helpers.tile_constants import FACE_C_DIM, calculate_tile_size_bytes
from helpers.utils import matmul_acc_atol, passed_test

# in0 is Float16_b. in1's L1 format comes per tile from the metadata; MxInt8, the widest, is what the
# in1 buffer descriptor is programmed with, and format inference maps it to a Float16_b SrcA.
FORMATS = [
    InputOutputFormat(
        DataFormat.Float16_b, DataFormat.Float16_b, input_format_B=DataFormat.MxInt8
    )
]

WEIGHT_ASSIGNMENTS = {
    "mxint8": lambda kt, ct: assign_uniform(kt, ct, DataFormat.MxInt8),
    "mxint4": lambda kt, ct: assign_uniform(kt, ct, DataFormat.MxInt4),
    "mxint2": lambda kt, ct: assign_uniform(kt, ct, DataFormat.MxInt2),
    "mixed": lambda kt, ct: assign_random(kt, ct, zero_fraction=0.25),
    "zero_rows": lambda kt, ct: assign_with_zero_rows(kt, ct),
}

# (ct_dim, kt_dim, M): a single tile, the widest row at full and minimum height, and odd sizes.
CT_KT_ROWS = [
    (1, 1, 1),
    (16, 2, 8),
    (16, 8, 1),
    (5, 3, 4),
    (2, 4, 2),
]

# DeepSeek weight distribution, as in the Blackhole test_matmul_custom_compressed_deepseek: the same
# shapes (M, K, N in datums) and the same tile-granular DEEPSEEK_T420 mix, its BFP levels mapped onto
# the Quasar MX formats of the same width.
DEEPSEEK_SHAPES = [
    (1, 256, 64),  #  8x2
    (1, 896, 32),  # 28x1
    (1, 256, 224),  #  8x7
    (1, 1792, 32),  # 56x1
]
_BFP_CODE_TO_MX = {
    FMT_CODE["bfp8"]: DataFormat.MxInt8,
    FMT_CODE["bfp4"]: DataFormat.MxInt4,
    FMT_CODE["bfp2"]: DataFormat.MxInt2,
    FMT_CODE["bfp0"]: ZERO_TILE,
}
# Each seed redraws the layout at the same distribution.
DEEPSEEK_SEEDS = [0]

# Prepacked operands go to L1 verbatim; StimuliConfig only reserves their regions, in units of one
# Float16_b output-shaped tile (2 * M * 16 datums).
_RESERVATION_FORMAT = DataFormat.Float16_b


def _reserved_tiles(num_bytes, tile_dimensions):
    tile_size = calculate_tile_size_bytes(
        _RESERVATION_FORMAT, tile_dimensions, format_tile_sizes
    )
    return max(1, -(-num_bytes // tile_size))


def _run_compressed_custom_mm(
    formats,
    math_fidelity,
    ct_dim,
    kt_dim,
    in0_rows,
    make_assignment,
    spec=StimuliSpec.uniform(low=0.0, high=1.0),
):
    torch_format = format_dict[formats.output_format]

    in0_dimensions = [in0_rows, kt_dim * TILE_DIM]
    in1_dimensions = [kt_dim * TILE_DIM, ct_dim * TILE_DIM]
    tile_dimensions = [in0_rows, TILE_DIM]

    src_A, _, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=in0_dimensions,
        stimuli_format_B=formats.input_format,  # generated in Float16_b, MX packing below
        input_dimensions_B=in1_dimensions,
        spec_A=spec,
        spec_B=spec,
    )
    in0 = src_A.reshape(in0_dimensions).to(torch_format)
    in1 = src_B.reshape(in1_dimensions).to(torch_format)

    assignment = make_assignment(kt_dim, ct_dim)
    in1_stream, metadata, in1_dequant = pack_compressed_in1(
        in1, kt_dim, ct_dim, assignment
    )

    golden_tensor = get_golden_generator(MatmulGolden)(
        in0,
        in1_dequant,
        formats.output_format,
        math_fidelity,
        input_A_dimensions=in0_dimensions,
        input_B_dimensions=in1_dimensions,
        tilize=False,
        input_A_format=formats.input_format,
        input_B_format=DataFormat.Float16_b,
    ).reshape(in0_rows, ct_dim * TILE_DIM)

    # in0's tiny-tile L1 layout is kt_dim tiles of two dense M x 16 faces, which is exactly the
    # dense face run the Blackhole custom_mm tests use.
    in0_faces = pack_in0_faces(in0, kt_dim, formats.input_format)

    configuration = TestConfig(
        "sources/quasar/compressed_custom_mm_quasar_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            DEST_SYNC(DestSync.Half),
        ],
        runtimes=[
            CRK_TILE_DIMM(ct_dim, 1, kt_dim),
            TILE_COUNT(ct_dim),
            NUM_FACES(num_faces=2, num_faces_A=2, num_faces_B=4),
            IN_FACE_DIMS(
                in0_face_r_dim=in0_rows,
                in0_face_c_dim=FACE_C_DIM,
                in1_face_r_dim=16,
                in1_face_c_dim=FACE_C_DIM,
            ),
            NUM_FACES_R_DIM(num_faces_r_dim_A=1, num_faces_r_dim_B=2),
            NUM_FACES_C_DIM(num_faces_c_dim_A=2, num_faces_c_dim_B=2),
        ],
        variant_stimuli=StimuliConfig(
            buffer_A=in0_faces,
            stimuli_A_format=formats.input_format,
            tile_count_A=kt_dim,
            buffer_B=in1_stream,
            stimuli_B_format=_RESERVATION_FORMAT,
            tile_count_B=_reserved_tiles(len(in1_stream), tile_dimensions),
            buffer_C=metadata,
            stimuli_C_format=_RESERVATION_FORMAT,
            tile_count_C=_reserved_tiles(len(metadata), tile_dimensions),
            stimuli_res_format=formats.output_format,
            tile_count_res=ct_dim,
            num_faces=2,
            face_r_dim=in0_rows,
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
        ),
        unpack_to_dest=False,
        dest_acc=DestAccumulation.No,
    )

    res_from_L1 = configuration.run().result
    res_tensor = dense_result_rowmajor(
        torch.tensor(res_from_L1, dtype=torch_format), ct_dim, in0_rows
    )

    assert (
        res_tensor.numel() == golden_tensor.numel()
    ), "Result tensor and golden tensor are not of the same length"

    assert passed_test(
        golden_tensor.flatten(),
        res_tensor.flatten(),
        formats.output_format,
        custom_atol=matmul_acc_atol(golden_tensor, kt_dim),
    ), "Assert against golden failed"


@pytest.mark.quasar
@parametrize(
    formats=FORMATS,
    math_fidelity=lambda formats: get_valid_math_fidelities(formats),
    weight_assignment=list(WEIGHT_ASSIGNMENTS),
    ct_kt_rows=CT_KT_ROWS,
)
def test_compressed_custom_mm_quasar(
    formats, math_fidelity, weight_assignment, ct_kt_rows
):
    ct_dim, kt_dim, in0_rows = ct_kt_rows
    _run_compressed_custom_mm(
        formats,
        math_fidelity,
        ct_dim,
        kt_dim,
        in0_rows,
        WEIGHT_ASSIGNMENTS[weight_assignment],
    )


@pytest.mark.quasar
@parametrize(
    formats=FORMATS,
    math_fidelity=lambda formats: get_valid_math_fidelities(formats),
    shape=DEEPSEEK_SHAPES,
    seed=DEEPSEEK_SEEDS,
)
def test_compressed_custom_mm_quasar_deepseek(formats, math_fidelity, shape, seed):
    M, K, N = shape

    def make_assignment(kt_dim, ct_dim):
        codes = generate_exact_assignment(K, N, DEEPSEEK_T420, seed=seed)
        return [_BFP_CODE_TO_MX[code] for code in codes]

    _run_compressed_custom_mm(
        formats,
        math_fidelity,
        N // TILE_DIM,
        K // TILE_DIM,
        M,
        make_assignment,
        # Zero-mean, so the long-K outputs (kt up to 56) spread out; with [0, 1] they all land near
        # K/4 and PCC compares little more than bf16 accumulation noise.
        spec=StimuliSpec.uniform(low=-1.0, high=1.0),
    )
