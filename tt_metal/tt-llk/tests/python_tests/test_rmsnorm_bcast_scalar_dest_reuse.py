# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
RMSNorm bcast-scalar DEST-reuse eltwise-binary LLK test (experimental, Blackhole only).

Exercises the experimental ``rmsnorm_bcast_scalar_dest_reuse`` LLKs
(``experimental/llk_math_rmsnorm_bcast_scalar_dest_reuse.h`` /
``experimental/llk_unpack_A_rmsnorm.h``), the strategy open-#2 op promoted by
PR #52709.

The op reuses a value already resident in DEST as a *broadcast scalar* SrcB and
fuses it with a freshly-unpacked input tile via an eltwise-binary FPU op
(``SRCB_BCAST_ALL``):

    DEST[dst][e] = A[e]  (op)  s        for every element e,

where ``s`` is the scalar taken from row 0 of DEST[src] and ``(op)`` is ELWADD
or ELWMUL. The C++ harness seeds DEST[src] with a single constant
``RMSNORM_SCALAR_SEED`` (via the SFPU fill kernel) so ``s`` is uniform and every
output lane is well-defined; the golden therefore validates the whole tile.
0.5 is chosen because it is exactly representable in bf16/fp16/fp32, so the
seeded scalar carries no quantization error into the golden.

Coverage (strategy §4): eltwise {ELWADD, ELWMUL}, num_tiles {1,2,3,7,8} bf16 /
{1..4} fp32, math_fidelity {LoFi, HiFi2, HiFi4} (>LoFi only for ELWMUL, a
hardware constraint mirrored in the LLK's high_fidelity MUL MOP), clear_dest
{False, True}, dest_acc {No, Yes}, num_faces {1, 2, 4} (via tile geometry), and
the blaze-only unpack_full_transpose {False, True} axis (the transpose path in
the LLK supports num_tiles==1 / num_faces==4 only, so it is swept only there).
"""

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import EltwiseBinaryGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    MathFidelity,
    MathOperation,
    format_dict,
)
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    MATH_FIDELITY,
    MATH_OP,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    RMSNORM_CLEAR_DEST,
    RMSNORM_UNPACK_FULL_TRANSPOSE,
    TEST_FACE_DIMS,
    TILE_COUNT,
)
from helpers.tile_shape import construct_tile_shape
from helpers.utils import passed_test

# Must match SCALAR_SEED in sources/rmsnorm_bcast_scalar_dest_reuse_test.cpp
# bit-for-bit. 0.5 is exact in bf16/fp16/fp32.
RMSNORM_SCALAR_SEED = 0.5

# Full 32x32 tile (num_faces=4) plus tiny tiles 16x32 (num_faces=2) and
# 16x16 (num_faces=1). Only num_faces varies; face_r_dim/face_c_dim stay at 16.
TILE_DIMENSIONS = [[32, 32], [16, 32], [16, 16]]


def _formats_for_dest_acc(dest_acc):
    """dest_acc=Yes needs native fp32 DEST, so the input must be Float32.
    dest_acc=No uses bf16 inputs (fp16_b) with a matching bf16 output."""
    if dest_acc == DestAccumulation.Yes:
        return [InputOutputFormat(DataFormat.Float32, DataFormat.Float32)]
    return [InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)]


def _num_tiles_for_dest_acc(dest_acc):
    """DEST half-sync holds 8 bf16 tiles or 4 fp32 tiles; the op writes
    DEST[0..num_tiles-1], so the cap is in DEST tile-slots."""
    return [1, 2, 3, 4] if dest_acc == DestAccumulation.Yes else [1, 2, 3, 7, 8]


def _fidelity_for_op(math_op):
    """Math fidelity > LoFi is only meaningful for ELWMUL (the LLK's
    high_fidelity MOP path is MUL-only); add is fidelity-agnostic."""
    if math_op == MathOperation.Elwmul:
        return [MathFidelity.LoFi, MathFidelity.HiFi2, MathFidelity.HiFi4]
    return [MathFidelity.LoFi]


def _transpose_for_tile(tile_dimensions, num_tiles):
    """The rmsnorm unpack transpose-of-faces path supports num_tiles==1 and
    num_faces==4 (the full 32x32 tile) only, per the LLK's LLK_ASSERTs."""
    if tile_dimensions == [32, 32] and num_tiles == 1:
        return [False, True]
    return [False]


@parametrize(
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    formats=lambda dest_acc: _formats_for_dest_acc(dest_acc),
    math_op=[MathOperation.Elwadd, MathOperation.Elwmul],
    math_fidelity=lambda math_op: _fidelity_for_op(math_op),
    tile_dimensions=TILE_DIMENSIONS,
    num_tiles=lambda dest_acc: _num_tiles_for_dest_acc(dest_acc),
    clear_dest=[False, True],
    unpack_full_transpose=lambda tile_dimensions, num_tiles: _transpose_for_tile(
        tile_dimensions, num_tiles
    ),
)
def test_rmsnorm_bcast_scalar_dest_reuse(
    dest_acc,
    formats,
    math_op,
    math_fidelity,
    tile_dimensions,
    num_tiles,
    clear_dest,
    unpack_full_transpose,
):
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip(
            "rmsnorm_bcast_scalar_dest_reuse is a Blackhole-only experimental LLK"
        )

    tile_shape = construct_tile_shape(tile_dimensions)
    elements_per_tile = tile_shape.total_tile_size()
    torch_format = format_dict[formats.output_format]

    # num_tiles contiguous dense tiles stacked in the row dimension.
    input_dimensions = [num_tiles * tile_dimensions[0], tile_dimensions[1]]

    # A ~ U[-1, 1] keeps A + 0.5 and A * 0.5 well inside the format's range.
    # Dense mode (tile_dimensions passed) lays out the real tiny-tile geometry.
    src_A, tile_cnt_A, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=tile_dimensions,
        spec_A=StimuliSpec.uniform(low=-1.0, high=1.0),
    )

    # The device gets only operand A; the scalar operand is produced on-device by
    # the SFPU fill seed (a uniform RMSNORM_SCALAR_SEED tile). Model that scalar
    # in the golden as a full-tile constant B so EltwiseBinaryGolden applies the
    # same fidelity masking the HW MUL MOP does.
    src_B_scalar = torch.full(
        (tile_cnt_A * elements_per_tile,),
        RMSNORM_SCALAR_SEED,
        dtype=format_dict[formats.input_format],
    )

    binary_golden = get_golden_generator(EltwiseBinaryGolden)
    golden_tensor = binary_golden(
        math_op,
        src_A,
        src_B_scalar,
        formats.output_format,
        math_fidelity,
        input_format=formats.input_format,
        # Scalar is exact (0.5); it is not a BFP/MX tile, so skip re-quantization.
        input_format_B=None,
        tile_shape=tile_shape,
    )

    configuration = TestConfig(
        "sources/rmsnorm_bcast_scalar_dest_reuse_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            MATH_OP(mathop=math_op),
            RMSNORM_CLEAR_DEST(clear_dest=clear_dest),
            RMSNORM_UNPACK_FULL_TRANSPOSE(unpack_full_transpose=unpack_full_transpose),
            # TILE_CNT is consumed as a compile-time template argument in all three
            # threads (e.g. _llk_unpack_A_rmsnorm_init_<TILE_CNT, ...>), so it must be
            # emitted as a file-scope constexpr, not a runtime struct field.
            TILE_COUNT(num_tiles),
            # The rmsnorm addr-mod configurator derives a DEST increment from
            # num_faces (dest.incr = 8 + (4 - num_faces) * 16). If num_faces is a
            # runtime value the whole addr_mod_t aggregate stops folding, and the
            # always-inline addr_mod_t::set() then fails the "n" (immediate) asm
            # constraint on the SETC16 register index ("impossible constraint in
            # 'asm'"). So num_faces must reach the init as a compile-time constant:
            # emit the face-dim counts as file-scope constexprs here.
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim, tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim, tile_shape.num_faces_c_dim),
        ],
        runtimes=[
            TEST_FACE_DIMS(face_r_dim=tile_shape.face_r_dim),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            # B is not streamed from L1 (produced on-device); pass A again just to
            # satisfy the harness's operand-B buffer wiring.
            src_A,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=num_tiles,
            num_faces=tile_shape.total_num_faces(),
            face_r_dim=tile_shape.face_r_dim,
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
        ),
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(golden_tensor), (
        f"Result tensor ({len(res_from_L1)}) and golden tensor "
        f"({len(golden_tensor)}) are not of the same length"
    )

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    # Every output lane is defined (uniform scalar), so validate the whole tile.
    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
