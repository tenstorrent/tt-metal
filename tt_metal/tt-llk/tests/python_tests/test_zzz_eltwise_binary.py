# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    BroadcastGolden,
    EltwiseBinaryGolden,
    TransposeGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BlocksCalculationAlgorithm,
    BroadcastType,
    DestAccumulation,
    DestSync,
    EltwiseBinaryReuseDestType,
    MathFidelity,
    MathOperation,
    Transpose,
    format_dict,
)
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import BuildMode, TestConfig
from helpers.test_variant_parameters import (
    BROADCAST_TYPE,
    DEST_SYNC,
    EN_DEST_REUSE,
    MATH_FIDELITY,
    MATH_OP,
    NUM_BLOCKS,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    NUM_TILES_IN_BLOCK,
    REUSE_DEST_TYPE,
    TEST_FACE_DIMS,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)
from helpers.tile_constants import FACE_C_DIM, SUPPORTED_TILE_SIZES, get_tile_params
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test

ALL_TILE_DIMENSIONS = [list(td) for td in SUPPORTED_TILE_SIZES]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _get_valid_formats(dest_acc):
    """
    Filter formats based on dest accumulation:
    - If dest accumulation is enabled, input must be Float32
    """
    all_formats = input_output_formats(
        [
            DataFormat.Bfp4_b,
            DataFormat.Bfp8_b,
            DataFormat.Float16_b,
            DataFormat.Float32,
        ],
        same=False,
    )
    if dest_acc == DestAccumulation.Yes:
        return [f for f in all_formats if f.input_format == DataFormat.Float32]
    return all_formats


def _get_valid_math_fidelity(formats, math_op=None):
    """
    Filter math fidelity based on input data format:
    - Bfp8_b: LoFi only
    - Float16_b: LoFi or HiFi2
    - Float32: HiFi3 and HiFi4

    Math fidelity > LoFi is only supported for Elwmul (hardware constraint),
    so non-multiply ops are restricted to LoFi regardless of format.
    """
    if math_op is not None and math_op != MathOperation.Elwmul:
        return [MathFidelity.LoFi]
    input_format = formats.input_format
    if input_format in [DataFormat.Bfp8_b, DataFormat.Bfp4_b]:
        return [MathFidelity.LoFi]
    elif input_format == DataFormat.Float16_b:
        return [MathFidelity.LoFi, MathFidelity.HiFi2]
    elif input_format == DataFormat.Float32:
        return [MathFidelity.HiFi3, MathFidelity.HiFi4]
    return [
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ]


def _get_valid_tile_dimensions(transpose_srca, broadcast_type):
    """
    Filter tile dimensions based on transpose and broadcast constraints:
    - Transpose only works for 32x32 tiles
    - 32x16 tiles are not supported for Column or Row broadcast
    """
    if transpose_srca == Transpose.Yes:
        return [[32, 32]]

    if broadcast_type in (BroadcastType.Column, BroadcastType.Row):
        return [td for td in ALL_TILE_DIMENSIONS if td != [32, 16]]

    return ALL_TILE_DIMENSIONS


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, repr=False)
class EltwiseBinaryConfig:
    dest_acc: DestAccumulation
    formats: InputOutputFormat
    math_op: MathOperation
    math_fidelity: MathFidelity
    broadcast_type: BroadcastType
    transpose_srca: Transpose
    input_dimensions: tuple
    tile_dimensions: tuple

    def __repr__(self):
        f = self.formats
        return (
            f"{self.dest_acc.name}-{f.input_format.name}->{f.output_format.name}"
            f"-{self.math_op.name}-{self.math_fidelity.name}"
            f"-{self.broadcast_type.name}-tr_{self.transpose_srca.name}"
            f"-{self.input_dimensions[0]}x{self.input_dimensions[1]}"
            f"-tile{self.tile_dimensions[0]}x{self.tile_dimensions[1]}"
        )


@dataclass(frozen=True, repr=False)
class EltwiseBinaryDestReuseConfig:
    reuse_dest_type: EltwiseBinaryReuseDestType
    formats: InputOutputFormat
    math_fidelity: MathFidelity
    math_op: MathOperation
    input_dimensions: tuple
    output_dimensions: tuple
    tile_dimensions: tuple

    def __repr__(self):
        f = self.formats
        return (
            f"{self.reuse_dest_type.name}-{f.input_format.name}->{f.output_format.name}"
            f"-{self.math_fidelity.name}-{self.math_op.name}"
            f"-in{self.input_dimensions[0]}x{self.input_dimensions[1]}"
            f"-out{self.output_dimensions[0]}x{self.output_dimensions[1]}"
            f"-tile{self.tile_dimensions[0]}x{self.tile_dimensions[1]}"
        )


# ---------------------------------------------------------------------------
# Sweep builders
# ---------------------------------------------------------------------------


def _sweep_eltwise_binary():
    """Pre-build all valid EltwiseBinaryConfig combinations.

    Loop order — template-affecting (hash) params outermost, runtime innermost:
    dest_acc → math_op → broadcast_type → math_fidelity → formats → transpose_srca → input_dimensions → tile_dimensions

    formats doesn't affect the compile hash (compile_time_formats=False,
    unpack_to_dest=False), so it's placed after all template params.
    """
    combos = []
    for da in [DestAccumulation.No, DestAccumulation.Yes]:
        for mop in [MathOperation.Elwmul, MathOperation.Elwadd, MathOperation.Elwsub]:
            for bt in [
                BroadcastType.None_,
                BroadcastType.Row,
                BroadcastType.Column,
                BroadcastType.Scalar,
            ]:
                for fmt in _get_valid_formats(da):
                    for mf in _get_valid_math_fidelity(fmt, mop):
                        for tr in [Transpose.Yes, Transpose.No]:
                            if tr == Transpose.Yes and bt == BroadcastType.Scalar:
                                continue
                            for idims in [[256, 32]]:
                                for td in _get_valid_tile_dimensions(tr, bt):
                                    combos.append(
                                        EltwiseBinaryConfig(
                                            dest_acc=da,
                                            formats=fmt,
                                            math_op=mop,
                                            math_fidelity=mf,
                                            broadcast_type=bt,
                                            transpose_srca=tr,
                                            input_dimensions=tuple(idims),
                                            tile_dimensions=tuple(td),
                                        )
                                    )
    return combos


def _sweep_eltwise_binary_bfp4_b():
    """Pre-build all valid EltwiseBinaryConfig combinations for bfp4_b test.

    Loop order — template-affecting outermost, runtime innermost:
    dest_acc → math_op → broadcast_type → formats → math_fidelity
    → transpose_srca → input_dimensions → tile_dimensions
    """
    bfp4_formats = [
        fmt
        for fmt in input_output_formats(
            [
                DataFormat.Bfp4_b,
                DataFormat.Float16_b,
                DataFormat.Bfp8_b,
                DataFormat.Float32,
            ]
        )
        if fmt.input_format == DataFormat.Bfp4_b
    ]
    combos = []
    for da in [DestAccumulation.No, DestAccumulation.Yes]:
        for mop in [MathOperation.Elwadd, MathOperation.Elwsub]:
            for bt in [
                BroadcastType.None_,
                BroadcastType.Row,
                BroadcastType.Column,
                BroadcastType.Scalar,
            ]:
                for fmt in bfp4_formats:
                    for mf in _get_valid_math_fidelity(fmt):
                        for tr in [Transpose.No]:
                            for idims in [[32, 32], [64, 32], [32, 64], [256, 32]]:
                                for td in _get_valid_tile_dimensions(tr, bt):
                                    combos.append(
                                        EltwiseBinaryConfig(
                                            dest_acc=da,
                                            formats=fmt,
                                            math_op=mop,
                                            math_fidelity=mf,
                                            broadcast_type=bt,
                                            transpose_srca=tr,
                                            input_dimensions=tuple(idims),
                                            tile_dimensions=tuple(td),
                                        )
                                    )
    return combos


def _sweep_eltwise_binary_dest_reuse():
    """Pre-build all valid EltwiseBinaryDestReuseConfig combinations.

    Loop order (template-affecting outer, runtime inner):
    reuse_dest_type -> formats -> math_fidelity -> math_op
    -> input_dimensions -> output_dimensions -> tile_dimensions
    """
    dest_reuse_formats = input_output_formats(
        [DataFormat.Float16_b, DataFormat.Float32, DataFormat.Bfp8_b],
        same=True,
    )
    combos = []
    for rdt in [
        EltwiseBinaryReuseDestType.DEST_TO_SRCA,
        EltwiseBinaryReuseDestType.DEST_TO_SRCB,
    ]:
        for fmt in dest_reuse_formats:
            for mf in [MathFidelity.LoFi]:
                for mop in [
                    MathOperation.Elwadd,
                    MathOperation.Elwsub,
                    MathOperation.Elwmul,
                ]:
                    for idims in [[512, 32]]:
                        for odims in [[128, 32]]:
                            for td in [[32, 32], [16, 32]]:
                                combos.append(
                                    EltwiseBinaryDestReuseConfig(
                                        reuse_dest_type=rdt,
                                        formats=fmt,
                                        math_fidelity=mf,
                                        math_op=mop,
                                        input_dimensions=tuple(idims),
                                        output_dimensions=tuple(odims),
                                        tile_dimensions=tuple(td),
                                    )
                                )
    return combos


def _sweep_eltwise_binary_int8():
    """Pre-build all valid EltwiseBinaryConfig combinations for int8 test.

    Loop order (template-affecting outer, runtime inner):
    dest_acc -> formats -> math_op -> math_fidelity -> broadcast_type
    -> transpose_srca -> input_dimensions -> tile_dimensions
    """
    combos = []
    fmt = InputOutputFormat(DataFormat.Int8, DataFormat.Int8)
    for da in [DestAccumulation.Yes]:
        for mop in [MathOperation.Elwadd, MathOperation.Elwsub]:
            for mf in [MathFidelity.LoFi]:
                for bt in [BroadcastType.None_]:
                    for tr in [Transpose.No]:
                        for idims in [[32, 32], [512, 32]]:
                            for td in _get_valid_tile_dimensions(tr, bt):
                                combos.append(
                                    EltwiseBinaryConfig(
                                        dest_acc=da,
                                        formats=fmt,
                                        math_op=mop,
                                        math_fidelity=mf,
                                        broadcast_type=bt,
                                        transpose_srca=tr,
                                        input_dimensions=tuple(idims),
                                        tile_dimensions=tuple(td),
                                    )
                                )
    return combos


# ---------------------------------------------------------------------------
# Pre-built config/ID lists
# ---------------------------------------------------------------------------

ALL_ELTWISE_BINARY_CONFIGS = _sweep_eltwise_binary()

ALL_BFP4B_CONFIGS = _sweep_eltwise_binary_bfp4_b()

ALL_DEST_REUSE_CONFIGS = _sweep_eltwise_binary_dest_reuse()

ALL_INT8_CONFIGS = _sweep_eltwise_binary_int8()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", ALL_ELTWISE_BINARY_CONFIGS)
def test_eltwise_binary(config: EltwiseBinaryConfig):
    dest_acc = config.dest_acc
    formats = config.formats
    math_op = config.math_op
    math_fidelity = config.math_fidelity
    broadcast_type = config.broadcast_type
    transpose_srca = config.transpose_srca
    input_dimensions = list(config.input_dimensions)
    tile_dimensions = list(config.tile_dimensions)

    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dimensions)
    num_faces = num_faces_r_dim * num_faces_c_dim

    tile_rows, tile_cols = tile_dimensions
    tile_cnt_A = (input_dimensions[0] // tile_rows) * (input_dimensions[1] // tile_cols)
    tile_cnt_B = tile_cnt_A

    effective_dest_acc = (
        DestAccumulation.Yes
        if formats.output_format == DataFormat.Float32
        else dest_acc
    )
    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        effective_dest_acc,
        formats,
        input_dimensions,
        tile_dimensions,
        BlocksCalculationAlgorithm.Standard,
    )

    stimuli = StimuliConfig(
        None,
        formats.input_format,
        None,
        formats.input_format,
        formats.output_format,
        tile_count_A=tile_cnt_A,
        tile_count_B=tile_cnt_B,
        tile_count_res=tile_cnt_A,
        num_faces=num_faces,
        face_r_dim=face_r_dim,
        tile_dimensions=tile_dimensions,
        use_dense_tile_dimensions=True,
    )

    configuration = TestConfig(
        "sources/eltwise_binary_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            BROADCAST_TYPE(broadcast_type),
            MATH_OP(mathop=math_op),
            DEST_SYNC(),
        ],
        runtimes=[
            UNPACK_TRANS_FACES(transpose_srca),
            UNPACK_TRANS_WITHIN_FACE(transpose_srca),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            NUM_BLOCKS(num_blocks),
            NUM_FACES_R_DIM(num_faces_r_dim),
            NUM_FACES_C_DIM(num_faces_c_dim),
            TEST_FACE_DIMS(face_r_dim=face_r_dim),
        ],
        variant_stimuli=stimuli,
        dest_acc=dest_acc,
        unpack_to_dest=False,
    )

    configuration.prepare()
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        pytest.skip(TestConfig.SKIP_JUST_FOR_COMPILE_MARKER)

    # Generate stimuli with correct face dimensions for smaller tiles
    src_A, _, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format_B,
        input_dimensions_B=input_dimensions,
        tile_dimensions=tile_dimensions,
    )

    binary_golden = get_golden_generator(EltwiseBinaryGolden)

    src_A_tilized = tilize_block(
        src_A,
        dimensions=input_dimensions,
        stimuli_format=formats.input_format,
        num_faces=num_faces,
        tile_dimensions=tile_dimensions,
        face_r_dim=face_r_dim,
    )
    src_B_tilized = tilize_block(
        src_B,
        dimensions=input_dimensions,
        stimuli_format=formats.input_format,
        num_faces=num_faces,
        tile_dimensions=tile_dimensions,
        face_r_dim=face_r_dim,
    )

    src_A_tilized_flat = src_A_tilized.flatten()
    src_B_tilized_flat = src_B_tilized.flatten()

    stimuli_A = src_A_tilized_flat
    stimuli_B = src_B_tilized_flat

    # Prepare golden src_A: apply tile-level transpose if enabled
    # Hardware does transpose_faces then transpose_within_faces during unpack
    golden_src_A = src_A_tilized_flat
    if transpose_srca == Transpose.Yes:
        transpose_golden = get_golden_generator(TransposeGolden)
        # Apply face transpose — also quantizes BFP formats to float16_b
        golden_src_A = transpose_golden.transpose_faces_multi_tile(
            src_A,
            formats.input_format,
            num_tiles=tile_cnt_A,
            tilize=True,
            untilize=False,
            input_dimensions=tuple(input_dimensions),
        )
        # Apply within-face transpose on already-quantized data.
        # Use Float32 to match hardware source register precision (TF32)
        # and avoid double-quantization — hardware only quantizes once
        # during unpack, then transposes at full precision.
        golden_src_A = transpose_golden.transpose_within_faces_multi_tile(
            golden_src_A,
            (
                DataFormat.Float16_b
                if formats.input_format in [DataFormat.Bfp4_b, DataFormat.Bfp8_b]
                else formats.input_format
            ),
            num_tiles=tile_cnt_A,
            tilize=False,
            untilize=False,
            input_dimensions=tuple(input_dimensions),
        )

    # Prepare golden src_B: apply broadcast if enabled
    golden_src_B = src_B_tilized_flat
    if broadcast_type != BroadcastType.None_:
        broadcast_golden = get_golden_generator(BroadcastGolden)
        golden_src_B = broadcast_golden(
            broadcast_type,
            src_B_tilized_flat,
            formats.input_format,
            num_faces=num_faces,
            tile_cnt=tile_cnt_A,
            face_r_dim=face_r_dim,
        )

    # When transpose/broadcast already quantized an operand (BFP -> float16_b),
    # pass None to skip re-quantization in EltwiseBinaryGolden.
    golden_input_format_A = (
        None if transpose_srca == Transpose.Yes else formats.input_format
    )
    golden_input_format_B = (
        None if broadcast_type != BroadcastType.None_ else formats.input_format
    )
    golden_tensor = binary_golden(
        math_op,
        golden_src_A,
        golden_src_B,
        formats.output_format,
        math_fidelity,
        input_format=golden_input_format_A,
        input_format_B=golden_input_format_B,
    )

    stimuli.set_buffers(stimuli_A, stimuli_B)

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), f"Result tensor ({len(res_from_L1)}) and golden tensor ({len(golden_tensor)}) are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"


@pytest.mark.parametrize("config", ALL_BFP4B_CONFIGS)
def test_eltwise_binary_bfp4_b(config: EltwiseBinaryConfig):
    dest_acc = config.dest_acc
    formats = config.formats
    math_op = config.math_op
    math_fidelity = config.math_fidelity
    broadcast_type = config.broadcast_type
    transpose_srca = config.transpose_srca
    input_dimensions = list(config.input_dimensions)
    tile_dimensions = list(config.tile_dimensions)

    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dimensions)
    num_faces = num_faces_r_dim * num_faces_c_dim

    tile_rows, tile_cols = tile_dimensions
    tile_cnt_A = (input_dimensions[0] // tile_rows) * (input_dimensions[1] // tile_cols)
    tile_cnt_B = tile_cnt_A

    effective_dest_acc = (
        DestAccumulation.Yes
        if formats.output_format == DataFormat.Float32
        else dest_acc
    )
    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        effective_dest_acc,
        formats,
        input_dimensions,
        tile_dimensions,
        BlocksCalculationAlgorithm.Standard,
    )

    stimuli = StimuliConfig(
        None,
        formats.input_format,
        None,
        formats.input_format,
        formats.output_format,
        tile_count_A=tile_cnt_A,
        tile_count_B=tile_cnt_B,
        tile_count_res=tile_cnt_A,
        num_faces=num_faces,
        face_r_dim=face_r_dim,
        tile_dimensions=tile_dimensions,
        use_dense_tile_dimensions=True,
    )

    configuration = TestConfig(
        "sources/eltwise_binary_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            BROADCAST_TYPE(broadcast_type),
            MATH_OP(mathop=math_op),
            DEST_SYNC(),
        ],
        runtimes=[
            UNPACK_TRANS_FACES(transpose_srca),
            UNPACK_TRANS_WITHIN_FACE(transpose_srca),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            NUM_BLOCKS(num_blocks),
            NUM_FACES_R_DIM(num_faces_r_dim),
            NUM_FACES_C_DIM(num_faces_c_dim),
            TEST_FACE_DIMS(face_r_dim=face_r_dim),
        ],
        variant_stimuli=stimuli,
        dest_acc=dest_acc,
        unpack_to_dest=False,
    )

    configuration.prepare()
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        pytest.skip(TestConfig.SKIP_JUST_FOR_COMPILE_MARKER)

    src_A, _, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format_B,
        input_dimensions_B=input_dimensions,
        tile_dimensions=tile_dimensions,
    )

    binary_golden = get_golden_generator(EltwiseBinaryGolden)

    src_A_tilized = tilize_block(
        src_A,
        dimensions=input_dimensions,
        stimuli_format=formats.input_format,
        num_faces=num_faces,
        tile_dimensions=tile_dimensions,
        face_r_dim=face_r_dim,
    )
    src_B_tilized = tilize_block(
        src_B,
        dimensions=input_dimensions,
        stimuli_format=formats.input_format,
        num_faces=num_faces,
        tile_dimensions=tile_dimensions,
        face_r_dim=face_r_dim,
    )

    src_A_tilized_flat = src_A_tilized.flatten()
    src_B_tilized_flat = src_B_tilized.flatten()

    stimuli_A = src_A_tilized_flat
    stimuli_B = src_B_tilized_flat

    golden_src_A = src_A_tilized_flat
    if transpose_srca == Transpose.Yes:
        transpose_golden = get_golden_generator(TransposeGolden)
        golden_src_A = transpose_golden.transpose_faces_multi_tile(
            src_A,
            formats.input_format,
            num_tiles=tile_cnt_A,
            tilize=True,
            untilize=False,
            input_dimensions=tuple(input_dimensions),
        )
        golden_src_A = transpose_golden.transpose_within_faces_multi_tile(
            golden_src_A,
            formats.input_format,
            num_tiles=tile_cnt_A,
            tilize=False,
            untilize=False,
            input_dimensions=tuple(input_dimensions),
        )

    golden_src_B = src_B_tilized_flat
    if broadcast_type != BroadcastType.None_:
        broadcast_golden = get_golden_generator(BroadcastGolden)
        golden_src_B = broadcast_golden(
            broadcast_type,
            src_B_tilized_flat,
            formats.input_format,
            num_faces=num_faces,
            tile_cnt=tile_cnt_A,
            face_r_dim=face_r_dim,
        )

    golden_tensor = binary_golden(
        math_op,
        golden_src_A,
        golden_src_B,
        formats.output_format,
        math_fidelity,
        input_format=formats.input_format,
    )

    stimuli.set_buffers(stimuli_A, stimuli_B)

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"


@pytest.mark.parametrize("config", ALL_DEST_REUSE_CONFIGS)
def test_eltwise_binary_dest_reuse(config: EltwiseBinaryDestReuseConfig):
    reuse_dest_type = config.reuse_dest_type
    formats = config.formats
    math_fidelity = config.math_fidelity
    math_op = config.math_op
    input_dimensions = list(config.input_dimensions)
    output_dimensions = list(config.output_dimensions)
    tile_dimensions = list(config.tile_dimensions)

    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dimensions)
    num_faces = num_faces_r_dim * num_faces_c_dim

    tile_rows, tile_cols = tile_dimensions
    tile_cnt_input = (input_dimensions[0] // tile_rows) * (
        input_dimensions[1] // tile_cols
    )
    tile_cnt_output = (output_dimensions[0] // tile_rows) * (
        output_dimensions[1] // tile_cols
    )

    assert tile_cnt_input % tile_cnt_output == 0, (
        f"Input tile count ({tile_cnt_input}) must be divisible by "
        f"output tile count ({tile_cnt_output})"
    )

    # Compute block/tile counts for output (determines dest register blocking)
    effective_dest_acc = (
        DestAccumulation.Yes
        if formats.output_format == DataFormat.Float32
        else DestAccumulation.No
    )
    output_num_blocks, output_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        effective_dest_acc,
        formats,
        output_dimensions,
        tile_dimensions,
        BlocksCalculationAlgorithm.Standard,
    )

    # Input has the same block count, but more tiles per block
    inner_dim = tile_cnt_input // tile_cnt_output
    input_tiles_in_block = inner_dim * output_tiles_in_block
    input_num_blocks = output_num_blocks

    stimuli = StimuliConfig(
        None,
        formats.input_format,
        None,
        formats.input_format,
        formats.output_format,
        tile_count_A=tile_cnt_input,
        tile_count_B=tile_cnt_input,
        tile_count_res=tile_cnt_output,
        num_faces=num_faces,
        face_r_dim=face_r_dim,
        tile_dimensions=tile_dimensions,
        use_dense_tile_dimensions=True,
    )

    configuration = TestConfig(
        "sources/eltwise_binary_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            BROADCAST_TYPE(BroadcastType.None_),
            MATH_OP(mathop=math_op),
            DEST_SYNC(),
            EN_DEST_REUSE(),
            REUSE_DEST_TYPE(reuse_dest_type=reuse_dest_type),
        ],
        runtimes=[
            UNPACK_TRANS_FACES(Transpose.No),
            UNPACK_TRANS_WITHIN_FACE(Transpose.No),
            NUM_TILES_IN_BLOCK(
                output_tiles_in_block,
                input_num_tiles_in_block=input_tiles_in_block,
                output_num_tiles_in_block=output_tiles_in_block,
            ),
            NUM_BLOCKS(
                output_num_blocks,
                input_num_blocks=input_num_blocks,
                output_num_blocks=output_num_blocks,
            ),
            NUM_FACES_R_DIM(num_faces_r_dim),
            NUM_FACES_C_DIM(num_faces_c_dim),
            TEST_FACE_DIMS(face_r_dim=face_r_dim),
        ],
        variant_stimuli=stimuli,
        dest_acc=DestAccumulation.No,
        unpack_to_dest=False,
    )

    configuration.prepare()
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        pytest.skip(TestConfig.SKIP_JUST_FOR_COMPILE_MARKER)

    src_A, _, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=tile_dimensions,
    )

    # Tilize inputs
    src_A_tilized = tilize_block(
        src_A,
        dimensions=input_dimensions,
        stimuli_format=formats.input_format,
        num_faces=num_faces,
        tile_dimensions=tile_dimensions,
        face_r_dim=face_r_dim,
    )
    src_B_tilized = tilize_block(
        src_B,
        dimensions=input_dimensions,
        stimuli_format=formats.input_format,
        num_faces=num_faces,
        tile_dimensions=tile_dimensions,
        face_r_dim=face_r_dim,
    )

    src_A_tilized_flat = src_A_tilized.flatten()
    src_B_tilized_flat = src_B_tilized.flatten()

    stimuli_A = src_A_tilized_flat
    stimuli_B = src_B_tilized_flat

    # Golden: simulate dest reuse.
    tile_elements = num_faces * face_r_dim * FACE_C_DIM
    torch_format = format_dict[formats.output_format]
    golden_tensor = torch.zeros(tile_cnt_output * tile_elements, dtype=torch_format)

    for out_t in range(tile_cnt_output):
        block_idx = out_t // output_tiles_in_block
        tile_in_block = out_t % output_tiles_in_block
        dest = torch.zeros(tile_elements, dtype=torch_format)

        for i in range(inner_dim):
            input_tile_idx = (
                block_idx * input_tiles_in_block
                + i * output_tiles_in_block
                + tile_in_block
            )
            start = input_tile_idx * tile_elements
            end = start + tile_elements

            a_tile = src_A_tilized_flat[start:end].to(torch_format)
            b_tile = src_B_tilized_flat[start:end].to(torch_format)

            if reuse_dest_type == EltwiseBinaryReuseDestType.DEST_TO_SRCA:
                srcA, srcB = dest.clone(), b_tile
            else:
                srcA, srcB = a_tile, dest.clone()

            if math_op == MathOperation.Elwadd:
                dest = srcA + srcB
            elif math_op == MathOperation.Elwsub:
                dest = srcA - srcB
            elif math_op == MathOperation.Elwmul:
                dest = dest + srcA * srcB

        out_start = out_t * tile_elements
        golden_tensor[out_start : out_start + tile_elements] = dest

    stimuli.set_buffers(stimuli_A, stimuli_B)

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    test_passed = passed_test(golden_tensor, res_tensor, formats.output_format)
    assert test_passed, "Assert against golden failed"


@pytest.mark.parametrize("config", ALL_INT8_CONFIGS)
def test_eltwise_binary_int8_format(config: EltwiseBinaryConfig):
    dest_acc = config.dest_acc
    formats = config.formats
    math_op = config.math_op
    math_fidelity = config.math_fidelity
    broadcast_type = config.broadcast_type
    transpose_srca = config.transpose_srca
    input_dimensions = list(config.input_dimensions)
    tile_dimensions = list(config.tile_dimensions)

    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dimensions)
    num_faces = num_faces_r_dim * num_faces_c_dim

    tile_rows, tile_cols = tile_dimensions
    tile_cnt_A = (input_dimensions[0] // tile_rows) * (input_dimensions[1] // tile_cols)
    tile_cnt_B = tile_cnt_A

    effective_dest_acc = (
        DestAccumulation.Yes
        if formats.output_format == DataFormat.Float32
        else dest_acc
    )
    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        effective_dest_acc,
        formats,
        input_dimensions,
        tile_dimensions,
        BlocksCalculationAlgorithm.Standard,
    )

    stimuli = StimuliConfig(
        None,
        formats.input_format,
        None,
        formats.input_format_B,
        formats.output_format,
        tile_count_A=tile_cnt_A,
        tile_count_B=tile_cnt_B,
        tile_count_res=tile_cnt_A,
        num_faces=num_faces,
        face_r_dim=face_r_dim,
        tile_dimensions=tile_dimensions,
        use_dense_tile_dimensions=True,
    )

    configuration = TestConfig(
        "sources/eltwise_binary_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            BROADCAST_TYPE(broadcast_type),
            MATH_OP(mathop=math_op),
            DEST_SYNC(),
        ],
        runtimes=[
            UNPACK_TRANS_FACES(transpose_srca),
            UNPACK_TRANS_WITHIN_FACE(transpose_srca),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            NUM_BLOCKS(num_blocks),
            NUM_FACES_R_DIM(num_faces_r_dim),
            NUM_FACES_C_DIM(num_faces_c_dim),
            TEST_FACE_DIMS(face_r_dim=face_r_dim),
        ],
        variant_stimuli=stimuli,
        dest_acc=dest_acc,
        unpack_to_dest=False,
    )

    configuration.prepare()
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        pytest.skip(TestConfig.SKIP_JUST_FOR_COMPILE_MARKER)

    src_A, _, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format_B,
        input_dimensions_B=input_dimensions,
        tile_dimensions=tile_dimensions,
    )

    # Use modulo to get even distribution in range -50 to +50 (avoids bunching at boundaries and avoid overflow so we can test exact results against golden)
    src_A = (src_A % 101) - 50
    src_B = (src_B % 101) - 50

    binary_golden = get_golden_generator(EltwiseBinaryGolden)

    src_A_tilized = tilize_block(
        src_A,
        dimensions=input_dimensions,
        stimuli_format=formats.input_format,
        num_faces=num_faces,
        tile_dimensions=tile_dimensions,
        face_r_dim=face_r_dim,
    )
    src_B_tilized = tilize_block(
        src_B,
        dimensions=input_dimensions,
        stimuli_format=formats.input_format,
        num_faces=num_faces,
        tile_dimensions=tile_dimensions,
        face_r_dim=face_r_dim,
    )

    src_A_tilized_flat = src_A_tilized.flatten()
    src_B_tilized_flat = src_B_tilized.flatten()

    stimuli_A = src_A_tilized_flat
    stimuli_B = src_B_tilized_flat

    golden_src_A = src_A_tilized_flat
    golden_src_B = src_B_tilized_flat

    golden_tensor = binary_golden(
        math_op,
        golden_src_A,
        golden_src_B,
        formats.output_format,
        math_fidelity,
        input_format=formats.input_format,
    )

    stimuli.set_buffers(stimuli_A, stimuli_B)

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    # Compare in tilized format
    test_passed = passed_test(
        golden_tensor, res_tensor, formats.output_format, print_errors=False
    )
    assert test_passed, "Assert against golden failed"
