# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    BroadcastGolden,
    EltwiseBinaryGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BroadcastType,
    DestAccumulation,
    DestSync,
    MathFidelity,
    MathOperation,
    format_dict,
)
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    BROADCAST_TYPE,
    INPUT_DIMENSIONS,
    MATH_FIDELITY,
    MATH_OP,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    TEST_FACE_DIMS,
    TILE_COUNT,
    TemplateParameter,
)
from helpers.tile_constants import get_tile_params
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

logger = logging.getLogger(__name__)


@dataclass
class CT_DIM(TemplateParameter):
    ct_dim: int

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t CT_DIM = {self.ct_dim};"


@dataclass
class PRESERVE_SRC_ZERO_FLAG(TemplateParameter):
    """Math sets the Src zero flag to keep before the custom init, as a preceding copy_tile_init does."""

    preserve: bool = False

    def convert_to_cpp(self) -> str:
        return f"constexpr bool PRESERVE_SRC_ZERO_FLAG = {str(self.preserve).lower()};"


# Row height of the srcA operand. 32 -> full 32x32 tiles (num_faces=4);
# 16 -> 16x32 "tiny tiles" (one face-row, num_faces=2).
_TILE_ROWS = [32, 16]


@parametrize(
    cpp_source=[
        "sources/multiple_tiles_eltwise_custom_test.cpp",
    ],
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
        ]
    ),
    mathop=[MathOperation.Elwsub, MathOperation.Elwmul],
    dest_acc=[DestAccumulation.No],
    math_fidelity=[
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
    broadcast_type=[BroadcastType.Column],
    tile_rows=_TILE_ROWS,
    # Widths up to 256 fit a single dest section; 1024 (32 tiles) spans multiple
    # sections, exercising destination bank switching in the custom pipeline.
    input_width=list(range(32, 257, 32)) + [1024],
)
def test_eltwise_bcast_col_custom(
    cpp_source,
    formats,
    mathop,
    dest_acc,
    math_fidelity,
    broadcast_type,
    tile_rows,
    input_width,
):
    if mathop != MathOperation.Elwmul and math_fidelity != MathFidelity.LoFi:
        pytest.skip("Fidelity does not affect Elwadd and Elwsub operations")
    # The MUL instantiation of the blocked bcast-col reuse scaffold exists only on Blackhole
    # (Wormhole has just the SUB-named wrapper).
    if (
        mathop == MathOperation.Elwmul
        and get_chip_architecture() != ChipArchitecture.BLACKHOLE
    ):
        pytest.skip("MUL bcast-col reuse scaffold is Blackhole-only")

    input_dimensions_A = [tile_rows, input_width]
    input_dimensions_B = [tile_rows, 32]

    # 32x32 tile => num_faces=4, 16x32 tiny tile => num_faces=2.
    tile_dims = [tile_rows, 32]
    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dims)
    num_faces = num_faces_r_dim * num_faces_c_dim

    ct_dim = input_dimensions_A[1] // 32
    # Each column tile is tile_rows x 32; the operand is a single row of ct_dim tiles.
    rt_dim = 1
    logger.info(
        "Running ct_dim=%d  srcA=%s  srcB=%s  num_faces=%d  face_r_dim=%d",
        ct_dim,
        input_dimensions_A,
        input_dimensions_B,
        num_faces,
        face_r_dim,
    )

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions_A,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions_B,
        tile_dimensions=tile_dims,
    )

    src_A_tilized = tilize_block(
        src_A,
        input_dimensions_A,
        formats.input_format,
        num_faces=num_faces,
        tile_dimensions=tile_dims,
        face_r_dim=face_r_dim,
    ).flatten()
    src_B_tilized = tilize_block(
        src_B,
        input_dimensions_B,
        formats.input_format,
        num_faces=num_faces,
        tile_dimensions=tile_dims,
        face_r_dim=face_r_dim,
    ).flatten()

    broadcast_golden = get_golden_generator(BroadcastGolden)
    src_B_broadcasted_tilized = broadcast_golden(
        broadcast_type,
        src_B_tilized,
        formats.input_format,
        num_faces=num_faces,
        tile_cnt=tile_cnt_B,
        face_r_dim=face_r_dim,
    )

    src_B_golden = untilize_block(
        src_B_broadcasted_tilized,
        formats.input_format,
        input_dimensions_B,
        num_faces=num_faces,
        tile_dimensions=tile_dims,
        face_r_dim=face_r_dim,
    ).flatten()

    src_B_2d = src_B_golden.view(input_dimensions_B[0], input_dimensions_B[1])
    src_B_expanded = src_B_2d.repeat(1, input_dimensions_A[1] // input_dimensions_B[1])
    src_B_golden_expanded = src_B_expanded.flatten()

    generate_golden = get_golden_generator(EltwiseBinaryGolden)
    golden_tensor = generate_golden(
        mathop, src_A, src_B_golden_expanded, formats.output_format, math_fidelity
    )

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half, dest_acc, formats, input_dimensions_A, tile_dimensions=tile_dims
    )

    configuration = TestConfig(
        cpp_source,
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            INPUT_DIMENSIONS(
                full_rt_dim=rt_dim,
                full_ct_dim=ct_dim,
                block_ct_dim=ct_dim,
                block_rt_dim=rt_dim,
            ),
            MATH_OP(mathop=mathop),
            BROADCAST_TYPE(broadcast_type),
            CT_DIM(ct_dim),
            PRESERVE_SRC_ZERO_FLAG(False),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces, num_faces, num_faces),
            TEST_FACE_DIMS(face_r_dim),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=StimuliConfig(
            src_A_tilized,
            formats.input_format,
            src_B_tilized,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            num_faces=num_faces,
            face_r_dim=face_r_dim,
            tile_dimensions=tile_dims,
            use_dense_tile_dimensions=True,
        ),
        dest_acc=dest_acc,
    )
    res_from_L1 = configuration.run().result

    res_from_L1 = untilize_block(
        res_from_L1,
        formats.output_format,
        input_dimensions_A,
        num_faces=num_faces,
        tile_dimensions=tile_dims,
        face_r_dim=face_r_dim,
    ).flatten()

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    # Spot-check: log a few elements per tile to verify srcA <op> bcast(srcB)
    tile_elems = tile_rows * 32
    for t in range(ct_dim):
        tile_start = t * tile_elems
        a_sample = src_A[tile_start : tile_start + 4].tolist()
        b_sample = src_B_golden_expanded[tile_start : tile_start + 4].tolist()
        g_sample = golden_tensor[tile_start : tile_start + 4].tolist()
        r_sample = res_tensor[tile_start : tile_start + 4].tolist()
        logger.info(
            "  tile %d: srcA[:4]=%s  bcast_srcB[:4]=%s  golden[:4]=%s  result[:4]=%s",
            t,
            a_sample,
            b_sample,
            g_sample,
            r_sample,
        )

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"


# Smallest positive bf16 denormal, 2^-133.
BF16_DENORMAL_MIN = 2.0**-133


def test_eltwise_bcast_col_custom_flushes_denormal_srcb_after_keep_flag():
    """The init must restore the Src zero flag a preceding datacopy left at keep.

    Even rows broadcast a bf16 denormal from SrcB, which a bf16 FPU op flushes to zero. Against srcA rows of
    2^100 a kept denormal gives a nonzero product, so those rows must come out exactly 0. Odd rows are normal
    and check the rest of the op. MUL only: a denormal is lost in a SUB result either way.
    """
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("MUL bcast-col reuse scaffold is Blackhole-only")
    mathop = MathOperation.Elwmul

    formats = input_output_formats([DataFormat.Float16_b])[0]
    tile_rows, ct_dim = 32, 2
    tile_dims = [tile_rows, 32]
    input_dimensions_A = [tile_rows, ct_dim * 32]
    input_dimensions_B = [tile_rows, 32]
    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dims)
    num_faces = num_faces_r_dim * num_faces_c_dim
    denormal_rows = torch.arange(tile_rows) % 2 == 0

    torch.manual_seed(0)
    src_A = torch.randn(input_dimensions_A)
    src_A[denormal_rows] = 2.0**100
    # One value per row, so the column broadcast reads it whichever column it takes.
    row_values = torch.randn(tile_rows)
    row_values[denormal_rows] = (
        BF16_DENORMAL_MIN * torch.randint(1, 128, (int(denormal_rows.sum()),)).float()
    )
    src_B = row_values[:, None].expand(input_dimensions_B).clone()
    flushed_B = torch.where(denormal_rows, 0.0, row_values)[:, None]
    golden = (src_A * flushed_B).to(torch.bfloat16)
    src_A = src_A.to(torch.bfloat16).flatten()
    src_B = src_B.to(torch.bfloat16).flatten()

    def _tilize(t, dims):
        return tilize_block(
            t,
            dims,
            formats.input_format,
            num_faces=num_faces,
            tile_dimensions=tile_dims,
            face_r_dim=face_r_dim,
        ).flatten()

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        DestAccumulation.No,
        formats,
        input_dimensions_A,
        tile_dimensions=tile_dims,
    )
    configuration = TestConfig(
        "sources/multiple_tiles_eltwise_custom_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(MathFidelity.LoFi),
            INPUT_DIMENSIONS(
                full_rt_dim=1,
                full_ct_dim=ct_dim,
                block_ct_dim=ct_dim,
                block_rt_dim=1,
            ),
            MATH_OP(mathop=mathop),
            BROADCAST_TYPE(BroadcastType.Column),
            CT_DIM(ct_dim),
            PRESERVE_SRC_ZERO_FLAG(True),
        ],
        runtimes=[
            TILE_COUNT(ct_dim),
            NUM_FACES(num_faces, num_faces, num_faces),
            TEST_FACE_DIMS(face_r_dim),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=StimuliConfig(
            _tilize(src_A, input_dimensions_A),
            formats.input_format,
            _tilize(src_B, input_dimensions_B),
            formats.input_format,
            formats.output_format,
            tile_count_A=ct_dim,
            tile_count_B=1,
            tile_count_res=ct_dim,
            num_faces=num_faces,
            face_r_dim=face_r_dim,
            tile_dimensions=tile_dims,
            use_dense_tile_dimensions=True,
        ),
        dest_acc=DestAccumulation.No,
    )
    res = untilize_block(
        configuration.run().result,
        formats.output_format,
        input_dimensions_A,
        num_faces=num_faces,
        tile_dimensions=tile_dims,
        face_r_dim=face_r_dim,
    ).flatten()
    res = torch.tensor(res, dtype=torch.bfloat16).reshape(input_dimensions_A)

    kept = res[denormal_rows].float()
    assert (
        torch.count_nonzero(kept) == 0
    ), f"denormal SrcB rows were not flushed: max |value| {kept.abs().max().item():.3e}"
    assert passed_test(
        golden.flatten(), res.flatten(), formats.output_format
    ), "Assert against golden failed"
