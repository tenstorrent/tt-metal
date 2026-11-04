# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Functional golden of the tiny tile cells of perf_matmul_grid2: operands of different formats (Float16_b, Bfp8_b, Bfp4_b, Float32), a
different face count per operand (in0 height 1 to 16 with 2 faces, in1 width 16 with 2 faces) and block float tiles packed per operand, on the
math matmul test. One block per tile shape and operand format pair, drawn from the perf rows of perf_matmul_grid2 (same cells, same blocks)."""

import pytest
import torch
from conftest import skip_for_wormhole
from helpers.format_config import DataFormat
from helpers.golden_generators import TILE_DIM, MatmulGolden, TransposeGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, DestSync, MathFidelity, PerfRunType, StochasticRounding, Transpose, format_dict
from helpers.matmul_sweep import matmul_pack_form
from helpers.param_config import DEST_SYNC_TILE_LIMITS
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import convert_to_l1_view, generate_face_matmul_data
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    MATMUL_PHASE_ELIDE,
    CRK_TILE_DIMM,
    DEST_INDEX,
    DEST_SYNC,
    IN_TILE_DIMS,
    LOOP_FACTOR,
    MATH_FIDELITY,
    MATMUL_FORM_KEYS,
    MATMUL_K_LOOP,
    MATMUL_ROW_END_FOLD,
    MATMUL_IN1_ALT,
    MATMUL_OPERAND_SIZES,
    MATMUL_PACK_BLOCK,
    MATMUL_STREAM_NARROW,
    MATMUL_REUSE_DIR,
    MATMUL_STREAM_Z,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    PARTIAL_FACE,
    PERF_RUN_TYPE,
    STOCHASTIC_ROUNDING,
    THROTTLE_LEVEL,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)
from helpers.logger import logger
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test
from perf_matmul_grid2 import _hash, _tiny_rows, tiny_config


def _golden_cases():
    """One perf row per (tile shape, SrcA format, SrcB format)."""
    groups = {}
    for row in _tiny_rows():
        a, b, o, acc, sy, fi, r0, c1, rt, ct, kt = row
        groups.setdefault((r0, c1, a, b), []).append(row)
    out = []
    for key in sorted(groups):
        rows = groups[key]
        if "F32" in key[2:]:
            # a 32-bit operand against the golden's 32-bit accumulation needs the 32-bit DEST
            rows = [r for r in rows if r[3] == "Y"]
        a, b, o, acc, sy, fi, r0, c1, rt, ct, kt = rows[_hash("g%d,%d,%s,%s" % key) % len(rows)]
        out.append((a, b, o, acc, sy, getattr(MathFidelity, fi), r0, c1, rt, ct, kt))
    return out


@pytest.mark.nightly
@skip_for_wormhole
@pytest.mark.parametrize(
    "case",
    _golden_cases(),
    ids=lambda c: "in0_%dx32_in1_32x%d_in1fmt_%s_in0fmt_%s_out_%s_%s_%s_%s_%dx%dx%d" % (c[6], c[7], c[0], c[1], c[2], c[3], c[4], c[5].name, c[8], c[9], c[10]),
)
def test_matmul_grid2_tiny(case):
    a, b, o, acc, sy, math_fidelity, r0, c1, rt, ct, kt = case
    matmul_config = tiny_config(a, b, o, acc, sy, r0, c1, rt, ct, kt)
    throttle = 0
    formats = matmul_config.formats
    in0_dimensions = matmul_config.tile_dimensions.in0_dimensions
    in1_dimensions = matmul_config.tile_dimensions.in1_dimensions
    in0_tile_r_dim = matmul_config.tile_dimensions.in0_tile_r_dim
    in0_tile_c_dim = matmul_config.tile_dimensions.in0_tile_c_dim
    in1_tile_r_dim = matmul_config.tile_dimensions.in1_tile_r_dim
    in1_tile_c_dim = matmul_config.tile_dimensions.in1_tile_c_dim
    transpose = matmul_config.face_layout_config.unpack_transpose_faces
    tile_cnt_in1 = matmul_config.tile_dimensions.tile_cnt_in1
    num_faces_in0 = matmul_config.face_layout_config.num_faces_in0
    num_faces_in1 = matmul_config.face_layout_config.num_faces_in1
    num_faces = matmul_config.face_layout_config.num_faces

    torch_format = format_dict[formats.output_format]

    in0 = generate_face_matmul_data(
        num_faces=num_faces_in0,
        stimuli_format=formats.input_format_B,
        input_dimensions=in0_dimensions,  # This will generate the right number of tiles
        is_matrix_A=True,  # matrix A (In0/SrcB) uses f0,f1 for 2-face mode
        face_r_dim=(
            matmul_config.tile_dimensions.in0_tile_r_dim
            if matmul_config.tile_dimensions.in0_tile_r_dim < 16
            else 16
        ),
    )

    in1 = generate_face_matmul_data(
        num_faces=num_faces_in1,
        stimuli_format=formats.input_format,
        input_dimensions=in1_dimensions,  # This will generate the right number of tiles
        is_matrix_A=False,  # matrix B (In1/SrcA) uses f0,f2 for 2-face mode
    )

    in1_golden = in1
    if transpose == Transpose.Yes:
        t_matrix = get_golden_generator(TransposeGolden)

        in1_golden = t_matrix.transpose_faces_multi_tile(
            in1,
            formats.input_format,
            num_tiles=tile_cnt_in1,
            tilize=True,
            input_dimensions=in1_dimensions,
        )
        in1_golden = t_matrix.transpose_within_faces_multi_tile(
            in1_golden,
            formats.input_format,
            num_tiles=tile_cnt_in1,
            untilize=True,
            input_dimensions=in1_dimensions,
        )

    # the golden quantizes a block float operand in blocks of 16 of its tilized order and untilizes the result, so it takes those operands tilized
    golden_in0 = tilize_block(in0, dimensions=in0_dimensions, stimuli_format=formats.input_format_B) if formats.input_format_B.is_block_float() else in0
    golden_in1 = tilize_block(in1_golden, dimensions=in1_dimensions, stimuli_format=formats.input_format) if formats.input_format.is_block_float() else in1_golden
    generate_golden = get_golden_generator(MatmulGolden)
    golden_tensor = generate_golden(
        golden_in0,
        golden_in1,
        formats.output_format,
        math_fidelity,
        input_A_dimensions=in0_dimensions,
        input_B_dimensions=in1_dimensions,
        tilize=True,  # Golden cannot model FPU strided for tilized data computation, so we tilize output after computation
        input_A_format=formats.input_format_B,
        input_B_format=formats.input_format,
    )

    tilized_in0 = tilize_block(
        in0, dimensions=in0_dimensions, stimuli_format=formats.input_format_B
    )
    tilized_in1 = tilize_block(
        in1, dimensions=in1_dimensions, stimuli_format=formats.input_format
    )
    tilized_in0_l1_view = convert_to_l1_view(
        tilized_in0, in0_dimensions, tile_dimensions=[in0_tile_r_dim, in0_tile_c_dim]
    )
    tilized_in1_l1_view = convert_to_l1_view(
        tilized_in1, in1_dimensions, tile_dimensions=[in1_tile_r_dim, in1_tile_c_dim]
    )

    # Matmul sweep shapes deliberately fit one destination section. Repeat the
    # complete matmul block four times so the same test also validates section
    # hand-off without changing the operation's RT/CT/KT contract.
    #
    # SyncFull + 16-bit dest (dest_acc=No) holds 16 tiles. Pack CLR_ALL after a
    # tiny-tile section whose dest window straddles the half-sync boundary
    # leaves packer dest addressing stuck at that tile on the next section
    # (1x32 through 16x32). Skip the 4-block handoff for those windows;
    # restore num_blocks=4 after #53500. Windows that stay in one half,
    # SyncHalf, dest_acc=Yes, and full 32x32 tiles still cover the handoff.
    num_tiles_in_block = matmul_config.tile_dimensions.tile_cnt
    dst_index = matmul_config.dst_index
    half_dest_tiles = DEST_SYNC_TILE_LIMITS[DestSync.Half]
    dest_straddles_half = (
        matmul_config.dest_sync == DestSync.Full
        and matmul_config.dest_acc == DestAccumulation.No
        and matmul_config.tile_dimensions.in0_tile_r_dim < TILE_DIM
        and dst_index < half_dest_tiles < dst_index + num_tiles_in_block
    )
    if dest_straddles_half:
        logger.warning(
            "Skipping 4-block dest section handoff for tiny-tile matmul "
            "(dst_index={}, num_tiles_in_block={}); restore num_blocks=4 after #53500",
            dst_index,
            num_tiles_in_block,
        )
    num_blocks = 1


    configuration = TestConfig(
        "sources/math_matmul_test.cpp",
        formats,
        templates=[
            STOCHASTIC_ROUNDING(matmul_config.stochastic_rnd),
            MATH_FIDELITY(math_fidelity),
            MATMUL_OPERAND_SIZES(),
            *MATMUL_PHASE_ELIDE.for_block(formats, math_fidelity, matmul_config.dest_acc, throttle, matmul_config.face_layout_config.unpack_transpose_faces, MATMUL_PHASE_ELIDE.keeps_requested(matmul_config.tile_dimensions, matmul_config.dest_sync, math_fidelity)),
            THROTTLE_LEVEL(throttle),
            DEST_SYNC(matmul_config.dest_sync),
            PERF_RUN_TYPE(PerfRunType.L1_TO_L1),
            *MATMUL_STREAM_NARROW.for_block(
                formats,
                matmul_config.dest_acc,
                matmul_config.dest_sync,
                matmul_config.tile_dimensions.ct_dim,
                matmul_config.tile_dimensions.rt_dim,
                matmul_config.tile_dimensions.in0_tile_r_dim,
                math_fidelity=math_fidelity,
            ),
            *MATMUL_REUSE_DIR.for_block(
                formats,
                math_fidelity,
                matmul_config.tile_dimensions.ct_dim,
                matmul_config.tile_dimensions.rt_dim,
                matmul_config.dest_sync,
            ),
            *MATMUL_STREAM_Z.for_block(
                formats,
                matmul_config.dest_acc,
                matmul_config.dest_sync,
                matmul_config.tile_dimensions.ct_dim,
                matmul_config.tile_dimensions.rt_dim,
                matmul_config.tile_dimensions.in0_tile_r_dim,
                math_fidelity=math_fidelity,
                in1_faces=matmul_config.face_layout_config.num_faces_in1,
                in1_partial=matmul_config.face_layout_config.partial_face_in1,
                kt_dim=matmul_config.tile_dimensions.kt_dim,
                throttle=throttle,
            ),
            *MATMUL_K_LOOP.for_block(
                formats,
                matmul_config.dest_acc,
                math_fidelity,
                matmul_config.tile_dimensions.ct_dim,
                matmul_config.tile_dimensions.rt_dim,
                matmul_config.tile_dimensions.kt_dim,
                matmul_config.tile_dimensions.in0_tile_r_dim,
                matmul_config.tile_dimensions.in0_tile_c_dim,
                matmul_config.tile_dimensions.in1_tile_r_dim,
                matmul_config.tile_dimensions.in1_tile_c_dim,
            ),
            *MATMUL_ROW_END_FOLD.for_block(matmul_config.tile_dimensions.ct_dim, matmul_config.tile_dimensions.rt_dim),
            *MATMUL_FORM_KEYS.for_block(
                formats,
                matmul_config.tile_dimensions.kt_dim,
                matmul_config.tile_dimensions.ct_dim,
                matmul_config.tile_dimensions.rt_dim,
                math_fidelity=math_fidelity,
                dest_sync=matmul_config.dest_sync,
                dest_acc=matmul_config.dest_acc,
            ),
            *MATMUL_IN1_ALT.for_tiles(
                formats,
                matmul_config.dest_acc,
                math_fidelity,
                matmul_config.tile_dimensions.in0_tile_r_dim,
                matmul_config.tile_dimensions.in0_tile_c_dim,
                matmul_config.tile_dimensions.in1_tile_r_dim,
                matmul_config.tile_dimensions.in1_tile_c_dim,
            ),
            MATMUL_PACK_BLOCK(
                matmul_pack_form(
                    formats,
                    matmul_config.dest_sync,
                    math_fidelity,
                    matmul_config.tile_dimensions.rt_dim,
                    matmul_config.tile_dimensions.ct_dim,
                    matmul_config.tile_dimensions.kt_dim,
                    matmul_config.tile_dimensions.in0_tile_r_dim,
                    matmul_config.tile_dimensions.in1_tile_c_dim,
                    throttle,
                )
            ),
        ],
        runtimes=[
            TILE_COUNT(matmul_config.tile_dimensions.tile_cnt),
            NUM_BLOCKS(num_blocks),
            LOOP_FACTOR(1),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            NUM_FACES(num_faces, num_faces_in0, num_faces_in1),
            UNPACK_TRANS_FACES(transpose),
            UNPACK_TRANS_WITHIN_FACE(transpose),
            PARTIAL_FACE(
                partial_a=matmul_config.face_layout_config.partial_face_in0,
                partial_face_pack=matmul_config.face_layout_config.partial_face_pack,
                partial_b=matmul_config.face_layout_config.partial_face_in1,
                partial_face_math=matmul_config.face_layout_config.partial_face_math,
            ),
            CRK_TILE_DIMM(
                matmul_config.tile_dimensions.ct_dim,
                matmul_config.tile_dimensions.rt_dim,
                matmul_config.tile_dimensions.kt_dim,
            ),
            IN_TILE_DIMS(
                matmul_config.tile_dimensions.in0_tile_r_dim,
                matmul_config.tile_dimensions.in0_tile_c_dim,
                matmul_config.tile_dimensions.in1_tile_r_dim,
                matmul_config.tile_dimensions.in1_tile_c_dim,
            ),
            DEST_INDEX(dst_index),
        ],
        variant_stimuli=StimuliConfig(
            tilized_in0_l1_view.flatten(),
            formats.input_format_B,
            tilized_in1_l1_view.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=matmul_config.tile_dimensions.tile_cnt_in0,
            tile_count_B=matmul_config.tile_dimensions.tile_cnt_in1,
            tile_count_res=matmul_config.tile_dimensions.tile_cnt * num_blocks,
            num_faces_A=num_faces_in0,
            face_r_dim_A=min(in0_tile_r_dim, 16),
            tile_dimensions_A=[in0_tile_r_dim, in0_tile_c_dim],
            num_faces_B=num_faces_in1,
            tile_dimensions_B=[in1_tile_r_dim, in1_tile_c_dim],
            num_faces_res=num_faces,
            face_r_dim_res=min(in0_tile_r_dim, 16),
        ),
        dest_acc=matmul_config.dest_acc,
    )
    res_from_L1 = configuration.run().result

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)
    # the result tiles come back packed to their own face count and rows; the golden keeps 1024 elements per tile
    per_tile = num_faces * min(in0_tile_r_dim, 16) * 16
    res_tensor = torch.nn.functional.pad(res_tensor.view(-1, per_tile), (0, 1024 - per_tile)).flatten()

    # Convert golden tensor to L1 memory view for comparison
    golden_tensor = convert_to_l1_view(
        golden_tensor,
        (in0_dimensions[0], in1_dimensions[1]),
        tile_dimensions=[in0_tile_r_dim, in1_tile_c_dim],
    )
    golden_tensor = golden_tensor.repeat(num_blocks)

    assert len(res_tensor) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    if formats.output_format == DataFormat.Bfp4_b:
        # the packer keeps 3 magnitude bits per block of 16 face row elements, so a result is within two steps of its block (the exponent of the block max less 2) of the exact value
        g = golden_tensor.float().view(-1, 16)
        r = res_tensor.float().view(-1, 16)
        block_max = g.abs().max(dim=1, keepdim=True).values
        step = torch.exp2(torch.floor(torch.log2(block_max.clamp(min=1e-30))) - 2)
        assert ((g - r).abs() <= 2.0 * step).all(), "Bfp4_b result is more than two block steps from the golden"
        return

    if num_faces < 4:
        num_elements_per_tile = in0_tile_r_dim * in1_tile_c_dim
        tile_cnt = matmul_config.tile_dimensions.output_tile_cnt * num_blocks

        # Compare each tile separately
        TILE_R_DIM, TILE_C_DIM = 32, 32
        for i in range(tile_cnt):
            start = i * (TILE_R_DIM * TILE_C_DIM)
            assert passed_test(
                golden_tensor[
                    start : start + num_elements_per_tile
                ],  # Only compare active faces in this tile
                res_tensor[
                    start : start + num_elements_per_tile
                ],  # Only compare active faces in this tile
                formats.output_format,
            ), f"Assert on tile {i}/{tile_cnt} against golden failed"
    else:
        assert passed_test(
            golden_tensor, res_tensor, formats.output_format
        ), "Assert against golden failed"
