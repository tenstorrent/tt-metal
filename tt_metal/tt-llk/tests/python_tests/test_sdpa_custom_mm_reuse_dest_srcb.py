# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# ADVANCE TEST: covers demo-fork experimental LLK sdpa_custom_mm_reuse_dest_srcb (tt-metal#47554 / tt-blaze#1971),
# pending promotion. Include path (shadow -I) repoint on promotion. Primitive differs from tt-blaze only in FPU<->SFPU
# signalling cadence (orthogonal to this numerical golden).
#
# sdpa_custom_mm_reuse_dest_srcb documented contract, plus the DEST layout measured on p100a:
#   - A K-reduction matmul that UNPACKS ONLY SrcA (in1, full [32,32] K-tiles) and REUSES SrcB FROM DEST (in0), moving
#     DEST rows into SrcB via MOVD2B once per K-iteration.
#   - Every tile it touches -- in0 K-tiles and output tiles alike -- is 16 DEST rows: an 8x32 logical tile packed into
#     ONE 16x16 face, DEST rows 0-7 holding logical columns 0-15 and rows 8-15 holding columns 16-31. in0 K-tile i is
#     read from src_index + i*16 and output tile j written at dst_index + j*16, both RAW DEST ROW offsets (the
#     primitive TT_SETC16s DEST_TARGET_REG_CFG_MATH_Offset directly rather than using a tile index).
#   - So it computes  out[0:8, :] = in0[0:8, :] @ in1,  in0 being [8, kt_dim*32] and in1 [kt_dim*32, nt_dim*32].
#   - in0 rows are NOT swept: the addrmod helper steps DEST by 8, so 8 is the only shape it implements.
#   - kt_dim: even, 2..256 by the header; capped at 4 here because the kernel seeds all K-tiles from the 4 faces of a
#     single A2D datacopy tile, and because the UNPACK-side semaphore fake posts kt_dim times (SEMPOST saturates at 15).
#   - nt_dim: 1..16 (SrcA tiles per K-iteration). Pinned to 1 here.
#   - The MATH primitive does t6_semaphore_wait_on_zero<STALL_MATH>(UNPACK_MATH_DONE) at the top of EVERY K-iteration,
#     before it retires any MVMUL. In an isolated compute-only kernel there is no SFPU op, so the .cpp fakes the SFPU
#     side from the UNPACK thread -- and those posts must be issued BEFORE the unpack execute, which otherwise spins in
#     wait_for_next_context(1) on SrcA banks that only MATH frees. That, the missing in0 unpack, and the SDPA-init /
#     datacopy-init ordering were the three causes of this file's 8 hanging variants; see the .cpp banner.
#
# Blackhole-only (@blackhole_only): the primitive headers resolve through a Blackhole-only shadow -I.

import torch
from conftest import blackhole_only
from helpers.advance_llk_includes import (  # noqa: F401  (module-scoped autouse fixture)
    advance_llk_include_paths,
)
from helpers.custom_mm_utils import (
    IN0_ROWS_SDPA,
    KT_DIMS,
    face_result_leading,
    pack_sdpa_dest_tile,
    sdpa_dest_tile_golden,
)
from helpers.device import BootMode
from helpers.format_config import DataFormat
from helpers.golden_generators import MatmulGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, MathFidelity, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    IN_FACE_DIMS,
    NUM_FACES,
    TILE_COUNT,
)
from helpers.tilize_untilize import tilize_block
from helpers.utils import matmul_acc_atol, passed_test

# LoFi-only, bf16-natural path.
SDPA_REUSE_FORMATS = input_output_formats([DataFormat.Float16_b])

NT_DIM = 1


@blackhole_only
@parametrize(
    formats=SDPA_REUSE_FORMATS,
    kt_dim=KT_DIMS,
)
def test_sdpa_custom_mm_reuse_dest_srcb(
    formats,
    kt_dim,
    boot_mode=BootMode.DEFAULT,
):
    in0_rows = IN0_ROWS_SDPA
    torch_format = format_dict[formats.output_format]

    in0_dimensions = [in0_rows, kt_dim * 32]
    in1_dimensions = [kt_dim * 32, NT_DIM * 32]

    spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, _, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=in0_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=in1_dimensions,
        spec_A=spec,
        spec_B=spec,
    )

    in0 = src_A.reshape(in0_dimensions).to(torch_format)
    in1 = src_B.reshape(in1_dimensions).to(torch_format)

    # in0 goes to L1 as ONE standard 4-face 32x32 tile whose face i is K-tile i in the op's 16x16-face packing.
    # MATH's single A2D datacopy then lands face i at DEST rows 16*i -- exactly where the execute reads K-tile i.
    in0_tile = pack_sdpa_dest_tile(in0, kt_dim, torch_format)

    # in1 is the ordinary tilized [kt_dim*32, 32] block: the unpack MOP walks kt_dim contiguous 32x32 tiles
    # (in1_k_stride == 1, nt_dim == 1).
    tilized_B = tilize_block(
        src_B, dimensions=in1_dimensions, stimuli_format=formats.input_format
    )

    # Row-major LoFi matmul, then repacked into the one-16x16-face-per-output-tile order the packer writes back.
    # LoFi, not a raw torch matmul: the FPU truncates the SrcA/SrcB mantissas before multiplying, which
    # biases a K-deep sum of positive values low by ~2% -- far outside atol if the golden multiplies at full
    # bf16 precision.
    matmul_rowmajor = get_golden_generator(MatmulGolden)(
        in0,
        in1,
        formats.output_format,
        MathFidelity.LoFi,
        input_A_dimensions=in0_dimensions,
        input_B_dimensions=in1_dimensions,
        tilize=False,
        input_A_format=formats.input_format,
        input_B_format=formats.input_format,
    ).reshape(in0_dimensions[0], in1_dimensions[1])
    golden_tensor = sdpa_dest_tile_golden(matmul_rowmajor, torch_format)

    configuration = TestConfig(
        "sources/sdpa_custom_mm_reuse_dest_srcb_test.cpp",
        formats,
        runtimes=[
            # num_faces_B is the in1 full-tile count (4) and is CROSSED into the unpA slot by the kernel;
            # num_faces is the pack count, 1 here because each output tile is a single 16x16 DEST face.
            NUM_FACES(num_faces=1, num_faces_A=1, num_faces_B=4),
            TILE_COUNT(NT_DIM),
            CRK_TILE_DIMM(NT_DIM, 1, kt_dim),
            IN_FACE_DIMS(in0_face_r_dim=in0_rows),
        ],
        variant_stimuli=StimuliConfig(
            in0_tile,
            formats.input_format,
            tilized_B.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=tile_cnt_B,
            tile_count_res=NT_DIM,
        ),
        dest_acc=DestAccumulation.No,
        boot_mode=boot_mode,
    )

    res_from_L1 = configuration.run().result
    res_tensor = face_result_leading(
        torch.tensor(res_from_L1, dtype=torch_format), NT_DIM
    )

    assert len(res_tensor) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        custom_atol=matmul_acc_atol(golden_tensor, kt_dim),
    ), "Assert against golden failed"
