# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# ADVANCE TEST: covers demo-fork experimental LLK sdpa_custom_mm (tt-metal#47554 / tt-blaze#1971), pending promotion.
# Include path (shadow -I) repoint on promotion. Primitive differs from tt-blaze only in FPU<->SFPU signalling cadence
# (orthogonal to this numerical golden).
#
# sdpa_custom_mm documented contract, plus what the header actually implements:
#   in0 -> SrcB (partial tile), in1 [32,32] -> SrcA, rt_dim == 1, ct_dim 1..16, kt_dim even 2..256, LoFi only.
#
#   in0 rows are NOT swept. sdpa_custom_mm_configure_addrmod opens with
#       constexpr std::uint32_t face_r_dim = 8;
#   and derives every dest step from it (dest.incr 8 within a tile, 2*8 == 16 between tiles), so 8 is the only in0
#   shape this primitive implements. The {1, 2, 4} rows the header banner advertises belong to the plain custom_mm
#   addrmod, which takes face_r_dim as an argument; sweeping them here only produced 30 variants that could not have
#   agreed with any golden. See helpers/custom_mm_utils.IN0_ROWS_SDPA.
#
#   Because face_r_dim is 8 and the tile step is 16, each output tile is exactly ONE 16x16 DEST face carrying an 8x32
#   logical tile (rows 0-7 are logical columns 0-15, rows 8-15 are columns 16-31) -- the demo's DEST convention, the
#   same one sdpa_custom_mm_reuse_dest_srcb and sdpa_reduce_row use.
#
# in0's L1 layout is the dense-face layout the custom_mm unpacker walks (kt_dim*2 faces of 8x16); see
# helpers/custom_mm_utils.pack_in0_faces.
#
# COVERAGE GAP -- mask_chunk is not exercised. The primitive takes a runtime mask_chunk flag: when true the unpacker
# unpacks an extra SrcB tile from base_address_mask through config context 1
# (llk_unpack_AB_sdpa_custom_mm.h:43-47) and MATH seeds DEST from it with ct_dim*2 MOVB2D MOV_8_ROW_BRCSTs instead of
# ZEROACCing it (llk_math_sdpa_custom_mm.h:102-109), so the op computes mask + in0 @ in1. It is a live runtime flag in
# production -- flash_mla.hpp:806 computes mask_last_chunk and threads it to sdpa.h:284. This test pins it false, so
# the SrcB-mask / MOVB2D branch is untested and this test is numerically identical to test_custom_mm apart from the
# DEST layout. #52721 / #47554 cover promotion, not masked-path coverage; a dedicated tracking issue is still needed.
#
# Blackhole-only (@blackhole_only): the primitive headers resolve through a Blackhole-only shadow -I.

import torch
from conftest import blackhole_only
from helpers.advance_llk_includes import (  # noqa: F401  (module-scoped autouse fixture)
    advance_llk_include_paths,
)
from helpers.custom_mm_utils import (
    CT_DIMS,
    IN0_ROWS_SDPA,
    KT_DIMS,
    face_result_leading,
    matmul_lofi_golden,
    pack_in0_faces,
    sdpa_dest_tile_golden,
)
from helpers.device import BootMode
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, format_dict
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
SDPA_MM_FORMATS = input_output_formats([DataFormat.Float16_b])


def _grid():
    return [(ct, kt) for ct in CT_DIMS for kt in KT_DIMS]


@blackhole_only
@parametrize(
    formats=SDPA_MM_FORMATS,
    ct_kt=_grid(),
)
def test_sdpa_custom_mm(
    formats,
    ct_kt,
    boot_mode=BootMode.DEFAULT,
):
    ct_dim, kt_dim = ct_kt
    in0_rows = IN0_ROWS_SDPA
    rt_dim = 1
    output_tile_cnt = rt_dim * ct_dim

    torch_format = format_dict[formats.output_format]

    in0_dimensions = [in0_rows, kt_dim * 32]
    in1_dimensions = [kt_dim * 32, ct_dim * 32]

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

    # Row-major LoFi matmul, then repacked into the one-16x16-face-per-output-tile order the packer writes back.
    matmul_rowmajor = matmul_lofi_golden(
        in0, in1, formats, in0_dimensions, in1_dimensions
    )
    golden_tensor = sdpa_dest_tile_golden(matmul_rowmajor, torch_format)

    in0_faces = pack_in0_faces(in0, kt_dim, formats.input_format)
    tilized_B = tilize_block(
        src_B, dimensions=in1_dimensions, stimuli_format=formats.input_format
    )

    configuration = TestConfig(
        "sources/sdpa_custom_mm_test.cpp",
        formats,
        runtimes=[
            # num_faces_A is in0's active face count and num_faces_B is in1's full 4; the kernel CROSSES them into the
            # unpB / unpA slots. num_faces is the pack count, 1 here (one 16x16 DEST face per output tile).
            NUM_FACES(num_faces=1, num_faces_A=2, num_faces_B=4),
            TILE_COUNT(output_tile_cnt),
            CRK_TILE_DIMM(ct_dim, rt_dim, kt_dim),
            IN_FACE_DIMS(in0_face_r_dim=in0_rows),
        ],
        variant_stimuli=StimuliConfig(
            in0_faces,
            formats.input_format,
            tilized_B.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=kt_dim,
            tile_count_B=tile_cnt_B,
            tile_count_res=output_tile_cnt,
        ),
        dest_acc=DestAccumulation.No,
        boot_mode=boot_mode,
    )

    res_from_L1 = configuration.run().result
    res_tensor = face_result_leading(
        torch.tensor(res_from_L1, dtype=torch_format), output_tile_cnt
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
