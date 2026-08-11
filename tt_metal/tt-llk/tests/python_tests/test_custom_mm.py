# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# ADVANCE TEST: covers demo-tree experimental LLK custom_mm (tt-metal#47554 / tt-blaze#1971), pending promotion into
# tt_llk_blackhole/llk_lib/experimental/. Include path below must be repointed to the canonical header on promotion.
# Primitive verified byte-identical to tt-blaze main as of this writing.
#
# custom_mm documented contract (llk_math_custom_mm.h / llk_unpack_AB_custom_mm.h header banners):
#   in0 tile shape: [{1, 2, 4, 8}, 32]   (partial-row tile -> SrcB, reused across output width)
#   in1 tile shape: [32, 32]             (full tile -> SrcA)
#   rt_dim: 1
#   ct_dim: any integer from 1 to 16
#   kt_dim: even number from 2 to 256 (inclusive)
#   fidelity: LoFi only (math init takes no MathFidelity template)
#   throttle: not supported
#
# in0 is NOT a run of padded 32x32 tiles. `_llk_unpack_AB_custom_mm_init_` sets
# unpB_x_end = unpB_face_r_dim * FACE_C_DIM - 1 and issues two UNPACRs per k-tile, with the SrcB L1 base programmed
# once and advanced by counters whose stride is a single face (datum_size * FACE_C_DIM * face_r_dim). So an in0 k-tile
# is 64 * in0_rows bytes and the buffer is kt_dim*2 DENSELY packed in0_rows x 16 faces -- see
# helpers/custom_mm_utils.pack_in0_faces, which matches what the silicon-validated compressed_utils.run_compressed
# emits. TILE_SIZE_UNPACK_A is dead on this path because tile_index_b is 0.
#
# The output is correspondingly a partial tile: [in0_rows, ct_dim*32], packed with dense_packing so the DEST
# tile-to-tile stride is 32 rows, and read back through helpers/custom_mm_utils.dense_result_rowmajor.
#
# Blackhole-only (@blackhole_only): the primitive headers resolve through a Blackhole-only shadow -I.

import torch
from conftest import blackhole_only
from helpers.advance_llk_includes import (  # noqa: F401  (module-scoped autouse fixture)
    advance_llk_include_paths,
)
from helpers.custom_mm_utils import (
    dense_result_rowmajor,
    matmul_acc_atol,
    matmul_grid,
    matmul_lofi_golden,
    pack_in0_faces,
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
from helpers.utils import passed_test

# LoFi-only, bf16-natural path. Bfp8_b input is also allowed by the LLK; keep the grid small for the advance test.
CUSTOM_MM_FORMATS = input_output_formats([DataFormat.Float16_b])


@blackhole_only
@parametrize(
    formats=CUSTOM_MM_FORMATS,
    ct_kt_rows=matmul_grid(),
)
def test_custom_mm(
    formats,
    ct_kt_rows,
    boot_mode=BootMode.DEFAULT,
):
    ct_dim, kt_dim, in0_rows = ct_kt_rows
    rt_dim = 1  # custom_mm contract: rt_dim is always 1
    output_tile_cnt = rt_dim * ct_dim

    torch_format = format_dict[formats.output_format]

    # in0 is [in0_rows, kt_dim*32] -- the partial tile the kernel actually consumes, not a padded 32-row tile.
    # in1 is [kt_dim*32, ct_dim*32].
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

    golden_tensor = matmul_lofi_golden(
        in0, in1, formats, in0_dimensions, in1_dimensions
    )

    in0_faces = pack_in0_faces(in0, kt_dim, torch_format)
    tilized_B = tilize_block(
        src_B, dimensions=in1_dimensions, stimuli_format=formats.input_format
    )

    configuration = TestConfig(
        "sources/custom_mm_test.cpp",
        formats,
        runtimes=[
            # num_faces_A is in0's active face count (its top two faces) and num_faces_B is in1's full 4; the kernel
            # CROSSES them into the unpB / unpA slots. num_faces is the pack count.
            NUM_FACES(num_faces=2, num_faces_A=2, num_faces_B=4),
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
