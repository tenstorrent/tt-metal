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
#   ct_dim: any integer from 1 to 16     (header ENFORCES 1..16 via the mop divide; no {1,2,3,4,5,6,8,10,12,14,16}
#                                          restriction is present in this revision of the header -- see report)
#   kt_dim: even number from 2 to 256 (inclusive)
#   fidelity: LoFi only (math init takes no MathFidelity template)
#   throttle: not supported
#
# Blackhole-only. Deliverable here is compile-green (compile-producer). On-device numerical verification is
# pending Blackhole hardware/CI; this host is Wormhole.

import torch
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
    IN0_FACE_R_DIM,
    NUM_FACES,
    TILE_COUNT,
)
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test

# LoFi-only, bf16-natural path. Bfp8_b input also allowed by the LLK; keep the grid small for the advance test.
CUSTOM_MM_FORMATS = input_output_formats([DataFormat.Float16_b])

# Honor the header contract. ct_dim in the documented allowed set, kt_dim even (2..256), in0 rows in {1, 2, 4, 8}.
CT_DIMS = [1, 2, 4, 8, 16]
KT_DIMS = [2, 4]
IN0_ROWS = [1, 2, 4, 8]


def _grid():
    combos = []
    for ct in CT_DIMS:
        for kt in KT_DIMS:
            for rows in IN0_ROWS:
                combos.append((ct, kt, rows))
    return combos


@parametrize(
    formats=CUSTOM_MM_FORMATS,
    ct_kt_rows=_grid(),
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

    # in0 is [rt_dim*32, kt_dim*32] (partial rows are modeled inside the LLK; host feeds full 32-row tiles and the
    # kernel only unpacks the top in0_rows rows of each in0 face). in1 is [kt_dim*32, ct_dim*32].
    input_A_dimensions = [rt_dim * 32, kt_dim * 32]
    input_B_dimensions = [kt_dim * 32, ct_dim * 32]

    spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_A_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_B_dimensions,
        spec_A=spec,
        spec_B=spec,
    )

    # LoFi golden: standard matmul with bf16 (LoFi) rounding. NOTE: this does not model custom_mm's exact packed
    # output tile layout (split_acc/dense_packing are off here, so the plain layout is the closest match). Exact
    # numerical agreement is validated only when run on Blackhole hardware.
    generate_golden = get_golden_generator(MatmulGolden)
    golden_tensor = generate_golden(
        src_A,
        src_B,
        formats.output_format,
        MathFidelity.LoFi,
        input_A_dimensions=input_A_dimensions,
        input_B_dimensions=input_B_dimensions,
        tilize=True,
        input_A_format=formats.input_format,
        input_B_format=formats.input_format,
    )

    tilized_A = tilize_block(
        src_A, dimensions=input_A_dimensions, stimuli_format=formats.input_format
    )
    tilized_B = tilize_block(
        src_B, dimensions=input_B_dimensions, stimuli_format=formats.input_format
    )

    configuration = TestConfig(
        "sources/custom_mm_test.cpp",
        formats,
        runtimes=[
            NUM_FACES(),
            TILE_COUNT(output_tile_cnt),
            CRK_TILE_DIMM(ct_dim, rt_dim, kt_dim),
            IN0_FACE_R_DIM(in0_rows),
        ],
        variant_stimuli=StimuliConfig(
            tilized_A.flatten(),
            formats.input_format,
            tilized_B.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=output_tile_cnt,
        ),
        dest_acc=DestAccumulation.No,
        boot_mode=boot_mode,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
