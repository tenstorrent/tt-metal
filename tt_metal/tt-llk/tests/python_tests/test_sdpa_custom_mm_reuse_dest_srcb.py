# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# ADVANCE TEST: covers demo-fork experimental LLK sdpa_custom_mm_reuse_dest_srcb (tt-metal#47554 / tt-blaze#1971),
# pending promotion. Include path (shadow -I) repoint on promotion. Primitive differs from tt-blaze only in FPU<->SFPU
# signalling cadence (orthogonal to this numerical golden).
#
# sdpa_custom_mm_reuse_dest_srcb documented contract
# (llk_math_sdpa_custom_mm_reuse_dest_srcb.h / llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb.h header banners):
#   - Custom K-reduction matmul that UNPACKS ONLY SrcA (in1, full [32,32] K-tiles) and REUSES SrcB FROM DEST (in0,
#     the [1,32] partial tile), moving DEST rows into SrcB via MOVD2B each K-iteration.
#   - Output height and width should be a SINGLE tile with tile shape [1, 32]: ct_dim == 1, rt_dim == 1.
#   - kt_dim: even number from 2 to 256 (inclusive). nt_dim: 1 to 16 (SrcA tiles per K-iteration).
#   - fidelity: LoFi (the demo compute API pins MATH_FIDELITY; the demo-fork primitive templates MathFidelity).
#   - The MATH primitive does t6_semaphore_wait_on_zero<STALL_MATH>(semaphore::UNPACK_MATH_DONE) at the top of every
#     K-iteration and (only in the signal_output branch) POSTs semaphore::FPU_SFPU. In the isolated compute-only
#     kernel there is no SFPU op, so the .cpp fakes the SFPU side: the UNPACK thread POSTs UNPACK_MATH_DONE once per
#     K-tile so MATH's wait clears, and we instantiate signal_output == false so nothing is posted on FPU_SFPU.
#
# This advance test exercises nt_dim == 1 (single output tile), LoFi, with the standard MatmulGolden. Like the
# custom_mm / sdpa_custom_mm analog tests the golden does not model the primitive's exact partial-tile DEST layout;
# exact numerical agreement is validated only when run on Blackhole hardware.
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

# LoFi-only, bf16-natural path. Keep the grid small for the advance test.
SDPA_REUSE_FORMATS = input_output_formats([DataFormat.Float16_b])

# Honor the header contract. ct_dim == 1 and rt_dim == 1 (single output tile [1,32]) are fixed; kt_dim even (2..256),
# in0 rows in {1, 2, 4, 8}.
KT_DIMS = [2, 4]
IN0_ROWS = [1, 2, 4, 8]


def _grid():
    combos = []
    for kt in KT_DIMS:
        for rows in IN0_ROWS:
            combos.append((kt, rows))
    return combos


@parametrize(
    formats=SDPA_REUSE_FORMATS,
    kt_rows=_grid(),
)
def test_sdpa_custom_mm_reuse_dest_srcb(
    formats,
    kt_rows,
    boot_mode=BootMode.DEFAULT,
):
    kt_dim, in0_rows = kt_rows
    # sdpa_custom_mm_reuse_dest_srcb contract: single output tile.
    ct_dim = 1
    rt_dim = 1
    output_tile_cnt = rt_dim * ct_dim

    torch_format = format_dict[formats.output_format]

    # in0 is the [1,32] partial tile reused from DEST as SrcB (host feeds full 32-row tiles; the kernel uses the top
    # in0_rows rows). in1 is [kt_dim*32, ct_dim*32] streamed into SrcA.
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

    # LoFi golden: standard matmul with bf16 (LoFi) rounding. NOTE: this does not model the reuse primitive's exact
    # partial-tile DEST layout; exact numerical agreement is validated only when run on Blackhole hardware.
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
        "sources/sdpa_custom_mm_reuse_dest_srcb_test.cpp",
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
