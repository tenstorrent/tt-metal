# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import MatmulGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, MathFidelity, format_dict
from helpers.param_config import parametrize
from helpers.sfpu_dispatch_constants import EXP_WITH_BASE_SCALE
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    CLAMP_NEGATIVE,
    CRK_TILE_DIMM,
    DEST_SYNC,
    ITERATIONS,
    MATH_FIDELITY,
    SFPU_SCALE_EN,
    THROTTLE_LEVEL,
    TILE_COUNT,
)
from helpers.tile_constants import (
    DEFAULT_TILE_C_DIM,
    DEFAULT_TILE_R_DIM,
    MAX_TILE_ELEMENTS,
)
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test

# A: TILE_CNT stacked tiles, B: one tile. Output tile t = exp(A[t] x B).
TILE_CNT = 4
INPUT_A_DIMENSIONS = [DEFAULT_TILE_R_DIM * TILE_CNT, DEFAULT_TILE_C_DIM]
INPUT_B_DIMENSIONS = [DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM]


# The math thread keeps two replay-buffer users live across the tile loop: the matmul
# MOP body, recorded once at init in the FPU half, and the accurate exp body, recorded
# in the SFPU half on every call. Each output tile is a matmul followed by exp on the
# same Dest tile, with both inits done once before the loop -- the compute-kernel shape
# mm_init + exp_tile_init, then matmul_tiles / exp_tile per tile. Tile 0's matmul runs
# before any exp has recorded; every later tile's matmul replays whatever the exp
# record left in the FPU half, so a record that crosses the split shows up as tile 0
# right and tiles 1.. wrong.
#
# scale_en runs the scaled exp, exp(EXP_WITH_BASE_SCALE * x) -- the longest loop body
# of the accurate bf16 kernel on both architectures.
@parametrize(
    math_fidelity=[MathFidelity.LoFi, MathFidelity.HiFi4],
    scale_en=[False, True],
)
def test_matmul_exp_interleave(math_fidelity, scale_en):
    torch.manual_seed(0)
    # The accurate exp's bf16-Dest kernel is the one under test.
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    dest_acc = DestAccumulation.No
    torch_format = format_dict[formats.output_format]

    # |A x B| stays around 1 so exp is well inside bf16 range and a wrong product is
    # not hidden by saturation.
    spec = StimuliSpec.uniform(low=-0.3, high=0.3)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=INPUT_A_DIMENSIONS,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=INPUT_B_DIMENSIONS,
        spec_A=spec,
        spec_B=spec,
    )
    assert tile_cnt_A == TILE_CNT and tile_cnt_B == 1

    generate_golden = get_golden_generator(MatmulGolden)
    matmul_golden = generate_golden(
        src_A,
        src_B,
        formats.output_format,
        math_fidelity,
        input_A_dimensions=INPUT_A_DIMENSIONS,
        input_B_dimensions=INPUT_B_DIMENSIONS,
        tilize=True,
        input_A_format=formats.input_format,
        input_B_format=formats.input_format,
    )
    exp_scale = EXP_WITH_BASE_SCALE if scale_en else 1.0
    golden_tensor = torch.exp(exp_scale * matmul_golden.float()).to(torch_format)

    tilized_A = tilize_block(
        src_A, dimensions=INPUT_A_DIMENSIONS, stimuli_format=formats.input_format
    )
    tilized_B = tilize_block(
        src_B, dimensions=INPUT_B_DIMENSIONS, stimuli_format=formats.input_format
    )

    configuration = TestConfig(
        "sources/matmul_exp_interleave_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            DEST_SYNC(),
            THROTTLE_LEVEL(),
            APPROX_MODE(),
            CLAMP_NEGATIVE(),
            ITERATIONS(),
            SFPU_SCALE_EN(scale_en),
        ],
        runtimes=[TILE_COUNT(TILE_CNT), CRK_TILE_DIMM(1, 1, 1)],
        variant_stimuli=StimuliConfig(
            tilized_A.flatten(),
            formats.input_format,
            tilized_B.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=TILE_CNT,
        ),
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result
    assert len(res_from_L1) == len(golden_tensor)
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    # Per-tile verdict: it names the tiles that went wrong, which is the signature of a
    # replay-buffer crossing (tile 0 right, the rest wrong). The failing tiles' mismatches
    # are dumped before the assert.
    def tile_slice(tensor, t):
        return tensor[t * MAX_TILE_ELEMENTS : (t + 1) * MAX_TILE_ELEMENTS]

    bad_tiles = [
        t
        for t in range(TILE_CNT)
        if not passed_test(
            tile_slice(golden_tensor, t),
            tile_slice(res_tensor, t),
            formats.output_format,
            print_errors=False,
        )
    ]
    for t in bad_tiles:
        print(f"tile {t}:")
        passed_test(
            tile_slice(golden_tensor, t),
            tile_slice(res_tensor, t),
            formats.output_format,
            print_errors=True,
        )
    assert not bad_tiles, (
        f"exp(matmul) wrong on tiles {bad_tiles} of {TILE_CNT}: the matmul program replayed "
        "after the first exp no longer matches the one recorded at init"
    )
