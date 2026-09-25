# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from conftest import skip_for_quasar
from helpers.format_config import DataFormat
from helpers.golden_generators import MatmulGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, MathFidelity
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    MATH_FIDELITY,
    NUM_FACES,
    THROTTLE_LEVEL,
    TILE_COUNT,
)
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test


@skip_for_quasar
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    kt_dim=[3, 20],
)
def test_matmul_in1_column_stride_and_restore(formats, dest_acc, kt_dim):
    """Strided in1 columns, then contiguous columns without unpacker reinit.

    Four output columns exercise address stepping in both destination modes.
    Stride 20 matches K640 MLA; stride 3 covers an odd, non-power-of-two layout.
    The second result uses different data so stale/repeated output cannot pass.
    """
    ct_dim = 4
    a_shape, b_shape = (32, kt_dim * 32), (kt_dim * 32, ct_dim * 32)
    # Small binary fractions keep products and running sums exactly representable
    # in LoFi / 16-bit destination mode, isolating address errors from rounding.
    generator = torch.Generator().manual_seed(42)
    a = torch.randint(-1, 2, a_shape, generator=generator).to(torch.bfloat16) / 8
    b_strided = (
        torch.randint(-1, 2, b_shape, generator=generator).to(torch.bfloat16) / 8
    )
    b_contiguous = (
        torch.randint(-1, 2, b_shape, generator=generator).to(torch.bfloat16) / 8
    )

    def tiled(tensor, shape):
        return tilize_block(
            tensor, dimensions=shape, stimuli_format=formats.input_format
        ).flatten()

    # Reorder complete tiles, preserving the face layout inside each tile.
    # Logical B[k, c] is stored at c * kt_dim + k in phase 0.
    strided_tiles = (
        tiled(b_strided, b_shape)
        .reshape(kt_dim, ct_dim, 1024)
        .permute(1, 0, 2)
        .contiguous()
        .flatten()
    )
    contiguous_tiles = tiled(b_contiguous, b_shape)
    # Keep wrong-stride reads inside allocated stimulus memory if restoration breaks.
    in1 = torch.cat(
        [strided_tiles, contiguous_tiles, torch.zeros_like(contiguous_tiles)]
    )
    configuration = TestConfig(
        "sources/matmul_in1_column_stride_test.cpp",
        formats,
        templates=[MATH_FIDELITY(MathFidelity.LoFi), THROTTLE_LEVEL(0)],
        runtimes=[
            NUM_FACES(),
            TILE_COUNT(2 * ct_dim),
            CRK_TILE_DIMM(ct_dim, 1, kt_dim),
        ],
        variant_stimuli=StimuliConfig(
            tiled(a, a_shape),
            formats.input_format,
            in1,
            formats.input_format,
            formats.output_format,
            tile_count_A=kt_dim,
            tile_count_B=3 * kt_dim * ct_dim,
            tile_count_res=2 * ct_dim,
        ),
        dest_acc=dest_acc,
    )
    result = torch.tensor(configuration.run().result, dtype=torch.bfloat16)
    assert result.numel() == 2 * ct_dim * 1024
    golden = get_golden_generator(MatmulGolden)
    for phase, b in enumerate((b_strided, b_contiguous)):
        expected = golden(
            a,
            b,
            formats.output_format,
            MathFidelity.LoFi,
            input_A_dimensions=a_shape,
            input_B_dimensions=b_shape,
            tilize=True,
            input_A_format=formats.input_format,
            input_B_format=formats.input_format,
        )
        actual = result[phase * ct_dim * 1024 : (phase + 1) * ct_dim * 1024]
        assert passed_test(expected, actual, formats.output_format), (
            "Strided matmul mismatch"
            if phase == 0
            else "Contiguous matmul after stride restore mismatch"
        )
