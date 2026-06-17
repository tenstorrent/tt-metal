# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-chip behavioural repro for tt-metal#47016 (and #47049, same root cause).

A tilize -> BFP8 matmul pipeline runs on one Tensix core. The only variable is
the z-dim value that _llk_unpack_tilize_uninit_ restores into the tile
descriptor, exposed here as UNINIT_NUM_FACES:

    UNINIT_NUM_FACES == 4  -> z-dim correct for the BFP matmul -> PASS
    UNINIT_NUM_FACES == 2  -> z-dim mis-sizes the BFP exponent array -> FAIL

This is the deciding-variable A/B from the issue, reduced to a single chip and a
single compile-time knob. The pre-#45179 code always restored 4; #45179 made it
the operand's num_faces, so a num_faces != 4 operand feeding a Bfp8_b matmul
(e.g. the tt_transformers lm_head) decodes corrupt exponents.

Run:
    cd tt_metal/tt-llk/tests/python_tests
    pytest test_repro_47016.py -s
"""

import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import MatmulGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, MathFidelity, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    MATH_FIDELITY,
    UNINIT_NUM_FACES,
    generate_input_dim,
)
from helpers.tilize_untilize import tilize
from helpers.utils import passed_test


@parametrize(
    # Bfp8_b is the compressed operand whose exponent array the descriptor z-dim
    # sizes; this is what makes the wrong z-dim observable.
    formats=input_output_formats([DataFormat.Bfp8_b], same=True),
    dest_acc=[DestAccumulation.No],
    math_fidelity=[MathFidelity.HiFi2],
    # 4 = fixed/correct, 2 = reproduces the #45179 regression.
    uninit_num_faces=[4, 2],
)
def test_repro_47016(formats, dest_acc, math_fidelity, uninit_num_faces):
    torch_format = format_dict[formats.output_format]
    input_dimensions = [32, 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    generate_golden = get_golden_generator(MatmulGolden)
    golden_tensor = tilize(
        generate_golden(
            src_A,
            src_B,
            formats.output_format,
            math_fidelity,
            input_A_dimensions=input_dimensions,
            input_B_dimensions=input_dimensions,
            input_A_format=formats.input_format,
            input_B_format=formats.input_format,
        )
    ).to(torch_format)

    L1_to_L1_iterations = 2
    configuration = TestConfig(
        "sources/repro_47016_tilize_uninit_bfp.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            MATH_FIDELITY(math_fidelity),
            UNINIT_NUM_FACES(uninit_num_faces),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=dest_acc,
        L1_to_L1_iterations=L1_to_L1_iterations,
    )

    res_from_L1 = configuration.run().result
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    ok = passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        L1_to_L1_iterations=L1_to_L1_iterations,
    )

    if uninit_num_faces == 4:
        assert (
            ok
        ), "z-dim=4 should match golden; if this fails the harness/format path needs adjustment, not the LLK"
    else:
        # The regression: wrong restored z-dim corrupts the BFP matmul output.
        assert not ok, (
            "Expected the BFP matmul to be CORRUPT when the tilize-uninit restores "
            "z-dim != 4 (the #45179 regression). It matched golden instead, which "
            "would mean this build already has the z-dim=4 fix."
        )
