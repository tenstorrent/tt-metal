# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-chip behavioural repro for tt-metal#47016 (and #47049, same root cause).

This is the *genuine* "unpack-tilize followed by matmul" pattern from the issues
(the lm_head path: in-kernel tilize of activations, then a matmul against
bfloat8_b weights). It uses the real fused tilize->matmul kernel
(matmul_unpack_tilize_test.cpp), which drives all three cores with the proper
PACK_DONE handshake:

    run 0: unpack-TILIZE A and B from L1 -> datacopy -> PACK back to L1 as Bfp8_b
    run 1: _llk_unpack_tilize_uninit_(z=UNINIT_NUM_FACES) -> matmul on the
           (Bfp8_b) tilized operands

Because the matmul now consumes BFP-compressed operands, the tile-descriptor
z-dim that the uninit restores actually matters: it sizes the per-tile exponent
array (NumBlobs = BlobsPerXYPlane * ZDim * WDim).

    UNINIT_NUM_FACES == 4  -> FIXED behaviour   -> matches golden    -> PASS
    UNINIT_NUM_FACES == 2  -> #45179 regression  -> corrupt logits    -> FAIL

A correct run prints BOTH parametrizations passing: the z=4 control reproduces
golden, and the z=2 case is corrupt (which is what the z=2 case asserts).

The data-format inference model cannot express "tilize Float16_b, then matmul a
Bfp8_b operand" across two runs (it would infer math=Bfp8_b, which the FPU can't
do), so we build the two-run formats_array by hand: math stays Float16_b
everywhere; only the tilized L1 intermediate is Bfp8_b, so the matmul is a real
BFP consumer.

Run:
    cd tt_metal/tt-llk/tests/python_tests
    pytest test_repro_47016.py -s
"""

import torch
from helpers.format_config import DataFormat, FormatConfig, InputOutputFormat
from helpers.golden_generators import MatmulGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, MathFidelity, format_dict
from helpers.param_config import parametrize
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

# Activations enter in L1 as Float16_b and are tilized in-kernel; the tilized
# operands are stored as Bfp8_b so the matmul is a BFP-compressed consumer
# (exactly like the lm_head's bfloat8_b weights).
_L1_INPUT = DataFormat.Float16_b
_TILIZED = DataFormat.Bfp8_b
_OUTPUT = DataFormat.Float16_b


def _bfp_matmul_formats():
    """Two-run formats_array for the fused tilize->matmul kernel.

    run 0 (tilize): Float16_b in L1 -> math Float16_b -> pack Bfp8_b to L1.
    run 1 (matmul): unpack Bfp8_b -> math Float16_b -> pack Float16_b.

    math is Float16_b in both runs (the FPU cannot operate in Bfp8_b); only the
    tilized L1 intermediate is Bfp8_b, which is what makes the matmul a genuine
    BFP exponent-array consumer subject to the z-dim regression.
    """
    tilize_run = FormatConfig(
        unpack_A_src=_L1_INPUT,
        unpack_A_dst=_L1_INPUT,
        pack_src=_L1_INPUT,
        pack_dst=_TILIZED,
        math=_L1_INPUT,
    )
    matmul_run = FormatConfig(
        unpack_A_src=_TILIZED,
        unpack_A_dst=_TILIZED,
        pack_src=_L1_INPUT,
        pack_dst=_OUTPUT,
        math=_L1_INPUT,
    )
    return [tilize_run, matmul_run]


@parametrize(
    math_fidelity=[MathFidelity.HiFi4],
    # 4 = fixed/correct restore value, 2 = reproduces the #45179 regression.
    uninit_num_faces=[4, 2],
)
def test_repro_47016(math_fidelity, uninit_num_faces):
    torch_format = format_dict[_OUTPUT]
    input_dimensions = [32, 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=_L1_INPUT,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=_L1_INPUT,
        input_dimensions_B=input_dimensions,
    )

    # Golden quantizes the operands to Bfp8_b (matching the device's tilized
    # intermediate), then matmuls. The result is tilized to compare against L1.
    generate_golden = get_golden_generator(MatmulGolden)
    golden_tensor = tilize(
        generate_golden(
            src_A,
            src_B,
            _OUTPUT,
            math_fidelity,
            input_A_dimensions=input_dimensions,
            input_B_dimensions=input_dimensions,
            input_A_format=_TILIZED,
            input_B_format=_TILIZED,
        )
    ).to(torch_format)

    L1_to_L1_iterations = 2
    configuration = TestConfig(
        "sources/repro_47016_tilize_uninit_bfp.cpp",
        # Inference would pick math=Bfp8_b for a BFP matmul operand (invalid), so
        # we drive the kernel with a hand-built formats_array (set below). A valid
        # Float16_b/Float16_b InputOutputFormat is passed here only so tile sizes,
        # dest_acc, and the runtime-args struct are sized correctly.
        formats=InputOutputFormat(_L1_INPUT, _OUTPUT),
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            MATH_FIDELITY(math_fidelity),
            UNINIT_NUM_FACES(uninit_num_faces),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_A,
            _L1_INPUT,
            src_B,
            _L1_INPUT,
            _OUTPUT,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=DestAccumulation.No,
        L1_to_L1_iterations=L1_to_L1_iterations,
    )

    # Override the inferred (Float16_b) formats with the hand-built BFP pipeline.
    # formats_config is a runtime input (excluded from the build hash), so this
    # is picked up by write_runtimes_to_L1 without affecting compilation.
    configuration.formats_config = _bfp_matmul_formats()

    res_from_L1 = configuration.run().result
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    ok = passed_test(golden_tensor, res_tensor, _OUTPUT)

    if uninit_num_faces == 4:
        assert ok, (
            "Control (z-dim=4, the FIXED restore value) should match golden. "
            "If this fails, the fused tilize->BFP-matmul baseline itself is off, "
            "not the bug."
        )
    else:
        assert not ok, (
            "Expected the Bfp8_b matmul to be CORRUPT when the tilize-uninit "
            "restores z-dim != 4 (the #45179 regression). It matched golden "
            "instead, which would mean this build already has the z-dim=4 fix."
        )
