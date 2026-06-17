# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-chip behavioural repro for tt-metal#47016 (and #47049, same root cause).

This is the *genuine* "unpack-tilize followed by matmul" pattern from the issues
(the lm_head path: in-kernel tilize of activations, then a matmul against
bfloat8_b weights). The fused tilize->matmul kernel drives all three cores with
the proper PACK_DONE handshake:

    run 0: unpack-TILIZE A and B from L1 -> datacopy -> PACK back to L1 as Bfp8_b
    run 1: matmul straight on the (Bfp8_b) tilized operands, with NO
           reconfig(FACE_ROW_MAJOR) before it (the model binding does not reconfig).

Why this is the faithful repro
------------------------------
For a BFP-compressed operand the unpack tile-descriptor z-dim sizes the per-tile
exponent / RowStart arrays (NumBlobs = BlobsPerXYPlane * ZDim * WDim). That z-dim
is PERSISTENT operand state - written only by _llk_unpack_hw_configure_ and by
reconfig(FACE_ROW_MAJOR). Neither the tilize datapath nor the matmul unpack
*execute* maintains it (both drive face iteration from the ADC counters). So the
matmul decodes its exponents using whatever value is sitting in the descriptor
when it runs.

In the model that value is NOT guaranteed to be 4: it is whatever a prior op left
there, and nothing between the tilize and the matmul refreshes it. The kernel
models that by stamping the SEC0/SEC1 descriptor z-dim to STALE_DESC_Z just before
the matmul (no reconfig to fix it):

    STALE_DESC_Z == 4 : control - descriptor already holds the full-tile count, so
                        the BFP matmul is correct regardless of the fix.
    STALE_DESC_Z == 2 : the bug condition - a non-full-tile z reaches the BFP
                        matmul. Correct ONLY if _llk_unpack_AB_matmul_init_ itself
                        re-establishes the descriptor z-dim from its num_faces.

So this test FAILS (STALE_DESC_Z=2) whenever nothing in the matmul unpack path owns
the z-dim - which is the state of #45179 AND of the partial "just stop writing z in
uninit" attempt (why that still broke the model). It PASSES both parametrizations
only once the matmul init programs the z-dim. A register-only test cannot catch
this, because the corruption is in the BFP exponent decode, not a face count.

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
    STALE_DESC_Z,
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
    # Descriptor z-dim the matmul inherits from prior ops (no reconfig refreshes it).
    # 4 = full-tile control; 2 = stale non-full-tile state that triggers #47016.
    stale_desc_z=[4, 2],
)
def test_repro_47016(math_fidelity, stale_desc_z):
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
            STALE_DESC_Z(stale_desc_z),
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

    assert ok, (
        f"Fused tilize->Bfp8_b matmul must match golden with an inherited descriptor "
        f"z-dim of {stale_desc_z}. A failure at stale_desc_z=2 means nothing in the "
        f"matmul unpack path re-establishes the tile-descriptor z-dim, so the stale "
        f"non-full-tile value mis-sizes the BFP exponent array -> the tt-metal#47016 "
        f"regression. The fix is _llk_unpack_AB_matmul_init_ programming z-dim from "
        f"its own num_faces."
    )
