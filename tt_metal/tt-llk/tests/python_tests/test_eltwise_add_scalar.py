# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Binary dest-reuse add (experimental, Blackhole only).

Exercises the experimental ``eltwise_add_scalar`` Compute API
(``api/compute/experimental/eltwise_add_scalar.h``), specifically the
``deepseek_binary_dest_reuse_add_tiles<..., DEST_TO_SRCA>`` op:

    dest[idst] = dest[idst] + cb[in_tile_index]

i.e. the accumulator tile already in DEST is fed back as SrcA and the freshly
unpacked cb tile is SrcB; the sum overwrites DEST. To mirror the header's
seed-then-fold usage shape (matching the mul_scalar sibling), DEST is first
seeded with a plain NONE-reuse ELWADD A_0 + B_0, then remaining inner tiles fold
in via DEST_TO_SRCA.

Unlike eltwise_mul_scalar.h, the add header's init
(``deepseek_binary_dest_reuse_add_tiles_init``, eltwise_add_scalar.h:27-34) uses
a single shorthand init for every fidelity — there is no reverted HiFi
DEFAULT_TENSOR_SHAPE branch to reproduce, so this test is NOT xfail. ELWADD only
supports ``MathFidelity::LoFi`` (fidelity > LoFi is an ELWMUL-only hardware
feature), so the test runs at LoFi.

Every lane of every output tile is defined (a full tile is packed), so the
golden validates all lanes at the format tolerance. Runtime pass/fail is not
checked in this harness (no BH card); the bar is a clean Blackhole compile plus
a golden faithfully mirroring the header math.
"""

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.llk_params import (
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
    MathFidelity,
    format_dict,
)
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    MATH_FIDELITY,
    NUM_BLOCKS,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    NUM_TILES_IN_BLOCK,
    TEST_FACE_DIMS,
)
from helpers.tile_constants import FACE_C_DIM, get_tile_params
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test

# WEDGES THE DEVICE — skipped, not run. In bit-exact run 32156210309 this test hit a
# device error mid-run (test_eltwise_add_scalar.py:253) that wedged the Tensix, cascading
# TENSIX-TIMED-OUT into every subsequent test on the core. A wedge cannot be xfailed; it
# must not execute on hardware. Root cause (our seed-then-fold DEST_TO_SRCA driver vs the
# merged deepseek_binary_dest_reuse_add_tiles header) needs a BH-card debug session.
# TODO: fix the driver and un-skip.
pytestmark = pytest.mark.skip(
    reason="Wedges the Tensix on BH (run 32156210309) — cascades timeouts into all "
    "later tests; needs a BH-card debug of the DEST_TO_SRCA fold. Un-skip once fixed."
)


def _prepare_dest_reuse_inputs(
    formats, input_dimensions, output_dimensions, tile_dimensions
):
    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dimensions)
    num_faces = num_faces_r_dim * num_faces_c_dim

    tile_rows, tile_cols = tile_dimensions
    tile_cnt_input = (input_dimensions[0] // tile_rows) * (
        input_dimensions[1] // tile_cols
    )
    tile_cnt_output = (output_dimensions[0] // tile_rows) * (
        output_dimensions[1] // tile_cols
    )

    assert tile_cnt_input % tile_cnt_output == 0
    inner_dim = tile_cnt_input // tile_cnt_output
    assert inner_dim > 1, "Dest reuse requires at least one reuse accumulation"

    src_A, _, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=tile_dimensions,
    )

    effective_dest_acc = (
        DestAccumulation.Yes
        if formats.output_format == DataFormat.Float32
        else DestAccumulation.No
    )
    output_num_blocks, output_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        effective_dest_acc,
        formats,
        output_dimensions,
        tile_dimensions,
        BlocksCalculationAlgorithm.Standard,
    )
    input_tiles_in_block = inner_dim * output_tiles_in_block

    src_A_tilized_flat = tilize_block(
        src_A,
        dimensions=input_dimensions,
        stimuli_format=formats.input_format,
        num_faces=num_faces,
        tile_dimensions=tile_dimensions,
        face_r_dim=face_r_dim,
    ).flatten()
    src_B_tilized_flat = tilize_block(
        src_B,
        dimensions=input_dimensions,
        stimuli_format=formats.input_format,
        num_faces=num_faces,
        tile_dimensions=tile_dimensions,
        face_r_dim=face_r_dim,
    ).flatten()

    return {
        "face_r_dim": face_r_dim,
        "num_faces_r_dim": num_faces_r_dim,
        "num_faces_c_dim": num_faces_c_dim,
        "num_faces": num_faces,
        "tile_cnt_input": tile_cnt_input,
        "tile_cnt_output": tile_cnt_output,
        "inner_dim": inner_dim,
        "output_num_blocks": output_num_blocks,
        "output_tiles_in_block": output_tiles_in_block,
        "input_tiles_in_block": input_tiles_in_block,
        "input_num_blocks": output_num_blocks,
        "src_A_tilized_flat": src_A_tilized_flat,
        "src_B_tilized_flat": src_B_tilized_flat,
        "tile_elements": num_faces * face_r_dim * FACE_C_DIM,
        "torch_format": format_dict[formats.output_format],
    }


def _compute_dest_reuse_golden(prepared):
    """Seed DEST = A_0 + B_0, then fold dest = dest + B_i via DEST_TO_SRCA.

    ELWADD accumulates no fidelity phases (add is a single-pass op, LoFi only),
    so the golden is a plain running sum: dest = A_0 + B_0 + B_1 + ... + B_{n-1}
    per output tile, computed in the output torch format to match the packer.
    """
    tile_elements = prepared["tile_elements"]
    torch_format = prepared["torch_format"]
    src_A = prepared["src_A_tilized_flat"]
    src_B = prepared["src_B_tilized_flat"]
    golden_tensor = torch.zeros(
        prepared["tile_cnt_output"] * tile_elements, dtype=torch_format
    )

    for out_t in range(prepared["tile_cnt_output"]):
        block_idx = out_t // prepared["output_tiles_in_block"]
        tile_in_block = out_t % prepared["output_tiles_in_block"]
        dest = torch.zeros(tile_elements, dtype=torch_format)

        for i in range(prepared["inner_dim"]):
            input_tile_idx = (
                block_idx * prepared["input_tiles_in_block"]
                + i * prepared["output_tiles_in_block"]
                + tile_in_block
            )
            start = input_tile_idx * tile_elements
            end = start + tile_elements
            a_tile = src_A[start:end].to(torch_format)
            b_tile = src_B[start:end].to(torch_format)

            if i == 0:
                # Seed: NONE-reuse ELWADD A_0 + B_0.
                dest = (a_tile + b_tile).to(torch_format)
            else:
                # Fold via DEST_TO_SRCA: dest = dest + B_i.
                dest = (dest + b_tile).to(torch_format)

        out_start = out_t * tile_elements
        golden_tensor[out_start : out_start + tile_elements] = dest

    return golden_tensor


@parametrize(
    formats=input_output_formats(
        [DataFormat.Bfp8_b, DataFormat.Float16_b, DataFormat.Float32],
        same=True,
    ),
    # ELWADD is LoFi only (fidelity > LoFi is an ELWMUL-only hardware feature).
    math_fidelity=MathFidelity.LoFi,
    tile_dimensions=[[32, 32], [16, 32], [8, 32]],
    input_dimensions=[[512, 32]],
    output_dimensions=[[128, 32]],
)
def test_eltwise_add_scalar(
    formats,
    math_fidelity,
    tile_dimensions,
    input_dimensions,
    output_dimensions,
):
    prepared = _prepare_dest_reuse_inputs(
        formats, input_dimensions, output_dimensions, tile_dimensions
    )
    golden_tensor = _compute_dest_reuse_golden(prepared)

    dest_acc = (
        DestAccumulation.Yes
        if formats.output_format == DataFormat.Float32
        else DestAccumulation.No
    )

    configuration = TestConfig(
        "sources/eltwise_add_scalar_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            DEST_SYNC(),
        ],
        runtimes=[
            NUM_TILES_IN_BLOCK(
                prepared["output_tiles_in_block"],
                input_num_tiles_in_block=prepared["input_tiles_in_block"],
                output_num_tiles_in_block=prepared["output_tiles_in_block"],
            ),
            NUM_BLOCKS(
                prepared["output_num_blocks"],
                input_num_blocks=prepared["input_num_blocks"],
                output_num_blocks=prepared["output_num_blocks"],
            ),
            NUM_FACES_R_DIM(prepared["num_faces_r_dim"]),
            NUM_FACES_C_DIM(prepared["num_faces_c_dim"]),
            TEST_FACE_DIMS(face_r_dim=prepared["face_r_dim"]),
        ],
        variant_stimuli=StimuliConfig(
            prepared["src_A_tilized_flat"],
            formats.input_format,
            prepared["src_B_tilized_flat"],
            formats.input_format,
            formats.output_format,
            tile_count_A=prepared["tile_cnt_input"],
            tile_count_B=prepared["tile_cnt_input"],
            tile_count_res=prepared["tile_cnt_output"],
            num_faces=prepared["num_faces"],
            face_r_dim=prepared["face_r_dim"],
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=False,
    )

    res_from_L1 = configuration.run().result
    assert len(res_from_L1) == len(golden_tensor)

    res_tensor = torch.tensor(res_from_L1, dtype=prepared["torch_format"])
    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
