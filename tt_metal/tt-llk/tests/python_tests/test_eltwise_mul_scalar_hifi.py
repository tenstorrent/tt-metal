# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Binary dest-reuse multiply, HiFi init path (experimental, Blackhole only).

Exercises the experimental ``eltwise_mul_scalar`` Compute API
(``api/compute/experimental/eltwise_mul_scalar.h``), specifically the
``deepseek_binary_dest_reuse_tiles<..., DEST_TO_SRCA>`` op:

    dest[idst] = dest[idst] * cb[in_tile_index]

i.e. the accumulator tile already in DEST is fed back as SrcA and the freshly
unpacked cb tile is SrcB; the product overwrites DEST. Because zero annihilates
a product, DEST is first seeded with a real value (a plain NONE-reuse ELWMUL
A_0 * B_0), then remaining inner tiles fold in via DEST_TO_SRCA. This is the
seed-then-fold shape of a ttnn silu(gate)*up MoE kernel.

REVERTED / XFAIL — HiFi init path. ``deepseek_binary_dest_reuse_tiles_init``
takes the GENERAL math init at HiFi (eltwise_mul_scalar.h:74-88), hard-coding
``ckernel::DEFAULT_TENSOR_SHAPE`` (a full 32x32 tile) instead of the kernel's
real tile shape. That mis-specialization HANGS the device on silicon (tt-blaze
#1760, strategy §9). The C++ reproduces the reverted HiFi init verbatim under
``HIFI_GENERAL_INIT`` and MUST COMPILE cleanly for Blackhole; runtime pass/fail
is not checked in this harness (no BH card), and on real silicon it hangs — so
this test is marked ``pytest.mark.xfail``.

Every lane of every output tile is defined (a full tile is packed), so the
golden validates all lanes at the format tolerance.
"""

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import EltwiseBinaryGolden
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
    MULSCALARHIFI_HIFI_INIT,
    NUM_BLOCKS,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    NUM_TILES_IN_BLOCK,
    TEST_FACE_DIMS,
)
from helpers.tile_constants import FACE_C_DIM, get_tile_params
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test


def _hifi_fidelity_for_format(formats):
    """HiFi only (the reverted general-init path is HiFi-specific), respecting the
    hardware fidelity/format rule: Float16_b -> HiFi2, Float32 -> HiFi3/HiFi4."""
    if formats.input_format == DataFormat.Float32:
        return [MathFidelity.HiFi3, MathFidelity.HiFi4]
    return [MathFidelity.HiFi2]


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


def _apply_mul_fidelity(
    math_fidelity, formats, torch_format, binary_golden, srcA, srcB
):
    """ELWMUL with per-fidelity-phase masking, matching the on-silicon math."""
    fidelity_iters = {
        MathFidelity.LoFi: 1,
        MathFidelity.HiFi2: 2,
        MathFidelity.HiFi3: 3,
        MathFidelity.HiFi4: 4,
    }[math_fidelity]
    result = None
    for fidelity_iter in range(fidelity_iters):
        a_m, b_m = binary_golden._apply_fidelity_masking(
            formats.output_format, srcA, srcB, fidelity_iter
        )
        phase = a_m.to(torch.float32) * b_m.to(torch.float32)
        result = phase if result is None else result + phase
    return result.to(torch_format)


def _compute_dest_reuse_golden(math_fidelity, formats, prepared):
    """Seed DEST = A_0 * B_0, then fold dest = dest * B_i via DEST_TO_SRCA."""
    tile_elements = prepared["tile_elements"]
    torch_format = prepared["torch_format"]
    src_A = prepared["src_A_tilized_flat"]
    src_B = prepared["src_B_tilized_flat"]
    golden_tensor = torch.zeros(
        prepared["tile_cnt_output"] * tile_elements, dtype=torch_format
    )
    binary_golden = EltwiseBinaryGolden()

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
                srcA, srcB = a_tile, b_tile
            else:
                # DEST_TO_SRCA: reused DEST is SrcA, cb tile is SrcB.
                srcA, srcB = dest.clone(), b_tile

            dest = _apply_mul_fidelity(
                math_fidelity, formats, torch_format, binary_golden, srcA, srcB
            )

        out_start = out_t * tile_elements
        golden_tensor[out_start : out_start + tile_elements] = dest

    return golden_tensor


@pytest.mark.skip(
    reason=(
        "REVERTED HiFi init HANGS the device on silicon (tt-blaze #1760, strategy §9): "
        "deepseek_binary_dest_reuse_tiles_init takes the general math init at HiFi with a "
        "hard-coded DEFAULT_TENSOR_SHAPE (eltwise_mul_scalar.h:74-88), mis-specializing the "
        "tile shape. A device hang CANNOT be xfailed -- it wedges every subsequent test on "
        "the Tensix (confirmed in run 32156210309) -- so this is skipped, not xfailed. "
        "Un-skip once the HiFi init is fixed."
    )
)
@parametrize(
    formats=input_output_formats(
        [DataFormat.Float16_b, DataFormat.Float32],
        same=True,
    ),
    math_fidelity=lambda formats: _hifi_fidelity_for_format(formats),
    tile_dimensions=[[32, 32], [16, 32], [8, 32]],
    input_dimensions=[[512, 32]],
    output_dimensions=[[128, 32]],
)
def test_eltwise_mul_scalar_hifi(
    formats,
    math_fidelity,
    tile_dimensions,
    input_dimensions,
    output_dimensions,
):
    prepared = _prepare_dest_reuse_inputs(
        formats, input_dimensions, output_dimensions, tile_dimensions
    )
    golden_tensor = _compute_dest_reuse_golden(math_fidelity, formats, prepared)

    dest_acc = (
        DestAccumulation.Yes
        if formats.output_format == DataFormat.Float32
        else DestAccumulation.No
    )

    configuration = TestConfig(
        "sources/eltwise_mul_scalar_hifi_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            DEST_SYNC(),
            # Select the reverted HiFi general-init path (#define HIFI_GENERAL_INIT).
            MULSCALARHIFI_HIFI_INIT(enabled=True),
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
