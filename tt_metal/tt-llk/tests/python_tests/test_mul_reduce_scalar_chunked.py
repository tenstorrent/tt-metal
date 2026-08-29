# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Chunked fused multiply + reduce-to-scalar LLK test (experimental, Blackhole only).

This is the REVERTED "chunked" driver for the experimental ``mul_reduce_scalar``
LLKs (promotion strategy §3, open-question #1). The non-chunked
``mul_reduce_scalar_tile`` caps ``num_tiles`` at the DEST half-sync capacity
(8 bf16 / 4 fp32) because every multiply product must be resident in DEST before
the reduce phase consumes it. The chunked driver processes the tile stream in
fixed-size chunks (each <= DEST capacity), reduces each chunk to a scalar, and
accumulates the chunk scalars into a running total held in DEST[0] between reduces:

    result = sum_over_all_tiles_and_elements(A * B)

stored in element ``[0]`` of the output tile. Chunking only changes the *order*
of accumulation, so the golden is identical to the non-chunked op. Only element
``[0]`` is defined (REDUCE_SCALAR pack mask); every other lane is unspecified, so
the test validates the reduced scalar alone.

Kernel B is held at 1.0 (matching the on-silicon gtest and
``fuser_config/fpu_reduce_scalar.yaml``), so A * B == A and the fused op reduces
to ``sum(A)`` over all tiles/elements.

XFAIL: on silicon the chunked result comes out ~5-30x too high. The suspected
cause is the between-chunk DEST[0] restore (the running scalar in DEST[0] is
clobbered / double-counted when the next chunk's multiply phase and fill sequence
re-touch DEST[0]). See promotion strategy §3. The test is expected to COMPILE
cleanly for Blackhole and to FAIL numerically at runtime.
"""

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, MathFidelity, format_dict
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    MATH_FIDELITY,
    MUL_REDUCE_SCALAR_CHUNK_SIZE,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    TILE_COUNT,
)
from helpers.tile_shape import construct_tile_shape
from helpers.utils import tolerances

# Inputs are always bf16; only the DEST/output precision varies.
FORMATS = [
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
    InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32),
]

# Full 32x32 tile (4 faces) plus the tiny tiles: 16x32 (2 faces) and 16x16 (1
# face). The reduce collapses every element to [0] regardless of tile geometry.
TILE_DIMENSIONS = [[32, 32], [16, 32], [16, 16]]

# Chunk size drives the whole point of this driver: it must fit the DEST slot
# budget (<= 8 bf16 / <= 4 fp32). Fixed at 4 so it is valid for both bf16 and
# native-fp32 DEST while still forcing multiple chunks for the larger tile counts.
CHUNK_SIZE = 4


def _dest_acc(output_format):
    """Native fp32 DEST is required whenever the output is Float32."""
    return (
        DestAccumulation.Yes
        if output_format == DataFormat.Float32
        else DestAccumulation.No
    )


def _num_tiles_for_format(formats):
    """Tile-stream lengths that span more than one chunk (CHUNK_SIZE=4), so the
    chunked driver's between-chunk accumulation is actually exercised. fp32 DEST
    holds only 4 slots per chunk; the counts here are total stream lengths, not
    per-chunk, so both formats use the same set."""
    return [4, 5, 8]


@pytest.mark.xfail(
    reason=(
        "Reverted chunked mul_reduce_scalar driver (promotion strategy §3, "
        "open-question #1): on-silicon result is ~5-30x too high, suspected the "
        "between-chunk DEST[0] restore double-counts the running scalar. Kept as "
        "a compile-clean placeholder until the driver is fixed."
    ),
    strict=False,
)
@parametrize(
    formats=FORMATS,
    math_fidelity=[MathFidelity.HiFi2, MathFidelity.HiFi4],
    num_tiles=_num_tiles_for_format,
    tile_dimensions=TILE_DIMENSIONS,
)
def test_mul_reduce_scalar_chunked(formats, math_fidelity, num_tiles, tile_dimensions):
    if get_chip_architecture() != ChipArchitecture.BLACKHOLE:
        pytest.skip("mul_reduce_scalar is a Blackhole-only experimental LLK")

    tile_shape = construct_tile_shape(tile_dimensions)
    elements_per_tile = tile_shape.total_tile_size()
    dest_acc = _dest_acc(formats.output_format)
    input_dimensions = [num_tiles * tile_dimensions[0], tile_dimensions[1]]

    # A ~ U[0, 1] mirrors the on-silicon gtest and keeps the accumulated sum well
    # inside bf16's dynamic range for the larger tile counts. Passing
    # tile_dimensions puts the generator in dense mode (real tiny-tile layout).
    src_A, tile_cnt_A, _, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=tile_dimensions,
        spec_A=StimuliSpec.uniform(low=0.0, high=1.0),
    )
    # B == 1.0 everywhere (matching the gtest and fpu_reduce_scalar.yaml):
    # A * B == A, so the fused op reduces to sum(A) over all tiles/elements.
    src_B = torch.ones(
        tile_cnt_B * elements_per_tile, dtype=format_dict[formats.input_format]
    )

    # Golden mirrors the non-chunked reference: the element-wise product summed
    # over every element of every tile, in fp32. Chunking is a pure reordering of
    # this sum, so the golden is unchanged from the non-chunked op.
    golden_scalar = float(
        (src_A.to(torch.float32) * src_B.to(torch.float32)).sum().item()
    )

    configuration = TestConfig(
        "sources/mul_reduce_scalar_chunked_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
        ],
        runtimes=[
            TILE_COUNT(num_tiles),
            MUL_REDUCE_SCALAR_CHUNK_SIZE(CHUNK_SIZE),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim, tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim, tile_shape.num_faces_c_dim),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=1,
            num_faces=tile_shape.total_num_faces(),
            face_r_dim=tile_shape.face_r_dim,
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
            sfpu=False,
        ),
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    assert (
        len(res_from_L1) == elements_per_tile
    ), f"Expected one {elements_per_tile}-element output tile, got {len(res_from_L1)}"

    # The reduced scalar lives in element [0]; every other lane is unspecified.
    device_scalar = float(res_from_L1[0])
    tol = tolerances[formats.output_format]
    assert abs(device_scalar - golden_scalar) <= tol.atol + tol.rtol * abs(
        golden_scalar
    ), (
        f"mul_reduce_scalar_chunked mismatch: device={device_scalar} golden={golden_scalar} "
        f"(num_tiles={num_tiles}, chunk={CHUNK_SIZE}, tile={tile_dimensions}, "
        f"fidelity={math_fidelity.name})"
    )
