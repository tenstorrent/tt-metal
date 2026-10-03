# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Chunked fused multiply + reduce-to-scalar LLK test (experimental, Blackhole only).

The kernel expands ``mul_reduce_scalar_chunked_tile`` (``api/compute/experimental/rmsnorm.h``)
into its LLK calls. ``CHUNK_SIZE`` is the API's ``dst_capacity``: DEST slot ``CHUNK_SIZE - 1``
holds the running scalar and the other slots stage the products of a batch of
``CHUNK_SIZE - 1`` input tiles, each cleared before its multiply, with the unpack and
math re-initialised between batches. The API takes rows longer than ``dst_capacity``.

B is held at 1.0 (matching the on-silicon gtest and ``fuser_config/fpu_reduce_scalar.yaml``),
so the op reduces to ``sum(A)`` over all tiles and elements, stored in element ``[0]`` of
the accumulator tile; every other lane is unspecified (REDUCE_SCALAR pack mask).
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

def _dest_acc(output_format):
    """Native fp32 DEST is required whenever the output is Float32."""
    return (
        DestAccumulation.Yes
        if output_format == DataFormat.Float32
        else DestAccumulation.No
    )


def _chunk_sizes_for_format(formats):
    """The API's dst_capacity: up to the DEST half-sync capacity (8 bf16 / 4 fp32 tiles)."""
    return [4] if _dest_acc(formats.output_format) == DestAccumulation.Yes else [4, 8]


def _num_tiles_for_chunk(chunk_size):
    """Rows longer than dst_capacity: two batches, a trailing one-tile batch, and four batches."""
    batch = chunk_size - 1
    return [batch + 2, 2 * batch + 1, 4 * batch]


@parametrize(
    formats=FORMATS,
    math_fidelity=[MathFidelity.HiFi2, MathFidelity.HiFi4],
    chunk_size=_chunk_sizes_for_format,
    num_tiles=_num_tiles_for_chunk,
    tile_dimensions=TILE_DIMENSIONS,
)
def test_mul_reduce_scalar_chunked(
    formats, math_fidelity, chunk_size, num_tiles, tile_dimensions
):
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
            MUL_REDUCE_SCALAR_CHUNK_SIZE(chunk_size),
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
        f"(num_tiles={num_tiles}, chunk={chunk_size}, tile={tile_dimensions}, "
        f"fidelity={math_fidelity.name})"
    )
