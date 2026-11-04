# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import format_dict
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import GATHER_COMPRESS_PARAMS

TILE_DIMENSIONS = [32, 32]
TILE_ELEMS = TILE_DIMENSIONS[0] * TILE_DIMENSIONS[1]
NUM_FACES = 4
FACE_R_DIM = 16

# Every format the gather / compress LLK accepts: 16-bit formats move 32-element vectors, 8-bit ones 64.
FORMATS_16BIT = [DataFormat.Float16_b]
FORMATS_8BIT = [DataFormat.Int8, DataFormat.UInt8]

# Value range per integer format; floats come from randn.
INT_RANGES = {
    DataFormat.Int8: (-127, 128),
    DataFormat.UInt8: (0, 256),
}


def elem_cnt_for(data_format: DataFormat) -> int:
    return 64 if data_format in FORMATS_8BIT else 32


def generate_source(data_format: DataFormat, gen: torch.Generator) -> torch.Tensor:
    if data_format in INT_RANGES:
        low, high = INT_RANGES[data_format]
        return torch.randint(low, high, (TILE_ELEMS,), generator=gen, dtype=torch.int64)
    return torch.randn(TILE_ELEMS, generator=gen).to(format_dict[data_format])


def make_gather_indices(elem_cnt: int, num_vectors: int, gen: torch.Generator):
    """Random indices in [0, elem_cnt), one per output lane (repeats allowed), a fresh set per vector.

    Returns the index bytes laid out one ARG FIFO entry per vector, and each vector's source lanes.
    """
    indices = torch.randint(
        0, elem_cnt, (num_vectors, elem_cnt), generator=gen, dtype=torch.uint8
    )
    return indices.flatten(), [row.to(torch.int64) for row in indices]


def make_compress_mask(elem_cnt: int, num_vectors: int, gen: torch.Generator):
    """Random keep mask packed LSB-first, one bit per element, a fresh mask per vector.

    Returns the mask bytes laid out one ARG FIFO entry per vector, and each vector's kept lanes.
    """
    keep = torch.randint(
        0, 2, (num_vectors, elem_cnt), generator=gen, dtype=torch.uint8
    )
    weights = torch.tensor([1 << b for b in range(8)], dtype=torch.int32)
    packed = (
        (keep.view(num_vectors, -1, 8).to(torch.int32) * weights)
        .sum(dim=2)
        .to(torch.uint8)
    )
    return packed.flatten(), [torch.nonzero(row, as_tuple=True)[0] for row in keep]


@pytest.mark.quasar
@pytest.mark.parametrize(
    "data_format",
    FORMATS_16BIT + FORMATS_8BIT,
    ids=lambda f: f.name,
)
@pytest.mark.parametrize("compress", [False, True], ids=["gather", "compress"])
def test_gather_compress_tile_quasar(data_format, compress):
    from helpers.chip_architecture import quasar_variant

    if quasar_variant() != "saturn":
        pytest.skip("Gather/compress requires TT_METAL_QUASAR_VARIANT=saturn")

    elem_cnt = elem_cnt_for(data_format)
    num_vectors = TILE_ELEMS // elem_cnt

    gen = torch.Generator().manual_seed(42)
    src_A = generate_source(data_format, gen)
    index_buffer, golden_src_idx = (
        make_compress_mask(elem_cnt, num_vectors, gen)
        if compress
        else make_gather_indices(elem_cnt, num_vectors, gen)
    )
    src_vectors = src_A.view(num_vectors, elem_cnt)
    # Compress defines only the first popcount(mask) lanes of each vector, so goldens are ragged.
    golden = [src_vectors[v, golden_src_idx[v]] for v in range(num_vectors)]

    formats = InputOutputFormat(input_format=data_format, output_format=data_format)
    configuration = TestConfig(
        "sources/quasar/gather_compress_tile_quasar_test.cpp",
        formats,
        templates=[GATHER_COMPRESS_PARAMS(compress, elem_cnt)],
        variant_stimuli=StimuliConfig(
            src_A,
            data_format,
            index_buffer,
            DataFormat.UInt8,
            data_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
            num_faces=NUM_FACES,
            face_r_dim=FACE_R_DIM,
            tile_dimensions=TILE_DIMENSIONS,
            use_dense_tile_dimensions=True,
        ),
        requires_vector_ext=True,
    )

    res_from_L1 = configuration.run().result
    assert len(res_from_L1) == TILE_ELEMS, "Result does not hold one 32x32 tile"
    res = torch.tensor(res_from_L1, dtype=src_A.dtype).view(num_vectors, elem_cnt)

    mismatches = [
        (v, i, golden[v][i].item(), res[v, i].item())
        for v in range(num_vectors)
        for i in range(len(golden[v]))
        if res[v, i] != golden[v][i]
    ]
    checked = sum(len(g) for g in golden)
    op = "compress" if compress else "gather"
    assert not mismatches, (
        f"{data_format.name} tile {op} returned the wrong element for {len(mismatches)} of "
        f"{checked} elements (vector, lane, expected, actual), first 32: "
        + str(mismatches[:32])
    )
