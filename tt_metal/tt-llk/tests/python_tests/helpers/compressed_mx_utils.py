# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host side of the Quasar compressed_custom_mm: per-tile MX weight streams and their metadata.

The Quasar op (tt_llk_quasar/llk_lib/experimental/llk_compressed_custom_mm_common.h) takes the
weights in1 [kt_dim * 32, ct_dim * 32] as a stream of 32x32 tiles in row-major (k, c) order, each
packed in its own MX format, with all-zero tiles left out. Next to it sits one metadata word per
tile, which is the UNPACR0 ARG_FIFO mode-1 entry with a stream-relative address:

    [27:20] tile L1 format (Quasar DataFormat code), 0 marks a zero tile, whose whole word is 0
    [19:0]  tile offset from the start of the stream, in 16 B units

The face-granular op (llk_unpack_face_compressed_custom_mm.h) does the same per 16x16 face: the
stream is one single-face MX tile per stored face, and each weight tile has four metadata words,
one per face in face order, with the same entry layout. Zero faces point at one shared all-zero
face at the start of the stream.

This is the Quasar counterpart of the Blackhole helpers in compressed_utils.py, which encode
3-bit BFP codes instead.
"""

import random

import numpy as np
import torch

from .format_config import DataFormat
from .golden_generators import TILE_DIM
from .pack import pack_mxint2, pack_mxint4, pack_mxint8
from .tile_constants import FACE_C_DIM
from .tilize_untilize import tilize, untilize
from .unpack import unpack_mxint2, unpack_mxint4, unpack_mxint8

# Quasar hardware codes of the weight formats, from enum DataFormat in
# tt_metal/hw/inc/internal/tt-2xx/quasar/tensix_types.h. The device reads them straight out of the
# metadata, so they must match that enum, not the Python DataFormat.
MX_FORMAT_CODE = {
    DataFormat.MxInt8: 2,
    DataFormat.MxInt4: 3,
    DataFormat.MxInt2: 11,
}

# An assignment entry of ZERO_TILE is an all-zero weight tile: no data in L1, metadata word 0.
ZERO_TILE = None
# The same for a 16x16 face of the face-granular op: no data of its own, its metadata word points
# at the shared zero face.
ZERO_FACE = None

FACE_DIM = FACE_C_DIM
# The shared zero face at the start of a face stream is packed in the narrowest format.
_ZERO_FACE_FORMAT = DataFormat.MxInt2

COMPRESSED_FORMATS = list(MX_FORMAT_CODE)

_ENTRY_FORMAT_SHIFT = 20
_ENTRY_ADDR_MASK = (1 << _ENTRY_FORMAT_SHIFT) - 1
_L1_ALIGNMENT = 16

_PACKERS = {
    DataFormat.MxInt8: pack_mxint8,
    DataFormat.MxInt4: pack_mxint4,
    DataFormat.MxInt2: pack_mxint2,
}
_UNPACKERS = {
    DataFormat.MxInt8: unpack_mxint8,
    DataFormat.MxInt4: unpack_mxint4,
    DataFormat.MxInt2: unpack_mxint2,
}


def assign_uniform(kt_dim, ct_dim, data_format):
    """Every weight tile in ``data_format``."""
    return [data_format] * (kt_dim * ct_dim)


def assign_random(kt_dim, ct_dim, zero_fraction, seed=0):
    """Each weight tile independently zero with ``zero_fraction``, else a random MX format.

    Deterministic for a given seed, so a failing variant reproduces.
    """
    rng = random.Random(seed)
    return [
        ZERO_TILE if rng.random() < zero_fraction else rng.choice(COMPRESSED_FORMATS)
        for _ in range(kt_dim * ct_dim)
    ]


def assign_with_zero_rows(kt_dim, ct_dim, seed=0):
    """Random formats, with every odd k-row entirely zero.

    A k-row without a single stored tile reads nothing from the weight stream: every SrcA bank
    of that row is a zeroed one, multiplied against the row's SrcB like any other.
    """
    assignment = assign_random(kt_dim, ct_dim, zero_fraction=0.25, seed=seed)
    for k in range(1, kt_dim, 2):
        assignment[k * ct_dim : (k + 1) * ct_dim] = [ZERO_TILE] * ct_dim
    return assignment


def _pack_tile(tile, data_format):
    """Pack one row-major 32x32 tile; returns (L1 bytes, dequantized row-major tile)."""
    faces = tilize(tile.reshape(-1), stimuli_format=DataFormat.Float16_b)
    packed = bytes(_PACKERS[data_format](faces))
    dequant_faces = _UNPACKERS[data_format](packed)
    dequant = untilize(dequant_faces, stimuli_format=DataFormat.Float16_b)
    return packed, dequant.reshape(TILE_DIM, TILE_DIM)


def pack_compressed_in1(in1, kt_dim, ct_dim, assignment):
    """Pack in1 into the compressed weight stream and its metadata.

    Args:
        in1: Row-major weights [kt_dim * 32, ct_dim * 32].
        kt_dim, ct_dim: Weight grid in tiles.
        assignment: One entry per tile in (k, c) order, an MX DataFormat or ZERO_TILE.

    Returns:
        (stream, metadata, in1_dequant): the L1 images of the stream and of the metadata words,
        and in1 as the device sees it after MX quantization, for the golden.
    """
    assert len(assignment) == kt_dim * ct_dim, "one assignment entry per weight tile"

    stream = bytearray()
    entries = []
    in1_dequant = torch.zeros_like(in1, dtype=torch.bfloat16)

    for k in range(kt_dim):
        for c in range(ct_dim):
            data_format = assignment[k * ct_dim + c]
            if data_format is ZERO_TILE:
                entries.append(0)
                continue

            rows = slice(k * TILE_DIM, (k + 1) * TILE_DIM)
            cols = slice(c * TILE_DIM, (c + 1) * TILE_DIM)
            packed, dequant = _pack_tile(in1[rows, cols], data_format)

            offset_16B = len(stream) // _L1_ALIGNMENT
            assert (
                offset_16B <= _ENTRY_ADDR_MASK
            ), "weight stream exceeds the 20-bit offset"
            entries.append(
                offset_16B | (MX_FORMAT_CODE[data_format] << _ENTRY_FORMAT_SHIFT)
            )
            stream += packed
            in1_dequant[rows, cols] = dequant

    metadata = np.array(entries, dtype=np.uint32).tobytes()
    return bytes(stream), metadata, in1_dequant


def expand_tiles_to_faces(assignment, kt_dim, ct_dim):
    """Expand a per-tile assignment ((k, c) order) to faces: every face takes its tile's entry.

    Returns the (2 * kt_dim) x (2 * ct_dim) face grid in row-major order, the order the face
    assignment generators and ``pack_face_compressed_in1`` use.
    """
    faces_r = faces_c = TILE_DIM // FACE_DIM
    grid = [[None] * (ct_dim * faces_c) for _ in range(kt_dim * faces_r)]
    for k in range(kt_dim):
        for c in range(ct_dim):
            entry = assignment[k * ct_dim + c]
            for fr in range(faces_r):
                for fc in range(faces_c):
                    grid[k * faces_r + fr][c * faces_c + fc] = entry
    return [entry for row in grid for entry in row]


def _pack_face(face, data_format):
    """Pack one row-major 16x16 face as a single-face MX tile; returns (L1 bytes, dequantized face)."""
    packed = bytes(
        _PACKERS[data_format](face.reshape(-1), num_faces=1, face_r_dim=FACE_DIM)
    )
    dequant = _UNPACKERS[data_format](packed, num_faces=1, face_r_dim=FACE_DIM)
    return packed, dequant.reshape(FACE_DIM, FACE_DIM)


def pack_face_compressed_in1(in1, kt_dim, ct_dim, assignment):
    """Pack in1 into the face-compressed weight stream and its metadata.

    Args:
        in1: Row-major weights [kt_dim * 32, ct_dim * 32].
        kt_dim, ct_dim: Weight grid in tiles.
        assignment: One entry per 16x16 face over the (2 * kt_dim) x (2 * ct_dim) face grid in
            row-major order, an MX DataFormat or ZERO_FACE.

    Returns:
        (stream, metadata, in1_dequant): the L1 images of the stream and of the metadata words,
        and in1 as the device sees it after MX quantization, for the golden.
    """
    faces_r = faces_c = TILE_DIM // FACE_DIM
    assert (
        len(assignment) == kt_dim * faces_r * ct_dim * faces_c
    ), "one assignment entry per weight face"

    # The shared zero face opens the stream, so a zero face's entry is its format at offset 0.
    zero_face, _ = _pack_face(torch.zeros(FACE_DIM, FACE_DIM), _ZERO_FACE_FORMAT)
    stream = bytearray(zero_face)
    zero_entry = MX_FORMAT_CODE[_ZERO_FACE_FORMAT] << _ENTRY_FORMAT_SHIFT
    entries = []
    in1_dequant = torch.zeros_like(in1, dtype=torch.bfloat16)

    for k in range(kt_dim):
        for c in range(ct_dim):
            # Face order within the tile: 0 top-left, 1 top-right, 2 bottom-left, 3 bottom-right.
            for fr in range(faces_r):
                for fc in range(faces_c):
                    data_format = assignment[
                        (k * faces_r + fr) * (ct_dim * faces_c) + c * faces_c + fc
                    ]
                    if data_format is ZERO_FACE:
                        entries.append(zero_entry)
                        continue

                    rows = slice(
                        k * TILE_DIM + fr * FACE_DIM, k * TILE_DIM + (fr + 1) * FACE_DIM
                    )
                    cols = slice(
                        c * TILE_DIM + fc * FACE_DIM, c * TILE_DIM + (fc + 1) * FACE_DIM
                    )
                    packed, dequant = _pack_face(in1[rows, cols], data_format)

                    offset_16B = len(stream) // _L1_ALIGNMENT
                    assert (
                        offset_16B <= _ENTRY_ADDR_MASK
                    ), "weight stream exceeds the 20-bit offset"
                    entries.append(
                        offset_16B
                        | (MX_FORMAT_CODE[data_format] << _ENTRY_FORMAT_SHIFT)
                    )
                    stream += packed
                    in1_dequant[rows, cols] = dequant

    metadata = np.array(entries, dtype=np.uint32).tobytes()
    return bytes(stream), metadata, in1_dequant
