# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Getting tensors into and out of L1 for the heavyweight data-transfer blocks.

L1 holds bytes, so the blocks take bytes. There is no quantization step anywhere
in this package: storing a tensor as MxFp8R *is* the quantization, and by the
time those bytes exist the loss has already happened. Unpacking them just reads
back what is there.

This module is the thin dispatch over the real codecs in :mod:`helpers.pack` and
:mod:`helpers.unpack` — one entry point each way, so a chain of blocks can hand
L1 buffers to each other the way the hardware does.
"""

import inspect
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence, Union

import torch
from helpers.format_config import DataFormat
from helpers.llk_params import format_tile_sizes
from helpers.pack import (
    pack_bfp2_b,
    pack_bfp4_b,
    pack_bfp8_b,
    pack_bfp16,
    pack_fp8_e4m3,
    pack_fp16,
    pack_fp32,
    pack_int8,
    pack_int16,
    pack_int32,
    pack_mxfp4,
    pack_mxfp8p,
    pack_mxfp8r,
    pack_mxint2,
    pack_mxint4,
    pack_mxint8,
    pack_uint8,
    pack_uint16,
    pack_uint32,
)
from helpers.tile_constants import (
    FACE_C_DIM,
    MAX_FACE_R_DIM,
    MAX_NUM_FACES,
    calculate_tile_size_bytes,
)
from helpers.unpack import unpack_res_tiles

#: Tensor -> L1 bytes, per format. Mirrors ``StimuliConfig.get_packer``, which is
#: what the test harness uses to write operands into device L1.
PACKERS: Dict[DataFormat, Callable] = {
    DataFormat.Float32: pack_fp32,
    DataFormat.Float16: pack_fp16,
    DataFormat.Float16_b: pack_bfp16,
    DataFormat.Fp8_e4m3: pack_fp8_e4m3,
    DataFormat.Bfp8_b: pack_bfp8_b,
    DataFormat.Bfp4_b: pack_bfp4_b,
    DataFormat.Bfp2_b: pack_bfp2_b,
    DataFormat.MxFp8R: pack_mxfp8r,
    DataFormat.MxFp8P: pack_mxfp8p,
    DataFormat.MxFp4: pack_mxfp4,
    DataFormat.MxInt8: pack_mxint8,
    DataFormat.MxInt4: pack_mxint4,
    DataFormat.MxInt2: pack_mxint2,
    DataFormat.Int32: pack_int32,
    DataFormat.UInt32: pack_uint32,
    DataFormat.Int16: pack_int16,
    DataFormat.UInt16: pack_uint16,
    DataFormat.Int8: pack_int8,
    DataFormat.UInt8: pack_uint8,
}


def _call_accepted(fn: Callable, tensor: torch.Tensor, **kwargs):
    """Call `fn` passing only the keyword arguments it declares.

    The packers take different subsets of the tile geometry — ``pack_fp16`` takes
    none, ``pack_bfp8_b`` takes faces, the MX packers also take the SrcS layout
    flags and the extra rounding controls. Filtering by signature keeps one call site.
    """
    accepted = inspect.signature(fn).parameters
    return fn(tensor, **{k: v for k, v in kwargs.items() if k in accepted})


def datums_per_tile(
    num_faces: int = MAX_NUM_FACES, face_r_dim: int = MAX_FACE_R_DIM
) -> int:
    """Datums one tile holds at this geometry."""
    return num_faces * face_r_dim * FACE_C_DIM


def tile_dimensions_for(num_faces: int, face_r_dim: int) -> list:
    """``[rows, cols]`` for a tile of `num_faces` faces of `face_r_dim` rows.

    Faces tile the 32-datum row before they stack, so up to two sit side by
    side and the rest go underneath: 1 face is 16 wide, 2 or more are 32.
    """
    faces_c = min(num_faces, 2)
    return [(num_faces // faces_c) * face_r_dim, faces_c * FACE_C_DIM]


def tile_bytes_for(
    l1_format: DataFormat,
    num_faces: int = MAX_NUM_FACES,
    face_r_dim: int = MAX_FACE_R_DIM,
    use_srcs: bool = False,
    dest_acc: bool = False,
) -> int:
    """Bytes one tile of `l1_format` occupies in L1, as the harness sizes it.

    Delegates to :func:`helpers.tile_constants.calculate_tile_size_bytes`, the
    same sizing ``StimuliConfig`` uses, because a datum count is not enough:

    * the **BFP** packers hold a minimum of 16 exponents, so a tile with fewer
      than 8 rows per face occupies more bytes than its datums imply -- 48
      rather than 34 for a 1x32 ``Bfp8_b`` tile.
    * with **use_srcs** an MX tile is written as 16-byte-aligned per-slice
      blocks, 1152 bytes rather than the dense 1056, and 1280 under
      ``dest_acc``.

    Computing the stride from ``num_bytes_per_tile(datums)`` misses both, which
    truncates a single-tile read and misaligns every tile after the first.
    """
    return calculate_tile_size_bytes(
        l1_format,
        tile_dimensions_for(num_faces, face_r_dim),
        format_tile_sizes,
        use_srcs=use_srcs,
        dest_acc=dest_acc,
    )


def pack_to_l1(
    tensor: torch.Tensor,
    l1_format: DataFormat,
    *,
    tile_count: Optional[int] = None,
    num_faces: int = MAX_NUM_FACES,
    face_r_dim: int = MAX_FACE_R_DIM,
    use_srcs: bool = False,
    dest_acc: bool = False,
) -> List[int]:
    """Lay `tensor` out in L1 as `l1_format`, returning the bytes.

    This is where precision is lost for the block-scaled formats — the bytes that
    come back are the quantized truth, and nothing downstream re-quantizes.

    The codecs in :mod:`helpers.pack` handle exactly one tile, so a multi-tile
    tensor is split and packed tile by tile, matching how ``unpack_res_tiles``
    reads it back. `tile_count` defaults to what the tensor holds at this
    geometry.
    """
    packer = PACKERS.get(l1_format)
    if packer is None:
        raise ValueError(_no_codec_message(l1_format))
    if tensor.dtype is torch.bfloat16:
        # numpy has no bfloat16 and most packers go straight to .numpy().
        # float32 holds every bf16 value exactly, so this is lossless.
        tensor = tensor.to(torch.float32)

    flat = tensor.reshape(-1)
    per_tile = datums_per_tile(num_faces, face_r_dim)
    if tile_count is None:
        tile_count = max(1, flat.numel() // per_tile)

    packed: List[int] = []
    for tile in range(tile_count):
        chunk = flat[tile * per_tile : (tile + 1) * per_tile]
        if chunk.numel() == 0:
            break
        tile_bytes = _call_accepted(
            packer,
            chunk,
            num_faces=num_faces,
            face_r_dim=face_r_dim,
            use_srcs=use_srcs,
            dest_acc=dest_acc,
        )
        packed.extend(tile_bytes)
    return packed


#: L1 formats this module can actually move bytes for. Derived from
#: :data:`PACKERS` because the two directions cover exactly the same formats --
#: there is no format with a packer and no unpacker, or the reverse -- so one
#: gate is enough for both. Deliberately separate from an architecture's
#: ``SUPPORTED_L1_FORMATS``, which says what the *hardware* can hold: a format
#: can be perfectly real on the device and still have no codec here.
MODELLED_L1_FORMATS: FrozenSet[DataFormat] = frozenset(PACKERS)


def _no_codec_message(l1_format: DataFormat) -> str:
    return (
        f"{l1_format} has no L1 codec in this golden, so its bytes cannot be "
        f"written or read here. This is a gap in the model, not a statement "
        f"about the hardware. Modelled formats: "
        f"{sorted(str(f) for f in MODELLED_L1_FORMATS)}."
    )


def unpack_from_l1(
    packed: Union[Sequence[int], bytes],
    l1_format: DataFormat,
    *,
    tile_count: Optional[int] = None,
    tile_stride_bytes: Optional[int] = None,
    num_faces: int = MAX_NUM_FACES,
    face_r_dim: int = MAX_FACE_R_DIM,
    use_srcs: bool = False,
    dest_acc: bool = False,
    twos_complement: bool = False,
) -> torch.Tensor:
    """Read `packed` L1 bytes back as values, exactly as the unpacker sees them.

    `tile_stride_bytes` defaults to what :func:`pack_to_l1` actually writes at
    this geometry, via :func:`tile_bytes_for` -- which accounts for the BFP
    exponent minimum and the ``use_srcs`` slice layout, neither of which a
    datum count captures. Left to its own devices ``unpack_res_tiles`` assumes a
    full 32x32 tile stride for backward compatibility, which is only correct
    when the geometry really is 32x32; pass the device's stride explicitly when
    reading a buffer laid out some other way.

    `tile_count` defaults to however many whole tiles the buffer holds.
    """
    if l1_format not in MODELLED_L1_FORMATS:
        # unpack_res_tiles would reach a bare dict lookup and raise KeyError
        # with nothing but the format name in it.
        raise ValueError(_no_codec_message(l1_format))
    packed = list(packed)
    if tile_stride_bytes is None:
        tile_stride_bytes = tile_bytes_for(
            l1_format, num_faces, face_r_dim, use_srcs, dest_acc
        )
    if tile_count is None:
        tile_count = max(1, len(packed) // tile_stride_bytes)
    return unpack_res_tiles(
        packed,
        l1_format,
        tile_count=tile_count,
        tile_stride_bytes=tile_stride_bytes,
        num_faces=num_faces,
        face_r_dim=face_r_dim,
        use_srcs=use_srcs,
        dest_acc=dest_acc,
        twos_complement=twos_complement,
    )
