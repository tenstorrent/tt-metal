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
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
from helpers.format_config import DataFormat
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
from helpers.tile_constants import FACE_C_DIM
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


def datums_per_tile(num_faces: int = 4, face_r_dim: int = 16) -> int:
    """Datums one tile holds at this geometry."""
    return num_faces * face_r_dim * FACE_C_DIM


def pack_to_l1(
    tensor: torch.Tensor,
    l1_format: DataFormat,
    *,
    tile_count: Optional[int] = None,
    num_faces: int = 4,
    face_r_dim: int = 16,
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
        raise ValueError(f"No packer for {l1_format}")
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


def unpack_from_l1(
    packed: Union[Sequence[int], bytes],
    l1_format: DataFormat,
    *,
    tile_count: Optional[int] = None,
    tile_stride_bytes: Optional[int] = None,
    num_faces: int = 4,
    face_r_dim: int = 16,
    use_srcs: bool = False,
    dest_acc: bool = False,
    twos_complement: bool = False,
) -> torch.Tensor:
    """Read `packed` L1 bytes back as values, exactly as the unpacker sees them.

    `tile_stride_bytes` defaults to the **dense** size of one tile at this
    geometry, which is what :func:`pack_to_l1` writes. Left to its own devices
    ``unpack_res_tiles`` assumes a full 32x32 tile stride for backward
    compatibility, which is only correct when the geometry really is 32x32 —
    pass the device's stride explicitly when reading a buffer laid out that way.

    `tile_count` defaults to however many whole tiles the buffer holds.
    """
    packed = list(packed)
    if tile_stride_bytes is None:
        tile_stride_bytes = l1_format.num_bytes_per_tile(
            datums_per_tile(num_faces, face_r_dim)
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
