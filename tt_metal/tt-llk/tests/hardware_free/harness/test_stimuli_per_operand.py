# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Per-operand geometry of ``StimuliConfig`` (tiny tile matmul with operands of different formats and face counts) and block float
packing of partial faces. Host-only: no toolchain, no device."""

from __future__ import annotations

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.pack import pack_bfp4_b, pack_bfp8_b
from helpers.stimuli_config import StimuliConfig
from helpers.unpack import unpack_bfp4_b, unpack_bfp8_b


def _cfg(**kw):
    return StimuliConfig(
        None,
        DataFormat.Bfp8_b,
        None,
        DataFormat.Bfp4_b,
        DataFormat.Float16_b,
        tile_count_A=2,
        tile_count_B=3,
        **kw,
    )


def test_default_geometry_is_unchanged():
    cfg = _cfg()
    assert not cfg.per_operand_geometry
    assert cfg.operand_unpack_words is None
    assert (cfg.num_faces_A, cfg.num_faces_B, cfg.face_r_dim_A, cfg.face_r_dim_B) == (
        4,
        4,
        16,
        16,
    )
    assert cfg.tile_size_A_bytes == 1088 and cfg.tile_size_B_bytes == 576


def test_per_operand_tile_sizes_and_unpacker_words():
    cfg = _cfg(
        num_faces_A=2,
        face_r_dim_A=8,
        tile_dimensions_A=[8, 32],
        num_faces_B=2,
        tile_dimensions_B=[32, 16],
    )
    assert cfg.per_operand_geometry
    # Bfp8_b 8x32: 16 exponents + 256 mantissas; Bfp4_b 32x16: 32 exponents + 256 mantissa bytes
    assert cfg.tile_size_A_bytes == 16 + 256
    assert cfg.tile_size_B_bytes == 32 + 256
    assert cfg.operand_unpack_words == (
        cfg.tile_size_B_bytes // 16,
        cfg.tile_size_A_bytes // 16,
    )
    assert cfg.buf_b_addr == cfg.buf_a_addr + 2 * cfg.tile_size_A_bytes
    assert cfg.buf_res_addr == cfg.buf_b_addr + 3 * cfg.tile_size_B_bytes


@pytest.mark.parametrize(
    "pack,unpack,sizes",
    [(pack_bfp8_b, unpack_bfp8_b, 1), (pack_bfp4_b, unpack_bfp4_b, 0.5)],
)
@pytest.mark.parametrize(
    "num_faces,face_r_dim", [(1, 16), (2, 16), (2, 8), (2, 4), (2, 2), (2, 1), (1, 4)]
)
def test_partial_face_block_float_roundtrip(pack, unpack, sizes, num_faces, face_r_dim):
    torch.manual_seed(7)
    tile = torch.rand(1024, dtype=torch.bfloat16)
    n = num_faces * face_r_dim * 16
    packed = pack(tile, num_faces=num_faces, face_r_dim=face_r_dim)
    assert len(packed) == max(n // 16, 16) + int(n * sizes)
    got = unpack(bytes(packed), num_faces=num_faces, face_r_dim=face_r_dim)
    # the blocks are rows of 16, so a partial tile equals the head of the full tile's roundtrip
    ref = unpack(bytes(pack(tile)))[:n]
    assert torch.equal(got[:n], ref)


@pytest.mark.parametrize(
    "fmt,pack", [(DataFormat.Bfp8_b, pack_bfp8_b), (DataFormat.Bfp4_b, pack_bfp4_b)]
)
@pytest.mark.parametrize(
    "num_faces,face_r_dim", [(2, 16), (2, 8), (2, 1), (1, 4), (1, 16)]
)
def test_unpack_res_tiles_partial_block_float(fmt, pack, num_faces, face_r_dim):
    """Result tiles of a partial block float output sit at the full tile stride; each reads back its exponents, padding and mantissas."""
    from helpers.llk_params import format_tile_sizes
    from helpers.unpack import unpack_res_tiles

    torch.manual_seed(3)
    n = num_faces * face_r_dim * 16
    stride = format_tile_sizes[fmt]
    tiles = [torch.rand(1024, dtype=torch.bfloat16) for _ in range(3)]
    raw = bytearray(3 * stride)
    for i, t in enumerate(tiles):
        p = bytes(pack(t, num_faces=num_faces, face_r_dim=face_r_dim))
        raw[i * stride : i * stride + len(p)] = p
    got = unpack_res_tiles(bytes(raw), fmt, 3, False, num_faces, face_r_dim)
    assert len(got) == 3 * n
    for i, t in enumerate(tiles):
        ref = (unpack_bfp8_b if fmt == DataFormat.Bfp8_b else unpack_bfp4_b)(
            bytes(pack(t))
        )[:n]
        assert torch.equal(torch.as_tensor(got[i * n : (i + 1) * n]), ref)
