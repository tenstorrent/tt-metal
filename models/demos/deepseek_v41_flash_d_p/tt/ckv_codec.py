# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU codec of the DeepSeek-V4.1 compressed-KV row formats (de-risk L0; the oracle for the L1 device tests).

Row = 512 values (head_dim; RoPE already applied to the last 64). Both formats are 288 bytes per row.

Format A (reference-exact, model.py fp4_act_quant(block 16, e4m3 scale) = reference/kernels.py:fp4_quant_dequant):
    bytes [0, 256)   e2m1 codes, value 2i in the LOW nibble of byte i, value 2i + 1 in the high nibble
                     code = sign << 3 | g, g indexes the grid {0, .5, 1, 1.5, 2, 3, 4, 6} (= the e2m1 bit pattern)
    bytes [256, 288) e4m3fn scale of group j (values 16 j .. 16 j + 15), s = round_e4m3(max(amax, 6 2^-9) / 6)
    value = grid[g] (-1)^sign * s   (exact in bf16: <= 2 + 3 significant bits)

Format B (row-packed bfloat4_b fragments, tt-metal's host packer rule, tt_metal/impl/data_format/blockfloat_common.cpp
convert_u32_to_bfp / get_max_exp, is_exp_a = false):
    bytes [0, 32)    shared exponent of group j = max fp32 biased exponent of its 16 values (0 for all zero / denormal)
    bytes [32, 288)  sign-magnitude nibbles (sign << 3 | m, m in 0..7), value 2i low nibble of byte 32 + i
    m = RNE((1.f 2^23) >> (E - e), 21 bits), clamped to 7 (no exponent bump); value = (-1)^s m / 4 2^(E - 127)
  A 32-row band of B rows is a bfloat4_b TILE per 32 columns (``rows_to_tiles``): for tile c the row's groups 2c
  (faces 0 / 2) and 2c + 1 (faces 1 / 3): 2 exponent bytes + 2 x 8 mantissa bytes per row, no re-rounding.
"""

from __future__ import annotations

import torch

ROW = 512
GROUP = 16
NGROUPS = ROW // GROUP
ROW_BYTES = 288
FP4_MAX = 6.0
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
_MIDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)


def _pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """[n, 512] int (0..15) -> [n, 256] uint8, element 2i in the low nibble."""
    c = codes.to(torch.int32).view(codes.shape[0], -1, 2)
    return (c[..., 0] | (c[..., 1] << 4)).to(torch.uint8)


def _unpack_nibbles(b: torch.Tensor) -> torch.Tensor:
    """[n, 256] uint8 -> [n, 512] int64 codes."""
    b = b.to(torch.int64)
    return torch.stack([b & 15, b >> 4], dim=-1).flatten(-2)


# ------------------------------------------------------------------------------------------------ format A
def round_e4m3(v: torch.Tensor) -> torch.Tensor:
    return v.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()


def encode_a(x: torch.Tensor) -> torch.Tensor:
    """fp32 rows [n, 512] -> [n, 288] uint8 (format A)."""
    xg = x.float().reshape(-1, NGROUPS, GROUP)
    amax = xg.abs().amax(dim=-1, keepdim=True)
    s = round_e4m3(amax.clamp_min(FP4_MAX * 2.0**-9) / FP4_MAX)
    v = (xg / s).clamp(-FP4_MAX, FP4_MAX)
    a = v.abs()
    idx = torch.bucketize(a, _MIDS)
    tie = (idx < len(_MIDS)) & (a == _MIDS[idx.clamp(max=len(_MIDS) - 1)])
    idx = torch.where(tie & (idx % 2 == 1), idx + 1, idx)
    sign = torch.signbit(v).to(torch.int64)  # copysign keeps -0 (code 8): same value
    codes = (sign << 3) | idx
    sb = s.squeeze(-1).to(torch.float8_e4m3fn).view(torch.uint8)
    return torch.cat([_pack_nibbles(codes.reshape(x.shape[0], ROW)), sb], dim=-1)


def decode_a(b: torch.Tensor) -> torch.Tensor:
    """[n, 288] uint8 -> fp32 rows [n, 512]."""
    codes = _unpack_nibbles(b[:, :256])
    s = b[:, 256:].contiguous().view(torch.float8_e4m3fn).float()  # [n, 32]
    mag = E2M1[codes & 7]
    val = torch.where((codes >> 3) == 1, -mag, mag)
    return (val.view(-1, NGROUPS, GROUP) * s.unsqueeze(-1)).reshape(-1, ROW)


# ------------------------------------------------------------------------------------------------ format B
def encode_b(x: torch.Tensor, truncate: bool = False) -> torch.Tensor:
    """fp32 rows [n, 512] -> [n, 288] uint8 (format B; tt-metal's host bfloat4_b rule, or truncation)."""
    u = x.float().contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    exp = (u >> 23) & 0xFF
    sign = u >> 31
    man = (u & 0x7FFFFF) | (1 << 23)
    eg = exp.view(-1, NGROUPS, GROUP)
    shared = eg.amax(dim=-1, keepdim=True)
    diff = (shared - eg).clamp(max=62)
    m = man.view(-1, NGROUPS, GROUP) >> diff
    if truncate:
        q = m >> 21
    else:
        rv = m & ((1 << 21) - 1)
        q = m >> 21
        up = (rv > (1 << 20)) | ((rv == (1 << 20)) & ((q & 1) == 1))
        q = q + up.to(torch.int64)
    q = q.clamp(max=7)
    q = torch.where(eg == 0, torch.zeros_like(q), q)
    sg = torch.where(q == 0, torch.zeros_like(q), sign.view(-1, NGROUPS, GROUP))
    codes = (sg << 3) | q
    return torch.cat([shared.squeeze(-1).to(torch.uint8), _pack_nibbles(codes.reshape(-1, ROW))], dim=-1)


def decode_b(b: torch.Tensor) -> torch.Tensor:
    """[n, 288] uint8 -> fp32 rows [n, 512]."""
    e = b[:, :32].to(torch.int64)
    codes = _unpack_nibbles(b[:, 32:]).view(-1, NGROUPS, GROUP)
    mag = (codes & 7).float() * torch.pow(2.0, (e - 127 - 2).float()).unsqueeze(-1)
    return torch.where((codes >> 3) == 1, -mag, mag).reshape(-1, ROW)


def rows_to_tiles(rows: torch.Tensor) -> torch.Tensor:
    """32 format-B rows [32, 288] uint8 -> 16 bfloat4_b tiles [16, 576] uint8 in tt-metal's tile byte layout
    (64 exponent bytes: face 0 rows 0..15, face 1, face 2, face 3; then 4 faces x 16 rows x 8 mantissa bytes).
    This is the byte shuffle the gather kernel does in L1 (format B, read side)."""
    assert rows.shape == (32, ROW_BYTES)
    out = torch.zeros(16, 576, dtype=torch.uint8)
    for c in range(16):
        for r in range(32):
            fr = r % 16
            for half in range(2):  # face column
                face = (r // 16) * 2 + half
                g = 2 * c + half
                out[c, face * 16 + fr] = rows[r, g]
                m0 = 64 + face * 128 + fr * 8
                out[c, m0 : m0 + 8] = rows[r, 32 + 8 * g : 32 + 8 * g + 8]
    return out


def tiles_to_rows(tiles: torch.Tensor) -> torch.Tensor:
    """Inverse of rows_to_tiles: [16, 576] -> [32, 288] (format B, write side: packer tiles -> row records)."""
    rows = torch.zeros(32, ROW_BYTES, dtype=torch.uint8)
    for c in range(16):
        for r in range(32):
            fr = r % 16
            for half in range(2):
                face = (r // 16) * 2 + half
                g = 2 * c + half
                rows[r, g] = tiles[c, face * 16 + fr]
                m0 = 64 + face * 128 + fr * 8
                rows[r, 32 + 8 * g : 32 + 8 * g + 8] = tiles[c, m0 : m0 + 8]
    return rows
