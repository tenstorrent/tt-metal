# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host <-> device layout of HPCG vectors and index tables.

A level vector of length P lives on device as a [P / 32, 32] fp32 TILE tensor. The device kernels address it in
q order: q = page * 1024 + word, the order of the 4-byte words inside the DRAM pages (reference/problem.py). A
TILE page stores its 32 x 32 values face by face, so the host permutes each 1024-value tile once on upload and
once on download; the kernels then never deal with faces.
"""

import numpy as np
import torch
import ttnn

TILE = 1024


def word_of_flat(k):
    """Word index inside a 32 x 32 fp32 tile page for row-major position k (mod 1024) of the [rows, 32] tensor."""
    k = np.asarray(k) % TILE
    rr, c = k // 32, k % 32
    return ((rr // 16) * 2 + (c // 16)) * 256 + (rr % 16) * 16 + (c % 16)


_WORD_OF_FLAT = word_of_flat(np.arange(TILE))


def to_device(device, data):
    """q-ordered numpy vector (length a multiple of 1024) -> fp32 TILE tensor [P / 32, 32] on device."""
    data = np.asarray(data, np.float32)
    if data.size % TILE:
        raise ValueError(f"vector length {data.size} is not a multiple of {TILE}")
    a = data.reshape(-1, TILE)[:, _WORD_OF_FLAT].reshape(-1, 32)
    return ttnn.from_torch(torch.from_numpy(a), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)


def from_device(tensor):
    """fp32 TILE tensor -> q-ordered float64 numpy vector."""
    a = ttnn.to_torch(tensor).float().numpy().reshape(-1).astype(np.float64)
    out = np.empty_like(a)
    out.reshape(-1, TILE)[:, _WORD_OF_FLAT] = a.reshape(-1, TILE)
    return out


def u32_rows(device, a):
    """Integer table -> uint32 ROW_MAJOR tensor (index tables and per-core metadata)."""
    return ttnn.from_torch(
        torch.from_numpy(np.ascontiguousarray(a, np.int32)),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
