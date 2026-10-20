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

from models.experimental.hpcg.host import native

TILE = 1024


def word_of_flat(k):
    """Word index inside a 32 x 32 fp32 tile page for row-major position k (mod 1024) of the [rows, 32] tensor."""
    k = np.asarray(k) % TILE
    rr, c = k // 32, k % 32
    return ((rr // 16) * 2 + (c // 16)) * 256 + (rr % 16) * 16 + (c % 16)


_WORD_OF_FLAT = word_of_flat(np.arange(TILE))


def to_device(device, data):
    """q-ordered numpy vector (length a multiple of 1024) -> fp32 TILE tensor [P / 32, 32] on device."""
    data = np.ascontiguousarray(data, np.float32).reshape(-1)
    if data.size % TILE:
        raise ValueError(f"vector length {data.size} is not a multiple of {TILE}")
    lib = native.lib()
    if lib is not None:  # the same permutation in one parallel pass (numpy's fancy indexing ran at about 1 GB/s)
        a = native.zeros(data.size, np.float32)
        lib.hpcg_tile_permute(data, a, data.size)
        a = a.reshape(-1, 32)
    else:
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


def rows_to_device(device, a):
    """Data read only by the custom kernels (matrix values, transfer weights, column offsets) as ROW_MAJOR
    [N / 1024, 1024] with 4-byte words: each 4 KB DRAM page then holds words q in plain order, which is what the
    kernels read, so the host permutes nothing. int32 tables go up bit for bit under an fp32 label: the kernels only
    read their bytes, and an fp32 upload is a plain copy while TT-NN's uint32 upload converts element by element."""
    a = np.ascontiguousarray(a)
    if a.dtype == np.int32:
        a = a.view(np.float32)
    a = np.ascontiguousarray(a, np.float32).reshape(-1, TILE)
    return ttnn.from_torch(torch.from_numpy(a), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def zeros_device(device, P):
    """A zero vector of length P, zero-filled by the device: ttnn.zeros builds the zeros on the host and uploads them
    (9.5 ms for 28 MB on the P150), while allocating and filling on the device takes about 2 ms."""
    t = ttnn.empty(ttnn.Shape([P // 32, 32]), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    return ttnn.zeros_like(t)
