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
from models.experimental.hpcg.tt.constants import TILE


def word_of_flat(k):
    """Word index inside a 32 x 32 fp32 tile page for row-major position k (mod 1024) of the [rows, 32] tensor."""
    k = np.asarray(k) % TILE
    rr, c = k // 32, k % 32
    return ((rr // 16) * 2 + (c // 16)) * 256 + (rr % 16) * 16 + (c % 16)


_WORD_OF_FLAT = word_of_flat(np.arange(TILE))


def allocate(device, shape, layout):
    """A new fp32 DRAM tensor, not filled: ttnn.allocate_tensor_on_device, 3.0 us per call on the P150's host
    against 6.3 to 7.5 for ttnn.empty, whose Python wrapper costs the difference (a setup allocates about 40)."""
    return ttnn.allocate_tensor_on_device(ttnn.TensorSpec(tuple(shape), ttnn.float32, layout), device)


def upload_f32(device, a, layout=ttnn.ROW_MAJOR_LAYOUT):
    """A C-contiguous float32 numpy array -> a device tensor of its bytes: the ttnn.Tensor constructor that
    ttnn.from_torch ends in, without from_torch's Python argument checks (about 2 us per upload)."""
    return ttnn.Tensor(tensor=torch.from_numpy(a), data_type=ttnn.float32, device=device, layout=layout)


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
    """Integer table (index tables and per-core metadata) -> ROW_MAJOR device tensor of its 32-bit words. The words
    go up under an fp32 label, a plain copy (the kernels read raw bytes; TT-NN's uint32 upload converts element by
    element: 0.056 against 0.021 ms for a 130 x 1024 table). Values stay below 2^31 - 2^23, so no word is a NaN."""
    a = np.ascontiguousarray(a, np.int32)
    return upload_f32(device, a.view(np.float32))


def rows_to_device(device, a):
    """Data read only by the custom kernels (matrix values, transfer weights, column offsets) as ROW_MAJOR
    [N / 1024, 1024] with 4-byte words: each 4 KB DRAM page then holds words q in plain order, which is what the
    kernels read, so the host permutes nothing. int32 tables go up bit for bit under an fp32 label: the kernels only
    read their bytes, and an fp32 upload is a plain copy while TT-NN's uint32 upload converts element by element."""
    a = np.ascontiguousarray(a)
    if a.dtype == np.int32:
        a = a.view(np.float32)
    a = np.ascontiguousarray(a, np.float32).reshape(-1, TILE)
    return upload_f32(device, a)


def empty_device(device, P):
    """An fp32 TILE vector of length P, allocated on the device and not filled (its first user writes it)."""
    return allocate(device, (P // 32, 32), ttnn.TILE_LAYOUT)


def upload_zeros(device, P):
    """A small zero vector of length P as ROW_MAJOR [P / 1024, 1024] (the kernels read raw 4 KB pages, so the layout
    does not matter), uploaded from the host: no device program, so no kernel binary to load; and a plain copy, where a
    TILE upload tilizes on the host (0.6 ms the first time in a process)."""
    return upload_f32(device, np.zeros((P // TILE, TILE), np.float32))  # numpy: torch.zeros' first call costs ~0.5 ms


def empty_page(device):
    """One fp32 4 KB page on the device (ROW_MAJOR [1, 1024]), not filled: a CG scalar the device writes before it
    reads it. An allocation, not an upload."""
    return allocate(device, (1, TILE), ttnn.ROW_MAJOR_LAYOUT)


def scalar_device(device, value, only_first=False):
    """One 4 KB page holding value in every word (a CG scalar is word 0: element (0, 0) of a tile), uploaded from the
    host as ROW_MAJOR [1, 1024] (a plain copy). only_first: value in word 0 and zeros elsewhere (a per-core partial
    slot that DotFinishOp adds to exactly value)."""
    t = np.zeros((1, TILE), np.float32) if only_first else np.full((1, TILE), value, np.float32)
    t[0, 0] = value
    return upload_f32(device, t)
