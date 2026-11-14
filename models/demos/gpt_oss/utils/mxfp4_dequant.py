# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fast exact dequantization of the gpt-oss MXFP4 expert tensors.

The checkpoint stores each expert matrix as `_blocks` [E, N, G, 16] uint8 (two E2M1 nibbles per byte, low nibble
first: 32 values per block along K = 32 G) and `_scales` [E, N, G] uint8 (E8M0, 2^(s - 127) per block).
transformers' convert_moe_packed_tensors returns the [E, K, N] tensor (transposed) in BF16; on gpt-oss-120b it takes
~6.5 s per layer's gate_up_proj and ~3 s per down_proj (~330 M values/s: chunked torch indexing plus a transpose
copy). dequant_mxfp4 returns the same tensor bit for bit in one parallel numba pass that writes the transposed layout
directly: every output value is a lookup in a 256 x 16 table (scale byte x nibble) computed once with the very torch
ops of convert_moe_packed_tensors. Without numba (or for another dtype) it calls convert_moe_packed_tensors.
"""

import numpy as np
import torch

# E2M1 nibble -> value (bit 3 = sign), as transformers.integrations.mxfp4.FP4_VALUES
FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

_STATE = {}


def _table():
    """[256, 16] uint16: BF16 bits of FP4_VALUES[nib] * 2^(s - 127), computed as convert_moe_packed_tensors does
    (BF16 lookup, then torch.ldexp in place on the BF16 tensor)."""
    if "table" not in _STATE:
        sub = torch.tensor(FP4_VALUES, dtype=torch.bfloat16).reshape(1, 16).repeat(256, 1)
        exp = (torch.arange(256, dtype=torch.int32) - 127).reshape(256, 1)
        torch.ldexp(sub, exp, out=sub)
        _STATE["table"] = sub.view(torch.int16).numpy().view(np.uint16).copy()
    return _STATE["table"]


def _kernel():
    if "kernel" in _STATE:
        return _STATE["kernel"]
    try:
        import numba
    except ImportError:
        _STATE["kernel"] = None
        return None

    @numba.njit(parallel=True, cache=False)
    def kernel(blocks, scales, table, out):
        # blocks [E, N, G, 16] u8, scales [E, N, G] u8, table [256, 16] u16, out [E, 32 G, N] u16.
        # One work item per (expert, block index g): its 32 output rows are written along N.
        E, N, G, _ = blocks.shape
        for item in numba.prange(E * G):
            e = item // G
            g = item % G
            k0 = g * 32
            for n in range(N):
                row = table[scales[e, n, g]]
                for j in range(16):
                    b = blocks[e, n, g, j]
                    out[e, k0 + 2 * j, n] = row[b & 0xF]
                    out[e, k0 + 2 * j + 1, n] = row[b >> 4]

    _STATE["kernel"] = kernel
    return kernel


def dequant_mxfp4(blocks: torch.Tensor, scales: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    """blocks [E, N, G, 16] uint8, scales [E, N, G] uint8 -> [E, 32 G, N] in dtype, as convert_moe_packed_tensors."""
    kernel = _kernel() if (dtype == torch.bfloat16 and blocks.dim() == 4) else None
    if kernel is None:
        from transformers.integrations.mxfp4 import convert_moe_packed_tensors

        return convert_moe_packed_tensors(blocks, scales, dtype=dtype)
    E, N, G, B = blocks.shape
    assert B == 16 and tuple(scales.shape) == (E, N, G), (tuple(blocks.shape), tuple(scales.shape))
    out = np.empty((E, 32 * G, N), dtype=np.uint16)
    kernel(blocks.contiguous().numpy(), scales.contiguous().numpy(), _table(), out)
    return torch.from_numpy(out.view(np.int16)).view(torch.bfloat16)
