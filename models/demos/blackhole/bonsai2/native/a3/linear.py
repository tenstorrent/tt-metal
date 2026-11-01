# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One exact-ternary projection for single-chip decode: y = W x with W from the PQ2_0 GGUF.

forward(x): x [.., 1(32), K] bf16 32x32 tiles, row 0 live -> y [.., 1(32), N] 32x32 tiles in
`out_mem`, row 0 live, rest zero. N is padded on the weight side to a whole number of output
tiles per A3 worker; the padding columns are zero and never written.
"""

from __future__ import annotations

import os

import numpy as np

import ttnn

from . import op as a3
from .mlp import Shared


class TernaryLinear:
    def __init__(self, shared: Shared, name: str, cache_dir=None, out_mem=ttnn.L1_MEMORY_CONFIG):
        from models.demos.blackhole.bonsai2.native.gguf_ternary import read_ternary

        self.sh = shared
        self.out_mem = out_mem
        t = read_ternary(shared.gguf, name)
        n, k = t.codes.shape
        self.n, self.k = n, k
        assert k in shared.m, f"no Hadamard tensor for width {k}"
        per = 32 * len(shared.workers)  # output columns per round over all workers
        n_pad = -(-n // per) * per
        codes = np.zeros((k, n_pad), np.float32)
        codes[:, :n] = t.codes.T
        scales = np.zeros((k // 128, n_pad), np.float32)
        scales[:, :n] = t.scales.T
        cache = os.path.join(cache_dir, name) if cache_dir else None
        self.w, self.s = a3.weights_for_device(shared.device, codes, scales, cache)

    def forward(self, x, out_mem=None):
        sh = self.sh
        x_in = x
        if x.is_sharded() or x.memory_config().buffer_type != ttnn.BufferType.L1:
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        x8 = a3.masked_input_buffer(sh.device, sh.workers, self.k)
        a3.prep(x, sh.m[self.k], x8, sh.workers, src_tile32=True)
        out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(x.shape)[:-1] + [self.n]), ttnn.bfloat16,
                                             ttnn.TILE_LAYOUT, sh.device, out_mem or self.out_mem)
        a3.ternary_linear(x8, self.w, self.s, out, sh.workers, cache=False, n_real=self.n)
        ttnn.deallocate(x8)
        if x is not x_in:
            ttnn.deallocate(x)
        return out
