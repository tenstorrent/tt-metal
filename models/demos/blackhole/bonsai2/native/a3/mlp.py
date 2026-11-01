# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Exact-ternary SwiGLU MLP for single-chip decode (BONSAI_NATIVE=pq2_0).

forward(x): x [1, 1, 32, 5120] bf16 32x32 tiles in L1, row 0 live (decode, one user).
  prep(x)        H(D x) at 5120 into the shared masked buffer
  A3 gate|up     one op over (gate, up) column pairs, writes silu(gate) * up as 1x32 tiles
  prep(h)        H(D h) at 17408
  A3 down        writes [1, 1, 32, 5120] 32x32 tiles, row 0 live, rest zero
Weights come straight from the PQ2_0 GGUF (codes + fp16 scales, no requantization).
Everything shared across layers (H tensors, masked buffers, h) lives in one Shared.
"""

from __future__ import annotations

import os

import numpy as np
import torch

import ttnn

from . import op as a3

GGUF = os.path.expanduser("~/avan/models/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf")
WPB = 4  # workers per DRAM bank


class Shared:
    """Per-device state common to every layer's MLP."""

    def __init__(self, device, gguf_path=GGUF):
        from models.demos.blackhole.bonsai2.native.gguf_ternary import read_gguf
        from models.demos.blackhole.bonsai2.native.hadamard import BlockHadamard

        self.device = device
        self.gguf = read_gguf(gguf_path)
        self.had = BlockHadamard.from_metadata(self.gguf.kv)
        self.workers = a3.worker_cores(device, WPB)
        self.m = {k: a3.hadamard_tensor(device, self.had, k) for k in (5120, 17408)}
        self.x8 = {k: a3.masked_input_buffer(device, self.workers, k) for k in (5120, 17408)}
        self.h = ttnn.from_torch(torch.zeros(1, 1, 1, 17408), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                 tile=ttnn.Tile([1, 32]), device=device, memory_config=ttnn.L1_MEMORY_CONFIG)


class TernaryMLP:
    def __init__(self, shared: Shared, layer: int, cache_dir=None):
        from models.demos.blackhole.bonsai2.native.gguf_ternary import read_ternary

        self.sh = shared
        dev = shared.device
        cache = (lambda name: os.path.join(cache_dir, f"blk{layer}.{name}")) if cache_dir else (lambda name: None)
        g = read_ternary(shared.gguf, f"blk.{layer}.ffn_gate.weight")
        u = read_ternary(shared.gguf, f"blk.{layer}.ffn_up.weight")
        d = read_ternary(shared.gguf, f"blk.{layer}.ffn_down.weight")
        # codes [N, K] -> [K, N]; scales [N, K/128] -> [K/128, N]; gate/up as column pairs
        gu_codes = a3.interleave_cols(g.codes.T.astype(np.float32), u.codes.T.astype(np.float32))
        gu_scales = a3.interleave_cols(g.scales.T.astype(np.float32), u.scales.T.astype(np.float32))
        self.w_gu, self.s_gu = a3.weights_for_device(dev, gu_codes, gu_scales, cache("gu"))
        self.w_d, self.s_d = a3.weights_for_device(dev, d.codes.T.astype(np.float32), d.scales.T.astype(np.float32),
                                                   cache("down"))

    def forward(self, x):
        sh = self.sh
        a3.prep(x, sh.m[5120], sh.x8[5120], sh.workers, src_tile32=True)
        a3.ternary_linear(sh.x8[5120], self.w_gu, self.s_gu, sh.h, sh.workers, gu=True)
        a3.prep(sh.h, sh.m[17408], sh.x8[17408], sh.workers, src_tile32=False)
        out = ttnn.allocate_tensor_on_device(x.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT,  # residual-stream logical shape
                                             sh.device, ttnn.L1_MEMORY_CONFIG)
        return a3.ternary_linear(sh.x8[17408], self.w_d, self.s_d, out, sh.workers, gu=False)


_shared = {}


def shared_for(device):
    """One Shared per device (H tensors and buffers are common to all layers)."""
    if id(device) not in _shared:
        _shared[id(device)] = Shared(device)
    return _shared[id(device)]
