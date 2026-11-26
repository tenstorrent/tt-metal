# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Plain interleaved copy of a bank-streamed decode weight, for the chunked prefill (data movement only).

The decode keeps its projection weights as BankLinear pages (tt/fused/bank_linear.py): the same tiles as the weight
cache's plain [K, N] tensor, byte for byte, in another page order. PfPages(device).plain(bank, cols) rebuilds
column ranges [(c0, c1), ...] of the plain tensor, side by side, as a fresh [1, 1, K, sum c1 - c0] TILE DRAM tensor: one program copies each of its
pages from its page of bank.weight. No value is touched, so the result is the cache tensor's tiles. The caller
deallocates it after use.
"""

import torch

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F

NB = 16  # pages per batch (the reader fills one batch while the writer drains the other)


class PfPages:
    def __init__(self, device):
        self.device = device
        self._idx = {}  # (id(bank), c0, c1) -> (bank, uint32 index tensor, K, N)

    def index(self, bank, cols):
        """The device index tensor of column ranges cols (multiples of 32), built once per (bank, cols). Build them
        all before a decode capture: a tensor allocated after it can sit in DRAM the decode replays write."""
        key = (id(bank), tuple(cols))
        e = self._idx.get(key)
        if e is None or e[0] is not bank:
            assert all(c % 32 == 0 for r in cols for c in r), cols
            pm = bank.page_map()
            m = torch.cat([pm[:, c0 // 32 : c1 // 32] for c0, c1 in cols], dim=1).reshape(-1)
            assert (m >= 0).all(), "columns not stored in the bank weight"
            n = m.numel()
            pad = torch.zeros(-(-n // 256) * 256, dtype=torch.int64)
            pad[:n] = m
            t = ttnn.from_torch(pad.to(torch.int32).view(1, 1, -1, 256), dtype=ttnn.uint32,
                                layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
            e = self._idx[key] = (bank, t, bank.K, sum(c1 - c0 for c0, c1 in cols))
        return e

    def plain(self, bank, cols):
        _, idx, K, N = self.index(bank, cols)
        src = bank.weight
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, K, N]), src.dtype, ttnn.TILE_LAYOUT, self.device,
                                             ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        pages = K // 32 * (N // 32)
        grid = self.device.compute_with_storage_grid_size()
        n_cores = min(grid.x * grid.y, -(-pages // NB))
        cores, crs = F.core_list(self.device, n_cores)
        r_args, w_args = {}, {}
        for i, core in enumerate(cores):
            a, b = pages * i // n_cores, pages * (i + 1) // n_cores
            r_args[core] = [src.buffer_address(), idx.buffer_address(), a, b]
            w_args[core] = [out.buffer_address(), a, b]
        pb = ttnn.Tile([32, 32]).get_tile_size(src.dtype)
        named = dict(pb=pb, nb=NB)
        ks = [F.kernel("pf_pages_reader.cpp", ttnn.ReaderConfigDescriptor(), named, r_args, crs),
              F.kernel("pf_pages_writer.cpp", ttnn.WriterConfigDescriptor(), named, w_args, crs)]  # fmt: skip
        cbs = [F.cb(0, src.dtype, 2 * NB, pb, cores=crs), F.cb(1, ttnn.bfloat16, 1, 1024, cores=crs)]
        return F.run([src, idx, out], ks, cbs)
