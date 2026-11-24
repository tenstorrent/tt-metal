# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device backend of the capacity tiers (tt/hy4_chainstore.py ``Backend``). ttnn is imported lazily; every method raises on any error (the ChainStore then recomputes cold).

* copy_pages same row   : Hy4PageCopy.copy(row, [(src, dst)]) (row-local page-list copy, B45b kernel; source and destination ranges must be disjoint)
* copy_pages cross row  : Hy4Xrow.copy_off (stage-in, all_gather over the row axis, stage-out on the destination row; different page offsets)
* to_host / from_host   : the same stage tensors: stage-in of a page chunk (every row), read the stage of the donor row's devices to host (uint32, dtype blind); restore = write the chunk into the stage of the
                          destination row (other rows zero, skipped by the control page) and stage-out. Payload = list of (tensor name, chunk index, torch uint32 [cols, cp*units, page_words]).
STATUS: [NC] none of this ran on a device (hy4cap had no device by order); the E1 microbenchmark (tools/pcie_bw.py) measures the host path bandwidth, demo scenarios tailpark / hostspill test the data.
"""
from __future__ import annotations

import os

from models.demos.blackhole.hy4_preview.tt import hy4_xrow as X
from models.demos.blackhole.hy4_preview.tt.hy4_chainstore import Backend


class DeviceBackend(Backend):
    def __init__(self, md, tensors, xrow=None, bcopy=None):
        self.md, self.tensors, self.xrow, self.bcopy = md, tensors, xrow, bcopy
        self.supports_cross_row = xrow is not None
        self.rows, self.cols = int(md.shape[0]), int(md.shape[1])
        self._dcache = {}
        self.h2d_mode = os.environ.get("HY4_HT_H2D", "full")

    def describe(self):
        return f"row-local={'bcopy' if self.bcopy is not None else 'none'} cross-row={'xrow' if self.xrow else 'none'} host={'stage' if self.xrow else 'none'}"

    def can_copy(self, src_row, dst_row, npages):
        """cross-row copy = stage-in + all_gather over the rows (a fresh DRAM tensor of rows x stage bytes per chip, freed behind the stage-out) + stage-out. hy4fix: port1 got two TT_FATAL out-of-memory here (bank 4.03 GB, 4.2 MB free,
        23 MB buffer): require the gathered buffer per bank (x2 for interleaved rounding, + HY4_CAP_HEADROOM_MB margin) to fit in the largest contiguous free block of a bank. Same-row copies allocate nothing.
        """
        if src_row == dst_row or self.xrow is None:
            return True, ""
        try:
            import ttnn

            v = ttnn.get_memory_view(self.md, ttnn.BufferType.DRAM)
            free = int(v.largest_contiguous_bytes_free_per_bank)
            x = self.xrow
            stage_b = max(x.cp_kv * X.KV_UNITS * X.KV_PAGE_B, x.cp_key * X.KEY_UNITS * x.key_page)
            banks = max(1, int(self.md.dram_grid_size().x))
            need = 2 * (self.rows * stage_b // banks) + int(float(os.environ.get("HY4_CAP_HEADROOM_MB", "4")) * 1e6)
            self.last_headroom = (free, need)
            if free < need:
                return False, f"largest free block {free / 1e6:.1f} MB per bank < {need / 1e6:.1f} MB needed"
        except Exception as e:  # noqa: BLE001  (the check must never be the failure)
            return True, f"headroom check unavailable: {type(e).__name__}"
        return True, ""

    def copy_pages(self, src_row, src_page0, dst_row, dst_page0, npages):
        if src_row == dst_row:
            if self.bcopy is None:
                raise RuntimeError("no row-local page copy built")
            return self.bcopy.copy(src_row, [(src_page0 + i, dst_page0 + i) for i in range(npages)])
        if self.xrow is None:
            raise RuntimeError("no cross-row copy built (HY4_XROW=1)")
        return self.xrow.copy_off(src_row, src_page0, dst_row, dst_page0, npages)

    # ---- host tier (stage tensors of Hy4Xrow)
    def _stage_of(self, name):
        x = self.xrow
        kind = x.kind[name]
        pb = X.KV_PAGE_B if kind == "kv" else x.key_page
        u, cp = (X.KV_UNITS, x.cp_kv) if kind == "kv" else (X.KEY_UNITS, x.cp_key)
        return x.stage[(kind, pb)], kind, pb, u, cp

    def _row_devs(self, tensor, row):
        """the ``cols`` single-device views of mesh tensor ``tensor`` that sit on mesh row ``row`` (row-major device order of get_device_tensors, verified by the device round-trip test tests/dev_hosttier_rt.py)"""
        import ttnn

        d = self._dcache.get(id(tensor))
        if d is None:
            d = ttnn.get_device_tensors(tensor)
            self._dcache[id(tensor)] = d
        return d[row * self.cols : (row + 1) * self.cols]

    def tensor_groups(self, npages, slice_mb):
        """number of tensor groups (= slices) a spill of npages is cut into: per-chip bytes of the chain / slice_mb (0 = one slice, no grouping)"""
        if slice_mb <= 0 or self.xrow is None:
            return 0
        per_chip = npages * 3_750_912
        return int(max(1, min(len(self.xrow.tensors), -(-per_chip // int(slice_mb * 1e6)))))

    def to_host(self, row, page0, npages, group=None):
        """read ONLY the donor row's devices (hy4fix E1: ttnn.to_torch over the whole 4x8 mesh moved 4x the bytes at 0.06 GB/s useful; per-device reads reach ~0.75-0.9 GB/s per row); no per-op device sync (the read is ordered behind the stage-in on CQ0), the cols reads are enqueued non-blocking and drained once."""
        import ttnn

        if self.xrow is None:
            raise RuntimeError("host tier needs the xrow stage tensors (HY4_XROW=1)")
        import torch

        x, out = self.xrow, []
        tl = x.tensors
        if group is not None:
            g, G = group
            tl = tl[g * len(tl) // G : (g + 1) * len(tl) // G]
        for name, t in tl:
            stage, kind, pb, u, cp = self._stage_of(name)
            for ci, (k, n) in enumerate(X.chunk_plan(npages, cp)):
                x._prog(t, stage, x.ctl_in, n * u, (page0 + k) * u, 0, pb)
                hs = [ttnn.from_device(d, blocking=False) for d in self._row_devs(stage, row)]
                ttnn.synchronize_device(self.md)
                blk = torch.stack([ttnn.to_torch(h).reshape(-1, pb // 4) for h in hs])  # [cols, cp*u, pb/4]
                out.append(
                    (name, ci, n, blk[:, : n * u].clone())
                )  # trim the padding of a partial chunk (a 17-page slice used to store 128 pages: port2 host RSS 549 GB)
        return out

    def from_host(self, payload, row, page0, npages):
        """write the chunks into the stage of the destination row's devices only (other rows keep stale stage data: their control page says skip) and stage-out."""
        import torch

        import ttnn

        x = self.xrow
        if x is None:
            raise RuntimeError("host tier needs the xrow stage tensors (HY4_XROW=1)")
        ctl = x._ctl(0, row)
        try:
            for name, ci, n, blk in payload:
                t = dict(x.tensors)[name]
                stage, kind, pb, u, cp = self._stage_of(name)
                k = ci * cp
                if k >= npages:
                    continue  # the promoted hit is shorter than the parked chain
                mk = lambda b_: ttnn.from_torch(
                    b_.reshape(1, 1, -1, pb // 4).to(torch.int32).contiguous(),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                if blk.shape[1] < cp * u:  # trimmed payload: pad back to the stage shape (rows past n*u are never read)
                    blk = blk.view(torch.int32) if blk.dtype != torch.int32 else blk
                    blk = torch.cat(
                        [blk, torch.zeros((blk.shape[0], cp * u - blk.shape[1], blk.shape[2]), dtype=torch.int32)],
                        dim=1,
                    )
                mine = [mk(b) for b in blk]
                if (
                    self.h2d_mode == "sparse"
                ):  # shards only for the destination row (if the runtime accepts a sparse host mesh tensor)
                    shards = [mine[c] if r == row else None for r in range(self.rows) for c in range(self.cols)]
                    host = ttnn.from_host_shards([s_ for s_ in shards if s_ is not None], ttnn.MeshShape(1, self.cols))
                else:  # full: other rows get the SAME host shard (never used: their control page says skip)
                    shards = [mine[c] if r == row else mine[0] for r in range(self.rows) for c in range(self.cols)]
                    host = ttnn.from_host_shards(shards, ttnn.MeshShape(self.rows, self.cols))
                ttnn.copy_host_to_device_tensor(host, stage)
                n = min(n, npages - k)
                x._prog(stage, t, ctl, n * u, 0, (page0 + k) * u, pb)
            ttnn.synchronize_device(self.md)
        finally:
            ttnn.deallocate(ctl)
