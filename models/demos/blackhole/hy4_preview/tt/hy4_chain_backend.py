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

from models.demos.blackhole.hy4_preview.tt import hy4_xrow as X
from models.demos.blackhole.hy4_preview.tt.hy4_chainstore import Backend


class DeviceBackend(Backend):
    def __init__(self, md, tensors, xrow=None, bcopy=None):
        self.md, self.tensors, self.xrow, self.bcopy = md, tensors, xrow, bcopy
        self.supports_cross_row = xrow is not None
        self.rows, self.cols = int(md.shape[0]), int(md.shape[1])

    def describe(self):
        return f"row-local={'bcopy' if self.bcopy is not None else 'none'} cross-row={'xrow' if self.xrow else 'none'} host={'stage' if self.xrow else 'none'}"

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

    def to_host(self, row, page0, npages):
        import ttnn

        if self.xrow is None:
            raise RuntimeError("host tier needs the xrow stage tensors (HY4_XROW=1)")
        x, out = self.xrow, []
        comp = ttnn.ConcatMesh2dToTensor(self.md, dims=(0, 1), mesh_shape=tuple(self.md.shape))
        for name, t in x.tensors:
            stage, kind, pb, u, cp = self._stage_of(name)
            for ci, (k, n) in enumerate(X.chunk_plan(npages, cp)):
                x._prog(t, stage, x.ctl_in, n * u, (page0 + k) * u, 0, pb)
                ttnn.synchronize_device(self.md)
                full = ttnn.to_torch(stage, mesh_composer=comp)  # [rows, cols, cp*u, pb/4]
                out.append((name, ci, n, full[row].clone()))
        return out

    def from_host(self, payload, row, page0, npages):
        import torch

        import ttnn

        x = self.xrow
        if x is None:
            raise RuntimeError("host tier needs the xrow stage tensors (HY4_XROW=1)")
        ctl = x._ctl(0, row)
        mp = ttnn.ShardTensor2dMesh(self.md, dims=(0, 1), mesh_shape=tuple(self.md.shape))
        try:
            for name, ci, n, blk in payload:
                t = dict(x.tensors)[name]
                stage, kind, pb, u, cp = self._stage_of(name)
                full = torch.zeros((self.rows,) + tuple(blk.shape), dtype=blk.dtype)
                full[row] = blk
                host = ttnn.from_torch(full, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mp)
                ttnn.copy_host_to_device_tensor(host, stage)
                k = ci * cp
                if k >= npages:
                    continue  # the promoted hit is shorter than the parked chain
                n = min(n, npages - k)
                x._prog(stage, t, ctl, n * u, 0, (page0 + k) * u, pb)
            ttnn.synchronize_device(self.md)
        finally:
            ttnn.deallocate(ctl)
