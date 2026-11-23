# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Prefill attention-sink logits in one program: drop-in for ttnn.permute(_cols(qgrs, c0, c0 + H), (0, 3, 2, 1)).

qgrs [1, 1, T, QC] bf16 TILE; out [1, H, T, 1] bf16 TILE: out[h, t] = qgrs[t, c0 + h]. Per tile row R the one qgrs
tile holding the H columns is multiplied by H 0/1 selection tiles (S_h: element (h, 0) = 1), which moves column h to
column 0 and zeros the rest: one bf16 term per element, so the output equals the permute's, padding included.
"""

import torch

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F


class PfSink:
    def __init__(self, device, heads=24):
        self.device, self.heads = device, heads
        sel = torch.zeros(heads, 32, 32)
        sel[torch.arange(heads), torch.arange(heads), 0] = 1.0
        self.sel = ttnn.from_torch(sel.view(1, 1, heads * 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=device)  # fmt: skip

    def __call__(self, qgrs, c0):
        t, qc = qgrs.shape[-2], qgrs.shape[-1]
        assert qgrs.dtype == ttnn.bfloat16 and qgrs.layout == ttnn.TILE_LAYOUT and t % 32 == 0, qgrs
        assert c0 % 32 == 0 and c0 + self.heads <= qc, (c0, qc)
        h = self.heads
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, h, t, 1]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device,
                                             ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        rows = t // 32
        grid = self.device.compute_with_storage_grid_size()
        n_cores = min(rows, grid.x * grid.y)
        cores, crs = F.core_list(self.device, n_cores)
        per = [list(range(rows))[i::n_cores] for i in range(n_cores)]
        r_args = {c: [qgrs.buffer_address(), self.sel.buffer_address(), len(rs)] + rs for c, rs in zip(cores, per)}
        w_args = {c: [out.buffer_address(), len(rs)] + rs for c, rs in zip(cores, per)}
        c_args = {c: [len(rs)] for c, rs in zip(cores, per)}
        named = dict(h=h, qt=-(-qc // 32), col=c0 // 32, rt=rows)
        named = F.fixed_rt(named, r_args, w_args, c_args)
        kernels = [
            F.kernel("pf_sink_reader.cpp", ttnn.ReaderConfigDescriptor(), named, r_args, crs),
            F.kernel("pf_sink_writer.cpp", ttnn.WriterConfigDescriptor(), named, w_args, crs),
            F.kernel("pf_sink_compute.cpp", F.compute_config([]), named, c_args, crs),
        ]
        bf = ttnn.bfloat16
        cbs = [F.cb(0, bf, 2, cores=crs), F.cb(1, bf, h, cores=crs), F.cb(16, bf, 4, cores=crs)]
        return F.run([qgrs, self.sel, out], kernels, cbs)
