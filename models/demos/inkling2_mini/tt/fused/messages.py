# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Loop-pass message split (E10d): residual.MessageAttnRes.all_messages output [1, 1, 48, C] fp32 (row 4 l + i =
message i of loop layer l: q, k, v, mlp) -> per layer (qkv [1, 1, 3, C], mlp [1, 1, 1, C]) fp32 row tensors with
dense twins, in ONE program (72 cores, one output tile column each; data movement only, exact copies).

Replaces model._messages' 24 selector matmuls per pass (one program each, tf32-rounded copies) and lets the message
gates read the messages' dense twins (3 pages) instead of gathering each row (144 small reads)."""

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F


class FusedMessageSplit:
    def __init__(self, device, n_layers=12, c=2304):
        self.device, self.n_layers, self.c = device, n_layers, c
        self.out = F.OutCache(device)
        self.bufs = [
            (self.out.get_twin(("qkv", l), 3, c, ttnn.float32), self.out.get_twin(("mlp", l), 1, c, ttnn.float32))
            for l in range(n_layers)
        ]

    def __call__(self, msgs):
        """msgs: [1, 1, 4 * n_layers, C] fp32 TILE (DRAM). Returns [(qkv, mlp)] per loop layer (the cached
        row tensors: valid until the next call)."""
        F.check_row_input(msgs)
        assert msgs.dtype == ttnn.float32 and msgs.shape[-2] == 4 * self.n_layers, (msgs.dtype, msgs.shape)
        ntc = self.c // 32
        cores, crs = F.core_list(self.device, ntc)
        common = []
        for (q, qd), (m, md) in self.bufs:
            common += [q.buffer_address(), qd.buffer_address(), m.buffer_address(), md.buffer_address()]
        named = dict(ntc=ntc, pages=F.pages(self.c), n_layers=self.n_layers)
        rt = {core: [msgs.buffer_address(), j] for j, core in enumerate(cores)}
        k = F.kernel("msg_split.cpp", ttnn.ReaderConfigDescriptor(), named, rt, crs)
        k.common_runtime_args = common
        n_tr = -(-4 * self.n_layers // 32)
        cbs = [F.cb(0, ttnn.float32, n_tr, cores=crs)]
        io = [msgs] + [t for pair in self.bufs for (r, d) in pair for t in (r, d)]
        F.run(io, [k], cbs)
        return [(q, m) for (q, _), (m, _) in self.bufs]
