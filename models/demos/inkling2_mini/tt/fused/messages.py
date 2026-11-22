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


class FusedMessages:
    """Drop-in for model._messages (MessageAttnRes.all_messages + the split): the loop messages of one pass from
    the previous pass's keys, in 2 programs. (1) rms(h12_in) -> row 0 of a private stack (attn-res seal, 1 core).
    (2) 12 cores, core l = loop layer l: the attn-res kernels with k = 1 (states rms(h12_in), rms(h17_out)) and
    nq = 4 queries (q, k, v, mlp of layer l, pre-scaled by 1/sqrt(C)); its writer puts messages 0..2 into the
    layer's qkv row tensor and message 3 into its mlp row tensor, each with a dense twin (split3)."""

    def __init__(self, device, weights, name="msg_q", n_layers=12, c=2304):
        from models.demos.inkling2_mini.tt.fused.attnres import FusedRowOps

        w = weights[name]
        w = ttnn.to_torch(w) if isinstance(w, ttnn.Tensor) else w
        q = w.float().reshape(n_layers, 32, -1)[:, :4] * c**-0.5  # [12, 4, C], as MessageAttnRes
        self.device, self.n_layers, self.c = device, n_layers, c
        self.rows = FusedRowOps(device, c)
        self.q = [F.dense_tensor(device, q[l]) for l in range(n_layers)]
        self.out = F.OutCache(device)
        self.bufs = [
            (self.out.get_twin(("qkv", l), 3, c, ttnn.float32), self.out.get_twin(("mlp", l), 1, c, ttnn.float32))
            for l in range(n_layers)
        ]

    def __call__(self, h12_in, h17_out):
        """h12_in, h17_out: [1,1,1,C] rows (the previous pass's input of L12 and output of L17).
        Returns [(qkv [1,1,3,C], mlp [1,1,1,C])] per loop layer (cached: valid until the next call)."""
        from models.demos.inkling2_mini.tt.fused.attnres import f32_bits

        r = self.rows
        r.run(h12_in, 0)  # rms(h12_in) -> stack row 0
        F.check_row_input(h17_out)
        T, ntc, nq, k = r.T, self.c // 32, 4, 1
        cores, crs = F.core_list(self.device, self.n_layers)
        named = dict(ntc=ntc, pages=T, nq=nq, k=k, has_weight=0, out_rows=3, split3=1)
        rd, wr = {}, {}
        for l, core in enumerate(cores):
            (qr, qd), (mr, md) = self.bufs[l]
            rd[core] = [*F.input_args(h17_out), self.q[l].buffer_address(), r.stack.buffer_address(),
                        r.scaler.buffer_address(), 0]  # fmt: skip
            wr[core] = [r.stack.buffer_address(), qr.buffer_address(), qd.buffer_address(), mr.buffer_address(),
                        md.buffer_address()]  # fmt: skip
        cnamed = dict(pages=T, nq=nq, k=k, has_weight=0, inv_c=f32_bits(1.0 / self.c), eps=f32_bits(r.eps))
        kernels = [
            F.kernel("attnres_reader.cpp", ttnn.ReaderConfigDescriptor(), named, rd, crs),
            F.kernel("attnres_writer.cpp", ttnn.WriterConfigDescriptor(), named, wr, crs),
            F.kernel("attnres_compute.cpp", F.compute_config([], full_sync=True), cnamed, None, crs),
        ]
        f32, d = ttnn.float32, k + 1
        cbs = [
            F.cb(0, F.input_dtype(h17_out), T, cores=crs),
            F.cb(1, f32, nq * T, cores=crs),
            F.cb(2, f32, k * T, cores=crs),
            F.cb(3, f32, 1, cores=crs),
            F.cb(16, f32, T, cores=crs),
            F.cb(17, f32, nq * T, cores=crs),
            F.stage_cb(24, ntc, crs),
            F.cb(25, f32, 1, cores=crs),
            F.cb(26, f32, 1, cores=crs),
            F.cb(27, f32, nq * d, cores=crs),
            F.cb(28, f32, nq * d, cores=crs),
        ]
        tw = F.twin(h17_out)
        io = [h17_out] + ([tw[0]] if tw else []) + [r.scaler, *self.q, r.stack]
        io += [t for pair in self.bufs for (a, b) in pair for t in (a, b)]
        F.run(io, kernels, cbs)
        r.fresh.clear()  # the stack rows now hold rms(h12_in), rms(h17_out) of this call only
        return [(q, m) for (q, _), (m, _) in self.bufs]
