# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Prefill block attn-res over T token rows: drop-in for prefill_model._AttnRes (same ctor, same call).

Per token: softmax_d(q . rms(S_d) / sqrt(C)) . rms(S_d) over the depth stack, nq queries. The rms of the partial
state and the D1 logits matmuls ([T, C] x [C, 32]) stay ttnn ops; one program does the rest, which the op chain ran
as ~2 D1 nq + 3 D1 full passes over [T, C] fp32 tensors: the depth softmax, the weight broadcast and the weighted
sums. Same SFPU ops in the same order as the chain (max, subtract, exp, add, reciprocal, multiply), so the outputs
are bit-equal to it.

Work unit: (tile row R of the T rows, a group of column tiles); units are spread over the compute grid.
"""


import ttnn
from models.demos.inkling2_mini.tt.fused import common as F

HIDDEN = 2304
EPS = 1e-6
COL_GROUPS = 3  # column tiles per unit = 72 / 3


class PfAttnRes:
    # Logits tables of sealed states: id(normed) -> (normed, [T, 32 M] fp32 table). A sealed state never changes and
    # every module's queries are fixed, so its logits for all M modules come from one matmul when it is sealed
    # (seal()); each call then only computes the logits of its partial state. Cleared per chunk (clear()).
    TABLES = {}

    @staticmethod
    def clear():
        PfAttnRes.TABLES.clear()

    @staticmethod
    def seal(normed, q_all, ckc):
        """Logits of a sealed normed state for every module (q_all: all_queries() on device)."""
        PfAttnRes.TABLES[id(normed)] = (
            normed,
            ttnn.matmul(normed, q_all, dtype=ttnn.float32, compute_kernel_config=ckc),
        )

    @staticmethod
    def all_queries(modules):
        """Index the modules (module i reads columns [32 i, 32 i + 32) of a table); [1, 1, C, 32 M] fp32 queries."""
        for i, m in enumerate(modules):
            m.index, m.n_modules = i, len(modules)
        return ttnn.concat([m.qT for m in modules], dim=-1)

    def __init__(self, dev, q, nq, ckc):
        """q: host [.., 32, C] (row j = query j); nq queries used; ckc: compute config of the norm / logits ops."""
        q = (ttnn.to_torch(q) if isinstance(q, ttnn.Tensor) else q).float().reshape(-1, HIDDEN)[:32].clone()
        q[nq:] = 0
        self.nq = nq
        qT = (q * HIDDEN**-0.5).T.contiguous().view(1, 1, HIDDEN, 32)  # 1/sqrt(C) folded
        self.qT = ttnn.from_torch(qT, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        self.ckc = ckc
        self.dev = dev
        self.index = None  # set by all_queries

    def __call__(self, normed, partial):
        """normed: list of rms-normed sealed states, fp32 [1,1,T,C] DRAM. Returns nq fp32 [1,1,T,C] outputs."""
        p32 = partial if partial.dtype == ttnn.float32 else ttnn.typecast(partial, ttnn.float32)
        ns = list(normed) + [ttnn.rms_norm(p32, epsilon=EPS, compute_kernel_config=self.ckc)]
        lg, stride, off, own = [], [], [], []
        for n in ns:
            e = PfAttnRes.TABLES.get(id(n)) if self.index is not None else None
            if e is not None and e[0] is n:  # sealed: this module's columns of the state's table
                lg.append(e[1]), stride.append(self.n_modules), off.append(self.index)
            else:
                lg.append(ttnn.matmul(n, self.qT, dtype=ttnn.float32, compute_kernel_config=self.ckc))
                stride.append(1), off.append(0), own.append(lg[-1])
        for t in ns + lg:
            assert t.dtype == ttnn.float32 and t.layout == ttnn.TILE_LAYOUT, (t.dtype, t.layout)
            assert t.memory_config().buffer_type == ttnn.BufferType.DRAM and not t.is_sharded(), t.memory_config()
        t_rows = ns[0].shape[-2]
        assert t_rows % 32 == 0 and ns[0].shape[-1] == HIDDEN, ns[0].shape
        d1, nq, wt = len(ns), self.nq, HIDDEN // 32
        outs = [
            ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, t_rows, HIDDEN]), ttnn.float32, ttnn.TILE_LAYOUT, self.dev,
                                           ttnn.DRAM_MEMORY_CONFIG)
            for _ in range(nq)
        ]  # fmt: skip
        g = wt // COL_GROUPS
        units = [(r, c * g, (c + 1) * g) for r in range(t_rows // 32) for c in range(COL_GROUPS)]
        grid = self.dev.compute_with_storage_grid_size()
        n_cores = min(len(units), grid.x * grid.y)
        cores, crs = F.core_list(self.dev, n_cores)
        per = [units[i::n_cores] for i in range(n_cores)]
        r_args, w_args, c_args = {}, {}, {}
        for c, us in zip(cores, per):
            flat = [v for unit in us for v in unit]
            r_args[c] = (
                [n.buffer_address() for n in ns] + [x.buffer_address() for x in lg] + stride + off + [len(us)] + flat
            )
            w_args[c] = [o.buffer_address() for o in outs] + [len(us)] + flat
            c_args[c] = [len(us)] + [j1 - j0 for _, j0, j1 in us]
        named = F.fixed_rt(dict(d1=d1, nq=nq, wt=wt), r_args, w_args, c_args)
        kernels = [
            F.kernel("pf_attnres_reader.cpp", ttnn.ReaderConfigDescriptor(), named, r_args, crs),
            F.kernel("pf_attnres_writer.cpp", ttnn.WriterConfigDescriptor(), named, w_args, crs),
            F.kernel("pf_attnres_compute.cpp", F.compute_config([0, 1, 3, 4, 5, 6, 7], full_sync=True), named, c_args,
                     crs),
        ]  # fmt: skip
        fp = ttnn.float32
        cbs = [F.cb(0, fp, d1, cores=crs), F.cb(1, fp, d1, cores=crs), F.cb(2, fp, d1, cores=crs),
               F.cb(3, fp, d1 * nq, cores=crs), F.cb(4, fp, d1 * nq, cores=crs), F.cb(5, fp, 2 * d1, cores=crs),
               F.cb(6, fp, 1, cores=crs), F.cb(7, fp, 1, cores=crs), F.cb(16, fp, 2 * nq, cores=crs)]  # fmt: skip
        F.run(ns + lg + outs, kernels, cbs)
        for x in own + [ns[-1]]:
            ttnn.deallocate(x)
        return outs
