# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4-Preview dense MLP (layer 0) and shared expert, TP8 over the mesh columns, on the DSV4 ``DSV41SharedExpertColTP`` (reused unchanged).

dense layer 0: SwiGLU 6144 -> 18432 -> 6144 (no clamp: the oracle's conv.clamp_dense=False), 2304 per column = 72 tiles, bfp8.
shared expert: 6144 -> 2048 -> 6144 with the swiglu_limit 10 clamp (oracle), 256 per column = 8 tiles, bfp8.
``forward`` returns this column's PARTIAL [1,1,T,6144]; the MoE block folds the shared partial into the routed partial before its
reduce_scatter, the dense layer sums the 8 column partials with ``allreduce_cols`` (reduce_scatter over the hidden dim + all_gather).
"""
import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.shared_expert_tp import DSV41SharedExpertColTP
from models.demos.blackhole.hy4_preview.tt.hy4_pf_opt import PfOpt

NO_CLAMP = 1.0e30


def _apply_pf(m, pf, rows=512):
    """HY4_PF_OPT mm2d (+ hifi2): explicit 2D-mcast program configs for the T=``rows`` prefill linears of a DSV41SharedExpertColTP (stock auto configs ran at 21% of the HiFi2 ideal)."""
    if not pf.mm2d:
        return m
    from models.demos.blackhole.hy4_preview.tt.hy4_pf_opt import make_pc

    m.pc1 = make_pc(rows, m.dim, 2 * m.ic)
    m.pc2 = make_pc(rows, m.ic, m.dim)
    return m


class SkinnyDecodeShared:
    """HY4_SKINNY=shared: the decode shared-expert partial on the bank-local streaming kernel (tt/hy4_skinny.SkinnyShared, T <= 16 rows); same forward contract as DSV41SharedExpertColTP."""

    def __init__(self, mesh_device, w0, w1, w2, limit):
        from models.demos.blackhole.hy4_preview.tt.hy4_skinny import SkinnyShared

        self.sk = SkinnyShared(mesh_device, w0, w1, w2, limit=limit)

    def forward(self, h):
        return self.sk(h, int(h.shape[2]))


def make_shared(mesh_device, w0, w1, w2, limit=10.0, decode=True, pf_opt=None, **kw):
    """w0/w1 [1,1,6144,2048], w2 [1,1,2048,6144] host ([in, out]). decode=True: 1D-mcast program configs for T <= 32 rows.
    HY4_SKINNY=shared (default OFF): decode=True returns the bank-local skinny kernel (T <= 16 rows, no extra kwargs).
    """
    if decode and not kw and pf_opt is None:
        from models.demos.blackhole.hy4_preview.tt.hy4_skinny import enabled

        if enabled("shared"):
            return SkinnyDecodeShared(mesh_device, w0, w1, w2, limit)
    base = dict(
        g1=(12, 8), g2=(12, 8), pn1=1, pn2=4, bw1=32, bw2=4
    )  # ict = 8 tiles per column: bw2 must divide 8; pn2*cores covers 192 out tiles
    base.update(kw)
    pf_opt = pf_opt if pf_opt is not None else PfOpt.from_env()  # HY4_PF_OPT, default OFF
    if pf_opt.hifi2 and not decode:
        base.setdefault("fid", "HiFi2")
    m = DSV41SharedExpertColTP(mesh_device, w0, w1, w2, limit=limit, cfg=decode, **base)
    return m if decode else _apply_pf(m, pf_opt)


def make_dense(mesh_device, w0, w1, w2, decode=True, pf_opt=None, **kw):
    """Dense layer 0: w0/w1 [1,1,6144,18432], w2 [1,1,18432,6144] host. 72 inter tiles per column: gate|up = 144 output tiles (pn1=2 -> 72 cores), down 192 out tiles (pn2=4 -> 48 cores)."""
    base = dict(g1=(12, 8), g2=(12, 8), pn1=2, pn2=4, bw1=32, bw2=8)
    base.update(kw)
    pf_opt = pf_opt if pf_opt is not None else PfOpt.from_env()  # HY4_PF_OPT, default OFF
    if pf_opt.hifi2 and not decode:
        base.setdefault("fid", "HiFi2")
    m = DSV41SharedExpertColTP(mesh_device, w0, w1, w2, limit=NO_CLAMP, cfg=decode, **base)
    return m if decode else _apply_pf(m, pf_opt)


def allreduce_cols(part, num_links=1, topology=None):
    """Sum the 8 column partials [1,1,T,D] -> replicated [1,1,T,D] (reduce_scatter over dim 3 on cluster_axis 1, then all_gather)."""
    topology = topology or ttnn.Topology.Ring
    rs = ttnn.reduce_scatter(
        part, dim=3, cluster_axis=1, num_links=num_links, topology=topology, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = ttnn.all_gather(
        rs, dim=3, cluster_axis=1, num_links=num_links, topology=topology, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn.deallocate(rs)
    return out
