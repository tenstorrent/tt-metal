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

NO_CLAMP = 1.0e30


def make_shared(mesh_device, w0, w1, w2, limit=10.0, decode=True, **kw):
    """w0/w1 [1,1,6144,2048], w2 [1,1,2048,6144] host ([in, out]). decode=True: 1D-mcast program configs for T <= 32 rows."""
    base = dict(
        g1=(12, 8), g2=(12, 8), pn1=1, pn2=4, bw1=32, bw2=4
    )  # ict = 8 tiles per column: bw2 must divide 8; pn2*cores covers 192 out tiles
    base.update(kw)
    return DSV41SharedExpertColTP(mesh_device, w0, w1, w2, limit=limit, cfg=decode, **base)


def make_dense(mesh_device, w0, w1, w2, decode=True, **kw):
    """Dense layer 0: w0/w1 [1,1,6144,18432], w2 [1,1,18432,6144] host. 72 inter tiles per column: gate|up = 144 output tiles (pn1=2 -> 72 cores), down 192 out tiles (pn2=4 -> 48 cores)."""
    base = dict(g1=(12, 8), g2=(12, 8), pn1=2, pn2=4, bw1=32, bw2=8)
    base.update(kw)
    return DSV41SharedExpertColTP(mesh_device, w0, w1, w2, limit=NO_CLAMP, cfg=decode, **base)


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
