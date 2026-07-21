# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Mesh-distributed KDA layer (Option B: SP × TP). Phase 8 distribution.
#
# TP (axis 1) shards the heads: projections are column-parallel (q/k/v/gate output-sharded) and the
# output projection is row-parallel + a TP all-reduce. The delta-rule recurrence, gate, short-conv and
# gated-RMSNorm are all per-head-local, so they run UNCHANGED on each chip's head-shard (reused from
# ttnn_kda_ops). SP (axis 0) sequence sharding + the cross-span state-scan are added in a later step.
#
# Collectives mirror tt/mla/mla.py (reduce_scatter_minimal_async + all_gather_async) and the CCL setup
# in tests/sparse_mla/test_sparse_mla_ccl_perf.py. See ../DISTRIBUTION.md, ../ROOFLINE.md.

from __future__ import annotations

import torch
import ttnn
from einops import rearrange

from models.demos.deepseek_v3_d_p.tt.tt_ccl import create_global_semaphores

from .ttnn_kda_chunk import chunk_kda_ttnn
from .ttnn_kda_ops import (
    causal_conv1d_silu_ttnn,
    kda_gate_ttnn,
    l2norm_ttnn,
    recurrent_kda_ttnn,
)

_CHUNK = 64  # chunked prefill when T % _CHUNK == 0, else token-recurrent

_MM = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)


def make_tp_semaphores(mesh_device):
    """Semaphore sets over the full compute grid: all_gather (2), reduce_scatter (3), barrier (1)."""
    g = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})
    ag = create_global_semaphores(mesh_device, cores, 0)                                  # 2
    rs = [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(3)]          # 3
    bar = ttnn.create_global_semaphore(mesh_device, cores, 0)
    return ag, rs, bar


class TtKimiDeltaAttentionMesh:
    """SP×TP mesh-distributed KDA. Weights sourced from a torch KimiDeltaAttentionRef state_dict.

    mesh_device.shape = (sp_size, tp_size); axis 0 = SP (sequence), axis 1 = TP (heads).
    This step distributes over **TP only** (SP=1 supported; SP>1 state-scan is the next step).
    """

    def __init__(self, ref, mesh_device, sp_axis: int = 0, tp_axis: int = 1, num_links: int = 1):
        self.md = mesh_device
        self.sp_axis, self.tp_axis = sp_axis, tp_axis
        self.TP = mesh_device.shape[tp_axis]
        self.SP = mesh_device.shape[sp_axis]
        assert ref.num_heads % self.TP == 0, f"num_heads {ref.num_heads} not divisible by TP {self.TP}"
        # SP handling: this step distributes over TP (head-shard); the sequence is REPLICATED over the
        # SP axis (redundant data-parallel). Real SP sequence-sharding + cross-span state-scan is the
        # next step (see ../ROOFLINE.md §state-scan). Replication keeps SP rows identical -> correct.
        self.sp_replicated = True
        self.H = ref.num_heads
        self.HV = ref.num_v_heads
        self.Hloc = self.HV // self.TP
        self.K = ref.head_k_dim
        self.V = ref.head_v_dim
        self.conv_size = ref.conv_size
        self.use_short_conv = ref.use_short_conv
        self.allow_neg_eigval = ref.allow_neg_eigval
        self.lower_bound = ref.lower_bound
        self.norm_eps = ref.norm_eps
        self.num_links = num_links
        self.topology = ttnn.Topology.Linear
        self.sem_ag, self.sem_rs, self.sem_bar = make_tp_semaphores(mesh_device)
        sd = ref.state_dict()
        mshape = tuple(mesh_device.shape)

        def shard(dim):
            dims = [None, None]
            dims[tp_axis] = dim
            return ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mshape, dims=dims)

        repl = ttnn.ReplicateTensorToMesh(mesh_device)

        def up(t, mapper, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                t.contiguous().to(torch.float32), dtype=ttnn.float32, layout=layout, device=mesh_device, mesh_mapper=mapper
            )

        # column-parallel in-projections: shard the OUT dim (weight stored [in,out] -> dim 1)
        self.w_q = up(sd["q_proj.weight"].T, shard(1))
        self.w_k = up(sd["k_proj.weight"].T, shard(1))
        self.w_v = up(sd["v_proj.weight"].T, shard(1))
        self.w_f0 = up(sd["f_proj.0.weight"].T, repl)  # bottleneck, replicated
        self.w_f1 = up(sd["f_proj.1.weight"].T, shard(1))
        self.w_b = up(sd["b_proj.weight"].T, shard(1))
        self.w_g0 = up(sd["g_proj.0.weight"].T, repl)
        self.w_g1 = up(sd["g_proj.1.weight"].T, shard(1))
        self.b_g1 = up(sd["g_proj.1.bias"].reshape(1, 1, -1), shard(2))
        # row-parallel out-projection: shard the IN dim (weight [in,out] -> dim 0)
        self.w_o = up(sd["o_proj.weight"].T, shard(0))
        # gate params: per value-head -> shard head dim
        self.A_log = up(sd["A_log"].reshape(self.HV, 1), shard(0))
        self.dt_bias = up(sd["dt_bias"].reshape(self.HV, self.K), shard(0))
        self.o_norm_w = up(sd["o_norm_weight"].reshape(1, 1, 1, self.V), repl)
        # depthwise conv taps: per-channel -> shard channel dim
        if self.use_short_conv:
            self.taps = {
                nm: [up(sd[nm][:, k].reshape(1, 1, -1), shard(2), layout=ttnn.TILE_LAYOUT) for k in range(self.conv_size)]
                for nm in ("q_conv", "k_conv", "v_conv")
            }

    def _lin(self, x, w):
        return ttnn.linear(x, w, compute_kernel_config=_MM)

    def _tp_all_reduce(self, x):
        """Sum per-TP-chip o_proj partials across the TP axis: reduce_scatter + all_gather (production
        form — moves ~(p-1)/p of the data per phase, no wasteful [.,.,TP,.] intermediate)."""
        if self.TP == 1:
            return x
        d = len(x.shape) - 1  # scatter/gather the hidden (last) dim; hidden % TP must be tile-aligned
        x = ttnn.experimental.reduce_scatter_minimal_async(
            x, persistent_output_buffers=None, dim=d,
            multi_device_global_semaphore=self.sem_rs, barrier_semaphore=self.sem_bar,
            num_links=self.num_links, memory_config=ttnn.DRAM_MEMORY_CONFIG, topology=self.topology, cluster_axis=self.tp_axis,
        )
        x = ttnn.experimental.all_gather_async(
            x, dim=d, multi_device_global_semaphore=self.sem_ag, barrier_semaphore=self.sem_bar,
            num_links=self.num_links, memory_config=ttnn.DRAM_MEMORY_CONFIG, topology=self.topology, cluster_axis=self.tp_axis,
        )
        return x

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        # input hidden replicated across all chips
        x = ttnn.from_torch(
            hidden_states.to(torch.float32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
            device=self.md, mesh_mapper=ttnn.ReplicateTensorToMesh(self.md),
        )
        q, k, v = self._lin(x, self.w_q), self._lin(x, self.w_k), self._lin(x, self.w_v)  # [B,T,Hloc*K] per chip
        if self.use_short_conv:
            q = causal_conv1d_silu_ttnn(q, self.taps["q_conv"], self.conv_size, self.md)
            k = causal_conv1d_silu_ttnn(k, self.taps["k_conv"], self.conv_size, self.md)
            v = causal_conv1d_silu_ttnn(v, self.taps["v_conv"], self.conv_size, self.md)
        else:
            q, k, v = ttnn.silu(q), ttnn.silu(k), ttnn.silu(v)

        g = self._lin(self._lin(x, self.w_f0), self.w_f1)
        beta = ttnn.sigmoid(self._lin(x, self.w_b))

        q = ttnn.reshape(q, [B, T, self.Hloc, self.K])
        k = ttnn.reshape(k, [B, T, self.Hloc, self.K])
        v = ttnn.reshape(v, [B, T, self.Hloc, self.V])
        g = ttnn.reshape(g, [B, T, self.Hloc, self.K])

        q, k = l2norm_ttnn(q), l2norm_ttnn(k)
        if self.allow_neg_eigval:
            beta = ttnn.multiply(beta, 2.0)
        g = kda_gate_ttnn(g, self.A_log, self.dt_bias, self.lower_bound)

        if T % _CHUNK == 0:
            o, _ = chunk_kda_ttnn(q, k, v, g, beta, device=self.md, chunk_size=_CHUNK)  # prefill, per-chip local heads
        else:
            o, _ = recurrent_kda_ttnn(q, k, v, g, beta, device=self.md)

        gate = ttnn.reshape(ttnn.add(self._lin(self._lin(x, self.w_g0), self.w_g1), self.b_g1), [B, T, self.Hloc, self.V])
        o_norm = ttnn.rms_norm(o, epsilon=self.norm_eps, weight=self.o_norm_w)
        o = ttnn.multiply(o_norm, ttnn.sigmoid(gate))
        o = ttnn.reshape(o, [B, T, self.Hloc * self.V])
        o = self._lin(o, self.w_o)        # row-parallel partial [B,T,hidden]
        o = self._tp_all_reduce(o)        # sum partials across TP
        return ttnn.to_torch(o, mesh_composer=ttnn.ConcatMeshToTensor(self.md, dim=0))[:B]
