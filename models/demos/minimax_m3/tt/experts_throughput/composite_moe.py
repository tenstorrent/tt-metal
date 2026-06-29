# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Composite expert-parallel MoE for MiniMax-M3 — a SELF-OWNED alternative to the DeepSeek
dispatch/combine all-to-all, which is incorrect for M3's real (skewed) gate routing distribution
(random routing PCC 0.998, real routing 0.607; not capacity, not the routing math, not bf4 — the DS
all-to-all itself). This is the CompositeRoutedExpert philosophy applied one level up: own the
dispatch/combine, correctness-first.

Mechanism (no all-to-all):
  * experts are EP-sharded 4/device (expert g -> device g // experts_per_chip, linear device index);
  * tokens are AllGathered across the SP axis so every device sees the full chunk;
  * each device runs ITS local experts on all tokens, weighted by the per-expert routing weight (the
    token's gate weight if it routed to that expert, else 0);
  * the per-device partials are summed across the whole mesh (all-reduce) -> full MoE output; then
    reduce-scattered back to the SP seq-shard so the output matches the residual layout [1, S_local, H].

Validated vs the from-scratch M3 golden on REAL layer-3 weights at EP=32 (PCC 0.9998).
PERF: recompute-all-tokens-per-expert is wasteful (optimize later: gather-by-expert or the fused kernel);
this is the FUNCTIONAL bring-up path.
"""

import torch

import ttnn

from .activation import apply_swiglu


class CompositeEPMoE:
    """EP MoE via AllGather(tokens) -> per-device weighted experts -> all-reduce(+reduce-scatter).

    routed_expert_weights: list[dict] length E (global), each {gate_proj/up_proj/down_proj} in HF
    (out_features, in_features) layout. Experts are assigned expert g -> linear device g//experts_per_chip.
    """

    def __init__(
        self,
        mesh_device,
        ccl_manager,
        mesh_config,
        routed_expert_weights,
        emb_dim,
        hidden_dim,
        num_experts,
        weights_dtype=ttnn.bfloat8_b,
        swiglu_limit=7.0,
        alpha=1.702,
    ):
        from types import SimpleNamespace

        self.mesh_device = mesh_device
        self.ccl = ccl_manager
        self.mesh_config = mesh_config
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.E = num_experts
        self._sw = SimpleNamespace(swiglu_limit=swiglu_limit, alpha=alpha)
        rows, cols = tuple(mesh_device.shape)
        self.rows, self.cols, self.ndev = rows, cols, rows * cols
        self.epc = num_experts // self.ndev  # experts per chip
        assert num_experts % self.ndev == 0, "num_experts must be divisible by num_devices"

        # Per-slot sharded weights: slot e tensor on device i holds expert (i*epc + e). Built by stacking
        # the per-device expert for each slot, reshaping to (rows, cols, ...) and 2D-sharding dims=(0,1).
        def shard_slot(per_expert, slot, transpose):
            t = torch.stack([per_expert[i * self.epc + slot] for i in range(self.ndev)])  # [ndev,a,b]
            if transpose:
                t = t.transpose(1, 2).contiguous()  # HF (out,in) -> ttnn.linear (in,out)
            a, b = t.shape[1], t.shape[2]
            t = t.reshape(rows, cols, a, b)
            return ttnn.from_torch(
                t,
                device=mesh_device,
                dtype=weights_dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(0, 1)),
            )

        gp = [w["gate_proj"].float() for w in routed_expert_weights]
        up = [w["up_proj"].float() for w in routed_expert_weights]
        dn = [w["down_proj"].float() for w in routed_expert_weights]
        self.gate_s = [shard_slot(gp, e, True) for e in range(self.epc)]  # [emb, hid] per device
        self.up_s = [shard_slot(up, e, True) for e in range(self.epc)]
        self.down_s = [shard_slot(dn, e, True) for e in range(self.epc)]  # [hid, emb] per device
        # this device's global expert id for each local slot: device i (linear r*cols+c) -> i*epc + e
        gid = torch.arange(self.ndev).reshape(rows, cols, 1) * self.epc + torch.arange(self.epc).reshape(1, 1, -1)
        self.gid_s = ttnn.from_torch(
            gid.to(torch.int32).reshape(rows, cols, 1, self.epc),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(0, 1)),
        )

    def __call__(self, x3d, topk_indices, topk_weights):
        """x3d [1, S_local, H] (SP seq-shard, full H, TP-replicated); topk_indices/weights [1, S_local, K].
        Returns [1, S_local, H] SP seq-sharded (full H)."""
        sp_axis = self.mesh_config.sp_axis  # rows
        H, K = self.emb_dim, topk_indices.shape[-1]

        # AllGather tokens + routing across SP rows -> every device holds the full chunk.
        x_full = self.mesh_config.allgather(x3d, self.ccl, axis=sp_axis, dim=1)  # [1, S_tot, H]
        idx_full = self.mesh_config.allgather(topk_indices, self.ccl, axis=sp_axis, dim=1)  # [1, S_tot, K]
        w_full = self.mesh_config.allgather(topk_weights, self.ccl, axis=sp_axis, dim=1)
        S_tot = x_full.shape[1]

        # Build per-device routing weight vec on host (idx/w are identical on every device post-gather).
        idx_t = ttnn.to_torch(ttnn.get_device_tensors(idx_full)[0]).long().reshape(S_tot, K)
        w_t = ttnn.to_torch(ttnn.get_device_tensors(w_full)[0]).float().reshape(S_tot, K)
        vec = torch.zeros(self.E, S_tot)
        for j in range(K):
            vec[idx_t[:, j], torch.arange(S_tot)] = w_t[:, j]
        # shard per slot: device i gets vec rows [i*epc + e]
        vec_s = []
        for e in range(self.epc):
            v = torch.stack([vec[i * self.epc + e] for i in range(self.ndev)]).reshape(self.rows, self.cols, S_tot, 1)
            vec_s.append(
                ttnn.from_torch(
                    v.to(torch.bfloat16),
                    device=self.mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        self.mesh_device, mesh_shape=(self.rows, self.cols), dims=(0, 1)
                    ),
                )
            )

        # Per-device weighted expert sum (partial over THIS device's local experts).
        partial = None
        for e in range(self.epc):
            g_out = ttnn.linear(x_full, self.gate_s[e])
            u_out = ttnn.linear(x_full, self.up_s[e])
            act = apply_swiglu(g_out, u_out, self._sw)
            oe = ttnn.linear(act, self.down_s[e])  # [1, S_tot, H]
            oe = ttnn.mul(oe, vec_s[e])  # weight (broadcast over H)
            partial = oe if partial is None else ttnn.add(partial, oe, output_tensor=partial)

        # Combine = sum the per-device partials across the WHOLE mesh, then SP-shard the seq dim back.
        # v1 (correctness-first): host-sum the 32 partials + re-shard. Each device's partial is its local
        # experts' contribution; summing all 32 gives the full MoE output. TODO(perf): replace with a
        # device reduce (all-reduce over TP cols + reduce-scatter over SP rows) via the CCL rs semaphores.
        parts = ttnn.get_device_tensors(partial)
        out_full = sum(ttnn.to_torch(p).float().reshape(S_tot, H) for p in parts).reshape(1, S_tot, H)
        return ttnn.from_torch(
            out_full.to(torch.bfloat16),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=(self.rows, self.cols), dims=(1, None)),
        )  # [1, S_local, H] SP seq-shard, TP-replicated
