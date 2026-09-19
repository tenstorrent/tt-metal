# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure-TTNN NemotronH expert bank for `nemotron_h_experts` (`mixer.experts`,
the un-gated MLP expert collection inside `NemotronHMoE` -- distinct from the
already-graduated `nemotron_h_mo_e` sibling, which wraps gate+experts+shared
expert as one component; this one is just the raw expert compute).

HF reference (`NemotronHExperts.forward`, modeling_nemotron_h.py):

    forward(hidden_states[T,H], top_k_index[T,K] int64, top_k_weights[T,K] fp32):
        for each expert e that has >=1 token routed to it:
            x = hidden_states[tokens routed to e]
            y = down_proj[e]( relu(up_proj[e](x)) ** 2 )   # relu2, no bias
            y *= top_k_weights[those tokens, their slot]
            scatter-add y back into the per-token output

That sparse gather/scatter is mathematically identical to the DENSE form used
here (and by the graduated `nemotron_h_mo_e` sibling): build a dense
(tokens, num_experts) routing-weight matrix from (top_k_index, top_k_weights)
(zero for experts a token didn't select), evaluate every expert on every
token, and weight-sum. `top_k_index`/`top_k_weights` never reach this stub as
device tensors (the PCC harness only converts the PRIMARY arg -- here
`hidden_states` -- via `ttnn.from_torch`; the rest stay plain torch), so the
scatter that builds the dense routing matrix runs on host, and only the
resulting (tokens, num_experts) matrix is uploaded.

Tensor-parallel (TP=2): EXPERT-parallel, mirroring `nemotron_h_mo_e` --
`up_proj`/`down_proj` are already native 3D (num_experts, ...) tensors, so
they shard directly on the expert axis (dim 0); each chip evaluates its local
E/TP experts against its own routing-matrix columns and the partial mixture
is all_reduced to recover the full sum.
"""
from __future__ import annotations

import torch

import ttnn


class TtNemotronHExperts:
    def __init__(self, device, torch_module) -> None:
        self.device = device

        self.num_experts = int(torch_module.num_experts)
        self.hidden_dim = int(torch_module.hidden_dim)
        self.intermediate_dim = int(torch_module.intermediate_dim)
        E = self.num_experts

        # nn.Linear-style weights (out, in); store pre-transposed (in, out) so
        # forward can do `x @ W` directly, matching F.linear(x, W) = x @ W.T.
        up_t = torch_module.up_proj.detach().float().transpose(-1, -2).contiguous()  # (E, hidden, inter)
        down_t = torch_module.down_proj.detach().float().transpose(-1, -2).contiguous()  # (E, inter, hidden)

        # ---- tensor-parallel (expert-parallel) config ----------------------
        import os as _os

        dev = self.device
        try:
            _is_mesh = isinstance(dev, ttnn.MeshDevice)
        except AttributeError:
            _is_mesh = False
        _mesh_shape = list(dev.shape) if _is_mesh else [1, 1]
        _shard = bool(_os.environ.get("TT_HW_PLANNER_SHARD_RUN")) and _is_mesh
        TP = _mesh_shape[-1] if _shard else 1
        _shard = _shard and TP > 1 and (E % TP == 0)
        self._shard = _shard
        self._TP = TP
        self._tp_axis = len(_mesh_shape) - 1
        self._mesh_shape = _mesh_shape

        if _shard:
            Eloc = E // TP

            # Expert axis split on the HOST, one upload per LOCAL expert index:
            # chunk[d] is global expert d*Eloc+j, so ShardTensor2dMesh on dim 0
            # hands chip d exactly its own expert j. Same placement as slicing
            # the full stack on device, ~200x faster to build (measured
            # 2026-09-06: a per-expert device ttnn.slice out of the 1.3 GB tiled
            # stack costs seconds EACH).
            def _per_expert(stack, out_rows, out_cols):
                mats = []
                for j in range(Eloc):
                    chunk = torch.stack([stack[d * Eloc + j] for d in range(TP)], dim=0)
                    t = ttnn.from_torch(
                        chunk,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=dev,
                        mesh_mapper=ttnn.ShardTensor2dMesh(dev, mesh_shape=_mesh_shape, dims=(None, 0)),
                    )
                    mats.append(ttnn.reshape(t, [out_rows, out_cols]))
                return mats

            self._up = _per_expert(up_t, self.hidden_dim, self.intermediate_dim)
            self._down = _per_expert(down_t, self.intermediate_dim, self.hidden_dim)
            self._Eloc = Eloc
        else:
            self._up, self._down = [], []
            for e in range(E):
                self._up.append(self._devw(up_t[e]))
                self._down.append(self._devw(down_t[e]))
            self._Eloc = E

        # Per-chip expert selector for the DEVICE-SIDE routing path (see the
        # `routing_dense` kwarg on __call__). sel[d] is a one-hot (E, Eloc) that
        # projects a replicated full (tokens, E) routing matrix down to THIS
        # chip's contiguous expert window, sharded on dim 0 with the SAME
        # ShardTensor2dMesh call used for up_proj/down_proj so the columns line
        # up with the local expert weights. Identical to the pattern the
        # graduated `nemotron_h_mo_e` sibling already uses.
        self._sel = None
        if _shard:
            sel = torch.zeros(TP, E, self._Eloc)
            for d in range(TP):
                for j in range(self._Eloc):
                    sel[d, d * self._Eloc + j, j] = 1.0
            _sel_sh = ttnn.from_torch(
                sel,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                mesh_mapper=ttnn.ShardTensor2dMesh(dev, mesh_shape=_mesh_shape, dims=(None, 0)),
            )
            # squeeze to 2-D: a rank-3 rhs with batch 1 against a batch-B lhs is
            # a PARTIAL batch broadcast, which hangs ttnn.matmul (measured
            # 2026-09-06). A 2-D rhs broadcasts safely over any batch.
            self._sel = ttnn.reshape(_sel_sh, [E, self._Eloc])

        self.ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    # ------------------------------------------------------------------ #
    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    # ----------------------------- helpers ---------------------------- #
    def _is_mesh(self):
        try:
            if isinstance(self.device, ttnn.MeshDevice):
                return True
        except AttributeError:
            pass
        return hasattr(self.device, "get_device_ids") or hasattr(self.device, "get_devices")

    def _upload(self, torch_tensor, dtype, layout=ttnn.TILE_LAYOUT):
        if self._is_mesh():
            try:
                return ttnn.from_torch(
                    torch_tensor,
                    dtype=dtype,
                    layout=layout,
                    device=self.device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
                )
            except Exception:
                pass
        return ttnn.from_torch(torch_tensor, dtype=dtype, layout=layout, device=self.device)

    def _devw(self, torch_tensor, layout=ttnn.TILE_LAYOUT):
        return self._upload(torch_tensor.to(torch.bfloat16), ttnn.bfloat16, layout)

    def _dev(self, torch_tensor, layout=ttnn.TILE_LAYOUT):
        return self._upload(torch_tensor.float(), ttnn.float32, layout)

    def _fp32(self, t):
        if isinstance(t, ttnn.Tensor):
            if t.dtype != ttnn.float32:
                return ttnn.typecast(t, ttnn.float32)
            return t
        return self._dev(t.float())

    # ----------------------------- forward ---------------------------- #
    def __call__(self, hidden_states, top_k_index=None, top_k_weights=None, routing_dense=None, **kwargs):
        hs = self._fp32(hidden_states)
        if hs.layout != ttnn.TILE_LAYOUT:
            hs = ttnn.to_layout(hs, ttnn.TILE_LAYOUT)
        num_tokens = list(hs.shape)[0]
        E = self.num_experts

        # DEVICE-SIDE routing path: `routing_dense` is an already-on-device
        # (tokens, E) fp32 routing matrix produced by the chained pipeline's
        # on-device router. It replaces the host torch scatter_add_ below, which
        # would otherwise put host compute in the pipeline's hot path. The
        # expert loop, the expert-parallel weight split and the all_reduce are
        # unchanged either way.
        if routing_dense is not None:
            W_dev = routing_dense
            if W_dev.layout != ttnn.TILE_LAYOUT:
                W_dev = ttnn.to_layout(W_dev, ttnn.TILE_LAYOUT)
            if W_dev.dtype != ttnn.float32:
                W_dev = ttnn.typecast(W_dev, ttnn.float32)
            # self._sel is 2-D (E, Eloc), so this stays a (tokens, Eloc) matrix.
            W_sh = ttnn.matmul(W_dev, self._sel, compute_kernel_config=self.ckc) if self._shard else W_dev
            return self._mix(hs, W_sh, num_tokens)

        # Dense (tokens, E) routing-weight matrix, built on host from the
        # (index, weight) pairs -- top_k_index/top_k_weights arrive as plain
        # torch tensors (never converted to ttnn by the harness).
        idx = top_k_index if isinstance(top_k_index, torch.Tensor) else torch.zeros(num_tokens, 1, dtype=torch.long)
        wts = top_k_weights if isinstance(top_k_weights, torch.Tensor) else torch.zeros(num_tokens, 1)
        W_full = torch.zeros(num_tokens, E, dtype=torch.float32)
        W_full.scatter_add_(1, idx.long(), wts.float())

        out = None
        Eloc = self._Eloc
        if self._shard:
            # Shard the (tokens, E) routing matrix on the SAME expert axis
            # (dim 1 here == the weights' dim 0) with the SAME ShardTensor2dMesh
            # call used for up_proj/down_proj, so chip d's local W_sh columns
            # line up with chip d's local expert weights (both are the
            # contiguous window [d*Eloc, (d+1)*Eloc)).
            W_sh = ttnn.from_torch(
                W_full,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.device, mesh_shape=self._mesh_shape, dims=(None, 1)),
            )
        else:
            W_sh = self._dev(W_full)

        return self._mix(hs, W_sh, num_tokens)

    def _mix(self, hs, W_sh, num_tokens):
        """The expert-parallel mixture itself: pure ttnn, identical for the
        host-routing and device-routing entry paths above."""
        out = None
        Eloc = self._Eloc
        hs_bf = ttnn.typecast(hs, ttnn.bfloat16)
        for e in range(Eloc):
            up = ttnn.matmul(hs_bf, self._up[e], compute_kernel_config=self.ckc)  # (T, inter) bf16
            act = ttnn.relu(up)
            ttnn.deallocate(up)
            act = ttnn.multiply(act, act)  # relu2
            down = ttnn.matmul(act, self._down[e], compute_kernel_config=self.ckc)  # (T, hidden) bf16
            ttnn.deallocate(act)
            down_f = ttnn.typecast(down, ttnn.float32)
            ttnn.deallocate(down)
            we = ttnn.slice(W_sh, [0, e], [num_tokens, e + 1])  # (T,1) fp32, this chip's local column
            contrib = ttnn.multiply(down_f, we)
            ttnn.deallocate(down_f)
            ttnn.deallocate(we)
            out = contrib if out is None else ttnn.add(out, contrib)
        ttnn.deallocate(hs_bf)
        ttnn.deallocate(W_sh)

        if self._shard:
            out = ttnn.all_reduce(out, cluster_axis=self._tp_axis, topology=ttnn.Topology.Linear)
        return ttnn.typecast(out, ttnn.bfloat16)


# Module-level `build` — primary test entry point.
def build(device, torch_module=None):
    return TtNemotronHExperts.build(device, torch_module)


# Backward-compatible slug shim.
def nemotron_h_experts(device, torch_module=None):
    return TtNemotronHExperts.build(device, torch_module)
