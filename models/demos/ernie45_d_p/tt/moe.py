# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ERNIE MoE (64 routed experts top-6 + 2 shared) on a 1x4 mesh.

Router: replicated, fp32 (softmax probs; bias-corrected selection; renormalized weights), producing a
dense routing matrix R [1,1,S,64] (weights at the chosen experts, 0 elsewhere) on every chip.
Routed experts: expert-parallel, chip c owns experts 16c..16c+15. Bring-up version is "dense-EP":
every local expert runs on all S tokens and is scaled by its R column (exact, 16/6 more FLOPs than
necessary). Shared experts: TP4 (768 of 3072 FFN columns per chip). routed + shared partials are summed
per chip and reduced with a single all_reduce.
"""

from __future__ import annotations

import ttnn
from models.demos.ernie45_d_p.reference.ernie_ref import ErnieConfig, LayerWeights
from models.demos.ernie45_d_p.tt.common import COMPUTE_HIFI2, COMPUTE_HIFI4, cache_name, replicate, shard
from models.demos.ernie45_d_p.tt.ops import TtSwiGLU, all_reduce

NUM_CHIPS = 4


class TtRouter:
    def __init__(self, mesh, cfg: ErnieConfig, layer: int, w: LayerWeights):
        self.cfg = cfg
        E = cfg.moe_num_experts
        self.w = replicate(
            mesh,
            w.router.float().T.contiguous()[None, None],
            dtype=ttnn.float32,
            cache=cache_name(f"L{layer}", "router"),
        )
        self.bias = replicate(
            mesh, w.e_bias.float().reshape(1, 1, 1, E), dtype=ttnn.float32, cache=cache_name(f"L{layer}", "e_bias")
        )

    def __call__(self, x):
        """x: [1,1,S,H] bf16 replicated -> (dense routing R [1,1,S,E] bf16, topk idx, topk weights)."""
        xf = ttnn.typecast(x, ttnn.float32)
        logits = ttnn.linear(xf, self.w, compute_kernel_config=COMPUTE_HIFI4)
        ttnn.deallocate(xf)
        probs = ttnn.softmax(logits, dim=-1, numeric_stable=True, compute_kernel_config=COMPUTE_HIFI4)
        ttnn.deallocate(logits)
        sel = ttnn.add(probs, self.bias)
        _, idx = ttnn.topk(sel, k=self.cfg.moe_k, dim=-1, largest=True, sorted=True)
        ttnn.deallocate(sel)
        wts = ttnn.gather(probs, dim=-1, index=idx)
        wsum = ttnn.sum(wts, dim=-1, keepdim=True)
        wts = ttnn.div(wts, wsum)
        ttnn.deallocate(wsum)
        dense = ttnn.scatter(ttnn.zeros_like(probs), dim=-1, index=idx, src=wts)
        ttnn.deallocate(probs)
        return ttnn.typecast(dense, ttnn.bfloat16), idx, wts


class TtMoE:
    def __init__(self, mesh, cfg: ErnieConfig, layer: int, w: LayerWeights):
        self.mesh, self.cfg, self.layer = mesh, cfg, layer
        self.router = TtRouter(mesh, cfg, layer, w)
        self.shared = TtSwiGLU(mesh, w.w_gate, w.w_up, w.w_down, name=f"L{layer}/shared")
        E = cfg.moe_num_experts
        self.e_local = E // NUM_CHIPS
        # Slot j on chip c holds expert 16c + j: stack over chips, shard dim 0.
        order = lambda j: [c * self.e_local + j for c in range(NUM_CHIPS)]  # noqa: E731
        self.gate, self.up, self.down = [], [], []
        for j in range(self.e_local):
            ids = order(j)
            nm = f"L{layer}/experts/slot{j:02d}"
            self.gate.append(
                shard(mesh, w.e_gate[ids].transpose(1, 2).contiguous()[:, None], 0, cache=cache_name(nm, "gate"))
            )
            self.up.append(
                shard(mesh, w.e_up[ids].transpose(1, 2).contiguous()[:, None], 0, cache=cache_name(nm, "up"))
            )
            self.down.append(
                shard(mesh, w.e_down[ids].transpose(1, 2).contiguous()[:, None], 0, cache=cache_name(nm, "down"))
            )

    def routed_partial(self, x, dense_routing):
        seq = x.shape[-2]
        local = ttnn.mesh_partition(dense_routing, dim=3, cluster_axis=1)  # [1,1,S,16]: chip c's experts
        acc = None
        for j in range(self.e_local):
            g = ttnn.linear(x, self.gate[j], compute_kernel_config=COMPUTE_HIFI2)
            u = ttnn.linear(x, self.up[j], compute_kernel_config=COMPUTE_HIFI2)
            h = ttnn.mul(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
            ttnn.deallocate(g)
            ttnn.deallocate(u)
            y = ttnn.linear(h, self.down[j], compute_kernel_config=COMPUTE_HIFI2)
            ttnn.deallocate(h)
            wj = ttnn.slice(local, [0, 0, 0, j], [1, 1, seq, j + 1])
            yw = ttnn.mul(y, wj)
            ttnn.deallocate(y)
            ttnn.deallocate(wj)
            if acc is None:
                acc = yw
            else:
                acc2 = ttnn.add(acc, yw)
                ttnn.deallocate(acc)
                ttnn.deallocate(yw)
                acc = acc2
        ttnn.deallocate(local)
        return acc

    def __call__(self, x, debug: dict | None = None):
        dense, idx, wts = self.router(x)
        if debug is not None:
            debug.update(routing=dense, topk_idx=idx, topk_w=wts)
        else:
            ttnn.deallocate(idx)
            ttnn.deallocate(wts)
        routed = self.routed_partial(x, dense)
        ttnn.deallocate(dense)
        shared = self.shared(x)
        if debug is not None:
            debug.update(routed_partial=routed, shared_partial=shared)
        tot = ttnn.add(routed, shared)
        if debug is None:
            ttnn.deallocate(routed)
            ttnn.deallocate(shared)
        out = all_reduce(tot)
        ttnn.deallocate(tot)
        return out
