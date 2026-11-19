# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MoE router: softmax over 128 experts, top-4, renormalized.

Adapted from moe/tt_moe_gate_prefill.py (GateComputeMode.GPT_DEVICE: matmul, then
ttnn.topk on the logits, then a softmax over the selected ones). softmax -> top-4 -> renormalize equals top-4 on the
logits -> softmax over those 4, exactly (routed_scaling_factor 1.0). Per chip (SP row r of 8, TP column t of 4):

    x       [1, 1, 640, 1536] ffn_norm -> all_gather(axis 1) -> x_full [1, 1, 640, 6144]
    logits  x_full @ gate^T, gate [128, 6144] replicated in ROUTER_WEIGHT_DTYPE, fp32 out   [1, 1, 640, 128]
    top-4   ttnn.topk(logits, k) -> indices [640, k]; sel = sum_j (expert_id == indices[:, j]) [640, 128] 0 / 1
    dense   e = exp(logits - max(logits)) * sel; router = e / sum(e)                       [1, 1, 640, 128] fp32

The dense form is the reference's component boundary ([S, 128], the 4 weights at their experts, 0 elsewhere), built
without a scatter (no fp32 TILE path) and without a softmax over a 4-wide padded tile. Every TP column
computes the same rows (replicated over axis 1).
"""

import torch

import ttnn

from models.demos.deepseek_v3_d_p.tt.mistral_large_4 import tp_gather
from models.demos.deepseek_v3_d_p.tt.mistral_large_4.settings import settings


def _compute_config(mesh):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, settings.get("ROUTER_MATH_FIDELITY")),
        math_approx_mode=False,
        fp32_dest_acc_en=settings.get("ROUTER_FP32_ACC"),
        packer_l1_acc=False,
    )


class TtRouter:
    # read at run time, so a changed setting applies to an already loaded model
    @property
    def cfg(self):
        return _compute_config(self.mesh)

    @property
    def topology(self):
        return getattr(ttnn.Topology, settings.get("ROUTER_TOPOLOGY"))

    def __init__(self, mesh, gate: torch.Tensor, top_k: int, routed_scale: float = 1.0):
        """gate: [E, H] (checkpoint order, host, any float dtype)."""
        self.mesh, self.top_k, self.routed_scale = mesh, top_k, routed_scale
        self.last_indices = None
        # [E, H] read with transpose_b, as mlp.py; replicated (128 experts x 6144 fp32 = 3.1 MB per chip)
        self.gate = ttnn.from_torch(
            gate.float()[None, None].contiguous(),
            device=mesh,
            dtype=getattr(ttnn, settings.get("ROUTER_WEIGHT_DTYPE")),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        # expert ids [1, 1, 1, E] fp32, compared with each top-k index column to build the dense selection
        self.expert_ids = ttnn.from_torch(
            torch.arange(gate.shape[0], dtype=torch.float32).reshape(1, 1, 1, -1),
            device=mesh,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: ffn_norm [1, 1, S/8, H/4] bf16 per chip -> dense routing [1, 1, S/8, E] fp32 per chip (TP-replicated)."""
        x_full = tp_gather.gather(x, self.topology, self.gate.shape[-1])
        logits = ttnn.matmul(
            x_full,
            self.gate,
            transpose_b=True,
            dtype=ttnn.float32,
            compute_kernel_config=self.cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tp_gather.done(x_full, x)
        values, indices = ttnn.topk(logits, k=self.top_k, dim=-1, largest=True, sorted=True)
        rows = logits.shape[-2]
        if settings.get("ROUTER_OUT") == "topk" and values.dtype == ttnn.float32:
            # softmax over the k selected logits only ([rows, k] in place of the dense [rows, E]); the experts
            # take these weights and ids as they are (no second top-k): the same fp32 exp terms and sum
            ttnn.deallocate(logits)
            top = ttnn.slice(values, [0, 0, 0, 0], [1, 1, rows, 1])
            shifted = ttnn.sub(values, top, dtype=ttnn.float32)
            ttnn.deallocate(values)
            ttnn.deallocate(top)
            ex = ttnn.exp(shifted, fast_and_approximate_mode=False)
            ttnn.deallocate(shifted)
            denom = ttnn.sum(ex, dim=-1, keepdim=True, compute_kernel_config=self.cfg)
            if self.routed_scale != 1.0:
                denom = ttnn.multiply(denom, 1.0 / self.routed_scale)
            w = ttnn.div(ex, denom, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(ex)
            ttnn.deallocate(denom)
            self.last_indices = indices
            return w
        # the cut must be the exact fp32 logit: a rounded one would move experts in or out
        if settings.get("ROUTER_SELECT") == "threshold" and values.dtype == ttnn.float32:
            # the sorted top-k values give the row max (column 0) and the cut (column k - 1)
            ttnn.deallocate(indices)
            top = ttnn.slice(values, [0, 0, 0, 0], [1, 1, rows, 1])
            cut = ttnn.slice(values, [0, 0, 0, self.top_k - 1], [1, 1, rows, self.top_k])
            ttnn.deallocate(values)
            sel = ttnn.ge(logits, cut, dtype=ttnn.float32)
            ttnn.deallocate(cut)
            return self._normalize(logits, top, sel)
        ttnn.deallocate(values)
        idx = ttnn.typecast(indices, ttnn.float32)  # exact: expert ids < 2^24
        ttnn.deallocate(indices)
        sel = None
        for j in range(self.top_k):
            col = ttnn.slice(idx, [0, 0, 0, j], [1, 1, rows, j + 1])
            hit = ttnn.eq(self.expert_ids, col, dtype=ttnn.float32)
            if sel is None:
                sel = hit
            else:
                sel = ttnn.add(sel, hit, dtype=ttnn.float32)
                ttnn.deallocate(hit)
        ttnn.deallocate(idx)
        top = ttnn.max(logits, dim=-1, keepdim=True)
        return self._normalize(logits, top, sel)

    def take_indices(self):
        """ROUTER_OUT=topk: the expert ids of the latest call ([rows, k]); the caller owns (deallocates) them."""
        idx, self.last_indices = self.last_indices, None
        return idx

    def _normalize(self, logits, top, sel):
        """exp(logits - row max) at the selected experts, divided by their sum; deallocates its inputs."""
        shifted = ttnn.sub(logits, top, dtype=ttnn.float32)
        ttnn.deallocate(logits)
        ttnn.deallocate(top)
        ex = ttnn.exp(shifted, fast_and_approximate_mode=False)
        ttnn.deallocate(shifted)
        e = ttnn.mul(ex, sel, dtype=ttnn.float32)
        ttnn.deallocate(ex)
        ttnn.deallocate(sel)
        denom = ttnn.sum(e, dim=-1, keepdim=True, compute_kernel_config=self.cfg)
        if self.routed_scale != 1.0:
            denom = ttnn.multiply(denom, 1.0 / self.routed_scale)
        out = ttnn.div(e, denom, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(e)
        ttnn.deallocate(denom)
        return out
