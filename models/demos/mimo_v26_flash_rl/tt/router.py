# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Sigmoid noaux_tc top-8 router of the MiMo MoE layers on the [8, 4] mesh.

Per chip, one chunk of 5120 tokens (each TP column routes the same 640 rows of its SP row, the experts' dispatch
group input):
    x [1, 1, 640, 4096] bf16 (ffn_norm, gathered over TP)
    -> logits = x @ gate^T  (gate [256, 4096] bf16 replicated; HiFi4, fp32 accumulation, fp32 out)
    -> scores = sigmoid(logits) fp32
    -> choice = scores + (bias - c) fp32   (c: a bf16 constant near the bias mean; top-k is shift invariant)
    -> topk(8) indices -> mask = scatter(zeros, idx, ones) (bf16 0 / 1, exactly 8 per row) -> fp32
    -> routing = scores * mask / sum(scores * mask)   [1, 1, 640, 256] fp32, zero off the selected experts

Precision (known issues "A large correction bias makes noaux_tc top-k selection precision-bound" and
"moe_grouped_topk selects on TF32 keys"): the bias is 1.7..2.2, so the choice score is fp32 and recentred; the
fused moe_grouped_topk (TF32 keys) is not used. The dense output comes from a bf16 0 / 1 mask scattered at topk's
indices (no fp32 TILE scatter in ttnn; a mask on the 8th value selects 9 on an exact tie, ROUTER_MASK "kth"); the
weights are the unbiased sigmoid scores, renormalised.
"""

from __future__ import annotations

import torch

import ttnn

from .settings import settings


class Router:
    def __init__(self, mesh, ck, layer_idx: int, max_rows: int = 640):
        cfg = ck.config
        assert cfg["scoring_func"] == "sigmoid" and cfg["topk_method"] == "noaux_tc"
        assert cfg["n_group"] == 1 and cfg["topk_group"] == 1, "group-limited routing not implemented"
        self.mesh = mesh
        self.top_k = cfg["num_experts_per_tok"]
        self.norm_topk = cfg["norm_topk_prob"]
        self.routed_scale = cfg.get("routed_scaling_factor") or 1.0
        p = f"model.layers.{layer_idx}.mlp.gate"
        w = ck.raw(f"{p}.weight").to(torch.bfloat16)  # [E, H]
        self.n_experts = w.shape[0]
        # config moe_router_dtype (bf16): upstream runs the correction bias in bf16 (the reference's rounding)
        rdtype = getattr(torch, cfg.get("moe_router_dtype") or "float32")
        bias = ck.raw(f"{p}.e_score_correction_bias").to(rdtype).to(torch.float32)
        c = bias.mean().to(torch.bfloat16).to(torch.float32)  # bf16 constant: bias - c is exact in fp32
        rep = ttnn.ReplicateTensorToMesh(mesh)
        self.w = ttnn.from_torch(
            w.view(1, 1, *w.shape),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        self.bias = ttnn.from_torch(
            (bias - c).view(1, 1, 1, -1),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        self.mask_mode = settings.get("ROUTER_MASK")
        if self.mask_mode == "scatter":  # load-time constants, sliced per call
            bf = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            self.zeros = ttnn.from_torch(torch.zeros(1, 1, max_rows, self.n_experts), mesh_mapper=rep, **bf)
            self.ones = ttnn.from_torch(torch.ones(1, 1, max_rows, self.top_k), mesh_mapper=rep, **bf)
        self.mm_cfg = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, settings.get("ROUTER_FIDELITY")),
            math_approx_mode=False,
            fp32_dest_acc_en=settings.get("ROUTER_FP32_ACC"),
            packer_l1_acc=True,
        )
        self.red_cfg = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def __call__(self, x):
        """x [1, 1, R, H] bf16 (the column's gathered ffn_norm rows) -> dense routing [1, 1, R, E] fp32."""
        dram = ttnn.DRAM_MEMORY_CONFIG
        logits = ttnn.linear(
            x, self.w, transpose_b=True, dtype=ttnn.float32, memory_config=dram, compute_kernel_config=self.mm_cfg
        )
        scores = ttnn.sigmoid(logits, memory_config=dram)
        logits.deallocate(True)
        choice = ttnn.add(scores, self.bias, dtype=ttnn.float32, memory_config=dram)
        vals, idx = ttnn.topk(choice, k=self.top_k, dim=-1, largest=True, sorted=True, memory_config=dram)
        if self.mask_mode == "scatter":
            # exactly top_k ones per row at topk's indices (a tie at the 8th value cannot select 9)
            vals.deallocate(True)
            choice.deallocate(True)
            rows = idx.shape[2]
            part = rows != self.zeros.shape[2]  # a full-size slice returns the constant itself: keep it
            zeros, ones = self.zeros, self.ones
            if part:
                zeros = ttnn.slice(zeros, [0, 0, 0, 0], [1, 1, rows, self.n_experts], memory_config=dram)
                ones = ttnn.slice(ones, [0, 0, 0, 0], [1, 1, rows, self.top_k], memory_config=dram)
            mask_bf = ttnn.scatter(zeros, -1, idx, ones, memory_config=dram)
            if part:
                zeros.deallocate(True)
                ones.deallocate(True)
            idx.deallocate(True)
            mask = ttnn.typecast(mask_bf, ttnn.float32, memory_config=dram)
            mask_bf.deallocate(True)
        else:
            idx.deallocate(True)
            kth = ttnn.slice(vals, [0, 0, 0, self.top_k - 1], [1, 1, vals.shape[2], self.top_k], memory_config=dram)
            vals.deallocate(True)
            mask = ttnn.ge(choice, kth, dtype=ttnn.float32, memory_config=dram)
            choice.deallocate(True)
            kth.deallocate(True)
        sel = ttnn.multiply(scores, mask, dtype=ttnn.float32, memory_config=dram)
        scores.deallocate(True)
        mask.deallocate(True)
        if self.top_k > 1 and self.norm_topk:
            s = ttnn.sum(sel, dim=-1, keepdim=True, memory_config=dram, compute_kernel_config=self.red_cfg)
            if self.routed_scale != 1.0:
                s = ttnn.multiply(s, 1.0 / self.routed_scale, memory_config=dram)
            out = ttnn.div(sel, s, memory_config=dram)
            sel.deallocate(True)
            s.deallocate(True)
            return out
        if self.routed_scale != 1.0:
            out = ttnn.multiply(sel, self.routed_scale, memory_config=dram)
            sel.deallocate(True)
            return out
        return sel
