# SPDX-License-Identifier: Apache-2.0
"""Experimental dedicated expert kernel; never selected silently."""

import torch

import ttnn

from .fusion_candidates import DRAM, FusedDecoder


class ExpertCandidate(FusedDecoder):
    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        self = super().from_state_dict(*args, **kwargs)
        self.expert_lists = {}
        for role, w in self.experts.items():
            k, n = tuple(w.shape)[-2:]
            self.expert_lists[role] = [
                ttnn.reshape(ttnn.slice(w, (0, e, 0, 0), (1, e + 1, k, n)), (k, n)) for e in range(384)
            ]

        def table(v):
            return ttnn.from_torch(
                v, device=self.device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM
            )

        self.expert_ids = table(torch.arange(384, dtype=torch.int32))
        self.expert_offsets = {n: table(torch.arange(384, dtype=torch.int32) * n) for n in [32, 64, 96, 128, 160]}
        return self

    def _moe(self, x):
        # x [1,1,T,H]. A group shares sparse expert work; per-token scores
        # remain distinct and mask every unselected token/expert pair.
        tokens = x.shape[-2]
        logits = self._linear(
            x if self.router_cast else ttnn.typecast(x, ttnn.float32), self.router, dtype=ttnn.float32
        )
        choice = ttnn.add(logits, self.expert_bias)
        values, ids = ttnn.topk(choice, k=6, dim=-1)
        scores = logits if self.routing_fused else ttnn.sigmoid(logits)
        selected = ttnn.scatter(
            ttnn.zeros_like(ttnn.typecast(scores, ttnn.bfloat16)),
            dim=-1,
            index=ids,
            src=ttnn.ones_like(ttnn.typecast(values, ttnn.bfloat16)),
        )
        routing = (
            ttnn.multiply(scores, selected, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
            if self.routing_fused
            else ttnn.multiply(scores, selected)
        )
        sparsity = ttnn.to_layout(ttnn.max(selected, dim=2, keepdim=True), ttnn.ROW_MAJOR_LAYOUT)
        sparsity = ttnn.typecast(sparsity, ttnn.bfloat16)

        physical = (tokens + 31) // 32 * 32
        counts = ttnn.reshape(
            ttnn.to_layout(
                ttnn.typecast(ttnn.multiply(ttnn.to_layout(sparsity, ttnn.TILE_LAYOUT), float(tokens)), ttnn.uint32),
                ttnn.ROW_MAJOR_LAYOUT,
            ),
            (384,),
        )
        inp = ttnn.reshape(x, (tokens, 2560))
        if tokens != physical:
            inp = ttnn.pad(inp, ((0, physical - tokens), (0, 0)), 0.0)
        # Diagnostic workaround: current kernel bounds write-only offsets by input capacity.
        inp = ttnn.pad(inp, ((0, 383 * physical), (0, 0)), 0.0)
        inp = ttnn.to_layout(inp, ttnn.ROW_MAJOR_LAYOUT)
        output = ttnn.empty(
            (384 * physical, 2560), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=DRAM
        )
        out = ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
            inp,
            self.expert_lists["gate_proj"],
            self.expert_lists["up_proj"],
            self.expert_lists["down_proj"],
            counts,
            self.expert_ids,
            input_m_tiles=physical // 32,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            ),
            core_grid=ttnn.CoreCoord(8, 8),
            output=output,
            expert_region_offsets=self.expert_offsets[physical],
            read_x_at_offset=False,
        )
        down = ttnn.slice(ttnn.reshape(out, (1, 384, physical, 2560)), (0, 0, 0, 0), (1, 384, tokens, 2560))
        if self.reduce_fused and (tokens <= 32 or self.reduce_all):
            routed_parts = []
            for start in range(0, tokens, 32):
                count = min(32, tokens - start)
                scores_part = ttnn.slice(routing, (0, 0, start, 0), (1, 1, start + count, 384))
                scores_bf16 = ttnn.typecast(scores_part, ttnn.bfloat16)
                scores_rm = ttnn.reshape(ttnn.to_layout(scores_bf16, ttnn.ROW_MAJOR_LAYOUT), (count, 1, 1, 384))
                mask = ttnn.reshape(ttnn.permute(scores_bf16, (0, 3, 2, 1)), (1, 384, count, 1))
                piece = ttnn.slice(down, (0, 0, start, 0), (1, 384, start + count, 2560))
                clean = ttnn.where(mask, piece, 0.0)
                part = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
                    clean,
                    self.reduce_indices[32],
                    self.reduce_mapping,
                    1,
                    split_size=2560,
                    cluster_axis=1,
                    scores_tensor=scores_rm,
                    compute_kernel_config=self.compute,
                )[0]
                routed_parts.append(part)
            routed = ttnn.concat(routed_parts, dim=2) if len(routed_parts) > 1 else routed_parts[0]
        else:
            routing = ttnn.reshape(ttnn.permute(routing, (0, 3, 2, 1)), (1, 384, tokens, 1))
            weighted = ttnn.where(
                routing,
                (
                    ttnn.multiply(down, routing, dtype=ttnn.float32)
                    if self.mixed_weight
                    else ttnn.multiply(ttnn.typecast(down, ttnn.float32), routing)
                ),
                0.0,
            )
            routed = ttnn.sum(weighted, dim=1, keepdim=True)
        if self.shared_packed:
            both = self._linear(x, self.shared["gate_up"])
            width = both.shape[-1] // 2
            sg = ttnn.slice(both, (0, 0, 0, 0), (1, 1, tokens, width))
            su = ttnn.slice(both, (0, 0, 0, width), (1, 1, tokens, 2 * width))
        else:
            sg = self._linear(x, self.shared["gate_proj"], activation="silu" if self.shared_matmul_act else None)
            su = self._linear(x, self.shared["up_proj"])
        middle_shared = (
            ttnn.multiply(sg, su)
            if self.shared_matmul_act
            else (
                ttnn.multiply(sg, su, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
                if self.activation_fused
                else ttnn.multiply(ttnn.silu(sg), su)
            )
        )
        shared = self._linear(middle_shared, self.shared["down_proj"])
        return (
            ttnn.add(routed, shared, dtype=ttnn.bfloat16)
            if self.outputcast
            else ttnn.add(ttnn.typecast(routed, ttnn.bfloat16), shared)
        )
