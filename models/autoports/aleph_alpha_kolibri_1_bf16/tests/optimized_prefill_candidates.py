# SPDX-License-Identifier: Apache-2.0
"""Grouped active-token prefill and dedicated-expert precision experiments."""
import json
import os

import torch

import ttnn

from ..tt.optimized_decoder import DRAM, OptimizedDecoder


class PrefillCandidate(OptimizedDecoder):
    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        self = super().from_state_dict(state_dict, **kwargs)
        self.prefill_options = json.loads(os.environ.get("OPT_PREFILL", "{}"))
        dtype = getattr(ttnn, self.prefill_options.get("dtype", "bfloat16"))
        if dtype != ttnn.bfloat16:
            self.prefill_experts = {
                role: [ttnn.typecast(w, dtype) for w in weights] for role, weights in self.prefill_experts.items()
            }
        self.expert_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, self.prefill_options.get("fidelity", "HiFi4")),
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        self.dispatch_table = ttnn.from_torch(
            torch.zeros(1, 384, dtype=torch.int32),
            device=self.device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=DRAM,
        )
        return self

    def _moe(self, x, *, prefill=False):
        tokens = x.shape[-2]
        if not prefill or not self.prefill_options.get("dispatch", False):
            return super()._moe(x, prefill=prefill)
        logits = self._linear(x, self.router, dtype=ttnn.float32)
        _, ids = ttnn.topk(ttnn.add(logits, self.expert_bias), k=6, dim=-1)
        if tokens > 128:
            zero = ttnn.to_layout(ttnn.repeat(self.mask_zero, (1, 1, (tokens + 127) // 128, 1)), ttnn.ROW_MAJOR_LAYOUT)
            one = ttnn.to_layout(ttnn.repeat(self.mask_one, (1, 1, (tokens + 127) // 128, 1)), ttnn.ROW_MAJOR_LAYOUT)
            zero = ttnn.slice(zero, (0, 0, 0, 0), (1, 1, tokens, 384))
            one = ttnn.slice(one, (0, 0, 0, 0), (1, 1, tokens, 6))
        else:
            zero = ttnn.slice(self.mask_zero_rm, (0, 0, 0, 0), (1, 1, tokens, 384))
            one = ttnn.slice(self.mask_one_rm, (0, 0, 0, 0), (1, 1, tokens, 6))
        selected = ttnn.to_layout(ttnn.scatter(zero, dim=-1, index=ids, src=one), ttnn.TILE_LAYOUT)
        counts = ttnn.sum(ttnn.typecast(selected, ttnn.float32), dim=2, keepdim=True)
        aligned = ttnn.multiply(ttnn.ceil(ttnn.multiply(counts, 1 / 32)), 32)
        offsets = ttnn.subtract(ttnn.cumsum(aligned, dim=3), aligned)

        def ints(t, shape):
            return ttnn.reshape(ttnn.to_layout(ttnn.typecast(t, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), shape)

        offsets = ints(offsets, (1, 384))
        counts = ints(counts, (1, 384))
        capacity = ((6 * tokens + 31 * 384 + 31) // 32) * 32
        dispatch_ids = ttnn.to_layout(ttnn.typecast(ids, ttnn.uint16), ttnn.ROW_MAJOR_LAYOUT)
        routed, metadata = ttnn.experimental.deepseek_prefill.dispatch(
            ttnn.reshape(x, (1, tokens, 2560)),
            ttnn.reshape(dispatch_ids, (1, tokens, 6)),
            offsets,
            self.dispatch_table,
            dispatch_group_size=1,
            experts_per_chip=384,
            num_routed_experts=384,
            num_experts_per_tok=6,
            metadata_len=3,
            max_dispatch_buffer_token_size=capacity,
            cluster_axis=0,
        )
        routed = ttnn.reshape(routed, (capacity, 2560))
        output = ttnn.empty(
            (capacity, 2560), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=DRAM
        )
        down = ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
            routed,
            self.prefill_experts["gate_proj"],
            self.prefill_experts["up_proj"],
            self.prefill_experts["down_proj"],
            ttnn.reshape(counts, (384,)),
            self.expert_ids,
            input_m_tiles=(tokens + 31) // 32,
            compute_kernel_config=self.expert_compute,
            core_grid=ttnn.CoreCoord(*self.prefill_options.get("grid", [8, 8])),
            output=output,
            expert_region_offsets=ttnn.reshape(offsets, (384,)),
            read_x_at_offset=True,
        )
        combined = ttnn.experimental.deepseek_prefill.combine(
            ttnn.reshape(down, (1, 1, capacity, 2560)),
            metadata,
            ttnn.reshape(counts, (1, 1, 384)),
            ttnn.reshape(offsets, (1, 1, 384)),
            dispatch_group_size=1,
            experts_per_chip=384,
            num_experts_per_tok=6,
            seq_len_per_chip=tokens,
            cluster_axis=0,
        )
        scores = ttnn.sigmoid(ttnn.gather(logits, -1, index=ids))
        scores = ttnn.to_layout(ttnn.typecast(scores, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT)
        scores = ttnn.reshape(scores, (1, 1, tokens, 6, 1))
        routed = ttnn.experimental.deepseek_prefill.post_combine_reduce(
            combined, scores, dispatch_ids, self.dispatch_table, expert_dim=3, output_memory_config=DRAM
        )
        both = self._linear(x, self.shared["gate_up"])
        gate = ttnn.slice(both, (0, 0, 0, 0), (1, 1, tokens, 512))
        up = ttnn.slice(both, (0, 0, 0, 512), (1, 1, tokens, 1024))
        middle = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        return ttnn.add(routed, self._linear(middle, self.shared["down_proj"]), dtype=ttnn.bfloat16)
