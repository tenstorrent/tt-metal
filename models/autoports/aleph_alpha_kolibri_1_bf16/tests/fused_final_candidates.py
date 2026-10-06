# SPDX-License-Identifier: Apache-2.0
"""Explicit final-graph trials for constant masks and shared fused SwiGLU."""

import os

import ttnn

from ..tt.fused_decoder import DRAM, FusedDecoder


class FinalGraphCandidate(FusedDecoder):
    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        import torch

        self = super().from_state_dict(state_dict, **kwargs)

        def convert(x):
            return ttnn.from_torch(
                x.contiguous(), device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=DRAM
            )

        if os.environ.get("FUSION_IMPL") in ("mask", "mask_rm", "mask_rm128"):
            rows = 128 if os.environ.get("FUSION_IMPL") == "mask_rm128" else 32
            self.mask_zero = convert(torch.zeros(1, 1, rows, 384))
            self.mask_one = convert(torch.ones(1, 1, rows, 6))
            self.mask_zero_rm = ttnn.to_layout(self.mask_zero, ttnn.ROW_MAJOR_LAYOUT)
            self.mask_one_rm = ttnn.to_layout(self.mask_one, ttnn.ROW_MAJOR_LAYOUT)
        else:
            prefix = f"model.layers.{self.layer_idx}.mlp.shared_experts."
            state_dict = {k.removeprefix(f"model.layers.{self.layer_idx}."): v for k, v in state_dict.items()}
            prefix = "mlp.shared_experts."
            packed = torch.cat([state_dict[prefix + k + ".weight"].T for k in ("gate_proj", "up_proj")], dim=-1)
            self.shared_interleaved = convert(packed.reshape(2560, 2, 16, 32).permute(0, 2, 1, 3).reshape(2560, 1024))
        return self

    def _moe(self, x, *, prefill=False):
        tokens = x.shape[-2]
        logits = self._linear(x, self.router, dtype=ttnn.float32)
        choice = ttnn.add(logits, self.expert_bias)
        values, ids = ttnn.topk(choice, k=6, dim=-1)
        scores = logits
        if os.environ.get("FUSION_IMPL") in ("mask", "mask_rm", "mask_rm128"):
            rows = self.mask_zero.shape[-2]
            use_rm = os.environ.get("FUSION_IMPL") in ("mask_rm", "mask_rm128") and tokens <= rows
            zero = self.mask_zero_rm if use_rm else self.mask_zero
            one = self.mask_one_rm if use_rm else self.mask_one
            if tokens > rows:
                zero = ttnn.repeat(zero, (1, 1, (tokens + rows - 1) // rows, 1))
                one = ttnn.repeat(one, (1, 1, (tokens + rows - 1) // rows, 1))
            zero = ttnn.slice(zero, (0, 0, 0, 0), (1, 1, tokens, 384))
            one = ttnn.slice(one, (0, 0, 0, 0), (1, 1, tokens, 6))
        else:
            zero = ttnn.zeros_like(ttnn.typecast(scores, ttnn.bfloat16))
            one = ttnn.ones_like(ttnn.typecast(values, ttnn.bfloat16))
        selected = ttnn.scatter(
            zero,
            dim=-1,
            index=ids,
            src=one,
        )
        selected = ttnn.to_layout(selected, ttnn.TILE_LAYOUT)
        routing = ttnn.multiply(scores, selected, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        active = ttnn.max(selected, dim=2, keepdim=True)

        if prefill and tokens >= 64:
            counts = ttnn.reshape(
                ttnn.to_layout(
                    ttnn.typecast(ttnn.multiply(ttnn.typecast(active, ttnn.float32), float(tokens)), ttnn.uint32),
                    ttnn.ROW_MAJOR_LAYOUT,
                ),
                (384,),
            )
            offsets = ttnn.reshape(
                ttnn.to_layout(
                    ttnn.typecast(ttnn.multiply(self.expert_sequence, float(tokens)), ttnn.uint32),
                    ttnn.ROW_MAJOR_LAYOUT,
                ),
                (384,),
            )
            # Current dedicated op bounds output-only offsets against input capacity.
            # Padding avoids the defect without changing shared code; only the first
            # tokens rows are read (read_x_at_offset=False).
            expanded = ttnn.pad(ttnn.reshape(x, (tokens, 2560)), ((0, 383 * tokens), (0, 0)), 0.0)
            expanded = ttnn.to_layout(expanded, ttnn.ROW_MAJOR_LAYOUT)
            output = ttnn.empty(
                (384 * tokens, 2560),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=DRAM,
            )
            down = ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
                expanded,
                self.prefill_experts["gate_proj"],
                self.prefill_experts["up_proj"],
                self.prefill_experts["down_proj"],
                counts,
                self.expert_ids,
                input_m_tiles=tokens // 32,
                compute_kernel_config=self.expert_compute,
                core_grid=ttnn.CoreCoord(8, 8),
                output=output,
                expert_region_offsets=offsets,
                read_x_at_offset=False,
            )
            down = ttnn.reshape(down, (1, 384, tokens, 2560))
        else:
            sparsity = ttnn.to_layout(active, ttnn.ROW_MAJOR_LAYOUT)

            def sparse(a, w, n, active=False):
                return ttnn.sparse_matmul(
                    a,
                    w,
                    sparsity=sparsity,
                    nnz=None,
                    is_input_a_sparse=active,
                    is_input_b_sparse=True,
                    program_config=self._sparse_config(tokens, n),
                    compute_kernel_config=self.compute,
                    dtype=ttnn.bfloat16,
                    memory_config=DRAM,
                )

            gate_up = ttnn.reshape(sparse(x, self.experts["gate_up"], 1024), (1, 384, tokens, 1024))
            gate = ttnn.slice(gate_up, (0, 0, 0, 0), (1, 384, tokens, 512))
            up = ttnn.slice(gate_up, (0, 0, 0, 512), (1, 384, tokens, 1024))
            middle = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
            down = ttnn.reshape(sparse(middle, self.experts["down_proj"], 2560, True), (1, 384, tokens, 2560))
        # The fused reducer allocates one score tile per expert and row tile.
        # Limit its internal row group to 32 so score tiles fit in L1. Public
        # sequence lengths stay arbitrary; prefill planning pads physical chunks.
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
                self.reduce_indices,
                self.reduce_mapping,
                1,
                split_size=2560,
                cluster_axis=1,
                scores_tensor=scores_rm,
                compute_kernel_config=self.compute,
            )[0]
            routed_parts.append(part)
        routed = ttnn.concat(routed_parts, dim=2) if len(routed_parts) > 1 else routed_parts[0]
        if os.environ.get("FUSION_IMPL") == "minimal":
            middle_shared = ttnn.experimental.minimal_matmul(
                x, self.shared_interleaved, fuse_swiglu=True, compute_kernel_config=self.compute, dtype=ttnn.bfloat16
            )
        else:
            both = self._linear(x, self.shared["gate_up"])
            width = both.shape[-1] // 2
            sg = ttnn.slice(both, (0, 0, 0, 0), (1, 1, tokens, width))
            su = ttnn.slice(both, (0, 0, 0, width), (1, 1, tokens, 2 * width))
            middle_shared = ttnn.multiply(sg, su, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        shared = self._linear(middle_shared, self.shared["down_proj"])
        return ttnn.add(routed, shared, dtype=ttnn.bfloat16)


class RowMajorPadCandidate(FusedDecoder):
    def _moe(self, x, *, prefill=False):
        tokens = x.shape[-2]
        logits = self._linear(x, self.router, dtype=ttnn.float32)
        choice = ttnn.add(logits, self.expert_bias)
        _, ids = ttnn.topk(choice, k=6, dim=-1)
        scores = logits
        if tokens <= 128:
            zero, one = self.mask_zero_rm, self.mask_one_rm
        else:
            # Larger public chunk choices remain supported entirely on device.
            repeats = (tokens + 127) // 128
            zero = ttnn.repeat(self.mask_zero, (1, 1, repeats, 1))
            one = ttnn.repeat(self.mask_one, (1, 1, repeats, 1))
        zero = ttnn.slice(zero, (0, 0, 0, 0), (1, 1, tokens, 384))
        one = ttnn.slice(one, (0, 0, 0, 0), (1, 1, tokens, 6))
        selected = ttnn.scatter(zero, dim=-1, index=ids, src=one)
        selected = ttnn.to_layout(selected, ttnn.TILE_LAYOUT)
        routing = ttnn.multiply(scores, selected, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        active = ttnn.max(selected, dim=2, keepdim=True)

        if prefill and tokens >= 64:
            counts = ttnn.reshape(
                ttnn.to_layout(
                    ttnn.typecast(ttnn.multiply(ttnn.typecast(active, ttnn.float32), float(tokens)), ttnn.uint32),
                    ttnn.ROW_MAJOR_LAYOUT,
                ),
                (384,),
            )
            offsets = ttnn.reshape(
                ttnn.to_layout(
                    ttnn.typecast(ttnn.multiply(self.expert_sequence, float(tokens)), ttnn.uint32),
                    ttnn.ROW_MAJOR_LAYOUT,
                ),
                (384,),
            )
            # Current dedicated op bounds output-only offsets against input capacity.
            # Padding avoids the defect without changing shared code; only the first
            # tokens rows are read (read_x_at_offset=False).
            rows = ttnn.to_layout(ttnn.reshape(x, (tokens, 2560)), ttnn.ROW_MAJOR_LAYOUT)
            expanded = ttnn.pad(rows, ((0, 383 * tokens), (0, 0)), 0.0)
            output = ttnn.empty(
                (384 * tokens, 2560),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=DRAM,
            )
            down = ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
                expanded,
                self.prefill_experts["gate_proj"],
                self.prefill_experts["up_proj"],
                self.prefill_experts["down_proj"],
                counts,
                self.expert_ids,
                input_m_tiles=tokens // 32,
                compute_kernel_config=self.expert_compute,
                core_grid=ttnn.CoreCoord(8, 8),
                output=output,
                expert_region_offsets=offsets,
                read_x_at_offset=False,
            )
            down = ttnn.reshape(down, (1, 384, tokens, 2560))
        else:
            sparsity = ttnn.to_layout(active, ttnn.ROW_MAJOR_LAYOUT)

            def sparse(a, w, n, active=False):
                return ttnn.sparse_matmul(
                    a,
                    w,
                    sparsity=sparsity,
                    nnz=None,
                    is_input_a_sparse=active,
                    is_input_b_sparse=True,
                    program_config=self._sparse_config(tokens, n),
                    compute_kernel_config=self.compute,
                    dtype=ttnn.bfloat16,
                    memory_config=DRAM,
                )

            gate_up = ttnn.reshape(sparse(x, self.experts["gate_up"], 1024), (1, 384, tokens, 1024))
            gate = ttnn.slice(gate_up, (0, 0, 0, 0), (1, 384, tokens, 512))
            up = ttnn.slice(gate_up, (0, 0, 0, 512), (1, 384, tokens, 1024))
            middle = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
            down = ttnn.reshape(sparse(middle, self.experts["down_proj"], 2560, True), (1, 384, tokens, 2560))
        # The fused reducer allocates one score tile per expert and row tile.
        # Limit its internal row group to 32 so score tiles fit in L1. Public
        # sequence lengths stay arbitrary; prefill planning pads physical chunks.
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
                self.reduce_indices,
                self.reduce_mapping,
                1,
                split_size=2560,
                cluster_axis=1,
                scores_tensor=scores_rm,
                compute_kernel_config=self.compute,
            )[0]
            routed_parts.append(part)
        routed = ttnn.concat(routed_parts, dim=2) if len(routed_parts) > 1 else routed_parts[0]
        both = self._linear(x, self.shared["gate_up"])
        width = both.shape[-1] // 2
        sg = ttnn.slice(both, (0, 0, 0, 0), (1, 1, tokens, width))
        su = ttnn.slice(both, (0, 0, 0, width), (1, 1, tokens, 2 * width))
        middle_shared = ttnn.multiply(sg, su, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        shared = self._linear(middle_shared, self.shared["down_proj"])
        return ttnn.add(routed, shared, dtype=ttnn.bfloat16)
