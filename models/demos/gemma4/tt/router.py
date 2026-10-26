# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 Router: RMSNorm -> scale -> linear -> softmax -> topk -> sum-normalize -> per_expert_scale -> scatter

Fully on-device, trace-compatible. Returns dense routing weights [1,1,S,E] on device for sparse_matmul.
Gemma4 uses softmax-THEN-topk (opposite of GPT-OSS), and sum-normalizes the
top-k weights linearly (NOT a second softmax — that would diverge from HF).
"""


import torch

import ttnn
from models.demos.gemma4.tt.decode_matmul import DecodeLinear, flag, is_decode_rows
from models.demos.gemma4.tt.rms_norm import RMSNorm
from models.demos.gemma4.utils.general_utils import get_cache_file_name
from models.demos.gemma4.utils.substate import substate

# GEMMA4_ROUTER_DECODE_1D=1: explicit 1D-multicast decode config on the router projection (N = num_experts is
# 4 tiles, so the auto config runs 4 cores with a 2-tile K block: 36 us for a 0.7 MB weight on the 26B-A4B);
# a 22-tile K block takes it to 10 us (P150, trace replay), same HiFi2 numerics.
_ROUTER_DECODE_BLOCK_W = 22
# GEMMA4_ROUTER_FOLD_SCALE=1: decode rows apply HF's ``norm(x) * scale * hidden**-0.5`` as one rms_norm with
# the folded weight (3 programs -> 1 per layer). Prefill keeps the fp32 norm + two multiplies.


class Gemma4Router:
    def __init__(self, mesh_device, hf_config, state_dict, tensor_cache_path=None, dtype=ttnn.bfloat16):
        self.mesh_device = mesh_device
        self.num_experts = hf_config.num_experts
        self.top_k = hf_config.top_k_experts
        self.hidden_size = hf_config.hidden_size
        self.scalar_root_size = self.hidden_size**-0.5

        self.norm = RMSNorm(
            mesh_device=mesh_device,
            hf_config=hf_config,
            state_dict={},
            tensor_cache_path=f"{tensor_cache_path}/norm" if tensor_cache_path else None,
            with_scale=False,
        )

        if state_dict:
            scale_weight = state_dict["scale"].reshape(1, 1, 1, -1)
            proj_weight = substate(state_dict, "proj")["weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
            per_expert_scale_raw = state_dict["per_expert_scale"]
        else:
            scale_weight = None
            proj_weight = None
            per_expert_scale_raw = None

        is_mesh = hasattr(mesh_device, "shape")
        replicate_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

        # Tag the router proj cache filename with its dtype so flipping
        # ``router`` precision in precision_overrides.json doesn't reuse a
        # stale cache. ``scale`` and ``per_expert_scale`` are tiny and stay
        # at bfloat16 — no dtype suffix needed.
        from models.demos.gemma4.tt.precision import dtype_to_str

        dtype_suffix = f"_{dtype_to_str(dtype)}"

        self.scale = ttnn.as_tensor(
            scale_weight,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=replicate_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "scale"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.proj_weight = ttnn.as_tensor(
            proj_weight,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=replicate_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, f"proj.weight{dtype_suffix}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.proj = DecodeLinear(
            self.proj_weight,
            mesh_device,
            k=self.hidden_size,
            n=self.num_experts,
            enabled=flag("GEMMA4_ROUTER_DECODE_1D"),
            max_block_w=_ROUTER_DECODE_BLOCK_W,
        )

        # Folded decode norm gamma: scale * hidden**-0.5, derived ON DEVICE from the cached ``scale`` (a warm
        # tensor cache hands this constructor dataless placeholders, so a host-side product would be
        # persisted as zeros under a new cache name). TILE [1, 1, 1, H] is a valid rms_norm gamma layout.
        self.scale_folded = None
        if flag("GEMMA4_ROUTER_FOLD_SCALE"):
            self.scale_folded = ttnn.multiply(self.scale, self.scalar_root_size)

        # Per-expert scale on device [1, 1, 1, num_experts] for broadcast mul with dense routing
        if per_expert_scale_raw is not None:
            self.per_expert_scale = ttnn.as_tensor(
                per_expert_scale_raw.reshape(1, 1, 1, -1),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=replicate_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, "per_expert_scale"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.per_expert_scale = None
        self._fused = None  # (pes ROW_MAJOR, base tensor) of the fused router tail, built on first use
        self._packed = None  # (pes ROW_MAJOR, zero base) of the packed router tail (kernel rows), built on first use

    def _norm_scale(self, hidden_states):
        """Steps 1-2: ``norm(x) * scale * hidden**-0.5`` (the folded one-program form on decode rows when
        GEMMA4_ROUTER_FOLD_SCALE=1)."""
        if self.scale_folded is not None and is_decode_rows(hidden_states):
            return ttnn.rms_norm(hidden_states, weight=self.scale_folded, epsilon=self.norm.eps)
        normed = self.norm.forward(hidden_states)
        scaled = ttnn.mul(normed, self.scale)
        normed.deallocate(True)
        return ttnn.mul(scaled, self.scalar_root_size)

    def fused_routing(self, hidden_states, expert_weights, mesh_config, prenormed=False):
        """GEMMA4_MOE_FUSED=1 decode rows: norm, scale and projection as before, then ``custom_ops/router_topk``
        (softmax, top-k, sum-normalize, per-expert scale, this chip's expert columns) in one program -> the
        ``FusedRouting`` tables the fused expert path consumes. ``prenormed``: ``hidden_states`` already is
        ``norm(x) * scale * hidden**-0.5`` (the layer's fused norm program emits it, GEMMA4_LAYER_NORM_FUSED=1)."""
        from models.demos.gemma4.custom_ops.router_topk import RouterTopk
        from models.demos.gemma4.tt.experts.decode import FusedRouting

        if self._fused is None:
            pes = ttnn.to_layout(self.per_expert_scale, ttnn.ROW_MAJOR_LAYOUT)
            base = RouterTopk.make_base(
                self.mesh_device, mesh_config, expert_weights.num_local_experts, expert_weights.ep, expert_weights.tp
            )
            self._fused = (pes, base)
        scaled = hidden_states if prenormed else self._norm_scale(hidden_states)
        logits = self.proj(scaled)
        if not prenormed:
            scaled.deallocate(True)
        w, sparsity = RouterTopk.op(
            logits, self._fused[0], self._fused[1], expert_weights.num_local_experts, self.top_k
        )
        logits.deallocate(True)
        return FusedRouting(w, sparsity)

    def fused_sparse_routing(self, hidden_states):
        """GEMMA4_MOE_FUSED=1 on the decode MoE kernel rows: norm, scale and projection as before, then
        ``custom_ops/router_topk.op_packed`` (softmax, top-k, sum-normalize, per-expert scale, ids) in one program ->
        the ``PackedRouting`` rows ``tt/experts/moe_decode.py`` feeds the kernel through ``custom_ops/moe_frame_prep``
        (replaces ``softmax, topk, sum, mul, gather, div`` and their fill-pads: 10 programs -> 1)."""
        from models.demos.gemma4.custom_ops.router_topk import RouterTopk
        from models.demos.gemma4.tt.experts.decode import PackedRouting

        assert self.per_expert_scale is not None, "the packed router tail folds per_expert_scale"
        if self._packed is None:
            pes = ttnn.to_layout(self.per_expert_scale, ttnn.ROW_MAJOR_LAYOUT)
            is_mesh = hasattr(self.mesh_device, "shape")
            base = ttnn.from_torch(
                torch.zeros(1, 1, 1, 16, dtype=torch.int32),
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh else None,
            )
            self._packed = (pes, base)
        scaled = self._norm_scale(hidden_states)
        logits = self.proj(scaled)
        scaled.deallocate(True)
        packed = RouterTopk.op_packed(logits, self._packed[0], self._packed[1], self.top_k)
        logits.deallocate(True)
        return PackedRouting(packed, self.top_k)

    def __call__(self, hidden_states, sparse=False):
        """
        Route tokens to experts. Fully on-device, trace-compatible.

        ``sparse=True`` returns ``(values, indices)`` of shape [1, 1, seq_len, top_k] (bf16 / uint16, TILE):
        the sum-normalized top-k weights with the per-expert scale gathered at the selected experts, i.e. the
        same numbers the dense [.., num_experts] matrix carries at its non-zero columns, without the scatter
        and without the consumer re-selecting the top-k (the decode MoE kernel takes them directly).

        Args:
            hidden_states: [1, 1, seq_len, hidden_size] on device

        Returns:
            dense_routing: [1, 1, seq_len, num_experts] on device — weights at selected experts, zeros elsewhere
        """
        if self.scale_folded is not None and is_decode_rows(hidden_states):
            # 1+2. RMSNorm with the folded scale * hidden**-0.5 gamma (decode rows; see _ROUTER fold note)
            scaled = ttnn.rms_norm(hidden_states, weight=self.scale_folded, epsilon=self.norm.eps)
        else:
            # 1. RMSNorm — on device
            normed = self.norm.forward(hidden_states)

            # 2. Scale — on device
            scaled = ttnn.mul(normed, self.scale)
            normed.deallocate(True)
            scaled = ttnn.mul(scaled, self.scalar_root_size)

        # 3. Linear projection → [1, 1, seq_len, num_experts] — on device
        expert_scores = self.proj(scaled)
        scaled.deallocate(True)

        # 4. Softmax over all experts — on device
        router_probs = ttnn.softmax(expert_scores, dim=-1)
        expert_scores.deallocate(True)

        # 5. TopK — on device → values [1,1,S,k], indices [1,1,S,k]
        top_k_values, top_k_indices = ttnn.topk(router_probs, k=self.top_k, dim=-1)

        # 6. Sum-normalize top-k weights so they sum to 1 per token. HF Gemma4
        # divides by the sum here; a second softmax would compress the
        # distribution nonlinearly and diverge from the reference.
        top_k_sum = ttnn.sum(top_k_values, dim=-1, keepdim=True)
        if sparse:
            if self.per_expert_scale is not None:
                # HF: weights = normalize(topk(probs)) * per_expert_scale[idx]; the scale is gathered at the
                # selected experts from the scaled probabilities (same selection, same values).
                scaled = ttnn.mul(router_probs, self.per_expert_scale)
                top_k_values.deallocate(True)
                top_k_values = ttnn.gather(scaled, dim=-1, index=top_k_indices)
                scaled.deallocate(True)
            top_k_values = ttnn.div(top_k_values, top_k_sum)
            top_k_sum.deallocate(True)
            router_probs.deallocate(True)
            return top_k_values, top_k_indices
        top_k_values = ttnn.div(top_k_values, top_k_sum)
        top_k_sum.deallocate(True)

        # 7. Scatter into dense [1,1,S,E] — fully on device
        dense_routing = ttnn.scatter(
            ttnn.zeros_like(router_probs),
            dim=-1,
            index=top_k_indices,
            src=top_k_values,
        )
        router_probs.deallocate(True)
        top_k_values.deallocate(True)
        top_k_indices.deallocate(True)

        # 8. Per-expert scale — broadcast mul on device
        if self.per_expert_scale is not None:
            dense_routing = ttnn.mul(dense_routing, self.per_expert_scale)

        return dense_routing
