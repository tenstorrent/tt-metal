# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5-35B-A3B specific MoE implementations for TTNN.

This module contains Qwen-specific subclasses that inherit from the GLM base classes
in moe.py. Key differences:
- TTNNQwenMoERouterDecode: Uses softmax activation instead of sigmoid
- TTNNQwenExperts: Uses sparse_matmul with fused w1/w3 (gate/up) projections
- TTNNQwen3MoE: Handles Qwen's shared_expert (singular) and optional shared_expert_gate

Environment Variables:
- TT_QWEN_CPU_EXPERTS: Set to "1" to use CPU fallback for experts (for debugging).
  When enabled, TTNNQwenExperts is NOT created and the PyTorch experts are used instead.
"""

import os
import torch
import ttnn

from models.experimental.tt_symbiote.core.module import run_on_devices, DeviceArch, tree_map
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.core.run_config import DistributedTensorConfig
from models.experimental.tt_symbiote.modules.moe import (
    TTNNMoERouterDecode,
    TTNNExperts,
    TTNNMoE,
    TTNNGlm4MoeTopkRouter,
    TTNNGlm4MoeMLP,
    Glm4MoeRouteTokenToExperts,
    even_int_div,
    _make_sparse_matmul_program_config,
    SPARSITY_BLOCK_SIZE,
)


class TTNNQwenMoERouterDecode(TTNNMoERouterDecode):
    """Qwen-specific router using softmax activation instead of sigmoid.

    Qwen3 architecture uses softmax for routing scores, while GLM-4 uses sigmoid.
    This subclass overrides the forward method to use softmax.

    Inheritance:
        - from_torch(): Inherited (unchanged)
        - preprocess_weights_impl(): Inherited (unchanged)
        - move_weights_to_device_impl(): Inherited (unchanged)
        - forward(): OVERRIDDEN - uses softmax instead of sigmoid
    """

    def forward(self, logits: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Forward pass with softmax activation for Qwen3 architecture.

        The only difference from parent is line 22 uses ttnn.softmax instead of ttnn.sigmoid.
        Also adds epsilon (1e-20) to denominator to match PyTorch reference and prevent division by zero.
        """
        r = self._fallback_torch_layer

        if logits.layout != ttnn.TILE_LAYOUT:
            logits = ttnn.to_layout(logits, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        logits = ttnn.reshape(logits, ttnn.Shape((1, 1, logits.shape[0], logits.shape[1])))
        if logits.dtype != ttnn.float32:
            logits_f32 = ttnn.typecast(logits, ttnn.float32)
            ttnn.deallocate(logits)
        else:
            logits_f32 = logits

        # KEY DIFFERENCE: Qwen uses softmax activation instead of sigmoid
        scores_f32 = ttnn.softmax(logits_f32, dim=-1)

        T = scores_f32.shape[2]
        n_experts = scores_f32.shape[3]
        n_group = r.n_group
        experts_per_group = n_experts // n_group

        bias_rm = self._bias_dev
        bias_rep_rm = ttnn.repeat(bias_rm, ttnn.Shape((1, 1, T, 1)))
        bias = ttnn.to_layout(bias_rep_rm, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Convert bias to float32 for stable addition
        if bias.dtype != ttnn.float32:
            bias_f32 = ttnn.typecast(bias, ttnn.float32)
            ttnn.deallocate(bias)
        else:
            bias_f32 = bias

        scores_with_bias_f32 = ttnn.add(scores_f32, bias_f32)
        ttnn.deallocate(bias_f32)

        top_k = r.top_k

        if n_group <= r.topk_group:
            # Pass 1: rough BF16 topk(k+1) to find coarse threshold
            scores_bf16_p1 = ttnn.typecast(scores_with_bias_f32, ttnn.bfloat16)
            rough_vals, _ = ttnn.topk(scores_bf16_p1, k=top_k + 1, dim=3, largest=True, sorted=True)
            ttnn.deallocate(scores_bf16_p1)
            # (k+1)-th value gives coarse threshold.
            rough_thr_bf16 = ttnn.slice(rough_vals, [0, 0, 0, top_k], [1, 1, T, top_k + 1])
            ttnn.deallocate(rough_vals)
            rough_thr_f32 = ttnn.typecast(rough_thr_bf16, ttnn.float32)
            ttnn.deallocate(rough_thr_bf16)
            # Center scores around the decision boundary (float32 precision preserved)
            scores_c1 = ttnn.sub(scores_with_bias_f32, rough_thr_f32)
            ttnn.deallocate(rough_thr_f32)
            ttnn.deallocate(scores_with_bias_f32)

            # Pass 2: refined BF16 topk(k+1) on centered scores
            scores_bf16_p2 = ttnn.typecast(scores_c1, ttnn.bfloat16)
            refined_vals, _ = ttnn.topk(scores_bf16_p2, k=top_k + 1, dim=3, largest=True, sorted=True)
            ttnn.deallocate(scores_bf16_p2)
            # Second threshold is now near 0 -> BF16 step ~ 0.0001 (very precise)
            refined_thr_bf16 = ttnn.slice(refined_vals, [0, 0, 0, top_k], [1, 1, T, top_k + 1])
            ttnn.deallocate(refined_vals)
            refined_thr_f32 = ttnn.typecast(refined_thr_bf16, ttnn.float32)
            ttnn.deallocate(refined_thr_bf16)
            scores_c2 = ttnn.sub(scores_c1, refined_thr_f32)
            ttnn.deallocate(scores_c1)
            ttnn.deallocate(refined_thr_f32)

            # Final pass: exact topk(k) on doubly-centered scores
            scores_bf16_final = ttnn.typecast(scores_c2, ttnn.bfloat16)
            ttnn.deallocate(scores_c2)
            _, topk_expert_idx = ttnn.topk(scores_bf16_final, k=top_k, dim=3, largest=True, sorted=True)
            ttnn.deallocate(scores_bf16_final)
        else:
            # Group-based selection: apply same 3-pass centering after masking
            scores_bf16 = ttnn.typecast(scores_with_bias_f32, ttnn.bfloat16)
            ttnn.deallocate(scores_with_bias_f32)

            # group scores
            grouped = ttnn.reshape(scores_bf16, ttnn.Shape((1, T, n_group, experts_per_group)))
            top2_scores, _ = ttnn.topk(grouped, k=2, dim=3)
            ttnn.deallocate(grouped)
            group_scores = ttnn.sum(top2_scores, dim=3)
            ttnn.deallocate(top2_scores)

            # top-k groups
            _, topk_group_idx = ttnn.topk(group_scores, k=r.topk_group, dim=2)
            ttnn.deallocate(group_scores)

            # group mask via scatter
            input_mask_rm = ttnn.repeat(self._scatter_input_dev, ttnn.Shape((1, 1, T, 1)))
            src_rm = ttnn.repeat(self._scatter_src_dev, ttnn.Shape((1, 1, T, 1)))
            idx_rm = ttnn.to_layout(topk_group_idx, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(topk_group_idx)
            idx_4d = ttnn.unsqueeze(idx_rm, dim=1)
            ttnn.deallocate(idx_rm)
            active_groups_rm = ttnn.scatter(input=input_mask_rm, index=idx_4d, src=src_rm, dim=3)
            ttnn.deallocate(idx_4d)

            # expert mask
            active_groups_rm = ttnn.reshape(active_groups_rm, ttnn.Shape((1, T, n_group, 1)))
            expert_mask_rm = ttnn.repeat(active_groups_rm, ttnn.Shape((1, 1, 1, experts_per_group)))
            ttnn.deallocate(active_groups_rm)
            expert_mask_rm = ttnn.reshape(expert_mask_rm, ttnn.Shape((1, 1, T, n_experts)))
            expert_mask = ttnn.to_layout(expert_mask_rm, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(expert_mask_rm)

            # top-k active experts
            masked_scores = ttnn.mul(scores_bf16, expert_mask)
            ttnn.deallocate(scores_bf16)
            ttnn.deallocate(expert_mask)
            _, topk_expert_idx = ttnn.topk(masked_scores, k=top_k, dim=3)
            ttnn.deallocate(masked_scores)

        # gather raw softmax scores (no bias) for weights
        topk_weights = ttnn.gather(scores_f32, dim=3, index=topk_expert_idx)
        ttnn.deallocate(scores_f32)

        # normalise
        denom = ttnn.sum(topk_weights, dim=3, keepdim=True)
        # KEY DIFFERENCE: Add epsilon to match PyTorch reference and prevent division by zero
        denom = ttnn.add(denom, 1e-20)
        topk_weights = ttnn.div(topk_weights, denom)
        ttnn.deallocate(denom)

        # apply routing scale
        scale_rep_rm = ttnn.repeat(self._scale_dev, ttnn.Shape((1, 1, T, 1)))
        scale_bf16 = ttnn.to_layout(scale_rep_rm, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if scale_bf16.dtype != ttnn.float32:
            scale_f32 = ttnn.typecast(scale_bf16, ttnn.float32)
            ttnn.deallocate(scale_bf16)
        else:
            scale_f32 = scale_bf16
        topk_weights = ttnn.mul(topk_weights, scale_f32)
        ttnn.deallocate(scale_f32)

        # Reshape outputs to (T, top_k).
        topk_expert_idx = ttnn.reshape(topk_expert_idx, ttnn.Shape((T, r.top_k)))
        topk_weights = ttnn.reshape(topk_weights, ttnn.Shape((T, r.top_k)))
        return topk_expert_idx, topk_weights


class TTNNQwenExperts(TTNNExperts):
    """Qwen-specific experts using sparse_matmul with fused w1/w3 projections.

    This subclass overrides preprocess_weights_impl() to pre-reshape weights to 4D
    and forward() to use sparse_matmul with fused gate/up projections. This approach
    eliminates duplicate memory bandwidth by reading the input tensor once instead
    of twice.

    Inheritance:
        - __init__(): Inherited (unchanged)
        - _get_num_experts_per_device(): Inherited (unchanged)
        - from_torch(): Inherited (unchanged)
        - preprocess_weights_impl(): OVERRIDDEN - creates fused w1_w3 weights, shards on dim=1
        - move_weights_to_device_impl(): OVERRIDDEN - simplified (no reshape needed)
        - forward(): OVERRIDDEN - uses fused sparse_matmul
    """

    def preprocess_weights_impl(self):
        """Preprocess expert weights: reshape to 4D on host, convert to bfloat16, TILE_LAYOUT.

        Creates fused w1_w3 weights for single sparse_matmul, eliminating duplicate memory
        bandwidth by reading the input tensor once instead of twice.
        Shape: (num_experts, H, I) -> (1, num_experts, H, 2*I) for fused w1_w3
        """
        # Reshape to 4D on host (torch) before converting to ttnn
        torch_w1_4d = self.torch_w1_proj.unsqueeze(0).to(torch.bfloat16)
        torch_w3_4d = self.torch_w3_proj.unsqueeze(0).to(torch.bfloat16)
        torch_w2_4d = self.torch_w2_proj.unsqueeze(0).to(torch.bfloat16)

        # Create fused w1_w3 weights for single sparse_matmul
        torch_w1_w3_fused = torch.cat([torch_w1_4d, torch_w3_4d], dim=-1)
        self.tt_w1_w3_proj = ttnn.from_torch(
            torch_w1_w3_fused,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=1),
        )
        del torch_w1_w3_fused

        # w2 for down projection
        self.tt_w2_proj = ttnn.from_torch(
            torch_w2_4d,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=1),
        )

        del self.torch_w1_proj
        del self.torch_w3_proj
        del self.torch_w2_proj

    def move_weights_to_device_impl(self):
        """Move preprocessed weights to device - simplified since weights are already 4D.

        Moves fused w1_w3 weights and w2 weights to device, and creates program configs
        for sparse_matmul operations.
        """
        self.num_experts_per_device = self._get_num_experts_per_device(self.config, self.device)
        self.num_devices = self.device.get_num_devices()
        self.num_dispatch_devices = self.device.get_num_devices()

        # Move fused w1_w3 weights to device
        self.tt_w1_w3_proj = ttnn.to_device(self.tt_w1_w3_proj, self.device)

        # Move w2 weights to device
        self.tt_w2_proj = ttnn.to_device(self.tt_w2_proj, self.device)

        # Create expert mapping tensors for all-to-all ops
        self.expert_mapping_tensors = ttnn.from_torch(
            torch.eye(self.num_devices, dtype=torch.int32)
            .repeat_interleave(self.num_experts_per_device, dim=0)
            .unsqueeze(0)
            .unsqueeze(0),
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            dtype=ttnn.uint16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # Create remap topk mask for expert token remap
        self.remap_topk_mask = ttnn.from_torch(
            torch.ones((1, self.num_dispatch_devices, 1, self.num_experts), dtype=torch.bfloat16),
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # Program configs for sparse_matmul operations
        hidden_tiles = self.hidden_size // ttnn.TILE_SIZE
        intermediate_tiles = self.intermediate_size // ttnn.TILE_SIZE

        # Fused gate/up program config (output is 2*intermediate_size)
        self._fused_gate_up_program_config = _make_sparse_matmul_program_config(
            device=self.device,
            out_features=int(self.intermediate_size * 2),  # 2*I for fused output
            in0_block_w=min(4, hidden_tiles),
            per_core_M=1,
        )

        # Down projection program config
        self._down_program_config = _make_sparse_matmul_program_config(
            device=self.device,
            out_features=int(self.hidden_size),
            in0_block_w=min(4, intermediate_tiles),
            per_core_M=1,
        )
        self._expert_compute_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    @run_on_devices(DeviceArch.T3K)
    def forward(
        self, x: ttnn.Tensor, topk_experts_indices: ttnn.Tensor, topk_experts_weights: ttnn.Tensor
    ) -> ttnn.Tensor:
        """Execute expert pipeline using fused sparse_matmul.

        Uses sparse_matmul with fused W1/W3 weights to compute only activated experts.
        This eliminates duplicate memory bandwidth by reading the input tensor once.

        Args:
            x: Input tensor of shape (batch_size_per_device, 1, seq_len, hidden_size)
            topk_experts_indices: Expert indices of shape (batch_size_per_device*seq_len, num_experts_per_tok)
            topk_experts_weights: Expert weights of shape (batch_size_per_device*seq_len, num_experts_per_tok)

        Returns:
            Output tensor of shape (1, 1, batch_size_per_device*seq_len, hidden_size)
        """
        # Extract dimensions
        batch_size_per_device = x.shape[0]
        seq_len = x.shape[2]
        batch_size = batch_size_per_device * self.num_dispatch_devices

        # Decode mode detection for L1 memory optimization
        is_decode_mode = seq_len == 1
        decode_memory_config = ttnn.L1_MEMORY_CONFIG if is_decode_mode else ttnn.DRAM_MEMORY_CONFIG
        tokens_per_device = batch_size_per_device * seq_len

        # Store original num_tokens for unpadding later
        original_num_tokens = tokens_per_device

        if topk_experts_indices.dtype != ttnn.uint16:
            if topk_experts_indices.layout != ttnn.TILE_LAYOUT:
                topk_experts_indices = ttnn.to_layout(
                    topk_experts_indices,
                    ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            topk_experts_indices = ttnn.typecast(topk_experts_indices, ttnn.uint16)

        # Padding: Use SPARSITY_BLOCK_SIZE for sparse matmul
        pad_block_size = SPARSITY_BLOCK_SIZE

        total_tokens = tokens_per_device * self.num_dispatch_devices
        pad_amount = 0

        if total_tokens % pad_block_size != 0:
            padded_tokens_per_device = (
                ((total_tokens + pad_block_size - 1) // pad_block_size) * pad_block_size // self.num_dispatch_devices
            )
            if padded_tokens_per_device * self.num_dispatch_devices < total_tokens:
                padded_tokens_per_device += 1
            pad_amount = padded_tokens_per_device - tokens_per_device

            if pad_amount > 0:
                # Pad x: (batch, 1, seq_len, hidden) -> (batch, 1, padded_seq_len, hidden)
                x = ttnn.pad(x, padding=((0, 0), (0, 0), (0, pad_amount), (0, 0)), value=0.0)

                # Pad indices: (tokens, k) -> (padded_tokens, k) with value 0 (route to expert 0)
                topk_experts_indices = ttnn.pad(topk_experts_indices, padding=((0, pad_amount), (0, 0)), value=0)

                # Pad weights: (tokens, k) -> (padded_tokens, k) with value 0.0 (zero weight = no contribution)
                topk_experts_weights = ttnn.pad(topk_experts_weights, padding=((0, pad_amount), (0, 0)), value=0.0)

                # Update tokens_per_device and seq_len for padded tensors
                tokens_per_device = padded_tokens_per_device
                seq_len = tokens_per_device // batch_size_per_device
                total_tokens = tokens_per_device * self.num_dispatch_devices

        x = ttnn.typecast(x, ttnn.bfloat16)

        # STEP 1: PREPARE INPUTS FOR ALL_TO_ALL_DISPATCH
        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x_rm = ttnn.reshape(x_rm, shape=(batch_size_per_device, 1, seq_len, self.hidden_size))

        topk_experts_indices_rm = ttnn.to_layout(topk_experts_indices, ttnn.ROW_MAJOR_LAYOUT)
        topk_experts_indices_rm = ttnn.reshape(
            topk_experts_indices_rm, shape=(batch_size_per_device, 1, seq_len, self.num_experts_per_tok)
        )

        # STEP 2: ALL_TO_ALL_DISPATCH - Route tokens to expert devices
        all_to_all_dispatch_output, all_to_all_dispatch_metadata = ttnn.all_to_all_dispatch(
            x_rm,
            topk_experts_indices_rm,
            self.expert_mapping_tensors,
            cluster_axis=1,
            memory_config=decode_memory_config,
        )
        ttnn.deallocate(x_rm)
        ttnn.deallocate(topk_experts_indices_rm)

        # STEP 3: PREPARE DISPATCH OUTPUT FOR EXPERT COMPUTATION
        post_dispatch = ttnn.reshape(all_to_all_dispatch_output, shape=(1, 1, total_tokens, self.hidden_size))
        post_dispatch = ttnn.to_layout(post_dispatch, ttnn.TILE_LAYOUT)

        # STEP 4: EXPERT COMPUTATION using fused sparse_matmul
        # NO ttnn.repeat - this is the key optimization!
        # Generate sparsity tensor to only compute activated experts

        num_tokens = total_tokens

        # Generate sparsity tensor
        remap_topk_mask_expanded = ttnn.repeat(self.remap_topk_mask, ttnn.Shape((1, batch_size_per_device, 1, 1)))
        _, sparsity_t = ttnn.moe_expert_token_remap(
            remap_topk_mask_expanded,
            self.expert_mapping_tensors,
            all_to_all_dispatch_metadata,
            reduction_size=SPARSITY_BLOCK_SIZE,
        )

        num_sparse_blocks = num_tokens // SPARSITY_BLOCK_SIZE
        x_sparse = ttnn.reshape(post_dispatch, shape=(1, num_sparse_blocks, SPARSITY_BLOCK_SIZE, self.hidden_size))

        # Fused gate/up projection - single sparse_matmul for both w1 (gate) and w3 (up)
        # Output shape: [1, num_sparse_blocks, SPARSITY_BLOCK_SIZE, 2*intermediate_size]
        w1_w3_out = ttnn.sparse_matmul(
            x_sparse,
            self.tt_w1_w3_proj,  # Fused weights: [1, E, H, 2*I]
            sparsity=sparsity_t,
            output_tile=ttnn.Tile([SPARSITY_BLOCK_SIZE, ttnn.TILE_SIZE]),
            program_config=self._fused_gate_up_program_config,
            compute_kernel_config=self._expert_compute_cfg,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            memory_config=decode_memory_config,
        )
        ttnn.deallocate(x_sparse)

        # Split fused output into w1 (gate) and w3 (up) components
        # Shape: [1, num_sparse_blocks, SPARSITY_BLOCK_SIZE, 2*I] -> two [1, num_sparse_blocks, SPARSITY_BLOCK_SIZE, I]
        intermediate_size = self.intermediate_size

        # Get actual tensor shape to build correct slice indices
        actual_shape = list(w1_w3_out.shape)
        rank = len(actual_shape)

        # Build slice indices dynamically - only slice the last dimension
        # First half: w1 (gate projection)
        slice_start_w1 = [0] * rank
        slice_end_w1 = list(actual_shape)
        slice_end_w1[-1] = intermediate_size

        w1_out = ttnn.slice(
            w1_w3_out,
            slice_start=slice_start_w1,
            slice_end=slice_end_w1,
        )

        # Second half: w3 (up projection)
        slice_start_w3 = [0] * rank
        slice_start_w3[-1] = intermediate_size
        slice_end_w3 = list(actual_shape)

        w3_out = ttnn.slice(
            w1_w3_out,
            slice_start=slice_start_w3,
            slice_end=slice_end_w3,
        )
        ttnn.deallocate(w1_w3_out)

        # SwiGLU activation: silu(gate) * up
        w1_activated = ttnn.silu(w1_out, memory_config=decode_memory_config)
        ttnn.deallocate(w1_out)
        intermediate = ttnn.mul(w1_activated, w3_out, memory_config=decode_memory_config)
        ttnn.deallocate(w1_activated)
        ttnn.deallocate(w3_out)

        intermediate = ttnn.squeeze(intermediate, 0)
        intermediate = ttnn.squeeze(intermediate, 1)

        # Down projection (w2) with sparse_matmul
        expert_output = ttnn.sparse_matmul(
            intermediate,
            self.tt_w2_proj,
            sparsity=sparsity_t,
            output_tile=ttnn.Tile([SPARSITY_BLOCK_SIZE, ttnn.TILE_SIZE]),
            program_config=self._down_program_config,
            compute_kernel_config=self._expert_compute_cfg,
            is_input_a_sparse=True,
            is_input_b_sparse=False,
            memory_config=decode_memory_config,
        )
        ttnn.deallocate(intermediate)

        # Reshape to expected format
        expert_output = ttnn.permute(expert_output, (1, 0, 2, 3))
        expert_output = ttnn.reshape(
            expert_output, shape=(1, self.num_experts_per_device, num_tokens, self.hidden_size)
        )

        ttnn.deallocate(post_dispatch)

        # STEP 5: PREPARE EXPERT OUTPUT FOR ALL_TO_ALL_COMBINE
        expert_output = ttnn.reshape(
            expert_output,
            shape=(self.num_experts_per_device, 1, total_tokens, self.hidden_size),
        )
        expert_output = ttnn.to_layout(expert_output, ttnn.ROW_MAJOR_LAYOUT)

        # Reshape to match combine expected format
        expert_output = ttnn.reshape(
            expert_output, shape=(self.num_experts_per_device, batch_size, seq_len, self.hidden_size)
        )

        # STEP 6: ALL_TO_ALL_COMBINE - Route expert outputs back to token positions
        combined_output = ttnn.all_to_all_combine(
            expert_output,
            all_to_all_dispatch_metadata,
            self.expert_mapping_tensors,
            cluster_axis=1,
            memory_config=decode_memory_config,
        )
        ttnn.deallocate(expert_output)
        ttnn.deallocate(all_to_all_dispatch_metadata)

        # STEP 7: APPLY ROUTING WEIGHTS AND REDUCE ACROSS EXPERTS
        actual_shape = list(combined_output.shape)
        if len(actual_shape) == 5:
            combined_output = ttnn.reshape(combined_output, shape=(self.num_experts_per_tok, 1, -1, self.hidden_size))
        else:
            combined_output = ttnn.reshape(
                combined_output, shape=(self.num_experts_per_tok, 1, tokens_per_device, self.hidden_size)
            )
        combined_output = ttnn.to_layout(combined_output, ttnn.TILE_LAYOUT)

        # Prepare routing weights for broadcasting
        topk_experts_weights_rm = ttnn.to_layout(topk_experts_weights, ttnn.ROW_MAJOR_LAYOUT)
        # topk_experts_weights shape: [tokens, K] -> transpose to [K, tokens]
        topk_experts_weights_rm = ttnn.permute(topk_experts_weights_rm, (1, 0))
        # Now [K, tokens] -> [K, 1, tokens, 1] for broadcasting
        topk_experts_weights_rm = ttnn.unsqueeze(topk_experts_weights_rm, 1)
        topk_experts_weights_rm = ttnn.unsqueeze(topk_experts_weights_rm, 3)
        topk_experts_weights_tile = ttnn.to_layout(topk_experts_weights_rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(topk_experts_weights_rm)

        # Broadcast multiply: [K, 1, tokens, 1] * [K, 1, tokens, hidden] -> [K, 1, tokens, hidden]
        weighted_output = ttnn.mul(
            combined_output,
            topk_experts_weights_tile,
        )
        ttnn.deallocate(combined_output)
        ttnn.deallocate(topk_experts_weights_tile)

        # Sum over experts dimension
        final_output = ttnn.sum(weighted_output, dim=0, keepdim=True)
        ttnn.deallocate(weighted_output)

        # UNPAD: Remove padding added at the start to restore original token count
        if original_num_tokens != tokens_per_device:
            final_output = ttnn.slice(
                final_output,
                slice_start=[0, 0, 0, 0],
                slice_end=[1, 1, original_num_tokens, self.hidden_size],
                slice_step=[1, 1, 1, 1],
            )

        return final_output


class TTNNQwen3MoE(TTNNMoE):
    """TTNN MoE for Qwen3.5-35B-A3B architecture with 256 experts, top-8 routing.

    Handles the Qwen3_5MoeSparseMoeBlock structure:
    - gate: Qwen3_5MoeTopKRouter
    - experts: Qwen3_5MoeExperts (with gate_up_proj and down_proj)
    - shared_expert: Qwen3_5MoeMLP (singular, not plural like GLM)
    - shared_expert_gate: Optional gating for shared expert output

    Inheritance:
        - __init__(): Inherited (unchanged)
        - from_torch(): OVERRIDDEN - handles Qwen-specific structure
        - preprocess_weights_impl(): OVERRIDDEN - adds shared_expert_gate
        - move_weights_to_device_impl(): OVERRIDDEN - adds shared_expert_gate
        - forward(): OVERRIDDEN - adds shared_expert_gate support
        - _adapt_config(): NEW static method
        - _consolidate_experts(): NEW static method
        - _adapt_gate(): NEW static method
    """

    @classmethod
    def from_torch(cls, torch_moe):
        """Create TTNNQwen3MoE from PyTorch Qwen3_5MoeSparseMoeBlock module.

        KEY DIFFERENCES from parent:
        1. Gets config from torch_moe.experts.config (not torch_moe.config)
        2. Uses TTNNQwenMoERouterDecode instead of TTNNMoERouterDecode
        3. Uses TTNNQwenExperts instead of TTNNExperts
        4. Accesses shared_expert (singular) instead of shared_experts (plural)
        5. Handles optional shared_expert_gate
        """
        # 1. Adapt config to match expected structure
        adapted_config = cls._adapt_config(torch_moe)

        # 2. Consolidate experts from Qwen3 format
        consolidated_experts = cls._consolidate_experts(torch_moe.experts, adapted_config)

        # 3. Adapt gate to match expected structure
        adapted_gate = cls._adapt_gate(torch_moe.gate)

        # 4. Create module instance
        module = cls(adapted_config)
        module._fallback_torch_layer = torch_moe

        # 5. Initialize submodules using parent's pattern
        module.gate = TTNNGlm4MoeTopkRouter.from_parameters(adapted_gate.weight, adapted_gate.e_score_correction_bias)

        # KEY DIFFERENCE: Use Qwen-specific router with softmax activation
        module.route_tokens_to_experts = TTNNQwenMoERouterDecode.from_torch(
            Glm4MoeRouteTokenToExperts(
                adapted_gate.e_score_correction_bias,
                adapted_config.n_routed_experts,
                adapted_config.n_group,
                adapted_config.topk_group,
                adapted_config.num_experts_per_tok,
                True,  # norm_topk_prob (Qwen3 uses normalized probabilities)
                adapted_config.routed_scaling_factor,
            )
        )

        # KEY DIFFERENCE: Use Qwen-specific experts with batched matmul
        # Check if CPU experts fallback is enabled for debugging
        use_cpu_experts = os.environ.get("TT_QWEN_CPU_EXPERTS", "0").lower() in ("1", "true", "yes")
        if use_cpu_experts:
            # Keep original PyTorch experts for CPU execution (for debugging accuracy issues)
            module.experts = torch_moe.experts
            module._use_cpu_experts = True
            print("[DEBUG] TT_QWEN_CPU_EXPERTS=1: Using CPU fallback for experts")
        else:
            module.experts = TTNNQwenExperts.from_torch(consolidated_experts)
            module._use_cpu_experts = False

        # KEY DIFFERENCE: Qwen3 uses singular "shared_expert" not "shared_experts"
        module.shared_experts = TTNNGlm4MoeMLP.from_torch(torch_moe.shared_expert)

        # Store replicated gate weight for preprocessing
        module._gate_weight_torch = adapted_gate.weight.to(torch.bfloat16)

        # KEY DIFFERENCE: Store shared_expert_gate weight for gating the shared expert output
        if hasattr(torch_moe, "shared_expert_gate"):
            module._shared_expert_gate_weight_torch = torch_moe.shared_expert_gate.weight.to(torch.bfloat16)
        else:
            module._shared_expert_gate_weight_torch = None

        return module

    @staticmethod
    def _adapt_config(torch_moe):
        """Adapt Qwen3 MoE config to match Glm4MoeConfig structure.

        KEY DIFFERENCES:
        1. Gets config from torch_moe.experts.config (not torch_moe.config)
        2. num_experts -> n_routed_experts
        3. Provides default n_group=4, topk_group=2 (Qwen3 doesn't have these)
        """
        original_config = torch_moe.experts.config

        class AdaptedConfig:
            pass

        config = AdaptedConfig()

        # Map Qwen3 attributes to Glm4MoeConfig naming
        config.hidden_size = original_config.hidden_size
        config.moe_intermediate_size = original_config.moe_intermediate_size
        config.num_experts_per_tok = original_config.num_experts_per_tok

        # Key difference: num_experts -> n_routed_experts
        config.n_routed_experts = original_config.num_experts

        # Qwen3 doesn't have n_group and topk_group - use defaults that work
        # For 256 experts with top-8 routing:
        # n_group=4 means 256/4=64 experts per group
        # topk_group=2 means select top-2 groups
        config.n_group = 4
        config.topk_group = 2

        # Scaling factor - use 1.0 if not specified
        config.routed_scaling_factor = getattr(original_config, "routed_scaling_factor", 1.0)

        # Additional attributes needed by TTNNExperts
        config.hidden_act = getattr(original_config, "hidden_act", "silu")

        return config

    @staticmethod
    def _consolidate_experts(qwen_experts, config):
        """Adapt Qwen3_5MoeExperts to the structure expected by TTNNExperts.from_torch().

        Qwen3 experts already have the right shape:
        - gate_up_proj: [num_experts, 2*intermediate_size, hidden_size]
        - down_proj: [num_experts, hidden_size, intermediate_size]

        This method wraps them in an object with the expected attributes.
        """

        class ConsolidatedExperts:
            pass

        consolidated = ConsolidatedExperts()

        # Qwen3 gate_up_proj already has the right shape
        consolidated.gate_up_proj = qwen_experts.gate_up_proj
        consolidated.down_proj = qwen_experts.down_proj
        consolidated.config = config

        return consolidated

    @staticmethod
    def _adapt_gate(qwen_gate):
        """Adapt Qwen3_5MoeTopKRouter to match expected structure with e_score_correction_bias.

        KEY DIFFERENCE: Qwen3 router may not have e_score_correction_bias, so we create zeros.
        """

        class AdaptedGate:
            pass

        adapted = AdaptedGate()
        adapted.weight = qwen_gate.weight

        # Qwen3 router doesn't have e_score_correction_bias - create zeros
        if hasattr(qwen_gate, "e_score_correction_bias"):
            adapted.e_score_correction_bias = qwen_gate.e_score_correction_bias
        else:
            # Create zeros tensor with shape [num_experts]
            adapted.e_score_correction_bias = torch.zeros(qwen_gate.weight.shape[0])

        return adapted

    def preprocess_weights_impl(self):
        """Preprocess weights including shared_expert_gate.

        Extends parent to also preprocess shared_expert_gate weight if present.
        """
        # Call parent preprocess for gate weight and submodules
        super().preprocess_weights_impl()

        # KEY DIFFERENCE: Preprocess shared_expert_gate weight if present
        if self._shared_expert_gate_weight_torch is not None:
            # Shape: [1, hidden_size] -> transpose to [hidden_size, 1] for linear
            self._shared_expert_gate_tt_host = ttnn.from_torch(
                self._shared_expert_gate_weight_torch.T.contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )

    def move_weights_to_device_impl(self):
        """Move weights to device including shared_expert_gate.

        Extends parent to also move shared_expert_gate weight to device.
        """
        # Call parent move_weights_to_device
        super().move_weights_to_device_impl()

        # KEY DIFFERENCE: Move shared_expert_gate weight to device with replication
        if self._shared_expert_gate_weight_torch is not None:
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None
            gate_torch = ttnn.to_torch(self._shared_expert_gate_tt_host)
            self._shared_expert_gate_tt = ttnn.from_torch(
                gate_torch,
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mesh_mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

    def set_output_tensors_config_impl(self, output_tensors):
        """Set output tensor config for col-sharded output.

        The reduce_scatter output is col-sharded (each device has [batch, seq, hidden_size/8]).
        We need to use ConcatMeshToTensor on dim=-1 to concatenate the shards.
        """

        def set_col_sharded_config(e):
            if isinstance(e, TorchTTNNTensor) and e.ttnn_tensor is not None:
                if self.device is not None and self.device.get_num_devices() > 1:
                    # Use ConcatMeshToTensor on dim=-1 only (not batch dim)
                    mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=-1)
                    mesh_mapper = ttnn.ShardTensorToMesh(self.device, dim=-1)

                    def logical_shape_for_col_sharded(shape):
                        """Compute logical shape by multiplying last dim by num_devices."""
                        shape_list = list(shape)
                        num_devices = self.device.get_num_devices()
                        shape_list[-1] = shape_list[-1] * num_devices
                        return tuple(shape_list)

                    config = DistributedTensorConfig(
                        mesh_mapper=mesh_mapper,
                        mesh_composer=mesh_composer,
                        logical_shape_fn=logical_shape_for_col_sharded,
                    )
                    e.set_distributed_tensor_config(config)
            return e

        # Only set col-sharded config if in distributed mode
        if self.device is None or self.device.get_num_devices() <= 1:
            return super().set_output_tensors_config_impl(output_tensors)

        return tree_map(set_col_sharded_config, output_tensors)

    @run_on_devices(DeviceArch.T3K)
    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass with shared_expert_gate support.

        KEY DIFFERENCE: Applies sigmoid gating to shared expert output:
            shared_output = sigmoid(linear(x, gate_weight)) * shared_expert(x)
        """
        self.num_devices = self.device.get_num_devices()
        self.num_dispatch_devices = self.device.get_num_devices()
        self.num_experts_per_device = even_int_div(self.config.n_routed_experts, self.num_devices)
        # Store original input for shared experts
        residual = x

        # 1. All-gather to revert tensor parallelism
        x = ttnn.all_gather(
            x,
            dim=-1,
            num_links=1,
            topology=ttnn.Topology.Linear,
        )

        # 2. MoE gate routing
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if x.dtype != ttnn.float32:
            x_f32 = ttnn.typecast(x, ttnn.float32)
        else:
            x_f32 = x
        router_logits_f32 = ttnn.linear(
            x_f32,
            self._gate_weight_tt,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )
        if x_f32 is not x:
            ttnn.deallocate(x_f32)
        # Convert back to bfloat16 for router (ttnn.softmax / ttnn.topk require bf16)
        router_logits = ttnn.typecast(router_logits_f32, ttnn.bfloat16)
        ttnn.deallocate(router_logits_f32)

        T = router_logits.shape[-2]
        router_logits = ttnn.reshape(router_logits, ttnn.Shape((T, self.n_routed_experts)))

        topk_experts_indices, topk_experts_weights = self.route_tokens_to_experts(router_logits)

        x = ttnn.unsqueeze(x, 1)  # Add experts dimension for compatibility with experts module

        # 3. Experts handle dispatch -> compute -> combine -> weight
        if getattr(self, "_use_cpu_experts", False):
            # CPU EXPERTS FALLBACK: Convert to PyTorch, run experts, convert back
            # This is for debugging - bypasses TTNN expert computation
            x_cpu = ttnn.squeeze(x, 1)  # Remove experts dimension for CPU experts
            x_cpu = ttnn.to_layout(x_cpu, ttnn.ROW_MAJOR_LAYOUT)

            # IMPORTANT: After all_gather on hidden dim, all devices have IDENTICAL data.
            # ConcatMeshToTensor concatenates all device copies, creating num_devices x copies.
            # We only need ONE copy, so slice to take only the first device's portion.
            num_devices = self.device.get_num_devices()

            x_torch_full = ttnn.to_torch(x_cpu, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
            x_batch_per_device = x_torch_full.shape[0] // num_devices
            x_torch = x_torch_full[:x_batch_per_device]  # Take only first device's data
            x_torch = x_torch.view(-1, x_torch.shape[-1])  # Flatten to (tokens, hidden)

            # Convert indices and weights to PyTorch
            # Extract underlying TTNN tensor from TorchTTNNTensor wrapper
            idx_ttnn = topk_experts_indices
            idx_rm = ttnn.to_layout(idx_ttnn, ttnn.ROW_MAJOR_LAYOUT)
            idx_torch_full = ttnn.to_torch(idx_rm, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
            idx_batch_per_device = idx_torch_full.shape[0] // num_devices
            idx_torch = idx_torch_full[:idx_batch_per_device]  # Take only first device's data
            idx_torch = idx_torch.to(torch.int64)

            wgt_ttnn = topk_experts_weights
            wgt_rm = ttnn.to_layout(wgt_ttnn, ttnn.ROW_MAJOR_LAYOUT)
            wgt_torch_full = ttnn.to_torch(wgt_rm, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
            wgt_batch_per_device = wgt_torch_full.shape[0] // num_devices
            wgt_torch = wgt_torch_full[:wgt_batch_per_device]  # Take only first device's data

            # Call PyTorch experts
            routed_torch = self.experts(x_torch, idx_torch, wgt_torch)
            routed_torch = routed_torch.view(1, 1, -1, routed_torch.shape[-1])

            # Convert back to TTNN (replicate across devices since we'll reduce-scatter)
            routed_output = ttnn.from_torch(
                routed_torch.to(torch.bfloat16),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            routed_output = self.experts(x, topk_experts_indices, topk_experts_weights)

        # 4. Reduce-scatter final output.
        n_rs = self.device.get_num_devices()  # devices along cluster_axis=1
        # Extract underlying TTNN tensor - handle both wrapped and raw tensors
        routed_out = routed_output
        if n_rs > 1:
            routed_out = ttnn.mul(routed_out, 1.0 / float(n_rs))
        routed_output = ttnn.reduce_scatter(
            routed_out,
            dim=3,
            num_links=1,
            cluster_axis=1,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # 5. Add shared experts output with gating
        shared_output = self.shared_experts(residual)

        # KEY DIFFERENCE: Apply shared expert gate if present
        if self._shared_expert_gate_weight_torch is not None:
            # Compute gate values: sigmoid(linear(x_gathered, gate_weight))
            x_for_gate = ttnn.squeeze(x, 1)  # Remove experts dimension added earlier
            if x_for_gate.layout != ttnn.TILE_LAYOUT:
                x_for_gate = ttnn.to_layout(x_for_gate, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            gate_logits = ttnn.linear(
                x_for_gate,
                self._shared_expert_gate_tt,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            gate_values = ttnn.sigmoid(gate_logits)
            ttnn.deallocate(gate_logits)
            # Gate the shared expert output - broadcast gate_values (shape [..., 1]) to shared_output shape
            shared_output_gated = ttnn.mul(shared_output, gate_values)
            ttnn.deallocate(gate_values)
            output = ttnn.add(routed_output, shared_output_gated)
            ttnn.deallocate(shared_output_gated)
        else:
            output = ttnn.add(routed_output, shared_output)

        output = ttnn.squeeze(output, 1)  # Remove experts dimension

        return output
