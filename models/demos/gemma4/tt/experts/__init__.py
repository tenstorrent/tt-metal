# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 Routed Experts module.

Decode (seq_len=1): on-device via sparse_matmul with top-k sparsity
Prefill (seq_len>1): on-device via sparse_matmul with all-ones sparsity (gpt_oss pattern)
"""

import os

import ttnn

from .decode import decode_forward
from .prefill import create_prefill_sparsity, prefill_forward
from .weights import ExpertWeights, load_expert_weights
from models.demos.gemma4.tt.ccl import ccl_allreduce, ccl_lane_gather_rows, ccl_lane_scatter_rows


class Gemma4ExpertConfig:
    """Configuration for the routed experts, derived from HF config."""

    def __init__(self, hf_config):
        self.hidden_size = hf_config.hidden_size
        self.num_experts = hf_config.num_experts
        self.top_k = hf_config.top_k_experts
        self.moe_intermediate_size = hf_config.moe_intermediate_size


class Gemma4Experts:
    def __init__(
        self,
        mesh_device,
        config,
        state_dict,
        ccl_manager,
        mesh_config,
        program_config,
        weight_dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
        layer_idx=0,
        num_layers=1,
    ):
        self.mesh_device = mesh_device
        self.config = config
        self.ccl_manager = ccl_manager
        self.mesh_config = mesh_config

        # Load weights to device for sparse_matmul
        self.weights = load_expert_weights(
            mesh_device=mesh_device,
            config=config,
            state_dict=state_dict,
            mesh_config=mesh_config,
            weight_dtype=weight_dtype,
            tensor_cache_path=tensor_cache_path,
        )
        # Whole-expert sharding: every chip holds E / (ep * tp) experts at full width (ep over the
        # lane axis, tp over the other), and the dense routing is partitioned to its expert
        # columns on device. The ep-axis combine sums the lanes' partial outputs, the tp-axis
        # all-reduce the columns' partials, exactly as before; only the per-chip expert set and
        # the tensor widths changed (Qwen3.6 MoE layout).
        self.ep = self.weights.ep
        self.num_local_experts = self.weights.num_local_experts
        self.local_expert_mask = None
        # Prefill sparsity: all-ones over the local experts (reused for every prefill call)
        self.prefill_sparsity = create_prefill_sparsity(mesh_device, self.num_local_experts)
        # Optional token-dispatch prefill (DeepSeek dispatch/combine + fused GeGLU experts): prompts of at
        # least GEMMA4_MOE_DISPATCH_MIN_TOKENS rows route tokens to the chips owning their experts instead of
        # running every chip's experts on every token; shorter calls (the traced decode frame) keep the
        # sparse_matmul path above.
        self.dispatch_experts = None
        self.dispatch_min_tokens = int(os.environ.get("GEMMA4_MOE_DISPATCH_MIN_TOKENS", "256"))
        is_mesh = hasattr(mesh_device, "shape")
        if os.environ.get("GEMMA4_MOE_DISPATCH_PREFILL", "0") == "1" and is_mesh and mesh_device.get_num_devices() > 1:
            # imported here: the DeepSeek stack needs the GeluTanhGlu ttnn build, which the default path does not
            from models.demos.gemma4.tt.experts.dispatch_prefill import Gemma4DispatchExperts

            placeholder = bool(getattr(state_dict, "is_placeholder", False))
            hf_list = (
                None
                if placeholder
                else Gemma4DispatchExperts.hf_expert_list(state_dict["gate_up_proj"], state_dict["down_proj"])
            )
            self.dispatch_experts = Gemma4DispatchExperts(
                mesh_device, config, hf_list, tensor_cache_path=tensor_cache_path, weights_dtype=weight_dtype
            )
        # Optional decode MoE kernel for the lane decode frame (token dispatch within the column, moe_compute):
        # calls of at most the frame's rows take it; longer prefill calls keep the paths above.
        self.moe_decode = None
        if os.environ.get("GEMMA4_MOE_DECODE_KERNEL", "0") == "1" and is_mesh and mesh_device.get_num_devices() > 1:
            from models.demos.gemma4.tt.experts.moe_decode import Gemma4MoEDecode

            assert not getattr(state_dict, "is_placeholder", False), "the decode MoE kernel needs real expert weights"
            self.moe_decode = Gemma4MoEDecode(
                mesh_device,
                config,
                state_dict["gate_up_proj"],
                state_dict["down_proj"],
                layer_idx=layer_idx,
                num_layers=num_layers,
            )

    def __call__(self, hidden_states, dense_routing):
        lane_sharded = bool(self.mesh_config is not None and getattr(self.mesh_config, "lane_sharded", False))
        # With EP=1 every lane holds the full expert set, so lane-local rows need no exchange.
        if lane_sharded and self.ep > 1:
            # Lanes hold different rows: gather every lane's rows (and their routing), run the
            # local experts on all of them, then reduce_scatter the rows back to their lanes
            # (summing the ep partials) and all-reduce over tp -- the SharedMLP choreography.
            # Decode hands each lane 1..32 rows; the gathered tensor then has lanes*rows rows and
            # takes the (tile-granular) prefill path, so pad every lane to a tile of rows first and
            # slice the lane's own rows back after the scatter. Zero routing on the pad rows keeps
            # them out of the expert sums.
            rows = hidden_states.shape[2]
            pad_rows = (-rows) % 32
            if pad_rows:
                hidden_states = ttnn.pad(hidden_states, [(0, 0), (0, 0), (0, pad_rows), (0, 0)], value=0.0)
                dense_routing = ttnn.pad(dense_routing, [(0, 0), (0, 0), (0, pad_rows), (0, 0)], value=0.0)
            hidden_states = ccl_lane_gather_rows(hidden_states, self.mesh_config)
            dense_routing = ccl_lane_gather_rows(dense_routing, self.mesh_config)
            owner = getattr(self.mesh_config, "_g4_single_lane_prefill_owner", None)
            if owner is not None and os.environ.get("GEMMA4_EP_LANE_FASTPATH", "1") != "0":
                # Single-user prefill: only the owner lane's rows are real. Every lane still needs
                # them (the experts are sharded over the lanes), but running the local experts on
                # the other lanes' scratch rows would be 1 - 1/lanes wasted work. Slice the owner's
                # block out of the gather, compute 1x rows, and all-reduce the ep partials so every
                # lane ends up holding the owner's result (the other lanes' rows are discarded
                # downstream anyway).
                block = int(hidden_states.shape[2]) // self.mesh_config.lanes
                lo, hi = owner * block, (owner + 1) * block
                hidden_owner = ttnn.slice(hidden_states, [0, 0, lo, 0], [1, 1, hi, hidden_states.shape[-1]])
                routing_owner = ttnn.slice(dense_routing, [0, 0, lo, 0], [1, 1, hi, dense_routing.shape[-1]])
                hidden_states.deallocate(True)
                dense_routing.deallocate(True)
                if self._use_decode_kernel(hidden_owner):
                    out = self.moe_decode.forward(hidden_owner, routing_owner)  # full sum on every chip
                elif self._use_dispatch(hidden_owner):
                    # already the full sum on every chip
                    out = self.dispatch_experts.forward(hidden_owner, routing_owner)
                else:
                    out = self._forward(hidden_owner, routing_owner, combine=False)
                    out = ccl_allreduce(
                        out, self.mesh_config, self.ccl_manager, axis=self.mesh_config.ep_axis, group_size=self.ep
                    )
                    if self.mesh_config.tp > 1:
                        out = ccl_allreduce(out, self.mesh_config, self.ccl_manager)
            elif self._use_dispatch(hidden_states) or self._use_decode_kernel(hidden_states):
                # full sum of every lane's rows on every chip: each lane keeps its own block of rows
                if self._use_decode_kernel(hidden_states):
                    out = self.moe_decode.forward(hidden_states, dense_routing)
                else:
                    out = self.dispatch_experts.forward(hidden_states, dense_routing)
                out = ttnn.mesh_partition(out, dim=2, cluster_axis=self.mesh_config.ep_axis)
            else:
                out = self._forward(hidden_states, dense_routing, combine=False)
                out = ccl_lane_scatter_rows(out, self.mesh_config)
            if pad_rows:
                out = ttnn.slice(out, [0, 0, 0, 0], [1, 1, rows, out.shape[-1]])
            return out
        if self._use_decode_kernel(hidden_states):
            return self.moe_decode.forward(hidden_states, dense_routing)
        if self._use_dispatch(hidden_states):
            return self.dispatch_experts.forward(hidden_states, dense_routing)
        return self._forward(hidden_states, dense_routing, combine=True)

    def _use_decode_kernel(self, hidden_states):
        return self.moe_decode is not None and 1 < int(hidden_states.shape[2]) <= self.moe_decode.frame_rows

    def _use_dispatch(self, hidden_states):
        return self.dispatch_experts is not None and int(hidden_states.shape[2]) >= self.dispatch_min_tokens

    def _forward(self, hidden_states, dense_routing, combine=True):
        """
        Expert forward — fully on-device.

        Args:
            hidden_states: [1, 1, seq_len, hidden_size] on device
            dense_routing: [1, 1, seq_len, num_experts] on device

        Returns:
            output: [1, 1, seq_len, hidden_size] on device
        """
        seq_len = hidden_states.shape[2]

        if seq_len == 1:
            return decode_forward(
                local_expert_mask=self.local_expert_mask,
                combine=combine,
                hidden_states=hidden_states,
                routing_weights=dense_routing,
                weights=self.weights,
                config=self.config,
                mesh_config=self.mesh_config,
                mesh_device=self.mesh_device,
                ccl_manager=self.ccl_manager,
            )
        else:
            assert seq_len % 32 == 0, f"Prefill seq_len must be a multiple of 32, got {seq_len}"
            return prefill_forward(
                num_local_experts=self.num_local_experts,
                combine=combine,
                hidden_states=hidden_states,
                routing_weights=dense_routing,
                weights=self.weights,
                config=self.config,
                prefill_sparsity=self.prefill_sparsity,
                mesh_config=self.mesh_config,
                mesh_device=self.mesh_device,
                ccl_manager=self.ccl_manager,
            )
