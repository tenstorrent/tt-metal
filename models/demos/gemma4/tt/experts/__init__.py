# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 Routed Experts module.

Decode (seq_len=1): on-device via sparse_matmul with top-k sparsity
Prefill (seq_len>1): on-device via sparse_matmul with all-ones sparsity (gpt_oss pattern)
"""

import ttnn

from .decode import decode_forward
from .prefill import create_local_expert_mask, create_prefill_sparsity, prefill_forward
from .weights import ExpertWeights, load_expert_weights
from models.demos.gemma4.tt.ccl import ccl_lane_gather_rows, ccl_lane_scatter_rows


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
        # Expert parallelism over the mesh axis that is not TP: every device keeps the full
        # (TP-sliced) expert set, but row ``r`` of the ep axis only ever computes experts
        # [r*E/ep, (r+1)*E/ep). The static local-expert mask zeroes the routing of the other
        # experts, so ``sparse_matmul`` neither reads nor computes them; the ep-axis combine
        # below sums the rows' partial outputs. EP=1 keeps the previous all-ones behaviour.
        self.ep = int(getattr(mesh_config, "ep", 1) or 1) if mesh_config is not None else 1
        self.num_local_experts = config.num_experts // self.ep
        self.local_expert_mask = None
        if self.ep > 1:
            self.local_expert_mask = create_local_expert_mask(mesh_device, mesh_config, config.num_experts, self.ep)
        # Prefill sparsity: all-ones, or the local-expert mask under EP (reused for every prefill call)
        self.prefill_sparsity = create_prefill_sparsity(
            mesh_device, config.num_experts, mesh_config=mesh_config if self.ep > 1 else None, ep=self.ep
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
            out = self._forward(hidden_states, dense_routing, combine=False)
            out = ccl_lane_scatter_rows(out, self.mesh_config)
            if pad_rows:
                out = ttnn.slice(out, [0, 0, 0, 0], [1, 1, rows, out.shape[-1]])
            return out
        return self._forward(hidden_states, dense_routing, combine=True)

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
