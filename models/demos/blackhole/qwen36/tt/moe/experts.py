# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Routed experts. Dispatches decode vs prefill on the caller's explicit
mode, not the input shape: multi-user decode puts the B users on dim-2, so a shape test
would misroute B>1 decode into the prefill branch (decode_forward handles any batch)."""

from .decode import decode_forward
from .prefill import create_prefill_sparsity, prefill_forward
from .weights import load_expert_weights


class Qwen36Experts:
    def __init__(self, mesh_device, config, state_dict, tensor_cache_path=None, tt_ccl=None, topology=None):
        self.mesh_device = mesh_device
        self.config = config
        self.num_devices = config.num_devices
        self.tt_ccl = tt_ccl
        self.topology = topology
        self.weights = load_expert_weights(mesh_device, config, state_dict, tensor_cache_path)
        # Expert-parallel: each device computes only its expert shard, so the all-ones prefill
        # sparsity is sized to the per-device expert count (num_experts / num_devices).
        experts_per_device = config.num_experts // self.num_devices if self.num_devices > 1 else config.num_experts
        self.prefill_sparsity = create_prefill_sparsity(mesh_device, experts_per_device)

    def __call__(self, hidden_states, dense_routing, mode="decode"):
        """hidden_states [1,1,S,H] (S=batch in decode, seq in prefill), dense_routing
        [1,1,S,E] -> [1,1,S,H/tp]. mode ('decode'|'prefill') selects the path."""
        if mode == "decode":
            return decode_forward(
                hidden_states,
                dense_routing,
                self.weights,
                self.config,
                mesh_device=self.mesh_device,
                tt_ccl=self.tt_ccl,
                num_devices=self.num_devices,
                topology=self.topology,
            )
        seq_len = hidden_states.shape[2]
        assert seq_len % 32 == 0, f"Prefill seq_len must be a multiple of 32, got {seq_len}"
        return prefill_forward(
            hidden_states,
            dense_routing,
            self.weights,
            self.config,
            self.prefill_sparsity,
            mesh_device=self.mesh_device,
            tt_ccl=self.tt_ccl,
            num_devices=self.num_devices,
            topology=self.topology,
        )
