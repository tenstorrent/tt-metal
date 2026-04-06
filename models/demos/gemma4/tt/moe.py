# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 MoE Block: Router + Routed Experts — fully on device.

Router returns dense routing weights [1,1,S,E] on device.
Experts use sparse_matmul with that routing pattern.
No CPU round-trip.
"""

import ttnn
from models.demos.gemma4.tt.experts import Gemma4ExpertConfig, Gemma4Experts
from models.demos.gemma4.tt.router import Gemma4Router
from models.demos.gemma4.utils.substate import substate


class MoEBlock:
    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        ccl_manager,
        mesh_config,
        dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
    ):
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.hidden_size = hf_config.hidden_size

        self.router = Gemma4Router(
            mesh_device=mesh_device,
            hf_config=hf_config,
            state_dict=substate(state_dict, "router") if state_dict else {},
            tensor_cache_path=f"{tensor_cache_path}/router" if tensor_cache_path else None,
        )

        expert_config = Gemma4ExpertConfig(hf_config)
        self.experts = Gemma4Experts(
            mesh_device=mesh_device,
            config=expert_config,
            state_dict=substate(state_dict, "experts") if state_dict else {},
            ccl_manager=ccl_manager,
            mesh_config=mesh_config,
            program_config=None,
            weight_dtype=dtype,
            tensor_cache_path=f"{tensor_cache_path}/experts" if tensor_cache_path else None,
        )

    def __call__(self, router_input, expert_input):
        """
        MoE forward — fully on device.

        Args:
            router_input: [1, 1, seq_len, hidden_size] on device (for router linear)
            expert_input: [1, 1, seq_len, hidden_size] on device (normed, for experts)

        Returns:
            output: [1, 1, seq_len, hidden_size] on device
        """
        dense_routing = self.router(router_input)
        return self.experts(expert_input, dense_routing)
