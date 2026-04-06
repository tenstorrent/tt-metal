# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 MoE Block: Router + Routed Experts.

This wraps the router (softmax-then-topk) and expert computation.
In Gemma4, the decoder layer combines shared_mlp output + moe output.

The MoE forward:
1. Router receives flattened residual (pre-MLP hidden states)
2. Router returns top_k_weights, top_k_indices
3. Experts receive normed hidden states + routing info
4. Experts compute weighted sum of expert outputs
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

    def __call__(self, router_input_tt, expert_input_torch):
        """
        MoE forward: route tokens then compute expert outputs.

        Args:
            router_input_tt: [1, 1, seq_len, hidden_size] on TT device (for router linear)
            expert_input_torch: [seq_len, hidden_size] torch tensor (normed, for experts)

        Returns:
            output: torch.Tensor [seq_len, hidden_size] (expert output, on CPU)
        """
        top_k_weights, top_k_indices = self.router(router_input_tt)
        return self.experts(expert_input_torch, top_k_indices, top_k_weights)
