# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 MoE Block: Router + Routed Experts — fully on device.

Router returns dense routing weights [1,1,S,E] on device.
Experts use sparse_matmul with that routing pattern.
No CPU round-trip.
"""

import ttnn
from models.demos.gemma4.tt.decode_matmul import flag, is_decode_rows
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
        router_dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        layer_idx=0,
    ):
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.hidden_size = hf_config.hidden_size

        self.router = Gemma4Router(
            mesh_device=mesh_device,
            hf_config=hf_config,
            state_dict=substate(state_dict, "router") if state_dict else {},
            tensor_cache_path=f"{tensor_cache_path}/router" if tensor_cache_path else None,
            dtype=router_dtype,
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
            layer_idx=layer_idx,
            num_layers=getattr(hf_config, "num_hidden_layers", 1),
        )

    def fused_decode_ok(self) -> bool:
        """GEMMA4_MOE_FUSED=1 on tp-only decode rows (whole experts, no lanes, no decode kernel, a per-expert scale)
        and an expert count / top-k the router program takes (``RouterTopk.supports``: a tile multiple of experts up
        to 256, k <= 32; the CI's reduced-expert configurations fall back to the shared chain)."""
        from models.demos.gemma4.custom_ops.router_topk import RouterTopk

        lane_sharded = bool(self.mesh_config is not None and getattr(self.mesh_config, "lane_sharded", False))
        return (
            flag("GEMMA4_MOE_FUSED")
            and not lane_sharded
            and self.experts.ep == 1
            and self.experts.moe_decode is None
            and self.router.per_expert_scale is not None
            and RouterTopk.supports(self.router.num_experts, self.router.top_k)
        )

    def fused_packed_routing_ok(self) -> bool:
        """GEMMA4_MOE_FUSED=1 on the decode MoE kernel rows: the packed router program (``RouterTopk.supports``
        with its 32-byte packed rows, k a multiple of 8)."""
        from models.demos.gemma4.custom_ops.router_topk import RouterTopk

        return (
            flag("GEMMA4_MOE_FUSED")
            and self.router.per_expert_scale is not None
            and RouterTopk.supports(self.router.num_experts, self.router.top_k, packed=True)
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
        if self.fused_decode_ok() and is_decode_rows(expert_input):
            # GEMMA4_MOE_FUSED=1: router tail, GeGLU and expert combine as custom ops (tp-only decode frames)
            return self.experts(
                expert_input, self.router.fused_routing(router_input, self.experts.weights, self.mesh_config)
            )
        if self.experts.wants_sparse_routing(expert_input):
            if self.fused_packed_routing_ok() and is_decode_rows(expert_input):
                # GEMMA4_MOE_FUSED=1 on the kernel rows (lanes / tp): the router tail as one program, its packed
                # (weights | ids) rows go through custom_ops/moe_frame_prep into the kernel's dispatch
                return self.experts(expert_input, self.router.fused_sparse_routing(router_input))
            # decode MoE kernel rows: top-k straight from the router (no dense scatter, no second top-k)
            return self.experts(expert_input, self.router(router_input, sparse=True))
        dense_routing = self.router(router_input)
        return self.experts(expert_input, dense_routing)
