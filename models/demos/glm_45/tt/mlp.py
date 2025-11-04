# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
MoE MLP: Router + Experts with minimal abstraction
"""
import ttnn
from models.demos.glm_45.utils.general_utils import get_cache_file_name
from models.demos.glm_45.utils.substate import substate

from .experts import Experts
from .topk import TopKRouter


class DenseGLU:
    """Dense MLP path (GLM style): down = down_proj(act(gate) * up)

    Expects weights under keys: gate_proj.weight, up_proj.weight, down_proj.weight
    Biases are optional and default to zeros if missing.
    """

    def __init__(self, mesh_device, hf_config, state_dict, tensor_cache_path=None, mesh_config=None):
        self.hidden_size = hf_config.hidden_size
        self.intermediate_size = hf_config.intermediate_size
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config

        # Load weights (biases optional)
        gate_w = state_dict["gate_proj.weight"].transpose(0, 1)
        up_w = state_dict["up_proj.weight"].transpose(0, 1)
        down_w = state_dict["down_proj.weight"].transpose(0, 1)

        gate_b = state_dict.get("gate_proj.bias")
        up_b = state_dict.get("up_proj.bias")
        down_b = state_dict.get("down_proj.bias")

        # Create TTNN tensors
        self.gate_w = ttnn.as_tensor(
            gate_w,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            cache_file_name=get_cache_file_name(tensor_cache_path, "gate_proj"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.up_w = ttnn.as_tensor(
            up_w,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            cache_file_name=get_cache_file_name(tensor_cache_path, "up_proj"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.down_w = ttnn.as_tensor(
            down_w,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            cache_file_name=get_cache_file_name(tensor_cache_path, "down_proj"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Biases default to zeros if not provided
        if gate_b is not None:
            self.gate_b = ttnn.as_tensor(
                gate_b.unsqueeze(0),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                cache_file_name=get_cache_file_name(tensor_cache_path, "gate_proj_bias"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.gate_b = None

        if up_b is not None:
            self.up_b = ttnn.as_tensor(
                up_b.unsqueeze(0),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                cache_file_name=get_cache_file_name(tensor_cache_path, "up_proj_bias"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.up_b = None

        if down_b is not None:
            self.down_b = ttnn.as_tensor(
                down_b.unsqueeze(0),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                cache_file_name=get_cache_file_name(tensor_cache_path, "down_proj_bias"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.down_b = None

        # Activation
        self.hidden_act = getattr(hf_config, "hidden_act", "silu")

    def __call__(self, hidden_states):
        # hidden_states: (B, S, H)
        gate = ttnn.linear(hidden_states, self.gate_w, bias=self.gate_b)
        up = ttnn.linear(hidden_states, self.up_w, bias=self.up_b)

        act_l = self.hidden_act.lower() if isinstance(self.hidden_act, str) else "silu"
        if act_l == "gelu" or act_l == "gelu_pytorch_tanh":
            gate = ttnn.gelu(gate)
        elif act_l == "relu":
            gate = ttnn.relu(gate)
        else:
            gate = ttnn.silu(gate)

        down_in = ttnn.mul(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(up)
        ttnn.deallocate(gate)

        out = ttnn.linear(down_in, self.down_w, bias=self.down_b)
        ttnn.deallocate(down_in)
        return out


class MLP:
    """Streamlined MoE MLP combining router and experts"""

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=None,
    ):
        # Decide between MoE and dense MLP based on available keys
        router_state_dict = substate(state_dict, "gate")

        experts_state_dict = substate(state_dict, "experts")
        shared_state_dict = substate(state_dict, "shared_experts")

        # Validate presence for router
        has_router = bool(router_state_dict) and ("weight" in router_state_dict)

        # Validate experts presence: aggregated or per-expert
        has_experts = False
        if bool(experts_state_dict):
            agg_ok = all(
                k in experts_state_dict for k in ["gate_up_proj", "gate_up_proj_bias", "down_proj", "down_proj_bias"]
            )
            per_ok = any(k.endswith("gate_proj.weight") for k in experts_state_dict.keys())
            has_experts = agg_ok or per_ok
            if not has_experts:
                raise ValueError(
                    "Malformed experts state: expected aggregated tensors (gate_up_proj, gate_up_proj_bias, down_proj, "
                    "down_proj_bias) or per-expert weights experts.{i}.(gate_proj|up_proj|down_proj).weight"
                )

        # Dense GLU path if standard dense MLP weights are present
        has_dense = all(k in state_dict for k in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"])

        # Enforce consistency: if one of router/experts exists, both must exist
        if has_router ^ has_experts:
            missing = "experts" if has_router else "router"
            raise ValueError(f"MoE configuration incomplete: missing {missing} weights for MLP")

        self.is_moe = has_router and has_experts
        # Shared experts must be present if required by config and this layer is MoE
        n_shared = getattr(hf_config, "n_shared_experts", 0) or 0
        if self.is_moe and n_shared > 0:
            required_shared = ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]
            missing_shared = [k for k in required_shared if k not in shared_state_dict]
            if missing_shared:
                raise ValueError(
                    f"Missing shared_experts weights: {missing_shared}. n_shared_experts={n_shared} requires them."
                )
        self.is_dense = (not self.is_moe) and has_dense

        if self.is_moe:
            # Initialize components with mesh_config
            self.router = TopKRouter(
                mesh_device,
                hf_config,
                router_state_dict,
                tensor_cache_path=get_cache_file_name(tensor_cache_path, "router"),
            )
            self.experts = Experts(
                mesh_device,
                hf_config,
                experts_state_dict,
                ccl_manager,
                dtype=dtype,
                tensor_cache_path=get_cache_file_name(tensor_cache_path, "experts"),
                mesh_config=mesh_config,
                shared_state_dict=shared_state_dict,
            )
            self._dense = None
        elif self.is_dense:
            self.router = None
            self.experts = None
            self._dense = DenseGLU(
                mesh_device,
                hf_config,
                state_dict,
                tensor_cache_path=get_cache_file_name(tensor_cache_path, "dense_mlp"),
                mesh_config=mesh_config,
            )
        else:
            # Neither MoE nor DenseMlp detected -> strict failure
            raise ValueError(
                "Unable to initialize MLP: neither valid MoE (router+experts) nor Dense GLU weights present."
            )

    def __call__(self, hidden_states):
        """Forward pass: route -> experts (MoE) or run dense GLU path"""
        if self.is_moe:
            router_scores, router_indices, router_logits = self.router(hidden_states)
            expert_output = self.experts(hidden_states, router_scores)
            return expert_output, router_scores
        else:
            out = self._dense(hidden_states)
            return out, None
