from __future__ import annotations

from typing import Any, List, Optional

import torch
import ttnn
from ttnn import ReplicateTensorToMesh

from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.common import Mode


class Qwen3CoderNextMoEConfig:
    """Config for Qwen3-Coder-Next MoE (matches Qwen3NextConfig / Qwen3NextSparseMoeBlock)."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        moe_intermediate_size: int,
        shared_expert_intermediate_size: int,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
    ):
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor


def _expert_weights_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    num_experts: int,
    moe_intermediate_size: int,
    hidden_size: int,
    mesh_device: ttnn.Device,
    layer_num: int,
    dummy_weights: bool,
    weight_cache_path: Optional[Any],
) -> tuple[List[ttnn.Tensor], List[ttnn.Tensor]]:
    """Load expert gate_up and down_proj from state_dict and place on device via ttnn."""
    gate_up = state_dict[f"{prefix}experts.gate_up_proj"]  # (num_experts, 2*interm, hidden)
    down_proj = state_dict[f"{prefix}experts.down_proj"]  # (num_experts, hidden, interm)
    gate_up_tt: List[ttnn.Tensor] = []
    down_tt: List[ttnn.Tensor] = []
    for e in range(num_experts):
        # gate_up[e]: (2*interm, hidden) -> we store as (hidden, 2*interm) so that
        # input (.., hidden) @ W (hidden, 2*interm) yields (..., 2*interm).
        w_gu = gate_up[e].permute(1, 0).to(torch.bfloat16).unsqueeze(0).unsqueeze(0)
        cache_gu = None
        if not dummy_weights and weight_cache_path is not None:
            cache_gu = weight_cache_path / f"qwen3_moe_layer{layer_num}_expert{e}_gate_up"
        gate_up_tt.append(
            ttnn.as_tensor(
                w_gu,
                dtype=ttnn.bfloat16,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_gu,
                mesh_mapper=ReplicateTensorToMesh(mesh_device),
            )
        )
        # down_proj[e]: (hidden, interm) -> (1,1, interm, hidden) for matmul combined @ W.T
        w_down = down_proj[e].T.unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
        cache_down = None
        if not dummy_weights and weight_cache_path is not None:
            cache_down = weight_cache_path / f"qwen3_moe_layer{layer_num}_expert{e}_down"
        down_tt.append(
            ttnn.as_tensor(
                w_down,
                dtype=ttnn.bfloat16,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_down,
                mesh_mapper=ReplicateTensorToMesh(mesh_device),
            )
        )
    return gate_up_tt, down_tt


class TtQwen3CoderNextMoELayer(LightweightModule):
    """
    Qwen3-Coder-Next MoE layer using ttnn (no tt_symbiote).

    - Gate + ttnn softmax + top-k routing; expert compute on device via ttnn matmul/silu/mul.
    - State_dict: all weights (gate, experts, shared expert) loaded and placed on device via ttnn.
    - Shared expert: sigmoid(shared_expert_gate(x)) * shared_expert(x).
    """

    def __init__(
        self,
        mesh_device: ttnn.Device,
        state_dict: dict[str, torch.Tensor],
        state_dict_prefix: str,
        config: Qwen3CoderNextMoEConfig,
        layer_num: int,
        dtype: ttnn.DataType = ttnn.bfloat16,
        tt_ccl: Optional[Any] = None,
        dummy_weights: bool = False,
        weight_cache_path: Optional[Any] = None,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.config = config
        self.dtype = dtype
        self.tt_ccl = tt_ccl
        self.layer_num = layer_num
        prefix = state_dict_prefix

        # ----- Gate: (num_experts, hidden_size) -> (hidden_size, num_experts) for x @ W.T
        gate_w = state_dict[f"{prefix}gate.weight"]
        gate_w = gate_w.permute(1, 0).unsqueeze(0).unsqueeze(0)
        cache_gate = None
        if not dummy_weights and weight_cache_path is not None:
            cache_gate = weight_cache_path / f"qwen3_moe_layer{layer_num}_gate_weight"
        self.gate_weight = ttnn.as_tensor(
            gate_w.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_gate,
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )

        # ----- Experts: load from state_dict and place on device via ttnn
        self.gate_up_tt, self.down_tt = _expert_weights_from_state_dict(
            state_dict,
            prefix,
            config.num_experts,
            config.moe_intermediate_size,
            config.hidden_size,
            mesh_device,
            layer_num,
            dummy_weights,
            weight_cache_path,
        )

        # ----- Shared expert: gate_proj, up_proj, down_proj
        shared_prefix = f"{prefix}shared_expert."
        gate_proj_w = state_dict[f"{shared_prefix}gate_proj.weight"].permute(1, 0).unsqueeze(0).unsqueeze(0)
        up_proj_w = state_dict[f"{shared_prefix}up_proj.weight"].permute(1, 0).unsqueeze(0).unsqueeze(0)
        down_proj_w = state_dict[f"{shared_prefix}down_proj.weight"].permute(1, 0).unsqueeze(0).unsqueeze(0)
        cache_shared_gate = None
        if not dummy_weights and weight_cache_path is not None:
            cache_shared_gate = weight_cache_path / f"qwen3_moe_layer{layer_num}_shared_gate_proj"
        self.shared_gate_proj = ttnn.as_tensor(
            gate_proj_w.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_shared_gate,
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )
        cache_up = (
            weight_cache_path / f"qwen3_moe_layer{layer_num}_shared_up_proj"
            if (not dummy_weights and weight_cache_path is not None)
            else None
        )
        cache_down = (
            weight_cache_path / f"qwen3_moe_layer{layer_num}_shared_down_proj"
            if (not dummy_weights and weight_cache_path is not None)
            else None
        )
        self.shared_up_proj = ttnn.as_tensor(
            up_proj_w.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_up,
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )
        self.shared_down_proj = ttnn.as_tensor(
            down_proj_w.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_down,
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )

        # ----- Shared expert gate: (1, hidden_size)
        seg_w = state_dict[f"{prefix}shared_expert_gate.weight"]
        seg_w = seg_w.permute(1, 0).unsqueeze(0).unsqueeze(0)
        self.shared_expert_gate_weight = ttnn.as_tensor(
            seg_w.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ReplicateTensorToMesh(mesh_device),
        )

        self._hidden_size = config.hidden_size
        self._num_experts = config.num_experts
        self._top_k = config.num_experts_per_tok

    def forward(self, inputs: ttnn.Tensor, mode: Mode = Mode.DECODE) -> ttnn.Tensor:
        """
        inputs: (1, 1, seq_len, hidden_size) or (1, 1, batch*seq, hidden_size).
        """
        orig_shape = list(inputs.shape)
        batch_seq = orig_shape[-2]
        hidden = orig_shape[-1]
        interm = self.config.moe_intermediate_size

        # Ensure 4D
        if len(orig_shape) == 3:
            inputs = ttnn.unsqueeze(inputs, 0)
        input_4d = inputs

        # 1) Router logits on device: x @ gate.T
        router_logits = ttnn.matmul(
            input_4d,
            self.gate_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # 2) Routing in ttnn: softmax -> topk -> gather weights -> normalize -> scale
        if router_logits.layout != ttnn.TILE_LAYOUT:
            router_logits = ttnn.to_layout(
                router_logits,
                ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        probs = ttnn.softmax(router_logits, dim=-1)
        router_logits.deallocate(True)
        topk_values, topk_indices = ttnn.topk(probs, k=self._top_k, dim=-1, largest=True, sorted=False)
        if topk_values is not None:
            topk_values.deallocate(True)
        topk_weights = ttnn.gather(probs, dim=3, index=topk_indices)
        probs.deallocate(True)
        if self.config.norm_topk_prob:
            denom = ttnn.sum(topk_weights, dim=3, keepdim=True)
            denom = ttnn.add(denom, 1e-20, output_tensor=denom)
            topk_weights = ttnn.div(topk_weights, denom)
            denom.deallocate(True)
        if self.config.routed_scaling_factor != 1.0:
            topk_weights = ttnn.mul(topk_weights, self.config.routed_scaling_factor)

        # 3) Expert forward and mixing entirely on TTNN.
        # Build per-expert, per-token weights using TTNN scatter on device.
        T_len = batch_seq
        num_experts = self._num_experts
        expert_input = ttnn.from_torch(
            torch.zeros((1, 1, T_len, num_experts), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ReplicateTensorToMesh(self.mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        expert_weights = ttnn.scatter(
            input=expert_input,
            index=topk_indices,
            src=topk_weights,
            dim=3,
        )
        expert_input.deallocate(True)
        topk_indices.deallocate(True)
        topk_weights.deallocate(True)

        routed_output: Optional[ttnn.Tensor] = None
        for e in range(num_experts):
            gate_up_out = ttnn.matmul(
                input_4d,
                self.gate_up_tt[e],
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            gate_part = ttnn.slice(gate_up_out, (0, 0, 0, 0), (1, 1, batch_seq, interm))
            up_part = ttnn.slice(gate_up_out, (0, 0, 0, interm), (1, 1, batch_seq, 2 * interm))
            gate_up_out.deallocate(True)
            gate_act = ttnn.silu(gate_part)
            combined = ttnn.mul(gate_act, up_part)
            gate_part.deallocate(True)
            up_part.deallocate(True)
            gate_act.deallocate(True)
            expert_out = ttnn.matmul(
                combined,
                self.down_tt[e],
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            combined.deallocate(True)

            # Slice per-expert weights: (1,1,T,1) and apply on device.
            weights_e_tt = ttnn.slice(expert_weights, (0, 0, 0, e), (1, 1, T_len, e + 1))
            weighted_expert = ttnn.mul(expert_out, weights_e_tt)
            weights_e_tt.deallocate(True)
            expert_out.deallocate(True)

            if routed_output is None:
                routed_output = weighted_expert
            else:
                routed_output = ttnn.add(routed_output, weighted_expert)
                weighted_expert.deallocate(True)

        expert_weights.deallocate(True)
        assert routed_output is not None

        # 4) Shared expert: down(silu(gate_proj(x)) * up_proj(x))
        gate_out = ttnn.matmul(
            input_4d,
            self.shared_gate_proj,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        up_out = ttnn.matmul(
            input_4d,
            self.shared_up_proj,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        gate_out = ttnn.silu(gate_out)
        shared_in = ttnn.mul(gate_out, up_out)
        gate_out.deallocate(True)
        up_out.deallocate(True)
        shared_out = ttnn.matmul(
            shared_in,
            self.shared_down_proj,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        shared_in.deallocate(True)

        # 5) Shared expert gate: sigmoid(shared_expert_gate(x))
        gate_scale = ttnn.matmul(
            input_4d,
            self.shared_expert_gate_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        gate_scale = ttnn.sigmoid(gate_scale)
        shared_out = ttnn.mul(shared_out, gate_scale)
        gate_scale.deallocate(True)

        # 6) Add routed + gated shared
        output = ttnn.add(routed_output, shared_out)
        routed_output.deallocate(True)
        shared_out.deallocate(True)
        input_4d.deallocate(True)

        return output
