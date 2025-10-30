# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import math
import random

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


class _DummyHFConfig:
    def __init__(
        self, *, hidden_size: int, moe_intermediate_size: int, n_routed_experts: int, num_experts_per_tok: int
    ):
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_act = "silu"


def _build_state_dict(num_experts: int, hidden_size: int, intermediate_size: int, seed: int = 0):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    # Aggregated layout expected by Experts: gate_up_proj[..., ::2] -> gate, [..., 1::2] -> up
    gate_up = torch.randn((num_experts, hidden_size, 2 * intermediate_size), dtype=torch.bfloat16, generator=g)
    down = torch.randn((num_experts, intermediate_size, hidden_size), dtype=torch.bfloat16, generator=g)
    return {"gate_up_proj": gate_up, "down_proj": down}


def _build_topk_mask(batch: int, seq: int, num_experts: int, k: int, seed: int = 0) -> torch.Tensor:
    # Build a (B,S,E) mask with exactly k active experts per token
    random.seed(seed)
    mask = torch.zeros((batch, seq, num_experts), dtype=torch.bfloat16)
    for b in range(batch):
        for s in range(seq):
            idxs = random.sample(range(num_experts), k)
            mask[b, s, idxs] = 1.0
    return mask


def _pt_reference(tokens: torch.Tensor, state_dict: dict, mask: torch.Tensor) -> torch.Tensor:
    # Fast CPU reference: loop over experts only, compute GEMMs on active tokens per expert.
    # tokens: (B,S,H), mask: (B,S,E)
    B, S, H = tokens.shape
    E = mask.shape[-1]
    gate_up = state_dict["gate_up_proj"].contiguous()  # (E, H, 2I)
    I = gate_up.shape[-1] // 2
    gate_w = gate_up[..., ::2].contiguous().to(torch.float32)  # (E,H,I)
    up_w = gate_up[..., 1::2].contiguous().to(torch.float32)  # (E,H,I)
    down_w = state_dict["down_proj"].contiguous().to(torch.float32)  # (E,I,H)

    X = tokens.to(torch.float32).reshape(B * S, H).contiguous()  # (BS,H)
    M = mask.to(torch.float32).reshape(B * S, E).contiguous()  # (BS,E)
    Y = torch.zeros((B * S, H), dtype=torch.float32)

    for e in range(E):
        m_e = M[:, e]
        idx = torch.nonzero(m_e, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        x_e = X.index_select(0, idx)  # (#act,H)
        g = x_e.matmul(gate_w[e])  # (#act,I)
        u = x_e.matmul(up_w[e])  # (#act,I)
        a = torch.nn.functional.silu(g).mul_(u)  # (#act,I)
        y_e = a.matmul(down_w[e])  # (#act,H)
        Y.index_add_(0, idx, y_e)

    return Y.view(B, S, H)


def _pt_reference_moe_batch(
    hidden_states: torch.Tensor,
    state_dict: dict,
    routing_weights: torch.Tensor,
    alpha: float,
    limit: float,
) -> torch.Tensor:
    # Follow GPT-OSS Experts batch path exactly (seq_len multiple of 32, dense routing)
    # hidden_states: (1, S, H), routing_weights: (S, E)
    assert hidden_states.dim() == 3 and hidden_states.shape[0] == 1
    B, S, H = hidden_states.shape
    E = routing_weights.shape[-1]
    gate_up = state_dict["gate_up_proj"].contiguous()  # (E, H, 2I) interleaved: even->gate, odd->up
    gate_up_bias = state_dict["gate_up_proj_bias"].contiguous()  # (E, 2I) interleaved: even->gate, odd->up
    down_w = state_dict["down_proj"].contiguous()  # (E, I, H)
    down_b = state_dict["down_proj_bias"].contiguous()  # (E, H)
    I = gate_up.shape[-1] // 2

    X = hidden_states.to(torch.float32).reshape(S, H)  # (S,H)
    RW = routing_weights.to(torch.float32)  # (S,E)

    # Compute per-expert gate/up, add bias, clamp as in GPT-OSS
    Y_sum = torch.zeros((S, H), dtype=torch.float32)
    for e in range(E):
        # Dense path for seq_len>1, every expert active with weight RW[s,e]
        # Split packed gate_up and bias by interleaving to match TT Experts path
        ge = X.matmul(gate_up[e, :, ::2].to(torch.float32))  # (S,I)
        ue = X.matmul(gate_up[e, :, 1::2].to(torch.float32))  # (S,I)
        ge = ge + gate_up_bias[e, ::2].to(torch.float32)
        ue = ue + gate_up_bias[e, 1::2].to(torch.float32)
        ge = torch.clamp(ge, max=limit)
        ue = torch.clamp(ue, min=-limit, max=limit)
        glu = ge * torch.sigmoid(ge * alpha)
        down_in = (ue + 1.0) * glu
        ye = down_in.matmul(down_w[e].to(torch.float32)) + down_b[e].to(torch.float32)  # (S,H)
        # Weight by routing weight and accumulate
        ye = ye * RW[:, e].unsqueeze(1)
        Y_sum += ye

    return Y_sum.unsqueeze(0).to(hidden_states.dtype)


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_moe_end_to_end_batch(mesh_device):
    torch.manual_seed(0)
    random.seed(0)

    # Dimensions chosen to mirror typical GPT-OSS shapes
    batch_size = 1
    seq_len = 32  # multiple of TILE_SIZE (32)
    hidden_size = 2048
    intermediate_size = 7168
    num_experts = 8
    num_experts_per_tok = 2  # not used in dense path but part of config
    ep = 1

    # Build aggregated expert weights with biases
    g = torch.Generator(device="cpu")
    g.manual_seed(123)
    gate_up = torch.randn((num_experts, hidden_size, 2 * intermediate_size), dtype=torch.bfloat16, generator=g)
    gate_up_bias = torch.randn((num_experts, 2 * intermediate_size), dtype=torch.bfloat16, generator=g)
    down = torch.randn((num_experts, intermediate_size, hidden_size), dtype=torch.bfloat16, generator=g)
    down_bias = torch.randn((num_experts, hidden_size), dtype=torch.bfloat16, generator=g)

    # Inputs and routing weights (dense routing for seq_len>1)
    hidden_states = torch.randn((batch_size, seq_len, hidden_size), dtype=torch.bfloat16)
    routing_weights = torch.ones((seq_len, num_experts), dtype=torch.bfloat16) / num_experts

    # Reference output (CPU)
    alpha = 1.702
    limit = 7.0  # use a reasonable clamp limit (GPT-OSS pulls from config)
    ref = _pt_reference_moe_batch(
        hidden_states,
        {
            "gate_up_proj": gate_up,
            "gate_up_proj_bias": gate_up_bias,
            "down_proj": down,
            "down_proj_bias": down_bias,
        },
        routing_weights,
        alpha=alpha,
        limit=limit,
    ).to(torch.bfloat16)

    # Build TTNN tensors, following GPT-OSS experts.py
    from models.demos.gpt_oss.config import MeshConfig

    mesh_config = MeshConfig(mesh_device.shape, tp=mesh_device.shape[1], ep=mesh_device.shape[0])
    expert_dim = intermediate_size
    intermediate_size_per_device = mesh_config.shard_size(expert_dim)

    gate_proj = gate_up[..., ::2].reshape(1, num_experts, hidden_size, expert_dim)
    up_proj = gate_up[..., 1::2].reshape(1, num_experts, hidden_size, expert_dim)
    gate_proj_bias = gate_up_bias[..., ::2].reshape(1, num_experts, 1, expert_dim)
    up_proj_bias = gate_up_bias[..., 1::2].reshape(1, num_experts, 1, expert_dim)
    down_proj = down.reshape(1, num_experts, expert_dim, hidden_size)
    down_proj_bias = down_bias.reshape(1, num_experts, 1, hidden_size)

    col_mesh_mapper = mesh_config.column_parallel(mesh_device)
    row_mesh_mapper = mesh_config.row_parallel(mesh_device)

    weight_dtype = ttnn.bfloat16
    tt_gate_proj = ttnn.as_tensor(
        gate_proj,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=col_mesh_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_up_proj = ttnn.as_tensor(
        up_proj,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=col_mesh_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_gate_proj_bias = ttnn.as_tensor(
        gate_proj_bias,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=col_mesh_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_up_proj_bias = ttnn.as_tensor(
        up_proj_bias,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=col_mesh_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_down_proj = ttnn.as_tensor(
        down_proj,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=row_mesh_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # For TP>1 GPT-OSS pads bias; here TP=1 so no padding required
    tt_down_proj_bias = ttnn.as_tensor(
        down_proj_bias,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Hidden states and routing weights
    tt_hidden_states = ttnn.from_torch(hidden_states, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_routing_weights = ttnn.from_torch(
        routing_weights, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    # Prefill sparsity as in GPT-OSS (EP sharding); EP=1 -> all experts active
    tokens_per_ep = num_experts // mesh_config.ep
    sparsity = torch.zeros(1, 1, mesh_config.ep, num_experts)
    for i in range(mesh_config.ep):
        sparsity[:, :, i, i * tokens_per_ep : (i + 1) * tokens_per_ep] = torch.ones(1, 1, 1, tokens_per_ep)
    tt_prefill_sparsity = ttnn.from_torch(
        sparsity,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(dims=(-2, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device),
    )

    # Reshape hidden states to 4D if seq_len>1 (group_size = seq_len//32)
    TILE_SIZE = 32
    hidden_states_4D = ttnn.unsqueeze_to_4D(tt_hidden_states)
    group_size = 1
    if seq_len > 1:
        hidden_states_4D = ttnn.reshape(hidden_states_4D, (1, seq_len // TILE_SIZE, TILE_SIZE, hidden_size))
        group_size = seq_len // TILE_SIZE

    tt_sparsity = ttnn.to_layout(ttnn.unsqueeze_to_4D(tt_routing_weights), ttnn.ROW_MAJOR_LAYOUT)
    if seq_len > 1:
        tt_sparsity = ttnn.repeat(tt_prefill_sparsity, (1, 1, group_size, 1))

    output_tile = ttnn.Tile([32, 32])

    def sparse_matmul_program(core_x, core_y, m, n):
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=max(32, m) // 32,
            per_core_N=int(math.ceil(n / 32)) // (core_x * core_y),
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )

    def batched_sparse_matmul_program(core_x, core_y, m, n):
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=max(32, m) // 32,
            per_core_N=int(math.ceil(n / 32)) // (core_x * core_y),
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )

    num_experts_per_tok_dense = (num_experts // mesh_config.ep) * group_size

    # gate
    gate = ttnn.sparse_matmul(
        hidden_states_4D,
        tt_gate_proj,
        sparsity=tt_sparsity,
        nnz=num_experts_per_tok_dense,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=sparse_matmul_program(8, 7, hidden_states_4D.shape[2], tt_gate_proj.shape[3]),
    )
    if seq_len > 1:
        gate = ttnn.transpose(gate, 1, 3)
    gate = ttnn.reshape(gate, (batch_size, num_experts, seq_len, intermediate_size_per_device))
    gate = ttnn.add(gate, tt_gate_proj_bias, output_tensor=gate)
    gate = ttnn.clamp(gate, min=None, max=limit)

    # up
    up = ttnn.sparse_matmul(
        hidden_states_4D,
        tt_up_proj,
        sparsity=tt_sparsity,
        nnz=num_experts_per_tok_dense,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=sparse_matmul_program(8, 7, hidden_states_4D.shape[2], tt_up_proj.shape[3]),
    )
    if seq_len > 1:
        up = ttnn.transpose(up, 1, 3)
    up = ttnn.reshape(up, (batch_size, num_experts, seq_len, intermediate_size_per_device))
    up = ttnn.add(up, tt_up_proj_bias, output_tensor=up)
    up = ttnn.clamp(up, min=-limit, max=limit)

    # SWiGLU and prepare down input
    glu = gate * ttnn.sigmoid(gate * alpha)
    down_in0 = (up + 1) * glu
    ttnn.deallocate(glu)
    ttnn.deallocate(up)
    ttnn.deallocate(gate)
    down_in0 = ttnn.reshape(down_in0, (1, num_experts, seq_len, intermediate_size_per_device))
    if seq_len > 1:
        tt_sparsity = tt_prefill_sparsity
        num_experts_per_tok_dense = num_experts // mesh_config.ep

    # down (batched sparse matmul, input A is sparse)
    down = ttnn.sparse_matmul(
        down_in0,
        tt_down_proj,
        sparsity=tt_sparsity,
        nnz=num_experts_per_tok_dense,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
        is_input_a_sparse=True,
        program_config=batched_sparse_matmul_program(8, 8, down_in0.shape[2], tt_down_proj.shape[-1]),
    )
    next_states = ttnn.reshape(down, (batch_size, num_experts, seq_len, hidden_size)) + tt_down_proj_bias

    # Combine experts using routing weights
    if seq_len > 1:
        tt_routing_weights = tt_routing_weights * ttnn.reshape(tt_prefill_sparsity, (1, num_experts))
    routing_weights_t = ttnn.permute(tt_routing_weights, (1, 0))
    routing_weights_t = ttnn.reshape(routing_weights_t, (batch_size, num_experts, seq_len, 1))
    next_states = ttnn.mul(next_states, routing_weights_t, output_tensor=next_states)
    next_states = ttnn.sum(next_states, dim=1, keepdim=True)
    next_states = ttnn.reshape(next_states, (batch_size, seq_len, hidden_size))

    # Compare
    out = ttnn.to_torch(ttnn.get_device_tensors(next_states)[0]).to(torch.bfloat16)
    assert_with_pcc(ref, out, 0.99)
