# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Unit test for ttnn.experimental.deepseek.moe.fused_experts (full preloaded-experts FFN).

The op takes *all* experts' weights and uses the routing weights to select which
experts to run. For the routing-selected ("hit") experts, in ascending hit-id order,
it computes the gate_up matmul, the SwiGLU gate, the down matmul, *and* the
routing-weighted accumulation into a single output row on device:

    gu     = x @ gate_up_w[hit_ids[i]]                          # [1, H] @ [H, 2I] -> [1, 2I]
    act    = silu(clamp(gu[:I], max=L)) * clamp(gu[I:], -L, L)  # -> [1, I]
    output = sum_i routing_weights[hit_ids[i]] * (act @ down_w[hit_ids[i]])  # -> [1, H]

The I SwiGLU columns are distributed across the compute grid: each SwiGLU core owns a
2-tile (64-column) slice and needs both the gate columns [64c, 64c+64) and the paired up
columns [I+64c, I+64c+64) of the gate_up weight, kept in a *single* [H, 128] DRAM shard
(host-permuted into per-core [gate_64 | up_64] blocks). The down matmul contracts over the
full I, so each SwiGLU core scatters its activation slice to core {0,0}, which gathers the
full activation and broadcasts it to every core; each core then multiplies it by its
[I, H/64] down shard to produce its 64-column slice of each expert's [1, H] row, scales it
by the expert's routing weight (SCALAR broadcast) and accumulates across experts. The output
tensor is [1, 1, H] in TILE layout (the decode token row padded to a 32-row tile), BFLOAT16.

Decode-only: sequence length T == 1.
"""

import pytest
import torch
import ttnn
import random

from models.common.utility_functions import comp_pcc, comp_allclose


# fused_experts uses an 8x8 compute grid; each active core owns a 2-tile SwiGLU output
# slice (64 cols), reading a [H, 128] (gate 64 | up 64) gate_up shard.
FUSED_EXPERTS_GRID = 8
FUSED_EXPERTS_NUM_CORES = FUSED_EXPERTS_GRID * FUSED_EXPERTS_GRID
BH_NUM_DRAM_BANKS = 8
COLS_PER_CORE = 64  # SwiGLU output columns per core (2 tiles)


def _nd_sharded_dram_memory_config(
    rows: int, cols: int, shard_width: int, dram_core_range_set: ttnn.CoreRangeSet
) -> ttnn.MemoryConfig:
    """ND-sharded DRAM: ``rows`` × ``shard_width`` per shard, round-robin over the DRAM banks."""
    assert cols % shard_width == 0, f"last dim {cols} must divide evenly into shards of {shard_width}"
    dram_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=[rows, shard_width],
        grid=dram_core_range_set,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    return ttnn.MemoryConfig(ttnn.BufferType.DRAM, dram_nd_shard_spec)


def _interleave_gate_up(w: torch.Tensor, block: int = COLS_PER_CORE) -> torch.Tensor:
    """Permute a [K, 2I] gate_up weight into per-core [gate_block | up_block] order so each
    [K, 2*block] shard holds a core's gate columns followed by its paired up columns.

    gate = w[:, :I], up = w[:, I:]; output column (c*2*block + h*block + t) == w[:, h*I + c*block + t].
    """
    k, two_i = w.shape
    intermediate = two_i // 2
    blocks = intermediate // block
    return w.reshape(k, 2, blocks, block).permute(0, 2, 1, 3).reshape(k, two_i).contiguous()


def _swiglu(gu: torch.Tensor, intermediate: int, limit: float) -> torch.Tensor:
    """Reference SwiGLU on a [tokens, 2I] gate_up output -> [tokens, I]."""
    gate = torch.clamp(gu[:, :intermediate], max=limit)
    up = torch.clamp(gu[:, intermediate:], min=-limit, max=limit)
    return torch.nn.functional.silu(gate) * up


@pytest.mark.parametrize(
    "hidden, intermediate, num_experts, num_nonzero",
    [
        # DeepSeek-V4-Flash config sizes (hidden_size=4096, moe_intermediate_size=2048).
        # The model has n_routed_experts=256; we use fewer here to keep DRAM/host memory
        # tractable for a unit test (each [4096, 4096] gate_up weight is ~32 MB).
        (4096, 2048, 64, 6),
    ],
)
def test_fused_experts_gate_up(device, hidden, intermediate, num_experts, num_nonzero):
    torch.manual_seed(0)
    limit = 7.0
    tokens = 1  # decode: sequence length T == 1
    two_intermediate = 2 * intermediate

    x = (torch.rand((tokens, hidden), dtype=torch.bfloat16) - 0.5).float()
    x_flat = x.reshape(1, 1, tokens, hidden)

    # Routing weights [T, E]; nonzero columns are the routing-selected ("hit") experts.
    # The op runs the gate_up matmul only for those, in ascending hit-id order.
    routing = torch.rand((tokens, num_experts), dtype=torch.bfloat16).float() + 0.5
    nonzero_cols = random.sample(range(num_experts), num_nonzero)
    for c in range(num_experts):
        if c not in nonzero_cols:
            routing[:, c] = 0.0
    routing_4d = routing.reshape(1, 1, tokens, num_experts)
    # Device scans experts 0..E-1, so the hit ids land in ascending order.
    hit_ids = sorted(nonzero_cols)

    gate_up_weights = [
        (torch.rand((hidden, two_intermediate), dtype=torch.bfloat16) - 0.5).float() for _ in range(num_experts)
    ]
    down_weights = [
        (torch.rand((intermediate, hidden), dtype=torch.bfloat16) - 0.5).float() for _ in range(num_experts)
    ]
    # Permute each gate_up weight into per-core [gate_64 | up_64] blocks so each [H, 128]
    # shard holds everything a core needs for its SwiGLU output slice in one NoC read.
    gate_up_perm = [_interleave_gate_up(w) for w in gate_up_weights]

    def to_tt(t, layout, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.from_torch(t, dtype=dtype, device=device, layout=layout, memory_config=memory_config)

    dram_core_ranges = [
        ttnn.CoreRange(ttnn.CoreCoord(bank_id, 0), ttnn.CoreCoord(bank_id, 0)) for bank_id in range(BH_NUM_DRAM_BANKS)
    ]
    dram_core_range_set = ttnn.CoreRangeSet(dram_core_ranges)

    # Each gate_up shard is one core's [H, 128] (gate 64 | up 64) slice.
    gate_up_mem_config = _nd_sharded_dram_memory_config(
        hidden, two_intermediate, 2 * COLS_PER_CORE, dram_core_range_set
    )
    down_mem_config = _nd_sharded_dram_memory_config(
        intermediate, hidden, hidden // FUSED_EXPERTS_NUM_CORES, dram_core_range_set
    )

    x_tt = to_tt(x_flat, ttnn.TILE_LAYOUT)
    routing_tt = ttnn.from_torch(
        routing_4d,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gate_up_tt = [
        to_tt(w, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat4_b, memory_config=gate_up_mem_config) for w in gate_up_perm
    ]
    down_tt = [to_tt(w, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat4_b, memory_config=down_mem_config) for w in down_weights]

    tt_out = ttnn.experimental.deepseek.moe.fused_experts(
        x_tt,
        routing_weights=routing_tt,
        gate_up_weights=gate_up_tt,
        down_weights=down_tt,
        num_experts=num_nonzero,
        intermediate_size=intermediate,
        swiglu_limit=limit,
    )

    out_torch = ttnn.to_torch(tt_out).float()  # [1, 1, H]
    assert list(out_torch.shape) == [1, 1, hidden], f"unexpected output shape {out_torch.shape}"

    # The op returns the routing-weighted sum over the selected experts:
    #   out = sum_i routing_weights[hit_ids[i]] * (swiglu(x @ gate_up_w) @ down_w).
    # Reference uses the bf16-rounded input and routing weights to match the device path; the
    # chained bf4 matmuls add quantization error, so PCC (not exact match) is checked.
    x_dev = ttnn.to_torch(x_tt).float().reshape(tokens, hidden)
    rw_dev = ttnn.to_torch(routing_tt).float().reshape(num_experts)
    ref = torch.zeros((tokens, hidden), dtype=torch.float32)
    for e in hit_ids:
        gu = (x_dev @ gate_up_weights[e]).reshape(tokens, two_intermediate)  # [1, 2I]
        act = _swiglu(gu, intermediate, limit)  # [1, I]
        ref = ref + rw_dev[e] * (act @ down_weights[e])  # [1, H], weighted-accumulated

    got = out_torch.reshape(tokens, hidden)
    passing, pcc_msg = comp_pcc(ref, got, pcc=0.98)
    assert passing, f"weighted-sum output mismatch: {pcc_msg} | {comp_allclose(ref, got)}"
