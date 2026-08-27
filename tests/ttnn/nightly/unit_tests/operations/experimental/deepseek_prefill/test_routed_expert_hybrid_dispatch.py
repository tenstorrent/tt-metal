# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Splitting the routed experts across both ops must change nothing but the timing.

A hybrid forward runs `unified_routed_expert_moe` over the experts above a token
threshold and `moe_fused_swiglu` over those at or below it, both reading the SAME
device-resident counts vector and dropping whatever falls outside their band. The
two bands have to tile the experts exactly once: no expert done twice, none missed,
and none of the zero-count ones touched by either.

So the check is agreement, not PCC against torch: every expert is compared against
the SAME op running it unbanded. A band that leaks would show up as one expert's
region holding another op's numbers, or stale allocation garbage.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc, is_blackhole

TILE = 32
EMB = 7168
HIDDEN = 2048
GRID = ttnn.CoreCoord(11, 8)
NUM_ROUTED_EXPERTS = 384
CAPACITY = 512

# Straddles the threshold in both directions, includes zero-count experts, and puts a
# ragged tail on each side so neither band gets a uniformly full schedule.
COUNTS = (96, 512, 0, 300, 256, 0, 480, 128)
THRESHOLD = 256


def _to_device(tensor, dtype, layout, device):
    return ttnn.from_torch(
        tensor.contiguous(), dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
@pytest.mark.skipif(not is_blackhole(), reason="the hybrid dispatch is Blackhole-only")
def test_hybrid_matches_each_op_unbanded(device):
    torch.manual_seed(20260827)
    experts = len(COUNTS)

    host = [
        (
            torch.randn((EMB, HIDDEN), dtype=torch.bfloat16) * 2.0e-2,
            torch.randn((EMB, HIDDEN), dtype=torch.bfloat16) * 2.0e-2,
            torch.randn((HIDDEN, EMB), dtype=torch.bfloat16) * 2.0e-2,
        )
        for _ in range(experts)
    ]
    gates = [_to_device(w[0], ttnn.bfloat4_b, ttnn.TILE_LAYOUT, device) for w in host]
    ups = [_to_device(w[1], ttnn.bfloat4_b, ttnn.TILE_LAYOUT, device) for w in host]
    downs = [_to_device(w[2], ttnn.bfloat4_b, ttnn.TILE_LAYOUT, device) for w in host]

    total_rows = CAPACITY * experts
    x_host = torch.randn((1, 1, total_rows, EMB), dtype=torch.bfloat16)
    x = _to_device(x_host, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device)

    global_ids = [(11 + 37 * e) % NUM_ROUTED_EXPERTS for e in range(experts)]
    idx = _to_device(torch.tensor(global_ids, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device)
    counts_host = torch.zeros(NUM_ROUTED_EXPERTS, dtype=torch.int32)
    offsets_host = torch.zeros(NUM_ROUTED_EXPERTS, dtype=torch.int32)
    for e, gid in enumerate(global_ids):
        counts_host[gid] = COUNTS[e]
        offsets_host[gid] = e * CAPACITY
    counts = _to_device(counts_host, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device)
    offsets = _to_device(offsets_host, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device)

    fused_op = ttnn.experimental.deepseek_prefill.moe_fused_swiglu
    unified_op = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe

    def run_unified(min_active_tokens):
        return unified_op(
            x,
            offsets,
            counts,
            idx,
            gates,
            ups,
            downs,
            max_dispatched_tokens_per_expert=CAPACITY,
            min_active_tokens=min_active_tokens,
        )

    # Reference: each op run over ALL experts, unbanded.
    unified_all = ttnn.to_torch(run_unified(0))[0, 0].float()
    fused_ref_out = run_unified(0)  # allocation the fused reference writes into
    fused_op(
        x,
        gates,
        ups,
        downs,
        counts,
        idx,
        input_m_tiles=CAPACITY // TILE,
        core_grid=GRID,
        output=fused_ref_out,
        expert_region_offsets=offsets,
        read_x_at_offset=True,
    )
    fused_all = ttnn.to_torch(fused_ref_out)[0, 0].float()

    # Hybrid: unified above the threshold allocates, fused at or below writes in.
    hybrid_out = run_unified(THRESHOLD + 1)
    fused_op(
        x,
        gates,
        ups,
        downs,
        counts,
        idx,
        input_m_tiles=CAPACITY // TILE,
        core_grid=GRID,
        output=hybrid_out,
        expert_region_offsets=offsets,
        read_x_at_offset=True,
        max_active_tokens=THRESHOLD,
    )
    hybrid = ttnn.to_torch(hybrid_out)[0, 0].float()

    failures = []
    for e, count in enumerate(COUNTS):
        if count == 0:
            continue
        base = e * CAPACITY
        rows = hybrid[base : base + count]
        # Which op OWNS this expert decides what its rows must equal.
        owner = "fused" if count <= THRESHOLD else "unified"
        expect = (fused_all if owner == "fused" else unified_all)[base : base + count]
        if not torch.isfinite(rows).all():
            failures.append(f"expert {e} (count {count}, {owner}) non-finite")
            continue
        passed, msg = comp_pcc(expect, rows, 0.999)
        if not passed:
            failures.append(f"expert {e} (count {count}, {owner}) vs same-op reference: {msg}")
    assert not failures, "hybrid dispatch disagreed with the unbanded run: " + "; ".join(failures)
