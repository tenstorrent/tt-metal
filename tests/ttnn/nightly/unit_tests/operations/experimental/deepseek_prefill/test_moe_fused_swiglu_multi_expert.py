# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Every local expert in ONE moe_fused_swiglu dispatch.

The op loops the experts device-side off `counts[global_expert_idx_table[e]]`, so the cases that
matter are the ones where consecutive experts disagree: a zero count (skipped with no CB traffic and
no collective round), a ragged tail (the last M-block of an expert is smaller than M_BLOCK, which
the NEXT expert's block 0 must not inherit), and a grouped-M expert next to an ungrouped one.
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
WEIGHT_SCALE = 2.0e-2

# Per-expert token counts, one case per boundary the expert loop has to survive. `m_eff` is the
# power of two the last M-block rounds up to, so 256 is one full block, 96 is a short block and 512
# is two full blocks (which is also where grouped-M turns on).
COUNT_PATTERNS = {
    "uniform-full": (256, 256, 256),
    "with-zeros": (0, 256, 0, 256),
    "ragged-tail": (200, 256, 300),
    "short-blocks": (96, 96, 96),
    "short-then-full": (96, 256),
    "full-then-short": (256, 96),
    "short-then-ragged": (96, 300),
    "mixed": (0, 200, 512, 96, 0, 300),
}
CAPACITY = 512


def _to_device(tensor, dtype, layout, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        tensor.contiguous(),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config,
    )


def _reference(x_rows, gate, up, down):
    gate_out = x_rows @ gate.float()
    up_out = x_rows @ up.float()
    return (torch.nn.functional.silu(gate_out) * up_out) @ down.float()


@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
@pytest.mark.parametrize("pattern", list(COUNT_PATTERNS), ids=list(COUNT_PATTERNS))
@pytest.mark.skipif(not is_blackhole(), reason="moe_fused_swiglu is Blackhole-only")
def test_moe_fused_swiglu_runs_every_local_expert_in_one_dispatch(device, pattern):
    torch.manual_seed(20260827)
    counts_per_expert = COUNT_PATTERNS[pattern]
    experts = len(counts_per_expert)

    # Distinct weights per expert: a shared set would pass even if every expert read expert 0's
    # base address, which is exactly the failure the per-expert runtime-arg table can introduce.
    host_weights = [
        (
            torch.randn((EMB, HIDDEN), dtype=torch.bfloat16) * WEIGHT_SCALE,
            torch.randn((EMB, HIDDEN), dtype=torch.bfloat16) * WEIGHT_SCALE,
            torch.randn((HIDDEN, EMB), dtype=torch.bfloat16) * WEIGHT_SCALE,
        )
        for _ in range(experts)
    ]
    w_gates, w_ups, w_downs = ([], [], [])
    for gate, up, down in host_weights:
        w_gates.append(_to_device(gate, ttnn.bfloat4_b, ttnn.TILE_LAYOUT, device))
        w_ups.append(_to_device(up, ttnn.bfloat4_b, ttnn.TILE_LAYOUT, device))
        w_downs.append(_to_device(down, ttnn.bfloat4_b, ttnn.TILE_LAYOUT, device))

    # One shared dispatched buffer: every expert reads its own region and writes it back, which is
    # what makes a per-expert base-address mix-up visible as a wrong region rather than as noise.
    region_rows = CAPACITY
    total_rows = region_rows * experts
    x_host = torch.randn((1, 1, total_rows, EMB), dtype=torch.bfloat16)
    x = _to_device(x_host, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device)
    output = ttnn.empty(
        [1, 1, total_rows, EMB],
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    global_ids = [(11 + 37 * local) % NUM_ROUTED_EXPERTS for local in range(experts)]
    idx = _to_device(torch.tensor(global_ids, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device)

    counts_host = torch.zeros(NUM_ROUTED_EXPERTS, dtype=torch.int32)
    offsets_host = torch.zeros(NUM_ROUTED_EXPERTS, dtype=torch.int32)
    for local, global_id in enumerate(global_ids):
        counts_host[global_id] = counts_per_expert[local]
        offsets_host[global_id] = local * region_rows
    counts = _to_device(counts_host, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device)
    offsets = _to_device(offsets_host, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device)

    ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
        x,
        w_gates,
        w_ups,
        w_downs,
        counts,
        idx,
        input_m_tiles=region_rows // TILE,
        core_grid=GRID,
        output=output,
        expert_region_offsets=offsets,
        read_x_at_offset=True,
    )
    actual = ttnn.to_torch(output)[0, 0].float()

    # Every expert is scored before anything asserts: a per-expert PCC list localises a wrong
    # region or a stale weight base far faster than the first failing slice does.
    scores = []
    for local, count in enumerate(counts_per_expert):
        if count == 0:
            continue
        base = local * region_rows
        gate, up, down = host_weights[local]
        reference = _reference(x_host[0, 0, base : base + count].float(), gate, up, down)
        rows = actual[base : base + count]
        if not torch.isfinite(rows).all():
            bad = ~torch.isfinite(rows)
            bad_rows = bad.any(dim=1).nonzero().flatten().tolist()
            bad_cols = bad.any(dim=0).nonzero().flatten().tolist()
            raise AssertionError(
                f"expert {local} (count {count}) non-finite: {int(bad.sum())} of {bad.numel()} elems; "
                f"rows {bad_rows[:8]}..{bad_rows[-4:] if len(bad_rows) > 8 else ''} (n={len(bad_rows)}); "
                f"cols {bad_cols[:8]}..{bad_cols[-4:] if len(bad_cols) > 8 else ''} (n={len(bad_cols)})"
            )
        passed, message = comp_pcc(reference, rows, 0.97)
        scores.append((local, count, passed, message))
    assert all(entry[2] for entry in scores), "per-expert PCC: " + "; ".join(
        f"expert {i} (count {c}): {m}" for i, c, _, m in scores
    )
