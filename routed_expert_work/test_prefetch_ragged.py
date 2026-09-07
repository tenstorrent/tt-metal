# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Correctness gate for the CROSS-EXPERT weight prefetch, on the case that can actually break it.

The shipped functional suite is single-expert (`counts = idx_tensor([active_tokens])`), so it never
enters the prefetch path at all. Two properties have to hold and neither is observable with the
uniform, all-non-zero, identical-weight dispatches the perf harness uses:

  1. Expert i+1 must consume ITS OWN weights. With identical weights per expert an off-by-one is
     invisible, so every expert here gets its own scale.
  2. An expert with count 0 never enters the block loop, so a prefetch issued FOR it is never
     consumed and never barriered. If the "already prefetched" flag is not retired there, its reads
     race the next real read into the same slot and the expert AFTER it silently runs on the skipped
     expert's weights. `counts = [c, 0, c]` is exactly that shape: expert 0 prefetches for the
     skipped expert 1, and expert 2 must still read its own.

Every expert writes the same rows, so the graded output is the last non-zero expert's -- which is
the one both failure modes corrupt. Run with and without
MOE_FUSED_SWIGLU_DEFINES="MOE_WG_PREFETCH=1,MOE_WU_PREFETCH=1"; the numbers must be identical.
"""
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import ACTIVATION_SILU, TorchExpert
from tests.ttnn.utils_for_testing import comp_pcc

ALLOCATED = 5120
EMB, HID = 7168, 2048
GRID = ttnn.CoreCoord(11, 8)

# (counts per local expert, id) -- zeros in front, middle and back, and a ragged tail count.
_CASES = [
    ([256, 0, 256], "zero_middle"),
    ([0, 251, 0, 768], "zero_front_and_back"),
    ([0, 0, 0], "all_zero"),
    ([251, 768, 3001], "ragged_dense"),
]


@pytest.mark.parametrize("counts, case_id", _CASES, ids=[c[1] for c in _CASES])
def test_prefetch_ragged(device, counts, case_id):
    n_experts = len(counts)
    torch.manual_seed(42)
    base = {
        "gate_proj": torch.randn(HID, EMB, dtype=torch.float32) * 0.02,
        "up_proj": torch.randn(HID, EMB, dtype=torch.float32) * 0.02,
        "down_proj": torch.randn(EMB, HID, dtype=torch.float32) * 0.02,
    }

    def scale(e):
        return 1.0 + e / 4.0

    def to_dev(t, dtype, layout):
        return ttnn.from_torch(
            t.contiguous(), dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    w_gate = [to_dev((base["gate_proj"] * scale(e)).T, ttnn.bfloat4_b, ttnn.TILE_LAYOUT) for e in range(n_experts)]
    w_up = [to_dev((base["up_proj"] * scale(e)).T, ttnn.bfloat4_b, ttnn.TILE_LAYOUT) for e in range(n_experts)]
    w_down = [to_dev((base["down_proj"] * scale(e)).T, ttnn.bfloat4_b, ttnn.TILE_LAYOUT) for e in range(n_experts)]

    x = torch.zeros(ALLOCATED, EMB, dtype=torch.float32)
    max_count = max(counts)
    if max_count:
        x[:max_count] = torch.randn(max_count, EMB, dtype=torch.float32)
    tt_x = to_dev(x.reshape(1, 1, ALLOCATED, EMB), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)

    idx = to_dev(torch.tensor(list(range(n_experts)), dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    tt_counts = to_dev(torch.tensor(counts, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

    out = ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
        tt_x,
        w_gate,
        w_up,
        w_down,
        tt_counts,
        idx,
        input_m_tiles=ALLOCATED // ttnn.TILE_SIZE,
        core_grid=GRID,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        ),
    )
    got = ttnn.to_torch(out)[0, 0]

    # The LAST expert with work owns the graded rows.
    last = max((e for e, c in enumerate(counts) if c), default=None)
    if last is None:
        logger.info(f"PFRAGGED {case_id}: no expert has work, dispatch-only")
        print(f"PFRAGGED {case_id}: dispatch-only OK", flush=True)
        return
    c = counts[last]
    weights = {k: v * scale(last) for k, v in base.items()}
    with torch.no_grad():
        ref = TorchExpert(EMB, HID, weights, activation=ACTIVATION_SILU)(x[:c])
    _, pcc = comp_pcc(ref, got[:c])
    line = f"PFRAGGED {case_id}: counts={counts} graded_expert={last} count={c} pcc={float(pcc):.6f}"
    logger.info(line)
    print(line, flush=True)
    assert float(pcc) > 0.97, line
