# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Minimal reproduction (tt_metal_tracker-q171.2.11): moe_fused_swiglu's output for a token depends on how many
tokens its expert has.

One chip, one expert at GLM-5.3's routed-expert shape (GLM53Config: emb 6144, moe intermediate 2048), bfloat4_b
weights, the op called as test_moe_fused_swiglu.py calls it. The expert's first ``SHARED`` dispatch rows are the
same in every run; only the runtime count (the rows after them, real or not) changes. A count-independent kernel
would give bit-identical shared rows. Each run is also graded against the fp32 golden over its shared rows, so the
log shows whether a difference is within the op's own accuracy.

In TtMoe the count is the number of tokens routed to the expert in the chunk, so a token's routed-expert output
depends on the other tokens of the chunk (q171.2.11: the eager forward with padding awareness off routes the
padding rows and changes the real rows' MoE output; with every expert on unified_routed_expert_moe it does not).
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config

GRID = ttnn.CoreCoord(11, 8)  # as test_moe_fused_swiglu.py
ALLOCATED = 5120
SHARED = 96


def comp_pcc(a: torch.Tensor, b: torch.Tensor) -> tuple[bool, float]:
    a0, b0 = a.double().flatten() - a.double().mean(), b.double().flatten() - b.double().mean()
    return True, float(a0 @ b0 / (a0.norm() * b0.norm()))


COUNTS = (96, 128, 160, 192, 224, 256, 288, 320, 512, 1024, 3200)


@pytest.mark.skipif(not is_blackhole(), reason="moe_fused_swiglu is Blackhole-only")
def test_moe_fused_swiglu_count_dependence(device):
    emb, hidden, scale = GLM53Config.EMB_SIZE, GLM53Config.MOE_INTERMEDIATE_SIZE, 0.02
    torch.manual_seed(0)
    # torch.nn.Linear convention (out_features, in_features), as test_single_routed_expert.expert_weights draws them.
    weights = {
        "gate_proj": torch.randn(hidden, emb) * scale,
        "up_proj": torch.randn(hidden, emb) * scale,
        "down_proj": torch.randn(emb, hidden) * scale,
    }
    rows = torch.randn(ALLOCATED, emb, dtype=torch.float32)  # every count's active rows are a prefix of these

    def dev(t, dtype, layout):
        return ttnn.from_torch(
            t.contiguous(), dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    w = [dev(weights[k].T, ttnn.bfloat4_b, ttnn.TILE_LAYOUT) for k in ("gate_proj", "up_proj", "down_proj")]
    idx = dev(torch.tensor([0], dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    with torch.no_grad():
        x = rows[:SHARED]
        golden = (torch.nn.functional.silu(x @ weights["gate_proj"].T) * (x @ weights["up_proj"].T)) @ weights[
            "down_proj"
        ].T

    def run(count: int) -> torch.Tensor:
        x_host = torch.zeros(ALLOCATED, emb)
        x_host[:count] = rows[:count]
        out = ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
            dev(x_host.reshape(1, 1, ALLOCATED, emb), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
            [w[0]],
            [w[1]],
            [w[2]],
            dev(torch.tensor([count], dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            idx,
            input_m_tiles=ALLOCATED // 32,
            core_grid=GRID,
            activation=ttnn.RoutedExpertActivation.Silu,
        )
        shared = ttnn.to_torch(out)[0, 0, :SHARED].float()
        _, pcc = comp_pcc(golden, shared)
        logger.info(f"MOE_FUSED_COUNT count={count}: shared rows vs fp32 golden PCC {pcc:.6f}")
        return shared

    outs = {count: run(count) for count in COUNTS}
    repeat = run(COUNTS[0])
    logger.info(f"MOE_FUSED_COUNT count={COUNTS[0]} repeated: shared rows equal {torch.equal(repeat, outs[COUNTS[0]])}")

    # Which counts give bit-identical shared rows: classes of equal outputs, in count order.
    classes = []
    for count in COUNTS:
        for cls in classes:
            if torch.equal(outs[cls[0]], outs[count]):
                cls.append(count)
                break
        else:
            classes.append([count])
    logger.info(f"MOE_FUSED_COUNT classes of bit-identical shared rows: {classes}")

    base = outs[COUNTS[0]]
    for count in COUNTS[1:]:
        d = (outs[count] - base).abs()
        _, pcc = comp_pcc(base, outs[count])
        logger.info(
            f"MOE_FUSED_COUNT count={count} vs count={COUNTS[0]}: shared rows equal {torch.equal(outs[count], base)}, "
            f"rows differing {int((d.amax(-1) > 0).sum())}/{SHARED}, max|diff| {float(d.max()):.4g}, "
            f"mean|out| {float(base.abs().mean()):.4g}, PCC {pcc:.6f}"
        )
