# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""moe_fused_swiglu against unified_routed_expert_moe, one harness, one axis.

The two ops were previously timed separately -- different profilers, different
dispatch axes, and the fused side on ND-sharded weights while the composite ran
interleaved. Those offsets sit in the same band as the at-scale deltas, so the
at-scale verdict was not decidable. Here both ops run back to back in one
process on one dispatch axis over the SAME interleaved weight tensors, so a
single tracy CSV carries both and the only difference left is the kernel.

Weights are DRAM-interleaved because the composite accepts nothing else. That
costs the fused op its ND-shard placement win (worth ~8-14% at low M), so these
numbers are the FLOOR of the fused advantage, not what a shipped config sees.
"""

import os
import time

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config

TILE = 32
GRID = ttnn.CoreCoord(11, 8)
CAPACITY = 5120
REPS = int(os.environ.get("MOE_FUSED_SWIGLU_VS_REPS", "9"))
COUNTS = tuple(
    int(v) for v in os.environ.get("MOE_FUSED_SWIGLU_VS_COUNTS", "0,64,128,256,512,1024,2048,4096,5120").split(",")
)


class KimiK3SituConfig:
    """Kimi K3 routed-expert shape; kept local, the kernel sweep does not import the model package."""

    EMB_SIZE = 3584
    MOE_INTERMEDIATE_SIZE = 3072
    NUM_ROUTED_EXPERTS = 384


MODEL_CASES = (
    (KimiK26Config, ttnn.RoutedExpertActivation.Silu, "kimi-k26"),
    (GLM51Config, ttnn.RoutedExpertActivation.Silu, "glm-51"),
    (KimiK3SituConfig, ttnn.RoutedExpertActivation.SituGlu, "k3-situ"),
    (DeepSeekV4ProConfig, ttnn.RoutedExpertActivation.Silu, "ds-v4-pro"),
    (DeepSeekV4FlashConfig, ttnn.RoutedExpertActivation.Silu, "ds-v4-flash"),
)


def _to_device(tensor, dtype, layout, device):
    return ttnn.from_torch(
        tensor.contiguous(), dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _bench(call, device):
    """Per-iteration wall clock with the queue saturated; warmup excluded."""
    call()
    ttnn.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(REPS):
        call()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - start) / REPS * 1e6


@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
@pytest.mark.parametrize("model_config,activation,name", MODEL_CASES, ids=[c[2] for c in MODEL_CASES])
@pytest.mark.skipif(not is_blackhole(), reason="moe_fused_swiglu is Blackhole-only")
def test_fused_vs_unified(device, model_config, activation, name):
    torch.manual_seed(20260827)
    emb = model_config.EMB_SIZE
    hidden = model_config.MOE_INTERMEDIATE_SIZE
    num_experts = model_config.NUM_ROUTED_EXPERTS
    global_expert_id = 137 % num_experts

    weights = [
        _to_device(torch.randn(shape, dtype=torch.bfloat16) * 2.0e-2, ttnn.bfloat4_b, ttnn.TILE_LAYOUT, device)
        for shape in ((emb, hidden), (emb, hidden), (hidden, emb))
    ]
    x = _to_device(
        torch.randn((1, 1, CAPACITY, emb), dtype=torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device
    )
    expert_ids = _to_device(
        torch.tensor([global_expert_id], dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device
    )
    offsets = _to_device(torch.zeros(num_experts, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device)

    counts = {}
    for c in COUNTS:
        host = torch.zeros(num_experts, dtype=torch.int32)
        host[global_expert_id] = c
        counts[c] = _to_device(host, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device)

    fused_op = ttnn.experimental.deepseek_prefill.moe_fused_swiglu
    unified_op = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe

    rows = []
    for c in COUNTS:
        fused_us = _bench(
            lambda: fused_op(
                x,
                [weights[0]],
                [weights[1]],
                [weights[2]],
                counts[c],
                expert_ids,
                input_m_tiles=CAPACITY // TILE,
                core_grid=GRID,
                activation=activation,
            ),
            device,
        )
        unified_us = _bench(
            lambda: unified_op(
                x,
                offsets,
                counts[c],
                expert_ids,
                [weights[0]],
                [weights[1]],
                [weights[2]],
                CAPACITY,
                activation=activation,
            ),
            device,
        )
        rows.append((c, fused_us, unified_us))

    print(f"\n### {name}  emb={emb} hidden={hidden}  (interleaved weights, COL, wall clock, {REPS} reps)")
    print(f"{'tokens':>7} {'fused us':>10} {'unified us':>11} {'speedup':>8}")
    for c, f_us, u_us in rows:
        print(f"{c:>7} {f_us:>10.1f} {u_us:>11.1f} {u_us / f_us:>7.2f}x")
