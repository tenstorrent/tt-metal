# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc

EMB = 3584
HIDDEN = 3072
COUNT = 32
GRID = ttnn.CoreCoord(11, 8)
GLOBAL_EXPERT_ID = 137


def _to_device(tensor, dtype, layout, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        tensor.contiguous(),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config,
    )


def _situ_glu(gate, up):
    gate = gate.float()
    up = up.float()
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(up / 25.0))


@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
@pytest.mark.skipif(not is_blackhole(), reason="moe_fused_swiglu is Blackhole-only")
@pytest.mark.parametrize("weight_scale", [2.0e-2, 2.0e-1], ids=["model-scale", "saturation-scale"])
def test_moe_fused_situ_glu_3584x3072_rm_bf16_bfp4(device, weight_scale):
    """Requested Kimi K3 shape: RM bf16 input, bfp4 weights, SiTU-GLU in the fused kernel."""
    torch.manual_seed(20260819)
    x_host = torch.randn((1, 1, COUNT, EMB), dtype=torch.bfloat16)
    host_weights = [
        torch.randn(shape, dtype=torch.bfloat16) * weight_scale
        for shape in ((EMB, HIDDEN), (EMB, HIDDEN), (HIDDEN, EMB))
    ]

    x = _to_device(x_host, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device)
    weights = [_to_device(weight, ttnn.bfloat4_b, ttnn.TILE_LAYOUT, device) for weight in host_weights]

    counts_host = torch.zeros(384, dtype=torch.int32)
    counts_host[GLOBAL_EXPERT_ID] = COUNT
    counts = _to_device(counts_host, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device)
    offsets = _to_device(torch.zeros(384, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device)
    expert_ids = _to_device(
        torch.tensor([GLOBAL_EXPERT_ID], dtype=torch.int32),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
    )

    output = ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
        x,
        [weights[0]],
        [weights[1]],
        [weights[2]],
        counts,
        expert_ids,
        input_m_tiles=COUNT // 32,
        core_grid=GRID,
        activation=ttnn.RoutedExpertActivation.SituGlu,
    )
    actual = ttnn.to_torch(output)[0, 0, :COUNT].float()
    composite_output = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
        x,
        offsets,
        counts,
        expert_ids,
        [weights[0]],
        [weights[1]],
        [weights[2]],
        COUNT,
        activation=ttnn.RoutedExpertActivation.SituGlu,
    )
    composite_actual = ttnn.to_torch(composite_output)[0, 0, :COUNT].float()
    gate = x_host[0, 0].float() @ host_weights[0].float()
    up = x_host[0, 0].float() @ host_weights[1].float()
    reference = _situ_glu(gate, up) @ host_weights[2].float()

    assert torch.isfinite(actual).all()
    assert torch.isfinite(composite_actual).all()
    assert_with_pcc(reference, actual, pcc=0.97)
    assert_with_pcc(reference, composite_actual, pcc=0.97)
    # Cross-check between two DISTINCT kernels: moe_fused_swiglu against the
    # composite's own. They compute the same math but block and accumulate
    # differently, so this is a genuine agreement bound, not a self-comparison.
    assert_with_pcc(actual, composite_actual, pcc=0.999)
