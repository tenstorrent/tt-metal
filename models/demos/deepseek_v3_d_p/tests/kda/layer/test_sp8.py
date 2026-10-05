# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""KDA layer at SP8 x TP1 on an 8x1 mesh with FABRIC_1D, 640 tokens per SP rank."""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda import kda_forward_reference
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    collect_mesh_accuracy_and_determinism_results,
    random_weights,
    reconstruct_convolution_at_sp_rank,
    reconstruct_sp_tp_tensor,
    reconstruct_state_at_sp_rank,
)
from models.demos.deepseek_v3_d_p.tt.kda.config import KDAProgramConfig, KDARecurrenceProgramConfig
from models.demos.deepseek_v3_d_p.tt.kda.kda import ttKDA
from models.tt_transformers.tt.ccl import TT_CCL
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate, make_actual_start

SP_AXIS = 0
TP_AXIS = 1

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("mesh_device", [(8, 1)], indirect=True),
    pytest.mark.parametrize("device_params", [fabric_1d_device_params()], indirect=True),
]


@pytest.mark.parametrize(
    "sequence,summary_group_chunks",
    [
        pytest.param(8 * ttnn.TILE_SIZE, 1, id="SP8xTP1-32-per-rank"),
        pytest.param(5120, 20, id="SP8xTP1-640-per-rank"),
    ],
)
def test_sp8_tp1_matches_reference(mesh_device: ttnn.MeshDevice, sequence: int, summary_group_chunks: int) -> None:
    config = KDAConfig(hidden_size=128, num_heads=8, head_k_dim=32, head_v_dim=32, conv_kernel_size=4, norm_eps=1e-5)
    weights = random_weights(config)
    hidden = torch.randn(1, sequence, config.hidden_size, generator=torch.Generator().manual_seed(8101)).to(
        torch.bfloat16
    )
    expected_output, expected_state = kda_forward_reference(hidden, weights, config)
    expected_convolution = torch.cat(
        (expected_state.q_convolution, expected_state.k_convolution, expected_state.v_convolution), dim=-1
    ).to(torch.bfloat16)
    layer = ttKDA(
        mesh_device,
        config,
        weights,
        tt_ccl=TT_CCL(mesh_device),
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        program_config=KDAProgramConfig(
            recurrence=KDARecurrenceProgramConfig(summary_group_chunks=summary_group_chunks)
        ),
        active_seq_len=sequence,
    )
    hidden_tt = ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(1, None), mesh_shape=tuple(mesh_device.shape)),
    )
    sp_size = tuple(mesh_device.shape)[SP_AXIS]
    local_width = config.num_heads * config.head_k_dim

    def run() -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        state = layer.allocate_state(batch_size=1)
        with ttnn.manage_config("throw_exception_on_fallback", True):
            output_tt, state = layer.forward(hidden_tt, state, make_actual_start(layer.device))
        return output_tt, state.recurrent, state.convolution

    (output_tt, recurrent_tt, convolution_tt), mismatch_markers = collect_mesh_accuracy_and_determinism_results(run)
    actual_output = reconstruct_sp_tp_tensor(output_tt, mesh_device, SP_AXIS, TP_AXIS, tp_dim=2, sp_dim=1)
    assert_accurate(expected_output.to(actual_output.dtype), actual_output, name="SP8 output", pcc_threshold=0.9995)
    for sp_rank in range(sp_size):
        actual_recurrent = reconstruct_state_at_sp_rank(recurrent_tt, mesh_device, SP_AXIS, TP_AXIS, sp_rank)
        actual_convolution = reconstruct_convolution_at_sp_rank(
            convolution_tt, mesh_device, SP_AXIS, TP_AXIS, sp_rank, local_width
        )
        assert_accurate(
            expected_state.recurrent.to(actual_recurrent.dtype),
            actual_recurrent,
            name=f"SP8 recurrent sp_rank={sp_rank}",
            pcc_threshold=0.9995,
        )
        assert_accurate(
            expected_convolution.to(actual_convolution.dtype),
            actual_convolution,
            name=f"SP8 convolution sp_rank={sp_rank}",
            pcc_threshold=0.9995,
        )
    assert all(marker.item() == 0 for marker in mismatch_markers), "SP8 layer is not bit-identical across runs"
