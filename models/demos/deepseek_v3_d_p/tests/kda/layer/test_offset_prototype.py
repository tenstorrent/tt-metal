# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end mapping probe for the throwaway full-reshard offset path."""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda import KDAReferenceState, kda_forward_reference
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    random_weights,
    reconstruct_convolution_at_sp_rank,
    reconstruct_sp_tp_tensor,
    reconstruct_state_at_sp_rank,
)
from models.demos.deepseek_v3_d_p.tt.kda.config import KDAProgramConfig, KDARecurrenceProgramConfig
from models.demos.deepseek_v3_d_p.tt.kda.kda import ttKDA
from models.demos.deepseek_v3_d_p.tt.mla.utils import rotated_chip_positions
from models.tt_transformers.tt.ccl import TT_CCL
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate

pytestmark = run_for_blackhole()

_GLOBAL_SEQUENCE = 5120


def _physical_order(tensor: torch.Tensor, actual_start: int, sp_size: int) -> torch.Tensor:
    local_sequence = _GLOBAL_SEQUENCE // sp_size
    positions = rotated_chip_positions(actual_start, sp_size, local_sequence)
    relative_positions = torch.tensor(positions, dtype=torch.int64).flatten() - actual_start
    return tensor.index_select(1, relative_positions)


def _to_sp_input(hidden: torch.Tensor, mesh_device: ttnn.MeshDevice, sp_axis: int) -> ttnn.Tensor:
    mesh_dims: list[int | None] = [None, None]
    mesh_dims[sp_axis] = 1
    return ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=tuple(mesh_dims),
            mesh_shape=tuple(mesh_device.shape),
        ),
    )


def _run_offset_case(
    layer: ttKDA,
    mesh_device: ttnn.MeshDevice,
    hidden: torch.Tensor,
    golden_output: torch.Tensor,
    golden_state: KDAReferenceState,
    actual_start: int,
    tensor_parallel_axis: int,
) -> None:
    sequence_parallel_axis = 1 - tensor_parallel_axis
    sp_size = tuple(mesh_device.shape)[sequence_parallel_axis]
    local_sequence = _GLOBAL_SEQUENCE // sp_size
    physical_hidden = _physical_order(hidden, actual_start, sp_size)
    expected_physical_output = _physical_order(golden_output, actual_start, sp_size).to(torch.bfloat16)
    hidden_tt = _to_sp_input(physical_hidden, mesh_device, sequence_parallel_axis)
    initial_state = layer.allocate_state()

    with ttnn.manage_config("throw_exception_on_fallback", True):
        output_tt, state = layer.forward(hidden_tt, initial_state, actual_start=actual_start)

    actual_output = reconstruct_sp_tp_tensor(
        output_tt,
        mesh_device,
        sequence_parallel_axis,
        tensor_parallel_axis,
        tp_dim=2,
        sp_dim=1,
    )
    expected_convolution = torch.cat(
        (golden_state.q_convolution, golden_state.k_convolution, golden_state.v_convolution), dim=-1
    ).to(torch.bfloat16)
    assert_accurate(
        expected_physical_output,
        actual_output,
        name=f"offset={actual_start} physical output",
        pcc_threshold=0.999,
    )
    boundary_rank = (actual_start // local_sequence) % sp_size
    for sp_rank in {0, boundary_rank}:
        assert_accurate(
            golden_state.recurrent,
            reconstruct_state_at_sp_rank(
                state.recurrent,
                mesh_device,
                sequence_parallel_axis,
                tensor_parallel_axis,
                sp_rank,
            ),
            name=f"offset={actual_start} SP{sp_rank} recurrent state",
            pcc_threshold=0.999,
        )
        assert_accurate(
            expected_convolution,
            reconstruct_convolution_at_sp_rank(
                state.convolution,
                mesh_device,
                sequence_parallel_axis,
                tensor_parallel_axis,
                sp_rank,
                local_width=layer.config.num_heads * layer.config.head_k_dim,
            ),
            name=f"offset={actual_start} SP{sp_rank} convolution state",
            pcc_threshold=0.999,
        )


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device,tensor_parallel_axis,device_params",
    [
        pytest.param((1, 8), 0, fabric_1d_device_params(), id="SP8xTP1"),
        pytest.param((2, 4), 1, fabric_1d_device_params(), id="SP2xTP4"),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_offset_full_reshard_matches_natural_kda(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
    device_params: dict,
) -> None:
    del device_params
    config = KDAConfig(
        hidden_size=128,
        num_heads=8,
        head_k_dim=32,
        head_v_dim=32,
        conv_kernel_size=4,
        norm_eps=1e-5,
    )
    weights = random_weights(config)
    hidden = torch.randn(
        1,
        _GLOBAL_SEQUENCE,
        config.hidden_size,
        generator=torch.Generator().manual_seed(960),
    ).to(torch.bfloat16)
    golden_output, golden_state = kda_forward_reference(hidden, weights, config)
    sequence_parallel_axis = 1 - tensor_parallel_axis
    layer = ttKDA(
        mesh_device,
        config,
        weights,
        sp_axis=sequence_parallel_axis,
        tp_axis=tensor_parallel_axis,
        tt_ccl=None if tuple(mesh_device.shape)[tensor_parallel_axis] == 1 else TT_CCL(mesh_device),
        program_config=KDAProgramConfig(
            recurrence=KDARecurrenceProgramConfig(local_scan_strategy="grouped", summary_group_chunks=20),
            gated_rms_output_dtype=ttnn.bfloat16,
        ),
    )
    local_sequence = _GLOBAL_SEQUENCE // tuple(mesh_device.shape)[sequence_parallel_axis]

    for actual_start in (0, local_sequence, 960):
        _run_offset_case(
            layer,
            mesh_device,
            hidden,
            golden_output,
            golden_state,
            actual_start,
            tensor_parallel_axis,
        )
