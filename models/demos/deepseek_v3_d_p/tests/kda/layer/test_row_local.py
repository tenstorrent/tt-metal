# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Row-local KDA (tt-blaze G53F-0250): ``ttKDA(sp_axis=None)`` on a multi-row mesh runs each mesh row as its own
sequence -- the row-local prefill waves give each row a different prompt. Every row's output and final state must match
the reference of ITS prompt, and rows must not mix (a row-swapped input swaps the rows' results)."""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda import kda_forward_reference
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    _device_shards,
    random_weights,
    reconstruct_convolution_at_sp_rank,
    reconstruct_state_at_sp_rank,
)
from models.demos.deepseek_v3_d_p.tt.kda.config import KDAProgramConfig
from models.demos.deepseek_v3_d_p.tt.kda.kda import ttKDA
from models.tt_transformers.tt.ccl import TT_CCL
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate, make_actual_start

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True),
    pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True),
]

ROW_AXIS, TP_AXIS = 0, 1


def _row_input(hidden_rows: torch.Tensor, mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
    """[rows, S, H] -> row r's devices hold prompt r as [1, S, H], replicated over the TP columns."""
    return ttnn.from_torch(
        hidden_rows,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=tuple(mesh_device.shape)),
    )


def _row_output(output_tt: ttnn.Tensor, mesh_device: ttnn.MeshDevice, row: int) -> torch.Tensor:
    """Row ``row``'s [1, S, H]: its TP shards concatenated over the hidden dim (the reduce-scatter's split)."""
    shards = _device_shards(output_tt)
    columns = tuple(mesh_device.shape)[1]
    parts = []
    for column in range(columns):
        shard = shards[row * columns + column]
        if shard.ndim == 4:
            shard = shard.reshape(shard.shape[0], shard.shape[-2], shard.shape[-1])
        parts.append(shard)
    return torch.cat(parts, dim=2)


@pytest.mark.parametrize("sequence", [256, 1024])
def test_row_local_rows_match_their_own_prompts(mesh_device: ttnn.MeshDevice, sequence: int) -> None:
    config = KDAConfig(hidden_size=128, num_heads=8, head_k_dim=32, head_v_dim=32, conv_kernel_size=4, norm_eps=1e-5)
    weights = random_weights(config)
    rows = tuple(mesh_device.shape)[ROW_AXIS]
    gen = torch.Generator().manual_seed(7311)
    prompts = torch.randn(rows, sequence, config.hidden_size, generator=gen).to(torch.bfloat16)
    layer = ttKDA(
        mesh_device,
        config,
        weights,
        tt_ccl=TT_CCL(mesh_device),
        sp_axis=None,
        tp_axis=TP_AXIS,
        program_config=KDAProgramConfig(gated_rms_output_dtype=ttnn.bfloat16),
        active_seq_len=sequence,
    )
    assert layer.sequence_parallel_size == 1 and layer.active_seq_len_local == sequence
    local_heads = config.num_heads // tuple(mesh_device.shape)[TP_AXIS]
    local_width = local_heads * config.head_k_dim

    def run(order):
        state0 = layer.allocate_state(batch_size=1)
        with ttnn.manage_config("throw_exception_on_fallback", True):
            out, state = layer.forward(
                _row_input(prompts[list(order)], mesh_device), state0, make_actual_start(layer.device)
            )
        results = []
        for row in range(rows):
            results.append(
                (
                    _row_output(out, mesh_device, row),
                    reconstruct_state_at_sp_rank(state.recurrent, mesh_device, ROW_AXIS, TP_AXIS, row),
                    reconstruct_convolution_at_sp_rank(
                        state.convolution, mesh_device, ROW_AXIS, TP_AXIS, row, local_width
                    ),
                )
            )
        return results

    straight = run(range(rows))
    for row in range(rows):
        ref_out, ref_state = kda_forward_reference(prompts[row : row + 1], weights, config)
        ref_conv = torch.cat((ref_state.q_convolution, ref_state.k_convolution, ref_state.v_convolution), dim=-1)
        out, rec, conv = straight[row]
        assert_accurate(
            ref_out.to(torch.bfloat16), out, name=f"row {row} output", pcc_threshold=0.999, nrmse_threshold=0.02
        )
        assert_accurate(
            ref_state.recurrent, rec, name=f"row {row} recurrent", pcc_threshold=0.999, nrmse_threshold=0.03
        )
        assert_accurate(
            ref_conv.to(torch.bfloat16), conv, name=f"row {row} conv", pcc_threshold=0.999, nrmse_threshold=0.004
        )
    # rows are independent: the reversed prompt order gives the same per-prompt results, bit for bit
    swapped = run(range(rows - 1, -1, -1))
    for row in range(rows):
        for a, b in zip(straight[row], swapped[rows - 1 - row]):
            assert torch.equal(a, b), f"prompt {row}: the row-local result depends on its row"
