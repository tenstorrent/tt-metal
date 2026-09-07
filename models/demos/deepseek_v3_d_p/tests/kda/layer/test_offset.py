# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""KDA correctness when MLA supplies an offset-rotated, block-cyclic activation.

The layer must be equivalent to processing ``[actual_start, actual_start + G)`` in
natural chronological order, so every case here compares against the natural-order
reference after undoing MLA's row permutation. Output and BOTH replacement carries
are checked, because an ordering bug can leave the output plausible while
corrupting the state that feeds the next chunk.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda import kda_forward_reference
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    collect_mesh_accuracy_and_determinism_results,
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

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True),
    pytest.mark.parametrize(
        "device_params",
        [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
        indirect=True,
    ),
]

SEQUENCE = 1280
PCC_THRESHOLD = 0.999


def _mla_row_permutation(actual_start: int, sp_size: int, local_rows: int) -> torch.Tensor:
    """Natural-order index carried by each MLA row, flattened in chip-major order."""
    positions = rotated_chip_positions(actual_start, sp_size, local_rows)
    return torch.tensor([position - actual_start for chip in positions for position in chip])


def _to_sp_input(hidden: torch.Tensor, mesh_device: ttnn.MeshDevice, sp_axis: int) -> ttnn.Tensor:
    mesh_dims = [None, None]
    mesh_dims[sp_axis] = 1
    return ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=tuple(mesh_dims), mesh_shape=tuple(mesh_device.shape)),
    )


def _build_layer(mesh_device, config, weights, sp_axis, tp_axis) -> ttKDA:
    return ttKDA(
        mesh_device,
        config,
        weights,
        tt_ccl=TT_CCL(mesh_device),
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        program_config=KDAProgramConfig(
            recurrence=KDARecurrenceProgramConfig(summary_group_chunks=8),
            gated_rms_output_dtype=ttnn.bfloat16,
            output_projection_math_fidelity=ttnn.MathFidelity.HiFi2,
        ),
    )


def _reference_case() -> tuple[KDAConfig, object, torch.Tensor, torch.Tensor, object]:
    config = KDAConfig(
        hidden_size=128,
        num_heads=8,
        head_k_dim=32,
        head_v_dim=32,
        conv_kernel_size=4,
        norm_eps=1e-5,
    )
    weights = random_weights(config)
    hidden = torch.randn(1, SEQUENCE, config.hidden_size, generator=torch.Generator().manual_seed(4211)).to(
        torch.bfloat16
    )
    expected_output, expected_state = kda_forward_reference(hidden, weights, config)
    return config, weights, hidden, expected_output.to(torch.bfloat16), expected_state


def _assert_matches_reference(
    *,
    output_tt,
    state,
    permutation,
    expected_output,
    expected_state,
    mesh_device,
    sp_axis,
    tp_axis,
    config,
    label: str,
) -> None:
    """Undo MLA's row permutation, then compare output and both carries."""
    rotated_output = reconstruct_sp_tp_tensor(output_tt, mesh_device, sp_axis, tp_axis, tp_dim=2, sp_dim=1)
    natural_output = torch.empty_like(rotated_output)
    natural_output[:, permutation, :] = rotated_output

    assert_accurate(expected_output, natural_output, name=f"{label} output", pcc_threshold=PCC_THRESHOLD)

    expected_convolution = torch.cat(
        (expected_state.q_convolution, expected_state.k_convolution, expected_state.v_convolution), dim=-1
    ).to(torch.bfloat16)
    local_heads = config.num_heads // tuple(mesh_device.shape)[tp_axis]
    local_width = local_heads * config.head_k_dim

    for sp_rank in range(tuple(mesh_device.shape)[sp_axis]):
        assert_accurate(
            expected_state.recurrent,
            reconstruct_state_at_sp_rank(state.recurrent, mesh_device, sp_axis, tp_axis, sp_rank),
            name=f"{label} sp_rank={sp_rank} recurrent",
            pcc_threshold=PCC_THRESHOLD,
        )
        assert_accurate(
            expected_convolution,
            reconstruct_convolution_at_sp_rank(state.convolution, mesh_device, sp_axis, tp_axis, sp_rank, local_width),
            name=f"{label} sp_rank={sp_rank} convolution",
            pcc_threshold=PCC_THRESHOLD,
        )


@pytest.mark.parametrize("tensor_parallel_axis", [0, 1])
def test_device_boundary_offsets_match_natural_order(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
) -> None:
    """Offsets landing between chips rotate causal order without splitting a chip.

    Every boundary chip is exercised, so a rotation that only happens to work for
    chip zero fails here.
    """
    sp_axis = 1 - tensor_parallel_axis
    sp_size = tuple(mesh_device.shape)[sp_axis]
    local_rows = SEQUENCE // sp_size

    config, weights, hidden, expected_output, expected_state = _reference_case()
    layer = _build_layer(mesh_device, config, weights, sp_axis, tensor_parallel_axis)

    for boundary_chip in range(sp_size):
        actual_start = boundary_chip * local_rows
        permutation = _mla_row_permutation(actual_start, sp_size, local_rows)
        hidden_tt = _to_sp_input(hidden[:, permutation, :], mesh_device, sp_axis)
        with ttnn.manage_config("throw_exception_on_fallback", True):
            output_tt, state = layer.forward(hidden_tt, layer.allocate_state(batch_size=1), actual_start)
        _assert_matches_reference(
            output_tt=output_tt,
            state=state,
            permutation=permutation,
            expected_output=expected_output,
            expected_state=expected_state,
            mesh_device=mesh_device,
            sp_axis=sp_axis,
            tp_axis=tensor_parallel_axis,
            config=config,
            label=f"tp_axis={tensor_parallel_axis} start={actual_start}",
        )


@pytest.mark.parametrize("tensor_parallel_axis", [0, 1])
def test_zero_offset_is_deterministic_and_matches_reference(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
) -> None:
    """Offset zero must stay bit-identical run to run and equal to the reference."""
    sp_axis = 1 - tensor_parallel_axis
    sp_size = tuple(mesh_device.shape)[sp_axis]
    local_rows = SEQUENCE // sp_size

    config, weights, hidden, expected_output, expected_state = _reference_case()
    layer = _build_layer(mesh_device, config, weights, sp_axis, tensor_parallel_axis)
    hidden_tt = _to_sp_input(hidden, mesh_device, sp_axis)

    def run():
        with ttnn.manage_config("throw_exception_on_fallback", True):
            output_tt, state = layer.forward(hidden_tt, layer.allocate_state(batch_size=1), 0)
        return output_tt, state.recurrent, state.convolution

    (output_tt, recurrent_tt, convolution_tt), mismatch_markers = collect_mesh_accuracy_and_determinism_results(run)
    assert all(marker.item() == 0 for marker in mismatch_markers), "offset-aware KDA is not bit-identical"

    _assert_matches_reference(
        output_tt=output_tt,
        state=SimpleNamespace(recurrent=recurrent_tt, convolution=convolution_tt),
        permutation=_mla_row_permutation(0, sp_size, local_rows),
        expected_output=expected_output,
        expected_state=expected_state,
        mesh_device=mesh_device,
        sp_axis=sp_axis,
        tp_axis=tensor_parallel_axis,
        config=config,
        label=f"tp_axis={tensor_parallel_axis} start=0",
    )
