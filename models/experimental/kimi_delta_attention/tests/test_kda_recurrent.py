# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Blackhole correctness tests for the fused recurrent KDA operation."""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.experimental.kimi_delta_attention.reference import kda_recurrent_reference, l2_norm_reference

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True),
]


def _to_device(tensor: torch.Tensor, device: ttnn.Device) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _assert_pcc(name: str, golden: torch.Tensor, actual: torch.Tensor, threshold: float = 0.999) -> None:
    passed, pcc = comp_pcc(golden, actual, pcc=threshold)
    max_abs = (golden.float() - actual.float()).abs().max().item()
    print(f"{name}: PCC={pcc:.6f}, max_abs={max_abs:.6e}")
    assert passed, f"{name} PCC {pcc:.6f} < {threshold}"


def _run_case(device: ttnn.Device, heads: int, key_dim: int, value_dim: int, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(1, 1, heads, key_dim, generator=generator)
    k = torch.randn(1, 1, heads, key_dim, generator=generator)
    v = torch.randn(1, 1, heads, value_dim, generator=generator)
    gate = -0.2 * torch.rand(1, 1, heads, key_dim, generator=generator)
    beta = torch.sigmoid(torch.randn(1, 1, heads, generator=generator))
    state = 0.05 * torch.randn(1, heads, key_dim, value_dim, generator=generator)
    golden_output, golden_state = kda_recurrent_reference(q, k, v, gate, beta, state)

    q_scaled = l2_norm_reference(q) * (key_dim**-0.5)
    k_unit = l2_norm_reference(k)
    q_tt = _to_device(q_scaled.reshape(heads, 1, key_dim), device)
    k_tt = _to_device(k_unit.reshape(heads, 1, key_dim), device)
    v_tt = _to_device(v.reshape(heads, 1, value_dim), device)
    decay_tt = _to_device(gate.exp().reshape(heads, key_dim, 1), device)
    beta_tt = _to_device(beta.reshape(heads, 1, 1), device)
    state_tt = _to_device(state.reshape(heads, key_dim, value_dim), device)

    with ttnn.manage_config("throw_exception_on_fallback", True):
        output_tt, final_state_tt = ttnn.transformer.kda_recurrent_step(
            q_tt,
            k_tt,
            v_tt,
            decay_tt,
            beta_tt,
            state_tt,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    actual_output = ttnn.to_torch(output_tt).reshape(1, 1, heads, value_dim)
    actual_state = ttnn.to_torch(final_state_tt).reshape(1, heads, key_dim, value_dim)
    label = f"H={heads},K={key_dim},V={value_dim},seed={seed}"
    _assert_pcc(f"{label} output", golden_output, actual_output)
    _assert_pcc(f"{label} state", golden_state, actual_state)


@pytest.mark.parametrize(
    "heads,key_dim,value_dim",
    [
        (2, 32, 32),
        (32, 128, 128),
    ],
)
def test_fused_kda_recurrent_pcc_and_program_cache(
    device: ttnn.Device,
    heads: int,
    key_dim: int,
    value_dim: int,
) -> None:
    _run_case(device, heads, key_dim, value_dim, seed=211)
    _run_case(device, heads, key_dim, value_dim, seed=223)
