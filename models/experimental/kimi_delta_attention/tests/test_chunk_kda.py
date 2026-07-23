# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Blackhole PCC tests for the chunk-parallel KDA operation."""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.experimental.kimi_delta_attention.reference import kda_recurrent_reference, l2_norm_reference

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True),
]


def _to_device(tensor: torch.Tensor, device: ttnn.Device, dtype: ttnn.DataType) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _assert_pcc(name: str, golden: torch.Tensor, actual: torch.Tensor, threshold: float = 0.999) -> None:
    passed, pcc = comp_pcc(golden, actual, pcc=threshold)
    max_abs = (golden.float() - actual.float()).abs().max().item()
    print(f"{name}: PCC={pcc:.6f}, max_abs={max_abs:.6e}")
    assert passed, f"{name} PCC {pcc:.6f} < {threshold}"


@pytest.mark.parametrize("sequence,heads,key_dim,value_dim", [(32, 2, 32, 32), (64, 32, 128, 128)])
def test_chunk_kda_pcc(
    device: ttnn.Device,
    sequence: int,
    heads: int,
    key_dim: int,
    value_dim: int,
) -> None:
    generator = torch.Generator().manual_seed(401 + sequence + heads)
    shape = (1, sequence, heads)
    q = torch.randn(*shape, key_dim, generator=generator)
    k = torch.randn(*shape, key_dim, generator=generator)
    v = torch.randn(*shape, value_dim, generator=generator)
    gate = -0.02 * torch.rand(*shape, key_dim, generator=generator)
    beta = torch.sigmoid(torch.randn(*shape, generator=generator))
    state = 0.02 * torch.randn(1, heads, key_dim, value_dim, generator=generator)
    golden_output, golden_state = kda_recurrent_reference(q, k, v, gate, beta, state)

    q_tt = _to_device(l2_norm_reference(q), device, ttnn.bfloat16)
    k_tt = _to_device(l2_norm_reference(k), device, ttnn.bfloat16)
    v_tt = _to_device(v, device, ttnn.bfloat16)
    gate_tt = _to_device(gate, device, ttnn.float32)
    beta_tt = _to_device(beta, device, ttnn.float32)
    state_tt = _to_device(state, device, ttnn.float32)
    with ttnn.manage_config("throw_exception_on_fallback", True):
        output_tt, final_state_tt = ttnn.transformer.chunk_kda(
            q_tt,
            k_tt,
            v_tt,
            gate_tt,
            beta_tt,
            initial_state=state_tt,
            output_final_state=True,
            chunk_size=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    actual_output = ttnn.to_torch(output_tt)
    actual_state = ttnn.to_torch(final_state_tt)
    label = f"H={heads},K={key_dim},V={value_dim},T={sequence}"
    _assert_pcc(f"{label} output", golden_output, actual_output)
    _assert_pcc(f"{label} state", golden_state, actual_state)
