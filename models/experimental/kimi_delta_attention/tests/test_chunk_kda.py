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


@pytest.mark.parametrize(
    "sequence,heads,key_dim,value_dim,flat_v,flat_qk,flat_g,math_fidelity",
    [
        (32, 2, 32, 32, False, False, False, None),
        (32, 2, 32, 32, True, True, True, None),
        (64, 32, 128, 128, False, False, False, None),
        (64, 32, 128, 128, True, False, True, None),
        (64, 32, 128, 128, True, True, True, "HiFi2"),
        (256, 4, 128, 128, True, True, True, None),
        (512, 4, 128, 128, True, True, True, None),
    ],
)
def test_chunk_kda_pcc(
    device: ttnn.Device,
    sequence: int,
    heads: int,
    key_dim: int,
    value_dim: int,
    flat_v: bool,
    flat_qk: bool,
    flat_g: bool,
    math_fidelity: str | None,
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

    q_input = q.reshape(1, sequence, heads * key_dim) if flat_qk else l2_norm_reference(q)
    k_input = k.reshape(1, sequence, heads * key_dim) if flat_qk else l2_norm_reference(k)
    q_tt = _to_device(q_input, device, ttnn.bfloat16)
    k_tt = _to_device(k_input, device, ttnn.bfloat16)
    v_input = v.reshape(1, sequence, heads * value_dim) if flat_v else v
    v_tt = _to_device(v_input, device, ttnn.bfloat16)
    gate_input = gate.reshape(1, sequence, heads * key_dim) if flat_g else gate
    gate_tt = _to_device(gate_input, device, ttnn.float32)
    beta_tt = _to_device(beta, device, ttnn.float32)
    state_tt = _to_device(state, device, ttnn.float32)
    compute_kernel_config = (
        ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=getattr(ttnn.MathFidelity, math_fidelity), fp32_dest_acc_en=True
        )
        if math_fidelity is not None
        else None
    )
    with ttnn.manage_config("throw_exception_on_fallback", True):
        output_tt, final_state_tt = ttnn.transformer.chunk_kda(
            q_tt,
            k_tt,
            v_tt,
            gate_tt,
            beta_tt,
            initial_state=state_tt,
            output_final_state=True,
            output_head_major=flat_qk,
            chunk_size=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=compute_kernel_config,
        )

    actual_output = ttnn.to_torch(output_tt)
    if flat_qk:
        actual_output = actual_output.reshape(1, heads, sequence, value_dim).permute(0, 2, 1, 3)
    actual_state = ttnn.to_torch(final_state_tt)
    label = f"H={heads},K={key_dim},V={value_dim},T={sequence},flat_v={flat_v},flat_qk={flat_qk},flat_g={flat_g},math_fidelity={math_fidelity}"
    _assert_pcc(f"{label} output", golden_output, actual_output)
    _assert_pcc(f"{label} state", golden_state, actual_state)


@pytest.mark.parametrize("sequence", [512, 640])
def test_chunk_kda_affine_summary_matches_final_state(device: ttnn.Device, sequence: int) -> None:
    """A span summary must reproduce the recurrence for any incoming state."""
    heads, dim = 2, 32
    generator = torch.Generator().manual_seed(1138)
    q = torch.randn(1, sequence, heads * dim, generator=generator)
    k = torch.randn(1, sequence, heads * dim, generator=generator)
    v = torch.randn(1, sequence, heads * dim, generator=generator)
    gate = -0.02 * torch.rand(1, sequence, heads * dim, generator=generator)
    beta = torch.sigmoid(torch.randn(1, sequence, heads, generator=generator))
    initial_state = 0.02 * torch.randn(1, heads, dim, dim, generator=generator)

    q_tt = _to_device(q, device, ttnn.bfloat16)
    k_tt = _to_device(k, device, ttnn.bfloat16)
    v_tt = _to_device(v, device, ttnn.bfloat16)
    gate_tt = _to_device(gate, device, ttnn.float32)
    beta_tt = _to_device(beta, device, ttnn.float32)
    initial_state_tt = _to_device(initial_state, device, ttnn.float32)
    with ttnn.manage_config("throw_exception_on_fallback", True):
        transform_a_tt, transform_b_tt = ttnn.transformer.chunk_kda_affine_summary(
            q_tt,
            k_tt,
            v_tt,
            gate_tt,
            beta_tt,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        output_tt, final_state_tt = ttnn.transformer.chunk_kda(
            q_tt,
            k_tt,
            v_tt,
            gate_tt,
            beta_tt,
            initial_state=initial_state_tt,
            output_final_state=True,
            output_head_major=True,
            chunk_size=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    assert final_state_tt is not None
    groups = sequence // (256 if sequence % 256 == 0 else 128)
    affine_threshold = 0.999 if sequence % 256 == 0 else 0.995
    grouped_prep = ttnn.transformer.chunk_kda_group_prepare(
        q_tt,
        k_tt,
        v_tt,
        gate_tt,
        beta_tt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    grouped_a_tt, grouped_b_tt = ttnn.transformer.chunk_kda_group_summary(
        grouped_prep,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    entries_tt = ttnn.transformer.kda_affine_prefix(
        grouped_a_tt,
        grouped_b_tt,
        ttnn.reshape(initial_state_tt, (heads, dim, dim)),
        groups,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    grouped_output_tt, grouped_final_state_tt = ttnn.transformer.chunk_kda_group_scan(
        grouped_prep,
        entries_tt,
        groups,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    _assert_pcc("KDA grouped summary A", ttnn.to_torch(transform_a_tt), ttnn.to_torch(grouped_a_tt), threshold=0.999)
    _assert_pcc("KDA grouped summary B", ttnn.to_torch(transform_b_tt), ttnn.to_torch(grouped_b_tt), threshold=0.999)
    transform_a = ttnn.to_torch(grouped_a_tt).reshape(heads, groups, dim, dim)
    transform_b = ttnn.to_torch(grouped_b_tt).reshape(heads, groups, dim, dim)
    entries = ttnn.to_torch(entries_tt).reshape(heads, groups, dim, dim)
    expected_final_state = torch.matmul(transform_a[:, -1].float(), entries[:, -1].float()) + transform_b[:, -1].float()
    expected_final_state = expected_final_state.reshape_as(initial_state)
    _assert_pcc(
        "KDA affine span summary", expected_final_state, ttnn.to_torch(final_state_tt), threshold=affine_threshold
    )
    _assert_pcc(
        "KDA grouped scan output",
        ttnn.to_torch(output_tt),
        ttnn.to_torch(grouped_output_tt),
        threshold=affine_threshold,
    )
    _assert_pcc(
        "KDA grouped scan final state",
        ttnn.to_torch(final_state_tt),
        ttnn.to_torch(grouped_final_state_tt).reshape_as(initial_state),
        threshold=affine_threshold,
    )
