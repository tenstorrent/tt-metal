# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Blackhole PCC tests for the chunk-parallel KDA operation."""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.experimental.kimi_delta_attention.reference.ops import kda_recurrent_reference, l2_norm_reference
from models.experimental.kimi_delta_attention.tests.utils import assert_all_finite, compare_cpu_device

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
    assert_all_finite(f"{name} CPU reference", golden)
    assert_all_finite(f"{name} device result", actual)
    passed, pcc = comp_pcc(golden, actual, pcc=threshold)
    max_abs = (golden.float() - actual.float()).abs().max().item()
    print(f"{name}: PCC={pcc:.6f}, max_abs={max_abs:.6e}")
    assert passed, f"{name} PCC {pcc:.6f} < {threshold}"


@pytest.mark.parametrize(
    "sequence,heads,key_dim,value_dim,flat_v,flat_qk,flat_g,math_fidelity,summary_group_chunks,production_tuning,output_head_major",
    [
        (32, 2, 32, 32, False, False, False, None, None, False, False),
        (32, 2, 32, 32, True, True, True, None, None, False, True),
        (33, 2, 32, 32, False, False, False, None, None, False, True),
        (64, 32, 128, 128, False, False, False, None, None, False, False),
        (64, 32, 128, 128, True, False, True, None, None, False, False),
        (64, 32, 128, 128, True, True, True, "HiFi2", None, False, True),
        (256, 4, 128, 128, True, True, True, None, None, False, True),
        (512, 4, 128, 128, True, True, True, None, None, False, True),
        (2816, 12, 128, 128, True, True, True, None, 21, True, True),
        (5120, 2, 32, 32, True, True, True, None, 8, True, True),
        (5120, 1, 128, 128, True, True, True, None, 8, True, True),
        (5152, 2, 32, 32, True, True, True, None, 8, False, True),
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
    summary_group_chunks: int | None,
    production_tuning: bool,
    output_head_major: bool,
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
    production_compute_config = (
        ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True
        )
        if production_tuning
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
            output_head_major=output_head_major,
            chunk_size=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=compute_kernel_config,
            summary_group_chunks=summary_group_chunks or 8,
            affine_summary_dtype=ttnn.bfloat16 if production_tuning else ttnn.float32,
            affine_prefix_compute_kernel_config=production_compute_config,
            grouped_scan_output_dtype=ttnn.bfloat16 if production_tuning else ttnn.float32,
            grouped_scan_compute_kernel_config=production_compute_config,
            use_bf16_prep_intermediates=production_tuning,
        )

    assert final_state_tt.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    actual_output = ttnn.to_torch(output_tt)
    if output_head_major:
        actual_output = actual_output.reshape(1, heads, sequence, value_dim).permute(0, 2, 1, 3)
    actual_state = ttnn.to_torch(final_state_tt)
    label = f"H={heads},K={key_dim},V={value_dim},T={sequence},flat_v={flat_v},flat_qk={flat_qk},flat_g={flat_g},math_fidelity={math_fidelity},summary_group_chunks={summary_group_chunks},production_tuning={production_tuning}"
    _, output_failures = compare_cpu_device(
        f"{label} output",
        golden_output,
        actual_output,
        pcc_threshold=0.999,
    )
    _, state_failures = compare_cpu_device(
        f"{label} state",
        golden_state,
        actual_state,
        pcc_threshold=0.999,
    )
    failures = output_failures + state_failures
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("production_tuning", [False, True])
def test_grouped_summary_preserves_weak_decay(device: ttnn.Device, production_tuning: bool) -> None:
    sequence, heads, dim = 5120, 1, 32
    generator = torch.Generator().manual_seed(9401)
    shape = (1, sequence, heads, dim)
    q = torch.randn(*shape, generator=generator)
    k = torch.randn(*shape, generator=generator)
    v = torch.randn(*shape, generator=generator)
    gate = torch.full(shape, -1e-5)
    beta = torch.sigmoid(torch.randn(1, sequence, heads, generator=generator))
    state = 0.02 * torch.randn(1, heads, dim, dim, generator=generator)
    golden_output, golden_state = kda_recurrent_reference(q, k, v, gate, beta, state)
    production_compute_config = (
        ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True
        )
        if production_tuning
        else None
    )
    production_kwargs = {
        "affine_summary_dtype": ttnn.bfloat16 if production_tuning else ttnn.float32,
        "affine_prefix_compute_kernel_config": production_compute_config,
        "grouped_scan_output_dtype": ttnn.bfloat16 if production_tuning else ttnn.float32,
        "grouped_scan_compute_kernel_config": production_compute_config,
        "use_bf16_prep_intermediates": production_tuning,
    }

    with ttnn.manage_config("throw_exception_on_fallback", True):
        output_tt, final_state_tt = ttnn.transformer.chunk_kda(
            _to_device(l2_norm_reference(q), device, ttnn.bfloat16),
            _to_device(l2_norm_reference(k), device, ttnn.bfloat16),
            _to_device(v, device, ttnn.bfloat16),
            _to_device(gate, device, ttnn.float32),
            _to_device(beta, device, ttnn.float32),
            initial_state=_to_device(state, device, ttnn.float32),
            output_final_state=True,
            chunk_size=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            summary_group_chunks=8,
            **production_kwargs,
        )

    ttnn.synchronize_device(device)

    with ttnn.manage_config("throw_exception_on_fallback", True):
        output_two_tt, final_state_two_tt = ttnn.transformer.chunk_kda(
            _to_device(l2_norm_reference(q), device, ttnn.bfloat16),
            _to_device(l2_norm_reference(k), device, ttnn.bfloat16),
            _to_device(v, device, ttnn.bfloat16),
            _to_device(gate, device, ttnn.float32),
            _to_device(beta, device, ttnn.float32),
            initial_state=_to_device(state, device, ttnn.float32),
            output_final_state=True,
            chunk_size=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            summary_group_chunks=2,
            **production_kwargs,
        )

    label = f"weak-decay production_tuning={production_tuning}"
    _assert_pcc(f"{label} grouped output", golden_output, ttnn.to_torch(output_tt))
    _assert_pcc(
        f"{label} group-size output invariance",
        ttnn.to_torch(output_two_tt),
        ttnn.to_torch(output_tt),
        threshold=0.9999,
    )
    _assert_pcc(
        f"{label} group-size state invariance",
        ttnn.to_torch(final_state_two_tt),
        ttnn.to_torch(final_state_tt),
        threshold=0.9999,
    )
    _assert_pcc(f"{label} grouped state", golden_state, ttnn.to_torch(final_state_tt))
