# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Direct device coverage for the fused KDA gated RMS normalization."""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda.ops import sigmoid_gated_rms_norm_reference
from models.demos.deepseek_v3_d_p.tests.kda.utils import assert_accurate, assert_bit_identical

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True),
]


@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("output_dtype", [ttnn.float32, ttnn.bfloat16])
def test_kda_gated_rms_norm_matches_reference_cache_and_trace(
    device: ttnn.Device, input_dtype: ttnn.DataType, output_dtype: ttnn.DataType
) -> None:
    """Cover Kimi-K3 TP8-local geometry, program reuse, and trace replay."""
    batch = 1
    sequence = 64
    num_heads = 12
    value_dim = 128
    epsilon = 1e-5
    generator = torch.Generator().manual_seed(319)

    inputs = torch.randn(
        batch * num_heads,
        sequence,
        value_dim,
        generator=generator,
        dtype=torch.float32,
    )
    if input_dtype == ttnn.bfloat16:
        inputs = inputs.to(torch.bfloat16)
    gate = torch.randn(
        batch,
        sequence,
        num_heads * value_dim,
        generator=generator,
        dtype=torch.bfloat16,
    )
    weight = torch.randn(value_dim, generator=generator, dtype=torch.bfloat16)
    expected = sigmoid_gated_rms_norm_reference(
        inputs.reshape(batch, num_heads, sequence, value_dim).permute(0, 2, 1, 3),
        gate.reshape(batch, sequence, num_heads, value_dim),
        weight,
        eps=epsilon,
    ).reshape(batch, sequence, num_heads * value_dim)

    input_tt = ttnn.from_torch(
        inputs,
        dtype=input_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gate_tt = ttnn.from_torch(
        gate,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    weight_tt = ttnn.from_torch(
        weight,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def run() -> ttnn.Tensor:
        return ttnn.transformer.kda_gated_rms_norm(
            input_tt,
            gate_tt,
            weight_tt,
            num_heads,
            epsilon=epsilon,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=compute_config,
            output_dtype=output_dtype,
        )

    with ttnn.manage_config("throw_exception_on_fallback", True):
        output_tt = run()
    assert output_tt.dtype == output_dtype
    cache_entries = device.num_program_cache_entries()
    with ttnn.manage_config("throw_exception_on_fallback", True):
        repeated_tt = run()
    ttnn.synchronize_device(device)
    assert device.num_program_cache_entries() == cache_entries

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    with ttnn.manage_config("throw_exception_on_fallback", True):
        traced_tt = run()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    for _ in range(2):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)

    for name, actual_tt in (
        ("eager", output_tt),
        ("repeated", repeated_tt),
        ("traced", traced_tt),
    ):
        actual = ttnn.to_torch(actual_tt)
        assert_accurate(expected, actual, name=name, pcc_threshold=0.999)

    ttnn.release_trace(device, trace_id)


def test_kda_gated_rms_norm_determinism(device: ttnn.Device) -> None:
    batch, sequence, num_heads, value_dim = 1, 64, 12, 128
    generator = torch.Generator().manual_seed(1319)
    inputs = torch.randn(batch * num_heads, sequence, value_dim, generator=generator)
    gate = torch.randn(batch, sequence, num_heads * value_dim, generator=generator, dtype=torch.bfloat16)
    weight = torch.randn(value_dim, generator=generator, dtype=torch.bfloat16)

    def to_device(tensor: torch.Tensor, dtype: ttnn.DataType) -> ttnn.Tensor:
        return ttnn.from_torch(
            tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    input_tt = to_device(inputs, ttnn.float32)
    gate_tt = to_device(gate, ttnn.bfloat16)
    weight_tt = to_device(weight, ttnn.bfloat16)
    results = []
    for _ in range(3):
        output_tt = ttnn.transformer.kda_gated_rms_norm(
            input_tt,
            gate_tt,
            weight_tt,
            num_heads,
            epsilon=1e-5,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.synchronize_device(device)
        results.append(ttnn.to_torch(output_tt))

    for iteration, output in enumerate(results[1:], start=1):
        assert_bit_identical(results[0], output, name=f"gated RMSNorm iteration {iteration}")
