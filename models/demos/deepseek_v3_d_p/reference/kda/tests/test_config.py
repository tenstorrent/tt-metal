# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for KDA model configuration."""

import pytest

import ttnn
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig, KDAProgramConfig, KDARecurrenceProgramConfig
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import (
    KimiK3Config,
    kimi_k3_kda_config,
    kimi_k3_program_config,
)


def test_target_config_mapping() -> None:
    config = KDAConfig.from_model_config(
        {
            "hidden_size": 2304,
            "rms_norm_eps": 1e-5,
            "linear_attn_config": {
                "head_dim": 128,
                "num_heads": 32,
                "short_conv_kernel_size": 4,
            },
        }
    )

    assert config.hidden_size == 2304
    assert config.num_heads == 32
    assert config.head_k_dim == config.head_v_dim == 128
    assert config.q_dim == config.k_dim == config.v_dim == 4096
    assert config.conv_kernel_size == 4


def test_kimi_k3_config_mapping() -> None:
    config = kimi_k3_kda_config()

    assert config.hidden_size == KimiK3Config.HIDDEN_SIZE == 7168
    assert config.num_heads == KimiK3Config.KDA_NUM_HEADS == 96
    assert config.head_k_dim == config.head_v_dim == 128
    assert config.conv_kernel_size == 4
    assert config.use_full_rank_gate
    assert config.gate_lower_bound == -5.0


def test_program_config_ccl_topology() -> None:
    assert KDAProgramConfig().tp_ccl_topology == ttnn.Topology.Linear
    configured = kimi_k3_program_config(tp_ccl_topology=ttnn.Topology.Ring)
    assert configured.tp_ccl_topology == ttnn.Topology.Ring


def test_program_config_affine_summary_dtype() -> None:
    assert KDAProgramConfig().recurrence.affine_summary_dtype == ttnn.bfloat16
    assert kimi_k3_program_config(tp_ccl_topology=ttnn.Topology.Linear).recurrence.affine_summary_dtype == ttnn.bfloat16


def test_program_config_recurrent_state_dtype() -> None:
    assert KDAProgramConfig().recurrence.recurrent_state_dtype == ttnn.float32
    assert kimi_k3_program_config(tp_ccl_topology=ttnn.Topology.Linear).recurrence.recurrent_state_dtype == ttnn.float32


def test_program_config_affine_prefix_math_fidelity() -> None:
    assert KDAProgramConfig().recurrence.affine_prefix_math_fidelity == ttnn.MathFidelity.HiFi2
    assert (
        kimi_k3_program_config(tp_ccl_topology=ttnn.Topology.Linear).recurrence.affine_prefix_math_fidelity
        == ttnn.MathFidelity.HiFi2
    )


def test_program_config_grouped_scan_output_dtype() -> None:
    assert KDAProgramConfig().recurrence.scan_output_dtype == ttnn.bfloat16
    assert kimi_k3_program_config(tp_ccl_topology=ttnn.Topology.Linear).recurrence.scan_output_dtype == ttnn.bfloat16


def test_program_config_grouped_scan_math_fidelity() -> None:
    assert KDAProgramConfig().recurrence.grouped_scan_math_fidelity == ttnn.MathFidelity.HiFi2
    assert (
        kimi_k3_program_config(tp_ccl_topology=ttnn.Topology.Linear).recurrence.grouped_scan_math_fidelity
        == ttnn.MathFidelity.HiFi2
    )


def test_program_config_gated_rms_output_dtype() -> None:
    assert KDAProgramConfig().gated_rms_output_dtype == ttnn.float32
    assert kimi_k3_program_config(tp_ccl_topology=ttnn.Topology.Linear).gated_rms_output_dtype == ttnn.bfloat16


def test_program_config_output_projection_math_fidelity() -> None:
    assert KDAProgramConfig().output_projection_math_fidelity == ttnn.MathFidelity.HiFi4
    assert (
        kimi_k3_program_config(tp_ccl_topology=ttnn.Topology.Linear).output_projection_math_fidelity
        == ttnn.MathFidelity.HiFi2
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("qkv_dtype", ttnn.float32),
        ("gate_dtype", ttnn.float32),
        ("beta_dtype", ttnn.bfloat16),
        ("recurrent_state_dtype", ttnn.bfloat16),
        ("affine_summary_dtype", ttnn.float32),
        ("scan_output_dtype", ttnn.float32),
    ],
)
def test_recurrence_config_rejects_noncanonical_dtype(field: str, value: ttnn.DataType, expect_error) -> None:
    with expect_error(ValueError, field):
        KDARecurrenceProgramConfig(**{field: value})


def test_recurrence_config_uses_measured_memory_placements() -> None:
    config = KDARecurrenceProgramConfig()

    assert config.preparation_memory_config == ttnn.DRAM_MEMORY_CONFIG
    assert config.prefix_memory_config == ttnn.DRAM_MEMORY_CONFIG
    assert config.distributed_working_memory_config == ttnn.L1_MEMORY_CONFIG
    assert config.output_memory_config == ttnn.DRAM_MEMORY_CONFIG


@pytest.mark.parametrize(
    "field,value",
    [
        ("preparation_memory_config", ttnn.L1_MEMORY_CONFIG),
        ("prefix_memory_config", ttnn.L1_MEMORY_CONFIG),
        ("distributed_working_memory_config", ttnn.DRAM_MEMORY_CONFIG),
        ("output_memory_config", ttnn.L1_MEMORY_CONFIG),
    ],
)
def test_recurrence_config_rejects_unmeasured_memory_placement(
    field: str, value: ttnn.MemoryConfig, expect_error
) -> None:
    with expect_error(ValueError, field):
        KDARecurrenceProgramConfig(**{field: value})


@pytest.mark.parametrize("field", ["hidden_size", "num_heads", "head_k_dim", "head_v_dim"])
def test_config_rejects_nonpositive_dimensions(field: str, expect_error) -> None:
    values = {
        "hidden_size": 64,
        "num_heads": 2,
        "head_k_dim": 32,
        "head_v_dim": 32,
        "conv_kernel_size": 4,
        "norm_eps": 1e-5,
    }
    values[field] = 0
    with expect_error(ValueError, field):
        KDAConfig(**values)


def test_config_rejects_non_four_tap_convolution(expect_error) -> None:
    with expect_error(ValueError, "conv_kernel_size=4"):
        KDAConfig(
            hidden_size=64,
            num_heads=2,
            head_k_dim=32,
            head_v_dim=32,
            conv_kernel_size=3,
            norm_eps=1e-5,
        )
