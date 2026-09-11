# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for KDA semantic configuration."""

import pytest

from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kda.tests.helpers import make_config


def test_model_config_mapping() -> None:
    config = KDAConfig.from_model_config(
        {
            "hidden_size": 2304,
            "rms_norm_eps": 1e-5,
            "linear_attn_config": {
                "head_dim": 128,
                "num_heads": 32,
                "short_conv_kernel_size": 4,
                "use_full_rank_gate": True,
                "gate_lower_bound": -5.0,
            },
        }
    )
    assert (config.q_dim, config.k_dim, config.v_dim) == (4096, 4096, 4096)
    assert config.use_full_rank_gate
    assert config.gate_lower_bound == -5.0


def test_nested_text_config_mapping() -> None:
    base = make_config()
    mapped = KDAConfig.from_model_config(
        {
            "text_config": {
                "hidden_size": base.hidden_size,
                "rms_norm_eps": base.norm_eps,
                "linear_attn_config": {
                    "head_dim": base.head_k_dim,
                    "num_heads": base.num_heads,
                    "short_conv_kernel_size": base.conv_kernel_size,
                },
            }
        }
    )
    assert mapped == base


@pytest.mark.parametrize("field", ["hidden_size", "num_heads", "head_k_dim", "head_v_dim"])
def test_config_rejects_nonpositive_dimensions(field: str, expect_error) -> None:
    values = make_config().__dict__.copy()
    values[field] = 0
    with expect_error(ValueError, field):
        KDAConfig(**values)


def test_config_rejects_invalid_numerical_policy(expect_error) -> None:
    values = make_config().__dict__.copy()
    with expect_error(ValueError, "conv_kernel_size=4"):
        KDAConfig(**(values | {"conv_kernel_size": 3}))
    with expect_error(ValueError, "norm_eps"):
        KDAConfig(**(values | {"norm_eps": 0.0}))
    for norm_eps in (float("nan"), float("inf")):
        with expect_error(ValueError, "norm_eps"):
            KDAConfig(**(values | {"norm_eps": norm_eps}))
    with expect_error(ValueError, "gate_lower_bound"):
        KDAConfig(**(values | {"gate_lower_bound": 0.0}))


def test_config_module_does_not_import_ttnn() -> None:
    import models.demos.deepseek_v3_d_p.reference.kda.config as config_module

    assert "ttnn" not in config_module.__dict__
