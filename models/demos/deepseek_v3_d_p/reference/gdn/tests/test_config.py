# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the GDN configuration invariants."""

from dataclasses import asdict

import pytest

from models.demos.deepseek_v3_d_p.reference.gdn.config import GDNConfig
from models.demos.deepseek_v3_d_p.reference.gdn.tests.helpers import TINY


def test_derived_dimensions() -> None:
    # 2 K heads x 16 + 2 K heads x 16 + 6 V heads x 16 = 32 + 32 + 96.
    assert (TINY.group, TINY.q_dim, TINY.k_dim, TINY.v_dim, TINY.conv_dim) == (3, 32, 32, 96, 160)


@pytest.mark.parametrize(
    "field", ["hidden_size", "num_key_heads", "num_value_heads", "head_k_dim", "head_v_dim", "conv_kernel_size"]
)
def test_rejects_nonpositive_dimensions(field: str, expect_error) -> None:
    with expect_error(ValueError, field):
        GDNConfig(**(asdict(TINY) | {field: 0}))


def test_rejects_value_heads_not_a_multiple_of_key_heads(expect_error) -> None:
    with expect_error(ValueError, "multiple of num_key_heads"):
        GDNConfig(**(asdict(TINY) | {"num_value_heads": 5}))


def test_rejects_unsupported_numerical_policy(expect_error) -> None:
    with expect_error(ValueError, "conv_kernel_size=4"):
        GDNConfig(**(asdict(TINY) | {"conv_kernel_size": 3}))
    for norm_eps in (0.0, float("nan"), float("inf")):
        with expect_error(ValueError, "norm_eps"):
            GDNConfig(**(asdict(TINY) | {"norm_eps": norm_eps}))


@pytest.mark.parametrize("activation", ["swish", "gelu", None])
def test_rejects_noncanonical_output_gate_activation(activation, expect_error) -> None:
    """Aliases (swish) are resolved by the model-config boundary; the config takes only canonical names."""
    with expect_error(ValueError, "output_gate_activation"):
        GDNConfig(**(asdict(TINY) | {"output_gate_activation": activation}))


def test_reference_package_does_not_import_ttnn() -> None:
    import models.demos.deepseek_v3_d_p.reference.gdn.config as config_module
    import models.demos.deepseek_v3_d_p.reference.gdn.layer as layer_module

    assert "ttnn" not in config_module.__dict__
    assert "ttnn" not in layer_module.__dict__
