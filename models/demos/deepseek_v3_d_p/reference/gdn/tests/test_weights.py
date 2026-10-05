# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the canonical GDN host-weight schema."""

import torch

from models.demos.deepseek_v3_d_p.reference.gdn.tests.helpers import TINY, random_weights
from models.demos.deepseek_v3_d_p.reference.gdn.weights import validate_gdn_weights


def test_accepts_canonical_weights() -> None:
    validate_gdn_weights(random_weights(TINY), TINY)


def test_reports_missing_weight(expect_error) -> None:
    weights = random_weights(TINY)
    del weights["dt_bias"]
    with expect_error(ValueError, r"missing GDN weights: \['dt_bias'\]"):
        validate_gdn_weights(weights, TINY)


def test_reports_unexpected_weight(expect_error) -> None:
    """A Qwen3-Next-style fused layout is not silently ignored next to the canonical keys."""
    weights = random_weights(TINY)
    weights["in_proj_ba.weight"] = torch.empty(2 * TINY.num_value_heads, TINY.hidden_size)
    with expect_error(ValueError, r"unexpected GDN weights: \['in_proj_ba.weight'\]"):
        validate_gdn_weights(weights, TINY)


def test_reports_wrong_shape(expect_error) -> None:
    """q/k carry K heads: an in_proj_qkv sized as if q/k had V heads is rejected."""
    weights = random_weights(TINY)
    weights["in_proj_qkv.weight"] = torch.empty(3 * TINY.v_dim, TINY.hidden_size)
    with expect_error(ValueError, r"in_proj_qkv\.weight shape \(288, 64\) != \(160, 64\)"):
        validate_gdn_weights(weights, TINY)
