# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the KDA reference weight contract."""

import torch

from models.demos.deepseek_v3_d_p.reference.kda.tests.helpers import make_config, random_weights
from models.demos.deepseek_v3_d_p.reference.kda.weights import validate_kda_weights


def test_weight_validation_reports_exact_name_and_shape(expect_error) -> None:
    config = make_config(use_full_rank_gate=True)
    weights = random_weights(config)
    weights["q_proj.weight"] = torch.empty(config.q_dim, config.hidden_size + 1)

    with expect_error(ValueError, r"q_proj\.weight shape .* !="):
        validate_kda_weights(weights, config)


def test_weight_validation_reports_exact_missing_name(expect_error) -> None:
    config = make_config(use_full_rank_gate=True)
    weights = random_weights(config)
    del weights["q_proj.weight"]

    with expect_error(ValueError, "missing KDA weight: q_proj.weight"):
        validate_kda_weights(weights, config)
