# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the KDA reference weight contract."""

from dataclasses import replace

import torch

from models.demos.deepseek_v3_d_p.reference.kda.tests.helpers import make_config, random_weights
from models.demos.deepseek_v3_d_p.reference.kda.weights import normalize_kda_state_dict, validate_kda_weights
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_equal


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


def test_normalize_state_dict_trims_kimi_k3_padded_a_log() -> None:
    config = replace(make_config(use_full_rank_gate=True), num_heads=96)
    state_dict = random_weights(config)
    padded = torch.arange(128, dtype=torch.float32)
    state_dict["A_log"] = padded

    normalized = normalize_kda_state_dict(state_dict, config)

    assert normalized["A_log"].shape == (1, 1, config.num_heads, 1)
    assert_equal(padded[: config.num_heads], normalized["A_log"].reshape(-1), name="trimmed A_log")
    assert state_dict["A_log"] is padded


def test_normalize_state_dict_rejects_unsupported_a_log_padding(expect_error) -> None:
    config = replace(make_config(use_full_rank_gate=True), num_heads=96)
    state_dict = random_weights(config)
    state_dict["A_log"] = torch.arange(127, dtype=torch.float32)

    with expect_error(ValueError, "A_log has 127 entries"):
        normalize_kda_state_dict(state_dict, config)


def test_normalize_state_dict_requires_a_log(expect_error) -> None:
    config = make_config()
    state_dict = random_weights(config)
    del state_dict["A_log"]

    with expect_error(ValueError, "missing KDA weight: A_log"):
        normalize_kda_state_dict(state_dict, config)
