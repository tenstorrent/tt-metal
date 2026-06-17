# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

from models.common.utility_functions import torch_random
from tests.ttnn.utils_for_testing import assert_numeric_metrics

pytestmark = pytest.mark.use_module_device


def _ok_candidate_entries(result):
    return [entry for entry in result["candidate_timings_us"] if entry["status"] == "ok"]


def test_linear_auto_config_accepts_host_rhs_and_bias_when_disabled(device):
    torch.manual_seed(0)

    input_shape_a = (1, 1, 32, 64)
    input_shape_b = (64, 64)
    torch_input_tensor_a = torch_random(input_shape_a, -0.1, 0.1, dtype=torch.float32)
    torch_input_tensor_b = torch_random(input_shape_b, -0.1, 0.1, dtype=torch.float32)
    torch_bias = torch_random((64,), -0.1, 0.1, dtype=torch.float32)
    torch_output_tensor = torch.nn.functional.linear(
        torch_input_tensor_a,
        torch_input_tensor_b.T.contiguous(),
        bias=torch_bias,
    )

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    output_tensor = ttnn.linear(
        input_tensor_a,
        torch_input_tensor_b,
        bias=torch_bias,
        auto_config=False,
        dtype=ttnn.bfloat16,
    )
    output_tensor = ttnn.to_torch(output_tensor)

    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        atol=0.001 * input_shape_b[0],
        rtol=0.016 * input_shape_b[0],
        frobenius_threshold=0.001 * input_shape_b[0],
        pcc_threshold=0.999,
        check_ulp=False,
    )


def test_explain_matmul_force_retune_picks_fastest_candidate_and_reuses_cache(monkeypatch, tmp_path, device):
    monkeypatch.setenv("TTNN_AUTO_MATMUL_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TTNN_AUTO_MATMUL_VERSION", "unit-test-version")
    monkeypatch.setenv("TTNN_AUTO_MATMUL_FORCE_RETUNE", "1")

    torch.manual_seed(0)
    torch_input_tensor_a = torch_random((1, 1, 32, 64), -0.1, 0.1, dtype=torch.float32)
    torch_input_tensor_b = torch_random((64, 64), -0.1, 0.1, dtype=torch.float32)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    tuned = ttnn.experimental.auto_config.explain_matmul(
        input_tensor_a,
        input_tensor_b,
        dtype=ttnn.bfloat16,
        allow_tuning=True,
    )

    ok_candidates = _ok_candidate_entries(tuned)
    assert ok_candidates, "Expected at least one successfully measured candidate"

    winner_entry = next(entry for entry in ok_candidates if entry["descriptor"] == tuned["winner"])
    winner_average_us = winner_entry["average_us"]
    assert winner_average_us == min(entry["average_us"] for entry in ok_candidates)

    default_entries = [entry for entry in ok_candidates if entry["descriptor"]["kind"] == "default_matmul"]
    if default_entries:
        assert default_entries[0]["average_us"] >= winner_average_us

    monkeypatch.delenv("TTNN_AUTO_MATMUL_FORCE_RETUNE", raising=False)
    cached = ttnn.experimental.auto_config.explain_matmul(
        input_tensor_a,
        input_tensor_b,
        dtype=ttnn.bfloat16,
        allow_tuning=False,
    )

    assert cached["cache_hit"] is True
    assert cached["winner"] == tuned["winner"]
