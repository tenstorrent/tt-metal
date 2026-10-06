# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the layer-0 input math of the GDN real-text inputs, on hand-verifiable values."""

from __future__ import annotations

import torch

from models.demos.deepseek_v3_d_p.tests.gdn.cases import GDNPreparedCacheMiss, build_gdn_case, registered_gdn_case
from models.demos.deepseek_v3_d_p.tests.gdn.text_input import (
    gdn_text_input_cache_path,
    hyper_connection_input_mix,
    qwen_input_norm,
)


def test_input_norm_scales_by_one_plus_weight() -> None:
    # x = [3, 4]: RMS = sqrt(12.5), x / RMS = [0.84853, 1.13137]; weight [0, 1] -> scale [1, 2].
    out = qwen_input_norm(torch.tensor([[3.0, 4.0]]), torch.tensor([0.0, 1.0]), eps=0.0)
    torch.testing.assert_close(out, torch.tensor([[0.848528, 2.262742]]))


def test_input_norm_groups_normalize_independently() -> None:
    out = qwen_input_norm(torch.tensor([[3.0, 4.0, 30.0, 40.0]]), torch.zeros(4), eps=0.0, group=2)
    torch.testing.assert_close(out, torch.tensor([[0.848528, 1.131371, 0.848528, 1.131371]]))


def test_hyper_connection_mix_of_identical_streams() -> None:
    # Zero mix weights: silu(0) = 0, sigmoid(0) = 1/2, so the mix is half the mean of the normalized identical streams.
    out = hyper_connection_input_mix(
        torch.tensor([[3.0, 4.0]]), torch.zeros(4), torch.zeros(3, 4), torch.zeros(4, 3), streams=2, eps=0.0
    )
    torch.testing.assert_close(out, torch.tensor([[0.424264, 0.565685]]))
    # A stream-1 norm scale of 3 (1 + w = 4) makes the stream mean (1 + 4) / 2 of the normalized embedding.
    norm_weight = torch.tensor([0.0, 0.0, 3.0, 3.0])
    out = hyper_connection_input_mix(
        torch.tensor([[3.0, 4.0]]), norm_weight, torch.zeros(3, 4), torch.zeros(4, 3), streams=2, eps=0.0
    )
    torch.testing.assert_close(out, torch.tensor([[0.848528 * 1.25, 1.131371 * 1.25]]))


def test_text_case_fails_fast_on_a_missing_input(monkeypatch, tmp_path, expect_error) -> None:
    monkeypatch.setenv("TT_LINEAR_LAYERS_SHARED_CACHE", str(tmp_path))
    monkeypatch.setenv("GDN_CACHE_MISS", "fail")
    spec = registered_gdn_case("qwen38_27b", "LB-A", "single", "real", "text")
    assert gdn_text_input_cache_path("qwen38_27b", 0, 1280).is_relative_to(tmp_path)
    with expect_error(GDNPreparedCacheMiss, "text input"):
        build_gdn_case(spec)
