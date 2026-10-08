# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary (#58723): fused_recurrent_gated_delta_rule.cpp through the qwen36 GDN tests, on the plain device
fixture (the module's mesh_device fixture leaves no device profiler report)."""
import importlib

import pytest
import torch

m = importlib.import_module("models.demos.blackhole.qwen36.tests.test_fused_recurrent_gdn")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_frgdn_decode(device, seed):
    torch.manual_seed(seed)
    m.test_fused_decode_matches_fla_naive(device, seed)


@pytest.mark.parametrize("K", [3, 4])
def test_frgdn_verify(device, K):
    torch.manual_seed(K)
    m.test_fused_verify_matches_fla_naive(device, K, 0)
