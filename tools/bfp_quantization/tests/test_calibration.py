# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from tt_bfp_quant import HessianAccumulator, capture_linear_inputs


def test_masked_chunked_moments_and_sample_cap():
    torch.manual_seed(18)
    x = torch.randn(3, 7, 5)
    mask = torch.rand(3, 7) > 0.2
    stats = HessianAccumulator(5, chunk_rows=3, max_samples=11)
    stats.add(x, mask)
    selected = x[mask][:11]
    assert stats.count == len(selected)
    torch.testing.assert_close(stats.value(), selected.T @ selected / len(selected))
    previous = stats.value().clone()
    stats.add(x, mask)
    assert stats.count == 11
    assert torch.equal(previous, stats.value())


def test_hooks_observe_exact_inputs_without_changing_outputs():
    torch.manual_seed(1)
    model = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.ReLU(), torch.nn.Linear(7, 3)).eval()
    x = torch.randn(4, 8, 5)
    with torch.inference_mode():
        before = model(x)
        with capture_linear_inputs(model, ["0", "2"], chunk_rows=9) as stats:
            actual = model(x)
    assert torch.equal(actual, before)
    xx = x.reshape(-1, 5)
    torch.testing.assert_close(stats["0"].value(), xx.T @ xx / len(xx))
    assert stats["0"].count == stats["2"].count == 32
    assert not model[0]._forward_pre_hooks and not model[2]._forward_pre_hooks


def test_hook_cleanup_on_exception_and_keyword_input():
    model = torch.nn.Sequential(torch.nn.Linear(5, 7))
    with pytest.raises(RuntimeError):  # allow-pytest.raises: CPU-only tests.
        with capture_linear_inputs(model, ["0"]) as stats:
            model[0](input=torch.randn(4, 5))
            assert stats["0"].count == 4
            raise RuntimeError("test")
    assert not model[0]._forward_pre_hooks


def test_empty_and_incompatible_capture_rejected():
    with pytest.raises(ValueError):  # allow-pytest.raises: CPU-only tests.
        HessianAccumulator(5).value()
    with pytest.raises(ValueError):  # allow-pytest.raises: CPU-only tests.
        with capture_linear_inputs(torch.nn.Sequential(torch.nn.ReLU()), ["0"]):
            pass
