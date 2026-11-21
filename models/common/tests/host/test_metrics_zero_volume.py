# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Zero-volume tensors in models.common.metrics.comp_allclose.

The PyTorch fallback reduced the delta with torch.max, which raises on a tensor with no
elements. The exception was caught by the helper's own handler, so two identical empty
results were reported as ``False`` with the text of the RuntimeError instead of passing.
An empty result against a differently shaped empty one also reported an error rather than a
plain mismatch.
"""

import torch

from models.common.metrics import comp_allclose


def test_equal_empty_tensors_pass():
    empty = torch.empty(0, 32)
    passing, message = comp_allclose(empty, empty.clone())
    assert passing, message
    assert "Error computing" not in message


def test_empty_tensors_of_different_shapes_fail_without_error_text():
    passing, message = comp_allclose(torch.empty(0, 32), torch.empty(0, 16))
    assert not passing
    assert "Error computing" not in message
    assert "Allclose check failed" in message


def test_empty_against_non_empty_fails_without_error_text():
    passing, message = comp_allclose(torch.empty(0, 32), torch.zeros(1, 32))
    assert not passing
    assert "Error computing" not in message


def test_non_empty_tensors_keep_their_result():
    passing, message = comp_allclose(torch.ones(2, 2), torch.ones(2, 2))
    assert passing, message
    assert "Max ATOL Delta" in message

    passing, message = comp_allclose(torch.ones(2, 2), torch.full((2, 2), 1.5))
    assert not passing
    assert "Allclose check failed" in message
