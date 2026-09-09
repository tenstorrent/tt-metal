# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run existing gather suites with exact BF16 checks in place of tolerance-only acceptance."""
import sys

import pytest
import torch
from tests.ttnn import utils_for_testing


def require_exact_bf16(original):
    def checked(expected, actual, *args, **kwargs):
        if isinstance(expected, torch.Tensor) and expected.dtype == torch.bfloat16:
            assert torch.equal(expected, actual), "Existing gather suite: valid BF16 results are not exactly equal"
        return original(expected, actual, *args, **kwargs)

    return checked


for name in ("assert_allclose", "assert_with_pcc"):
    setattr(utils_for_testing, name, require_exact_bf16(getattr(utils_for_testing, name)))
print("Exact BF16 checks enabled for existing gather allclose/PCC comparisons", flush=True)
raise SystemExit(pytest.main(sys.argv[1:]))
