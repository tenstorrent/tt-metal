# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""Deprecated entry point.

The full-checkpoint Devstral-2-123B PCC test was removed when the rewrite switched to per-layer
random-weight PCC (see ``test_ministral3_*.py``). Real Hub-checkpoint validation should run via the
demo (``models/experimental/devstral2_large/reference/devstral2_123b_inference.py``) instead.
"""

import pytest


def test_replaced_by_random_weight_pcc() -> None:
    pytest.skip("Per-layer PCC against HF reference (random weights) is now the canonical test suite.")
