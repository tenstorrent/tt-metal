# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Leaf helpers for gating tests on tensor-prefetcher support.

Kept free of model/tracy/nightly-test imports so a test that only needs the skip
gate does not pull in prefetcher_common's dependency chain at collection time.
"""

import pytest
import ttnn


def require_tensor_prefetcher(device):
    """Skip unless programmable DRAM cores are available on this device."""
    if not ttnn.experimental.is_tensor_prefetcher_supported(device):
        pytest.skip("programmable DRAM cores unavailable (need Blackhole and firmware >= 19.12.0.0)")
