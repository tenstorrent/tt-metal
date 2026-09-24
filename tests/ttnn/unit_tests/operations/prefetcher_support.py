# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Leaf helpers for gating tests on tensor-prefetcher support.

Kept free of model/tracy/nightly-test imports so a test that only needs the skip
gate does not pull in prefetcher_common's dependency chain at collection time.
"""

import pytest
import ttnn


def require_tensor_prefetcher(device):
    """Skip unless the tensor prefetcher is usable on this device.

    Two things make it unusable, and the skip names both because the fix differs: no programmable
    DRAM cores, or the streaming profiler holding the same DRISCs.
    """
    if not ttnn.experimental.is_tensor_prefetcher_supported(device):
        pytest.skip(
            "tensor prefetcher unsupported here: either programmable DRAM cores are unavailable "
            "(need Blackhole and firmware >= 19.12.0.0), or TT_METAL_STREAMING_PROFILER=1 is set and its "
            "relay holds the DRISCs the prefetcher needs"
        )
