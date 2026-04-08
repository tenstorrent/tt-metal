# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for TTML Python tests."""

import ttnn
import ttml


def param_to_numpy_bf16_current(param):
    """Read current BF16-backed value to avoid stale cached FULL views."""
    # Workaround for stale FULL cache behavior. Tracking: #41657
    return ttnn.to_torch(param.get_value(ttml.autograd.PreferredPrecision.HALF)).float().cpu().numpy()
