# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import ttnn


def test_moe_compute_reports_supported_weight_dtypes():
    assert tuple(ttnn.experimental.moe_compute_supported_weight_dtypes()) == (
        ttnn.bfloat4_b,
        ttnn.bfloat8_b,
        ttnn.bfloat16,
    )
