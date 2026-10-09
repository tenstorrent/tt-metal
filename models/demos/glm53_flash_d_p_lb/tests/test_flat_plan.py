# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Print flat_routed_expert's core plan for GLM's MoE shape (H 4096, I 2048, 36 local experts, 288 global, m 8192, bfp4)."""

import pytest

import ttnn


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_flat_plan(device):
    p = dict(
        ttnn._ttnn.operations.bringup.flat_routed_expert_plan(
            device, 4096, 2048, 36, 288, 8192, weights_bf8=False, pin=1
        )
    )
    for k in sorted(p):
        v = p[k]
        s = str(v)
        print(
            f"[plan] {k}: {s if len(s) < 300 else s[:300] + '...'}"
            + (f"  (len {len(v)})" if isinstance(v, (list, tuple)) else ""),
            flush=True,
        )
