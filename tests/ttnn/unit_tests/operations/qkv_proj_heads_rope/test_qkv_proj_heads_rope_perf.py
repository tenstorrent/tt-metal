# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Production perf rows (prompt table). Run with scripts/run_safe_pytest.sh --profile; read DEVICE KERNEL
DURATION per op from the profiler CSV. Production placement: x, tables, q/k/v in L1; tables cut to S."""

import pytest

from ttnn.operations.qkv_proj_heads_rope import qkv_proj_heads_rope

from .test_qkv_proj_heads_rope import check_outputs, device_inputs, make_inputs, reference

# (B, S, stock us, target us)
PERF_ROWS = [
    (8, 512, 226.3, 90),
    (16, 256, 228.5, 90),
    (32, 128, 246.4, 90),
    (8, 384, 178.8, 68),
    (4, 512, 123.5, 46),
    (8, 256, 124.1, 46),
    (2, 512, 75.0, 28),
    (4, 256, 79.8, 28),
]


@pytest.mark.parametrize("batch, seq, stock_us, target_us", PERF_ROWS, ids=[f"{b}x{s}" for b, s, *_ in PERF_ROWS])
def test_qkv_proj_heads_rope_perf(device, batch, seq, stock_us, target_us):
    x, w, b, cos, sin = make_inputs(batch, seq, seq)
    tensors = device_inputs(device, x, w, b, cos, sin)
    outputs = qkv_proj_heads_rope(*tensors)
    check_outputs(outputs, reference(x, w, b, cos, sin))
