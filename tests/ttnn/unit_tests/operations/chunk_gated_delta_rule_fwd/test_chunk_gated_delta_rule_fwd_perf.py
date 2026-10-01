# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-perf harness for chunk_gated_delta_rule_fwd (perf refinements' measuring stick).

Run under the Tracy device profiler:

    scripts/run_safe_pytest.sh --profile --run-all \
        tests/ttnn/unit_tests/operations/chunk_gated_delta_rule_fwd/test_chunk_gated_delta_rule_fwd_perf.py

Each case dispatches the op WARMUP + 1 times (the first compiles); the per-op CSV the profiler
emits (generated/profiler/reports/<ts>/ops_perf_results*.csv) carries one row per dispatch — take
the LAST row of each case's group as its device-kernel duration.  `PERF_CASES` is the perf target
plus the config-spanning guard set (one representative per distinct extent regime × dtype ×
state_mode); keep it in sync with op_requirements.md.
"""

from __future__ import annotations

import pytest

import ttnn
from ttnn.operations.chunk_gated_delta_rule_fwd import chunk_gated_delta_rule_fwd

from eval.golden_tests.chunk_gated_delta_rule_fwd.helpers import make_reference_inputs, quantize

WARMUP = 1

# (id, (B, T, H, K, V), chunk, dtype, state_mode)
PERF_CASES = [
    # --- perf target: LOOSE Qwen3.5 prefill (both LOOSE configs) ---
    ("loose_qwen35_bf16_h0", (1, 4096, 16, 128, 128), 64, ttnn.bfloat16, "with_h0"),
    ("loose_qwen35_fp32", (1, 4096, 16, 128, 128), 64, ttnn.float32, "no_h0"),
    # --- guard set ---
    ("h32_nv2_vs2_fp32", (1, 256, 32, 128, 128), 64, ttnn.float32, "no_h0"),
    ("items_per_core_gt1_fp32", (4, 128, 16, 64, 64), 32, ttnn.float32, "with_h0"),
    ("widev_nv8_bf16_h0", (1, 256, 4, 128, 256), 64, ttnn.bfloat16, "with_h0"),
    ("long_scan_fp32_h0", (1, 2048, 2, 128, 128), 64, ttnn.float32, "with_h0"),
    ("ragged_long_bf16", (1, 1000, 4, 128, 128), 64, ttnn.bfloat16, "no_h0"),
    ("tiny_fp32", (1, 32, 1, 32, 32), 32, ttnn.float32, "no_h0"),
]


@pytest.mark.parametrize("case", PERF_CASES, ids=[c[0] for c in PERF_CASES])
def test_perf(case, device):
    _, shape, chunk, dtype, state_mode = case
    ref = quantize(make_reference_inputs(shape, state_mode, seed=0), dtype)
    dev = {
        n: (None if t is None else ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device))
        for n, t in ref.items()
    }
    for _ in range(WARMUP + 1):
        outs = chunk_gated_delta_rule_fwd(
            dev["q"], dev["k"], dev["v"], dev["g"], dev["beta"], initial_state=dev["initial_state"], chunk_size=chunk
        )
        ttnn.synchronize_device(device)
        for t in outs:
            ttnn.deallocate(t)
