# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf cases for moe_fused_swiglu — run under `run_safe_pytest.sh --profile`.

    util = dram_read_bytes / (512e9 * device_kernel_time_s)

Read bytes = three bfp4 weight sets (count-independent) + ONE read of the real tokens at the
format's own granularity. Graded at emb 7168: count 128 -> 91.80 us / 0.566, 256 -> 108.00 us /
0.514, 512 -> 161.82 us / 0.388. emb 6144 and count=5120 carry no target but are reported.
"""

import pytest
import torch

import ttnn

from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu

TILE = 32
HIDDEN = 2048
BFP4_TILE = 576
NUM_GLOBAL_EXPERTS, NUM_LOCAL_EXPERTS, LOCAL_EXPERT_ID, GLOBAL_EXPERT_ID = 256, 8, 3, 137

# (emb, capacity, count) — the graded perf points plus the two report-only ones.
PERF_CASES = [
    (7168, 5120, 128),
    (7168, 5120, 256),
    (7168, 5120, 512),
    (7168, 1024, 256),  # same count at a small capacity: the allocation must cost nothing
    (6144, 5120, 256),  # report only
    (7168, 5120, 5120),  # report only (count == capacity)
]

_FORMATS = {"bf16_rm": (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT), "bfp8_tile": (ttnn.bfloat8_b, ttnn.TILE_LAYOUT)}


def read_bytes(count, emb, input_format):
    weights = 3 * (emb * HIDDEN // 1024) * BFP4_TILE
    if input_format == "bf16_rm":
        return weights + count * emb * 2.0
    return weights + ((count + TILE - 1) // TILE) * TILE * emb * 1.0625


def _build(emb, capacity, count, input_format, device):
    torch.manual_seed(42)
    x = torch.randn((1, 1, capacity, emb), dtype=torch.float32)
    if count < capacity:
        x[:, :, count:, :] = 100.0
    dt, lay = _FORMATS[input_format]
    tt_x = ttnn.from_torch(
        x.to(torch.bfloat16), dtype=dt, layout=lay, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_w = [
        ttnn.from_torch(
            torch.randn(s, dtype=torch.bfloat16),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for s in ((emb, HIDDEN), (emb, HIDDEN), (HIDDEN, emb))
    ]
    counts = torch.zeros(NUM_GLOBAL_EXPERTS, dtype=torch.int32)
    counts[GLOBAL_EXPERT_ID] = count
    idx = torch.tensor([(11 + 37 * i) % NUM_GLOBAL_EXPERTS for i in range(NUM_LOCAL_EXPERTS)], dtype=torch.int32)
    idx[LOCAL_EXPERT_ID] = GLOBAL_EXPERT_ID
    to_dev = lambda t: ttnn.from_torch(  # noqa: E731
        t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return tt_x, tt_w, to_dev(counts), to_dev(idx)


@pytest.mark.parametrize("emb, capacity, count", PERF_CASES)
@pytest.mark.parametrize("input_format", ["bf16_rm", "bfp8_tile"])
def test_perf(device, emb, capacity, count, input_format):
    tt_x, tt_w, tt_counts, tt_idx = _build(emb, capacity, count, input_format, device)
    out = moe_fused_swiglu(tt_x, tt_w[0], tt_w[1], tt_w[2], tt_counts, tt_idx, LOCAL_EXPERT_ID)
    assert list(out.shape) == [1, 1, capacity, emb]
    print(
        f"[perf] {input_format} emb={emb} cap={capacity} count={count} read_MB={read_bytes(count, emb, input_format) / 1e6:.3f}"
    )
