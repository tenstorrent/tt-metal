# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf cases for moe_fused_swiglu — run under `run_safe_pytest.sh --profile`.

    util = dram_read_bytes / (512e9 * device_kernel_time_s)

Read bytes = three bfp4 weight sets (count-independent) + ONE read of the real tokens at the
format's own granularity. Graded at emb 7168: count 128 -> 91.80 us / 0.566, 256 -> 108.00 us /
0.514, 512 -> 161.82 us / 0.388. emb 6144 and count=5120 carry no target but are reported.

WEIGHT PLACEMENT — this harness now builds the weights the way PERF 12 designs for. It used to hand
the op plain `DRAM_MEMORY_CONFIG` weights, which is a SILENTLY SLOWER call site: placement is not a
knob, `nd_shard_n_tiles()` reads the shard width off the tensors the caller supplies, and an
interleaved weight is correct but takes the uncoalesced one-request-per-tile stream. Measured at
88 cores, bf16_rm, 7 reps: interleaved 102.51 / 135.37 / 241.49 us against ND-sharded
91.34 / 130.67 / 229.58 at counts 128 / 256 / 512 — so the old harness understated the op by up to
11 % and could not reproduce Perf 12's own "count 128 target MET". Weights are per-expert load-time
constants, so a deployment shards them once; the bytes read are identical either way, which is why
the `util` denominator does not change. `MOE_PERF_WPLACE=interleaved` restores the old call site for
an A/B.
"""

import os

import pytest
import torch

import ttnn

from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_program_descriptor import (
    nd_shard_n_tiles,
    weight_memory_configs,
)


#: The op's worker grid is a PARAMETER now, not an environment knob. `MOE_GRID=11x8` selects the
#: 88-core configuration every graded number is quoted at; empty = the device's full grid. It is a
#: harness variable, passed through as `core_grid=`, so the op itself stays env-free.
def _core_grid():
    g = os.environ.get("MOE_GRID", "").strip().lower()
    if not g:
        return None
    x, y = g.split("x")
    return (int(x), int(y))


CORE_GRID = _core_grid()

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

#: The op's DESIGNED weight placement. See the module docstring for the measured cost of getting this
#: wrong; `interleaved` is kept only as an A/B.
WPLACE = os.environ.get("MOE_PERF_WPLACE", "nd_shard")


def weight_configs(device, emb, hidden, wplace):
    if wplace == "nd_shard":
        return weight_memory_configs(device, emb, hidden, core_grid=CORE_GRID)
    if wplace == "interleaved":
        return ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
    raise ValueError(f"unknown MOE_PERF_WPLACE {wplace!r}")


def assert_placement(tt_w, wplace):
    """Check the READER's own predicate, not the memory config we asked for.

    An interleaved weight is silently CORRECT — just slower — so a placement that failed to apply
    would otherwise be reported as a legitimate number against the graded target.
    """
    widths = [nd_shard_n_tiles(w) for w in tt_w]
    if wplace == "nd_shard":
        assert all(w > 0 for w in widths), f"asked for nd_shard but the reader sees interleaved: {widths}"
    else:
        assert all(w == 0 for w in widths), f"asked for interleaved but the reader sees shards: {widths}"
    return widths


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
    gate_up_mc, down_mc = weight_configs(device, emb, HIDDEN, WPLACE)
    tt_w = [
        ttnn.from_torch(
            torch.randn(s, dtype=torch.bfloat16),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=mc,
        )
        for s, mc in (((emb, HIDDEN), gate_up_mc), ((emb, HIDDEN), gate_up_mc), ((HIDDEN, emb), down_mc))
    ]
    assert_placement(tt_w, WPLACE)
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
    out = moe_fused_swiglu(tt_x, tt_w[0], tt_w[1], tt_w[2], tt_counts, tt_idx, LOCAL_EXPERT_ID, core_grid=CORE_GRID)
    assert list(out.shape) == [1, 1, capacity, emb]
    print(
        f"[perf] {input_format} emb={emb} cap={capacity} count={count} wplace={WPLACE} "
        f"read_MB={read_bytes(count, emb, input_format) / 1e6:.3f}"
    )
