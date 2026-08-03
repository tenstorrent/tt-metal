# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf 2 measurement harness — ONE test, the case list comes from the environment.

`run_safe_pytest.sh` does not preserve quoted `-k` expressions or `[...]` node-id brackets, so the
Perf-1 harness could not be pointed at a single cell. This one is selected with

    MOE_R2_CASES="7168,5120,256,bf16_rm;7168,5120,128,bf16_rm"  \
      scripts/run_safe_pytest.sh --profile <this file>

Default = the focus shape alone (emb 7168, capacity 5120, count 256, bf16_rm). Correctness is not
asserted here beyond shape (that is the golden suite's job); this file exists to produce ONE fresh
device-kernel duration per cell, plus the per-stage zones when the profiler is on.

Weights are placed at the op's DESIGNED ND shard — see `test_moe_fused_swiglu_perf.py`'s docstring for
the measurement; `MOE_PERF_WPLACE=interleaved` restores the old, slower call site. Any A/B run through
this harness must hold the placement fixed, since it is worth up to 11 % on its own.
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
#: N and the weight dtype are harness variables now — the op generalizes over both, and the
#: read-bytes denominator below must follow them or the reported utilisation silently lies.
HIDDEN = int(os.environ.get("MOE_HIDDEN", 2048))
WEIGHT_DTYPE = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}[
    os.environ.get("MOE_WDTYPE", "bfp4")
]
W_TILE = ttnn.tile_size(WEIGHT_DTYPE)
BFP4_TILE = 576
NUM_GLOBAL_EXPERTS, NUM_LOCAL_EXPERTS, LOCAL_EXPERT_ID, GLOBAL_EXPERT_ID = 256, 8, 3, 137

_FORMATS = {"bf16_rm": (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT), "bfp8_tile": (ttnn.bfloat8_b, ttnn.TILE_LAYOUT)}

#: The GUARD SET, one representative per distinct kernel path x layout x M regime. Used verbatim as
#: the no-regression set of the Perf-2 tournament.
GUARD_SET = (
    "7168,5120,128,bf16_rm;"
    "7168,5120,256,bf16_rm;"
    "7168,5120,512,bf16_rm;"
    "7168,1024,256,bf16_rm;"
    "6144,5120,256,bf16_rm;"
    "7168,5120,5120,bf16_rm;"
    "7168,5120,128,bfp8_tile;"
    "7168,5120,256,bfp8_tile;"
    "7168,5120,512,bfp8_tile;"
    "7168,1024,256,bfp8_tile;"
    "6144,5120,256,bfp8_tile;"
    "7168,5120,5120,bfp8_tile"
)

_DEFAULT = "7168,5120,256,bf16_rm"

#: DUPLICATED from `test_moe_fused_swiglu_perf.py` on purpose: pytest 9 imports test modules in
#: `importlib` mode, so a sibling test module is not importable by name and neither is `conftest`.
#: Twenty lines of duplication beats a sys.path hack in the harness that produces the graded numbers.
WPLACE = os.environ.get("MOE_PERF_WPLACE", "nd_shard")


def weight_configs(device, emb, hidden, wplace):
    if wplace == "nd_shard":
        return weight_memory_configs(device, emb, hidden, core_grid=CORE_GRID)
    if wplace == "interleaved":
        return ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
    raise ValueError(f"unknown MOE_PERF_WPLACE {wplace!r}")


def assert_placement(tt_w, wplace):
    """Check the READER's predicate — an interleaved weight is silently correct, just slower."""
    widths = [nd_shard_n_tiles(w) for w in tt_w]
    if wplace == "nd_shard":
        assert all(w > 0 for w in widths), f"asked for nd_shard but the reader sees interleaved: {widths}"
    else:
        assert all(w == 0 for w in widths), f"asked for interleaved but the reader sees shards: {widths}"
    return widths


def _cases():
    spec = os.environ.get("MOE_R2_CASES", _DEFAULT)
    if spec == "guard":
        spec = GUARD_SET
    out = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        emb, capacity, count, fmt = part.split(",")
        out.append((int(emb), int(capacity), int(count), fmt.strip()))
    return out


def read_bytes(count, emb, input_format):
    weights = 3 * (emb * HIDDEN // 1024) * W_TILE
    if input_format == "bf16_rm":
        return weights + count * emb * 2.0
    return weights + ((count + TILE - 1) // TILE) * TILE * emb * 1.0625


def _build(emb, capacity, count, input_format, device):
    torch.manual_seed(42)
    x = torch.randn((1, 1, capacity, emb), dtype=torch.float32)
    if count < capacity:
        x[:, :, count:, :] = 100.0  # hostile sentinel in the phantom rows
    dt, lay = _FORMATS[input_format]
    tt_x = ttnn.from_torch(
        x.to(torch.bfloat16), dtype=dt, layout=lay, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    gate_up_mc, down_mc = weight_configs(device, emb, HIDDEN, WPLACE)
    tt_w = [
        ttnn.from_torch(
            torch.randn(s, dtype=torch.bfloat16),
            dtype=WEIGHT_DTYPE,
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


@pytest.mark.parametrize("case", _cases(), ids=lambda c: f"{c[3]}_e{c[0]}_c{c[1]}_n{c[2]}")
def test_r2_perf(device, case):
    emb, capacity, count, input_format = case
    tt_x, tt_w, tt_counts, tt_idx = _build(emb, capacity, count, input_format, device)
    out = moe_fused_swiglu(tt_x, tt_w[0], tt_w[1], tt_w[2], tt_counts, tt_idx, LOCAL_EXPERT_ID, core_grid=CORE_GRID)
    assert list(out.shape) == [1, 1, capacity, emb]
    print(
        f"[r2perf] {input_format} emb={emb} cap={capacity} count={count} wplace={WPLACE} "
        f"read_MB={read_bytes(count, emb, input_format) / 1e6:.3f}"
    )
