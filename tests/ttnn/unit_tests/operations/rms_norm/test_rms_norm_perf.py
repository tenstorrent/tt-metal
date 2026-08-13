# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm — perf harness.  Correctness is the only assertion; timing comes
from the profiler CSV:

    scripts/run_safe_pytest.sh --profile --run-all \
        tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_perf.py

Each case runs ONCE per profiled run — device kernel time has no warm-up
transient, so a standing loop would just re-measure the same number.

`floor_1tile` is the ablation baseline: 1 core, 1 tile, no combine, so its
duration is pure fixed cost (dispatch + kernel boot).  Subtract it from every
other row to get the payload cost.

A/B knob — the row-group rectangle search:
    echo grid_wide     > /tmp/rms_norm_rect_search   # shipped: any rect_w x rect_h
    echo column_pinned > /tmp/rms_norm_rect_search   # rect_w == grid_x (or a divisor)

It is read from a FILE, not the environment, on purpose: under `--profile` the
real run executes inside `python -m tracy`, and an ad-hoc env var does not reach
that child — the warm pass would honour it and the profiled run would not,
silently measuring the same variant twice.

CAVEAT, verified on this box: the profiler CSV's CORE COUNT column is NOT
trustworthy for a generic op — it reported 56 for a run whose program provably
used 11.  Read DEVICE KERNEL DURATION, and get the geometry from
`rms_norm_program_descriptor._plan`.

Measured, Blackhole p150 (11x10 compute grid), bf16 + gamma, TILE layout:

    case                  cores   ns    ns-floor   MB    payload GB/s
    decode_w7168             56  13026     9512   0.92        97
    decode_w8192             64  14275    10761   1.05        98
    decode_w4096             32   9685     6171   0.52        85
    decode_2rows_w12288     110  22677    19163   3.15       164
    prefill_2048x1024       110  45638    42124   8.39       199
    batch4d                  64   8953     5439   1.05       193
    floor_1tile               1   3514        0      -         -

Conclusion: the fixed cost (dispatch + kernel boot) is ~3.5 us, and the payload
runs at ~100 GB/s in the decode regime and ~200 GB/s in the prefill regime.  The
A/B above moves the wall by <2% on every row even where it changes occupancy
5x (decode_w7168: 11 -> 56 cores), so per-core work is NOT what binds these
shapes — the decode rows are short-transfer DRAM-efficiency bound (S=4 tiles per
core) and everything carries the ~3.5 us floor.  The grid-wide search is kept
because it is the design's mandate and it is what leaves headroom at larger W;
what has to shorten before it shows is the fixed cost and the per-core transfer
size, not the core count.

The HIDDEN_TILES_PER_CORE_FLOOR sweep (op_design.md's "Grid synchronization"
perf lamp) says the same thing — every cell is inside +-1% of every other, i.e.
noise:

    case                  floor=2  floor=4  floor=8  floor=16
    decode_w7168            13183    13287    13194     13163
    decode_w8192            14170    14036    14189     14181
    decode_w4096             9716     9730     9695      9724
    decode_2rows_w12288     22776    23041    22824     22824
    prefill_2048x1024       45681    46017    46487     45420
    batch4d                  9009     9005     8968      9005
    floor_1tile              3622     3532     3506      3547

So the knob stays at its measured-catalog default of 4; it is live and this
harness re-measures it in one command when the surrounding costs change.
"""

import pathlib

import pytest
import torch
import ttnn

import ttnn.operations.rms_norm.rms_norm_program_descriptor as pd
from ttnn.operations.rms_norm import rms_norm

# Shapes op_design.md calls decisive, plus the two regime-pinned families and
# the fixed-cost floor.
PERF_SHAPES = [
    pytest.param((1, 1, 32, 7168), id="decode_w7168"),
    pytest.param((1, 1, 32, 8192), id="decode_w8192"),
    pytest.param((1, 1, 32, 4096), id="decode_w4096"),
    pytest.param((1, 1, 64, 12288), id="decode_2rows_w12288"),
    pytest.param((1, 1, 2048, 1024), id="prefill_2048x1024"),
    pytest.param((2, 4, 128, 256), id="batch4d"),
    pytest.param((32, 32), id="floor_1tile"),
]


def _column_pinned(gx, gy):
    """The pre-fix candidate set: rect_w is grid_x or a divisor of it."""
    out = []
    for rect_w in range(1, gx + 1):
        if gx % rect_w == 0:
            out.append((rect_w, 1))
    for rect_h in range(2, gy + 1):
        out.append((gx, rect_h))
    return [(w * h, w, h) for (w, h) in out]


_SELECTOR = pathlib.Path("/tmp/rms_norm_rect_search")
RECT_SEARCH = _SELECTOR.read_text().strip() if _SELECTOR.exists() else "grid_wide"

# The design's "Grid synchronization" perf lamp: sweep the hidden-granularity
# floor. Higher floor => fewer, fatter slices (bigger per-core NoC transfers,
# fewer combine contributors); lower floor => more cores, thinner transfers.
#     echo 8 > /tmp/rms_norm_hidden_floor
_FLOOR_SELECTOR = pathlib.Path("/tmp/rms_norm_hidden_floor")
HIDDEN_FLOOR = int(_FLOOR_SELECTOR.read_text().strip()) if _FLOOR_SELECTOR.exists() else None


@pytest.fixture(autouse=True)
def rect_search(monkeypatch):
    if RECT_SEARCH == "column_pinned":
        monkeypatch.setattr(pd, "_rect_candidates", _column_pinned)
    if HIDDEN_FLOOR is not None:
        monkeypatch.setattr(pd, "HIDDEN_TILES_PER_CORE_FLOOR", HIDDEN_FLOOR)
    return RECT_SEARCH


@pytest.mark.parametrize("shape", PERF_SHAPES)
def test_rms_norm_perf(device, shape, rect_search):
    torch.manual_seed(42)
    torch_x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    torch_gamma = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32).to(torch.bfloat16)

    x = ttnn.from_torch(
        torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out = ttnn.to_torch(rms_norm(x, gamma=gamma)).to(torch.float32)

    xf = torch_x.to(torch.float32)
    expected = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * torch_gamma.to(torch.float32).reshape(-1)
    a, b = out.flatten(), expected.flatten()
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    assert pcc > 0.995, f"{shape} {rect_search}: PCC {pcc}"
