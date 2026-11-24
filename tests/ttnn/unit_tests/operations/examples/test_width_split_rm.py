# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""width_split on Layout::ROW_MAJOR sticks with no combine (tilize): k column groups per tile-row.

See ttnn/ttnn/operations/examples/width_split/README.md ("ROW_MAJOR without a combine").

    # every k matches torch
    scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_width_split_rm.py -k correctness

    # sweep: device kernel ns per (tile-rows, W, k); skipped unless WS_RM_SWEEPS=1 (or EXAMPLES_SWEEPS=1);
    # WS_RM_CELLS="R_t:W,..." and WS_RM_KS="1,2,..." override; written to WS_RM_REPORT when set
    WS_RM_SWEEPS=1 scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_width_split_rm.py::test_width_split_rm_device_perf
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import datetime
import socket
import statistics
from pathlib import Path

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.unit_tests.operations.examples.perf_test_utils import (
    arch_label,
    chip_freq_mhz,
    grid_cores,
    grid_xy,
    interleaved_ns,
    sweep_gate,
)
from ttnn.operations.examples.width_split.rm_split import rm_split_layout, width_split_rm


def _make_input(device, h, w):
    torch.manual_seed(0)
    x = torch.randn((h, w), dtype=torch.float32).to(torch.bfloat16)
    tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    return tt, x


def _check(tt, x, k, kernel_iters=1, trim=False):
    out = ttnn.to_torch(width_split_rm(tt, k=k, trim=trim, kernel_iters=kernel_iters))
    assert torch.equal(out, x), f"k={k} trim={trim} {list(x.shape)}: tilize output differs from the input"


@pytest.mark.parametrize(
    "h, w, ks",
    [(32, 64, (1, 2)), (32, 2048, (1, 4, 64)), (64, 128, (1, 2, 4)), (96, 1024, (1, 8)), (2048, 256, (1, 2))],
)
def test_width_split_rm_correctness(device, h, w, ks):
    tt, x = _make_input(device, h, w)
    for k in ks:
        if (h // 32) * k > grid_cores(device) and h // 32 < grid_cores(device):
            continue
        for trim in (False, True):
            _check(tt, x, k, trim=trim)
            _check(tt, x, k, kernel_iters=2, trim=trim)


# (tile-rows, W): few tile-rows (the grid mostly idle), short and long sticks, then more tile-rows
_CELLS = [
    (1, 64),
    (1, 128),
    (1, 256),
    (1, 512),
    (1, 2048),
    (1, 8192),
    (2, 64),
    (2, 128),
    (2, 256),
    (2, 1024),
    (4, 64),
    (4, 256),
    (4, 2048),
    (16, 128),
    (16, 512),
    (16, 2048),
    (47, 256),
    (64, 256),
    (64, 2048),
]
_KS = (1, 2, 4, 8, 16, 32, 64)


def _env_cells():
    v = os.environ.get("WS_RM_CELLS")
    return [tuple(int(t) for t in c.split(":")) for c in v.split(",")] if v else _CELLS


@sweep_gate("WS_RM_SWEEPS")
def test_width_split_rm_device_perf(device):
    trials = int(os.environ.get("WS_RM_TRIALS", "3"))
    ks_all = tuple(int(t) for t in os.environ.get("WS_RM_KS", ",".join(map(str, _KS))).split(","))
    gx, gy = grid_xy(device)
    freq = chip_freq_mhz()
    lines = [
        "# width_split — Layout::ROW_MAJOR tilize, no combine: k column groups per tile-row",
        "",
        f"box={socket.gethostname()}  arch={arch_label(device)}  grid={gx}x{gy}  "
        f"clock={f'{freq:.0f}' if freq else 'unknown'} MHz  N={trials} (median)  kernel-iters=1 (per-launch latency)  "
        f"date={datetime.date.today().isoformat()}",
        "",
        "Interleaved DRAM ROW_MAJOR bf16 [32 R_t, W] in, interleaved DRAM TILE out. k = 1: whole tile-rows per "
        "Tensix core; k > 1: each tile-row split into k column groups (pieces of 2W/k bytes per stick). "
        "ns per k (helper defaults); best = fastest k and its speedup over k = 1. trim = the tilize helper with "
        "NoReconfigure + InitOnly (no reconfig after compute_kernel_hw_startup, no trailing uninit), re-measured "
        "interleaved against the default at k = 1 and at the best k.",
        "",
        "| R_t | W | stick B | "
        + " | ".join(f"k={k}" for k in ks_all)
        + " | best k | trim at k=1: default -> trimmed ns | trim at best k | k=1 default -> best k trimmed |",
        "|---:|---:|---:|" + "---:|" * len(ks_all) + "---|---|---|---:|",
    ]
    for rows_t, w in _env_cells():
        wt = w // 32
        tt, x = _make_input(device, rows_t * 32, w)
        ks = [
            k
            for k in ks_all
            if wt % k == 0 and (k == 1 or rows_t * k <= grid_cores(device) or rows_t >= grid_cores(device))
        ]
        for k in ks:
            _check(tt, x, k)
        runners = {k: (lambda k=k: width_split_rm(tt, k=k)) for k in ks}
        samples = interleaved_ns(device, runners, trials=trials, warmup=1)
        med = {k: statistics.median(v) for k, v in samples.items()}
        best = min(med, key=med.get)
        for k in {1, best}:
            _check(tt, x, k, trim=True)
        trim_runners = {
            "k1": (lambda: width_split_rm(tt, k=1)),
            "k1_trim": (lambda: width_split_rm(tt, k=1, trim=True)),
            "best": (lambda: width_split_rm(tt, k=best)),
            "best_trim": (lambda: width_split_rm(tt, k=best, trim=True)),
        }
        tmed = {
            n: statistics.median(v) for n, v in interleaved_ns(device, trim_runners, trials=trials, warmup=1).items()
        }
        cells = [f"{med[k]:.0f}" if k in med else "" for k in ks_all]
        lines.append(
            f"| {rows_t} | {w} | {2 * w} | " + " | ".join(cells) + f" | k={best} ({med[1] / med[best]:.2f}x, "
            f"{2 * w // best} B pieces, {len(rm_split_layout(device, rows_t, wt, best))} Tensix cores) | "
            f"{tmed['k1']:.0f} -> {tmed['k1_trim']:.0f} ({tmed['k1'] / tmed['k1_trim']:.2f}x) | "
            f"{tmed['best']:.0f} -> {tmed['best_trim']:.0f} ({tmed['best'] / tmed['best_trim']:.2f}x) | "
            f"{tmed['k1'] / tmed['best_trim']:.2f}x |"
        )
    report = "\n".join(lines) + "\n"
    logger.info("\n" + report)
    path = os.environ.get("WS_RM_REPORT", "")
    if path:
        Path(path).write_text(report)
