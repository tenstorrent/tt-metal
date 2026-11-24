# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Retile a TILE tensor between tile heights: via_row_major (baseline) vs face_rows vs runs.

See ttnn/ttnn/operations/examples/retile_face_regroup/README.md.

    scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_retile_face_regroup.py -k correctness

    # sweep, skipped unless RFR_SWEEPS=1 (or EXAMPLES_SWEEPS=1); RFR_CELLS="H:W:th_in:th_out,..." replaces the cells
    RFR_SWEEPS=1 scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_retile_face_regroup.py::test_retile_face_regroup_device_perf
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
    grid_xy,
    interleaved_ns,
    sweep_gate,
)
from ttnn.operations.examples.retile_face_regroup import VARIANTS, retile_face_regroup, unit_shape, work_split


def _make_input(device, h, w, th_in):
    torch.manual_seed(0)
    x = torch.randn((h, w), dtype=torch.float32).to(torch.bfloat16)
    tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, tile=ttnn.Tile([th_in, 32]))
    return tt, x


def _check(tt, x, th_out, variant, kernel_iters=1):
    out = retile_face_regroup(tt, th_out, variant=variant, kernel_iters=kernel_iters)
    assert tuple(out.tile.tile_shape) == (th_out, 32)
    assert torch.equal(ttnn.to_torch(out), x), f"{variant} {tt.tile.tile_shape}->{th_out} {list(x.shape)}: data differs"


_CORRECTNESS = [
    (32, 64, 16, 32),
    (32, 64, 32, 16),
    (64, 96, 1, 32),
    (32, 64, 32, 1),
    (64, 512, 8, 4),
    (128, 256, 32, 8),
    (32, 2048, 16, 32),
]


@pytest.mark.parametrize("cfg", _CORRECTNESS, ids=[f"{h}x{w}_{a}to{b}" for h, w, a, b in _CORRECTNESS])
def test_retile_face_regroup_correctness(device, cfg):
    h, w, th_in, th_out = cfg
    tt, x = _make_input(device, h, w, th_in)
    for variant in VARIANTS:
        _check(tt, x, th_out, variant)
        _check(tt, x, th_out, variant, kernel_iters=2)


# (H, W, th_in, th_out): tiny tensors (one unit), a few units, then a grid-filling tensor
_CELLS = [
    (32, 64, 16, 32),
    (32, 64, 32, 16),
    (32, 64, 1, 32),
    (32, 64, 32, 1),
    (32, 64, 8, 4),
    (32, 256, 16, 32),
    (32, 256, 32, 16),
    (128, 512, 16, 32),
    (128, 512, 1, 32),
    (2048, 2048, 16, 32),
    (2048, 2048, 32, 16),
    (2048, 2048, 1, 32),
]


def _cells():
    v = os.environ.get("RFR_CELLS")
    return [tuple(int(t) for t in c.split(":")) for c in v.split(",")] if v else _CELLS


@sweep_gate("RFR_SWEEPS")
def test_retile_face_regroup_device_perf(device):
    trials = int(os.environ.get("RFR_TRIALS", "3"))
    gx, gy = grid_xy(device)
    freq = chip_freq_mhz()
    lines = [
        "# retile_face_regroup — via_row_major vs face_rows vs runs",
        "",
        f"box={socket.gethostname()}  arch={arch_label(device)}  grid={gx}x{gy}  "
        f"clock={f'{freq:.0f}' if freq else 'unknown'} MHz  N={trials} (median)  kernel-iters=1 (per-launch latency)  "
        f"date={datetime.date.today().isoformat()}",
        "",
        "Interleaved DRAM bf16 TILE [H, W] with [th_in, 32] tiles in, [th_out, 32] tiles out. Units of "
        "max(th_in, th_out) rows x CW tile-columns over min(units, grid) Tensix cores, identical for all variants.",
        "",
        "| H | W | th_in -> th_out | Tensix cores | via_row_major ns | face_rows ns | runs ns | runs vs via_row_major |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for h, w, th_in, th_out in _cells():
        tt, x = _make_input(device, h, w, th_in)
        for v in VARIANTS:
            _check(tt, x, th_out, v)
        runners = {v: (lambda v=v: retile_face_regroup(tt, th_out, variant=v)) for v in VARIANTS}
        med = {v: statistics.median(s) for v, s in interleaved_ns(device, runners, trials=trials, warmup=1).items()}
        u, cw = unit_shape(th_in, th_out, w // 32)
        cores = len(work_split(device, (h // u) * (w // 32 // cw)))
        lines.append(
            f"| {h} | {w} | {th_in} -> {th_out} | {cores} | {med['via_row_major']:.0f} | {med['face_rows']:.0f} | "
            f"{med['runs']:.0f} | {med['via_row_major'] / med['runs']:.2f}x |"
        )
    report = "\n".join(lines) + "\n"
    logger.info("\n" + report)
    path = os.environ.get("RFR_REPORT", "")
    if path:
        Path(path).write_text(report)
