# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf-only bench for tilize — NOT part of the golden suite, NO PCC assert.

Underscore-prefixed so the correctness runs don't collect it. Measurement and
ablation need no correctness, and the golden INPUTS are deliberately tiny (they
cannot be bandwidth-bound, so they cannot measure what Track A optimizes).

    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/tilize/_bench_tilize.py

Prints `DEVICE KERNEL DURATION [ns]` per case (in-process device profiler) plus
the achieved DRAM bandwidth (read + write = 2x tensor bytes).

Shape regimes (op_design.md §9.4):
  (a) grid-filling square   [1,1,2048,2048]  — per-core DRAM efficiency
  (b) wide/short  MANDATORY [1,1,32,16384]   — NT_H=1: does the split fill the grid?
  (c) multi-block-per-core  [1,1,8192,1024]  — the only regime where a
                                               next-block overlap lever can show
  (d) smallest regime       [1,1,32,64]      — per-core-overhead levers (master.md B0)

Lever arms: every `levers=dict(<knob>=0)` case is the measured OFF arm of an
applied lever, so `eval/verify_levers.py` can find the knob and the ledger row
can carry BOTH numbers. `stub=` arms are the classification ablation (payload
removed, synchronization kept).
"""

import os

# In-process on-device profiler — all three, before the device opens.
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import pytest
import torch
import ttnn
from loguru import logger

from ttnn.operations.tilize import tilize
from ttnn.operations.tilize import tilize_program_descriptor as pd


_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

SHAPES = {
    "a_square": [1, 1, 2048, 2048],
    "b_wide_short": [1, 1, 32, 16384],
    "c_multiblock": [1, 1, 8192, 1024],
    "d_smallest": [1, 1, 32, 64],
}

_DTYPES = {"bf16": ttnn.bfloat16, "fp32": ttnn.float32}


def _read_kernel_ns(device):
    """On-device kernel ns over the programs dispatched since the last read."""
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for program in programs:
            results = getattr(program, "program_analyses_results", None) or {}
            entry = results.get(_DURATION_KEY)
            if entry is None:
                continue
            total += float(entry.duration)
            found = True
    return total if found else None


def _measure(device, shape, dtype, *, use_multicore=True, use_double_buffer=True, levers=None, ablate=None, label=""):
    """One warm launch (compile + program cache), then ONE measured launch.

    Device kernel duration has no warm-up transient, so a trial loop would just
    re-measure the same number (see /perf-measure "Measurement discipline").
    """
    levers = levers or {}
    ablate = ablate or {}
    saved = dict(pd.LEVERS)
    saved_ablate = dict(pd.ABLATE)
    pd.LEVERS.update(levers)
    pd.ABLATE.update(ablate)
    try:
        torch_input = torch.randn(shape).to(torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32)
        tt_input = ttnn.from_torch(
            torch_input,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        tilize(tt_input, use_multicore=use_multicore, use_double_buffer=use_double_buffer)
        ttnn.synchronize_device(device)
        _read_kernel_ns(device)  # flush the warm-up window

        out = tilize(tt_input, use_multicore=use_multicore, use_double_buffer=use_double_buffer)
        ttnn.synchronize_device(device)
        ns = _read_kernel_ns(device)
    finally:
        pd.LEVERS.update(saved)
        pd.ABLATE.update(saved_ablate)

    elem = 2 if dtype == ttnn.bfloat16 else 4
    tensor_bytes = 1
    for d in shape:
        tensor_bytes *= d
    tensor_bytes *= elem
    gbps = (2 * tensor_bytes) / ns if ns else float("nan")
    logger.info(f"BENCH tilize {label} shape={shape} ns={ns} GB/s={gbps:.1f}")
    assert ns is not None, "profiler produced no data (profiler-enabled build?)"
    return ns


# --- baseline: every regime x dtype ---------------------------------------
@pytest.mark.parametrize("regime", list(SHAPES))
@pytest.mark.parametrize("dtype_name", list(_DTYPES))
def test_bench_baseline(device, regime, dtype_name):
    _measure(device, SHAPES[regime], _DTYPES[dtype_name], label=f"baseline/{regime}/{dtype_name}")


# --- lever OFF arms (counterfactuals) --------------------------------------
@pytest.mark.parametrize(
    "regime",
    [
        r
        if r != "b_wide_short"
        else pytest.param(
            r,
            marks=pytest.mark.xfail(
                reason="w_split OFF on the wide/short shape OOMs L1: the input CB would be "
                "WT=512 tiles (op_design.md §1.3 candidate 2). The counterfactual cannot even "
                "be built — that IS the measurement.",
                strict=True,
                raises=RuntimeError,
            ),
        )
        for r in SHAPES
    ],
)
def test_bench_lever_w_split_off(device, regime):
    """A0/A1 grid fill: pure height split. On (b) this collapses to ONE core."""
    _measure(device, SHAPES[regime], ttnn.bfloat16, levers=dict(w_split=0), label=f"w_split=0/{regime}")


@pytest.mark.parametrize("regime", list(SHAPES))
def test_bench_lever_row_wise_off(device, regime):
    """master.md A1: split_work_to_cores column-wise (the binding default trap)."""
    _measure(device, SHAPES[regime], ttnn.bfloat16, levers=dict(row_wise=0), label=f"row_wise=0/{regime}")


@pytest.mark.parametrize("regime", list(SHAPES))
def test_bench_lever_block_write_off(device, regime):
    """master.md B7: one write barrier per tile page instead of per block."""
    _measure(device, SHAPES[regime], ttnn.bfloat16, levers=dict(block_write=0), label=f"block_write=0/{regime}")


@pytest.mark.parametrize("regime", list(SHAPES))
def test_bench_lever_double_buffer_off(device, regime):
    """master.md C16: depth-1 CBs — no read/compute/write overlap."""
    _measure(device, SHAPES[regime], ttnn.bfloat16, levers=dict(double_buffer=0), label=f"double_buffer=0/{regime}")


@pytest.mark.parametrize("regime", list(SHAPES))
def test_bench_lever_multicore_off(device, regime):
    """A0 baseline: the whole op on one core."""
    _measure(device, SHAPES[regime], ttnn.bfloat16, levers=dict(multicore=0), label=f"multicore=0/{regime}")


@pytest.mark.parametrize("regime", list(SHAPES))
def test_bench_lever_page_write_off(device, regime):
    """master.md B5: each tile page split into two half-page transactions."""
    _measure(device, SHAPES[regime], ttnn.bfloat16, levers=dict(page_write=0), label=f"page_write=0/{regime}")


@pytest.mark.parametrize("regime", list(SHAPES))
def test_bench_lever_noc_split_off(device, regime):
    """master.md B9: reader/writer NoC assignment swapped."""
    _measure(device, SHAPES[regime], ttnn.bfloat16, levers=dict(noc_split=0), label=f"noc_split=0/{regime}")


@pytest.mark.parametrize("regime", list(SHAPES))
def test_bench_lever_regime_select_off(device, regime):
    """master.md D20: no compile-time specialization — the pad reader on the aligned path."""
    _measure(device, SHAPES[regime], ttnn.bfloat16, levers=dict(regime_select=0), label=f"regime_select=0/{regime}")


@pytest.mark.parametrize("regime", list(SHAPES))
def test_bench_lever_fp32_dest_off(device, regime):
    """master.md F25: fp32 DEST + lossless unpack (the exactness gate) turned off."""
    _measure(device, SHAPES[regime], ttnn.float32, levers=dict(fp32_dest=0), label=f"fp32_dest=0/{regime}")


# --- classification ablation (op_design.md §9.1) ---------------------------
# Payload removed, synchronization intact. Output is WRONG by design — this is a
# measurement, never a correctness check. Peel stages CUMULATIVELY: stages
# overlap, so a single removal is only a lower bound on that stage's cost.
@pytest.mark.parametrize("regime", list(SHAPES))
@pytest.mark.parametrize(
    "ablate, name",
    [
        ({"compute": 1}, "no_compute"),
        ({"dm": 1}, "no_dm"),
        ({"compute": 1, "dm": 1}, "sync_only"),
    ],
    ids=["no_compute", "no_dm", "sync_only"],
)
def test_bench_ablation(device, regime, ablate, name):
    _measure(device, SHAPES[regime], ttnn.bfloat16, ablate=ablate, label=f"ablate:{name}/{regime}")
