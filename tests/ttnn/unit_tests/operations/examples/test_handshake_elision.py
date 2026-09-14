# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `handshake_elision` example.

A same-spec, fully L1-resident tilize run two ways: the standard three-kernel
skeleton with the per-tile-row CB credit protocol ("handshake"), and one kernel
with no CB protocol at all ("no_handshake"). See ttnn/ttnn/operations/examples/handshake_elision/README.md.

    # every variant x shard geometry is bitwise the same tilize
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_handshake_elision.py::test_handshake_elision_correctness

    # device kernel duration per arm over tiles/core (in-process profiler)
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_handshake_elision.py::test_handshake_elision_device_perf
"""

import os

# Enable the on-device profiler IN-PROCESS (needs all three, set before the device
# opens). Scoped to this module so it doesn't perturb other examples' measurement.
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
# Gag the C++ logger (per-read profiler histograms would bury the report); the
# loguru report below still prints and the numbers are unaffected.
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import pytest
import torch

import ttnn
from ttnn.operations.examples.handshake_elision import (
    handshake_elision,
    sharded_memory_config,
    VARIANTS,
    NO_HANDSHAKE,
    KERNELS,
)

from loguru import logger

TILE = 32


def _parse_shards(spec):
    out = []
    for item in spec.split(","):
        ht, wt = item.lower().split("x")
        out.append((int(ht), int(wt)))
    return tuple(out)


# Defaults are overridable via env so the CLI (python -m ...handshake_elision) can
# measure the caller's own geometry through the same in-process path.
SHARD_SWEEP = _parse_shards(os.environ.get("HE_SHARDS", "1x2,1x4,2x4,4x4,4x8,8x8"))  # (Ht, Wt) tiles per core
NUM_CORES = int(os.environ.get("HE_CORES", "4"))  # cores in row 0, one shard each
VARIANT_SWEEP = tuple(os.environ.get("HE_VARIANTS", ",".join(VARIANTS)).split(","))
KERNEL_ITERS = int(os.environ.get("HE_ITERS", "1"))
N_WARMUP = 5
N_PROFILE_ITERS = int(os.environ.get("HE_TRIALS", "10"))

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


def _make_input(device, shard_ht, shard_wt, num_cores):
    torch.manual_seed(0)
    shape = (shard_ht * TILE * num_cores, shard_wt * TILE)
    # Distinct values per position: a tile-row addressed at the wrong index would
    # still "look" plausible under a tolerance, so the gate below is bitwise.
    torch_input = (torch.rand(shape, dtype=torch.float32) * 2.0 - 1.0).to(torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=sharded_memory_config(shard_ht, shard_wt, num_cores),
    )
    return torch_input, tt_input


def _read_kernel_ns(device):
    """Sum of on-device kernel duration over programs dispatched since the last read."""
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


def _measure_ns(device, run_fn):
    """Average ns/op: warm up, flush, run N, read, divide by N."""
    for _ in range(N_WARMUP):
        run_fn()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # flush the warmup window
    for _ in range(N_PROFILE_ITERS):
        run_fn()
    total_ns = _read_kernel_ns(device)
    return total_ns / N_PROFILE_ITERS if total_ns is not None else None


# Both arms over the smallest and largest shard, a non-square one, a 1-core and a
# many-core grid, and a multi-iteration loop (the "no_handshake" arm must re-address
# from the CB base every iteration; the baseline must re-publish and re-retire).
_CORRECTNESS_CASES = [
    (variant, ht, wt, cores, iters)
    for variant in VARIANTS
    for (ht, wt, cores, iters) in [(1, 2, 1, 1), (1, 2, 4, 1), (2, 4, 2, 1), (8, 8, 4, 1), (4, 4, 4, 3), (1, 1, 5, 2)]
]


@pytest.mark.parametrize("variant,shard_ht,shard_wt,num_cores,kernel_iters", _CORRECTNESS_CASES)
def test_handshake_elision_correctness(device, variant, shard_ht, shard_wt, num_cores, kernel_iters):
    """Every arm produces the tiled layout of the input, bitwise."""
    torch_input, tt_input = _make_input(device, shard_ht, shard_wt, num_cores)
    out = handshake_elision(
        tt_input,
        variant=variant,
        shard_ht=shard_ht,
        shard_wt=shard_wt,
        num_cores=num_cores,
        kernel_iters=kernel_iters,
    )
    assert out.layout == ttnn.TILE_LAYOUT
    result = ttnn.to_torch(out)
    assert list(result.shape) == list(torch_input.shape), f"{result.shape} != {torch_input.shape}"
    assert torch.equal(
        result, torch_input
    ), f"{variant}: tilize is not bitwise exact (max |diff| {(result.float() - torch_input.float()).abs().max()})"


def test_handshake_elision_device_perf(device):
    """Measure device kernel duration for each arm over tiles/core.

    Correctness lives in test_handshake_elision_correctness; this only measures and
    reports (perf is evidence, never a pass/fail -- the only assertion is that the
    profiler produced a number).
    """
    results = {}
    for shard_ht, shard_wt in SHARD_SWEEP:
        _, tt_input = _make_input(device, shard_ht, shard_wt, NUM_CORES)
        for variant in VARIANT_SWEEP:
            value = _measure_ns(
                device,
                lambda: handshake_elision(
                    tt_input,
                    variant=variant,
                    shard_ht=shard_ht,
                    shard_wt=shard_wt,
                    num_cores=NUM_CORES,
                    kernel_iters=KERNEL_ITERS,
                ),
            )
            assert value is not None, f"profiler produced no data for {variant} shard={shard_ht}x{shard_wt}"
            results[(shard_ht, shard_wt, variant)] = value
        ttnn.deallocate(tt_input)

    arch = os.environ.get("ARCH_NAME", "unknown")
    base = VARIANT_SWEEP[0]
    lines = [
        "",
        "=== handshake_elision device perf (resident-L1 same-spec tilize, no NoC traffic) ===",
        f"    cores={NUM_CORES} (row 0, one shard each)  dtype=bfloat16  arch={arch}  iters={KERNEL_ITERS}  trials={N_PROFILE_ITERS}",
        "    no_handshake = the compute kernel's compile-time constant: 0 = per-tile-row CB protocol with a",
        "    reader that publishes and a writer that retires, 1 = no CB protocol at all (tile-index addressing).",
        f"    ratio = {base} ns / this arm's ns  (>1 means this arm is faster)",
        "",
        f"    {'shard':>7}  {'tiles/core':>10}  {'variant':<12}  {'no_handshake':>12}  {'kernels':>7}  {'ns/op':>9}  {'ns/tile':>8}  {'ratio':>6}",
    ]
    for shard_ht, shard_wt in SHARD_SWEEP:
        tiles = shard_ht * shard_wt
        ref = results[(shard_ht, shard_wt, base)]
        for variant in VARIANT_SWEEP:
            v = results[(shard_ht, shard_wt, variant)]
            lines.append(
                f"    {f'{shard_ht}x{shard_wt}':>7}  {tiles:>10}  {variant:<12}  {NO_HANDSHAKE[variant]:>12}  {KERNELS[variant]:>7}  "
                f"{v:>9.1f}  {v / (tiles * KERNEL_ITERS):>8.1f}  {ref / v:>5.2f}x"
            )
        lines.append("")
    logger.info("\n".join(lines))
