# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared measurement and assertion helpers for the examples' perf tests.

One place for the policies every example's helps / does-not-help ("regime") test and every
measurement sweep follows, so "median of N interleaved trials" means the same thing in each
of them:

  * `interleaved_ns`   - interleaved A/B(/C...) measurement of device kernel duration.
  * `check_regime`     - asserts a speedup against a bound derived from the number measured on
                         THIS architecture, with a noise margin and one confirmation re-measure.
  * `require_cores` / `require_grid` - skip (never fail) on a compute grid that is too small.
  * `sweep_gate`       - opt-in marker for measurement sweeps that only log or write a report.

Import as:

    from tests.ttnn.unit_tests.operations.examples.perf_test_utils import check_regime, interleaved_ns
"""

import os
import socket
import statistics

import pytest
from loguru import logger

import ttnn

DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# Relative distance a bound keeps from the number it was derived from. A median of ~10 interleaved
# trials moved by up to ~7% between identical runs on the DRAM-bound cells (three regimes with a
# 2-5% margin failed once and passed on re-run), so 5% is not enough.
NOISE_MARGIN = 0.10
# A "helps" regime whose bound would fall below this is too small an effect to assert: it is logged.
MIN_HELPS_BOUND = 1.02

SWEEPS_ENV = "EXAMPLES_SWEEPS"

KINDS = ("helps", "does_not_help", "measure")


# ---------------------------------------------------------------------------------------------
# Device identity
# ---------------------------------------------------------------------------------------------


def arch_key(device):
    """Architecture as the key used in `measured={...}`: "wh" (Wormhole B0), "bh" (Blackhole), or
    the lower-cased arch name for anything else."""
    name = str(device.arch()).rsplit(".", 1)[-1].upper()
    return {"WORMHOLE_B0": "wh", "BLACKHOLE": "bh"}.get(name, name.lower())


def arch_label(device):
    """Architecture as printed in logs and reports: "WH_B0", "BH", or the arch name."""
    name = str(device.arch()).rsplit(".", 1)[-1].upper()
    return {"WORMHOLE_B0": "WH_B0", "BLACKHOLE": "BH"}.get(name, name)


def chip_freq_mhz():
    """AICLK in MHz that the device profiler stamped in the header of its device log, or None when
    there is no log yet. Read it AFTER the first profiler read of the session: the profiler
    (re)creates the log then."""
    home = os.environ.get("TT_METAL_HOME", os.getcwd())
    path = os.path.join(home, "generated", "profiler", ".logs", "profile_log_device.csv")
    try:
        with open(path) as f:
            header = f.readline()
    except OSError:
        return None
    for part in header.split(","):
        if "CHIP_FREQ" in part:
            try:
                return float(part.split(":")[1])
            except (IndexError, ValueError):
                return None
    return None


def grid_xy(device):
    """(x, y) of the compute-with-storage grid: Tensix-core columns, Tensix-core rows."""
    grid = device.compute_with_storage_grid_size()
    return grid.x, grid.y


def grid_cores(device):
    """Number of Tensix cores in the compute-with-storage grid."""
    x, y = grid_xy(device)
    return x * y


def stamp(device):
    """One-line provenance for a measured number: box, arch, grid, clock."""
    x, y = grid_xy(device)
    return (
        f"box={socket.gethostname()} arch={arch_label(device)} grid={x}x{y}={x * y} Tensix cores "
        f"CHIP_FREQ={chip_freq_mhz()} MHz"
    )


def is_simulator():
    """True when the run targets the TT simulator: device kernel durations mean nothing there."""
    return bool(os.environ.get("TT_METAL_SIMULATOR"))


# ---------------------------------------------------------------------------------------------
# Grid guards: a case that needs more Tensix cores than the board has is skipped, never failed
# and never silently bent to fit (Wormhole n150 / n300 have at most 8x8, Blackhole p150b 13x10).
# ---------------------------------------------------------------------------------------------


def require_cores(device, num_cores, what="this case"):
    """pytest.skip unless the compute grid has at least `num_cores` Tensix cores."""
    have = grid_cores(device)
    if have < num_cores:
        x, y = grid_xy(device)
        pytest.skip(f"{what} needs {num_cores} Tensix cores; this grid is {x}x{y} = {have}")


def require_grid(device, min_x=1, min_y=1, what="this case"):
    """pytest.skip unless the compute grid has at least `min_x` columns and `min_y` rows."""
    x, y = grid_xy(device)
    if x < min_x or y < min_y:
        pytest.skip(f"{what} needs a grid of at least {min_x}x{min_y} Tensix cores (x by y); this grid is {x}x{y}")


# ---------------------------------------------------------------------------------------------
# Opt-in gate for measurement sweeps
# ---------------------------------------------------------------------------------------------


def sweeps_enabled(*env_names):
    """True when EXAMPLES_SWEEPS=1 or any of the example's own switches in `env_names` is "1"."""
    return any(os.environ.get(name) == "1" for name in (SWEEPS_ENV,) + tuple(env_names))


def sweep_gate(*env_names):
    """Marker for a test that only measures (logs a table / writes a report) and asserts no
    behaviour a correctness test does not already cover. Such a test is skipped by default, so a
    plain run of the directory (the CI sanity group runs it with `-x` under a time budget) costs
    correctness tests plus the regimes only. It runs when EXAMPLES_SWEEPS=1, or when one of the
    example's own switches (`env_names`, set by its CLI) is "1"."""
    names = (SWEEPS_ENV,) + tuple(env_names)
    return pytest.mark.skipif(
        not sweeps_enabled(*env_names),
        reason=f"measurement sweep: set {' or '.join(name + '=1' for name in names)} to run it",
    )


# ---------------------------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------------------------


def read_kernel_ns(device):
    """Read the device profiler and return the summed DEVICE KERNEL DURATION [ns] of every program
    launched since the previous read, or None when the profiler produced no data."""
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data() or {}
    total, found = 0.0, False
    for programs in per_chip.values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(DURATION_KEY)
            if entry is None:
                continue
            total += float(entry.duration)
            found = True
    return total if found else None


def interleaved_ns(device, runners, *, trials, launches=1, warmup=2, per_launch_divisor=1):
    """Measure every runner in `runners` ({name: zero-arg callable that launches the op once})
    interleaved, and return {name: [ns, ...]} with `trials` samples each.

    Policy (identical for every example):
      * warm-up: each runner is launched `warmup` times (compiles and caches its program), the
        device is synchronized and that profiler window is read and discarded;
      * a trial of one runner = `launches` launches read as ONE profiler window, reported as the
        mean device kernel duration per launch divided by `per_launch_divisor` (pass the
        in-kernel repeat count there to get ns per repeat);
      * rounds are interleaved (A, B, A, B, ...) so clock or thermal drift hits every runner, and
        the starting runner rotates per round so none of them always runs first.
    """
    names = list(runners)
    assert names, "interleaved_ns needs at least one runner"
    for _ in range(max(1, warmup)):
        for name in names:
            runners[name]()
    ttnn.synchronize_device(device)
    read_kernel_ns(device)  # discard the warm-up window

    samples = {name: [] for name in names}
    for trial in range(trials):
        start = trial % len(names)
        for name in names[start:] + names[:start]:
            for _ in range(launches):
                runners[name]()
            total = read_kernel_ns(device)
            assert total is not None, f"device profiler produced no duration for {name} (profiler-enabled build?)"
            samples[name].append(total / launches / per_launch_divisor)
    return samples


def spread(values):
    """'median [min-max]' of a sample list, for log lines."""
    return f"{statistics.median(values):.1f} [{min(values):.1f}-{max(values):.1f}]"


# ---------------------------------------------------------------------------------------------
# Regime assertion
# ---------------------------------------------------------------------------------------------


def _as_range(value):
    if isinstance(value, (tuple, list)):
        lo, hi = value
        return float(lo), float(hi)
    return float(value), float(value)


def regime_bound(kind, measured, arch):
    """Bound a regime is asserted against on `arch`, derived from the speedup measured there.

    `measured` maps an arch key ("wh", "bh") to the speedup (baseline ns / lever ns) measured on
    that architecture: a number, or a (low, high) pair when repeated runs gave a range.

    Returns (bound, note). `bound` is None when the regime is logged but not asserted; `note`
    says why.

      helps          speedup >= min(1 + 0.5 * (low - 1), low / (1 + NOISE_MARGIN)):
                     half the measured gain, and never closer than NOISE_MARGIN to the
                     measurement. Not asserted when that falls below MIN_HELPS_BOUND (a measured
                     gain under ~1.12x cannot be told from flat at this noise level).
      does_not_help  speedup <= max(1.0, high * (1 + NOISE_MARGIN)): a measured loss must stay a
                     non-gain, a measured null may wander up by the noise margin.
      measure        never asserted (the verdict differs between boards, or sits on the edge).
    """
    if kind not in KINDS:
        raise ValueError(f"unknown regime kind {kind!r}; expected one of {KINDS}")
    if kind == "measure":
        return None, "measure-only regime"
    if not measured or arch not in measured:
        known = ", ".join(sorted(measured)) if measured else "none"
        return None, f"no measured reference for arch '{arch}' (measured on: {known})"
    low, high = _as_range(measured[arch])
    if kind == "helps":
        bound = min(1.0 + 0.5 * (low - 1.0), low / (1.0 + NOISE_MARGIN))
        if bound < MIN_HELPS_BOUND:
            return None, f"measured gain {low:.2f}x is within the {NOISE_MARGIN:.0%} noise margin"
        return bound, f"measured {low:.2f}x on {arch}"
    return max(1.0, high * (1.0 + NOISE_MARGIN)), f"measured {high:.2f}x on {arch}"


def _holds(kind, speedup, bound):
    return speedup >= bound if kind == "helps" else speedup <= bound


def check_regime(device, name, kind, measured, measure, *, detail=""):
    """Measure one regime and assert its direction on the architectures it was measured on.

    name      regime id, used in the log line and the failure message.
    kind      "helps" | "does_not_help" | "measure".
    measured  {arch key: speedup measured on that arch}, e.g. {"wh": 6.19, "bh": (7.1, 7.3)}.
              An architecture missing from it is logged, not asserted - a number measured on one
              board says nothing certain about another grid or clock.
    measure   zero-arg callable returning (baseline_samples, lever_samples), two lists of ns from
              one interleaved measurement (see `interleaved_ns`). It is called once; if the bound
              is missed it is called ONCE more and the bound is re-checked on the pooled samples,
              so a single noisy window does not fail the test.
    detail    free text appended to the log line (shape, Tensix-core counts, ...).

    Correctness of both variants is the caller's job and comes BEFORE this call.
    Returns the speedup (baseline median / lever median) of the samples the verdict was taken on.
    """
    arch = arch_key(device)
    bound, note = regime_bound(kind, measured, arch)

    base, lever = (list(samples) for samples in measure())
    speedup = statistics.median(base) / statistics.median(lever)
    confirmed = ""
    asserted = bound is not None and not is_simulator()
    if asserted and not _holds(kind, speedup, bound):
        first = speedup
        more_base, more_lever = measure()
        base += list(more_base)
        lever += list(more_lever)
        speedup = statistics.median(base) / statistics.median(lever)
        confirmed = f" | first pass {first:.3f}x missed the bound, pooled with a confirmation pass"

    if bound is None:
        verdict = f"logged, not asserted ({note})"
    elif not asserted:
        verdict = "logged, not asserted (simulator)"
    else:
        verdict = f"asserted {'>=' if kind == 'helps' else '<='} {bound:.3f}x ({note})"
    logger.info(
        f"REGIME {name} [{kind}] {detail + ' | ' if detail else ''}"
        f"baseline {spread(base)} ns | lever {spread(lever)} ns | speedup {speedup:.3f}x | "
        f"{len(base)} trials per variant | {verdict}{confirmed} | {stamp(device)}"
    )
    if asserted:
        op = ">=" if kind == "helps" else "<="
        assert _holds(kind, speedup, bound), (
            f"{name}: expected speedup {op} {bound:.3f}x ({note}), got {speedup:.3f}x after a confirmation pass "
            f"[baseline {spread(base)} ns, lever {spread(lever)} ns, {stamp(device)}]"
        )
    return speedup
