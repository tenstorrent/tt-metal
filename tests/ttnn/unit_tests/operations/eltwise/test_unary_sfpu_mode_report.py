# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ttnn.SfpuMode report: what each of the four modes costs and buys, per op and dtype.

For every mode-capable op (test_unary_sfpu_mode.OPS) and every SfpuMode this measures

- accuracy on a dense in-domain stimulus (every bf16 value in the domain; 32768 log/linear
  samples for float32): max and mean ULP against torch in float64, rounded to the output dtype;
- the Strict special-value contract (test_unary_sfpu_mode.STRICT_SPECIALS, float32 only):
  probes answered correctly out of the table size;
- device time on an L1 height-sharded tensor (TILES_PER_CORE tiles on every core, so the SFPU,
  not DRAM, sets the pace): DEVICE KERNEL and TRISC1 (math) duration per tile, median of
  PERF_ITERATIONS runs.

The table is logged at the end of the module, and written to $SFPU_MODE_REPORT when that is set.
The perf columns need the device profiler, enabled before the device opens:

    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 \\
    TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES=1 SFPU_MODE_REPORT=sfpu_mode_report.md \\
    pytest tests/ttnn/unit_tests/operations/eltwise/test_unary_sfpu_mode_report.py

Without it they read n/a and only accuracy is measured.
"""

import os
import statistics

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.unit_tests.operations.eltwise.test_unary_sfpu_mode import (
    ALL_MODES,
    OPS,
    STRICT_SPECIALS,
    TORCH_DTYPE,
    _bits,
    _tile_of,
    _to_device,
)

# op -> (scale, lo, hi). "log": sign * 2**u with u uniform in [log2 lo, log2 hi]; "linear": x uniform in [lo, hi].
# Inside every mode's domain (Relaxed included), wide enough to reach many binades of the result.
DOMAINS = {
    "softsign": ("log", 2.0**-8, 2.0**8),
    "exp2": ("linear", -60.0, 60.0),
    "reciprocal": ("log", 2.0**-60, 2.0**60),
}
FP32_SAMPLES = 32768
TILES_PER_CORE = 64
PERF_ITERATIONS = 10
KERNEL_NS = "DEVICE KERNEL DURATION [ns]"
MATH_NS = "DEVICE TRISC1 KERNEL DURATION [ns]"

_ROWS = []


def _profiler_enabled():
    return os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"


def _in_domain(op, x):
    scale, lo, hi = DOMAINS[op]
    if scale == "log":
        return (x.abs() >= lo) & (x.abs() <= hi)
    return (x >= lo) & (x <= hi)


def _domain_input(op, dtype):
    """Every bf16 value in the op's domain, or FP32_SAMPLES seeded float32 samples, padded to whole tiles by
    repeating the stimulus (never with zeros: they would be lanes outside the domain)."""
    if dtype == ttnn.bfloat16:
        every = torch.arange(-(2**15), 2**15, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        x = every[torch.isfinite(every) & _in_domain(op, every.float())]
    else:
        generator = torch.Generator().manual_seed(0)
        u = torch.rand(FP32_SAMPLES, generator=generator, dtype=torch.float64)
        scale, lo, hi = DOMAINS[op]
        if scale == "log":
            log_lo, log_hi = torch.log2(torch.tensor([lo, hi], dtype=torch.float64))
            sign = torch.where(torch.rand(FP32_SAMPLES, generator=generator) < 0.5, -1.0, 1.0).double()
            x = sign * torch.exp2(log_lo + u * (log_hi - log_lo))
        else:
            x = lo + u * (hi - lo)
        x = x.float()
        x = x[_in_domain(op, x)]
    width = 1024
    n = x.numel()
    total = -(-n // (32 * width)) * 32 * width
    return x.repeat(-(-total // n))[:total].reshape(1, 1, -1, width), n


def _ordinal(t):
    """Monotone integer image of a float tensor's bit patterns: adjacent floats differ by 1, and +-0 both map to 0."""
    bits = _bits(t).to(torch.int64)
    magnitude = bits & (0x7FFFFFFF if t.dtype == torch.float32 else 0x7FFF)
    return torch.where(bits < 0, -magnitude, magnitude)


def _ulp_stats(result, x, op, dtype):
    _, torch_fn, _, _ = OPS[op]
    golden = torch_fn(x.double()).to(TORCH_DTYPE[dtype])
    finite = torch.isfinite(golden)
    assert torch.isfinite(result[finite]).all(), f"{op}: non-finite result where torch is finite"
    ulp = (_ordinal(result[finite]) - _ordinal(golden[finite])).abs().double()
    return ulp.max().item(), ulp.mean().item()


def _specials_passed(device, op, sfpu_mode):
    """Strict special-value probes (float32) this mode gets right, out of the table size."""
    ttnn_fn, _, _, _ = OPS[op]
    table = STRICT_SPECIALS[op]
    x = _tile_of([v for v, _ in table], ttnn.float32)
    expected = _tile_of([e for _, e in table], ttnn.float32).flatten()[: len(table)]
    got = ttnn.to_torch(ttnn_fn(_to_device(x, ttnn.float32, device), sfpu_mode=sfpu_mode)).flatten()[: len(table)]
    ok = torch.where(torch.isnan(expected), torch.isnan(got), _bits(got) == _bits(expected))
    return int(ok.sum().item()), len(table)


def _sharded_input(device, op, dtype):
    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    x, n = _domain_input(op, dtype)
    tiles = num_cores * TILES_PER_CORE
    x = x.flatten()[:n].repeat(tiles * 1024 // n + 1)[: tiles * 1024].reshape(1, 1, tiles * 32, 32)
    memory_config = ttnn.create_sharded_memory_config(
        x.shape,
        core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x),
        strategy=ttnn.ShardStrategy.HEIGHT,
    )
    tt_x = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    return tt_x, memory_config


def _ns_per_tile(device, run):
    """Median DEVICE KERNEL and TRISC1 duration per tile over PERF_ITERATIONS runs of *run*."""
    run()  # compile and populate the program cache outside the measured window
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    for _ in range(PERF_ITERATIONS):
        run()
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    programs = [program for programs in ttnn.get_latest_programs_perf_data().values() for program in programs]
    assert len(programs) == PERF_ITERATIONS, f"expected {PERF_ITERATIONS} profiled programs, got {len(programs)}"

    def median(name):
        return statistics.median(p.program_analyses_results[name].duration for p in programs) / TILES_PER_CORE

    return median(KERNEL_NS), median(MATH_NS)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bfloat16", "float32"])
@pytest.mark.parametrize("op", list(OPS))
def test_sfpu_mode_report(device, op, dtype):
    """Measures every SfpuMode of *op* on *dtype*; the Strict modes must pass every special-value probe."""
    ttnn_fn, _, _, legacy_mode = OPS[op]
    x, n = _domain_input(op, dtype)
    tt_x = _to_device(x, dtype, device)
    if _profiler_enabled():
        tt_sharded, sharded_config = _sharded_input(device, op, dtype)

    rows = []
    for sfpu_mode in ALL_MODES:
        result = ttnn.to_torch(ttnn_fn(tt_x, sfpu_mode=sfpu_mode)).flatten()[:n]
        max_ulp, mean_ulp = _ulp_stats(result, x.flatten()[:n], op, dtype)
        specials = _specials_passed(device, op, sfpu_mode) if dtype == ttnn.float32 else None
        if specials is not None and sfpu_mode in (ttnn.SfpuMode.PreciseStrict, ttnn.SfpuMode.ApproximateStrict):
            assert specials[0] == specials[1], f"{op} {sfpu_mode}: {specials[0]}/{specials[1]} special probes"
        kernel_ns = math_ns = None
        if _profiler_enabled():
            kernel_ns, math_ns = _ns_per_tile(
                device, lambda: ttnn_fn(tt_sharded, sfpu_mode=sfpu_mode, memory_config=sharded_config)
            )
        rows.append(
            {
                "op": op,
                "dtype": "bf16" if dtype == ttnn.bfloat16 else "fp32",
                "mode": sfpu_mode.name + (" (L)" if sfpu_mode == legacy_mode else ""),
                "kernel_ns": kernel_ns,
                "math_ns": math_ns,
                "max_ulp": max_ulp,
                "mean_ulp": mean_ulp,
                "specials": specials,
                "inputs": n,
            }
        )
    base = rows[0]["kernel_ns"]
    for row in rows:
        row["speedup"] = base / row["kernel_ns"] if row["kernel_ns"] else None
    _ROWS.extend(rows)


def _markdown(rows):
    def num(value, fmt):
        return "n/a" if value is None else format(value, fmt)

    lines = [
        "| op | dtype | mode | kernel ns/tile | math ns/tile | speedup | max ULP | mean ULP | strict specials |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        specials = "—" if r["specials"] is None else f"{r['specials'][0]}/{r['specials'][1]}"
        lines.append(
            f"| {r['op']} | {r['dtype']} | {r['mode']} | {num(r['kernel_ns'], '.1f')} | {num(r['math_ns'], '.1f')} "
            f"| {num(r['speedup'], '.2f')}x | {r['max_ulp']:.0f} | {r['mean_ulp']:.3f} | {specials} |"
        )
    return "\n".join(lines)


@pytest.fixture(scope="module", autouse=True)
def _report():
    yield
    if not _ROWS:
        return
    table = _markdown(_ROWS)
    logger.info("SfpuMode report ((L) = the mode the mode-less call runs; speedup vs PreciseStrict):\n" + table)
    path = os.environ.get("SFPU_MODE_REPORT")
    if path:
        with open(path, "w") as f:
            f.write(table + "\n")
