# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `mcast_topology` example — work-split shape vs multicast shape.

Operand DELIVERY only; there is no compute kernel. Each core fetches the operand blocks a tiled
`C[M,N] = A[M,K] @ B[K,N]` would need for its share of the output, and writes a few probe tiles so
delivery can be proven without computing anything.

Both variants use the SAME fixed 2-D work split (grid rows carry M, grid columns carry N), so every
core holds the same slices either way. Only the transport differs:
  per_core_dram — every core reads its own A row-slice and B column-slice from DRAM. Redundant by
                  construction: Gc cores read the same A slice, Gr cores read the same B slice.
  mcast_1d_pair — each slice is read once per line and broadcast: Mcast1D(PerRow) for A,
                  Mcast1D(PerColumn) for B.

See ttnn/ttnn/operations/examples/mcast_topology/README.md.

    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_mcast_topology.py::test_mcast_topology_delivery
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_mcast_topology.py::test_mcast_topology_device_perf
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import socket
import statistics
from pathlib import Path

import pytest
import torch

import ttnn
from ttnn.operations.examples.mcast_topology import mcast_topology, VARIANTS
from ttnn.operations.examples.mcast_topology.mcast_topology import PROBES, core_assignment, layout

from loguru import logger

TILE = 32

# Short M, long N — the grid-starved regime. Mt=8 tile-rows can occupy at most 8 cores on the
# M axis alone, while Nt=32 offers 32 more units of independent parallelism.
MT = int(os.environ.get("MCT_MT", "8"))  # output tile-rows  (M = 256)
NT = int(os.environ.get("MCT_NT", "32"))  # output tile-cols (N = 1024)
KT = int(os.environ.get("MCT_KT", "4"))  # contraction tiles (K = 128)

_sel = os.environ.get("MCT_VARIANTS", "all")
SELECTED = VARIANTS if _sel == "all" else tuple(v for v in VARIANTS if v in _sel.split(","))

N_WARMUP = 3
N_PROFILE_ITERS = int(os.environ.get("MCT_TRIALS", "5"))
_INNER = 5
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
REPORT_PATH = os.environ.get(
    "MCT_REPORT",
    str(Path(__file__).resolve().parents[5] / "ttnn/ttnn/operations/examples/mcast_topology/report.md"),
)


def _operands(device, mt=MT, nt=NT, kt=KT):
    """Distinct value per tile, so a misdelivered block is unmistakable rather than plausible."""
    a = torch.arange(mt * kt, dtype=torch.float32).reshape(mt, kt)
    a = a.repeat_interleave(TILE, 0).repeat_interleave(TILE, 1)
    b = torch.arange(kt * nt, dtype=torch.float32).reshape(kt, nt) + 1000.0
    b = b.repeat_interleave(TILE, 0).repeat_interleave(TILE, 1)
    to = lambda t: ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return to(a), to(b), a, b


def _tile(t, ti, tj):
    return t[ti * TILE : (ti + 1) * TILE, tj * TILE : (tj + 1) * TILE]


def _expected_probes(device, a, b, variant, mt=MT, nt=NT, kt=KT):
    """The exact tiles each core must have received, in probe-slot order."""
    assign = core_assignment(device, mt, nt)
    out = {}
    for m0, n0, mloc, nloc, slot in assign.values():
        out[slot] = [
            _tile(a, m0, 0),  # A[m0, 0]
            _tile(b, 0, n0),  # B[0, n0]
            _tile(b, kt - 1, n0 + nloc - 1),  # B[Kt-1, n0+Nloc-1]
        ]
    return out


def _read_kernel_ns(device):
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
    for _ in range(N_WARMUP):
        run_fn()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)
    samples = []
    for _ in range(N_PROFILE_ITERS):
        for _ in range(_INNER):
            run_fn()
        total = _read_kernel_ns(device)
        if total is None:
            return None, None
        samples.append(total / _INNER)
    med = statistics.median(samples)
    return med, (statistics.pstdev(samples) / med * 100.0 if med else float("nan"))


@pytest.mark.parametrize("variant", VARIANTS)
def test_mcast_topology_delivery(device, variant):
    """The only pass/fail: every core received exactly the operand block its output share needs.

    Checked tile-exactly (not by PCC) — delivery is a routing question, so a probe either is the
    right tile or it is not.
    """
    tt_a, tt_b, a, b = _operands(device)
    got = ttnn.to_torch(mcast_topology(tt_a, tt_b, variant=variant)).to(torch.float32)
    expected = _expected_probes(device, a, b, variant)

    assert got.shape[0] == len(expected) * PROBES * TILE, f"probe count mismatch: {got.shape}"
    for slot, tiles in expected.items():
        for p, want in enumerate(tiles):
            have = got[(slot * PROBES + p) * TILE : (slot * PROBES + p + 1) * TILE, :]
            assert torch.equal(have, want.to(torch.bfloat16).to(torch.float32)), (
                f"{variant}: core slot {slot} probe {p} got {have[0, 0].item()} want {want[0, 0].item()} "
                f"— the multicast landed the wrong block on this core"
            )


def test_mcast_topology_device_perf(device):
    """Device kernel duration: 1-D work split (2-D mcast) vs 2-D work split (two 1-D mcasts)."""
    arch = os.environ.get("ARCH_NAME", str(device.arch()))
    box = socket.gethostname()
    lay = layout(device, MT, NT)
    gx, gy = lay["grid"]
    total_cores = gx * gy

    tt_a, tt_b, a, b = _operands(device)

    rows = []
    for variant in SELECTED:
        # Delivery gate first — a "win" must mean the same operands, delivered faster.
        got = ttnn.to_torch(mcast_topology(tt_a, tt_b, variant=variant)).to(torch.float32)
        for slot, tiles in _expected_probes(device, a, b, variant).items():
            for p, want in enumerate(tiles):
                have = got[(slot * PROBES + p) * TILE : (slot * PROBES + p + 1) * TILE, :]
                assert torch.equal(have, want.to(torch.bfloat16).to(torch.float32))
        med, spread = _measure_ns(device, lambda v=variant: mcast_topology(tt_a, tt_b, variant=v))
        rows.append((variant, lay["rows"], lay["cols"], lay["cores"], med, spread))

    base = rows[0][4]
    header = (
        f"mcast_topology  box={box}  arch={arch}  grid={gx}x{gy} ({total_cores} cores)  "
        f"M={MT}t N={NT}t K={KT}t  delivery only (no compute)   "
        f"N={N_PROFILE_ITERS} (median of {_INNER}-launch windows)"
    )
    logger.info(header)
    lines = []
    for variant, r, c, cores, med, spread in rows:
        shape = "per-core DRAM reads" if variant == "per_core_dram" else "2x Mcast1D (PerRow + PerColumn)"
        rel = f"  -> {base / med:.2f}x" if (med and base) else ""
        occ = 100.0 * cores / total_cores
        line = (
            f"  {variant:<14} split={r}x{c}  cores={cores:>3}/{total_cores} ({occ:>3.0f}%)  "
            f"{shape:<32} {med:>10.0f} ns ±{spread:.1f}%{rel}"
        )
        logger.info(line)
        lines.append(line)

    Path(REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "a") as f:
        f.write(f"\n## {arch} — {box}\n\n```\n{header}\n" + "\n".join(lines) + "\n```\n")


def _measure_interleaved_ns(device, run_fns, trials=10):
    """Median/min/max ns per op per callable, windows interleaved A,B,A,B,... to limit drift."""
    for fn in run_fns.values():
        for _ in range(N_WARMUP):
            fn()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)
    samples = {k: [] for k in run_fns}
    for _ in range(trials):
        for k, fn in run_fns.items():
            for _ in range(_INNER):
                fn()
            total = _read_kernel_ns(device)
            assert total is not None, f"profiler produced no data for {k}"
            samples[k].append(total / _INNER)
    return {k: (statistics.median(v), min(v), max(v)) for k, v in samples.items()}


# (Mt, Nt, Kt, regime, threshold). bf16 tiles. Gr x Gc = largest divisors of Mt / Nt fitting the grid.
#   helps:          Mt=8, Nt=32, Kt=4 -> 8x8 = 64 Tensix cores, lines of 8 -> measured 1.89x on WH B0
#   does_not_help:  Mt=4, Nt=4,  Kt=4 -> 4x4 = 16 Tensix cores, lines of 4 -> measured 1.01x
#                   Mt=2, Nt=2,  Kt=4 -> 2x2 =  4 Tensix cores, lines of 2 -> measured 0.72x
_REGIME_CASES = [
    pytest.param(8, 32, 4, "helps", 1.45, id="helps_8x8_64cores"),
    pytest.param(4, 4, 4, "does_not_help", 1.08, id="does_not_help_4x4_16cores"),
    pytest.param(2, 2, 4, "does_not_help", 1.0, id="does_not_help_2x2_4cores"),
]


@pytest.mark.parametrize("mt,nt,kt,regime,threshold", _REGIME_CASES)
def test_mcast_topology_regimes(device, mt, nt, kt, regime, threshold):
    """When two 1-D multicasts beat per-core DRAM reads, and when they do not.

    The saving is DRAM reads removed (line length x fewer); the cost is the per-line semaphore
    handshake and the sender's serial read. With short lines (few cores per row/column) the saving is
    small and the fixed handshake cost is not amortized.
    """
    # The regime thresholds below were calibrated on Wormhole B0 (n150). The boundary between
    # "helps" and "does not help" moves with the arch (grid size, DRAM banks, NoC, clock), so on
    # another arch re-measure the regimes and add its own thresholds before enabling this test.
    if not ttnn.device.is_wormhole_b0(device):
        pytest.skip("regime thresholds are calibrated on Wormhole B0 only; re-measure on this arch first")
    lay = layout(device, mt, nt)
    tt_a, tt_b, a, b = _operands(device, mt, nt, kt)

    fns = {}
    for variant in VARIANTS:
        got = ttnn.to_torch(mcast_topology(tt_a, tt_b, variant=variant)).to(torch.float32)
        for slot, tiles in _expected_probes(device, a, b, variant, mt, nt, kt).items():
            for p, want in enumerate(tiles):
                have = got[(slot * PROBES + p) * TILE : (slot * PROBES + p + 1) * TILE, :]
                assert torch.equal(have, want.to(torch.bfloat16).to(torch.float32)), f"{variant}: wrong probe"
        fns[variant] = lambda v=variant: mcast_topology(tt_a, tt_b, variant=v)

    res = _measure_interleaved_ns(device, fns)
    base, mc = res["per_core_dram"][0], res["mcast_1d_pair"][0]
    speedup = base / mc
    logger.info(
        f"[mcast_topology regimes] {regime}: Mt={mt} Nt={nt} Kt={kt} split={lay['rows']}x{lay['cols']} "
        f"({lay['cores']} Tensix cores) | per_core_dram {base:.0f} ns "
        f"[{res['per_core_dram'][1]:.0f}-{res['per_core_dram'][2]:.0f}]  mcast_1d_pair {mc:.0f} ns "
        f"[{res['mcast_1d_pair'][1]:.0f}-{res['mcast_1d_pair'][2]:.0f}]  speedup {speedup:.2f}x  "
        f"trials=10x{_INNER}"
    )
    if regime == "helps":
        assert speedup >= threshold, f"expected speedup >= {threshold:.2f}x, got {speedup:.2f}x"
    else:
        assert speedup <= threshold, f"expected speedup <= {threshold:.2f}x (no gain), got {speedup:.2f}x"
