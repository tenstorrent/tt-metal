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
from tests.ttnn.unit_tests.operations.examples.report_gate import report_target

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
# None unless the caller opted in (--write-reports / EXAMPLES_WRITE_REPORTS=1 / MCT_REPORT=<path>).
REPORT_PATH = report_target(
    "MCT_REPORT",
    Path(__file__).resolve().parents[5] / "ttnn/ttnn/operations/examples/mcast_topology/report.md",
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

    if REPORT_PATH:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REPORT_PATH, "a") as f:
            f.write(f"\n## {arch} — {box}\n\n```\n{header}\n" + "\n".join(lines) + "\n```\n")
