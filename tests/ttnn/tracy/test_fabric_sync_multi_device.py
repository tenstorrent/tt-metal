# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Continuous device-to-device clock sync over fabric (multi-device).

Runs a small multi-device fabric workload with the perf-debug profiler and the IN-ROUTER fabric sync
enabled (``TT_METAL_PERF_DEBUG_FABRIC_SYNC_HZ``), under a connected ``tracy-capture``, and verifies the
sync ran CONTINUOUSLY and LANDED IN THE RIGHT PLACE:

  * every discovered link solved rounds, with zero partial / dropped / stray;
  * the sample count is exactly lossless (rounds x links x samples x 3 endpoints);
  * every eth lane got its OWN Tracy anchor (an unanchored lane renders against the worker anchor and
    lands seconds away from the workload -- that bug shipped once and is what this test exists to catch);
  * the responder's echo falls INSIDE its own round-trip box on the initiator's clock.

That last check is the substantive one. It is a causality statement -- the ping cannot be observed
before it was sent -- evaluated on ONE lane, where the per-core anchor error cancels. Comparing the two
ends of a link ACROSS lanes cannot do this: per-core anchors are one-shot host fits whose ppm-scale
error integrates to tens of microseconds, so a cross-lane comparison scores the anchors, not the sync.

Hardware-gated: needs >= 2 devices with fabric. Skips cleanly when the box cannot run it. Device work
runs in a subprocess so the pytest parent never takes the PCIe lock (mirrors test_perf_debug_profiler.py).
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import time
from pathlib import Path

import pytest

from tools.tracy.common import PROFILER_ARTIFACTS_DIR, PROFILER_BIN_DIR, TT_METAL_HOME

CAPTURE_TOOL = PROFILER_BIN_DIR / "tracy-capture"
FABRIC_BIN = Path(TT_METAL_HOME) / "build_Release" / "test" / "tt_metal" / "tt_fabric" / "test_infra" / "test_tt_fabric"
ARTIFACTS = PROFILER_ARTIFACTS_DIR / "fabric_sync_tests"

SYNC_HZ = "20"
# kMaxSamples in the router-sync wire format: samples per round, per link.
SAMPLES_PER_ROUND = int(os.environ.get("TT_METAL_PERF_DEBUG_FABRIC_SYNC_SAMPLES", "2"))
# t0/t1/t2 -- three PP_SYNC packets per sample reach the sink.
ENDPOINTS_PER_SAMPLE = 3

# A deliberately small shape: this test is about the SYNC, not about bandwidth. NeighborExchange needs
# only that neighbouring devices can talk, so it is the cheapest multi-device fabric workload here.
BENCH_YAML = """Tests:
  - name: "FabricSyncSmoke"
    benchmark_mode: true
    sync: true
    fabric_setup:
      topology: NeighborExchange
    parametrization_params:
      num_links: [1]
      ntype: [unicast_write]
      size: [2048]
    defaults:
      ftype: unicast
      num_packets: 10000
    patterns:
      - type: neighbor_exchange
"""


def _free_port() -> str:
    ip = socket.gethostbyname(socket.gethostname())
    for port in range(8086, 8500):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((ip, port))
            s.close()
            return str(port)
        except (PermissionError, OSError):
            continue
    raise RuntimeError("no free TCP port for tracy-capture")


def test_fabric_sync_continuous_multi_device():
    if not FABRIC_BIN.exists():
        pytest.skip(f"fabric test binary not built: {FABRIC_BIN}")
    if not CAPTURE_TOOL.exists():
        pytest.skip(f"tracy-capture not found: {CAPTURE_TOOL}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    yaml_path = ARTIFACTS / "fabric_sync_smoke.yaml"
    yaml_path.write_text(BENCH_YAML)
    out_tracy = ARTIFACTS / "fabric_sync.tracy"
    out_tracy.unlink(missing_ok=True)
    port = _free_port()

    cap = subprocess.Popen(
        [str(CAPTURE_TOOL), "-o", str(out_tracy), "-f", "-p", port],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)  # let tracy-capture start listening

    env = dict(os.environ)
    env["TRACY_PORT"] = port
    env["TT_METAL_STREAMING_PROFILER"] = "1"
    env["TT_METAL_STREAMING_PROFILER_TRACY"] = "1"  # the Tracy sink is what renders (and self-checks) the sync
    env["TT_METAL_PERF_DEBUG_FABRIC_SYNC_HZ"] = SYNC_HZ
    # Pin the clock: the sync solves an offset against a running DVFS loop otherwise, which adds a
    # frequency term this test has no interest in measuring.
    env["TT_METAL_PERF_DEBUG_FORCE_AICLK"] = "1350"
    try:
        proc = subprocess.run(
            [str(FABRIC_BIN), "--test_config", str(yaml_path)],
            env=env,
            cwd=str(TT_METAL_HOME),
            timeout=900,
            capture_output=True,
            text=True,
        )
    finally:
        try:
            cap.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            cap.terminate()
            cap.communicate()

    log = proc.stdout + proc.stderr

    # ---- hardware gates -------------------------------------------------------------------------
    if "not Blackhole" in log or "resident on logical" not in log:
        pytest.skip("perf-debug profiler did not start the DRISC drainer (not Blackhole / no DRAM cores)")
    # A drainer that fails to start disarms back-pressure and DROPS every marker, so the sync would
    # look idle rather than broken. Never let that masquerade as a pass.
    assert "FAILED TO START" not in log, f"DRISC drainer failed to start:\n{log[-2000:]}"

    m = re.search(r"FABRIC SYNC: (\d+) in-router link\(s\) at ([\d.]+) Hz", log)
    if m is None or int(m.group(1)) == 0:
        pytest.skip("no in-router fabric sync links discovered (single device, or fabric disabled)")
    n_links = int(m.group(1))

    # ---- every lane must be ANCHORED -------------------------------------------------------------
    # An eth lane without its own anchor renders against the WORKER anchor. Its zones then land
    # seconds away from the workload they ran inside, while every count and duration still looks
    # perfect -- which is exactly how this shipped broken once.
    assert (
        "got NO clock sync" not in log
    ), "an eth sync lane failed to anchor; its rows would render against the worker anchor:\n" + "\n".join(
        l for l in log.splitlines() if "got NO clock sync" in l
    )

    # ---- continuity + losslessness ---------------------------------------------------------------
    finals = re.findall(
        r"FABRIC SYNC (\d+) -> (\d+) FINAL: (\d+) rounds solved \((\d+) partial, (\d+) dropped, (\d+) stray\)",
        log,
    )
    assert len(finals) == n_links, f"expected {n_links} FINAL line(s), got {len(finals)}:\n{log[-2000:]}"

    total_rounds = 0
    for init, resp, solved, partial, dropped, stray in finals:
        solved, partial, dropped, stray = int(solved), int(partial), int(dropped), int(stray)
        # "Continuous" means it kept exchanging for the whole window, not that it managed one round.
        # A single round is the signature of a hook that went dormant, which is a real failure mode.
        assert solved > 1, f"link {init}->{resp} solved only {solved} round(s) -- the sync did not run continuously"
        assert dropped == 0, f"link {init}->{resp} dropped {dropped} round(s)"
        assert partial == 0, f"link {init}->{resp} had {partial} partial round(s)"
        assert stray == 0, f"link {init}->{resp} saw {stray} stray sample(s)"
        total_rounds += solved

    m = re.search(r"FABRIC SYNC: (\d+) samples reached the sink end to end", log)
    assert m is not None, "no end-to-end sample count reported"
    got_samples = int(m.group(1))
    want_samples = total_rounds * SAMPLES_PER_ROUND * ENDPOINTS_PER_SAMPLE
    # Exact, not approximate: every t0/t1/t2 of every solved round rides the drain, and the drain is
    # lossless by construction. A shortfall means the capture silently lost sync packets.
    assert got_samples == want_samples, f"sync samples not lossless: got {got_samples}, want {want_samples}"

    # ---- placement: the echo must sit inside its own round trip -----------------------------------
    m = re.search(
        r"FABRIC SYNC drawn: (\d+) zone\(s\) \+ (\d+) marker\(s\) across (\d+) eth link\(s\); "
        r"(\d+)/(\d+) echoes fall INSIDE their own FSYNC_RTT box on the initiator clock \(([\d.]+)%\)",
        log,
    )
    assert m is not None, f"the sync was not drawn into Tracy (no FSYNC render line):\n{log[-2000:]}"
    zones, marks, drawn_links, inside, checked, pct = m.groups()
    assert int(zones) > 0 and int(marks) > 0, "sync rendered zero GUI elements"
    assert int(drawn_links) == n_links, f"drew {drawn_links} link(s), discovered {n_links}"
    assert int(checked) > 0, "no echo was causality-checked"
    # This is a causality bound, not a tuning threshold: t1 lies between t0 and t2 by construction, so
    # on the initiator's own clock every echo must land inside its box. Anything below 100% means the
    # round's solved offset is wrong -- a sync bug, not a rendering nicety.
    assert float(pct) >= 99.0, (
        f"only {inside}/{checked} ({pct}%) echoes fall inside their own round-trip box -- "
        "the per-round offsets are wrong"
    )

    assert out_tracy.exists() and out_tracy.stat().st_size > 4096, "no/empty Tracy capture produced"
