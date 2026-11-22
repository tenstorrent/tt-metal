# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.21 (owner): the trace ceiling of the all-device model. A standalone measurement, not the model's path and not a
gate: the profile rung's last chunk (the golden 53k prefix loaded, chunk 53248-55296, the spec's layers, shipped
settings) is run untraced as the profile runs it, then captured once with ttnn trace and replayed; both walls are
the median of warm runs of the layer stack (embedding, the golden inputs at layer gaps and the engram rows made
before, as the profile excludes the gap uploads; begin_chunk timed apart: it is the host input, not in a trace).

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \
        models/demos/deepseek_v41_flash_d_p/tt/perf_trace.py

What the capture needs here, and only here (the forward is unchanged):
  * the inputs stay at their addresses: layer 0's embedding, the golden inputs of the gap layers and the engram rows
    are made before the capture and never freed by it (the engram step takes a device clone of the rows: two small
    copies a chunk, inside the trace);
  * the same chunk start on every replay: the forward still picks per-start tables and scalars on the host (rope
    rows, cache-write index rows and device scalars, the window index class, the carry slices, the ring score's
    start / length, the top-k store's key), so a replay at another start is not this trace (printed below).
Measured P.21 (2x2, STATIC_TABLES=on, chunk 53248-55296, 8 layers): traced replay 87.0 ms a chunk (10 replays,
87.0-87.1), untraced layer stack median 103.7 ms (87.9-199.0), begin_chunk 38-40 ms warm (host); replay output vs
untraced PCC 1.000000. The replay's last-layer output is compared with the untraced output of the same chunk (the replays recompute the
chunk from the same inputs and the same prefix state)."""

from __future__ import annotations

import statistics
import time
import traceback

import pytest

import ttnn
from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference.golden import Golden
from models.demos.common.bringup.testing.harness import begin_chunk, spec
from models.demos.deepseek_v41_flash_d_p.bringup import hooks
from models.demos.deepseek_v41_flash_d_p.tt.common import DRAM

RUNG = "last"
WARM = 10  # untraced warm-up runs (program cache; address-keyed descriptor caches: the profile's WALL_WARM)
SETTLE = 2  # untraced runs in a row with no new address-keyed flat-op descriptor before the capture
RUNS = 10  # timed runs per side
TRACE_REGION = 512 * 1024 * 1024

# What a chunk-to-chunk trace still needs (code reading, tt/*.py at P.21): host values per start baked into the
# captured commands. Printed with the measurement.
BLOCKERS = [
    "rope rows: RopeTable.rows / CompressRope.rows slice the cos / sin tables at host offsets (start, start / r)",
    "cache writes: WindowTables.srv_row(k = (start mod S) / 32) picks the server-order row on the host; "
    "update_padded_kv_cache(kv_actual_global=start, valid_global=end) scalars; the r2 caches' DeviceScalars.at(start)",
    "window: TtWindowCache.carry slices local rows at host offsets of start (or the kept rows when start follows), "
    "the pad mask scalar end - start, the start class (< 128) picks index lists / compaction permutations",
    "indexer: ring_indexer_score_dsa chunk_start_idx / kv length per start (the key range grows with the prefix), "
    "the candidate pool / re-index key widths per start",
    "state: the top-k and candidate stores are dicts keyed by the chunk start (new tensors per chunk)",
    "address-keyed programs: WeightedCombine passes flat_routed_expert_with_program its full descriptor when the "
    "(x, arena, words, relays, counts, token index) addresses are new (``seen``), and a capture refuses to compile "
    "('Cannot load new binaries during trace capture'): the capture needs a run whose addresses were all seen (6 "
    "untraced runs here); a real prefill needs the arena / words / relay buffers preallocated",
    "inputs: begin_chunk uploads the engram rows into a new tensor per chunk and embed the ids (host inputs: a trace "
    "needs them copied into fixed input tensors)",
]


def _forward(model, state, layers, starts, inputs, start):
    """The profile's layer loop on device-resident inputs (never freed here); returns the last layer's output."""
    h = None
    for i in layers:
        if i in starts:
            if h is not None:
                model.free(h)
            h = inputs[i]
        h2 = model.layer(i, h, start, state)
        if h is not inputs.get(i):
            model.free(h)
        h = h2
    return h


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "device_params", [dict(hooks.device_params(spec()), trace_region_size=TRACE_REGION)], indirect=True, ids=["trace"]
)
@pytest.mark.parametrize("mesh_device", [tuple(spec().mesh)], indirect=True, ids=["x".join(map(str, spec().mesh))])
def test_trace_ceiling(mesh_device):
    s = spec()
    rung = s.rung(RUNG)
    g = Golden.for_rung(s, RUNG)
    layers = [i for i in s.layers() if i in g.layers]
    chunk = rung["chunk"]
    start = rung["seq"] - chunk
    model = s.hooks().device_model(mesh_device, s, layers, lm_head=False)
    state = model.new_state(rung["seq"])
    for i in layers:
        state.load_prefix(i, g.state(i, at=start), start)
    tokens = g.tokens()[start:]
    starts = {i for k, i in enumerate(layers) if k == 0 or layers[k - 1] != i - 1}
    want = g.layer(g.n_chunks - 1, layers[-1])["out"].float()

    # begin_chunk alone (host: hash, memory-mapped lookup, upload of the engram rows)
    bc = []
    for _ in range(3):
        t0 = time.time()
        begin_chunk(model, tokens, start)
        model.wait_engram()  # ENGRAM_ASYNC=on (P.6): the worker's jobs done before the sync
        ttnn.synchronize_device(mesh_device)
        bc.append((time.time() - t0) * 1e3)
    rows = dict(model._engram_dev)  # the chunk's engram rows, kept for every run (and the trace)
    model._engram_dev.clear()
    model.take_engram_rows = lambda layer: ttnn.clone(rows[layer], memory_config=DRAM)
    inputs = {}
    for i in starts:
        inputs[i] = model.embed(tokens) if i == 0 else model.from_host(g.layer(g.n_chunks - 1, i)["in"].float())

    def untraced():
        if state.memo is not None:
            state.memo.clear()
        return _forward(model, state, layers, starts, inputs, start)

    walls = []
    out = None
    for k in range(WARM + RUNS):
        t0 = time.time()
        out = untraced()
        ttnn.synchronize_device(mesh_device)
        if k >= WARM:
            walls.append((time.time() - t0) * 1e3)
        if k < WARM + RUNS - 1:
            model.free(out)
    ref = model.to_host(out).float()
    model.free(out)
    untraced_ms = statistics.median(walls)

    # capture after SETTLE more untraced runs (programs whose cache key holds buffer addresses must have seen the
    # capture's addresses; capture records commands only and refuses to compile)
    from models.demos.deepseek_v41_flash_d_p.tt import weighted_combine as WC

    def n_seen():
        return sum(len(wc.seen) for _, wc in WC._STATES.values())

    settle, quiet = 0, 0
    while quiet < SETTLE and settle < 40:  # until SETTLE runs in a row add no flat-op descriptor key
        before = n_seen()
        model.free(untraced())
        settle += 1
        quiet = quiet + 1 if n_seen() == before else 0
    ttnn.synchronize_device(mesh_device)
    print(f"TRACE settle: {settle} untraced runs before the capture ({n_seen()} flat-op keys seen, quiet {quiet})")
    if state.memo is not None:
        state.memo.clear()
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    try:
        traced_out = _forward(model, state, layers, starts, inputs, start)
    except Exception as e:  # what blocks capturing the forward as it is
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        print(f"TRACE capture failed: {type(e).__name__}: {str(e).splitlines()[0]}")
        print(
            "TRACE at:\n"
            + "".join(
                l
                for l in traceback.format_exc().splitlines(True)
                if "deepseek_v41" in l or "bringup/" in l or "line" in l
            )
        )
        raise
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    tw = []
    for k in range(2 + RUNS):
        t0 = time.time()
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        if k >= 2:
            tw.append((time.time() - t0) * 1e3)
    traced_ms = statistics.median(tw)
    got = model.to_host(traced_out).float()
    ttnn.release_trace(mesh_device, tid)
    pcc_same = metrics.pcc(got, ref)
    pcc_gold = metrics.pcc(got, want)
    print(f"TRACE begin_chunk (host input, outside the layers): {[round(x, 1) for x in bc]} ms")
    print(f"TRACE untraced layer stack: median {untraced_ms:.1f} ms of {[round(x, 1) for x in walls]}")
    print(f"TRACE traced replay:        median {traced_ms:.1f} ms of {[round(x, 1) for x in tw]}")
    print(
        f"TRACE replay output vs untraced pcc {pcc_same:.6f}, vs golden pcc {pcc_gold:.6f} (untraced vs golden "
        f"{metrics.pcc(ref, want):.6f})"
    )
    print("TRACE what still blocks a chunk-to-chunk trace:\n" + "".join(f"  - {b}\n" for b in BLOCKERS))
    assert pcc_same > 0.999, pcc_same
