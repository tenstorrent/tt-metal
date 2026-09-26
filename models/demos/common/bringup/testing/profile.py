# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Performance review, measurement half: warm per-section, per-chip device time of one long chunk.

Runs the last chunk of the profile rung (spec ``perf.rung``, default the last rung with ``prefix_from_golden``) after
loading the golden state prefix: once to compile, once unsynced for the real wall time, once with section profiling.
Writes ``<results>/<task>_profile.json`` (read by the dashboard and by the opportunity list) and records
chunk_wall_ms, host_transfers_per_layer, pcc_chunk_out (last layer's output vs the golden), prefill_ms_full (opt-in), prefill_tok_s,
prefill_chunk_ms_c<nn>, device_ms_total, device_ms_<phase>, device_ms_chip<c>, host_overhead_ms, profiled_programs.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference.golden import Golden
from models.demos.common.bringup.testing import profiler
from models.demos.common.bringup.testing.host_transfers import HostTransfers


def profile_rung(s) -> str:
    name = s.get("perf.rung")
    if name:
        return name
    cands = [r["name"] for r in s.data["ladder"] if r.get("prefix_from_golden")]
    return cands[-1] if cands else s.data["ladder"][-1]["name"]


def full_prefill(s, model, state, layers, rung, tokens) -> dict | None:
    """The whole target prefill, warm, with nothing read back: embed -> every layer -> final norm for every chunk, one
    device sync at the end. A first pass compiles every position-dependent program; the second is timed
    (prefill_ms_full, prefill_tok_s); a third syncs after each chunk for the per-chunk curve (prefill_chunk_ms_c<nn>).
    Excludes the LM head and sampling (a host LM head would not be device time). Needs the full layer stack."""
    if layers != list(range(s.num_layers)):
        return None
    seq, chunk = rung["seq"], rung["chunk"]
    n = seq // chunk

    def once(sync_each=False) -> list[float]:
        marks, t0 = [], time.time()
        for c in range(n):
            s0 = c * chunk
            h = model.embed(tokens[s0 : s0 + chunk])
            for i in layers:
                h2 = model.layer(i, h, s0, state)
                model.free(h)
                h = h2
            if c == n - 1:
                h2 = model.final_norm(h)
                model.free(h)
                h = h2
            model.free(h)
            if sync_each:
                model.sync()
                marks.append(time.time())
        model.sync()
        return [t0] + marks + [time.time()]

    once()  # compile
    t = once()
    total = t[-1] - t[0]
    metrics.record("prefill_ms_full", round(total * 1e3, 1))
    metrics.record("prefill_seq", seq)
    metrics.record("prefill_chunk", chunk)
    metrics.record("prefill_tok_s", round(seq / total, 1))
    t = once(sync_each=True)
    per = [round((b - a) * 1e3, 1) for a, b in zip(t[:-2], t[1:-1])]
    for c, ms in enumerate(per):
        metrics.record(f"prefill_chunk_ms_c{c:02d}", ms)
    print(
        f"full prefill {seq} tokens in {n} chunks of {chunk}: {total:.2f}s warm ({seq / total:.0f} tok/s); per chunk {per}"
    )
    return {"tokens": seq, "chunk": chunk, "ms": total * 1e3, "chunk_ms": per}


def run_profile(s, mesh, rung_name: str | None = None) -> dict:
    rung_name = rung_name or profile_rung(s)
    rung = s.rung(rung_name)
    g = Golden.for_rung(s, rung_name)
    layers = [i for i in s.layers() if i in g.layers]
    chunk = rung["chunk"]
    start = rung["seq"] - chunk
    model = s.hooks().device_model(mesh, s, layers, lm_head=False)
    state = model.new_state(rung["seq"])
    for i in layers:
        state.load_prefix(i, g.state(i), start)
    tokens = g.tokens()[start:]
    starts = {i for k, i in enumerate(layers) if k == 0 or layers[k - 1] != i - 1}

    host = []

    def run(count=False):
        h = None
        for i in layers:
            if i in starts:
                if h is not None:
                    model.free(h)
                h = model.embed(tokens) if i == 0 else model.from_host(g.layer(g.n_chunks - 1, i)["in"].float())
            if count:
                with HostTransfers() as ht:
                    h2 = model.layer(i, h, start, state)
                host.append(ht.total)
            else:
                h2 = model.layer(i, h, start, state)
            model.free(h)
            h = h2
        model.sync()
        if count:  # the counting run is untimed: also check this chunk is still right (cheap accuracy for perf work)
            got = model.to_host(h).float()
            metrics.record("pcc_chunk_out", metrics.pcc(got, g.layer(g.n_chunks - 1, layers[-1])["out"].float()))
        model.free(h)

    run()  # compile and fill the program cache: performance is measured warm only
    t0 = time.time()
    run()
    wall = time.time() - t0
    run(count=True)  # warm, apart from the timed run: host round-trips inside the forward pass (agent rule 5)
    metrics.record("host_transfers_per_layer", max(host))
    # The full-target prefill takes minutes; it is for a final number, not for every perf iteration (spec perf.full_prefill
    # or BRINGUP_FULL_PREFILL=1).
    want_full = s.get("perf.full_prefill", False) or os.environ.get("BRINGUP_FULL_PREFILL") == "1"
    full = full_prefill(s, model, state, layers, rung, g.tokens()) if want_full else None

    profiler.enable(mesh)
    run()
    profiler.signpost("end")
    prof = profiler.result()
    profiler.disable()

    total = sum(prof["kernel_ns"].values())
    assert total > 0, f"device profiler returned no durations; set {profiler.PROFILER_ENV}"
    phases = defaultdict(float)
    for sec, ns in prof["kernel_ns"].items():
        phases[profiler.phase_of(sec)] += ns
    for ph, ns in phases.items():
        metrics.record(f"device_ms_{ph}", round(ns / 1e6, 2))
    for sec, ns in prof["kernel_ns"].items():  # sub-sections a module marks itself ("attention.sdpa")
        if "." in sec:
            metrics.record(f"device_ms_{re.sub(r'[^A-Za-z0-9]+', '_', sec)}", round(ns / 1e6, 2))
    per_chip = defaultdict(float)
    for d in prof["kernel_ns_dev"].values():
        for c, ns in d.items():
            per_chip[c] += ns
    for c, ns in sorted(per_chip.items()):
        metrics.record(f"device_ms_chip{c}", round(ns / 1e6, 2))
    metrics.record("chunk_wall_ms", round(wall * 1e3, 1))
    metrics.record("chunk_start", start)
    metrics.record("chunk_len", chunk)
    metrics.record("device_model_hybrid", int("Hybrid" in type(model).__name__))
    metrics.record("device_ms_total", round(total / 1e6, 2))
    metrics.record("host_overhead_ms", round(wall * 1e3 - total / 1e6, 1))
    metrics.record("profiled_programs", sum(prof["programs"].values()))

    out = {
        "rung": rung_name,
        "chunk": [start, rung["seq"]],
        "layers": layers,
        "wall_ms": wall * 1e3,
        "device_ms": {ph: ns / 1e6 for ph, ns in sorted(phases.items(), key=lambda x: -x[1])},
        "sections_ms": {k: v / 1e6 for k, v in sorted(prof["kernel_ns"].items(), key=lambda x: -x[1])},
        "sections_ms_per_chip": {k: {str(c): v / 1e6 for c, v in d.items()} for k, d in prof["kernel_ns_dev"].items()},
        "programs": prof["programs"],
        "settings": getattr(model, "perf_settings", lambda: {})(),
        "full_prefill": full,
    }
    path = metrics.results_dir() / f"{metrics.task_id()}_profile.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    s.profiles_dir.mkdir(parents=True, exist_ok=True)
    (s.profiles_dir / f"{metrics.task_id()}_{time.strftime('%Y%m%dT%H%M%S')}.json").write_text(
        json.dumps(out, indent=1)
    )
    print(f"chunk [{start},{rung['seq']}): wall {wall * 1e3:.0f} ms, device {total / 1e6:.0f} ms")
    for sec, ms in list(out["sections_ms"].items())[:15]:
        print(f"  {sec:40s} {ms:9.1f} ms {100 * ms * 1e6 / total:5.1f}%")
    return out
