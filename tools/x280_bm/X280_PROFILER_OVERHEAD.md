<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# X280 profiler — end-to-end overhead

What it costs a real model to be profiled by the X280 perf-debug path, measured as the change in the
model's own reported inference time.

Companion to [`FINDINGS.md`](FINDINGS.md) (which covers the pipeline's internals and bandwidth work).
This file answers one narrow question: **if I turn the profiler on, how much slower does the model get,
and which stage is responsible?**

---

## Method

Four configurations, so the cost can be attributed rather than just totalled:

| cfg | what runs | isolates |
|---|---|---|
| **A** | no profiling at all | baseline |
| **B** | kernel instrumentation + X280 drain, **Tracy sink off** (`TT_METAL_PERF_DEBUG_NO_TRACY=1`) | device + drain cost |
| **C** | the real thing: instrumentation + drain + Tracy capture under `python -m tracy` | total cost |
| **D** | `python -m tracy --no-device` — wrapper on, **no** device profiling | the wrapper's own host cost |

`B` vs `C` separates the host sink from everything upstream of it. `D` vs `A` proves the tracy wrapper
isn't quietly contributing.

Rules the sweep follows, each one earned by getting it wrong at least once:

- **Interleave the configs within each rep** (A,B,C,D, A,B,C,D, …), never block them. The box drifts;
  blocking makes drift look like an effect.
- **A separate `TT_METAL_CACHE` per instrumentation state.** The JIT cache does **not** key on
  `PROFILE_KERNEL`, so a shared cache silently hands uninstrumented kernels to a profiled run — the run
  passes, drains 0 markers, and looks fast.
- **Warm each cache once and discard that run**, so every measured rep is a 100% cache hit. A cold ResNet
  compile is ~30 s against ~1.5 s warm; mixing the two swamps the signal.
- **`tt-smi -r 0` before every run**, including the baselines, so the reset is not itself a variable.
  (perf_debug does not self-reset; without it profzone bring-up fails `half_broken=true`.)
- **A hard `timeout` around every run.** See the trap below — one config can block forever.
- Report the **median of 3** and print the raw values. If the per-config spread overlaps the gap between
  configs, there is no result.

Driver and logs live on the box at `/localdev/$USER/e2e_overhead/` (`run.sh`, `extract.py`, `*.log`).

---

## ★ Trap: `python -m tracy` forces device profiling, and that HANGS the model without a drainer

`tools/tracy/__main__.py:404` — the `--no-device` option **defaults to `True`**, so the wrapper sets
`TT_METAL_DEVICE_PROFILER=1` by itself. Any "tracy wrapper but no device profiling" control therefore needs
an explicit **`--no-device`**.

Without it, the run compiles instrumented kernels and executes them with **no X280 consumer**. Every stage
of this pipeline is lossless-**blocking**, so the per-RISC L1 SPSC rings fill and the producers block
*forever*. Working exactly as designed — but it presents as a profiler hang, mid-layer, at 0% progress.

It also writes `PROFILE_KERNEL=1` entries into the cache it was pointed at, so an accident here
contaminates the *uninstrumented* cache. **Wipe both caches after one.**

---

## Result 1 — ResNet-50 b16, trace replay (short window)

`test_perf_e2e_resnet50.py::test_perf_trace[16-0.0024-30-device_params0]`, bh-07 (p100a),
5 warmup + **15 measured** iterations, 3 reps. Branch `x280-host-ring`.

| config | ms/iter (median of 3) | FPS | vs OFF |
|---|---|---|---|
| A — OFF | **1.6417** | 9746 | — |
| B — X280 drain, sink off | **1.7205** | 9299 | **+4.8 %** |
| C — X280 full capture | **1.7160** | 9324 | **+4.5 %** |
| D — tracy wrapper only | **1.6388** | 9763 | −0.2 % (noise) |

Raw ms/iter — A: 1.6365 / 1.6417 / 1.6706 · B: 1.6917 / 1.7205 / 1.7213 · C: 1.7051 / 1.7160 / 1.7204 ·
D: 1.6373 / 1.6388 / 1.6389. The A and B ranges do not overlap, so ~4.6 % is a real effect, not drift.

Health, identical in all six profiled runs: **1,581,952 markers** (bit-for-bit the same figure measured
earlier on bh-06 and bh-07 — this capture is deterministic), **0 producer stalls**, 0 ring drops.

### What it means

**`B ≈ C` is the headline.** Adding the entire Tracy sink on top of the drain costs *nothing measurable*.
The overhead is the RISCs writing markers into their L1 rings — **not** the X280 drain and **not** host
consumption. This is the BroadcastRing decoupling paying off: before it, the host sink back-pressured all
the way to silicon (826 producer stalls on UFLD, relay0 host-waiting 84 % of the run). The host is now
genuinely off the critical path.

**`D ≈ A`** confirms none of the 4.6 % is tracy-wrapper tax.

Scale: **+79 µs on a 1.64 ms iteration**, ~75 K markers per iteration across the grid.

### Caveat

15 iterations is only ~25 ms of device time. Enough to separate these configs cleanly, but it says nothing
about whether overhead grows under sustained volume — which is what Result 2 is for.

---

## Result 2 — UFLD-v2, trace + 2CQ, 1000 iterations (sustained window)

The other end of the range: 99.2 M markers at ~5.2 M/s, 63× ResNet's volume, and the model that originally
exposed the back-pressure bug. This is where the drain and sink are actually loaded, so it is the test that
would reveal a `B`-vs-`C` divergence if one exists.

Needs a burst-sized `TT_METAL_PERF_DEBUG_RING_RECS` to stay at 0 drops (Tracy ingests ~0.8 M rec/s against
the model's ~5.2 M/s, so the default 4 M ring drops ~84 % structurally).

_(pending — sweep running)_
