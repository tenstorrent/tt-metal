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

`test_ufld_v2_e2e_performant.py::test_ufldv2_e2e_performant[1-device_params0]`, bh-07, trace + 2 CQ,
batch 1, **1000 measured iterations**, 3 reps. `TT_METAL_PERF_DEBUG_RING_RECS=134217728` (128 M records
≈ 3.1 GB) on B and C — Tracy ingests ~0.8 M rec/s against the model's much higher rate, so the default
4 M ring drops ~84 % structurally.

| config | ms/iter (median of 3) | FPS | vs OFF | wall |
|---|---|---|---|---|
| A — OFF | **1.8360** | 545 | — | 10.1 s |
| B — X280 drain, sink off | **1.8680** | 535 | **+1.7 %** | 10.6 s |
| C — X280 full capture | **1.8680** | 535 | **+1.7 %** | 29.7 s |
| D — tracy wrapper only | **1.8360** | 545 | ±0.0 % | 11.4 s |

Every rep reproduced its config's value exactly. That is not a copy-paste error: the test rounds to 6
decimal places and averages over 1000 iterations, so run-to-run variance lands below the 1 µs print
resolution. Health: **99,187,072 markers**, **0 producer stalls** on both sockets, **0 ring drops**
(consumer took 99,187,072 of a 128 M ring) — in all six profiled runs.

### `B == C`, now with a much sharper demonstration

`C`'s **wall clock is 3× `B`'s** — 29.7 s against 10.6 s — because Tracy is serializing a ~490 MB capture.
And the measured inference time is **identical to the microsecond**. Twenty seconds of sink work, zero
cost to the model. Under the old inline design that same work sat directly on the drain thread and
back-pressured into silicon; here it is entirely behind the BroadcastRing and lands after the timed loop.

### The two windows disagree, and the sustained one is the trustworthy figure

| | window | markers | overhead |
|---|---|---|---|
| ResNet-50 | 15 iters ≈ 25 ms | 1.58 M | +79 µs/iter (+4.6 %) |
| UFLD-v2 | 1000 iters ≈ 1.84 s | 99.2 M | +32 µs/iter (+1.7 %) |

UFLD emits *more* markers per iteration yet costs 2.5× **less** per iteration. Overhead is not scaling
with volume, which is the opposite of what a per-marker cost would do.

**Leading hypothesis (not yet proven): a fixed per-run cost sitting inside ResNet's timed region.** Its
timed region is `pipeline.enqueue(host_inputs).pop_all()` — 15 enqueues *plus a full flush and host
readback of all 15 outputs*. A one-time profiler-induced cost at that flush of only ~0.7 ms would fully
account for the gap, since ResNet divides it by 15 while UFLD divides it by 1000. The arithmetic fits, but
a plausible arithmetic fit is not evidence.

> **This hypothesis was REFUTED by Result 3 below** — no editing of the model file was needed, because the
> four extra models happen to span windows of 1, 10 and 15 iterations. Overhead does not track window
> length. Left in place because the reasoning is the sort that looks convincing and is wrong.

---

## Result 3 — four more models: overhead is 0.3 % to 10 %, and dispatch mode is what predicts it

Configs **A** and **C** only, 3 reps interleaved, same rules. Blackhole-native model coverage is thin;
the four below are what actually runs (see "models that do not run" at the end).

Collected with everything measured so far, sorted by overhead:

| model | dispatch | window | A ms/iter | C ms/iter | overhead | markers |
|---|---|---|---|---|---|---|
| ResNet-50 b32 | trace + 2 CQ | 15 | 2.4448 | 2.4533 | **+0.3 %** | 1.61 M |
| SD 1.4 UNet | trace | **1** | 45.2950 | 45.7416 | **+1.0 %** | 2.50 M |
| UFLD-v2 | trace + 2 CQ | **1000** | 1.8360 | 1.8680 | **+1.7 %** | 99.2 M |
| ResNet-50 b16 | trace | 15 | 1.6417 | 1.7160 | **+4.5 %** | 1.58 M |
| ResNet-50 b16 | **non-trace** | 15 | 12.3480 | 13.5970 | **+10.1 %** | 1.58 M |
| VGG-UNet | trace + 2 CQ | 10 | 3.2430 | 2.9290 | **−9.7 %** ⚠️ | 0.56 M |

All profiled runs: **0 producer stalls, 0 ring drops.**

### Two hypotheses die here

**Window length does not predict overhead.** 1 iteration → +1.0 %; 15 iterations → anywhere from +0.3 % to
+10.1 %; 1000 iterations → +1.7 %. There is no trend. The Result-2 fixed-cost story is dead.

**Marker count does not predict overhead either.** UFLD emits **99.2 M** markers for +1.7 %. ResNet-50 b16
non-trace emits **1.58 M** — 63× fewer — for **+10.1 %**. And the cleanest pair available: ResNet-50 b16
trace vs non-trace are the *same model at the same batch* with essentially the same marker count
(1,581,952 vs 1,578,688), yet absolute overhead is **+79 µs vs +1249 µs — 16× apart.**

### What does predict it: **program dispatch count**

The trace/non-trace pair isolates exactly one variable. Trace replay issues **one** program launch per
iteration; non-trace dispatches every op individually from the host. Per-launch profiler work (init,
finish, control-buffer handling) is therefore paid hundreds of times per iteration instead of once.

That ranks the whole table correctly: every trace-based configuration lands at **≤ +4.5 %**, and the single
non-trace configuration is the outlier at **+10.1 %**.

Practical consequence: **quote overhead per dispatch mode, never as one number.** For trace-replay
workloads — which is what perf work usually targets — the honest figure is **≤ 5 %**, and typically 1–2 %.

### ⚠️ VGG-UNet runs 9.7 % FASTER with the profiler on, and I cannot explain it

This is not noise and not a sign error. Four configurations, 3 interleaved reps each (medians):

| A — off | D — tracy wrapper only | B — X280, sink off | C — X280 full |
|---|---|---|---|
| 3.243 ms | 3.196 ms | **2.834 ms** | **2.929 ms** |

What that rules out:
- **Not the tracy wrapper** — `D ≈ A` (−1.4 %).
- **Not the Tracy sink** — `B ≈ C`, consistent with every other result here.
- **Not chip warm-up from the profiler's device activity.** Tested directly: four consecutive config-A runs
  with `tt-smi -r 0` *only* before the first gave 3.236 / 3.229 / 3.463 / 3.447 ms — warm A is not faster,
  so the per-run reset is not penalising the baseline.
- **Not one-time setup captured inside the timed region** — `initialize_vgg_unet_trace_2cqs_inference()`
  (which does `begin/end_trace_capture`) runs *before* `t0`; the window is 10 `run()` calls and a sync.

So enabling instrumentation genuinely makes this model faster, reproducibly. Remaining candidates, none
tested: instrumented kernels shifting L1 allocation (profiler buffers) and with it sharding/NoC behaviour;
or the added per-zone code perturbing a dispatch/NoC overlap that was pathological in the uninstrumented
build. Both are guesses.

**Why this matters more than the sign:** a profiler that can move a workload ±10 % is not merely adding
time, it is *changing the thing being measured*. The 0-stall result proves the X280 never back-pressures
the producers — it does not prove the instrumented workload behaves like the uninstrumented one. Worth
resolving before anyone treats X280 numbers as ground truth for a model this size.

### Models that do not run on Blackhole (probed, 2026-07-28)

| model | blocker |
|---|---|
| ViT b10 | `ImportError: cannot import name 'HfFolder' from 'huggingface_hub'` — removed in hub ≥ 1.0. Fixing it means downgrading below 1.0, which transformers 5.x likely needs; not worth the venv. |
| SentenceBERT b8 | `TT_FATAL … shards along height 8 must not exceed number of cores 7` — model/device grid mismatch, unchanged from the earlier survey. |
| MobileNetV2, SegFormer | `@run_for_wormhole_b0()` — skipped on Blackhole by decorator. |

**Stable Diffusion needed an env fix and now works**: `diffusers` 0.35.1 could not import against the
pinned `transformers == 5.10.2` (`FLAX_WEIGHTS_NAME` was removed). Installed `diffusers 0.39.0` with
**`--no-deps`** specifically so transformers stays pinned — qwen36 depends on that pin, and upgrading
normally would trade one working model for another.
