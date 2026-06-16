# Plan: Nsight-like profiling for tt-metal — full HW counters (compute + NoC), mesh-scale, repeatable

Status: DESIGN ONLY — for an implementing agent. Target repo: `/home/smarton/tt-metal`.
**Scope: company-wide tt-metal profiler infrastructure, model-agnostic.** This is *not* an LTX
feature — every model (LTX, Llama, Wan, MoE, …) profiles through the same tt-metal device
profiler, so the deliverable is general infra. **LTX-2 is the first validation customer**, not
the scope: its audio decode + DiT step are convenient mesh-scale stress workloads to gate each
phase against. Nothing below is LTX-specific except the named validation targets.

Author context: perf work needs a *correct* bound classifier. Kernel durations alone are
insufficient — equal BRISC/TRISC durations are consistent with both "co-limited at 50% each"
and "one saturated, one idling in lockstep." Distinguishing them REQUIRES counters:
FPU/SFPU/MATH utilization, instruction-stall cycles, semaphore-wait cycles, and **NoC
bytes/transactions → bandwidth utilization %**. Deliver those per device zone (per op × core ×
RISC), at 32-chip mesh scale, in BOTH the CSV and the Tracy GUI tooltip — **stably, repeatably,
and fast enough to iterate**, for any model.

## Goal (definition of done)
For every device zone (op, on each core/RISC) expose, alongside the existing per-RISC durations:
- **Compute:** FPU/SFPU/MATH active cycles → FPU util %, SFPU util %; unpack/pack busy;
  instruction-stall %; **semaphore-wait cycles** (the cross-chip-halo signal).
- **NoC/memory:** bytes & transactions per NoC (0/1), **NoC BW util %**, **DRAM BW util %**,
  **ETH BW util %**, against the part's peaks (BH ≈280 TFLOPs HiFi2, ≈512 GB/s DRAM; parameterize
  per part — this serves all hardware, not one box).
- Rendered in: (a) `ops_perf_results_<ts>.csv` as derived util columns + raw counts; and
  (b) the **Tracy GUI zone tooltip** (hover a zone → see its counters).
- **Scope control:** select what gets captured/processed (N cores per op-grid, chip subset,
  counter groups) so a profile is fast and bounded by default, full-coverage on opt-in.
- **Stable & repeatable:** the same workload profiled twice yields the same numbers within a
  stated tolerance; deterministic selection; pinned build hash (no cache thrash between runs).
- Works on an arbitrary model's hot loop — validated on the **full LTX audio decode (~4400 ops)
  and a full DiT denoise step** on bh 4x8 — with **zero dropped markers** and post-process that
  completes in minutes within ~16 GB RAM.

## Prerequisites — already fixed, MUST be in the base branch before starting
Two profiler fixes exist on `smarton/profiler-infra-fixes` (off `ltx-perf`); land/rebase them
into the base before any of the phases below or you will re-hit them:
- **`profiler: fix fabric NoC-event build error (MeshRoutingFields qualifier)`**
  (`fabric_event_profiler.hpp`, 2 lines) — **hard prerequisite for Phase 3**: without it
  `PROFILE_NOC_EVENTS` / `--collect-noc-traces` does not compile on a 2D-mesh fabric build, so
  NoC-event capture can't start. (This fixes the *compile*; the *runtime* ethernet hang in
  Phase 3b is a separate, still-open problem this does NOT address.)
- **`profiler: survive duplicate-key marker reinsert on counter overflow`**
  (`profiler.cpp` `updateDeviceMarker`, ~12 lines) — **safety net for Phase 1**: a dropped-marker
  (overflow) run can re-emit a colliding sort key; before this fix that corrupts the marker
  `std::set` traversal and segfaults. Guarded under `had_dropped_markers` → overflow degrades to
  a partial report instead of crashing. Phase 1 deliberately pushes the buffer to its limit, so
  this must be present.

## What already exists (do NOT rebuild — verified by code map)
- Counter enum `PerfCounterType` (205 types: FPU/SFPU/MATH, UNPACK, PACK, L1_0..4, INSTRN
  incl. per-thread semaphore-wait): `tt_metal/tools/profiler/perf_counters.hpp:10-206`;
  `PERF_COUNTER_PROFILER_ID = 9090` (`:7`); `PerfCounter` union (`:209-222`).
- CLI: `--profiler-capture-perf-counters fpu,pack,unpack,l1_0,instrn,...` →
  `TT_METAL_PROFILE_PERF_COUNTERS` bitfield, `tools/tracy/__main__.py:155-301`
  (bits: fpu=0,pack=1,unpack=2,l1_0=3,l1_1=4,instrn=5,l1_2/3/4=6/7/8). **`sfpu` has NO CLI
  bit** — it rides the FPU group's register path; fix the CLI to accept it or document.
- Device capture: TRISC1 arms (`perf_counters.hpp:332-346`), BRISC reads + emits markers
  (`:407-428`). L1 banks mutually exclusive — one bank/run, enforced
  `tt_metal/llrt/rtoptions.cpp:867-881`; CLI already does **multi-pass** for >1 L1 group.
- NoC: 12-bit `counter_value` per NoC event (`event_metadata.hpp:92-119`), already written to
  `meta_data["noc_status_counter"]` + dst/src addr in `profiler.cpp:1973-1975`. NoC-event
  capture via `TT_METAL_DEVICE_PROFILER_NOC_EVENTS` (+ `_RPT_PATH`), `__main__.py:233-237`;
  emits `noc_trace*.json`, `topology.json`, `cluster_coordinates.json`. tt-npe (external,
  `/home/smarton/tt-npe`) turns those into NOC/DRAM/ETH BW util %.
- Perf-counter → zone correlation + util math: `tools/tracy/perf_counter_analysis.py`
  (`COUNTER_TYPE_NAMES`, `compute_perf_counter_metrics`) and `process_ops_logs.py:834,1176`
  (`extract_perf_counters`) → CSV columns "PM FPU UTIL (%)" etc.
- `meta_data` (nlohmann::json) lives on every `TTDeviceMarker`
  (`third_party/tracy/public/common/TracyTTDeviceData.hpp:81-246`; comparator sort key
  `(timestamp,chip,core_x,core_y,risc,marker_id)` at `:179-196`) and already carries the
  counter values (`profiler.cpp:2221-2224`).
- C++ fast post-process: `TT_METAL_PROFILER_CPP_POST_PROCESS=1` + `TT_METAL_PROFILER_MID_RUN_DUMP=1`
  → `ttnn.GetLatestProgramsPerfData()/GetAllProgramsPerfData()`
  (`ttnn/cpp/ttnn-nanobind/profiler.cpp`, `tt_metal/impl/profiler/profiler_analysis.{hpp,cpp}`).
  **Per-PROGRAM granularity today, NOT per-core/per-RISC, and NoC events are not folded in.**
- Buffer sizing: `profiler_state_manager.cpp:19-66` (`TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT`,
  default 1000) → `-DPROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC` at JIT
  (`jit_build/build.cpp:185-186`) → **changing it re-hashes every kernel (full recompile)**.
- Streaming alt: real-time profiler (PR #42905) — L1 ring buffer + D2H socket, no DRAM
  overflow, but **does NOT carry counters yet** (`impl/dispatch/realtime_profiler_*`).
- **Per-core/per-RISC read selection ALREADY EXISTS (verified) but is UNEXPOSED.**
  `readRiscProfilerResults` takes `riscs_to_include` (a `map<CoreCoord, set<RiscType>>`) and
  filters at readback — `profiler.cpp:1551-1552`: if set and doesn't contain this (core, risc),
  skip it. But `getVirtualCoresForProfiling` (`tt_metal_profiler.cpp:976`) returns *all* worker +
  eth cores and nothing passes a restricted set. Exposing this hook via env/API is the
  scope-control lever below — it shrinks **readback + post-process**, not the on-device buffer
  (which is sized per-RISC × `PROGRAM_SUPPORT_COUNT` across the whole grid, `profiler_state_manager.cpp:59`).

## The three real obstacles (everything else is plumbing)
1. **GUI gap:** the `tt_device` GPU-zone protocol (`TracyTTDevice.hpp:203-253`
   PushStart/PushEndMarker) has **no text/annotation slot**; `meta_data` is never streamed to
   the Tracy server, so the GUI can't show it. Needs a vendored-fork protocol change.
2. **Capture overflow at scale:** counter markers multiply per zone (FPU 3, UNPACK ~15,
   INSTRN 48, L1 16…) × 6 RISCs × 32 chips → blows the fixed per-RISC DRAM marker buffer
   (~192k markers ≈ 768000 B/RISC). This is what segfaulted/OOM'd the LTX work.
3. **Post-process cost:** legacy `process_ops_logs.py` loads the whole device CSV into pandas
   (chip×core×risc×counter×marker Cartesian) → >140 GB / OOM at mesh scale; NoC analysis is
   Python/tt-npe only and not joined to the per-zone compute view.
4. **No scope control:** capture is all-or-nothing (every worker + eth core). There's no way to
   say "3 cores per op-grid across all chips" even though the readback hook exists. This is what
   makes profiles slow *and* non-repeatable (volume varies run-to-run, OOM risk). Exposing scope
   control fixes iteration speed, OOM, AND repeatability at once.

---

## Phased plan (each phase independently lands value; MVP = Phases 0–3, GUI = Phase 4)

### Phase 0 — Trust the numbers (validation harness FIRST)
Counters are worthless if the util math is wrong. Build a micro-benchmark with **analytically
known** work before believing any %.
- A single matmul of known M,K,N → known FLOPs; a single DRAM-bound copy of known bytes.
- Capture counters, compute FPU util % and DRAM BW % via the existing
  `compute_perf_counter_metrics`, assert within tolerance of the analytical peak (BH
  reference: `interpretation.md` peak table — 280 TFLOPs HiFi2, 512 GB/s).
- Deliverable: `tests/ttnn/tracy/test_counter_utilization_sanity.py`; a PASS here is the gate
  for trusting every downstream number. **If util math is off, fix it here, not at scale.**

### Phase 1 — Lossless counter capture at mesh scale
Pick the lowest-risk path that achieves zero drops on the full decode + a DiT step:
- **1a. Right-size + pin the buffer.** Compute required markers/RISC for the chosen counter
  set and set `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT` to fit; **pin one value** and reuse it
  (avoid recompile thrash; `tracy.md` documents the hash dependency). Document the markers-per-
  zone formula from `perf_counters.hpp` so the size is derived, not guessed.
- **1b. Mid-run dump for long runs.** Use `TT_METAL_PROFILER_MID_RUN_DUMP=1` + a **coarse**
  `ttnn.ReadDeviceProfiler` cadence (per stage / few-hundred ops, after `synchronize_device`).
  NEVER per-op (mesh sync per flush garbles op-to-op latency — observed in LTX work).
- **1c. Multi-pass merge.** L1 banks are one-per-run; the CLI already multi-passes. Extend the
  post-process to **merge passes by zone key** so a single per-zone row carries all L1 banks +
  compute groups (today passes are separate captures). Same merge folds in the NoC-events pass.
- **Fallback if 1a–1c can't fit:** carry counters over the **real-time profiler ring buffer**
  (PR #42905) instead of the DRAM marker buffer — extend its mailbox/record format with a
  counter payload. Higher effort; only if buffer-sizing proves insufficient.
- Gate: full audio decode + 1 DiT step, `fpu|pack|unpack|instrn` + one L1 group/pass, **0
  "markers were dropped"** in the log (grep it; partial reports are a HARD failure).

### Phase 2 — Fast, bounded-memory post-process (kill the OOM)
- **2a. Per-RISC C++ fast path.** Extend `profiler_analysis.{hpp,cpp}` + the nanobind
  `GetLatestProgramsPerfData` to return **per-core/per-RISC** rows with counters (not just
  per-program), so the OOM-prone Python path is bypassed for the common case.
- **2b. If staying in Python, make it streaming.** Replace the pandas full-load in
  `process_ops_logs.py` with chunked/columnar (pyarrow/polars), **per-chip** processing with a
  thread pool, and incremental aggregation. Bound peak RAM (~16 GB target).
- Gate: full-decode post-process completes in minutes, < ~16 GB RAM, every op has compute-
  counter columns.

### Phase 2.5 — Scope control (the iteration-speed + repeatability lever)
Expose the verified `riscs_to_include` hook (`profiler.cpp:1551`) via env/API so a profile can
be bounded by default and full on opt-in. **Compute and NoC need DIFFERENT strategies — do not
treat them the same:**
- **Compute counters → SAMPLE per op-grid.** Ops are SPMD, so a few cores represent an op's
  compute. But cores must be **sampled from each op's actual grid** (recorded core range /
  `CORE COUNT`), NOT a fixed core mask — a fixed mask misses ops whose grid excludes those cores.
  Resolve grids via a cheap discovery pass (or post-hoc from the op's recorded core range).
- **NoC counters → CAPTURE ALL active cores, AGGREGATE per op.** Bandwidth is an
  aggregate-across-cores quantity (Σbytes ÷ time); per-core NoC load is non-uniform
  (DRAM-adjacent, corner, eth cores differ), so **sampling undercounts BW** — wrong tool. Instead
  read all NoC-active cores (worker BRISC/NCRISC + eth) but **sum to one per-op row** on read, so
  the BW total is correct and the row count still collapses. (Saves post-process rows, not
  readback — readback needs all cores to sum.)
- **Chip subset:** allow restricting to K chips for a quick look, but keep ≥1 edge + ≥1 interior
  (CCL/halo cost diverges across the mesh); full-mesh for the final number.
- **Counter-group subset:** capture only the groups asked for (fewer markers → less overflow).
- Gate: a bounded profile (e.g. 3 compute cores/op-grid + per-op NoC aggregate, all chips) on the
  validation workload reproduces the full-coverage per-op util numbers within a stated tolerance,
  and post-process drops ~1–2 orders of magnitude in rows/time.

### Phase 3 — Unified NoC/memory counters joined to the compute view
- **3a. Fold NoC into the per-zone row (aggregated across the op's cores — see Phase 2.5).**
  Sum `noc_status_counter` + fabric-event counts (`fabric_event_profiler.hpp`) over all the op's
  NoC-active cores → bytes/transactions per NoC; compute **NoC/DRAM/ETH BW util %** = Σbytes ÷
  (zone duration × peak BW). Emit as CSV columns next to FPU util %, so a single row answers
  "compute- vs bandwidth- vs sync-bound" unambiguously. (Per-op aggregate, not per-core rows.)
- **3b. Fix the NoC-trace ethernet wedge.** `--collect-noc-traces` recompiles fabric/erisc
  kernels with NoC instrumentation and has been observed to hang ethernet init ("Timed out
  waiting for active ethernet core to reset", `llrt.cpp`). The code map could NOT locate the
  root cause statically → this needs **on-device** repro + bisect (instrumented vs
  non-instrumented fabric build). Until fixed, NoC BW % at full mesh scale is blocked; the
  on-device `noc_status_counter` (3a) is the interim source.
- Gate: a known all-gather/halo op's measured bytes match analytical (e.g. tensor size ×
  participants); NoC BW % is sane (≤100%, >0 for CCL ops, ≈0 for local convs).

### Phase 4 — Tracy GUI zone tooltips (the Nsight-like view)
Highest effort (vendored-fork protocol change) — do AFTER the CSV path delivers value.
- **4a. Protocol slot.** Add a per-zone annotation to the `tt_device` GPU zone. Investigate
  first whether Tracy's existing `ZoneText`/`ZoneValue` is reachable from the serial GPU-zone
  path; if not, add a new `QueueType` (e.g. `GpuZoneAnnotationSerial`) carrying a compact
  string keyed to the zone. Touch points:
  - emit: `third_party/tracy/public/tracy/TracyTTDevice.hpp` (after PushEndMarker),
  - queue/protocol: `third_party/tracy/public/client/TracyProfiler.cpp`,
    `third_party/tracy/public/common/TracyQueue.hpp`,
    `third_party/tracy/public/common/TracyTTDeviceData.hpp`,
  - server render: the vendored Tracy server tooltip/zone-info function.
- **4b. Emit a compact counter set** (util % + the few key raw counts — FPU%, SFPU%, stall%,
  sem-wait, NoC BW%, DRAM BW%) from where `meta_data` is already populated
  (`profiler.cpp:2221-2224` / `:1973-1975`). Keep it small — tooltips, not the full 200-counter
  dump.
- **4c. Keep the fork rebasable.** Isolate the change behind clear `// TT:` markers and a
  single header so future Tracy upstream merges are tractable; consider filing upstream.
- Gate: open an archived `.tracy` in the GUI, hover a `Conv1dDepthwise` and an `AllGather`
  zone, see FPU%/NoC-BW%/stall% that match the CSV row.

### Phase 5 — Workflow + docs
- One-command capture that produces the unified CSV **and** an annotated `.tracy`, auto-archived
  to `~/traces/<ts>/` (reports dir wipes per run — never rely on it).
- Update the `tt:profiler` skill (`tracy.md`, `interpretation.md`) with the counter recipe,
  the markers-per-zone sizing formula, the multi-pass/L1 constraint, and the sampling guidance.
- Update `/home/smarton/profiling-recipe.md`.

## Sequencing & honest scope
- **MVP (Phases 0–3)** delivers the thing the perf work actually needs: a per-op row with
  compute **and** NoC utilization %, lossless and non-OOM at 32 chips. That alone fixes the
  "co-limited?" ambiguity. Do this first.
- **Phase 4 (GUI tooltips)** is a vendored-Tracy-fork protocol change — real profiler-team-
  scale work; valuable but not on the critical path to a correct bound call. Sequence it after
  the CSV path is trusted.
- This is **multi-week**, touches the JIT build, the device kernels, a vendored third-party
  tree, and host post-process. Land it phase-by-phase behind env gates; each phase has a
  validation gate above and must not regress the default (no-counter) profile.

## Repeatability requirement (applies to every phase — this is THE thing that makes it usable company-wide)
A profile that isn't repeatable isn't evidence. Enforce, and add a CI/regression gate for:
- **Pinned build hash:** fix `PROGRAM_SUPPORT_COUNT` + counter config so the kernel hash is
  constant run-to-run → 100% JIT cache hit, no silent recompile-induced variance.
- **Deterministic selection:** the per-op-grid core sample is a deterministic function of the
  op's grid + a fixed seed, so two runs pick the same cores.
- **Warm-only window:** profile steady-state (drain cold/load zones first); never rank the cold sum.
- **Variance gate:** same workload profiled twice → top-op device-time and util %s agree within a
  stated tolerance (e.g. ±2%). If not, the capture is not trustworthy — fail loud.
- **Archived artifacts:** unified CSV + annotated `.tracy` + the exact env/scope copied to a
  durable dir (`~/traces/<ts>/`); the reports dir wipes per run.

## Open questions for the implementing agent to resolve on-device (don't guess)
- Exact markers-per-zone for each counter group on Blackhole (derive from `perf_counters.hpp`,
  confirm against a real capture) → the buffer-sizing formula in 1a.
- Whether `GetLatestProgramsPerfData` can be extended to per-RISC without a protocol change, or
  if 2b (streaming Python) is the faster route.
- Root cause of the NoC-trace ethernet-init hang (3b) — requires a device bisect.
- Whether Tracy's upstream `ZoneText` is usable for GPU zones or a new `QueueType` is required
  (4a) — determines Phase 4 effort.
- How to resolve each op's core grid cheaply for per-op-grid sampling (2.5): a discovery pass vs
  reading the program's core ranges at dispatch vs post-hoc from recorded markers.
- Whether per-op NoC aggregation (3a/2.5) can happen on read (C++) to avoid ever materializing
  per-core NoC rows, keeping the bound-memory promise on the largest meshes.

## Relationship to other docs
- **Consumes nothing model-specific.** Model startup/iteration cost (weight load, mesh-workload
  assembly, per-model warmup) is OUT OF SCOPE — that's runtime work, tracked separately (see the
  profiling-iteration plan / per-model harnesses). This doc owns the *profiler*, not the model.
- The two `smarton/profiler-infra-fixes` commits are the entry condition (see Prerequisites).
