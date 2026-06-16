# LTX-2 Audio Decode — Deep Profiling & Optimization Plan (bh galaxy)

Target: make `decode_audio` fast on 4x8, and optimize 2x4 + 1x4 too.
Current 4x8 (eager, LTX_VOC_TRACE=0): audio 1.08s, E2E 8.4s. Loudbox 2x4 traced: 0.80s.

## What the data already says (from the flushed 4x8 profile, cpp_device_perf_report)
- Device-active = 495ms; **Conv3d 39.8% (197ms)** dominates, BinaryNg 12.7% (1027 ops),
  Conv1dDepthwise 12.5%, Ternary 8.8%, NeighborPad 7.1%, Concat 6.6%, Slice 4.4%. CCL only 8.2%.
- Per-RISC: BRISC 483ms / NCRISC 474ms (data-movement) ≈ TRISC 380ms (compute) →
  **convs look bandwidth/read-bound, not FPU-bound.**
- 4416 device ops → high dispatch count. OP-TO-OP LATENCY in that run is GARBAGE
  (per-conv ReadDeviceProfiler flushes injected multi-second mesh syncs). Must re-measure clean.

## Hard facts about the tooling (researched 2026-06-12)
- HW counters EXIST and are MEASURED (FPU/SFPU/Math/pack/unpack/INSTRN-stall + L1-bank NoC
  transaction counts), but **per-kernel-zone, NOT per-µs**. Per-µs sampling does NOT exist on TT.
  Enable: `python -m tracy --profiler-capture-perf-counters fpu,sfpu,pack,unpack,instrn,l1_0`.
  L1 banks are mutually exclusive → one run per bank group.
- `PM FPU UTIL %` = ESTIMATE (op perf-model COMPUTE ns / kernel ns); only ops with a perf model.
  `NOC/DRAM/ETH BW UTIL %` come from **tt-npe** (needs build + `--collect/--analyze-noc-traces`).
  `CB WAIT FRONT/RESERVE BACK [ns]` = REAL measured stalls.
- nsight-like per-RISC timeline = the `.tracy` file in the **Tracy GUI** — needs a desktop or
  `ssh -NL 8086:127.0.0.1:8086` tunnel. `csvexport-release` dumps zones headless (no timeline view).
- Clean op-to-op gaps: `python -m tracy -p -r --op-support-count 6000` (NO manual ReadDeviceProfiler).
  Overflow = silent zone drop; 6000 covers the ~4400-op decode.
- Trace replay removes host dispatch (one command replays whole program from DRAM) BUT can be
  slower: it resets the prefetcher cache + serializes ops back-to-back, losing the host/device
  overlap eager gets, and adds DRAM trace-buffer read BW. Explains 4x8 1.37s traced > 1.07s eager.

---

## STEP 0 (GATING — DO THIS FIRST, before any profiling/optimization)
**User directive: ~10 min per test run is way too slow for iteration. Find why and speed it up.
Get a fast turnaround FIRST; only then proceed to profiling/optimization (Phases 1+).**

The actual decode compute is ~1.1s; the rest of a run is overhead. Make iteration fast for ALL
the paths we actually use — eager AND traced, audio-only AND full E2E — not just the PCC/eager path.
We explicitly WANT to measure WARM (cache-hot) time and to make the TRACED path iterate fast (the
plan's open question "why is traced slower than eager on 4x8" can only be answered with fast warm
traced runs). Do NOT write off the cold cost as "intrinsic" — the point of Step 0 is that the
SECOND+ run should be fast.

Known overhead of a ~430s E2E run (~179s audio-only): ~100s 22B weight load, ~170s cold audio-kernel
JIT compile, ~70s gemma load that EVICTS the DiT/VAE → gen#0 reloads it, ~35s gen#0 trace capture, 2 gens.
Levers:
- WARM CACHE: is `.tt-metal-cache` reused run-to-run, or thrashing (profiler/op-support-count changes,
  cache-key churn) so every run pays the ~170s recompile? Fix it so warm runs skip the build — measure
  the WARM wall, that's the iteration number we care about.
- Persistent warm dev harness: load weights + compile ONCE, then run many decodes (eager and traced)
  in one process — the right tool for fast iteration on both paths.
- Skip gemma load + DiT eviction/reload when the prompt encode is cached (~70s); applies to E2E runs too.
- Load ONLY the audio decoder for the audio test (no 22B DiT / gemma / video VAE) — audio_only path.
- Keep BOTH eager and traced runnable fast; for traced, the warmup/capture should reuse a warm cache so
  a traced iteration isn't a fresh 170s+35s every time.
Target: warm audio-test turnaround in the low-minutes (ideally seconds for the decode itself), for eager
AND traced. PCC gate (test_audio_decode_girl conv1d-vs-torch >0.95) must still pass on the full path.
THEN proceed to Phase 1 (which includes the eager-vs-traced device-active comparison).

## PHASE 0 — Harness fixes (unblocks clean measurement)
0.1 Add `LTX_PROF_NOFLUSH=1` path to prof_girl_decode.py: skip the per-conv ReadDeviceProfiler
    flushes (rely on `--op-support-count`). Removes the 32GB-log + contaminated-gap pathology.
0.2 Parametrize prof_girl_decode for 4x8, 2x4, 1x4 (mesh + sp/tp + T-shard factor).
0.3 Clean the 62GB of stale logs in generated/profiler/.logs ONLY after Phase 1 extraction.

## PHASE 1 — Clean per-op + per-RISC + stall profile (headless, WORKS NOW)
For each of {4x8, 2x4, 1x4}:
  `python -m tracy -p -r --op-support-count 6000 -m pytest prof_girl_decode::...-k <cfg>` (LTX_PROF_NOFLUSH=1)
Extract per-op + per-stage (mel-VAE / vocoder / BWE):
  - DEVICE FW, per-RISC (BRISC/NCRISC/TRISC) busy, OP-TO-OP LATENCY (clean), CB WAIT/RESERVE stalls.
  Deliverable: per-stage table of device-active, dispatch-gap, compute-vs-DM split, stall ns.
  Answers: is each stage compute-bound, DM/bandwidth-bound, or dispatch-bound — per config.

1b. EAGER vs TRACED device-active (settles "why is trace slower on 4x8", folded in from Phase 5).
  On 4x8, profile BOTH use_trace=on and off (LTX_VOC_TRACE) and compare sum(DEVICE FW DURATION)
  (device-active) AND OP-TO-OP LATENCY:
   - device-active EQUAL, only wall differs -> delta is host/dispatch/sync; trace should have helped,
     so replay is inserting gaps (find them in the Tracy timeline / op-to-op).
   - device-active HIGHER under replay -> replay changes device behavior (CCL barriers, CB config,
     scheduling) -> that's the real cause. THIS is the measurement, not speculation.
  NOTE the 1.37 vs 1.07 delta is already confirmed REAL (not tracy/debug): both came from plain
  pytest E2E, identical except the trace flag (no tracy, no watcher/dprint — device_params only set
  fabric_config). Measurement overhead cancels in the delta.

1c. Measurement hygiene / clean-wall controls (so absolute numbers are trustworthy):
  - Use bh_4x8sp1tp0_LINEAR (line_params, no trace region) as the clean eager control — the _ring
    config reserves trace_region_size=300MB even eager, shrinking memory (equal in both my runs, but
    affects absolute placement). Compare ring-eager vs linear-eager to size the reservation cost.
  - Run once WITHOUT LTX_TIME_STAGES (drops the per-substage synchronize_device barriers) to get the
    bare decode wall; the stage-split syncs are ~ms but confirm.
  - E2E timing is otherwise clean: stage walls are time.time() around stages that end in a device->host
    readback (real sync), no inserted debug.

## PHASE 2 — HW perf counters (headless, WORKS NOW; per-zone)
Re-run with `--profiler-capture-perf-counters fpu,sfpu,pack,unpack,instrn` (+ a 2nd run l1_0 for
NoC-transaction counts). Per-op: FPU%, SFPU%, math-thread stall, pack/unpack, L1 NoC transactions.
  Deliverable: per-op-type measured FPU vs SFPU vs stall — confirms Conv3d bandwidth-bound hypothesis.

## PHASE 3 — NoC/DRAM/ETH bandwidth + congestion (tt-npe BUILT & VERIFIED 2026-06-12)
3.0 READY: tt-npe at /home/smarton/tt-npe (built, install/ present). Import verified under the
    tt-metal venv. Activate: `source python_env/bin/activate && source /home/smarton/tt-npe/ENV_SETUP`.
3.1 Capture: `python -m tracy -p -r --collect-noc-traces -m pytest <decode>` (device emits per-op NoC
    event traces; recompiles kernels w/ noc instrumentation — opt-in, has overhead).
3.2 Analyze: `process_ops_logs.py --analyze-noc-traces` → fills NOC UTIL %, DRAM BW UTIL %,
    DRAM-BW-per-ctrl, ETH BW UTIL %, congestion + npe_viz timeline. WITHOUT this, those columns are
    blank (NOT zero — don't misread blanks as "no traffic", as happened this session).
  Deliverable: per-op DRAM BW%, NOC%, ETH(CCL) BW%, per-link congestion, NoC timeline (npe_viz).
  Answers: which ops saturate DRAM/L1 BW; whether the bandwidth-bound convs hit a real BW wall.

## PHASE 4 — nsight-like timeline (needs desktop/tunnel)
Open `tracy_profile_log_host.tracy` in Tracy GUI via SSH tunnel for the per-core/per-RISC dispatch
timeline. Use to eyeball op overlap (eager) vs serialization (trace replay) and inter-op gaps.
  (No per-µs counters — Tracy shows per-zone; this is the visual stall/overlap view.)

4b. STRETCH (devtool, not critical-path): per-zone counters in the Tracy GUI tooltip.
  Today device zones reach the GUI via the GPU-zone protocol (gpuZoneBegin/End = context+srcloc+time,
  NO text field), so clicking a zone shows name/duration, not FPU%/SFPU%. The values already exist
  per marker in TTDeviceMarker.meta_data (nlohmann::json; already carries e.g. noc_status_counter) and
  profiler_analysis already correlates the 9090 perf-counter markers to their zones (that's how the CSV
  gets per-op counters). Missing piece: emit meta_data into the streamed tt_device GPU-zone as
  text/annotation + render it server-side. Cost: a vendored-Tracy-fork protocol change (add a text slot
  to the tt_device GPU-zone) + GUI tooltip render — a profiler-team feature, NOT a quick inline change.
  Files: tt_metal/impl/profiler/profiler.cpp (emit), tt_metal/third_party/tracy/public/{client/TracyProfiler.cpp,
  common/TracyTTDeviceData.hpp} (protocol/zone), Tracy server tooltip. Recommendation: file upstream;
  for THIS work the CSV per-op counters + the timeline suffice (don't block on it).

## PHASE 5 — Trace-vs-eager dispatch analysis
Profile eager vs traced clean (Phase 1 method) on 4x8. Compare OP-TO-OP LATENCY sum.
  Hypothesis: eager overlaps host-prep w/ device-exec; replay serializes + resets prefetcher.
  Decision: trace only helps when host-dispatch-bound (small mesh). The durable fix for BOTH
  is OP-COUNT REDUCTION (fewer, bigger ops) — see Phase 6.

## PHASE 6 — Optimize (the actual speedups), PCC-gated by test_audio_decode_girl
6.1 Conv3d (39.8%, bandwidth-bound): tune blockings for 4x8 T-shard=8 shapes; reduce activation/
    weight re-reads; evaluate non-conv3d routing for the small channel-mix convs. [lever 1]
6.2 Deeper-commit rebuild+test (conv3d -25% 5a092bf, conv1d 6ad44658/d55c0b16): rebuild per commit,
    measure real E2E effect on the now-confirmed-dominant Conv3d/Conv1d. [lever 2 / C]
6.3 Op-count reduction (4416 ops): fuse 1027 BinaryNg residual adds, fold 602 NeighborPad halo into
    conv reads (proven pattern from conv1d_depthwise), cut 678 Slice / 308 Concat. Helps device + dispatch.
6.4 Per-config tuning: pick the best mesh for audio (likely 2x4/1x4 — audio doesn't scale); consider
    decoding audio on a submesh while diffusion uses full 4x8.

## PHASE 7 — Config sweep + write-up
Final per-config (4x8/2x4/1x4) E2E + per-stage table, before/after each lever. Commit each
optimization with E2E stage timings in the message (established practice).

## Caveats / asks
- tt-npe (Phase 3) needs a build — confirm it's available or I clone it.
- Tracy GUI (Phase 4) needs desktop or SSH tunnel from your machine.
- Per-µs HW counter view (nsight-style) is NOT possible on TT — per-zone is the limit. The
  Tracy timeline + per-zone counters are the closest.
- Rebuilds (6.2) go through the tt build recipe, ~10-30min each.
