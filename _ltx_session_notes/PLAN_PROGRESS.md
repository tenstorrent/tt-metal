# LTX-2 Audio Decode — Plan Progress Log

Branch: smarton/optimizer/ltx-galaxy-v2. Timestamps in UTC / PT.

## STEP 0 — Fast turnaround (GATING)

### Diagnosis (from prior `audio_only.log`, 4x8, LTX_TRACED=1, 165.87s total)
Full timeline of the ~166s audio test:
- 04:39:48 process start -> 04:40:01 device open + VAE load (~12s)
- 04:40:01 -> 04:40:05 upsampler + audio decoder load (~4s)
- **04:40:05 -> 04:40:39 22B transformer cache load ~35s — PURE WASTE for audio_only**
  (decode_audio never touches the transformer/VAE/upsampler)
- 04:40:39 -> 04:42:13 **cold first decode ~94s** (`vocoder+bwe=93171.8ms`).
  NOT JIT compile: JIT cache stats = 1432/1455 hits (98.4%). This is first-run device-side
  op/program/mesh-workload assembly for the ~4400-op audio graph. This is the real floor.
- 04:42:13 -> 04:42:37 warm decodes + torch-oracle + PCC (~24s), PASSED, teardown to 04:42:45.

Warm steady-state: warm=878ms, PCC=0.99379 (LTX_TRACED=1 path).

### Levers identified
1. **audio_only must skip building+priming the transformer/VAE/upsampler** (saves ~35s).
   FOUND: `audio_only=True` only gated the *video warmup*; `__init__` still ran
   `_instantiate_modules` (built 22B transformer + all variants + VAE + upsampler) and
   `_prime_caches` (pushed 22B transformer weights to mesh). decode_audio uses NONE of these.
2. **Kernel-cache location mismatch.** Warm cache lives at `~/.cache/tt-metal-cache`
   (1432 cached kernels). The broker env sets `TT_METAL_CACHE=/home/smarton/.tt-metal-cache`
   which is nearly empty (2 entries) -> a broker run would trigger the ~170s cold JIT recompile.
   Fix: point runs at the warm cache.
3. WARM_REPS env knob added (default 5) to trim the steady-state loop for dev iteration;
   PCC oracle path untouched so the >0.95 gate is fully exercised.

### Changes made (not yet committed — validating first)
- `pipeline_ltx.py`: `audio_only` now threaded into `_instantiate_modules` + `_prime_caches`;
  when set, only the audio decoder shell is built and primed.
- `test_pipeline_ltx_distilled.py`: `WARM_REPS` env (default 5).
- `run_step0_audio.sh`: Step-0 validation (LTX_TRACED=0 -> PCC gate active).

### Changes made — NOW COMMITTED as b53d5d3 (this worktree)
- `pipeline_ltx.py`: `audio_only` threaded into `_instantiate_modules` + `_prime_caches`;
  when set, only the audio decoder shell is built+primed (skips 22B transformer/VAE/upsampler).
- `test_pipeline_ltx_distilled.py`: `WARM_REPS` env (default 5).
- `prof_girl_decode.py`: `LTX_PROF_NOFLUSH=1` + 1x4 case.

### MEASURED — b53d5d3 warm, job 012206-12 (2026-06-13 01:22-01:48 UTC / 18:22-18:41 PT)
Config: bh_4x8sp1tp0, LTX_TRACED=0 (eager, full conv1d+mac PCC oracle), WARM_REPS=3,
TT_METAL_CACHE=/home/smarton/.cache/tt-metal-cache, TT_METAL_HOME=worktree.
Audio test run TWICE in one job (2nd run = process-warm kernel cache).

| metric                | RUN 1 (cold cache) | RUN 2 (warm cache) |
|-----------------------|--------------------|--------------------|
| JIT kernel-cache hits | 0/1474 (0.0%)      | 1474/1474 (100%)   |
| cold first decode     | 509990 ms (~510s)  | 317037 ms (~317s)  |
|   - cold mel_vae      | 22175 ms           | 405 ms             |
|   - cold vocoder+bwe  | 487814 ms          | 316631 ms          |
| warm/decode (avg)     | 1051.9 ms          | 1066.1 ms          |
|   - warm mel_vae      | ~184 ms            | ~184 ms            |
|   - warm vocoder+bwe  | ~866 ms            | ~870 ms            |
| process wall          | 896 s (14m41s)     | 691 s (11m16s)     |
| PCC conv1d-vs-torch   | 0.99379 (PASS>.95) | 0.99379 (PASS)     |
| PCC mac-vs-torch      | 0.99613            | 0.99613            |

**audio_only effect (b53d5d3) CONFIRMED LIVE:** log shows only audio-decoder shells built;
audio_dec + audio_voc caches load in ~3s; NO 22B transformer load (the ~35s prior waste is gone).

**KEY FINDINGS / corrections to prior diagnosis:**
1. Warm steady-state decode = ~1.05s eager (mel 0.18s + vocoder+bwe 0.87s). Stable run-to-run.
2. The ~510→317s "cold" first decode is NOT JIT compile: run 2 had 100% kernel-cache hits and
   STILL paid 317s in the first vocoder+bwe call. Cold = device-side program/mesh-workload
   assembly for the ~4400-op audio graph (one-time per process). MUCH bigger than the prior
   ~94s estimate (that earlier number was LTX_TRACED=1; this eager LTX_TRACED=0 path also runs
   the extra mac-vocoder oracle pass, which roughly doubles run 1's cold).
3. Kernel-cache-key churn (lever #2) REPRODUCED & UNDERSTOOD: the pre-existing 56 entries in
   ~/.cache/tt-metal-cache were keyed for a DIFFERENT build → run 1 got 0 hits (full recompile,
   the ~170s the plan warned about). With TT_METAL_HOME pinned to the worktree the cache rebuilt
   once (57 entries) and run 2 hit 100%. So warm runs DO skip recompile now — but only after one
   priming run on this exact TT_METAL_HOME/build. Lock TT_METAL_HOME + TT_METAL_CACHE for reuse.
4. Process-per-iteration is dominated by the 317s cold assembly, not compile. The 14-min wall is
   mostly the cold decode + the CPU torch-oracle. => persistent-warm-dev-harness (load+assemble
   ONCE, loop many eager+traced decodes in one process) is the real fix; each extra decode ~1s.

### Harness/runnability notes (worktree has no build/venv)
- Worktree lacks build/ + python_env (worktrees don't get their own). b53d5d3 is PURE PYTHON
  (models/ only) so main's compiled binaries are valid. Symlinked into the worktree:
  build, build_Release, python_env, ttnn/ttnn/_ttnn.so -> /home/smarton/tt-metal/... .
- Broker needs PYTHON_ENV_DIR; env file /home/smarton/ltx_worktree_env.yaml points
  TT_METAL_HOME+PYTHONPATH at the worktree, PYTHON_ENV_DIR at main's python_env,
  TT_METAL_CACHE at the warm ~/.cache/tt-metal-cache.
- Scripts: /home/smarton/run_step0_b53d5d3_warm.sh (this measurement, 2x audio test).

### Persistent warm dev harness — BUILT + MEASURED, job 020258-13 (2026-06-13 02:03-02:09 UTC / 19:03-19:09 PT)
New file `models/tt_dit/tests/models/ltx/audio_warm_harness.py` (+ run_warm_harness_4x8.sh).
ONE process: build audio_only pipeline once -> one cold decode (pays the 317s device-side
assembly ONCE) -> then loops warm decodes EAGER then (in-process switch via release_traces +
flip pipeline._traced) TRACED, per-stage timed. No torch oracle (timing-only; PCC stays
test_audio_decode_girl's job). 12 decodes in 5m44s total -> after cold, each decode ~1s.

bh_4x8sp1tp0, conv1d path, kernel cache 100% hits (1454/1454):
| leg    | mel_vae | vocoder+bwe | total    |
|--------|---------|-------------|----------|
| cold   |   —     |     —       | 317367ms |
| EAGER  | 182.4ms |   864.0ms   | 1045.1ms |
| TRACED | 182.8ms |   912.0ms   | 1098.8ms | (LTX_VOC_TRACE=1, BWE/VAE trace off)

**Phase 1b answered (eager-vs-traced, isolated in ONE warm process, only the trace flag differs):**
TRACED is +53.7ms SLOWER (1098.8 vs 1045.1), and the entire delta is in vocoder+bwe
(912 vs 864ms); mel_vae is identical (182-183ms). Confirms the memory note's "trace
net-negative at real scale on 4x8" — replay serializes the vocoder ops / loses host-device
overlap. Magnitude here (+5%) is smaller than the earlier E2E 1.37-vs-1.07 (+28%) because this
isolates the audio decode (no DiT eviction/reload, no 2-gen variance). => durable fix is op-count
reduction, not trace (matches Phase 5/6). Iteration tool is ready for Phases 1-3 + 6.

### Step 0 status: COMPLETE — fast warm turnaround achieved for eager AND traced.
Warm decode iteration is ~1s/decode once the 317s one-time assembly is paid (harness amortizes
it across many decodes in one process). Kernel cache reuse fixed (pin TT_METAL_CACHE +
TT_METAL_HOME -> 100% hits after one priming run). The 317s cold is device-side mesh-workload
assembly, NOT JIT, so it cannot be cached away across processes — the persistent harness is the
answer (pay once, loop many). Proceeding to Phase 1 (clean per-op profile).

## PHASE 1 — clean per-op device profile (4x8 warm), job 022007-14 (2026-06-13 02:20-03:00 UTC / 19:20-20:00 PT)
prof_girl_decode now audio_only=True (no 22B build) + PROF_WARM=1 (cold-assembly zones drained
before the profiled decode). tracy -p -r --op-support-count 16000, LTX_PROF_NOFLUSH=1.
CSV: generated/profiler/reports/2026_06_13_02_56_29/ops_perf_results_2026_06_13_02_56_29.csv
(280960 rows = ops x 32 devices). device-active per op = max DEVICE FW DURATION across the 32
parallel devices; summed per op-type below. Wall 2373s (mostly one-time profiler-instrumented
kernel recompile into separate cache tt-metal-cache2908... — now cached for future prof runs).

NOTE: capture includes BOTH the cold prime and the warm decode (the cold ReadDeviceProfiler
drain didn't fully empty before the profiled decode), so ABSOLUTE ms are ~2x a single decode;
the PROPORTIONS (the deliverable) are clean. OP-TO-OP LATENCY is STILL garbage (NOFLUSH did not
fully clean it at this op count) — ignore the o2o column; use device-FW only.

Device-active by op-type (sum max-FW, % of 39189ms total):
| op                    | %    | cnt   | BRISC ms | NCRISC ms | TRISCmax ms |
|-----------------------|------|-------|----------|-----------|-------------|
| Conv3d                | 32.8 | 15424 | 12588    | 12477     | 12578       |
| BinaryNg              | 24.7 | 65728 |  3866    |  3772     |  3789       |
| Conv1dDepthwise       | 10.5 | 38464 |  3862    |  3616     |  3783       |
| Ternary               |  7.2 | 12800 |  2732    |  2709     |  2725       |
| NeighborPadAsync      |  6.2 | 38528 |  2331    |  2334     |     0 (DM)  |
| Slice                 |  5.7 | 43392 |  1368    |  1332     |     0 (DM)  |
| Concat                |  5.6 | 19712 |  2058    |  2044     |     0 (DM)  |
| Tilize/Untilize (4x)  |  5.7 |       |          |           |             |
| AllGatherAsync (CCL)  |  0.7 |   832 |   201    |   259     |     0       |
| (others <0.5% each)   |      |       |          |           |             |

**Phase 1 findings:**
1. Conv3d (32.8%) is the single dominant op — and it is NOT purely DM/read-bound as the prior
   flushed profile suggested: BRISC≈NCRISC≈TRISC (12588/12477/12578ms) are all ~equal, so compute
   (TRISC) is just as loaded as data-movement. It's balanced, ~saturated on all three RISCs.
2. BinaryNg is the surprise #2 at 24.7% with a MASSIVE 65728 invocations — scattered residual/bias
   adds. Its per-RISC is DM-heavy-ish but the sheer op COUNT is the story (op-count reduction lever).
3. Conv1dDepthwise 10.5%, Ternary 7.2%. NeighborPad/Slice/Concat are pure data-movement (TRISC=0),
   together ~17.5% — halo/reshape overhead that op-fusion could fold away.
4. CCL (AllGather) only 0.7% — audio decode is NOT CCL-bound on 4x8. (Matches plan: small mesh helps.)
=> Optimization levers (Phase 6), ranked: (A) Conv3d blocking/compute, (B) cut BinaryNg op-count
   (fuse residual/bias adds), (C) fold NeighborPad halo into conv reads + cut Slice/Concat.

## PHASE 6 — Optimization iterations (PCC-gated)

### BinaryNg source pinpointed (from Phase-1 CSV, per-device-per-decode)
The 65728 BinaryNg are NOT residual/bias adds — they are the T-shard tile-align
pad-image machinery `_set_tpad_tail` (audio_ops.py). Per device per decode:
- f32 MUL broadcast (mask × tensor): ~611/dev, ~48ms/dev — `x*M` (zeros) + `x*M`,`last*inv` (replicate)
- f32 ADD full: ~322/dev, ~16ms/dev — `add(xm,fill)` (replicate) + AMP residual adds
- bf16 (mel-VAE): ~80/dev, ~173ms/dev — high per-op cost, low count
Analytic count: 109 replicate (slice+2mul+add) + 115 zeros (1 mul) tail-sets per
vocoder forward, ×2 generators (main voc + bwe, both T-shard=8 on 4x8). The masks
are dispatch-cheap (~0.08ms each); per-op fusion can't cut them — only fewer CALLS helps.
The affine blend `x*M + last*inv` is an irreducible 3-op floor in ROW_MAJOR (mac/where/
addcmul are TILE-only; conversions would add ops). So the only lever is dedup/skip.

### Iter 1 — per-stage replicate tail-set hoist (commit 2b67b1e0e16), job 032826-15
The 3 AMP blocks of a stage share one input; each block's acts1[0] replicate-tailed
it identically. Hoisted to once per stage (shared read-only). Cuts ~72 redundant
BinaryNg/Slice ops/decode.
- PCC conv1d-vs-torch: 0.99379 (UNCHANGED — algebraically identical). Gate PASS.
- warm vocoder+bwe: ~865ms (reps 841/865/901/865) vs baseline 866ms — NEUTRAL (within noise)
- cold vocoder+bwe: 319193ms vs baseline 316631ms — NEUTRAL (within noise)
VERDICT: correct op-count cleanup, but NO measurable perf win — the ~72 cut ops are
1.6% of the 4400-op graph and dispatch-cheap, below the ±30ms warm noise floor.
LESSON: BinaryNg device-time (~64ms/decode warm = ~7% of vocoder+bwe) is much smaller
than the 24.7% Phase-1 proportion suggested — that capture INCLUDED the cold prime,
over-weighting assembly-heavy ops. Warm-decode reality: Conv3d dominates. Cutting
BinaryNg further needs HUNDREDS of ops removed (restructuring the tail-correctness
machinery = high PCC risk) for ≤7% of warm time. Pivoting to Conv3d (priority 2, 32.8%).

### Re-prioritization from the WARM per-stage breakdown (job 022007-14 CSV, re-analyzed)
Split the Phase-1 device-FW by dtype (bf16=mel-VAE, fp32=vocoder/bwe) and per-op, /2 for the
2 captured decodes. This corrects the plan's op-ranking, which had mixed the two stages and
included the cold prime:

| stage          | wall  | device-active/dev | dominant ops (ms/dev/decode)                                  |
|----------------|-------|-------------------|---------------------------------------------------------------|
| mel-VAE (bf16) | ~182ms| ~250ms            | Conv3d 142.8 (28 ops, 4-16ms EACH), BinaryNg 86.5             |
| vocoder+bwe    | ~865ms| ~362ms            | BinaryNg 64.5, Conv1dDepth 64.2, Conv3d 58.3, Ternary 43.8,   |
|                |       |                   | NeighborPad 38.0, Concat 32.1, Slice 24.1 — FLAT, ~3100 ops   |

KEY corrections to the Phase-1 ranking:
1. Conv3d's 32.8% is mostly the **mel-VAE** (bf16, 142.8ms/dev) — a handful (28) of LARGE convs
   (153x16x512->151x16x512 @ 4078us, 604x64x256 @ 16141us). The vocoder's fp32 Conv3d is only
   58ms. So "tune Conv3d" = tune the **mel-VAE 3D convs** (single-chip, replicated, compute-heavy
   per-op — a real blocking target), NOT the vocoder.
2. Ternary (7.2%) IS `ttnn.snake_beta` (fused activation, TernaryOpType::SNAKE_BETA), 200/decode
   — already a single fused op; not reducible without dropping activations.
3. The vocoder+bwe stage is **~58% dispatch-bound** (865ms wall vs 362ms device-active = ~503ms
   host/dispatch gap) and has NO single dominant op — it is bound by raw op COUNT (~3100/decode:
   947 BinaryNg + 601 Conv1dDepth + 602 NeighborPad + 676 Slice + 282 Concat). This is why trace
   was tried; trace is net-negative here (replay raises device-active), so the durable fix is a
   LARGE op-count cut, not micro-fusion. Small cuts (Iter 1's ~72) are lost in the noise.

### Strategy going forward (honest ROI ranking)
- BinaryNg per-op fusion (priority 1): EXHAUSTED at warm scale. The masks are dispatch-cheap
  broadcast ops; the affine blend is a 3-op ROW_MAJOR floor. Only a large structural cut to the
  T-shard tail machinery would help, at high PCC risk for ≤7% of warm device time. Not worth it.
- mel-VAE Conv3d blocking (was priority 2): the single largest compute block (142.8ms/dev). Real,
  but needs the Phase-2 measured-counter / tt-npe bound FIRST (per plan) and per-shape blocking
  tuning — a large, measurement-gated effort.
- NeighborPad/Slice/Concat fold (priority 3): ~94ms/dev device + a big share of the 503ms vocoder
  dispatch gap. Highest leverage on the dominant stage, but requires folding halo into conv reads
  (structural, PCC-risky). The proven conv1d_depthwise pattern is the template.

### Iter 2 — mel-VAE Conv3d output blocking (commit 74fc7765b0b), job 035133-16  *** WIN ***
The audio mel-VAE Conv2dViaConv3d combos (512/256/128 square, kT=1, W=mel_bins=16) had NO
_DEFAULT_BLOCKINGS entry -> hardcoded (Cin, 32, 1, 1, 1) default -> H_out=W_out=1 = one output
pixel per work-unit = 4-16ms/conv, ~143ms/dev (largest compute block in the decode). Added
entries blocking full mel width (W_out=16) + 8-row height chunk: (Cin, 32, 1, 8, 16).
- PCC conv1d-vs-torch: 0.99379 (UNCHANGED — blocking is math-invariant). Gate PASS.
- **warm mel-VAE stage: 182ms -> 56ms (3.3x)**. vocoder+bwe ~850ms (untouched).
- **total audio decode: ~1047ms -> ~907ms (~13% faster)**. Cold ~317s (unchanged).
- C_in_block kept = full Cin -> weight-prep cache key unchanged (audio_dec_cin55f0111e), no
  weight recompile; only the conv3d program JIT-compiles once for the new H/W block (cold only).
This is the real Conv3d lever (priority 2) — and it was a missing blocking-table entry, not a
kernel tune. Next: try larger H_out / C_out_block to squeeze the mel-VAE convs further.

### Iter 3 — push mel-VAE blocking harder (C_out 32->64, H_out 8->16), job 040559-17  *** NEUTRAL, NOT committed ***
Tried (Cin, 64, 1, 16, 16) for the same audio combos. warm mel-VAE ~52-56ms vs iter2's 56ms —
within noise, no additional win (PCC still 0.99379, no OOM). The convs are no longer
blocking-bound after iter2. Reverted to iter2's (Cin, 32, 1, 8, 16) — the smaller-L1, equally-fast
choice already committed (74fc7765b0b). mel-VAE blocking is now tapped out.

## PARADOX RESOLVED — vocoder+bwe is DEVICE-bound, not host-dispatch. job 044055-18 (2026-06-13 04:41-04:47 UTC / 21:41-21:47 PT)
Added PROBE_SPLIT=1 to audio_warm_harness.py: times the vocoder+bwe stage as main-vocoder vs
bwe-half separately (device-synced), eager AND traced, in one warm process. The main vocoder is
exactly the part that runs TRACED in the traced leg (LTX_VOC_TRACE=1) — so if its wall were
host-dispatch-bound, trace (which removes per-op host dispatch) would collapse it toward its
device-active floor.

| half          | EAGER   | TRACED  | delta            |
|---------------|---------|---------|------------------|
| main_vocoder  | 436.0ms | 453.7ms | +17.7ms (SLOWER) |
| bwe_half      | 448.9ms | 464.4ms | +15.5ms (SLOWER) |
| vocoder+bwe   | 862.6ms | 882.6ms | +20.0ms (SLOWER) |
(mel_vae 56ms both legs — iter2 blocking holding. eager total 920.8ms, traced 942.1ms.)

**ANSWER: the 503ms wall-vs-device gap is DEVICE-SIDE, not host dispatch.** Removing per-op host
dispatch (trace) did NOT shrink the main vocoder — it got +17.7ms SLOWER. A host-dispatch-bound
graph would have dropped toward its ~180ms device-active floor under trace; it did not move down
at all. So the gap is per-op device-side latency: CCL halo barriers (neighbor_pad_async / all_gather
semaphore waits), runtime op-launch/sync, and scheduling between the ~3100 small ops. Trace replay
even adds a small device-side serialization penalty (+15-18ms per half), which is why trace is
net-negative here. **=> The durable lever is OP-COUNT / DATA-MOVEMENT REDUCTION (fewer device-side
sync points), confirmed. Trace is permanently off for this stage.**

Second finding: vocoder+bwe splits ~evenly — main_vocoder ~436ms / bwe_half ~449ms. The BWE half
(bwe_generator + host-roundtrip mel-STFT + resample + clamp) is HALF the stage, not a sidecar.
Per-Activation1d data-movement (2 NeighborPad halo + 2 Slice polyphase + 1 Concat interleave, in
audio_resample.py UpSample1d/DownSample1d) is the dominant op-count source in BOTH halves.

### Iter 4 — polyphase upsample single-slice (commit 98543b27770), job 045854-19  *** WIN ***
The 2x polyphase UpSample1d sliced x_pad TWICE (in0 at offset 2, in1 at offset 3) to feed the two
phase convs. Phase 1's window = phase 0's advanced by one sample, so zero-padding the sub-tap
vectors (sub0 trailing zero, sub1 leading zero -> both K_sub+1 taps) folds the one-sample offset
INTO the filter and both convs read ONE shared slice base=x_pad[2:T_pad-2]. ~218 ttnn.slice
ops/decode removed (UpSample1d is called by every Activation1d; ~109/generator x2). Verified
bit-identical in torch across shapes before touching device code.
- PCC conv1d-vs-torch: **0.99379 (UNCHANGED** — algebraically identical). mac 0.99613. Gate PASS.
- **vocoder+bwe: 862.6 -> 819.6ms (-43ms, ~5%)**; main_vocoder 436.0->415.6, bwe_half 448.9->436.9.
- **total decode: ~920 -> 873ms warm (~5%)**. mel_vae 54-56ms (iter2 holding). cold ~311s.
- This CONFIRMS the paradox finding empirically: a pure op-COUNT cut (no compute change, same
  convs) removed device-side time well above the ±15ms noise floor (eager total 856-895 vs prior
  906-932 — cleanly separated). Op-count reduction IS the lever; the cut device time was the
  device-side per-op gap, exactly as the paradox probe predicted.

### Iter 5 — polyphase upsample drops the crop slice (commit 3454004c336), job 052355-20  *** WIN ***
The ratio-2 polyphase path only ever reads x_pad[2:T_pad-2]. Pad two fewer rows per side
(eff_pad = self.pad - 2) and consume the halo'd tensor directly: the iter4 base slice is GONE,
and each neighbor_pad halo exchange moves 2 fewer rows per side. ~218 more ttnn.slice/decode
removed (UpSample1d now has ZERO slices in the polyphase path). Verified bit-identical in torch
across shapes (incl. the replicate-pad edge behavior) before touching device code. Falls back to
full pad + slice when self.pad < 2.
- PCC conv1d-vs-torch: **0.99379 (UNCHANGED)**. mac 0.99613. Gate PASS.
- **vocoder+bwe: 819.6 -> 791.0ms (-29ms)**; main_vocoder 415.6->390.6, bwe_half 436.9->414.2.
- **total decode: 873 -> 843ms warm**. mel_vae 56ms. cold ~302s.
- **CUMULATIVE iters 4+5: vocoder+bwe 862.6 -> 791.0ms (-72ms, ~8%); total ~920 -> 843ms (~8.4%).**

### Status: two clean op-count wins landed. Remaining Python data-movement is hard to cut further.
The proven kernel-level halo-into-conv-read fold (the plan's primary structural target) needs a
C++ rebuild, which this worktree CANNOT do (it runs on main's compiled binaries via symlinks; pure
models/ Python only). The remaining big data-movement counts are: NeighborPad (~per-conv, distinct
tensors — not Python-mergeable without the kernel fold), the polyphase interleave Concat (already
the minimal 1-op interleave of two tensors), and per-conv halos. The two polyphase slice cuts were
the clean structural wins available in pure Python. Further large cuts (e.g. folding upsample halo
into the conv1d_depthwise read) require the kernel change → a build-capable workspace.

## PHASE 7 — KERNEL halo-fold attempt (build-capable main repo). 2026-06-13 (UTC) / (PT)

### Branch setup (DONE)
On the BUILD-capable main repo /home/smarton/tt-metal:
- Stashed 3 stray tracked edits on smarton/optimizer/ltx-galaxy-v2
  (`git stash push -u`: pipeline_ltx.py, prof_girl_decode.py, test_pipeline_ltx_distilled.py).
  Two untracked .mp4 artifacts left in place.
- `git checkout -b smarton/optimizer/ltx-audio-kernel 3454004c336` — NEW branch at the worktree
  tip (coexists with the held worktree branch worktree-agent-a03641bafa2c6612d). Verified HEAD =
  3454004c336 (cumulative ~843ms warm, PCC 0.99379). Harness + prof files present on the branch.
- Working tree clean (no source edits made this phase).

### GOAL re-examined: fold the per-conv causal-halo NeighborPad INTO the conv read.
Studied the template and the audio conv path end-to-end (static analysis; no build/run needed to
reach the verdict):

TEMPLATE — conv1d_depthwise stride-1 halo COALESCE
(ttnn/cpp/.../conv1d_depthwise/device/{program_factory.cpp, kernels/reader_conv1d_depthwise.cpp}):
its "coalesce" path (active when B==1) reads the per-block input-page UNION once from DRAM into an
L1 scratch CB, then gathers the K overlapping tap windows from L1 — cutting the ~K× redundant DRAM
RE-READ of **the same local tensor**. It is a LOCAL-tensor read-amplification fix. It does NOT
cross chips. (Audio waveform tensors are B==1, so this coalesce is already active for the vocoder
Conv1dDepthwise convs — no further win available there.)

AUDIO conv path (models/tt_dit/layers/{audio_ops.py, audio_resample.py}):
  _t_neighbor_pad (CCL neighbor_pad_async)  ->  depthwise_tap_filter -> ttnn.experimental.conv1d_depthwise
                                            (or  Conv1dViaConv3d.forward -> ttnn.experimental.conv3d)
On bh_4x8 the audio T axis is sharded factor=8 (pipeline_ltx.py: t_factor=max(mesh)=8 ->
ParallelFactor(factor=8)). So EVERY per-conv halo is a genuine CROSS-CHIP exchange across the 8
T-shards.

### VERDICT: halo-fold is INFEASIBLE in this path on the sharded (4x8) config. (documented negative)
Root cause, confirmed from the op sources:
1. neighbor_pad_async is a FABRIC (ethernet) CCL op
   (ttnn/cpp/.../neighbor_pad_async/device/neighbor_pad_async_program_factory.cpp:
   `#include fabric.hpp`; H/W "fabric writers" send boundary rows to the neighbor chip via fabric,
   "fabric readers" receive the halo from fabric into L1; uses GlobalSemaphores + barrier sems +
   num_links + topology). The halo data physically lives on NEIGHBORING CHIPS.
2. Both conv readers (conv1d_depthwise reader_conv1d_depthwise.cpp; conv3d reader_vol2col.cpp) read
   ONLY the LOCAL interleaved tensor via a `TensorAccessor` addressed by page index. They run on the
   data-movement RISCs with no fabric endpoint, no neighbor-device coords, no inter-chip semaphore
   handshake. A TensorAccessor cannot address a neighbor chip's pages.
3. conv3d's reader DOES already do internal T/H/W padding (padding_t/h/w via clampIndex/zeroPad),
   but that pads with LOCAL boundary values or zeros. The real causal halo needs the NEIGHBOR shard's
   ACTUAL samples; substituting local-clamp/zero at the 8 shard seams changes the math -> PCC fail.
=> "Folding the halo into the conv read" on the sharded path = embedding a fabric CCL RECEIVER
   inside the conv kernel. That re-implements neighbor_pad in-kernel (same fabric cores, same
   semaphores, same receive-before-compute barrier) — it does NOT cut op count OR device-side sync
   points (the dominant cost per the PARADOX-RESOLVED finding), at very high complexity + PCC risk.
   Not a win; not attempted on device (strong static evidence it cannot reduce device time).

The only LOCAL (foldable) pad in the conv path is the `external_pad_front` zeros+concat branch
(audio_ops.py:784) — but that is the UNSHARDED (factor==1) fallback, NOT exercised on bh_4x8
(factor=8). Folding it would save nothing in the target config.

### Re-evaluated cutting remaining per-conv NeighborPad/Concat with kernel freedom: still no lever.
- The per-conv NeighborPads are distinct cross-chip exchanges on distinct tensors at distinct conv
  boundaries — not mergeable without changing the dataflow graph (each conv genuinely needs its own
  boundary context). A kernel change does not make them fewer.
- The polyphase interleave Concat is already the minimal 1-op interleave of two tensors (iters 4/5).
- conv1d_depthwise coalesce is already active (B==1) — the local read-amplification is already cut.

### Build outcome: NO BUILD RUN. No C++ change was made — a build would compile an unchanged tree.
A documented-negative is the honest outcome here: the plan's primary structural target does not
exist in the sharded audio path (the halo is fabric/cross-chip, not a local re-read), so there is
no kernel edit that folds it without re-implementing the CCL op. PCC gate / warm decode unchanged
at the iter-5 baseline (vocoder+bwe ~791ms, total ~843ms, PCC 0.99379); no regression possible
(no edit). No commits this phase (no validated code change to land).

### Where the durable lever actually is (carry-forward, grounded by the PARADOX-RESOLVED probe)
The ~503ms wall-vs-device gap in vocoder+bwe is DEVICE-side per-op latency (CCL halo barriers +
op-launch/sync across ~3100 small ops), NOT host dispatch and NOT a local read-amp the conv reader
can absorb. The real future levers are op-GRAPH-level, not single-kernel:
  (a) FEWER cross-chip halo barriers — e.g. fuse consecutive same-axis halos, or run more of the
      vocoder UNSHARDED on T (gather once, compute many convs locally, partition once) so the
      per-conv neighbor_pad disappears. This trades CCL for redundant local compute; net win depends
      on conv count between halos vs. gather cost. Needs measurement, not a conv-kernel rewrite.
  (b) AllGather num_links/topology tuning (CCL only 0.7% here — low ceiling).
Both are device-graph restructurings in models/ (Python), not the conv reader/program-factory fold
the plan hypothesized. The conv-kernel fold is closed as infeasible.

## PHASE 8 — T-shard / submesh sweep + audio-decode submesh routing (build-capable main repo)
Branch smarton/optimizer/ltx-audio-kernel. 2026-06-13 06:24-07:37 UTC / 23:24-00:37 PT.
Confirms carry-forward lever (a): fewer cross-chip T-halo barriers by reducing T-shard count.

### T-shard sweep — warm audio decode, EAGER, conv1d path (jobs 062355-21, 064157-22)
audio_warm_harness.py extended with a 1x4 case; 2x4/1x4 open the FULL (4,8) galaxy and
create_submesh the slice (opening a 2x4/1x4 SUBSET standalone fails fabric router sync on
this machine — every audio decode runs on a submesh of the full galaxy). t_axis = larger
mesh axis, so on 4x8 T-shard=8; on 2x4/1x4 T-shard=4.

| config         | T-shard | mel_vae | vocoder+bwe | total    | cold     |
|----------------|---------|---------|-------------|----------|----------|
| bh_4x8sp1tp0   | 8       | 54.9ms  | 786.9ms     | 841.8ms  | 495.8s   |
| bh_2x4sp1tp0   | 4       | 40.9ms  | 646.8ms     | 687.8ms  | 141.4s   |
| bh_1x4sp1tp0   | 4       | 40.3ms  | 632.6ms     | 671.9ms  | 5.6s     |

KEY: halving T-shard (8->4) cut vocoder+bwe ~140-154ms (~18-20%), exactly the cross-chip
neighbor_pad-halo-barrier reduction the PARADOX-RESOLVED probe predicted. 1x4 (T-shard=4,
single row, fewest ethernet hops, no channel axis) is best, just beating 2x4. Cold
device-side mesh-workload assembly also collapses with fewer chips (496s -> 5.6s) — fewer
cross-chip program barriers to assemble. CCL is not the audio bottleneck on small meshes.

### Chosen config + IMPLEMENTED: env-gated audio submesh routing (commit 6182a3b34c0)
LTX_AUDIO_SUBMESH=RxC routes ONLY decode_audio onto an RxC submesh of the full mesh
(decode_audio is torch-in/torch-out and self-contained, so video keeps the whole mesh).
The pipeline builds the audio decoder/vocoder/CCL on self.audio_mesh_device (a create_submesh
slice) and keys the weight cache by the submesh shape. Default OFF -> audio_mesh_device IS
the full mesh, production path byte-identical to baseline.

VALIDATED on the full-galaxy production path (job 070157-23, LTX_AUDIO_SUBMESH=1x4, LTX_TRACED=0):
- conv1d-vs-torch **PCC=0.99379 (= baseline, gate PASS >0.95)**; mac 0.99613.
- **warm decode 673.9ms** (full 4x8 opened, audio routed to a 1x4 submesh) vs 841.8ms baseline
  = **-168ms / ~20% faster**. cold 5.5s. Matches the standalone 1x4 sweep (671.9ms).
DEFAULT-PATH HARD GATE re-validated (job 072557-27, NO submesh): test_audio_decode_girl
-k bh_4x8sp1tp0 LTX_TRACED=0 **PASSED, PCC 0.99379, warm 849.6ms, CLEAN teardown (rc=0)** —
no "Audio decode routed" line, so the 4x8 path creates no submesh and is unchanged.

### LIMITATION — why it ships gated (not default-on)
A create_submesh child SHARES the parent mesh's command queue. ttnn forbids closing a
cq-sharing child while the parent is alive (close HANGS — the documented "one mesh per CQ or
teardown hangs"), AND forbids closing the parent while the child is alive ("MeshDevice cq ID 0
is in use by child submesh"). So a routed audio submesh's lifetime is bound to the parent: it
is reclaimed only at process teardown. For a one-shot generation process this is fine (process
exits), but the pytest fixture's explicit mid-process parent-close trips the throw. Verified
empirically: explicit close hangs (job 071434-25, killed) / throws (071013-24, 072121-26).
release_audio_submesh() therefore only frees the audio device tensors and drops references;
it does not close the submesh. Default-on production routing needs a ttnn-side fix (separate
cq per submesh, or cascade close-children-first) — out of scope for this pure-Python lever.
=> SHIPPED as an opt-in capability + the validated 20% win; production 4x8 untouched.

### Carry-forward
- The submesh-routing win (20%, PCC-clean) is real and the durable lever (a) is now PROVEN:
  audio decode does NOT scale with chips — fewer T-shards = fewer cross-chip halo barriers =
  faster. The full-galaxy AV pipeline should route decode_audio to a 1x4/2x4 slice once ttnn
  supports clean child-submesh teardown (or via a process that exits after generation).
- Next pure-Python lever if needed: run the vocoder UNSHARDED on T within the submesh (gather
  once, many local convs, partition once) to drop the remaining per-conv neighbor_pad entirely
  — trades CCL for redundant local compute; measure conv-count-between-halos vs gather cost.

## PHASE 9 — SMALL-submesh sweep (1x1/1x2) + E2E push gate. 2026-06-14 08:03-09:18 UTC / 01:03-02:18 PT
Worktree agent-acc25593318152875 on ltx-perf HEAD 6182a3b (build symlinked from main).
Harness commit 0d27007987c (worktree branch, NOT ltx-perf) adds the 1x1/1x2 sweep cases.

### Full warm-decode-vs-chips curve — extends the sweep BELOW T-shard=4 (jobs 080305-38, 080840-40)
Warm, EAGER, conv1d, audio routed to the submesh (audio_warm_harness, median of 5, device-synced):
| config | chips | T-shard | mel_vae | vocoder+bwe | total warm | cold |
|--------|-------|---------|---------|-------------|------------|------|
| 1x1    | 1     | 1       | 39.0ms  | 1427.8ms    | 1462.7ms   | 102.8s* |
| 1x2    | 2     | 2       | 40.3ms  | 1010.7ms    | 1051.0ms   | 120.6s* |
| 1x4    | 4     | 4       | 39.9ms  | 632.9ms     | 675.3ms    | (5.6s prior, primed) |
| 2x4    | 8     | 4       | 40.9ms  | 646.8ms     | 687.8ms    | (141s prior) |
| 4x8    | 32    | 8       | 54.9ms  | 786.9ms     | 841.8ms    | (496s prior) |
*1x1/1x2 cold confounded by JIT recompile into the worktree-keyed cache (46.1%/58.9% hits) —
recompile+assembly, not clean assembly. Clean cold = the 4x8/2x4/1x4 prior column.

KEY: the curve is U-SHAPED, not monotonic. Minimum at T-shard=4 (1x4 633ms); BOTH 4x8 (T=8,
787ms, halo-barrier-bound) and 1x1 (T=1, 1428ms, per-chip-serial-compute-bound) are slower.
1x1 has ZERO cross-chip halos yet is the SLOWEST — so the lever is NOT "fewest chips," it is
"the T-shard count that balances fabric-halo barriers (∝ T-shard) against per-chip serial conv
compute (∝ 1/T-shard)." Corrects the prior "smaller = always faster" reading (that sweep only
went 8→4→4). Why+which numbers: /home/smarton/audio-perf.md.

### E2E push gate {1x1, 1x2} — NEITHER passes -> PUSHED NOTHING to ltx-perf
- 1x2 audio PCC (job 083228-41, test_audio_decode_girl bh_4x8sp1tp0, LTX_TRACED=0,
  LTX_AUDIO_SUBMESH=1x2): conv1d-vs-torch PCC=0.99664 (PASS), mac 0.99936, valid 6.01s audio.
  But the pytest teardown SIGABRTs (cq-sharing child close) — decode itself was clean.
- 1x2 FULL E2E (test_pipeline_distilled bh_4x8sp1tp0_ring, LTX_AUDIO_SUBMESH=1x2): HANGS at the
  submesh audio decode in BOTH traced (job 083228-41, warmup trace capture) and eager (job
  085025-44, after video Stage1 71.6s/upsample 13.5s/Stage2 42.2s/VAE 35.8s all clean). Audio
  decode works STANDALONE but DEADLOCKS on the child submesh AFTER the parent ran video on the
  shared CQ. => 1x2 FAILS the full-E2E gate (no audio+video produced).
- 1x1 INDETERMINATE: SIGKILLing the hung 1x2 eager E2E mid-CCL left the board FW-init stuck
  ("Device 0 Timeout waiting for physical cores ... failed to initialize FW! Try resetting the
  board") — confirmed by a bare device-open probe (job 091759-50). Cannot reset per safety rule,
  so 1x1 E2E unevaluated. DEVICE NEEDS AN OPERATOR tt-smi RESET.
- Per the rule (push fastest of {1x1,1x2} that passes BOTH gates, else push nothing): NEITHER
  achieved a clean full E2E, so NOTHING pushed to ltx-perf. The env-gated LTX_AUDIO_SUBMESH on
  ltx-perf (6182a3b) is unchanged. Root cause: create_submesh child shares the parent CQ; the
  shared-CQ child audio decode deadlocks after the parent issued video — the same ttnn cq
  limitation documented for teardown, now shown to break the live AV decode too. Default-on
  needs a per-submesh CQ (or cascade close-children-first) ttnn fix.
