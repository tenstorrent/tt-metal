# Mission — make LTX fresh-build iteration fast (audio kernel-compile reduction)

Autonomous optimization loop. Goal: cut the **fresh-build (cold kernel cache) audio warmup/compile
time** that dominates LTX E2E load. Kevin Mi (domain expert): galaxy fresh build ≈ **20 min**,
dominated by **audio** — "a lot of variants of kernels need to be compiled more than the other
modules." Evidence: last E2E showed **3117 unique kernels / 25,935 build-once dedup**;
`models/tt_dit/utils/conv3d.py` alone has ~277 blocking entries. The audio decode
(mel-VAE Conv3d + vocoder Conv1dViaConv3d/conv1d_depthwise + BWE) instantiates a kernel per
(shape, blocking, dtype, T-shard layout) → variant explosion.

Use superpowers (systematic-debugging — root-cause before fixes; TDD — measurement/gate FIRST;
verification-before-completion) and tt-buddy (tt:optimizer loop, tt:profiler, tt:run). Commit
frequently on the branch; append to the PROGRESS LOG after EVERY step so a relaunch resumes. Do
NOT push to ltx-perf — branch only; the owner reviews. Do NOT break audio output.

## HARD safety
- Device ONLY via broker MCP. Check queue first; QUEUE behind others. Never reset while a foreign
  tenant runs; reset only your OWN wedged proc + only if no foreign tenant. Never kill foreign
  jobs. SIGTERM only for your own orphans (never -9).
- Work in the worktree `/home/smarton/tt-metal/.claude/worktrees/audio-compile-opt` (branch
  `smarton/audio-compile-opt` off origin/ltx-perf @6182a3b), build+python_env symlinked from main.
  Changes here are Python (blockings/shapes) → reuse the prebuilt C++, no rebuild.
- Before editing source: Read .../tt-buddy/skills/buddy/code-comments.md. Before commits: Read
  commit-messages.md; footer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Timestamps UTC+PT.

## Metric + measurement protocol (TDD: build this FIRST, baseline before optimizing)
- **Primary metric:** cold-cache compile wall + unique-kernel count for the audio decode, measured
  via `test_audio_decode_girl -k bh_4x8sp1tp0` (audio_only path, has the conv1d-vs-torch PCC gate).
- **Cold-cache = reproducible:** point `TT_METAL_CACHE` at a FRESH empty temp dir each measurement
  (e.g. `/home/smarton/.cache/audio-compile-bench-<tag>`), wiped before the run, so every run pays
  the full compile. Grep the run for `JIT cache stats` (unique kernels, build count) + total wall +
  the audio cold time the test prints. Attribute compiles to audio vs other modules if the
  telemetry allows (audio_only build minimizes non-audio).
- Env: TT_METAL_HOME=<worktree>, TT_DIT_CACHE_DIR=/home/smarton/.cache/tt-dit, LTX_CHECKPOINT +
  GEMMA_PATH (the kevinmi paths in run_girl_e2e.sh), --timeout=0.
- Baseline run FIRST; record unique-kernel count + cold compile wall in the PROGRESS LOG.

## Levers (do all three; measure each independently; keep the wins that pass gates)
1. **Consolidate conv blockings** — map which of the ~277 conv3d/conv1d blockings the AUDIO path
   actually hits; dedup/merge near-equivalent ones so fewer distinct kernel hashes compile.
   TRADE-OFF: a coarser blocking may slow the warm decode — gate on warm decode time (≤10% regress).
2. **Shape bucketing/padding** — pad audio conv shapes into shared buckets so layers reuse kernels.
   NUMERICS RISK: padding can change output — gate hard on PCC.
3. **Precompile / shared cache** — populate + reuse a shared audio kernel cache so a fresh build
   hits cache instead of compiling. Amortizes even what variant-reduction can't remove.

## Gates (EVERY change must pass before commit)
- **Audio PCC:** conv1d-vs-torch PCC > 0.95 (the test's built-in gate) — MUST hold.
- **Warm decode time:** not regressed > 10% vs the pre-change warm decode (levers 1/2 trade compile
  for runtime — don't trade away the decode speed).
- **Compile metric improved:** unique-kernel count and/or cold compile wall measurably down.
- A lever that fails any gate is reverted, not committed; document why in the PROGRESS LOG.

## Loop + stop condition
tt:optimizer loop: measure → hypothesize → implement ONE lever → measure → gate → commit/revert →
repeat. STOP when: cold-cache audio compile wall is cut ≥50% from baseline (stretch: as low as it
goes), OR two consecutive levers each yield <5% with PCC/runtime gates binding (diminishing
returns). Then write a final summary + recommendation (what to land, the compile-vs-runtime
trade-offs, whether precompiled-cache shipping is the right call).

## PROGRESS LOG (append after every step; newest last)
- (created) baseline not yet measured. First: build the cold-cache measurement harness + record
  the baseline unique-kernel count + audio cold compile wall.
- [2026-06-15 ~19:05 UTC / 12:05 PT] SETUP + ROOT-CAUSE (no device yet; queued behind foreign
  sulphur run job 185336-38 in /home/sulphur/tt-metal — NOT disturbing it).
  * Worktree confirmed: branch smarton/audio-compile-opt, build->build_Release, python_env symlinked.
  * Gate test located: test_pipeline_ltx_distilled.py::test_audio_decode_girl -k bh_4x8sp1tp0.
    PCC gate = conv1d-vs-torch `c_pcc > 0.95` (+ localized-spike gate). Prints
    `AUDIO_GIRL ... cold=<ms>ms warm=<ms>ms`. audio_only=True skips the ~10min video warmup.
  * METRIC source nailed: tt_metal/jit_build/build_cache_telemetry.cpp:197 emits
    `JIT cache stats: {hits}/{total} hits (X%) [{cached} cached, {dedup} build-once dedup, ...]`
    at process teardown (BuildCacheTelemetry dtor -> dump_metrics, always on, no env gate).
    compiled = total - hits = the cold-compile count to minimize. Process-global, but audio_only
    minimizes non-audio; cross-iteration DELTA is the clean audio attribution.
  * Cold-cache = fresh empty TT_METAL_CACHE dir (rtoptions.cpp:407, defaults to temp if unset).
  * ROOT CAUSE of variant explosion (the reframe): the Python conv blocking table
    (utils/conv3d.py _FP32_BLOCKINGS / _DEFAULT_BLOCKINGS) is ALREADY deduped by
    (Cin, Cout, kernel) — Lever 1 (consolidate blockings) has little headroom for AUDIO.
    The real driver is the conv3d DEVICE-kernel hash: conv3d_program_factory.cpp:693-798 bakes
    T_in, T_out, num_patches(=T_out_block*H_out_block*W_out_block), AND dilation as COMPILE-TIME
    args. In the BigVGAN vocoder, T grows through 6 upsample stages (rates 5,2,2,2,2,2 =>
    cumulative x5/x10/x20/x40/x80/x160), so each of the 18 AMPBlock1 (3/stage) sees a DIFFERENT
    T => a distinct kernel hash even for identical (Cin,Cout,kernel,blocking). Plus per-branch
    dilation (1,3,5) multiplies it. => Lever 2 (T bucketing/padding to shared buckets) is the
    high-leverage lever for audio, and Lever 3 (precompile/shared cache) amortizes the rest.
  * HARNESS (TDD, committed before baseline): models/tt_dit/tests/models/ltx/audio_compile_bench.sh
    — wipes TT_METAL_CACHE, runs the gate test cold, greps JIT cache stats + AUDIO_GIRL.
  * NEXT: queue baseline behind foreign run; record compiled-kernel count + cold/warm + PCC.
- [2026-06-15 ~19:08 UTC / 12:08 PT] BASELINE ATTEMPT 1 (job 190740-39) FAILED — device, not code.
  Got the device after the foreign sulphur run released. Test errored in SETUP (16.6s) at
  create_submesh: root cause `Device 0 init: failed to initialize FW! Timeout (10000ms) waiting
  for physical cores 26-25,27-25` (metal_context.cpp:767 / risc_firmware_initializer.cpp:1402),
  then a teardown ethernet-core-27-25 timeout. Board was wedged when I grabbed it (likely the
  prior foreign run's mesh left in a bad state). NO reset performed — a foreign sulphur job
  (190746-40) started running immediately after and reset would abort it (HARD safety). Foreign
  run starting cleanly suggests the board self-recovered. Re-queued baseline as 190900-41 behind
  the foreign run. Audio conv inventory mapped statically meanwhile: 2 BigVGAN Vocoders (main:
  ch=1536, rates 5/2/2/2/2/2; BWE: ch~512 down to 32, rates 6/5/2/2/2), each = conv_pre(k7) +
  6 ConvTranspose ups + 18 AMPBlock1 (6 DilatedConv1d each: dil 1/3/5 + dil 1) + conv_post(k7).
  Per-stage unique (Cin,Cout,T) x per-branch dilation = the kernel-variant fan-out (T compile-time).
- [2026-06-15 ~20:50 UTC / 13:50 PT] RESUME (worker restarted; harness commit 2a7b31ad04a intact).
  Reviewed prior baseline attempts: 190900-41 "exit 0" was a FALSE POSITIVE — test ERRORED in
  setup (create_submesh) on the same `ethernet core 27-25` FW-init/teardown timeout; no JIT cache
  stats, no AUDIO_GIRL line emitted. 191157-43 (-9) and 192301-46 (interrupted) also failed.
  Root cause is the recurring wedged-board ethernet-27-25 init timeout, NOT our code. Since then
  the human (smarton) ran several `tt-smi -glx_reset` + eth-triage; latest triage 20:47 reports
  check_eth_status.py: PASS overall (transient retrains on Dev24 4-1 / Dev25 15-1 only, not 27-25).
  Device queue now FREE, no foreign tenant. Queuing baseline (timeout 3600s) as the clean window.
